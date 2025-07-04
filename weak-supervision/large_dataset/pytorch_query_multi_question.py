import ollama
import os
from tqdm import tqdm
import json
import argparse
import wandb
import pandas as pd
import sys
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, CLIPModel
from torch.nn.functional import cosine_similarity
import torch
import signal
import torch.nn.functional as F
from scipy.spatial.distance import cosine
import tempfile
import uuid
import shutil
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import torch.multiprocessing as mp
import torchvision.transforms as transforms


class GullyTileDataset(Dataset):
    """Custom PyTorch Dataset for loading tile image collages for gully detection."""
    
    def __init__(self, image_dir, tile_numbers, transform=None):
        """
        Args:
            image_dir (str): Directory containing the tile images
            tile_numbers (list): List of tile numbers to use
            transform (callable, optional): Optional transform to be applied on the collages
        """
        self.image_dir = image_dir
        self.tile_numbers = tile_numbers
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
    def __len__(self):
        return len(self.tile_numbers)
    
    def __getitem__(self, idx):
        tile_number = self.tile_numbers[idx]
        collage = self.collage_images(self.image_dir, tile_number)
        
        if collage is None:
            # Return a placeholder if no images found
            collage = Image.new('RGB', (1200, 1200), (0, 0, 0))
            # Convert to tensor
            tensor_collage = self.transform(collage)
            return {"tile_number": tile_number, "collage": tensor_collage, "valid": False}
        
        # Convert to tensor
        tensor_collage = self.transform(collage)
            
        return {"tile_number": tile_number, "collage": tensor_collage, "valid": True}
    
    def collage_images(self, base_path, tile_number):
        """
        Create a collage of all images for a specific tile number.
        Images follow patterns: rgb_{0-6}_tile_{tile_number}.tif and rgb_highres_tile_{tile_number}.tif
        
        Args:
            base_path: Base directory containing the images
            tile_number: The tile number to find images for
            
        Returns:
            PIL Image object with the collage
        """
        # Ensure base_path ends with a slash
        if not base_path.endswith('/'):
            base_path += '/'
        
        # Define patterns for finding the images
        image_patterns = [
            f"rgb_{i}_tile_{tile_number}.tif" for i in range(7)
        ] + [f"rgb_highres_tile_{tile_number}.tif"]
        
        # Find all existing images that match the patterns
        images = []
        for pattern in image_patterns:
            full_path = os.path.join(base_path, pattern)
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path)
                    # Convert to RGB if the image is in a different mode
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append((pattern, img))
                except Exception as e:
                    print(f"Error loading {pattern}: {e}")
        
        if not images:
            return None
        
        # Determine the grid size
        num_images = len(images)
        cols = min(4, num_images)  # Max 4 images per row
        rows = (num_images + cols - 1) // cols  # Ceiling division
        
        # Resize all images to a common size
        target_width, target_height = 300, 300  # Reasonable size for collage items
        
        # Create a new blank image for the collage
        collage_width = cols * target_width
        collage_height = rows * target_height
        collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
        
        # Add labels for each image
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(collage)
        
        # Place images in the collage
        for i, (name, img) in enumerate(images):
            # Calculate position
            row = i // cols
            col = i % cols
            x = col * target_width
            y = row * target_height
            
            # Resize image while preserving aspect ratio
            img_resized = img.resize((target_width, target_height), Image.LANCZOS)
            
            # Paste into collage
            collage.paste(img_resized, (x, y))
            
            # Add label
            short_names = ["2010", "2012", "2014", "2016", "2018", "2020", "2022", "2023"]
            short_name = short_names[i]
            draw.rectangle([x, y, x + len(short_name)*8, y + 20], fill=(0, 0, 0))
            draw.text((x + 5, y + 5), short_name, fill=(255, 255, 255))
        
        return collage


def get_class_embeddings(prompts, tokenizer, text_encoder, device):
    text_inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
    outputs = text_encoder(**text_inputs)
    text_embedding = outputs.pooler_output
    return text_embedding
    
def get_query_embedding(query_prompt, tokenizer, text_encoder, device):
    text_length = 77
    if len(query_prompt) > text_length: 
        query_prompt = query_prompt[:text_length]
    query_input = tokenizer(query_prompt, padding="max_length", return_tensors="pt").to(device)
    query_output = text_encoder(**query_input)
    query_embedding = query_output.pooler_output
    return query_embedding

def compute_scores_clip(class_embeddings, query_embedding, prompts):
    # Compute cosine similarity scores
    similarity_scores = cosine_similarity(query_embedding, class_embeddings, dim=1)
    
    # Find the highest matching score and corresponding item
    max_score_index = torch.argmax(similarity_scores).item()
    max_score = similarity_scores[max_score_index].item()
    best_match = prompts[max_score_index]
    
    return best_match

def compute_best_match(query_text, class_embeddings, class_dict, model_name):
    query_response = ollama.embed(model=model_name, input=query_text)
    query_embedding = query_response["embeddings"]
    query_embedding_tensor = torch.tensor(query_embedding[0])
    
    list_name_emb = list(class_embeddings.keys())
    current_best_score = -1.0  # Start with a low value for cosine similarity
    current_best_match = ""
    
    for class_name in list_name_emb:
        class_embeddings_tensor = torch.tensor(class_embeddings[class_name][0])
        # Compute the cosine similarity
        similarity_score = F.cosine_similarity(query_embedding_tensor.unsqueeze(0), class_embeddings_tensor.unsqueeze(0), dim=1)
        
        if similarity_score > current_best_score:
            current_best_score = similarity_score.item()
            current_best_match = class_name
    
    matched_label = class_dict[current_best_match] # integer representing class 
    matched_class_name = current_best_match
    return matched_class_name, matched_label 

def compute_scores(class_embeddings, query_embedding, prompts, temperature=0.8):
    scores = []
    # Compute cosine similarity scores
    for class_name in class_embeddings:
        similarity_scores = cosine_similarity(torch.tensor(query_embedding), torch.tensor(class_embeddings[class_name]), dim=1)
        similarity_scores = similarity_scores / temperature
        scores.append(similarity_scores.item())

    probabilities = F.softmax(torch.tensor(scores), dim=0)
    # Find the highest matching score and corresponding item
    max_prob_index = torch.argmax(probabilities).item()
    max_prob = probabilities[max_prob_index]
    best_match = prompts[max_prob_index]
    
    return best_match, probabilities, max_prob

def generate_context_embedding(class_names, model_name, options):
    prompt = "You are working on a difficult fine-grained image classification task, here are the only classes you can choose from"+class_names
    context_response = ollama.generate(model=model_name, prompt=prompt, options=options)
    return context_response['context']

def compute_class_embeddings(class_names_list, model_name, device):
    class_embeddings = {}
    print("Computing the class embeddings --")
    for class_name in tqdm(class_names_list):
        response = ollama.embed(model=model_name, input=class_name)
        class_embeddings[class_name] = response["embeddings"]
    
    return class_embeddings

def get_tile_numbers(base_path):
    tile_numbers = []
    for filename in os.listdir(base_path):
        if filename.startswith("rgb_0_" + "tile_"):
            tile_number = filename.split("tile_")[1].split(".tif")[0]
            tile_numbers.append(int(tile_number))
    return tile_numbers


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timed out!")

def save_to_wandb(results_file_path, vlm_model_name, llm_model_name):
    """
    Save the results JSON file to wandb.
    
    Args:
        results_file_path: Path to the results JSON file
        model_name: Name of the model used for generating results
    """
    # Load the results JSON file
    with open(results_file_path, 'r') as f:
        results_data = json.load(f)
    
    # Log the file to wandb
    wandb.log({f"{vlm_model_name.replace(':', '-')}-{llm_model_name.replace(':', '-')}-multi-question-results": wandb.Table(dataframe=pd.DataFrame(list(results_data.items()), columns=["tile_number", "class_label"]))}) 
    
    # Also save the raw JSON file as an artifact
    results_artifact = wandb.Artifact(f"{vlm_model_name.replace(':', '-')}-{llm_model_name.replace(':', '-')}-multi-question-labels", type="predictions")
    results_artifact.add_file(results_file_path)
    wandb.log_artifact(results_artifact)


def process_batch(batch, 
                  vlm_model_name, 
                  llm_model_name, 
                  questions,
                  prompt, 
                  options, 
                  tokenizer, 
                  text_encoder, 
                  class_embeddings, 
                  class_names_list, 
                  class_dict, 
                  timeout_duration, 
                  device):
    """Process a batch of tile collages."""
    results = {}
    
    # Extract data from batch dictionary
    tile_numbers = batch["tile_number"]
    collages = batch["collage"]
    valids = batch["valid"]
    
    for i in range(len(tile_numbers)):
        tile_number = tile_numbers[i].item()
        tensor_collage = collages[i]
        valid = valids[i].item()
        results[str(tile_number)] = {}
        results[str(tile_number)]["questions"] = {}
        
        if not valid:
            results[str(tile_number)]["class_label"] = -1
            continue
        

            
            # Create a temporary directory for this tile
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a unique temporary file name with PNG extension
            temp_img_path = os.path.join(temp_dir, f"collage_tile_{tile_number}_{uuid.uuid4()}.png")
            
            # Convert tensor back to PIL for saving
            # Denormalize the tensor
            # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            # img_tensor = tensor_collage * std + mean
            # img_tensor = img_tensor.clamp(0, 1)
            
            # Convert to PIL
            pil_image = transforms.ToPILImage()(tensor_collage.cpu())
            pil_image.save(temp_img_path, format="PNG")
            
            for question in questions:
                vlm_prompt = f"Given these set of 8 images of a tile taken over a period of 10 years, answer the following question: {question}"
                
                try:
                    # Set alarm for timeout
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_duration)
                    
                    response = ollama.generate(model=vlm_model_name, prompt=vlm_prompt, images=[temp_img_path], options=options)
                    
                    # Cancel the alarm if successful
                    signal.alarm(0)
                    
                    timed_out = False
                except TimeoutException:
                    print(f"Timeout occurred for tile {tile_number}. Skipping to next sample.")
                    timed_out = True
        
        
        
                if not timed_out:
                    results[str(tile_number)]["questions"][question] = response['response']
                else:
                    results[str(tile_number)]["questions"][question] = "Timeout"

        finally:
            # Clean up the temporary directory and its contents
            shutil.rmtree(temp_dir)

        context_prompt = "Your answer is only one word: Yes or No. Do not mention any other words or phrases."
        context_embedding = generate_context_embedding(context_prompt, llm_model_name, options)
        llm_prompt = f"Given these questions and their respective answers, do you think an ephemeral gully is formed in the region in discusion? {results[str(tile_number)]['questions']}"
        # print(llm_prompt)      
        try:
            # Set alarm for timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_duration)
            
            response = ollama.generate(model=llm_model_name, prompt=llm_prompt, context=context_embedding, options=options)
            
            # Cancel the alarm if successful
            signal.alarm(0)
            
            timed_out = False
        except TimeoutException:
            print(f"Timeout occurred for tile {tile_number}. Skipping to next sample.")
            timed_out = True

        if not timed_out:
            model_response = response['response']
            
            # Remove content between <think> tags
            import re
            cleaned_response = re.sub(r'<think>.*?</think>', '', model_response, flags=re.DOTALL)
            
            # Use the cleaned response for embedding
            query_embedding = get_query_embedding(cleaned_response, tokenizer, text_encoder, device)
            class_name = compute_scores_clip(class_embeddings, query_embedding, class_names_list)
            class_number = class_dict[class_name]
        else:
            # If timeout occurred, set class_number to -1
            class_number = -1
            
        results[str(tile_number)]["class_label"] = class_number
        results[str(tile_number)]["response"] = cleaned_response if not timed_out else "Timeout"
        print(results[str(tile_number)])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run gully detection using PyTorch dataset and multiple workers")

    parser.add_argument("--image_dir", type=str, required=False, help="Base dataset directory")
    parser.add_argument("--vlm_model_name", type=str, required=True, help="VLM model name")
    parser.add_argument("--llm_model_name", type=str, required=True, help="LLM model name")
    parser.add_argument("--prompt", type=str, required=False, help="VLM prompt")
    parser.add_argument("--results_dir", type=str, required=False, help="Folder name to save results")
    parser.add_argument("--timeout", type=int, default=50, help="time out duration to skip one sample")
    parser.add_argument("--model_unloading", action="store_true", help="Enables unloading mode. Every 100 sampels it unloades the model from the GPU to avoid crashing.")
    parser.add_argument("--runname", type=str, required=False)
    parser.add_argument("--projectname", type=str, required=False)
    parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")
    parser.add_argument("--tile", type=int, default=1, help="Tile number to analyze")
    parser.add_argument("--visualize", action="store_true", help="Visualize the collage using matplotlib")
    parser.add_argument("--logging", action="store_true", help="Enable verbose mode")
    parser.add_argument("--savingstep", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    args = parser.parse_args()

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using CUDA with {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available, using CPU")

    class_names = 'No, Yes'
    # model_name = args.model_name
    prompt = args.prompt
    results_dir = args.results_dir
    model_unloading = args.model_unloading
    runname = args.runname
    projectname = args.projectname
    nottest = args.nottest
    logging = args.logging
    num_workers = args.num_workers
    batch_size = args.batch_size
    LLM_Model_name = args.llm_model_name
    VLM_Model_name = args.vlm_model_name

    os.makedirs(results_dir, exist_ok=True)

    if logging:
        wandb.init(project=projectname, name=runname)
        
    print("Pulling Ollama Vision Model...")
    print(VLM_Model_name)
    ollama.pull(VLM_Model_name)
    print("Done Pulling Vision Model..")
    print("Pulling Ollama LLM Model...")
    print(LLM_Model_name)
    ollama.pull(LLM_Model_name)
    print("Done Pulling LLM Model..")
    timeout_duration = args.timeout

    options = {
        "seed": 123,
        "temperature": 0,
        "num_ctx": 2048,  # must be set, otherwise slightly random output
    }

    questions = [
        "Across the images, do you see a low point in the terrain where a temporary gully or depression begins to form without a permanent water feature? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Do narrow, winding paths or channels appear and become more defined across the sequence? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Over time, do any linear depressions or ruts become more pronounced along natural drainage lines or slopes? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Does a narrow, shallow channel become visibly deeper or more indented into the soil over the image sequence? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Are there areas where soil appears disturbed or vegetation is removed, showing progression over the images? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Across the sequence, does a specific path lack vegetation, suggesting an evolving or emerging channel? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Does the texture of the soil appear to change, becoming coarser or showing small pebbles in areas that develop into channels? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Can you see any clear starting and ending points for these channels becoming more evident over time? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Do small rills or grooves indicating water flow appear or deepen across the images? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Is there a gradual exposure of lighter-colored soil as the gully forms? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Are there sediment accumulations forming at the lower ends of channels as time progresses? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Across the images, do signs of water activity, like soil clumps or crusting, appear or intensify in specific areas? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Do shallow side walls along the channel areas become more defined as the sequence progresses? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Do deposits of silt or small debris appear and accumulate along any emerging channel paths? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Can you observe branching patterns that resemble temporary streams forming or intensifying over time? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Do areas with compacted soil or minor slope collapse become more noticeable along the developing channel edges? Please answer with YES or NO Only. DO NOT mention the reason.",
        "Are there indications of nearby human activity, such as tillage or machinery tracks, influencing the formation or expansion of these channels? Please answer with YES or NO Only. DO NOT mention the reason."
    ]

    # Initialize CLIP model
    model_id_clip = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Setting up CLIP..")
    tokenizer = CLIPTokenizer.from_pretrained(model_id_clip)
    text_encoder = CLIPTextModel.from_pretrained(model_id_clip).to(device)
    clip_model = CLIPModel.from_pretrained(model_id_clip).to(device)

    class_names_list = [name.strip() for name in class_names.split(',')]
    class_dict = {class_name: i for i, class_name in enumerate(class_names_list)}
    reverse_class_dict = {v: k for k, v in class_dict.items()}

    class_embeddings = get_class_embeddings(class_names_list, tokenizer, text_encoder, device)
    context_embedding = generate_context_embedding(class_names, VLM_Model_name, options)
    print("Done setting up clip...")
    
    # Get tile numbers
    tile_numbers = get_tile_numbers(args.image_dir)
    print(f"Found {len(tile_numbers)} tiles")
    
    if not nottest:
        # If testing, just use one tile
        tile_numbers = tile_numbers[:1]
    
    # Create transform pipeline for dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = GullyTileDataset(args.image_dir, tile_numbers, transform=transform)
    
    # Initialize dataloader with multiple workers
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Created DataLoader with {num_workers} workers and batch size {batch_size}")
    
    # Process batches
    all_results = {}
    start_time = time.time()

    

    print(prompt)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        batch_results = process_batch(
            batch, 
            vlm_model_name=VLM_Model_name,
            llm_model_name=LLM_Model_name,
            questions=questions,
            prompt=prompt,
            options=options,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            class_embeddings=class_embeddings,
            class_names_list=class_names_list,
            class_dict=class_dict,
            timeout_duration=timeout_duration,
            device=device
        )
        
        # Update all results
        all_results.update(batch_results)
        
        # Save results periodically
        if args.savingstep > 0 and (batch_idx + 1) % (args.savingstep // batch_size) == 0:
            results_file_path = f"{results_dir}/{VLM_Model_name.replace(':', '-')}-{LLM_Model_name.replace(':', '-')}-multi-question-labels.json"
            with open(results_file_path, "w") as f:
                json.dump(all_results, f)
                
            # Save results to wandb if logging is enabled
            if logging:
                save_to_wandb(results_file_path, VLM_Model_name, LLM_Model_name)
            
            print(f"Saved results after processing {(batch_idx + 1) * batch_size} samples")
            
        # Unload model periodically if requested
        if model_unloading and (batch_idx + 1) % (100 // batch_size) == 0:
            print("Unloading model from GPU to avoid memory issues...")
            torch.cuda.empty_cache()
            print("GPU memory cleared")
    
    # Save final results
    results_file_path = f"{results_dir}/{VLM_Model_name.replace(':', '-')}-{LLM_Model_name.replace(':', '-')}-multi-question-labels.json"
    with open(results_file_path, "w") as f:
        json.dump(all_results, f)
        
    # Save results to wandb if logging is enabled
    if logging:
        save_to_wandb(results_file_path, VLM_Model_name)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {len(all_results)} tiles in {elapsed_time:.2f} seconds")
    print(f"Average time per tile: {elapsed_time/len(all_results):.2f} seconds")
    print(f"Final results saved to {results_file_path}")


if __name__ == "__main__":
    # Enable multiprocessing for PyTorch DataLoader
    mp.set_start_method('spawn', force=True)
    main()
