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
        self.transform = transform
        
    def __len__(self):
        return len(self.tile_numbers)
    
    def __getitem__(self, idx):
        tile_number = self.tile_numbers[idx]
        collage = self.collage_images(self.image_dir, tile_number)
        
        if collage is None:
            # Return a placeholder if no images found
            collage = Image.new('RGB', (1200, 1200), (0, 0, 0))
            return {"tile_number": tile_number, "collage": collage, "valid": False}
        
        if self.transform:
            collage = self.transform(collage)
            
        return {"tile_number": tile_number, "collage": collage, "valid": True}
    
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

def save_to_wandb(results_file_path, model_name):
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
    wandb.log({f"{model_name.replace(':', '-')}-results": wandb.Table(dataframe=pd.DataFrame(list(results_data.items()), columns=["tile_number", "class_label"]))}) 
    
    # Also save the raw JSON file as an artifact
    results_artifact = wandb.Artifact(f"{model_name.replace(':', '-')}-labels", type="predictions")
    results_artifact.add_file(results_file_path)
    wandb.log_artifact(results_artifact)


def process_batch(batch, model_name, prompt, options, tokenizer, text_encoder, 
                 class_embeddings, class_names_list, class_dict, timeout_duration, device):
    """Process a batch of tile collages."""
    results = {}
    
    for item in batch:
        tile_number = item["tile_number"]
        collage = item["collage"]
        valid = item["valid"]
        
        if not valid:
            results[str(tile_number)] = -1
            continue
        
        # Create a temporary directory for this tile
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a unique temporary file name with PNG extension
            temp_img_path = os.path.join(temp_dir, f"collage_tile_{tile_number}_{uuid.uuid4()}.png")
            
            # Save the collage to the temporary path in PNG format
            collage.save(temp_img_path, format="PNG")
            
            # Prepare prompt if not provided
            if not prompt:
                prompt = f"Analyze this collage of satellite imagery (tile {tile_number}). Can you identify any gullies or erosion features? Describe what you see in detail, focusing on possible signs of soil erosion."
            
            try:
                # Set alarm for timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_duration)
                
                response = ollama.generate(model=model_name, prompt=prompt, images=[temp_img_path], options=options)
                
                # Cancel the alarm if successful
                signal.alarm(0)
                
                timed_out = False
            except TimeoutException:
                print(f"Timeout occurred for tile {tile_number}. Skipping to next sample.")
                timed_out = True
        
        finally:
            # Clean up the temporary directory and its contents
            shutil.rmtree(temp_dir)
        
        if not timed_out:
            model_response = response['response']
            text_length = 50
            if len(model_response) > text_length: 
                query_prompt = model_response[:text_length]
            else:
                query_prompt = model_response
            
            query_embedding = get_query_embedding(query_prompt, tokenizer, text_encoder, device)
            class_name = compute_scores_clip(class_embeddings, query_embedding, class_names_list)
            class_number = class_dict[class_name]
        else:
            # If timeout occurred, set class_number to -1
            class_number = -1
            
        results[str(tile_number)] = class_number
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run gully detection using PyTorch dataset and multiple workers")

    parser.add_argument("--image_dir", type=str, required=False, help="Base dataset directory")
    parser.add_argument("--model_name", type=str, required=True, help="VLM model name")
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
    model_name = args.model_name
    prompt = args.prompt
    results_dir = args.results_dir
    model_unloading = args.model_unloading
    runname = args.runname
    projectname = args.projectname
    nottest = args.nottest
    logging = args.logging
    num_workers = args.num_workers
    batch_size = args.batch_size

    os.makedirs(results_dir, exist_ok=True)

    if logging:
        wandb.init(project=projectname, name=runname)
        
    print("Pulling Ollama Model...")
    print(model_name)
    ollama.pull(model_name)
    print("Done Pulling..")
    timeout_duration = args.timeout

    options = {
        "seed": 123,
        "temperature": 0,
        "num_ctx": 2048,  # must be set, otherwise slightly random output
    }

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
    context_embedding = generate_context_embedding(class_names, model_name, options)
    print("Done setting up clip...")
    
    # Get tile numbers
    tile_numbers = get_tile_numbers(args.image_dir)
    print(f"Found {len(tile_numbers)} tiles")
    
    if nottest:
        # If testing, just use one tile
        tile_numbers = tile_numbers[:1]
    
    # Create dataset
    dataset = GullyTileDataset(args.image_dir, tile_numbers)
    
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
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        batch_results = process_batch(
            batch, model_name, prompt, options, tokenizer, text_encoder,
            class_embeddings, class_names_list, class_dict, timeout_duration, device
        )
        
        # Update all results
        all_results.update(batch_results)
        
        # Save results periodically
        if args.savingstep > 0 and (batch_idx + 1) % (args.savingstep // batch_size) == 0:
            results_file_path = f"{results_dir}/{model_name.replace(':', '-')}-labels.json"
            with open(results_file_path, "w") as f:
                json.dump(all_results, f)
                
            # Save results to wandb if logging is enabled
            if logging:
                save_to_wandb(results_file_path, model_name)
            
            print(f"Saved results after processing {(batch_idx + 1) * batch_size} samples")
            
        # Unload model periodically if requested
        if model_unloading and (batch_idx + 1) % (100 // batch_size) == 0:
            print("Unloading model from GPU to avoid memory issues...")
            torch.cuda.empty_cache()
            print("GPU memory cleared")
    
    # Save final results
    results_file_path = f"{results_dir}/{model_name.replace(':', '-')}-labels.json"
    with open(results_file_path, "w") as f:
        json.dump(all_results, f)
        
    # Save results to wandb if logging is enabled
    if logging:
        save_to_wandb(results_file_path, model_name)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {len(all_results)} tiles in {elapsed_time:.2f} seconds")
    print(f"Average time per tile: {elapsed_time/len(all_results):.2f} seconds")
    print(f"Final results saved to {results_file_path}")


if __name__ == "__main__":
    # Enable multiprocessing for PyTorch DataLoader
    mp.set_start_method('spawn', force=True)
    main()
