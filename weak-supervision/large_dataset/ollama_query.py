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
    similarity_scores = cosine_similarity(query_embedding, class_embeddings, dim=1)  # Shape: [37]
    
    # Find the highest matching score and corresponding item
    max_score_index = torch.argmax(similarity_scores).item()
    max_score = similarity_scores[max_score_index].item()
    best_match = prompts[max_score_index]
    
    # Print the result
   # print(f"Best match: {best_match} with a similarity score of {max_score:.4f}")
    return best_match

    
def compute_best_match(query_text, class_embeddings, class_dict, model_name):
    query_response = ollama.embed(model=model_name, input=query_text)
    query_embedding = query_response["embeddings"]
    query_embedding_tensor = torch.tensor(query_embedding[0])
    
    list_name_emb = list(class_embeddings.keys())
    current_best_score = -1.0  # Start with a low value for cosine similarity
    current_best_match = ""
    
    for class_name in list_name_emb:
        #print(f"Comparing with class: {class_name}")
        class_embeddings_tensor = torch.tensor(class_embeddings[class_name][0])
        # Compute the cosine similarity
        similarity_score = F.cosine_similarity(query_embedding_tensor.unsqueeze(0), class_embeddings_tensor.unsqueeze(0), dim=1)
        
        if similarity_score > current_best_score:
            current_best_score = similarity_score.item()  # Ensure it's a Python float for printing
            current_best_match = class_name
            #print(f"Current best match is: {current_best_match} with score: {current_best_score}")
    matched_label = class_dict[current_best_match] # integer representing class 
    matched_class_name  = current_best_match
    return matched_class_name, matched_label 

def compute_scores(class_embeddings, query_embedding, prompts, temperature=0.8):
    scores = []
    # Compute cosine similarity scores
    for class_name in class_embeddings:
        similarity_scores = cosine_similarity(torch.tensor(query_embedding), torch.tensor(class_embeddings[class_name]), dim=1)  # Shape: [37]
        similarity_scores = similarity_scores / temperature
        scores.append(similarity_scores.item())

    probabilities = F.softmax(torch.tensor(scores), dim=0)
    # Find the highest matching score and corresponding item

    max_prob_index = torch.argmax(probabilities).item()
    max_prob = probabilities[max_prob_index]
    best_match = prompts[max_prob_index]
    
    # Print the result
   # print(f"Best match: {best_match} with a similarity score of {max_score:.4f}")
    return best_match, probabilities, max_prob

def generate_context_embedding(class_names, model_name, options):
    prompt = "You are working on a difficult fine-grained image classification task, here are the only classes you can choose from"+class_names
    context_response = ollama.generate(model=model_name, prompt=prompt, options=options)
    return context_response['context']

def compute_class_embeddings(class_names_list, model_name, device) :
    class_embeddings = {}
    print("Computing the class embeddings --")
    for class_name in tqdm(class_names_list) :
        # print(class_name)
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

def collage_images(base_path, tile_number, visual=False):
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
                # print(f"Loaded {pattern}")
            except Exception as e:
                print(f"Error loading {pattern}: {e}")
    
    if not images:
        print(f"No images found for tile {tile_number}")
        return None
    
    # Determine the grid size
    num_images = len(images)
    cols = min(4, num_images)  # Max 4 images per row
    rows = (num_images + cols - 1) // cols  # Ceiling division
    
    # Resize all images to a common size (use size of first image)
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
        
        if visual and i == len(images) - 1:  # Show visualization after all images are processed
            # Convert collage to numpy array for matplotlib
            collage_array = np.array(collage)
            
            # Create figure with appropriate size
            plt.figure(figsize=(collage_width/100, collage_height/100))  # Size in inches
            plt.imshow(collage_array)
            plt.axis('off')  # Turn off axis
            plt.title(f'Tile {tile_number} Collage')
            plt.tight_layout()
            plt.show()
    
    # print(f"Created collage with {num_images} images for tile {tile_number}")
    return collage


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
    
    # print(f"Results saved to wandb as '{model_name}_labels'")

def main():
    parser = argparse.ArgumentParser(description="Run numerous experiments with varying VLM models on the Cars sign dataset")

    parser.add_argument("--image_dir", type=str, required=False, help="Base dataset directory")
    parser.add_argument("--model_name", type=str, required=True, help=" VLM model name")
    parser.add_argument("--prompt", type=str, required=False, help="VLM prompt")
    parser.add_argument("--results_dir", type=str, required=False, help="Folder name to save results")
    parser.add_argument("--timeout", type=int, default=100, help="time out duration to skip one sample")
    parser.add_argument("--model_unloading", action="store_true", help="Enables unloading mode. Every 100 sampels it unloades the model from the GPU to avoid crashing.")
    parser.add_argument("--runname", type=str, required=False)
    parser.add_argument("--projectname", type=str, required=False)
    parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")
    parser.add_argument("--tile", type=int, default=1, help="Tile number to analyze")
    parser.add_argument("--visualize", action="store_true", help="Visualize the collage using matplotlib")
    parser.add_argument("--logging", action="store_true", help="Enable verbose mode")
    parser.add_argument("--savingstep", type=int, default=100)
    args = parser.parse_args()

    class_names = 'No, Yes'
    model_name  = args.model_name
    prompt = args.prompt
    results_dir = args.results_dir
    model_unloading = args.model_unloading
    runname = args.runname
    projectname = args.projectname
    nottest = args.nottest
    logging = args.logging

    os.makedirs(results_dir, exist_ok=True)

    if logging:
        wandb.init(project=projectname, name=runname)
    print("Pulling Ollama Model...")
    print(model_name)
    ollama.pull(model_name)
    print("Done Pulling..")
    timeout_duration = args.timeout

    options= {  # new
                "seed": 123,
                "temperature": 0,
                "num_ctx": 2048, # must be set, otherwise slightly random output
            }

    model_id_clip  = "openai/clip-vit-large-patch14"
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Setting up CLIP..")
    tokenizer = CLIPTokenizer.from_pretrained(model_id_clip)
    text_encoder = CLIPTextModel.from_pretrained(model_id_clip).to(device)
    clip_model = CLIPModel.from_pretrained(model_id_clip).to(device)

    class_names_list = [name.strip() for name in class_names.split(',')]
    class_dict = {class_name : i for i, class_name in enumerate(class_names_list)}
    reverse_class_dict = {v: k for k, v in class_dict.items()}

    class_embeddings = get_class_embeddings(class_names_list, tokenizer, text_encoder, device)

    context_embedding = generate_context_embedding(class_names, model_name, options)
    print("Done setting up clip...")
    model_labels = {}
    text_length = 50
    count = 0

    # Base image directory
    # image_dir = '/home/Desktop/choroid/large_unlabeled_dataset/rgb_images/'

    tile_numbers = get_tile_numbers(args.image_dir) 
    print(tile_numbers)
    
    # Get tile number from command line
    for tile_number in tqdm(tile_numbers):
        visualize = args.visualize
        
        # Create a collage of all images for this tile
        # print(f"Creating collage for tile {tile_number}...")
        collage = collage_images(args.image_dir, tile_number, visual=visualize)
        
        if collage is None:
            print(f"No images found for tile {tile_number}. Exiting.")
            return
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a unique temporary file name with PNG extension (supported by Ollama)
            temp_img_path = os.path.join(temp_dir, f"collage_tile_{tile_number}_{uuid.uuid4()}.png")
            
            # Save the collage to the temporary path in PNG format
            # print(f"Saving collage to temporary path: {temp_img_path}")
            collage.save(temp_img_path, format="PNG")
            
            # Prepare a more specific prompt related to gully detection
            if not prompt:
                prompt = f"Analyze this collage of satellite imagery (tile {tile_number}). Can you identify any gullies or erosion features? Describe what you see in detail, focusing on possible signs of soil erosion."
            
            # Use the temporary file for ollama
            # print(f"Querying Ollama model with collage image...")
            response = ollama.generate(model=model_name, prompt=prompt, images=[temp_img_path], options=options)
            
            # Print the response
            # print("\nOllama Response:")
            # print(response['response'] if 'response' in response else response)
        
        finally:
            # Clean up the temporary directory and its contents
            # print("Cleaning up temporary files...")
            shutil.rmtree(temp_dir)
        
        model_response = response['response']
        if len(model_response) > text_length : 
            query_prompt = model_response[:text_length]
        else : 
            query_prompt = model_response
        query_embedding = get_query_embedding(query_prompt, tokenizer, text_encoder, device)
        class_name = compute_scores_clip(class_embeddings, query_embedding, class_names_list)
        class_number = class_dict[class_name]
        # print(f"Tile {tile_number}: {class_name} ({class_number})")
        model_labels[str(tile_number)] = class_number

        # print(model_labels)
        count += 1

        
        if args.savingstep > 0 and count % args.savingstep == 0:
            results_file_path = f"{results_dir}/{model_name.replace(':', '-')}-labels.json"
            with open(results_file_path, "w") as f:
                json.dump(model_labels, f)
                
            # Save results to wandb if logging is enabled
            if logging:
                save_to_wandb(results_file_path, model_name)

        if not nottest:
            break


if __name__ == "__main__":
    main()