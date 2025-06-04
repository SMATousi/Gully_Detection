import os
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import time

class GullyTileDataset(Dataset):
    def __init__(self, image_dir, tile_numbers, transform=None):
        self.image_dir = image_dir
        self.tile_numbers = tile_numbers
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.tile_numbers)

    def __getitem__(self, idx):
        tile_number = self.tile_numbers[idx]
        collage = self.collage_images(self.image_dir, tile_number)
        if collage is None:
            collage = Image.new('RGB', (1200, 1200), (0, 0, 0))
            tensor_collage = self.transform(collage)
            return {"tile_number": tile_number, "collage": tensor_collage, "valid": False}
        tensor_collage = self.transform(collage)
        return {"tile_number": tile_number, "collage": tensor_collage, "valid": True}

    def collage_images(self, base_path, tile_number):
        if not base_path.endswith('/'):
            base_path += '/'
        image_patterns = [
            f"rgb_{i}_tile_{tile_number}.tif" for i in range(7)
        ] + [f"rgb_highres_tile_{tile_number}.tif"]
        images = []
        for pattern in image_patterns:
            full_path = os.path.join(base_path, pattern)
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append((pattern, img))
                except Exception as e:
                    print(f"Error loading {pattern}: {e}")
        if not images:
            return None
        num_images = len(images)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
        target_width, target_height = 300, 300
        collage_width = cols * target_width
        collage_height = rows * target_height
        collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(collage)
        for i, (name, img) in enumerate(images):
            row = i // cols
            col = i % cols
            x = col * target_width
            y = row * target_height
            img_resized = img.resize((target_width, target_height), Image.LANCZOS)
            collage.paste(img_resized, (x, y))
            short_names = ["2010", "2012", "2014", "2016", "2018", "2020", "2022", "2023"]
            short_name = short_names[i]
            draw.rectangle([x, y, x + len(short_name)*8, y + 20], fill=(0, 0, 0))
            draw.text((x + 5, y + 5), short_name, fill=(255, 255, 255))
        return collage

def get_tile_numbers(base_path):
    tile_numbers = []
    for filename in os.listdir(base_path):
        if filename.startswith("rgb_0_" + "tile_"):
            tile_number = filename.split("tile_")[1].split(".tif")[0]
            tile_numbers.append(int(tile_number))
    return tile_numbers

def process_batch(batch, hf_model, hf_processor, prompt_text, class_dict, device):
    results = {}
    tile_numbers = batch["tile_number"]
    collages = batch["collage"]
    valids = batch["valid"]

    for i in range(len(tile_numbers)):
        tile_number = tile_numbers[i].item()
        tensor_collage = collages[i]
        valid = valids[i].item()

        if not valid:
            results[str(tile_number)] = -1
            continue

        pil_image = transforms.ToPILImage()(tensor_collage.cpu())
        
        inputs = hf_processor(images=pil_image, text=prompt_text, return_tensors="pt").to(device)
        
        class_number = -1 
        try:
            with torch.no_grad():
                generated_ids = hf_model.generate(**inputs, max_new_tokens=100) # Increased max_new_tokens
            generated_text = hf_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
            print(f"Tile {tile_number}: Model response: '{generated_text}'") # User added print
            if "yes" in generated_text:
                class_number = class_dict["Yes"]
            elif "no" in generated_text:
                class_number = class_dict["No"]
            else:
                print(f"Tile {tile_number}: Unexpected response: '{generated_text}'")
        except Exception as e:
            print(f"Error processing tile {tile_number} with model: {e}")

        results[str(tile_number)] = class_number
    return results

def main():
    parser = argparse.ArgumentParser(description="Run gully detection using a HuggingFace Vision-Language Model")
    parser.add_argument("--image_dir", type=str, required=True, help="Base dataset directory")
    parser.add_argument("--results_dir", type=str, required=True, help="Folder name to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--hf_model_id", type=str, default="Salesforce/blip-vqa-base", help="HuggingFace Model ID (e.g., Salesforce/blip-vqa-base, Salesforce/blip2-opt-2.7b)")
    parser.add_argument("--prompt_text", type=str, default="Is there a gully in this image? Answer with only 'Yes' or 'No'.", help="Prompt/question to ask the model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading HuggingFace model: {args.hf_model_id}")
    try:
        hf_processor = AutoProcessor.from_pretrained(args.hf_model_id)
        hf_model = AutoModelForVision2Seq.from_pretrained(args.hf_model_id).to(device)
    except Exception as e:
        print(f"Error loading model {args.hf_model_id}: {e}")
        print("Please ensure the model ID is correct and the model is suitable for vision-to-sequence tasks.")
        return
    print("HuggingFace model loaded.")

    class_names = 'No, Yes'
    class_dict = {name.strip(): i for i, name in enumerate(class_names.split(','))}
    
    tile_numbers = get_tile_numbers(args.image_dir)
    print(f"Found {len(tile_numbers)} tiles.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = GullyTileDataset(args.image_dir, tile_numbers, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))
    
    all_results = {}
    start_time = time.time()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        batch_results = process_batch(batch, hf_model, hf_processor, args.prompt_text, class_dict, device)
        all_results.update(batch_results)
        
        if (batch_idx + 1) % 10 == 0:
            results_file_path = f"{args.results_dir}/huggingface-vqa-labels.json"
            os.makedirs(args.results_dir, exist_ok=True)
            with open(results_file_path, "w") as f:
                json.dump(all_results, f)
            print(f"Saved intermediate results after processing {(batch_idx + 1) * args.batch_size} samples to {results_file_path}")

    results_file_path = f"{args.results_dir}/huggingface-vqa-labels.json"
    os.makedirs(args.results_dir, exist_ok=True)
    with open(results_file_path, "w") as f:
        json.dump(all_results, f)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {len(all_results)} tiles in {elapsed_time:.2f} seconds")
    print(f"Final results saved to {results_file_path}")

if __name__ == "__main__":
    main()

