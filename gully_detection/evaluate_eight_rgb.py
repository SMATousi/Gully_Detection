from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dataset import EightImageDataset_DEM_GT_Geo
from timm_flexiViT import Flexi_ViT_Gully_Classifier
import argparse
import os
import json
from tqdm import tqdm

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    # Initialize accelerator for distributed processing
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    
    print(f"Device: {device}")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained gully detection model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save prediction results JSON")
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="Confidence threshold for classification (default: 0.3)")
    parser.add_argument("--detailed_output", type=str, help="Path to save detailed prediction results with confidence scores")
    
    args = parser.parse_args()

    pos_dir = '/home/Desktop/final_pos_neg_test_data_merging_25_2/pos/rgb_images'  # Directory with positive tile images
    neg_dir = '/home/Desktop/final_pos_neg_test_data_merging_25_2/neg/rgb_images'  # Directory with negative tile images
    pos_dem_dir = '/home/Desktop/final_pos_neg_test_data_merging_25_2/pos/dem' # Directory with positive DEM tiles
    neg_dem_dir = '/home/Desktop/final_pos_neg_test_data_merging_25_2/neg/dem' # Directory with negative DEM tiles
    pos_gt_mask_dir = '/home/Desktop/final_pos_neg_test_data_merging_25_2/pos/ground_truth' # Directory with positive GT masks
    neg_gt_mask_dir = '/home/Desktop/final_pos_neg_test_data_merging_25_2/neg/ground_truth' # Directory with negative GT masks


    # Define transformations (no augmentation for evaluation)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create test dataset
    test_dataset = EightImageDataset_DEM_GT_Geo(
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        pos_dem_dir=pos_dem_dir,
        neg_dem_dir=neg_dem_dir,
        pos_gt_mask_dir=pos_gt_mask_dir,
        neg_gt_mask_dir=neg_gt_mask_dir,
        transform=transform,
        oversample=False
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")

    # Create model and load checkpoint
    model = Flexi_ViT_Gully_Classifier()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    test_loader = accelerator.prepare(test_loader)

    # Set model to evaluation mode
    model.eval()

    # Dictionary to store predictions
    predictions = {}
    detailed_predictions = {}
    ground_truth = {}

    # Collect all tile numbers
    all_tile_numbers = []
    for tile_files in test_dataset.data:
        # Extract tile number from the first image path in each tile
        tile_path = tile_files[0]
        tile_number = tile_path.split('_')[-1].split('.')[0]
        all_tile_numbers.append(tile_number)

    # Run evaluation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, dem_images, gt_masks, labels = batch
            
            # Get model predictions
            outputs = model(images)
            raw_scores = outputs.squeeze().detach().cpu().numpy()
            
            # Apply confidence threshold for classification
            preds = []
            for score in raw_scores:
                if score >= (0.5 + args.conf_threshold):
                    preds.append(1)  # Confident positive
                elif score <= (0.5 - args.conf_threshold):
                    preds.append(0)  # Confident negative
                else:
                    preds.append(-1)  # Uncertain
            
            # Store batch predictions
            batch_size = labels.size(0)
            for j in range(batch_size):
                batch_idx = i * args.batchsize + j
                if batch_idx < len(all_tile_numbers):
                    tile_number = all_tile_numbers[batch_idx]
                    predictions[tile_number] = int(preds[j])
                    detailed_predictions[tile_number] = {
                        "confidence": float(raw_scores[j]),
                        "classification": int(preds[j])
                    }
                    ground_truth[tile_number] = int(labels[j].cpu().numpy())
                    
                    # For metrics calculation, convert uncertain (-1) to closest class (0)
                    metric_pred = int(preds[j]) if preds[j] != -1 else 0
                    all_preds.append(metric_pred)
                    all_labels.append(int(labels[j].cpu().numpy()))

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save predictions to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {args.output_file}")

    # Save detailed predictions with confidence scores
    detailed_output = args.detailed_output if args.detailed_output else args.output_file.replace('.json', '_detailed.json')
    with open(detailed_output, 'w') as f:
        json.dump(detailed_predictions, f, indent=2)
    
    print(f"Detailed predictions with confidence scores saved to {detailed_output}")

    # Also save ground truth for comparison
    gt_file = args.output_file.replace('.json', '_ground_truth.json')
    with open(gt_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Ground truth saved to {gt_file}")
    
    # Print statistics on classification with confidence threshold
    uncertain_count = sum(1 for val in predictions.values() if val == -1)
    positive_count = sum(1 for val in predictions.values() if val == 1)
    negative_count = sum(1 for val in predictions.values() if val == 0)
    total_count = len(predictions)
    
    print(f"\nClassification with confidence threshold {args.conf_threshold}:")
    print(f"Positive: {positive_count} ({positive_count/total_count*100:.2f}%)")
    print(f"Negative: {negative_count} ({negative_count/total_count*100:.2f}%)")
    print(f"Uncertain: {uncertain_count} ({uncertain_count/total_count*100:.2f}%)")

if __name__ == "__main__":
    main()
