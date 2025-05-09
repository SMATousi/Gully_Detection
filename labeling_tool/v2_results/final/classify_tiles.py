#!/usr/bin/env python3
"""
This script analyzes the average_labels.json file and classifies tiles as positive or negative
based on configurable heuristics:

Default heuristics:
- Positive tiles: Tiles that have at least one image with a label of 4 AND at least two other images with labels above 3
- Negative tiles: Any tile where all images have average labels less than 1

The results are written to a new JSON file with a user-provided name.
"""

import json
import argparse
from pathlib import Path
import statistics

def classify_tiles(average_data, positive_heuristic='complex', negative_threshold=1, 
                  positive_threshold=3, min_label_4=1, min_others_above_3=2):
    """
    Classify tiles as positive or negative based on the given thresholds and heuristics.
    
    Args:
        average_data: The data from the average_labels.json file
        positive_heuristic: The type of heuristic to use for positive classification
                           'simple': Any tile with at least one image above positive_threshold
                           'complex': Tiles with at least min_label_4 images with label 4 AND
                                     at least min_others_above_3 other images with labels above 3
        negative_threshold: Tiles with all images below this threshold are negative
        positive_threshold: Used for 'simple' heuristic
        min_label_4: Minimum number of images with label 4 (for 'complex' heuristic)
        min_others_above_3: Minimum number of other images with labels above 3 (for 'complex' heuristic)
        
    Returns:
        A dictionary with positive and negative tiles
    """
    positive_tiles = {}
    negative_tiles = {}
    
    for tile_id, tile_data in average_data.items():
        # Extract all label values for this tile
        image_labels = list(tile_data["images"].values())
        
        # Calculate the average label for the entire tile
        tile_avg = statistics.mean(image_labels) if image_labels else 0
        
        # Apply positive classification based on selected heuristic
        is_positive = False
        
        if positive_heuristic == 'simple':
            # Simple heuristic: any image above threshold
            is_positive = any(label > positive_threshold for label in image_labels)
        
        elif positive_heuristic == 'complex':
            # Complex heuristic: at least one label 4 AND at least two others above 3
            # Count images with label 4
            label_4_count = sum(1 for label in image_labels if label == 4)
            
            # Count images with label above 3 (excluding those counted as label 4)
            above_3_count = sum(1 for label in image_labels if 3 <= label < 4)
            
            # Check if the criteria are met
            is_positive = (label_4_count >= min_label_4 and above_3_count >= min_others_above_3)
        
        # Store positive tiles
        if is_positive:
            positive_tiles[tile_id] = {
                "classification": "positive",
                "average_label": tile_avg,
                "image_labels": tile_data["images"]
            }
        
        # Check if all images have labels less than the negative threshold
        elif all(label < negative_threshold for label in image_labels):
            negative_tiles[tile_id] = {
                "classification": "negative",
                "average_label": tile_avg,
                "image_labels": tile_data["images"]
            }
    
    return {
        "positive_tiles": positive_tiles,
        "negative_tiles": negative_tiles
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Classify tiles based on average labels.')
    parser.add_argument('--input', type=str, default='average_labels.json',
                        help='Input JSON file with average labels (default: average_labels.json)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file name for classification results')
    parser.add_argument('--positive-heuristic', type=str, choices=['simple', 'complex'], default='complex',
                        help='Heuristic for positive classification (default: complex)')
    parser.add_argument('--negative-threshold', type=float, default=1.0,
                        help='Threshold for negative classification (default: 1.0)')
    parser.add_argument('--positive-threshold', type=float, default=3.0,
                        help='Threshold for simple positive classification (default: 3.0)')
    parser.add_argument('--min-label-4', type=int, default=1,
                        help='Minimum number of images with label 4 for complex heuristic (default: 1)')
    parser.add_argument('--min-others-above-3', type=int, default=2,
                        help='Minimum number of other images with labels above 3 for complex heuristic (default: 2)')
    
    args = parser.parse_args()
    
    # Define paths
    base_dir = Path("/root/Gully_Detection/labeling_tool/v2_results/final")
    input_file = base_dir / args.input
    output_file = base_dir / args.output
    
    # Read the average labels file
    print(f"Reading average labels from {input_file}")
    with open(input_file, 'r') as f:
        average_data = json.load(f)
    
    # Classify the tiles
    print(f"Classifying tiles using {args.positive_heuristic} heuristic")
    if args.positive_heuristic == 'complex':
        print(f"  Complex criteria: at least {args.min_label_4} images with label 4 AND at least {args.min_others_above_3} other images with labels above 3")
    else:
        print(f"  Simple criteria: at least one image with label > {args.positive_threshold}")
    print(f"  Negative criteria: all images with labels < {args.negative_threshold}")
    
    classification_results = classify_tiles(
        average_data, 
        positive_heuristic=args.positive_heuristic,
        negative_threshold=args.negative_threshold,
        positive_threshold=args.positive_threshold,
        min_label_4=args.min_label_4,
        min_others_above_3=args.min_others_above_3
    )
    
    # Add summary statistics
    positive_count = len(classification_results["positive_tiles"])
    negative_count = len(classification_results["negative_tiles"])
    total_tiles = len(average_data)
    
    classification_results["summary"] = {
        "total_tiles": total_tiles,
        "positive_tiles_count": positive_count,
        "negative_tiles_count": negative_count,
        "unclassified_tiles_count": total_tiles - positive_count - negative_count,
        "positive_percentage": (positive_count / total_tiles) * 100 if total_tiles > 0 else 0,
        "negative_percentage": (negative_count / total_tiles) * 100 if total_tiles > 0 else 0,
        "unclassified_percentage": ((total_tiles - positive_count - negative_count) / total_tiles) * 100 if total_tiles > 0 else 0,
        "classification_criteria": {
            "positive_heuristic": args.positive_heuristic,
            "negative_threshold": args.negative_threshold,
            "positive_threshold": args.positive_threshold,
            "min_label_4": args.min_label_4,
            "min_others_above_3": args.min_others_above_3
        }
    }
    
    # Write the results to the output file
    print(f"Writing classification results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(classification_results, f, indent=4)
    
    # Print summary
    print("\nClassification Summary:")
    print(f"Total tiles analyzed: {total_tiles}")
    print(f"Positive tiles: {positive_count} ({classification_results['summary']['positive_percentage']:.2f}%)")
    print(f"Negative tiles: {negative_count} ({classification_results['summary']['negative_percentage']:.2f}%)")
    print(f"Unclassified tiles: {classification_results['summary']['unclassified_tiles_count']} ({classification_results['summary']['unclassified_percentage']:.2f}%)")
    print("\nDone!")

if __name__ == "__main__":
    main()
