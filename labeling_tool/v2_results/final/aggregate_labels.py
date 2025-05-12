#!/usr/bin/env python
import json
import os
import glob
from collections import defaultdict

def process_labeler_data(file_path, heuristic="h1"):
    """
    Process a labeler's JSON file and apply the selected heuristic for tile labeling.
    
    Heuristic 1 (h1):
    - Label 1: Tile has at least one image with label 4 AND at least one other image with label > 3
    - Label 0: Otherwise
    - Exclude: All images in tile have label 2
    
    Heuristic 2 (h2):
    - Label 0: All images in the tile (excluding those with label 2) have labels that are 0 or 1
    - Label 1: Otherwise
    - Exclude: All images in tile have label 2
    """
    tile_labels = {}
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} tiles from {os.path.basename(file_path)}")
        print(f"Sample data (first 3 tiles):")
        for i, (tile_number, tile_data) in enumerate(data.items()):
            if i < 3:
                print(f"  Tile {tile_number}: {tile_data}")
            else:
                break
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return {}
    
    # Process each tile according to the heuristic
    for tile_number, tile_data in data.items():
        # Get all image labels for this tile
        if "images" in tile_data:
            image_labels = [int(label) for label in tile_data["images"].values()]
        else:
            # Skip tiles without image data
            print(f"Warning: Tile {tile_number} has no images field")
            continue
        
        # Skip tiles where all images have label 2
        if all(label == 2 for label in image_labels):
            continue
        
        # Check if at least one image has label 4 AND at least one other image has label > 3
        has_label_4 = False
        above_3_count = 0
        
        for label in image_labels:
            # if label == 4:
            if label > 2:
                has_label_4 = True
                above_3_count += 1
            elif label > 0:
                above_3_count += 1
        
        # Apply the selected heuristic
        if heuristic == "h1":
            # Heuristic 1: Label 1 if tile has at least one label 4 AND at least one other label > 3
            # if has_label_4 and above_3_count >= 2:
            if has_label_4:
                tile_labels[tile_number] = 1
            else:
                tile_labels[tile_number] = 0
                
            # Print detailed info for first few tiles to verify logic
            if len(tile_labels) <= 5:  
                print(f"Tile {tile_number} (H1): has_label_4={has_label_4}, above_3_count={above_3_count}, verdict={'positive' if tile_labels[tile_number] == 1 else 'negative'}")
        
        elif heuristic == "h2":
            # Heuristic 2: Label 0 if all images (excluding label 2) have labels 0 or 1
            # Filter out any images with label 2
            non_two_labels = [label for label in image_labels if label != 2]
            
            # Check if all remaining labels are 0 or 1
            if all(label <= 0 for label in non_two_labels):
                tile_labels[tile_number] = 0
            else:
                tile_labels[tile_number] = 1
                
            # Print detailed info for first few tiles to verify logic
            if len(tile_labels) <= 5:
                print(f"Tile {tile_number} (H2): all_labels_0_or_1={all(label <= 1 for label in non_two_labels)}, verdict={'negative' if tile_labels[tile_number] == 0 else 'positive'}")
    
    return tile_labels

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Aggregate labeler data using selected heuristic')
    parser.add_argument('--heuristic', type=str, choices=['h1', 'h2'], default='h1',
                        help='Heuristic to use: h1 (default, 1 if tile has label 4 AND another label > 3) or '
                             'h2 (0 if all images are labeled 0 or 1, otherwise 1)')
    args = parser.parse_args()
    
    print(f"Using heuristic: {args.heuristic}")
    if args.heuristic == 'h1':
        print("  - Label 1: Tile has at least one image with label 4 AND at least one other image with label > 3")
        print("  - Label 0: Otherwise")
    else:  # h2
        print("  - Label 0: All images in the tile (excluding those labeled 2) have labels 0 or 1")
        print("  - Label 1: Otherwise")
    
    # Directory containing labeler JSON files
    agg_dir = "/root/Gully_Detection/labeling_tool/v2_results/final/agg"
    
    # Check if directory exists
    if not os.path.exists(agg_dir):
        print(f"Error: Directory not found: {agg_dir}")
        print("Available directories:")
        parent_dir = os.path.dirname(agg_dir)
        if os.path.exists(parent_dir):
            print(os.listdir(parent_dir))
        return
    
    # List all json files in the directory
    print(f"Listing all JSON files in {agg_dir}:")
    json_files = glob.glob(os.path.join(agg_dir, "*.json"))
    for file in json_files:
        print(f"  - {os.path.basename(file)}")
    
    # Define labeler file paths (adjust as needed based on actual filenames)
    labelers = {
        "Ali": os.path.join(agg_dir, "Ali.json"),
        "Krystal": os.path.join(agg_dir, "Krystal.json"),
        "Dr_Lory": os.path.join(agg_dir, "Dr.Lory.json")
    }
    
    # Process each labeler's data
    all_labeler_results = {}
    for labeler_name, file_path in labelers.items():
        if os.path.exists(file_path):
            print(f"\nProcessing {labeler_name}'s labels from: {file_path}")
            all_labeler_results[labeler_name] = process_labeler_data(file_path, args.heuristic)
            print(f"Found {len(all_labeler_results[labeler_name])} valid tiles for {labeler_name}")
        else:
            print(f"\nWarning: File not found for {labeler_name}: {file_path}")
            # Try to find the file using glob pattern
            possible_files = glob.glob(os.path.join(agg_dir, f"*{labeler_name}*.json"))
            if possible_files:
                print(f"Found alternative file for {labeler_name}: {possible_files[0]}")
                all_labeler_results[labeler_name] = process_labeler_data(possible_files[0], args.heuristic)
                print(f"Found {len(all_labeler_results[labeler_name])} valid tiles for {labeler_name}")
            else:
                print(f"No alternative file found for {labeler_name}")
    
    # Combine results into a single dictionary
    combined_results = {}
    # Get all unique tile numbers
    all_tiles = set()
    for labeler_results in all_labeler_results.values():
        all_tiles.update(labeler_results.keys())
    
    # Organize by tile number
    for tile in all_tiles:
        combined_results[tile] = {
            labeler: results.get(tile, None) 
            for labeler, results in all_labeler_results.items()
        }
    
    # Collect statistics
    statistics = {
        "heuristic": args.heuristic,
        "total_tiles": len(all_tiles),
        "labelers": {},
        "agreement": {
            "all_agree": 0,
            "majority_positive": 0,
            "majority_negative": 0,
            "tie": 0
        },
        "overall": {
            "positive_tiles": 0,
            "negative_tiles": 0,
            "null_tiles": 0
        }
    }
    
    # Calculate statistics for each labeler
    for labeler_name, results in all_labeler_results.items():
        positive_count = sum(1 for label in results.values() if label == 1)
        negative_count = sum(1 for label in results.values() if label == 0)
        total_count = len(results)
        
        statistics["labelers"][labeler_name] = {
            "positive_tiles": positive_count,
            "negative_tiles": negative_count,
            "total_tiles": total_count,
            "positive_percentage": round(positive_count / total_count * 100, 2) if total_count > 0 else 0
        }
    
    # Calculate agreement statistics and overall counts
    for tile, labelers_verdict in combined_results.items():
        # Count votes
        positive_votes = sum(1 for verdict in labelers_verdict.values() if verdict == 1)
        negative_votes = sum(1 for verdict in labelers_verdict.values() if verdict == 0)
        null_votes = sum(1 for verdict in labelers_verdict.values() if verdict is None)
        total_votes = len(labelers_verdict)
        
        # Determine agreement
        if null_votes == total_votes:
            statistics["overall"]["null_tiles"] += 1
        elif positive_votes + negative_votes == total_votes:  # No nulls
            if positive_votes == total_votes:
                statistics["agreement"]["all_agree"] += 1
                statistics["overall"]["positive_tiles"] += 1
            elif negative_votes == total_votes:
                statistics["agreement"]["all_agree"] += 1
                statistics["overall"]["negative_tiles"] += 1
            elif positive_votes > negative_votes:
                statistics["agreement"]["majority_positive"] += 1
                statistics["overall"]["positive_tiles"] += 1
            elif negative_votes > positive_votes:
                statistics["agreement"]["majority_negative"] += 1
                statistics["overall"]["negative_tiles"] += 1
            else:  # Equal votes
                statistics["agreement"]["tie"] += 1
                # For ties, we could go either way - here we count as negative
                statistics["overall"]["negative_tiles"] += 1
        else:  # Some nulls, but not all
            if positive_votes > negative_votes:
                statistics["agreement"]["majority_positive"] += 1
                statistics["overall"]["positive_tiles"] += 1
            elif negative_votes > positive_votes:
                statistics["agreement"]["majority_negative"] += 1
                statistics["overall"]["negative_tiles"] += 1
            else:  # Equal votes
                statistics["agreement"]["tie"] += 1
                # For ties, we could go either way - here we count as negative
                statistics["overall"]["negative_tiles"] += 1
    
    # Add statistics to the output
    combined_results["_statistics"] = statistics
    
    # Save combined results
    output_path = f"/root/Gully_Detection/labeling_tool/v2_results/final/tile_binary_labels_{args.heuristic}.json"
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2, sort_keys=True)
    
    print(f"Combined labels saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for labeler, results in all_labeler_results.items():
        positive_count = sum(1 for label in results.values() if label == 1)
        total_count = len(results)
        if total_count > 0:
            print(f"{labeler}: {positive_count}/{total_count} positive tiles ({positive_count/total_count:.2%})")
        else:
            print(f"{labeler}: 0/0 positive tiles (0.00%)")
    
    # Print agreement statistics
    if "_statistics" in combined_results:
        stats = combined_results["_statistics"]
        print("\nAgreement Statistics:")
        print(f"Total tiles: {stats['total_tiles']}")
        print(f"All labelers agree: {stats['agreement']['all_agree']} tiles")
        print(f"Majority positive: {stats['agreement']['majority_positive']} tiles")
        print(f"Majority negative: {stats['agreement']['majority_negative']} tiles")
        print(f"Ties: {stats['agreement']['tie']} tiles")
        print(f"\nOverall analysis:")
        print(f"Positive tiles: {stats['overall']['positive_tiles']}")
        print(f"Negative tiles: {stats['overall']['negative_tiles']}")
        print(f"Null tiles: {stats['overall']['null_tiles']}")

if __name__ == "__main__":
    main()
