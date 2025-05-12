#!/usr/bin/env python
import json
import os
import glob
from collections import defaultdict

def create_binary_classification(input_file_path):
    """
    For each tile, assign the majority label among Ali, Dr_Lory, and Krystal.
    Example input for a tile:
    {
        "Ali": 1,
        "Dr_Lory": 0,
        "Krystal": 0
    }
    Output: {"tile_number": majority_label}
    """
    print(f"Processing file: {input_file_path}")
    base_name = os.path.basename(input_file_path)
    output_name = os.path.splitext(base_name)[0] + "_majority.json"
    output_path = os.path.join(os.path.dirname(input_file_path), output_name)

    try:
        with open(input_file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} tiles from {base_name}")
    except Exception as e:
        print(f"Error loading file {input_file_path}: {e}")
        return

    from collections import Counter
    majority_classifications = {}
    for tile_number, tile_data in data.items():
        if tile_number.startswith("_"):
            continue
        # tile_data should be a dict with keys Ali, Dr_Lory, Krystal
        labels = []
        for labeler in ["Ali", "Dr_Lory", "Krystal"]:
            if labeler in tile_data:
                label = tile_data[labeler]
                if isinstance(label, str) and label.isdigit():
                    label = int(label)
                labels.append(label)
        if labels:
            most_common_label = Counter(labels).most_common(1)[0][0]
            majority_classifications[tile_number] = most_common_label
    # Save output
    with open(output_path, 'w') as f:
        json.dump(majority_classifications, f, indent=2)
    print(f"Wrote majority labels for {len(majority_classifications)} tiles to {output_path}")
    # Save the binary classifications
    with open(output_path, 'w') as f:
        json.dump(majority_classifications, f, indent=2, sort_keys=True)
    
    print(f"Binary classifications saved to: {output_path}")
    print(f"Total tiles processed: {len(majority_classifications)}")
    # Count positive and negative classifications
    pos_count = sum(1 for val in majority_classifications.values() if val == 1)
    neg_count = sum(1 for val in majority_classifications.values() if val == 0)
    print(f"Classification summary: {pos_count} positive tiles, {neg_count} negative tiles")
    
def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create binary classification files from aggregated label data')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to a specific input file to process. If not provided, all JSON files in the agg directory will be processed.')
    
    args = parser.parse_args()
    
    agg_dir = "/root/Gully_Detection/labeling_tool/v2_results/final/agg"
    
    if args.input:
        # Process a specific file
        input_path = args.input
        if not os.path.isabs(input_path):
            # If a relative path is provided, make it absolute
            input_path = os.path.join(agg_dir, input_path)
        
        if os.path.isfile(input_path):
            create_binary_classification(input_path)
        else:
            print(f"Error: Input file {input_path} does not exist.")
    else:
        # Process all JSON files in the agg directory
        json_files = glob.glob(os.path.join(agg_dir, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {agg_dir}")
            return
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            if not os.path.basename(json_file).endswith("_agg.json"):  # Skip files that are already aggregated
                create_binary_classification(json_file)

if __name__ == "__main__":
    main()
