#!/usr/bin/env python3
"""
This script reads label data from three JSON files, calculates the average label for each image,
and writes the results to a new JSON file.
"""

import json
import os
import numpy as np
from pathlib import Path

def read_json_file(file_path):
    """Read and parse a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    # Define paths
    base_dir = Path("/root/Gully_Detection/labeling_tool/v2_results/final")
    labeler_files = [
        base_dir / "Ali.json",
        base_dir / "Dr.Lory.json",
        base_dir / "Krystal.json"
    ]
    output_file = base_dir / "average_labels.json"
    
    # Read all labeler files
    print(f"Reading label files...")
    labeler_data = []
    for file_path in labeler_files:
        print(f"  Reading {file_path.name}")
        labeler_data.append(read_json_file(file_path))
    
    # Initialize the output data structure
    average_data = {}
    
    # Process each tile
    print("Processing tiles and calculating averages...")
    # Get all unique tile numbers across all labelers
    all_tiles = set()
    for data in labeler_data:
        all_tiles.update(data.keys())
    
    # For each tile
    for tile in all_tiles:
        average_data[tile] = {"images": {}}
        
        # Get all unique image numbers for this tile across all labelers
        all_images = set()
        for data in labeler_data:
            if tile in data and "images" in data[tile]:
                all_images.update(data[tile]["images"].keys())
        
        # For each image
        for image in all_images:
            # Collect labels from all labelers
            labels = []
            for data in labeler_data:
                if (tile in data and 
                    "images" in data[tile] and 
                    image in data[tile]["images"]):
                    labels.append(data[tile]["images"][image])
            
            # Calculate the average if we have labels
            if labels:
                # Round to nearest integer
                avg_label = sum(labels) / len(labels)
                average_data[tile]["images"][image] = avg_label
    
    # Write the output file
    print(f"Writing results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(average_data, f, indent=4)
    
    print("Done!")
    
    # Print some statistics
    total_tiles = len(average_data)
    total_images = sum(len(tile_data["images"]) for tile_data in average_data.values())
    print(f"Processed {total_tiles} tiles with a total of {total_images} images.")

if __name__ == "__main__":
    main()
