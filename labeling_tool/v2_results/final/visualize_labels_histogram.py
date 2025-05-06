#!/usr/bin/env python3
"""
This script reads the average_labels.json file and creates a histogram
to visualize the distribution of labels.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import collections

def main():
    # Define paths
    base_dir = Path("/root/Gully_Detection/labeling_tool/v2_results/final")
    input_file = base_dir / "average_labels.json"
    output_file = base_dir / "label_histogram.png"
    
    # Read the average labels file
    print(f"Reading average labels from {input_file}")
    with open(input_file, 'r') as f:
        average_data = json.load(f)
    
    # Extract all label values
    all_labels = []
    for tile, tile_data in average_data.items():
        for image, label in tile_data["images"].items():
            all_labels.append(label)
    
    # Convert to numpy array for easier processing
    labels_array = np.array(all_labels)
    
    # Count the occurrences of each label
    label_counts = collections.Counter(all_labels)
    
    # Print some statistics
    print(f"Total number of labels: {len(all_labels)}")
    print(f"Label distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / len(all_labels)) * 100
        print(f"  Label {label}: {count} images ({percentage:.2f}%)")
    
    # Create bins for the histogram
    # Since we might have decimal values from averaging, create appropriate bins
    min_label = min(all_labels)
    max_label = max(all_labels)
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Create two plots: one for all labels and one for rounded labels
    plt.subplot(2, 1, 1)
    
    # Plot histogram with actual averaged values
    bins = np.arange(min_label - 0.25, max_label + 0.5, 0.25)
    plt.hist(all_labels, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Average Labels (Raw Values)', fontsize=14)
    plt.xlabel('Label Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add a second plot with rounded values for a clearer view of the main categories
    plt.subplot(2, 1, 2)
    
    # Round labels to nearest integer for the second plot
    rounded_labels = [round(label) for label in all_labels]
    rounded_counts = collections.Counter(rounded_labels)
    
    # Plot histogram with rounded values
    plt.hist(rounded_labels, bins=np.arange(-0.5, 5.5, 1), alpha=0.7, color='green', edgecolor='black')
    plt.title('Distribution of Average Labels (Rounded to Nearest Integer)', fontsize=14)
    plt.xlabel('Label Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(range(5))  # Assuming labels are 0-4
    plt.grid(True, alpha=0.3)
    
    # Add count labels on top of the bars
    for label, count in sorted(rounded_counts.items()):
        plt.text(label, count + 5, str(count), ha='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    plt.savefig(output_file, dpi=300)
    print(f"Histogram saved to {output_file}")
    
    # Show the plot
    plt.show()
    
    print("Done!")

if __name__ == "__main__":
    main()
