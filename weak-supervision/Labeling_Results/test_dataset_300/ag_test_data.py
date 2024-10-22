import json
import sys
from collections import defaultdict
from collections import Counter

# Function to load JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to find common tiles and calculate the percentage of agreement based on a threshold
def find_shared_tiles_and_agreement(labeler_data, threshold):
    # Find common tile numbers across all labelers
    common_tiles = set.intersection(*[set(labeler.keys()) for labeler in labeler_data.values()])
    
    total_common_tiles = len(common_tiles)
    agreed_tiles = 0

    print("\nTiles shared by all labelers and their labels:")
    for tile in common_tiles:
        tile_labels = []
        print(f"Tile {tile}:")
        for labeler, data in labeler_data.items():
            # Just get the first image's label in the tile
            first_image = next(iter(data[tile]))
            label = data[tile][first_image]['label']
            tile_labels.append(label)
            print(f"  {labeler}: Label {label}")

        # Count the frequency of each label
        label_count = Counter(tile_labels)
        most_common_label, most_common_count = label_count.most_common(1)[0]

        # Check if the most common label meets the threshold for agreement
        if most_common_count >= threshold:
            agreed_tiles += 1
    
    # Calculate the percentage of agreement
    agreement_percentage = (agreed_tiles / total_common_tiles) * 100 if total_common_tiles > 0 else 0
    print(f"\nTotal common tiles: {total_common_tiles}")
    print(f"Number of tiles where at least {threshold} labelers agree: {agreed_tiles}")
    print(f"Percentage of agreement: {agreement_percentage:.2f}%")

# Function to list all tiles with at least one label and labeler name
def list_all_tiles(labeler_data):
    all_tiles = defaultdict(list)
    
    for labeler, data in labeler_data.items():
        for tile in data.keys():
            all_tiles[tile].append(labeler)
    
    print("\nList of all tile numbers with at least one label:")
    for tile, labelers in all_tiles.items():
        print(f"Tile {tile} labeled by: {', '.join(labelers)}")
        # continue
    
    return all_tiles

# Main function to process multiple JSON files
def process_labeler_files(file_paths, threshold):
    labeler_data = {}
    
    # Load JSON files
    for file_path in file_paths:
        labeler_name = file_path.split('_')[0]  # Extract labeler name from file name
        labeler_data[labeler_name] = load_json(file_path)
    
    # Find shared tiles and calculate agreement with the threshold
    find_shared_tiles_and_agreement(labeler_data, threshold)
    list_all_tiles(labeler_data)

# Run the program with file paths as arguments and a threshold
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide JSON file paths and the agreement threshold as arguments.")
        sys.exit(1)

    file_paths = sys.argv[1:-1]
    threshold = int(sys.argv[-1])  # The last argument is the threshold
    process_labeler_files(file_paths, threshold)
