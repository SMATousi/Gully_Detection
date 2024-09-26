import json
import argparse
from collections import defaultdict

# Function to load the JSON data from a file
def load_json(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

# Tile index number dictionary (extracted from the previous key)
tile_index_dict = {
    "570": 1, "508": 2, "388": 3, "258": 4, "113": 5, "1069": 6, "591": 7, "238": 8, 
    "638": 9, "321": 10, "680": 11, "477": 12, "262": 13, "66": 14, "961": 15, "936": 16, 
    "245": 17, "548": 18, "800": 19, "18": 20, "1060": 21, "363": 22, "738": 23, "1136": 24, 
    "974": 25, "674": 26, "620": 27, "414": 28, "506": 29, "1008": 30, "1101": 31, "845": 32, 
    "699": 33, "1066": 34, "278": 35, "1139": 36, "1109": 37, "56": 38, "613": 39, "875": 40, 
    "122": 41, "775": 42, "133": 43, "170": 44, "284": 45, "318": 46, "340": 47, "46": 48, 
    "169": 49, "969": 50
}

# Function to apply label tolerance (treat 3 as 4 and 1 as 0)
def apply_label_tolerance(label):
    if label == 3:
        return 4
    elif label == 1:
        return 0
    return label

# Function to calculate agreement and disagreement levels
def compare_labels(datasets, with_tolerance=False):
    total_tiles = len(datasets[0])
    agreements = defaultdict(list)
    
    # Iterate through all the tiles in the first dataset
    for tile_number in datasets[0]:
        labels = []
        for dataset in datasets:
            if tile_number in dataset:
                first_image = next(iter(dataset[tile_number].values()))
                label = int(first_image['label'])
                if with_tolerance:
                    label = apply_label_tolerance(label)
                labels.append(label)

        # Check for agreement with or without tolerance
        if len(set(labels)) == 1:
            agreements[tile_number] = labels

    return agreements, total_tiles

# Main function
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare multiple JSON label files.")
    parser.add_argument('json_files', nargs='+', type=str, help="Paths to the JSON files to compare")

    # Parse the arguments
    args = parser.parse_args()

    if len(args.json_files) < 2:
        print("Please provide at least 2 JSON files for comparison.")
        return

    # Load all the JSON label files
    datasets = [load_json(file) for file in args.json_files]

    # Calculate agreement without tolerance
    agreements_without_tolerance, total_tiles = compare_labels(datasets, with_tolerance=False)
    
    # Calculate agreement with tolerance (3 as 4 and 1 as 0)
    agreements_with_tolerance, _ = compare_labels(datasets, with_tolerance=True)

    # Print agreements without tolerance
    print("\nAgreements without tolerance:")
    for tile_number, labels in agreements_without_tolerance.items():
        tile_index = tile_index_dict.get(tile_number, "Unknown")
        print(f"Tile Index = [{tile_index}], Number {tile_number}: labels = {labels}")

    # Print agreements with tolerance
    print("\nAgreements with tolerance (3 as 4 and 1 as 0):")
    for tile_number, labels in agreements_with_tolerance.items():
        tile_index = tile_index_dict.get(tile_number, "Unknown")
        print(f"Tile Index = [{tile_index}], Number {tile_number}: labels = {labels}")

    # Print summary
    total_agreements_without_tolerance = len(agreements_without_tolerance)
    total_agreements_with_tolerance = len(agreements_with_tolerance)

    print(f"\nTotal tiles: {total_tiles}")
    print(f"Agreements without tolerance: {total_agreements_without_tolerance/total_tiles}")
    print(f"Agreements with tolerance: {total_agreements_with_tolerance/total_tiles}")

if __name__ == "__main__":
    main()
