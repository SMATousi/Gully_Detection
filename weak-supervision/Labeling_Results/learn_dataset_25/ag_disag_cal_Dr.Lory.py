import json
import argparse

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

# Main function
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare two JSON label files.")
    parser.add_argument('json_file_2', type=str, help="Path to the second JSON file (Your labels)")

    # Parse the arguments
    args = parser.parse_args()

    # Load the two JSON label files
    data1 = load_json("./Dr.Lory_labels.json")
    data2 = load_json(args.json_file_2)

    # Initialize lists for agreement and disagreement
    agreements = []
    disagreements = []
    two_level_disagreements = 0
    more_than_two_level_disagreements = 0

    # Iterate through the tiles in the first dataset
    for tile_number, images1 in data1.items():
        if tile_number in data2:
            # Compare the label of the first image in both datasets
            first_image1 = next(iter(images1.values()))
            first_image2 = next(iter(data2[tile_number].values()))
            
            if first_image1['label'] == first_image2['label']:
                agreements.append((tile_number, first_image1['label'], first_image1['prefix']))
            else:
                disagreements.append((tile_number, first_image1['label'], first_image1['prefix'], first_image2['label'], first_image2['prefix']))

                # Calculate the absolute difference between the labels to check for disagreement levels
                if abs(int(first_image1['label']) - int(first_image2['label'])) == 1:
                    two_level_disagreements += 1
                else:
                    more_than_two_level_disagreements += 1

    total_tiles = len(data2)
    total_agreements = len(agreements)
    total_disagreements = len(disagreements)

    # Calculate percentages
    agreement_percentage = (total_agreements / total_tiles) * 100
    disagreement_percentage = (total_disagreements / total_tiles) * 100
    two_level_disagreement_percentage = (two_level_disagreements / total_tiles) * 100 if total_disagreements > 0 else 0
    more_than_two_level_disagreement_percentage = (more_than_two_level_disagreements / total_tiles) * 100 if total_disagreements > 0 else 0

    # Print all agreements
    print("\nAgreements:")
    for tile_number, label, prefix in agreements:
        tile_index = tile_index_dict.get(tile_number, "Unknown")
        print(f"Tile Index = [{tile_index}], Number {tile_number}: label = {label}")

    # Print all disagreements
    print("\nDisagreements:")
    for tile_number, label1, prefix1, label2, prefix2 in disagreements:
        tile_index = tile_index_dict.get(tile_number, "Unknown")
        print(f"Tile Index = [{tile_index}], Number {tile_number}: Dr. Lory label = {label1}; Your label = {label2}")

    # Print the summary percentages
    print(f"\nTotal tiles: {total_tiles}")
    print(f"Agreements: {total_agreements} ({agreement_percentage:.2f}%)")
    print(f"Disagreements: {total_disagreements} ({disagreement_percentage:.2f}%)")
    print(f"Two-level disagreements: {two_level_disagreements} ({two_level_disagreement_percentage:.2f}%)")
    print(f"More than two-level disagreements: {more_than_two_level_disagreements} ({more_than_two_level_disagreement_percentage:.2f}%)")

if __name__ == "__main__":
    main()
