import json
import argparse

# Function to load the JSON data
def load_json(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

# Main function
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a JSON label file.")
    parser.add_argument('json_file', type=str, help="Path to the JSON file")

    # Parse the arguments
    args = parser.parse_args()

    # Load the JSON data from the file provided as an argument
    data = load_json(args.json_file)

    # Iterate over the tiles and print the tile number, one label, and prefix
    i = 1
    for tile_number, images in data.items():
        # Get the first image's details
        first_image = next(iter(images.values()))

        # Print tile number, label, and prefix
        print(f"Tile index = {i}, Tile Number: {tile_number} - Label: {first_image['label']}")
        i += 1

if __name__ == "__main__":
    main()