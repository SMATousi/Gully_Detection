import json

# Path to the JSON file
json_file_path = './Krystal_labels.json'

# Load the JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Iterate over the tiles and print the tile number, one label, and prefix
for tile_number, images in data.items():
    # Get the first image's details
    first_image = next(iter(images.values()))
    
    # Print tile number, label, and prefix
    print(f"Tile Number: {tile_number} - Label: {first_image['label']} - Prefix: {first_image['prefix']}")