import json

# Function to load the JSON data from a file
def load_json(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

# Load the two JSON label files
json_file_path_1 = './Krystal_labels.json'
json_file_path_2 = './Dr.Lory_labels.json'  # Replace this with the actual second file path

data1 = load_json(json_file_path_1)
data2 = load_json(json_file_path_2)

# Initialize lists for agreement and disagreement
agreements = []
disagreements = []

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

# Print all agreements
print("\nAgreements:")
for tile_number, label, prefix in agreements:
    print(f"Tile {tile_number}: Both files label = {label}, Prefix = {prefix}")

# Print all disagreements
print("\nDisagreements:")
for tile_number, label1, prefix1, label2, prefix2 in disagreements:
    print(f"Tile {tile_number}: Krystal label = {label1}, Prefix = {prefix1}; Dr. Lory label = {label2}, Prefix = {prefix2}")

