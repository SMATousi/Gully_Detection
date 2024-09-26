import json
import matplotlib.pyplot as plt
import numpy as np

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

# Iterate through the tiles in the first dataset in order
for tile_number, images1 in data1.items():
    if tile_number in data2:
        # Compare the label of the first image in both datasets
        first_image1 = next(iter(images1.values()))
        first_image2 = next(iter(data2[tile_number].values()))
        
        if first_image1['label'] == first_image2['label']:
            agreements.append((tile_number, first_image1['label'], first_image1['prefix']))
        else:
            disagreements.append((tile_number, first_image1['label'], first_image1['prefix'], first_image2['label'], first_image2['prefix']))

# Create the visual key using Matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# Agreement Section
ax.text(0.1, 0.9, 'Agreements:', fontsize=16, fontweight='bold', color='green')

y_pos = 0.85
for tile_number, label, prefix in agreements:
    ax.text(0.1, y_pos, f"Tile {tile_number}: Label = {label}, Prefix = {prefix}", fontsize=12, color='black')
    y_pos -= 0.03

# Disagreement Section
ax.text(0.1, y_pos - 0.05, 'Disagreements:', fontsize=16, fontweight='bold', color='red')
y_pos -= 0.10

for tile_number, label1, prefix1, label2, prefix2 in disagreements:
    ax.text(0.1, y_pos, f"Tile {tile_number}: First file = Label {label1}, Prefix {prefix1}; Second file = Label {label2}, Prefix {prefix2}", 
            fontsize=12, color='black')
    y_pos -= 0.03

# Adjust layout and display the visual key
plt.tight_layout()
plt.show()