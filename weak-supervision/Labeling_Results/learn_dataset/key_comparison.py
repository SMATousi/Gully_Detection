import json

# Function to load the JSON data from a file
def load_json(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

# Load the learning key text file
key_file_path = './learning_key.txt'
json_file_path = './Pankaj_labels.json'

# Load and parse the text key
key_labels = {}
with open(key_file_path, 'r') as f:
    for line in f:
        if 'Number' in line:
            parts = line.split(':')
            tile_number = parts[0].strip().split()[-1]
            print(parts)
            label_part = parts[1].strip()
            if 'Krystal' in label_part or 'Dr. Lory' in label_part:
                labels = label_part.split(';')
                krystal_label = labels[0].split('=')[1].strip().split()[0]
                lory_label = labels[1].split('=')[1].strip().split()[0]
                # Use majority voting for the key label
                if krystal_label == lory_label:
                    key_labels[tile_number] = krystal_label
                else:
                    key_labels[tile_number] = None  # Disagreement between Krystal and Lory
            else:
                key_labels[tile_number] = label_part.split('=')[1].strip().split()[0]

# Load the JSON labels from Pankaj
pankaj_labels = load_json(json_file_path)['content'][0]
pankaj_labels = json.loads(pankaj_labels)

# Compare Pankaj's labels with the key labels
agreement_count = 0
disagreement_count = 0
disagreements = []

for tile_number, key_label in key_labels.items():
    if tile_number in pankaj_labels:
        pankaj_tile_data = pankaj_labels[tile_number]
        pankaj_label = next(iter(pankaj_tile_data.values()))['label']
        
        # Check if key label is set or disagreement between key producers
        if key_label is not None:
            if pankaj_label == key_label:
                agreement_count += 1
            else:
                disagreement_count += 1
                disagreements.append((tile_number, key_label, pankaj_label))

# Calculate percentage of agreement and disagreement
total_tiles = len(key_labels)
agreement_percentage = (agreement_count / total_tiles) * 100
disagreement_percentage = (disagreement_count / total_tiles) * 100

# Output the results
print(f"Agreement count: {agreement_count} ({agreement_percentage:.2f}%)")
print(f"Disagreement count: {disagreement_count} ({disagreement_percentage:.2f}%)")

# Print disagreements
if disagreements:
    print("\nDisagreements:")
    for tile_number, key_label, pankaj_label in disagreements:
        print(f"Tile {tile_number}: Key label = {key_label}, Pankaj label = {pankaj_label}")
else:
    print("\nNo disagreements.")
