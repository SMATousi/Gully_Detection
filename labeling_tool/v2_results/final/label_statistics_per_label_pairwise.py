import json
import os

# File paths
ali_path = os.path.join(os.path.dirname(__file__), 'Ali.json')
lory_path = os.path.join(os.path.dirname(__file__), 'Dr.Lory.json')
krystal_path = os.path.join(os.path.dirname(__file__), 'Krystal.json')

# Load annotation data
def load_annotations(path):
    with open(path, 'r') as f:
        return json.load(f)

ali_data = load_annotations(ali_path)
lory_data = load_annotations(lory_path)
krystal_data = load_annotations(krystal_path)

all_annotators = {'Ali': ali_data, 'Dr.Lory': lory_data, 'Krystal': krystal_data}
annotator_pairs = [("Ali", "Dr.Lory"), ("Ali", "Krystal"), ("Dr.Lory", "Krystal")]

# Gather all labels, images, and tiles
labels_set = set()
tile_numbers = set()
image_numbers = set()
for annot in all_annotators.values():
    tile_numbers.update(annot.keys())
    for tile in annot:
        image_numbers.update(annot[tile]['images'].keys())
        labels_set.update(annot[tile]['images'].values())
tile_numbers = sorted(tile_numbers, key=lambda x: int(x))
image_numbers = sorted(image_numbers, key=lambda x: int(x))
labels_list = sorted(labels_set, key=lambda x: int(x))

def is_flexible_agreement(a, b):
    if a == b:
        return True
    if (a in [0, 1] and b in [0, 1]) or (a in [3, 4] and b in [3, 4]):
        return True
    return False

def get_excluded_tiles():
    excluded = set()
    for tile in tile_numbers:
        for annot in all_annotators.values():
            labels = [annot.get(tile, {}).get('images', {}).get(img, None) for img in image_numbers]
            if all(label is not None for label in labels):
                if all(label == 2 for label in labels):
                    excluded.add(tile)
    return excluded

def count_pairwise_label_agreement_table(mode="strict", exclude2=False):
    pairwise_stats = {pair: {str(label): 0 for label in labels_list} for pair in annotator_pairs}
    excluded_tiles = get_excluded_tiles() if exclude2 else set()
    for tile in tile_numbers:
        if tile in excluded_tiles:
            continue
        for img in image_numbers:
            labels = {labeler: all_annotators[labeler].get(tile, {}).get('images', {}).get(img, None) for labeler in all_annotators}
            for pair in annotator_pairs:
                l1 = labels[pair[0]]
                l2 = labels[pair[1]]
                if l1 is not None and l2 is not None:
                    if mode == "strict":
                        if l1 == l2:
                            pairwise_stats[pair][str(l1)] += 1
                    elif mode == "flexible":
                        if is_flexible_agreement(l1, l2):
                            # Count under the 'base' label (use l1 if equal, else min(l1, l2))
                            if l1 == l2:
                                pairwise_stats[pair][str(l1)] += 1
                            elif (l1 in [0,1] and l2 in [0,1]):
                                pairwise_stats[pair]['0'] += 1
                            elif (l1 in [3,4] and l2 in [3,4]):
                                pairwise_stats[pair]['3'] += 1
    return pairwise_stats

def print_pairwise_label_agreement_table(pairwise_stats, title):
    print(f"\n{title}")
    for pair in pairwise_stats:
        print(f"\nPair: {pair[0]} vs {pair[1]}")
        print("Label | Agreement Count")
        print("------|----------------")
        for label in labels_list:
            print(f"  {label}   |      {pairwise_stats[pair][str(label)]}")

if __name__ == '__main__':
    # Strict
    strict_stats = count_pairwise_label_agreement_table(mode="strict", exclude2=False)
    print_pairwise_label_agreement_table(strict_stats, "STRICT AGREEMENT")
    # Flexible
    flexible_stats = count_pairwise_label_agreement_table(mode="flexible", exclude2=False)
    print_pairwise_label_agreement_table(flexible_stats, "FLEXIBLE AGREEMENT (0/1 and 3/4 also agree)")
    # Strict, excluding all-2 tiles
    strict_ex2_stats = count_pairwise_label_agreement_table(mode="strict", exclude2=True)
    print_pairwise_label_agreement_table(strict_ex2_stats, "STRICT AGREEMENT (excluding all-2 tiles)")
    # Flexible, excluding all-2 tiles
    flexible_ex2_stats = count_pairwise_label_agreement_table(mode="flexible", exclude2=True)
    print_pairwise_label_agreement_table(flexible_ex2_stats, "FLEXIBLE AGREEMENT (excluding all-2 tiles)")
