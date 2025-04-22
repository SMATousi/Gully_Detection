import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

# File paths
ali_path = os.path.join(os.path.dirname(__file__), 'Ali.json')
krystal_path = os.path.join(os.path.dirname(__file__), 'Krystal.json')
lory_path = os.path.join(os.path.dirname(__file__), 'Dr.Lory.json')

# Load annotation data
def load_annotations(path):
    with open(path, 'r') as f:
        return json.load(f)

ali_data = load_annotations(ali_path)
krystal_data = load_annotations(krystal_path)
lory_data = load_annotations(lory_path)

all_annotators = {'Ali': ali_data, 'Dr.Lory': lory_data, 'Krystal': krystal_data}

# Gather all unique tile numbers and image numbers
tile_numbers = set()
image_numbers = set()
for annot in all_annotators.values():
    tile_numbers.update(annot.keys())
    for tile in annot:
        image_numbers.update(annot[tile]['images'].keys())
tile_numbers = sorted(tile_numbers, key=lambda x: int(x))
image_numbers = sorted(image_numbers, key=lambda x: int(x))

def agreement_stats_per_image():
    results = {}
    for image_num in image_numbers:
        total = 0
        agree = 0
        for tile in tile_numbers:
            labels = []
            for annotator in all_annotators.values():
                label = annotator.get(tile, {}).get('images', {}).get(image_num, None)
                labels.append(label)
            if None not in labels:
                total += 1
                if labels[0] == labels[1] == labels[2]:
                    agree += 1
        percentage = 100 * agree / total if total > 0 else 0
        results[image_num] = {'agree': agree, 'total': total, 'percentage': percentage}
    return results

def plot_agreement_percentages(stats):
    imgs = list(stats.keys())
    percentages = [stats[img]['percentage'] for img in imgs]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(imgs, percentages, color='skyblue')
    plt.ylim(0, 100)
    plt.xlabel('Image Number')
    plt.ylabel('Agreement Percentage (%)')
    plt.title('Agreement Percentage (All 3 Labelers) per Image')
    plt.xticks(rotation=90)
    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

# --- Pairwise agreement calculation and visualization ---
def pairwise_agreement_stats_per_image():
    pairs = [
        ("Ali", "Dr.Lory"),
        ("Ali", "Krystal"),
        ("Dr.Lory", "Krystal")
    ]
    results = {p: {} for p in pairs}
    for image_num in image_numbers:
        for p in pairs:
            total = 0
            agree = 0
            for tile in tile_numbers:
                label1 = all_annotators[p[0]].get(tile, {}).get('images', {}).get(image_num, None)
                label2 = all_annotators[p[1]].get(tile, {}).get('images', {}).get(image_num, None)
                if label1 is not None and label2 is not None:
                    total += 1
                    if label1 == label2:
                        agree += 1
            percentage = 100 * agree / total if total > 0 else 0
            results[p][image_num] = {'agree': agree, 'total': total, 'percentage': percentage}
    return results

def is_flexible_agreement(a, b):
    if a == b:
        return True
    if (a in [0, 1] and b in [0, 1]) or (a in [3, 4] and b in [3, 4]):
        return True
    return False

def pairwise_flexible_agreement_stats_per_image():
    pairs = [
        ("Ali", "Dr.Lory"),
        ("Ali", "Krystal"),
        ("Dr.Lory", "Krystal")
    ]
    results = {p: {} for p in pairs}
    for image_num in image_numbers:
        for p in pairs:
            total = 0
            agree = 0
            for tile in tile_numbers:
                label1 = all_annotators[p[0]].get(tile, {}).get('images', {}).get(image_num, None)
                label2 = all_annotators[p[1]].get(tile, {}).get('images', {}).get(image_num, None)
                if label1 is not None and label2 is not None:
                    total += 1
                    if is_flexible_agreement(label1, label2):
                        agree += 1
            percentage = 100 * agree / total if total > 0 else 0
            results[p][image_num] = {'agree': agree, 'total': total, 'percentage': percentage}
    return results

def plot_pairwise_agreement_percentages(pairwise_stats, title):
    pairs = list(pairwise_stats.keys())
    num_imgs = len(image_numbers)
    plt.figure(figsize=(12, 6))
    width = 0.2
    x = np.arange(num_imgs)
    for i, p in enumerate(pairs):
        percentages = [pairwise_stats[p][img]['percentage'] for img in image_numbers]
        plt.bar(x + i*width, percentages, width, label=f"{p[0]} vs {p[1]}")
    plt.ylim(0, 100)
    plt.xlabel('Image Number')
    plt.ylabel('Agreement Percentage (%)')
    plt.title(title)
    plt.xticks(x + width, image_numbers, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- New: Exclude tiles where any labeler labeled all images as 2 ---
def get_excluded_tiles():
    excluded = set()
    for tile in tile_numbers:
        for annot_name, annot in all_annotators.items():
            labels = [annot.get(tile, {}).get('images', {}).get(img, None) for img in image_numbers]
            # Only consider if all images are present
            if all(label is not None for label in labels):
                if all(label == 2 for label in labels):
                    excluded.add(tile)
    return excluded

def pairwise_strict_exclude2_stats_per_image():
    pairs = [
        ("Ali", "Dr.Lory"),
        ("Ali", "Krystal"),
        ("Dr.Lory", "Krystal")
    ]
    excluded_tiles = get_excluded_tiles()
    results = {p: {} for p in pairs}
    for image_num in image_numbers:
        for p in pairs:
            total = 0
            agree = 0
            for tile in tile_numbers:
                if tile in excluded_tiles:
                    continue
                label1 = all_annotators[p[0]].get(tile, {}).get('images', {}).get(image_num, None)
                label2 = all_annotators[p[1]].get(tile, {}).get('images', {}).get(image_num, None)
                if label1 is not None and label2 is not None:
                    total += 1
                    if label1 == label2:
                        agree += 1
            percentage = 100 * agree / total if total > 0 else 0
            results[p][image_num] = {'agree': agree, 'total': total, 'percentage': percentage}
    return results

def pairwise_flexible_exclude2_stats_per_image():
    pairs = [
        ("Ali", "Dr.Lory"),
        ("Ali", "Krystal"),
        ("Dr.Lory", "Krystal")
    ]
    excluded_tiles = get_excluded_tiles()
    results = {p: {} for p in pairs}
    for image_num in image_numbers:
        for p in pairs:
            total = 0
            agree = 0
            for tile in tile_numbers:
                if tile in excluded_tiles:
                    continue
                label1 = all_annotators[p[0]].get(tile, {}).get('images', {}).get(image_num, None)
                label2 = all_annotators[p[1]].get(tile, {}).get('images', {}).get(image_num, None)
                if label1 is not None and label2 is not None:
                    total += 1
                    if is_flexible_agreement(label1, label2):
                        agree += 1
            percentage = 100 * agree / total if total > 0 else 0
            results[p][image_num] = {'agree': agree, 'total': total, 'percentage': percentage}
    return results

if __name__ == '__main__':
    stats = agreement_stats_per_image()
    print("Agreement stats per image:")
    for img, stat in stats.items():
        print(f"Image {img}: {stat['agree']} / {stat['total']} tiles agree ({stat['percentage']:.1f}%)")
    plot_agreement_percentages(stats)
    
    pairwise_stats = pairwise_agreement_stats_per_image()
    print("\nSTRICT AGREEMENT (exact match):")
    for p in pairwise_stats:
        print(f"\nPair: {p[0]} vs {p[1]}")
        for img, stat in pairwise_stats[p].items():
            print(f"Image {img}: {stat['agree']} / {stat['total']} tiles agree ({stat['percentage']:.1f}%)")
    plot_pairwise_agreement_percentages(pairwise_stats, 'Pairwise STRICT Agreement Percentage per Image')

    pairwise_flex_stats = pairwise_flexible_agreement_stats_per_image()
    print("\nFLEXIBLE AGREEMENT (0/1 and 3/4 also agree):")
    for p in pairwise_flex_stats:
        print(f"\nPair: {p[0]} vs {p[1]}")
        for img, stat in pairwise_flex_stats[p].items():
            print(f"Image {img}: {stat['agree']} / {stat['total']} tiles agree ({stat['percentage']:.1f}%)")
    plot_pairwise_agreement_percentages(pairwise_flex_stats, 'Pairwise FLEXIBLE Agreement (0/1, 3/4) per Image')

    pairwise_stats_ex2 = pairwise_strict_exclude2_stats_per_image()
    print("\nSTRICT AGREEMENT (excluding tiles where any labeler gave all 2s):")
    for p in pairwise_stats_ex2:
        print(f"\nPair: {p[0]} vs {p[1]}")
        for img, stat in pairwise_stats_ex2[p].items():
            print(f"Image {img}: {stat['agree']} / {stat['total']} tiles agree ({stat['percentage']:.1f}%)")
    plot_pairwise_agreement_percentages(pairwise_stats_ex2, 'STRICT Agreement (excluding all-2 tiles)')

    pairwise_flex_stats_ex2 = pairwise_flexible_exclude2_stats_per_image()
    print("\nFLEXIBLE AGREEMENT (excluding tiles where any labeler gave all 2s):")
    for p in pairwise_flex_stats_ex2:
        print(f"\nPair: {p[0]} vs {p[1]}")
        for img, stat in pairwise_flex_stats_ex2[p].items():
            print(f"Image {img}: {stat['agree']} / {stat['total']} tiles agree ({stat['percentage']:.1f}%)")
    plot_pairwise_agreement_percentages(pairwise_flex_stats_ex2, 'FLEXIBLE Agreement (excluding all-2 tiles)')
