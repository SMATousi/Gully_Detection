import os
import shutil
import random
from tqdm import tqdm

def copy_tile_images(src_dir, dest_dir, tile_number, pos_or_neg):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # Copy all 6 RGB images for the given tile
    for i in range(6):
        filename = f'rgb_{i}_tile_{tile_number}.tif'
        dest_name = f'{pos_or_neg}_rgb_{i}_tile_{tile_number}.tif'
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, dest_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
        else:
            print(f"Warning: {src_file} does not exist!")

def split_and_copy_tiles(pos_dir, neg_dir, output_dir, test_size=450):
    # Create directories for the split
    test_data_dir = os.path.join(output_dir, 'test_data')
    # pos_test_dir = os.path.join(test_data_dir, 'pos')
    # neg_test_dir = os.path.join(test_data_dir, 'neg')

    pos_data_dir = os.path.join(output_dir, 'pos_data')
    neg_data_dir = os.path.join(output_dir, 'neg_data')

    # Get unique tile numbers by scanning filenames in pos and neg folders
    pos_tiles = {filename.split('_tile_')[-1].split('.')[0] for filename in os.listdir(pos_dir) if 'tile' in filename}
    neg_tiles = {filename.split('_tile_')[-1].split('.')[0] for filename in os.listdir(neg_dir) if 'tile' in filename}

    # Convert to sorted lists to maintain a consistent order
    pos_tiles = sorted(pos_tiles)
    neg_tiles = sorted(neg_tiles)

    # Shuffle the tile numbers
    random.shuffle(pos_tiles)
    random.shuffle(neg_tiles)

    # Select 450 tiles (each containing 6 images) for the test set
    pos_test_tiles = pos_tiles[:test_size]
    neg_test_tiles = neg_tiles[:test_size]

    # The remaining tiles go to their respective directories
    pos_remaining_tiles = pos_tiles[test_size:]
    neg_remaining_tiles = neg_tiles[test_size:]

    # Copy test tiles (and their 6 images each) to test_data/pos and test_data/neg
    for tile in tqdm(pos_test_tiles):
        copy_tile_images(pos_dir, test_data_dir, tile, 'pos')
    for tile in tqdm(neg_test_tiles):
        copy_tile_images(neg_dir, test_data_dir, tile, 'neg')

    # Copy remaining tiles (and their 6 images each) to pos_data and neg_data
    for tile in tqdm(pos_remaining_tiles):
        copy_tile_images(pos_dir, pos_data_dir, tile, 'pos')
    for tile in tqdm(neg_remaining_tiles):
        copy_tile_images(neg_dir, neg_data_dir, tile, 'neg')

if __name__ == "__main__":
    # Define the source directories for pos and neg images
    pos_dir = "/home/macula/SMATousi/Gullies/ground_truth/organized_data/MO+IA__Test_data/all_pos/rgb_images"
    neg_dir = "/home/macula/SMATousi/Gullies/ground_truth/organized_data/MO+IA__Test_data/all_neg/rgb_images"

    # Define the output directory
    output_dir = "/home/macula/SMATousi/Gullies/ground_truth/organized_data/MO+IA__Test_data/divided_data"

    # Split and copy tiles
    split_and_copy_tiles(pos_dir, neg_dir, output_dir)
