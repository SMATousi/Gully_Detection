import rasterio
from rasterio.merge import merge
import glob
import os
import yaml
from utils import *
from tqdm import tqdm


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

placeholders = {
    'huc_number': config['huc_number']
}

config = replace_placeholders(config, placeholders)


rgb_path = config['rgb']['path']
dem_path = config['dem']['path']
MSIDS_path = config['MSDIS']['path_to_merge']
output_path = config['merge']['path']

# Function to merge TIFF files based on a common prefix within a specified directory
def merge_tiffs_by_prefix(prefix, input_dir, output_dir='merged_tiffs', MSDIS=False):
    # Create the directory for the merged files if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Search for TIFF files with the specified prefix in the given directory
    if MSDIS:
        search_criteria = os.path.join(input_dir, f"*.tif")
        tif_files = glob.glob(search_criteria)
    else:
        search_criteria = os.path.join(input_dir, f"{prefix}*.tif")
        tif_files = glob.glob(search_criteria)

    if not tif_files:
        print(f"No TIFF files found with prefix '{prefix}' in directory '{input_dir}'")
        return

    # List to store the datasets
    datasets = []

    # Open each TIFF file and append it to the list
    for tif in tqdm(tif_files):
        src = rasterio.open(tif)
        
        # Check the pixel height (transform[4]) and flip if necessary
        transform = src.transform
        if transform[4] < 0:  # Negative pixel height (upside-down raster)
            print(f"Flipping raster {tif} due to negative pixel height.")
            
            # Flip the raster by updating the transform (multiply the pixel height by -1)
            new_transform = rasterio.Affine(transform[0], transform[1], transform[2],
                                            transform[3], -transform[4], transform[5])
            
            # We will modify the src object by creating a flipped raster in memory.
            # No need to manually adjust the data, just flip the transform for merging.
            src = src.read()  # Read the raster data (numpy array)
            # Store the modified transform and dataset
            datasets.append((src, new_transform))
        else:
            # No flip needed, just add the dataset to the list
            datasets.append(src)

    # Merge the datasets into a single raster
    # Use the dataset objects directly (not the numpy arrays)
    # Extract the datasets from the list and merge them
    datasets_to_merge = [dataset for dataset in datasets]
    mosaic, out_trans = merge(datasets_to_merge)

    # Copy the metadata of the first dataset (after flipping or not)
    out_meta = datasets[0].meta.copy()

    # Update the metadata to reflect the dimensions of the merged raster
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    # Write the merged raster to a new TIFF file
    output_file = os.path.join(output_dir, f"{prefix}_merged.tif")
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Merged TIFF file created: {output_file}")

# Example usage
# input_directory = "/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/gully_detection/raw_data/HUC_071100060307/RGB"
# dem_input_dir = "/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/gully_detection/raw_data/HUC_071100060307/DEM"
# output_dir = "/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/gully_detection/raw_data/HUC_071100060307/merged"
# prefixes = ["tile_10_", "tile_12_", "tile_14_", "tile_16_", "tile_18_", "tile_20_"]
# dem_prefix = ["dem_tile_"]
# for prefix in prefixes:
#     merge_tiffs_by_prefix(prefix, rgb_path, output_path)
# for prefix in dem_prefix:
#     merge_tiffs_by_prefix(prefix, dem_path, output_path)
prefix=None
merge_tiffs_by_prefix(prefix, MSIDS_path, output_path, MSDIS=True)
