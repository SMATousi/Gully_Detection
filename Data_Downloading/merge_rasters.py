import rasterio
from rasterio.merge import merge
import glob
import os
import yaml
from utils import *


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

placeholders = {
    'huc_number': config['huc_number']
}

config = replace_placeholders(config, placeholders)


rgb_path = config['rgb']['path']
dem_path = config['dem']['path']
output_path = config['merge']['path']

# Function to merge TIFF files based on a common prefix within a specified directory
def merge_tiffs_by_prefix(prefix, input_dir, output_dir='merged_tiffs'):
    # Create the directory for the merged files if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Search for TIFF files with the specified prefix in the given directory
    search_criteria = os.path.join(input_dir, f"{prefix}*.tif")
    tif_files = glob.glob(search_criteria)

    if not tif_files:
        print(f"No TIFF files found with prefix '{prefix}' in directory '{input_dir}'")
        return

    # List to store the datasets
    datasets = []

    # Open each TIFF file and append it to the list
    for tif in tif_files:
        src = rasterio.open(tif)
        datasets.append(src)

    # Merge the datasets into a single raster
    mosaic, out_trans = merge(datasets)

    # Copy the metadata of the first dataset
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

    # Close all datasets
    for src in datasets:
        src.close()

    print(f"Merged TIFF saved as {output_file}")

# Example usage
# input_directory = "/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/gully_detection/raw_data/HUC_071100060307/RGB"
# dem_input_dir = "/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/gully_detection/raw_data/HUC_071100060307/DEM"
# output_dir = "/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/gully_detection/raw_data/HUC_071100060307/merged"
prefixes = ["tile_10_", "tile_12_", "tile_14_", "tile_16_", "tile_18_", "tile_20_", "tile_22_"]
dem_prefix = ["dem_tile_"]
for prefix in prefixes:
    merge_tiffs_by_prefix(prefix, rgb_path, output_path)
for prefix in dem_prefix:
    merge_tiffs_by_prefix(prefix, dem_path, output_path)
