import os
import argparse
import rasterio
from rasterio.merge import merge

def merge_tif_images(directory, output_file):
    # Build a list of full file paths for GeoTIFF files in the directory
    tif_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.tif')]
    tif_files.sort()  # Optional: sort files if you need a specific order
    
    # Open all the files using rasterio
    src_files_to_mosaic = []
    for fp in tif_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    
    # Merge the files using rasterio.merge
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Copy metadata from one of the input files (assuming they share the same CRS)
    out_meta = src_files_to_mosaic[0].meta.copy()
    
    # Update the metadata to reflect the number of layers, dimensions, and transform of the mosaic
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    
    # Write the mosaic to disk
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close the source datasets
    for src in src_files_to_mosaic:
        src.close()

def main():
    parser = argparse.ArgumentParser(
        description="Merge all GeoTIFF files in a directory into a single mosaic while preserving geospatial metadata."
    )
    parser.add_argument('directory', type=str, help="Directory containing GeoTIFF files")
    parser.add_argument('output', type=str, help="Output filename for the merged GeoTIFF")
    
    args = parser.parse_args()
    merge_tif_images(args.directory, args.output)

if __name__ == "__main__":
    main()
