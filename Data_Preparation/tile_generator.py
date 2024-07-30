import os
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import os
import rasterio
from rasterio.windows import Window

def check_rasters_alignment(dem_src, so_src):
    """
    Check if the DEM and SO rasters are aligned in terms of CRS, resolution, and dimensions.
    """
    if dem_src.crs != so_src.crs:
        raise ValueError("CRS mismatch between DEM and SO rasters")
    
    # if dem_src.res != so_src.res:
    #     print(dem_src.res)
    #     print(so_src.res)
    #     raise ValueError("Resolution mismatch between DEM and SO rasters")

    # if dem_src.bounds != so_src.bounds:
    #     print(dem_src.bounds)
    #     print(so_src.bounds)
    #     raise ValueError("Bounds mismatch between DEM and SO rasters")

def save_tile(raster, window, output_path):
    """
    Save a tile from the given raster and window to the specified output path.
    """
    tile = raster.read(1, window=window)
    transform = raster.window_transform(window)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=window.height,
        width=window.width,
        count=1,
        dtype=tile.dtype,
        crs=raster.crs,
        transform=transform,
    ) as dst:
        dst.write(tile, 1)

def generate_tiles(dem_path, so_path, output_dir_dem, output_dir_so, tile_size=(128, 128)):
    """
    Generate 128x128 tiles for both DEM and SO rasters.
    """
    with rasterio.open(dem_path) as dem_src, rasterio.open(so_path) as so_src:
        # Check if rasters are aligned
        check_rasters_alignment(dem_src, so_src)

        for j in tqdm(range(0, dem_src.height, tile_size[0])):
            for i in range(0, dem_src.width, tile_size[1]):
                window = Window(i, j, tile_size[0], tile_size[1])
                dem_tile_path = os.path.join(output_dir_dem, f'dem_tile_{i}_{j}.tif')
                so_tile_path = os.path.join(output_dir_so, f'so_tile_{i}_{j}.tif')

                save_tile(dem_src, window, dem_tile_path)
                save_tile(so_src, window, so_tile_path)





def save_tile(raster, window, output_path):
    tile = raster.read(window=window)
    transform = raster.window_transform(window)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=window.height,
        width=window.width,
        count=raster.count,
        dtype=raster.dtypes[0],
        crs=raster.crs,
        transform=transform,
    ) as dst:
        dst.write(tile)

def has_no_data(tile):
    return (tile == 0).any()

def generate_tiles(dem_path, so_path, rgb_paths, output_dirs, tile_size=(128, 128)):
    with rasterio.open(dem_path) as dem_src, rasterio.open(so_path) as so_src:
        tile_number = 0
        for j in tqdm(range(0, dem_src.height, tile_size[0])):
            for i in range(0, dem_src.width, tile_size[1]):
                window = Window(i, j, tile_size[0], tile_size[1])
                dem_tile = dem_src.read(window=window)
                so_tile = so_src.read(window=window)

                if has_no_data(dem_tile):
                    continue

                rgb_tiles = []
                for rgb_path in rgb_paths:
                    with rasterio.open(rgb_path) as rgb_src:
                        rgb_tile = rgb_src.read(window=window)
                        if has_no_data(rgb_tile):
                            rgb_tiles = []  # Clear the list and break if no-data found
                            break
                        rgb_tiles.append((rgb_tile, rgb_src.profile))

                if not rgb_tiles:  # Skip saving if any no-data found in RGB tiles
                    continue

                dem_tile_path = os.path.join(output_dirs['dem'], f'dem_tile_{tile_number}.tif')
                so_tile_path = os.path.join(output_dirs['so'], f'so_tile_{tile_number}.tif')
                save_tile(dem_src, window, dem_tile_path)
                save_tile(so_src, window, so_tile_path)

                for k, (rgb_tile, rgb_profile) in enumerate(rgb_tiles):
                    rgb_tile_path = os.path.join(output_dirs['rgb'], f'rgb{k}_tile_{tile_number}.tif')
                    with rasterio.open(
                        rgb_tile_path,
                        'w',
                        driver='GTiff',
                        height=window.height,
                        width=window.width,
                        count=rgb_profile['count'],
                        dtype=rgb_profile['dtype'],
                        crs=rgb_profile['crs'],
                        transform=rasterio.windows.transform(window, rgb_profile['transform'])
                    ) as dst:
                        dst.write(rgb_tile)
                
                tile_number = tile_number + 1


# Example usage
dem_path = '/home/macula/SMATousi/Gullies/ground_truth/organized_data/HUCs/HUC_102702060102-done/data/merged/dem_tile__merged.tif'
so_path = '/home/macula/SMATousi/Gullies/ground_truth/organized_data/HUCs/HUC_102702060102-done/data/merged/dem_tile__merged.tif'
rgb_paths = ['/home/macula/SMATousi/Gullies/ground_truth/organized_data/HUCs/HUC_102702060102-done/data/merged/tile_10__merged.tif', 
             '/home/macula/SMATousi/Gullies/ground_truth/organized_data/HUCs/HUC_102702060102-done/data/merged/tile_12__merged.tif', 
             '/home/macula/SMATousi/Gullies/ground_truth/organized_data/HUCs/HUC_102702060102-done/data/merged/tile_14__merged.tif',
             '/home/macula/SMATousi/Gullies/ground_truth/organized_data/HUCs/HUC_102702060102-done/data/merged/tile_16__merged.tif',
             '/home/macula/SMATousi/Gullies/ground_truth/organized_data/HUCs/HUC_102702060102-done/data/merged/tile_18__merged.tif',
             '/home/macula/SMATousi/Gullies/ground_truth/organized_data/HUCs/HUC_102702060102-done/data/merged/tile_20__merged.tif']
output_dirs = {
    'dem': '/home/macula/SMATousi/Gullies/ground_truth/organized_data/tiled_HUCs/HUC_102702060102/dem',
    'so': '/home/macula/SMATousi/Gullies/ground_truth/organized_data/tiled_HUCs/HUC_102702060102/so',
    'rgb': '/home/macula/SMATousi/Gullies/ground_truth/organized_data/tiled_HUCs/HUC_102702060102/rgb'
}

# Create output directories if they don't exist
for dir in output_dirs.values():
    if isinstance(dir, list):
        for d in dir:
            os.makedirs(d, exist_ok=True)
    else:
        os.makedirs(dir, exist_ok=True)

generate_tiles(dem_path, so_path, rgb_paths, output_dirs)