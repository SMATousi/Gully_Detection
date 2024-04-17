import os
import rasterio
from rasterio.windows import Window
import yaml
from tqdm import tqdm

with open('rgb_dem_prep_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


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

def is_crop_land_valid(tile):
    invalid_values_count = ((tile < 0) | ((tile > 61) & (tile != 127))).sum()
    return invalid_values_count <= 128*128 / 2

def find_starting_number(output_dirs):
    highest_number = -1
    for output_dir in output_dirs.values():
        for file in glob.glob(os.path.join(output_dir, '*.tif')):
            file_name = os.path.basename(file)
            parts = [int(part) for part in file_name.split('_') if part.isdigit()]
            if parts:
                highest_number = max(highest_number, max(parts))
    return highest_number + 1

# def generate_tiles(dem_path, rgb_paths, output_dirs, tile_size=(128, 128)):
#     with rasterio.open(dem_path) as dem_src:
#         for j in tqdm(range(0, dem_src.height, tile_size[0])):
#             for i in range(0, dem_src.width, tile_size[1]):
#                 window = Window(i, j, tile_size[0], tile_size[1])
#                 dem_tile = dem_src.read(window=window)
#                 # so_tile = so_src.read(window=window)

#                 if has_no_data(dem_tile):
#                     continue

#                 rgb_tiles = []
#                 for rgb_path in rgb_paths:
#                     with rasterio.open(rgb_path) as rgb_src:
#                         rgb_tile = rgb_src.read(window=window)
#                         if has_no_data(rgb_tile):
#                             rgb_tiles = []  # Clear the list and break if no-data found
#                             break
#                         rgb_tiles.append((rgb_tile, rgb_src.profile))

#                 if not rgb_tiles:  # Skip saving if any no-data found in RGB tiles
#                     continue

#                 dem_tile_path = os.path.join(output_dirs['dem'], f'dem_tile_{i}_{j}.tif')
#                 # so_tile_path = os.path.join(output_dirs['so'], f'so_tile_{i}_{j}.tif')
#                 save_tile(dem_src, window, dem_tile_path)
#                 # save_tile(so_src, window, so_tile_path)

#                 for k, (rgb_tile, rgb_profile) in enumerate(rgb_tiles):
#                     rgb_tile_path = os.path.join(output_dirs['rgb'], f'rgb{k}_tile_{i}_{j}.tif')
#                     with rasterio.open(
#                         rgb_tile_path,
#                         'w',
#                         driver='GTiff',
#                         height=window.height,
#                         width=window.width,
#                         count=rgb_profile['count'],
#                         dtype=rgb_profile['dtype'],
#                         crs=rgb_profile['crs'],
#                         transform=rasterio.windows.transform(window, rgb_profile['transform'])
#                     ) as dst:
#                         dst.write(rgb_tile)

def generate_tiles(dem_path, rgb_paths, output_dirs, tile_size=(128, 128)):
    starting_number = 0
    with rasterio.open(dem_path) as dem_src:
        for j in tqdm(range(0, dem_src.height, tile_size[0])):
            for i in range(0, dem_src.width, tile_size[1]):
                window = Window(i, j, tile_size[0], tile_size[1])
                dem_tile = dem_src.read(window=window)

                if has_no_data(dem_tile):
                    continue

                # cropland_tile = cropland_src.read(1, window=window)
                # if not is_crop_land_valid(cropland_tile):
                #     continue

                # so_tile = so_src.read(window=window)

                rgb_tiles = []
                for rgb_path in rgb_paths:
                    with rasterio.open(rgb_path) as rgb_src:
                        rgb_tile = rgb_src.read(window=window)
                        if has_no_data(rgb_tile):
                            rgb_tiles = []
                            break
                        rgb_tiles.append((rgb_tile, rgb_src.profile))

                if not rgb_tiles:
                    continue

                tile_number = starting_number
                dem_tile_path = os.path.join(output_dirs['dem'], f'dem_tile_{tile_number}.tif')
                # so_tile_path = os.path.join(output_dirs['so'], f'so_tile_{tile_number}.tif')
                save_tile(dem_src, window, dem_tile_path)
                # save_tile(so_src, window, so_tile_path)

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

                starting_number += 1  # Increment the tile number for the next tile


dem_path = config['dem']['path']

rgb_paths = config['rgb']['paths']

output_dirs = {
    'dem': config['dem']['out_path'],
    'rgb': config['rgb']['out_path']
}

# Create output directories if they don't exist
for dir in output_dirs.values():
    if isinstance(dir, list):
        for d in dir:
            os.makedirs(d, exist_ok=True)
    else:
        os.makedirs(dir, exist_ok=True)

generate_tiles(dem_path, rgb_paths, output_dirs)
