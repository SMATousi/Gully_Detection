import ee
import geemap
import logging
import multiprocessing
import os
import requests
import shutil
from retry import retry
import json
import yaml
import rasterio
import os
from rasterio.warp import transform_bounds
from utils import *

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

def get_region_points_utm(tif_path):
    """Extract bounding box in original UTM coordinates."""
    with rasterio.open(tif_path) as dataset:
        tif_crs = dataset.crs.to_string()  # Store UTM projection
        bounds = dataset.bounds  # (left, bottom, right, top)

        # Keep the coordinates in their original UTM projection
        region_points = {
            "point_1": [bounds.top, bounds.left],   # Top-left (N, E)
            "point_2": [bounds.top, bounds.right],  # Top-right (N, E)
            "point_3": [bounds.bottom, bounds.right],  # Bottom-right (N, E)
            "point_4": [bounds.bottom, bounds.left]   # Bottom-left (N, E)
        }
        return region_points, tif_crs  # Return UTM bounding box & CRS

# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

placeholders = {
    'huc_number': config['huc_number']
}

config = replace_placeholders(config, placeholders)

rgb_path = config['rgb']['path']
dem_path = config['dem']['path']

os.makedirs(rgb_path, exist_ok=True)
os.makedirs(dem_path, exist_ok=True)

print(config['rgb']['spec']['filter_dates'])

# Extract bounding box from the TIFF file in UTM
tif_file_path = config['region']['tif_file_path']
region_points, utm_crs = get_region_points_utm(tif_file_path)

# Fix: Add error margin when defining geometry
error_margin = 10  # Set a small non-zero error margin

# Convert to Earth Engine Polygon (Keep UTM coordinates)
region = ee.Geometry.Polygon(
    [
        [
            [region_points["point_1"][1], region_points["point_1"][0]],  # (E, N)
            [region_points["point_2"][1], region_points["point_2"][0]],
            [region_points["point_3"][1], region_points["point_3"][0]],
            [region_points["point_4"][1], region_points["point_4"][0]]
        ]
    ],
    proj=ee.Projection(utm_crs),  # Keep original UTM projection
).buffer(error_margin)  # Fix: Add buffer to define a non-zero error margin

print("Extracted Region Points (UTM):", region_points)
print("UTM Projection:", utm_crs)

if config['rgb']['choose']:

    for i, pair in enumerate(config['rgb']['spec']['filter_dates']):

        Map = geemap.Map()


        # Parameters
        params = {
            'dimensions': config['rgb']['spec']['dimension'],
            'format': config['rgb']['spec']['format'],
            'prefix': config['rgb']['spec']['prefix'][i],
            'processes': 25,
            'out_dir': rgb_path,
        }

        image = (
            ee.ImageCollection('USDA/NAIP/DOQQ')
            # ee.ImageCollection('SKYSAT/GEN-A/PUBLIC/ORTHO/RGB')
            .filterBounds(region)
            .filterDate(pair[0], pair[1])
            .mosaic()
            .clip(region)
            .select('R', 'G', 'B')
            .reproject(ee.Projection(utm_crs))
        )

        # Additional Function to create grid cells covering the region
        def create_grid(region, dx, dy):
            bounds = region.bounds().getInfo()['coordinates'][0]
            minx = min([coord[0] for coord in bounds])
            miny = min([coord[1] for coord in bounds])
            maxx = max([coord[0] for coord in bounds])
            maxy = max([coord[1] for coord in bounds])
            
            grid = []
            x = minx
            while x < maxx:
                y = miny
                while y < maxy:
                    cell = ee.Geometry.Rectangle([x, y, x + dx, y + dy])
                    grid.append(cell)
                    y += dy
                x += dx
            return grid

        # Define your grid size
        dx = 0.02  # Adjust grid size in x
        dy = 0.02  # Adjust grid size in y

        # Create grid
        grid = create_grid(region, dx, dy)

        def download_tile(index, cell):
            region = cell.bounds().getInfo()['coordinates']
            url = image.getDownloadURL(
                {
                    'region': region,
                    # 'dimensions': params['dimensions'],
                    'format': params['format'],
                    'scale': config['rgb']['spec']['scale']
                }
            )
            
            ext = 'tif'
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                r.raise_for_status()

            out_dir = os.path.abspath(params['out_dir'])
            basename = str(index).zfill(len(str(len(grid))))
            filename = f"{out_dir}/{params['prefix']}{basename}.{ext}"
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
            print("Done: ", basename)

        # Run the Download in Parallel
        pool = multiprocessing.Pool(params['processes'])
        pool.starmap(download_tile, enumerate(grid))
        pool.close()


if config['dem']['choose']:

    Map = geemap.Map()


    # Parameters
    params = {
        'dimensions': config['dem']['spec']['dimension'],
        'format': config['dem']['spec']['format'],
        'prefix': config['dem']['spec']['prefix'],
        'processes': 25,
        'out_dir': dem_path
    }

    image = (
        ee.ImageCollection("USGS/3DEP/1m")
        .filterBounds(region)
        .filterDate(config['dem']['spec']['filter_date_2'], config['dem']['spec']['filter_date_1'])
        .mosaic()
        .clip(region)
        .select('elevation')
    )

    # Additional Function to create grid cells covering the region
    def create_grid(region, dx, dy):
        bounds = region.bounds().getInfo()['coordinates'][0]
        minx = min([coord[0] for coord in bounds])
        miny = min([coord[1] for coord in bounds])
        maxx = max([coord[0] for coord in bounds])
        maxy = max([coord[1] for coord in bounds])
        
        grid = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                cell = ee.Geometry.Rectangle([x, y, x + dx, y + dy])
                grid.append(cell)
                y += dy
            x += dx
        return grid

    # Define your grid size
    dx = 0.02  # Adjust grid size in x
    dy = 0.02  # Adjust grid size in y

    # Create grid
    grid = create_grid(region, dx, dy)

    def download_tile(index, cell):
        region = cell.bounds().getInfo()['coordinates']
        url = image.getDownloadURL(
            {
                'region': region,
                'dimensions': params['dimensions'],
                'format': params['format'],
            }
        )
        
        ext = 'tif'
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()

        out_dir = os.path.abspath(params['out_dir'])
        basename = str(index).zfill(len(str(len(grid))))
        filename = f"{out_dir}/{params['prefix']}{basename}.{ext}"
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print("Done: ", basename)

    # Run the Download in Parallel
    pool = multiprocessing.Pool(params['processes'])
    pool.starmap(download_tile, enumerate(grid))
    pool.close()