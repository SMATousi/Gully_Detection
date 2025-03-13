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
import os
import rasterio
from utils import *

#ee.Authenticate()
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
# ee.Initialize()
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




with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

placeholders = {
    'huc_number': config['huc_number']
}

config = replace_placeholders(config, placeholders)


rgb_path = config['rgb']['path']
dem_path = config['dem']['path']

# extraction_path = config['kmz_address']['extraction_path']
# kmz_file_path = config['kmz_address']['kmz_file_path']

# bounding_box_coordinates = get_bounding_box_from_kml(extraction_path)
# bounding_box_coordinates = main_extracter(kmz_file_path, extraction_path)



os.makedirs(rgb_path, exist_ok=True)
os.makedirs(dem_path, exist_ok=True)

print(config['rgb']['spec']['filter_dates'])

# print(f"spec : {len(spec_config['point_1'])}")

if config['specs']['from_tif']:
    
    tif_file_path = config['region']['tif_file_path']
    region_points, utm_crs = get_region_points_utm(tif_file_path)

    region = ee.Geometry.Polygon(
        [
            [
            [region_points["point_1"][1], region_points["point_1"][0]],  # (E, N)
            [region_points["point_2"][1], region_points["point_2"][0]],
            [region_points["point_3"][1], region_points["point_3"][0]],
            [region_points["point_4"][1], region_points["point_4"][0]]
            ]
        ],
        None,
        False,
    )

else:
    region = ee.Geometry.Polygon(
        [
            [
                config['region']['point_1'],
                config['region']['point_2'],
                config['region']['point_3'],
                config['region']['point_4'],
            ]
        ],
        None,
        False,
    )

    # region = ee.Geometry.Polygon(
    #     [
    #         [
    #             bounding_box_coordinates[0],
    #             bounding_box_coordinates[1],
    #             bounding_box_coordinates[2],
    #             bounding_box_coordinates[3],
    #         ]
    #     ],
    #     None,
    #     False,
    # )

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
                # 'dimensions': params['dimensions'],
                'format': params['format'],
                'scale': config['dem']['spec']['scale']
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
