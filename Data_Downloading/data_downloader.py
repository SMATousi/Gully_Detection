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

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

rgb_path = config['rgb']['path']
dem_path = config['dem']['path']

# print(f"spec : {len(spec_config['point_1'])}")

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

if config['rgb']['choose']:

    Map = geemap.Map()


    # Parameters
    params = {
        'dimensions': config['rgb']['spec']['dimension'],
        'format': config['rgb']['spec']['format'],
        'prefix': config['rgb']['spec']['prefix'],
        'processes': 25,
        'out_dir': rgb_path,
    }

    image = (
        ee.ImageCollection('USDA/NAIP/DOQQ')
        .filterBounds(region)
        .filterDate(config['rgb']['spec']['filter_date_2'], config['rgb']['spec']['filter_date_1'])
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