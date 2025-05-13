import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import imageio
import rasterio

class SixImageDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, transform=None):
        self.data = []
        self.labels = []

        # Load all files and group by tiles
        pos_files = [f for f in os.listdir(pos_dir) if f.endswith('.tif')]
        neg_files = [f for f in os.listdir(neg_dir) if f.endswith('.tif')]
        pos_tiles = self.group_files_by_tile(pos_files)
        neg_tiles = self.group_files_by_tile(neg_files)

        # Handle class imbalance by oversampling the minority class
        max_len = max(len(pos_tiles), len(neg_tiles))
        if len(pos_tiles) > len(neg_tiles):
            neg_tiles = self.oversample(neg_tiles, max_len)
        else:
            pos_tiles = self.oversample(pos_tiles, max_len)

        # Combine and store
        self.store_tiles(pos_tiles, pos_dir, 1)
        self.store_tiles(neg_tiles, neg_dir, 0)
        
        self.transform = transform

    def group_files_by_tile(self, files):
        tile_dict = {}
        for file in files:
            tile_number = file.split('_')[-1].split('.')[0]
            if tile_number not in tile_dict:
                tile_dict[tile_number] = []
            tile_dict[tile_number].append(file)
        # Only include complete groups
        return [tile for tile in tile_dict.values() if len(tile) == 6]

    def oversample(self, tiles, target_length):
        # Repeat tiles until the desired length is achieved
        return random.choices(tiles, k=target_length)

    def store_tiles(self, tiles, directory, label):
        for tile_files in tiles:
            self.data.append([os.path.join(directory, f) for f in sorted(tile_files)])
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_1 = [imageio.imread(img_path).astype('uint8') for img_path in self.data[idx]]
        images = [transforms.functional.to_pil_image(image) for image in images_1]
        if self.transform:
            images = [self.transform(image) for image in images]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return images, label

    
class SixImageDataset_DEM_GT(Dataset):
    def __init__(self, pos_dir, 
                 neg_dir, 
                 pos_dem_dir, 
                 neg_dem_dir, 
                 pos_gt_mask_dir, 
                 neg_gt_mask_dir, 
                 transform=None):
        self.data = []
        self.labels = []
        self.dem_paths = []
        self.gt_mask_paths = []

        # Load all files and group by tiles
        pos_files = [f for f in os.listdir(pos_dir) if f.endswith('.tif')]
        neg_files = [f for f in os.listdir(neg_dir) if f.endswith('.tif')]
        pos_tiles = self.group_files_by_tile(pos_files)
        neg_tiles = self.group_files_by_tile(neg_files)

        # Handle class imbalance by oversampling the minority class
        max_len = max(len(pos_tiles), len(neg_tiles))
        if len(pos_tiles) > len(neg_tiles):
            neg_tiles = self.oversample(neg_tiles, max_len)
        else:
            pos_tiles = self.oversample(pos_tiles, max_len)

        # Combine and store
        self.store_tiles(pos_tiles, pos_dir, pos_dem_dir, pos_gt_mask_dir, 1)
        self.store_tiles(neg_tiles, neg_dir, neg_dem_dir, neg_gt_mask_dir, 0)
        
        self.transform = transform

    def group_files_by_tile(self, files):
        tile_dict = {}
        for file in files:
            tile_number = file.split('_')[-1].split('.')[0]
            if tile_number not in tile_dict:
                tile_dict[tile_number] = []
            tile_dict[tile_number].append(file)
        # Only include complete groups
        return [tile for tile in tile_dict.values() if len(tile) == 6]

    def oversample(self, tiles, target_length):
        # Repeat tiles until the desired length is achieved
        return random.choices(tiles, k=target_length)

    def store_tiles(self, tiles, directory, dem_dir, gt_mask_dir, label):
        for tile_files in tiles:
            tile_number = tile_files[0].split('_')[-1].split('.')[0]
            self.data.append([os.path.join(directory, f) for f in sorted(tile_files)])
            self.dem_paths.append(os.path.join(dem_dir, f"dem_tile_{tile_number}.tif"))
            if label == 1:
                self.gt_mask_paths.append(os.path.join(gt_mask_dir, f"ground_truth_tile_{tile_number}.tif"))
            else: 
                self.gt_mask_paths.append(os.path.join(gt_mask_dir, f"negative_ground_truth_tile_{tile_number}.tif"))
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_1 = [imageio.imread(img_path).astype('uint8') for img_path in self.data[idx]]
        images = [transforms.functional.to_pil_image(image) for image in images_1]
        dem_image = transforms.functional.to_pil_image(imageio.imread(self.dem_paths[idx]).astype('uint8'))
        gt_mask = transforms.functional.to_pil_image(imageio.imread(self.gt_mask_paths[idx]).astype('uint8'))
        if self.transform:
            images = [self.transform(image) for image in images]
            dem_image = self.transform(dem_image)
            gt_mask = self.transform(gt_mask)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return images, dem_image, gt_mask, label




class SixImageDataset_DEM_GT_Geo(Dataset):
    def __init__(self, pos_dir, 
                 neg_dir, 
                 pos_dem_dir, 
                 neg_dem_dir, 
                 pos_gt_mask_dir, 
                 neg_gt_mask_dir, 
                 transform=None):
        self.data = []
        self.labels = []
        self.dem_paths = []
        self.gt_mask_paths = []
        self.geo_info = []

        # Load all files and group by tiles
        pos_files = [f for f in os.listdir(pos_dir) if f.endswith('.tif')]
        neg_files = [f for f in os.listdir(neg_dir) if f.endswith('.tif')]
        pos_tiles = self.group_files_by_tile(pos_files)
        neg_tiles = self.group_files_by_tile(neg_files)

        # Handle class imbalance by oversampling the minority class
        max_len = max(len(pos_tiles), len(neg_tiles))
        if len(pos_tiles) > len(neg_tiles):
            neg_tiles = self.oversample(neg_tiles, max_len)
        else:
            pos_tiles = self.oversample(pos_tiles, max_len)

        # Combine and store
        self.store_tiles(pos_tiles, pos_dir, pos_dem_dir, pos_gt_mask_dir, 1)
        self.store_tiles(neg_tiles, neg_dir, neg_dem_dir, neg_gt_mask_dir, 0)
        
        self.transform = transform

    def group_files_by_tile(self, files):
        tile_dict = {}
        for file in files:
            tile_number = file.split('_')[-1].split('.')[0]
            if tile_number not in tile_dict:
                tile_dict[tile_number] = []
            tile_dict[tile_number].append(file)
        # Only include complete groups
        return [tile for tile in tile_dict.values() if len(tile) == 6]

    def oversample(self, tiles, target_length):
        # Repeat tiles until the desired length is achieved
        return random.choices(tiles, k=target_length)

    def store_tiles(self, tiles, directory, dem_dir, gt_mask_dir, label):
        for tile_files in tiles:
            tile_number = tile_files[0].split('_')[-1].split('.')[0]
            self.data.append([os.path.join(directory, f) for f in sorted(tile_files)])
            self.dem_paths.append(os.path.join(dem_dir, f"dem_tile_{tile_number}.tif"))
            if label == 1:
                self.gt_mask_paths.append(os.path.join(gt_mask_dir, f"ground_truth_tile_{tile_number}.tif"))
            else: 
                self.gt_mask_paths.append(os.path.join(gt_mask_dir, f"negative_ground_truth_tile_{tile_number}.tif"))
            self.labels.append(label)
            self.geo_info.append(self.extract_geo_info(os.path.join(directory, tile_files[0])))

    def extract_geo_info(self, file_path):
        with rasterio.open(file_path) as dataset:
            geo_transform = dataset.transform
            crs = dataset.crs
        return {'geo_transform': geo_transform, 'crs': crs}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_1 = [imageio.imread(img_path).astype('uint8') for img_path in self.data[idx]]
        images = [transforms.functional.to_pil_image(image) for image in images_1]
        dem_image = transforms.functional.to_pil_image(imageio.imread(self.dem_paths[idx]).astype('uint8'))
        gt_mask = transforms.functional.to_pil_image(imageio.imread(self.gt_mask_paths[idx]).astype('uint8'))
        if self.transform:
            # Generate a random seed for this tile
            seed = torch.randint(0, 2**32, (1,)).item()
            transformed_images = []
            for image in images:
                torch.manual_seed(seed)
                random.seed(seed)
                transformed_images.append(self.transform(image))
            images = transformed_images
            torch.manual_seed(seed)
            random.seed(seed)
            dem_image = self.transform(dem_image)
            torch.manual_seed(seed)
            random.seed(seed)
            gt_mask = self.transform(gt_mask)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        geo_info = self.geo_info[idx]
        return images, dem_image, gt_mask, label, geo_info



class EightImageDataset_DEM_GT_Geo(Dataset):
    def __init__(self, pos_dir, 
                 neg_dir, 
                 pos_dem_dir, 
                 neg_dem_dir, 
                 pos_gt_mask_dir, 
                 neg_gt_mask_dir, 
                 transform=None,
                 oversample=False):
        self.data = []
        self.labels = []
        self.dem_paths = []
        self.gt_mask_paths = []
        self.geo_info = []

        # Load all files and group by tiles
        pos_files = [f for f in os.listdir(pos_dir) if f.endswith('.tif')]
        neg_files = [f for f in os.listdir(neg_dir) if f.endswith('.tif')]
        pos_tiles = self.group_files_by_tile(pos_files)
        neg_tiles = self.group_files_by_tile(neg_files)

        # Handle class imbalance by oversampling the minority class
        if oversample:
            max_len = max(len(pos_tiles), len(neg_tiles))
            if len(pos_tiles) > len(neg_tiles):
                neg_tiles = self.oversample(neg_tiles, max_len)
            else:
                pos_tiles = self.oversample(pos_tiles, max_len)

        # Combine and store
        self.store_tiles(pos_tiles, pos_dir, pos_dem_dir, pos_gt_mask_dir, 1)
        self.store_tiles(neg_tiles, neg_dir, neg_dem_dir, neg_gt_mask_dir, 0)
        
        self.transform = transform
        self.resize_high = transforms.Resize((764,764))

    def group_files_by_tile(self, files):
        tile_dict = {}
        for file in files:
            tile_number = file.split('_')[-1].split('.')[0]
            if tile_number not in tile_dict:
                tile_dict[tile_number] = []
            tile_dict[tile_number].append(file)
        # Only include complete groups
        return [tile for tile in tile_dict.values() if len(tile) == 8]

    def oversample(self, tiles, target_length):
        # Repeat tiles until the desired length is achieved
        return random.choices(tiles, k=target_length)

    def store_tiles(self, tiles, directory, dem_dir, gt_mask_dir, label):
        for tile_files in tiles:
            tile_number = tile_files[0].split('_')[-1].split('.')[0]
            self.data.append([os.path.join(directory, f) for f in sorted(tile_files)])
            self.dem_paths.append(os.path.join(dem_dir, f"dem_tile_{tile_number}.tif"))
            if label == 1:
                self.gt_mask_paths.append(os.path.join(gt_mask_dir, f"ground_truth_tile_{tile_number}.tif"))
            else: 
                self.gt_mask_paths.append(os.path.join(gt_mask_dir, f"negative_ground_truth_tile_{tile_number}.tif"))
            self.labels.append(label)
            self.geo_info.append(self.extract_geo_info(os.path.join(directory, tile_files[0])))

    def extract_geo_info(self, file_path):
        with rasterio.open(file_path) as dataset:
            geo_transform = dataset.transform
            crs = dataset.crs
        return {'geo_transform': geo_transform, 'crs': crs}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_1 = [imageio.imread(img_path).astype('uint8') for img_path in self.data[idx]]
        images = [transforms.functional.to_pil_image(image).convert("RGB") for image in images_1]
        dem_image = transforms.functional.to_pil_image(imageio.imread(self.dem_paths[idx]).astype('uint8')).convert("RGB")
        gt_mask = transforms.functional.to_pil_image(imageio.imread(self.gt_mask_paths[idx]).astype('uint8'))
        if self.transform:
            # Generate a random seed for this tile
            seed = torch.randint(0, 2**32, (1,)).item()
            transformed_images = []
            for i, image in enumerate(images):
                torch.manual_seed(seed)
                random.seed(seed)
                if i == 1:
                    transformed_images.append(self.transform(self.resize_high(image)))
                else:
                    transformed_images.append(self.transform(image))
            images = transformed_images
            torch.manual_seed(seed)
            random.seed(seed)
            dem_image = self.transform(self.resize_high(dem_image))
            torch.manual_seed(seed)
            random.seed(seed)
            gt_mask = self.transform(gt_mask)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        geo_info = self.geo_info[idx]
        return images, dem_image, gt_mask, label