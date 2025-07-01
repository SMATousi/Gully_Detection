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
import json
from tqdm import tqdm

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



class EightImageDataset_DEM_GT_Geo_from_JSON(Dataset):
    def __init__(self, 
                 pos_dir, 
                 neg_dir, 
                 pos_dem_dir, 
                 neg_dem_dir, 
                 pos_gt_mask_dir, 
                 neg_gt_mask_dir, 
                 labels_json_path,
                 transform=None,
                 oversample=False):
        self.data = []
        self.labels = []
        self.dem_paths = []
        self.gt_mask_paths = []
        self.geo_info = []
        
        # Load labels from JSON file
        with open(labels_json_path, 'r') as f:
            tile_labels = json.load(f)
            
        # Load all files and group by tiles
        pos_files = [f for f in os.listdir(pos_dir) if f.endswith('.tif')]
        neg_files = [f for f in os.listdir(neg_dir) if f.endswith('.tif')]
        pos_tiles = self.group_files_by_tile(pos_files)
        neg_tiles = self.group_files_by_tile(neg_files)
        
        # Process tiles according to labels in JSON file
        # Create a lookup structure mapping tile numbers to their files and source directory
        tile_lookup = {}
        
        # Process positive directory tiles
        for tile_files in pos_tiles:
            tile_number = tile_files[0].split('_')[-1].split('.')[0]
            tile_lookup[tile_number] = {
                'files': tile_files,
                'src_dir': pos_dir,
                'dem_dir': pos_dem_dir,
                'gt_dir': pos_gt_mask_dir
            }
        
        # Process negative directory tiles
        for tile_files in neg_tiles:
            tile_number = tile_files[0].split('_')[-1].split('.')[0]
            tile_lookup[tile_number] = {
                'files': tile_files,
                'src_dir': neg_dir,
                'dem_dir': neg_dem_dir,
                'gt_dir': neg_gt_mask_dir
            }
            
        # Apply labels from JSON and collect labeled tiles
        labeled_pos_tiles = []
        labeled_neg_tiles = []
        
        for tile_number, label in tile_labels.items():
            if tile_number in tile_lookup:
                tile_info = tile_lookup[tile_number]
                if label == 1:
                    labeled_pos_tiles.append((tile_info['files'], tile_info['src_dir'], 
                                            tile_info['dem_dir'], tile_info['gt_dir']))
                else:
                    labeled_neg_tiles.append((tile_info['files'], tile_info['src_dir'], 
                                            tile_info['dem_dir'], tile_info['gt_dir']))
        
        # Handle class imbalance by oversampling the minority class
        if oversample:
            max_len = max(len(labeled_pos_tiles), len(labeled_neg_tiles))
            if len(labeled_pos_tiles) > len(labeled_neg_tiles):
                labeled_neg_tiles = random.choices(labeled_neg_tiles, k=max_len)
            else:
                labeled_pos_tiles = random.choices(labeled_pos_tiles, k=max_len)

        # Store the labeled tiles
        for tile_info in labeled_pos_tiles:
            files, src_dir, dem_dir, gt_dir = tile_info
            self.store_tiles(files, src_dir, dem_dir, gt_dir, 1)
        
        for tile_info in labeled_neg_tiles:
            files, src_dir, dem_dir, gt_dir = tile_info
            self.store_tiles(files, src_dir, dem_dir, gt_dir, 0)
        
        self.transform = transform
        self.resize_high = transforms.Resize((764,764))
        
        print(f"Dataset created with {len(self.data)} samples: {self.labels.count(1)} positive, {self.labels.count(0)} negative")

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

    def store_tiles(self, tile_files, dir_path, dem_dir, gt_mask_dir, label):
        # Make sure we're working with basenames only
        tile_files = [os.path.basename(f) if isinstance(f, str) and os.path.dirname(f) else f for f in tile_files]
            
        tile_number = tile_files[0].split('_')[-1].split('.')[0]
        
        # Check if the files exist in the directory
        file_paths = [os.path.join(dir_path, f) for f in sorted(tile_files)]
        for path in file_paths:
            if not os.path.exists(path):
                print(f"Warning: File does not exist: {path}")
        
        self.data.append(file_paths)
        
        dem_file = os.path.join(dem_dir, f"dem_tile_{tile_number}.tif")
        if not os.path.exists(dem_file):
            print(f"Warning: DEM file does not exist: {dem_file}")
        self.dem_paths.append(dem_file)
        
        # Use appropriate GT mask naming based on label
        if label == 1:
            gt_file = os.path.join(gt_mask_dir, f"ground_truth_tile_{tile_number}.tif")
        else:
            gt_file = os.path.join(gt_mask_dir, f"negative_ground_truth_tile_{tile_number}.tif")            
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth file does not exist: {gt_file}")
        
        self.gt_mask_paths.append(gt_file)
        
        self.labels.append(label)
        
        # Safety check for geo info extraction
        if os.path.exists(file_paths[0]):
            try:
                self.geo_info.append(self.extract_geo_info(file_paths[0]))
            except Exception as e:
                print(f"Error extracting geo info from {file_paths[0]}: {e}")
                # Add a placeholder instead of failing
                self.geo_info.append({'geo_transform': None, 'crs': None})
        else:
            print(f"Cannot extract geo info, file doesn't exist: {file_paths[0]}")
            self.geo_info.append({'geo_transform': None, 'crs': None})

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
        gt_mask = transforms.functional.to_pil_image(imageio.imread(self.dem_paths[idx]).astype('uint8')).convert("RGB")
        
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





class EightImageDataset_WS(Dataset):
    def __init__(self, image_dir,
                 label_model_results_path,
                 transform=None,
                 oversample=False):

        self.image_dir = image_dir
        self.transform = transform
        self.oversample = oversample
        self.label_model_results_path = label_model_results_path
        
        self.data = []
        self.labels = []
        self.geo_info = []

        self.label_model_results = self.load_label_model_results()
        
        # Load all files and group by tiles
        files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        tiles = self.group_files_by_tile(files)

        # Handle class imbalance by oversampling the minority class
        if self.oversample:
            max_len = max(len(tiles))
            tiles = self.oversample(tiles, max_len)

        # Combine and store
        self.store_tiles(tiles, image_dir, 1)
        
        self.resize_high = transforms.Resize((764,764))

    def load_label_model_results(self):
        return json.load(open(self.label_model_results_path, 'r'))

    def group_files_by_tile(self, files):
        tile_dict = {}
        for file in tqdm(files, desc='Grouping files by tile'):
            tile_number = file.split('_')[-1].split('.')[0]
            if tile_number not in tile_dict:
                tile_dict[tile_number] = []
            tile_dict[tile_number].append(file)
        # Only include complete groups
        return [tile for tile in tile_dict.values() if len(tile) == 8]

    def oversample(self, tiles, target_length):
        # Repeat tiles until the desired length is achieved
        return random.choices(tiles, k=target_length)

    def store_tiles(self, tiles, directory, label):
        for tile_files in tqdm(tiles, desc='Storing tiles'):
            tile_number = tile_files[0].split('_')[-1].split('.')[0]
            self.data.append([os.path.join(directory, f) for f in sorted(tile_files)])
            self.labels.append(label)
            # self.geo_info.append(self.extract_geo_info(os.path.join(directory, tile_files[0])))

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
        
        tile_number = self.data[idx][0].split('_')[-1].split('.')[0]
        target_probs = torch.tensor(self.label_model_results[tile_number]['proba'], dtype=torch.float32)
        target_label = torch.tensor(int(self.label_model_results[tile_number]['label']), dtype=torch.float32)

        # geo_info = self.geo_info[idx]

        return images, target_probs, target_label


class EightImageDataset_WS_GT(Dataset):
    def __init__(self, image_dir,
                 GT_labels_path,
                 transform=None,
                 oversample=False):

        self.image_dir = image_dir
        self.transform = transform
        self.oversample = oversample
        self.GT_labels_path = GT_labels_path
        
        self.data = []
        self.labels = []
        self.geo_info = []

        self.GT_labels = self.load_GT_labels()
        
        # Load all files and group by tiles
        files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        tiles = self.group_files_by_tile(files)

        # Handle class imbalance by oversampling the minority class
        if self.oversample:
            max_len = max(len(tiles))
            tiles = self.oversample(tiles, max_len)

        # Combine and store
        self.store_tiles(tiles, image_dir, 1)
        
        self.resize_high = transforms.Resize((764,764))

    def load_GT_labels(self):
        return json.load(open(self.GT_labels_path, 'r'))

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

    def store_tiles(self, tiles, directory, label):
        for tile_files in tiles:
            tile_number = tile_files[0].split('_')[-1].split('.')[0]
            self.data.append([os.path.join(directory, f) for f in sorted(tile_files)])
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
        
        target_label = torch.tensor(int(self.GT_labels[idx]), dtype=torch.float32)

        # geo_info = self.geo_info[idx]

        return images, target_label