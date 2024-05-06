import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import cv2
import imageio

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
        return torch.stack(images), label