import os
import matplotlib.pyplot as plt
from dataset import EightImageDataset_DEM_GT_Geo
from torchvision import transforms
import torch

# Example augmentation pipeline (customize as needed)
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(20),
])

# Example paths (edit these to your actual data locations)
pos_dir = '/home/Desktop/train_pos_neg/pos/rgb_images'  # Directory with positive tile images
neg_dir = '/home/Desktop/train_pos_neg/neg/rgb_images'  # Directory with negative tile images
pos_dem_dir = '/home/Desktop/train_pos_neg/pos/dem' # Directory with positive DEM tiles
neg_dem_dir = '/home/Desktop/train_pos_neg/neg/dem' # Directory with negative DEM tiles
pos_gt_mask_dir = '/home/Desktop/train_pos_neg/pos/ground_truth' # Directory with positive GT masks
neg_gt_mask_dir = '/home/Desktop/train_pos_neg/neg/ground_truth' # Directory with negative GT masks

# Instantiate the dataset with augmentation
# You can change 'augmentation' to None to see non-augmented images
dataset = EightImageDataset_DEM_GT_Geo(
    pos_dir=pos_dir,
    neg_dir=neg_dir,
    pos_dem_dir=pos_dem_dir,
    neg_dem_dir=neg_dem_dir,
    pos_gt_mask_dir=pos_gt_mask_dir,
    neg_gt_mask_dir=neg_gt_mask_dir,
    transform=augmentation,
    oversample=False
)

# Pick a sample index to visualize
idx = 11

images, dem_image, gt_mask, label, geo_info = dataset[idx]

for image in images:
    print(image.size)
# Plot all 8 images from the tile
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f'Image {i+1}')
    ax.axis('off')
plt.suptitle(f'Tile Images (Label: {label.item()}) [Augmented]')
plt.show()

# Plot DEM
plt.figure(figsize=(6, 6))
plt.imshow(dem_image, cmap='terrain')
plt.title('DEM Image [Augmented]')
plt.axis('off')
plt.show()

# Plot GT mask
plt.figure(figsize=(6, 6))
plt.imshow(gt_mask, cmap='gray')
plt.title('Ground Truth Mask [Augmented]')
plt.axis('off')
plt.show()

# Print geo info
print('Geo Info:', geo_info)
print('Label:', label)
print('len:', len(dataset))
