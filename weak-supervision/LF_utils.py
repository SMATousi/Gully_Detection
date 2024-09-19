import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from model import *
from dataset import *
from utils import *
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
from collections import Counter


def visualize_images_torch(images):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    if len(images) == 1:
        axes = [axes]
    if images[0].shape[1] == 1:
        for ax, img in zip(axes, images):
            ax.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')  # Change channel order for matplotlib
            ax.axis('off')
        plt.show()
    else:
        for ax, img in zip(axes, images):
            ax.imshow(img.permute(2, 3, 1, 0).squeeze())  # Change channel order for matplotlib
            ax.axis('off')
        plt.show()

def visualize_images_numpy(images, is_grayscale=False):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    if len(images) == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        if is_grayscale:
            ax.imshow(img, cmap='gray')  # Set colormap to gray for grayscale images
        else:
            ax.imshow(img)  # Change channel order for RGB images
        ax.axis('off')
    plt.show()
    
def convert_to_grayscale(images):
    grayscale_images = []
    for img in images:
        grayscale_img = torch.mean(img.squeeze(), dim=0, keepdim=True)  # Take the mean across the RGB channels
        grayscale_images.append(grayscale_img)
    return grayscale_images

def edge_detection_canny(images, threshold1=100, threshold2=200):
    edge_images = []
    for img in images:
        img_np = img.squeeze().numpy()  # Convert torch tensor to numpy array
        img_np = (img_np * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        edges = cv2.Canny(img_np, threshold1, threshold2)  # Apply Canny edge detection
        edge_images.append(edges)
    return edge_images

def line_detection(edge_images, threshold=50, min_line_length=10, max_line_gap=5):
    line_images = []
    line_coords = []
    for edges in edge_images:
        img_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green lines
                line_coords.append((x1, y1, x2, y2))
        line_images.append(img_color)
    return line_images, line_coords

# def check_repetitive_lines(line_coords, repetition_threshold=2):
#     lines_counter = Counter(line_coords)
#     for count in lines_counter.values():
#         if count >= repetition_threshold:
#             return 1
#     return 0

def check_repetitive_lines(line_coords, repetition_threshold=2, tolerance=5):
    def lines_equal(line1, line2, tol):
        return all(abs(a - b) <= tol for a, b in zip(line1, line2))

    line_counts = Counter()
    for line in line_coords:
        matched = False
        for existing_line in line_counts:
            if lines_equal(line, existing_line, tolerance):
                line_counts[existing_line] += 1
                matched = True
                break
        if not matched:
            line_counts[line] += 1

    for count in line_counts.values():
        if count >= repetition_threshold:
            return 1
    return 0

