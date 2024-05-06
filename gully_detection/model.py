import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob
import random
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatures(nn.Module):
    def __init__(self, output_size, saved_model_path):
        super(ResNetFeatures, self).__init__()
        resnet = models.resnet50(weights=None)
        resnet.fc = nn.Linear(2048, output_size)  # Adjust output size for feature extraction

        checkpoint = torch.load(saved_model_path)
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]
        resnet.load_state_dict(state_dict, strict=False)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, input_channels, dropout_rate=0.5):
        super(DenseNet, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_channels, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

class gully_detector(nn.Module):
    def __init__(self, resnet_output_size, model_choice, resnet_saved_model_path, dropout_rate=0.5):
        super(gully_detector, self).__init__()
        self.resnet = ResNetFeatures(output_size=resnet_output_size, saved_model_path=resnet_saved_model_path)
        self.dense_net = DenseNet(input_channels=6*2048*resnet_output_size*resnet_output_size, dropout_rate=dropout_rate)
        self.model_choice = model_choice

    def forward(self, dem, rgbs):
        features = [self.resnet(rgb) for rgb in rgbs]
        features = torch.cat(features, dim=1)
        combined_input = torch.cat((dem, features), dim=1)
        output = self.dense_net(combined_input)
        return output

