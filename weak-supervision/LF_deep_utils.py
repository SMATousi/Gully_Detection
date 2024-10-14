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

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)  # Ensure pretrained=False if loading custom weights
#         resnet.load_state_dict(torch.load(pre_trained_apth))
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Exclude the final classification layer

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # Flatten the output

# Define the MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    

class Gully_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Gully_Classifier, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor()
        self.classifier = MLPClassifier(input_size, hidden_size, output_size)
        
    def forward(self, images):

        features = [self.feature_extractor(image) for image in images]
        stacked_features = torch.stack(features, dim=1)
        output = self.classifier(stacked_features)


        return output