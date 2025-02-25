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
from transformers import ViTForImageClassification, ViTFeatureExtractor
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



class ViT_Gully_Classifier(nn.Module):
    def __init__(self, tandom_init_embeddings=False, freeze_layers=False):
        super(ViT_Gully_Classifier, self).__init__()
        self.preprocessor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
        self.preprocessor.do_resize = False
        self.preprocessor.do_rescale = False

        self.embedding = self.classifier.vit.embeddings
        if tandom_init_embeddings:
            print("------ The embeddings are initialized randomly ----------------")
            torch.nn.init.normal_(self.embedding.patch_embeddings.projection.weight, mean=0.0, std=0.02)

        self.encoder = self.classifier.vit.encoder
        self.layernorm = self.classifier.vit.layernorm
        self.final_layer = nn.Linear(in_features=6*768, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        if freeze_layers:
            self.freeze_layers()

    def freeze_layers(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.layernorm.parameters():
            param.requires_grad = False
        
        # for param in self.final_layer.parameters():
        #     param.requires_grad = False
        # for param in self.sigmoid.parameters():
        #     param.requires_grad = False
        
    def forward(self, list_of_images):

        # preprocessed_images = [self.preprocessor(images=image, return_tensors="pt")['pixel_values'] for image in list_of_images]
        embedding_outputs = [self.embedding(image) for image in list_of_images]
        encoder_outputs = [self.encoder(embedding_output) for embedding_output in embedding_outputs]
        layer_norm_outputs = [self.layernorm(encoder_output[0]) for encoder_output in encoder_outputs]
        pooled_outputs = [layer_norm_output[:, 0] for layer_norm_output in layer_norm_outputs]
        stacked_features = torch.concatenate(pooled_outputs, dim=1)
        output = self.final_layer(stacked_features)
        output = self.sigmoid(output)

        return output
