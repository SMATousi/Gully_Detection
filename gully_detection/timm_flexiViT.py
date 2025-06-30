import timm
import torch
import torch.nn as nn

class Flexi_ViT_Gully_Classifier(nn.Module):
    def __init__(self):
        super(Flexi_ViT_Gully_Classifier, self).__init__()

        self.model_128 = timm.create_model('flexivit_small.1200ep_in1k', pretrained=True, img_size=128, patch_size=16)
        self.model_764 = timm.create_model('flexivit_small.1200ep_in1k', pretrained=True, img_size=764, patch_size=95)
        self.final_layer = nn.Linear(in_features=8*24960, out_features=2, bias=True)
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)  # Use a small gain value
        nn.init.zeros_(self.final_layer.bias)  # Initialize bias to zero
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(8*24960)
        self.bn = nn.BatchNorm1d(8*24960)
        self.dropout = nn.Dropout(0.3)
        # Freeze all parameters in model_128 and model_764
        for param in self.model_128.parameters():
            param.requires_grad = False
        for param in self.model_764.parameters():
            param.requires_grad = False

        
    def forward(self, list_of_images):
        """
        Forward pass for Flexi_ViT_Gully_Classifier.
        Args:
            list_of_images (list of torch.Tensor): List of images as tensors.
        Returns:
            torch.Tensor: Output prediction.
        """
        # Extract features using forward_features (removes classification head)
        unpooled_features_128 = [self.model_128.forward_features(image) for i, image in enumerate(list_of_images) if i != 1]
        unpooled_features_764 = [self.model_764.forward_features(image) for i, image in enumerate(list_of_images) if i == 1] # This is for the high resolution image
        
        # print(unpooled_features_128[0].shape)
        # print(unpooled_features_764[0].shape)

        flatten_features_128 = [torch.flatten(unpooled_features, start_dim=1) for unpooled_features in unpooled_features_128]
        flatten_features_764 = [torch.flatten(unpooled_features, start_dim=1) for unpooled_features in unpooled_features_764]

        # print(flatten_features_128[0].shape)
        # print(flatten_features_764[0].shape)
        # Concatenate features along the batch dimension
        # Each element in flatten_features_* is shape [batch, features]
        stacked_128_features = torch.cat(flatten_features_128, dim=1) if flatten_features_128 else None
        stacked_764_features = flatten_features_764[0] if flatten_features_764 else None
        # print(stacked_128_features.shape)
        # print(stacked_764_features.shape)

        if stacked_128_features is not None and stacked_764_features is not None:
            stacked_features = torch.cat([stacked_128_features, stacked_764_features], dim=1)
        elif stacked_128_features is not None:
            stacked_features = stacked_128_features
        elif stacked_764_features is not None:
            stacked_features = stacked_764_features
        else:
            raise ValueError("No features extracted from input images.")
        stacked_features = self.norm(stacked_features)
        stacked_features = self.bn(stacked_features)
        stacked_features = self.dropout(stacked_features)
        output = self.final_layer(stacked_features)
        # print("output", output)
        # output = self.sigmoid(output)
        # print("output after sigmoid", output)

        return output

# torch.manual_seed(0)

# model = Flexi_ViT_Gully_Classifier()
# model = model.to('cuda')

# list_of_images = [torch.randn(1, 3, 128, 128).to('cuda'), torch.randn(1, 3, 764, 764).to('cuda'),
#                   torch.randn(1, 3, 128, 128).to('cuda'), torch.randn(1, 3, 128, 128).to('cuda'),
#                   torch.randn(1, 3, 128, 128).to('cuda'), torch.randn(1, 3, 128, 128).to('cuda')]
# output = model(list_of_images)
# print(output)