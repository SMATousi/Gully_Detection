from urllib.request import urlopen
from PIL import Image
import timm
import torch

# img = Image.open(urlopen(
#     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
# ))

model = timm.create_model('flexivit_small.1200ep_in1k', pretrained=True, img_size=128)
model = model.to('cuda')
model = model.eval()


img = torch.randn(1, 3, 128, 128).to('cuda')
print(model(img))