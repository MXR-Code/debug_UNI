import timm
import os
import logging
import timm
import torch
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = timm.create_model(model_name='vit_large_patch16_224',
                          img_size=224,
                          patch_size=16,
                          init_values=1e-5,
                          num_classes=0,
                          dynamic_img_size=True)
model = model.to(device=device)

batch_image = torch.rand((2, 3, 224, 224), device=device)
batch_size, num_channel, image_length, image_width = batch_image.shape
with torch.no_grad():
    features = model.forward(batch_image)

print()
