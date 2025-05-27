from torch.nn import Module
import os
import logging
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from resnet50_trunc import resnet50_trunc_imagenet
from huggingface_hub import login, hf_hub_download


class UNI2H(nn.Module):
    def __init__(self, uni_config=None,
                 pretrain_model_path=None):
        super().__init__()
        if uni_config is None:
            uni_config = {'model_name': 'vit_giant_patch14_224',
                          'img_size': 224,
                          'patch_size': 14,
                          'depth': 24,
                          'num_heads': 24,
                          'init_values': 1e-5,
                          'embed_dim': 1536,
                          'mlp_ratio': 2.66667 * 2,
                          'num_classes': 0,
                          'no_embed_class': True,
                          'mlp_layer': timm.layers.SwiGLUPacked,
                          'act_layer': torch.nn.SiLU,
                          'reg_tokens': 8,
                          'dynamic_img_size': True}
        self.model = timm.create_model(**uni_config)
        self.load_model(path=pretrain_model_path)

        self.constants_zoo = {'imagenet': {'mean': (0.485, 0.456, 0.406),
                                           'std': (0.229, 0.224, 0.225)},
                              'ctranspath': {'mean': (0.485, 0.456, 0.406),
                                             'std': (0.229, 0.224, 0.225)},
                              'openai_clip': {'mean': (0.48145466, 0.4578275, 0.40821073),
                                              'std': (0.26862954, 0.26130258, 0.27577711)},
                              'uniform': {'mean': (0.5, 0.5, 0.5),
                                          'std': (0.5, 0.5, 0.5)}}
        self.transform = self.evalution_transform()

    def load_model(self, path):
        if path is None:
            path = os.path.join('uni2-h', 'pytorch_model.bin')
            login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
            os.makedirs(name=path, exist_ok=True)
            hf_hub_download(repo_id='MahmoodLab/UNI2-h',
                            filename="pytorch_model.bin",
                            local_dir=path,
                            force_download=True)

        state_dict = torch.load(path, map_location="cpu")
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=True)

    def evalution_transform(self,
                            image_norm_type: str = 'imagenet',
                            image_resize: int = 224,
                            center_crop: bool = False):
        transform = []
        if image_resize > 0:
            transform.append(transforms.Resize(image_resize))
            if center_crop:
                transform.append(transforms.CenterCrop(image_resize))

        constants = self.constants_zoo[image_norm_type]
        mean = constants.get('mean')
        std = constants.get('std')

        transform.extend([transforms.ToTensor(),
                          transforms.Normalize(mean=mean, std=std)])
        transform = transforms.Compose(transform)

        return transform


    def forward(self, x):
        with torch.no_grad():
            x = self.transform(x)
            x = x.to(self.device)

            return self.model(x)

    def test_batch(self):
        imgs = torch.rand((2, 3, 224, 224), device=self.device)
        with torch.no_grad():
            features = self.forward(x=imgs)



