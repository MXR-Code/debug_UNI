from torch.nn import Module,Linear
import os
import logging
import timm
import torch
from torchvision import transforms
from huggingface_hub import login, hf_hub_download
import numpy as np
from PIL import Image



class UNI2H(Module):
    def __init__(self,
                 model_config=None,
                 is_pretrain_model=False,
                 pretrained_model_path=None,
                 huggingface_access_token=None,
                 num_class=0):
        super().__init__()
        if model_config is None:
            self.model_config = {'model_name': 'vit_giant_patch14_224',
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
        else:
            self.model_config = model_config

        self.encoder = timm.create_model(**self.model_config)
        if is_pretrain_model is False:
            self.load_model(pretrained_model_path=pretrained_model_path,
                            huggingface_access_token=huggingface_access_token)

        out_dim = self.test_forward()
        self.classifier = Linear(in_features=out_dim, out_features=num_class)


    def load_model(self, pretrained_model_path=None, huggingface_access_token=None):
        if pretrained_model_path is None:
            pretrained_model_path = os.path.join('uni2-h', 'pytorch_model.bin')
            # login with your User Access Token, found at https://huggingface.co/settings/tokens
            login(token=huggingface_access_token)
            os.makedirs(name=pretrained_model_path, exist_ok=True)
            hf_hub_download(repo_id='MahmoodLab/UNI2-h',
                            filename="pytorch_model.bin",
                            local_dir=pretrained_model_path,
                            force_download=True)

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        assert torch.is_tensor(x)
        out = self.encoder(x)

        return out

    def inference(self, x):
        out = self.classifier(x)
        return out

    def test_forward(self):
        device = next(self.encoder.parameters()).device
        batch_image = torch.rand((2, 3, 224, 224), device=device)
        batch_size, num_channel, image_length, image_width = batch_image.shape
        with torch.no_grad():
            batch_feature = self.forward(x=batch_image)
        out_dim = batch_feature.shape[-1]

        return out_dim

    def test_inference(self, x):
        out = self.classifier(x)



uni = UNI2H(is_pretrain_model=True)
uni.test_forward()
print()
