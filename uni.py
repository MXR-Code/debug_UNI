import os
import logging
import timm
import torch
from torchvision import transforms


def get_norm_constants(which_img_norm: str = 'imagenet'):
    constants_zoo = {
        'imagenet': {'mean': (0.485, 0.456, 0.406),
                     'std': (0.229, 0.224, 0.225)},
        'ctranspath': {'mean': (0.485, 0.456, 0.406),
                       'std': (0.229, 0.224, 0.225)},
        'openai_clip': {'mean': (0.48145466, 0.4578275, 0.40821073),
                        'std': (0.26862954, 0.26130258, 0.27577711)},
        'uniform': {'mean': (0.5, 0.5, 0.5),
                    'std': (0.5, 0.5, 0.5)}
    }
    constants = constants_zoo[which_img_norm]
    return constants.get('mean'), constants.get('std')


def get_eval_transforms(which_img_norm: str = 'imagenet',
                        img_resize: int = 224,
                        center_crop: bool = False):
    eval_transform = []

    if img_resize > 0:
        eval_transform.append(transforms.Resize(img_resize))

        if center_crop:
            eval_transform.append(transforms.CenterCrop(img_resize))

    mean, std = get_norm_constants(which_img_norm)

    eval_transform.extend([transforms.ToTensor(),
                           transforms.Normalize(mean=mean, std=std)
                           ])
    eval_transform = transforms.Compose(eval_transform)

    return eval_transform


def get_uni_encoder(checkpoint='pytorch_model.bin',
                    which_img_norm='imagenet',
                    img_resize=224,
                    center_crop=True,
                    test_batch=0,
                    device=None,
                    assets_dir='./checkpoint',
                    ):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    checkpoint_dir = os.path.join(assets_dir, 'uni')
    checkpoint_path = os.path.join(assets_dir, 'uni', checkpoint)

    assert which_img_norm == 'imagenet'
    if not os.path.isfile(checkpoint_path):
        from huggingface_hub import login, hf_hub_download
        login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
        os.makedirs(checkpoint_dir, exist_ok=True)
        hf_hub_download('MahmoodLab/UNI', filename="pytorch_model.bin", local_dir=checkpoint_dir, force_download=True)

    uni_kwargs = {'model_name': 'vit_large_patch16_224',
                  'img_size': 224,
                  'patch_size': 16,
                  'init_values': 1e-5,
                  'num_classes': 0,
                  'dynamic_img_size': True}
    model = timm.create_model(**uni_kwargs)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

    eval_transform = get_eval_transforms(
        which_img_norm=which_img_norm,
        img_resize=img_resize,
        center_crop=center_crop
    )

    logging.info(f'Missing Keys: {missing_keys}')
    logging.info(f'Unexpected Keys: {unexpected_keys}')
    logging.info(str(model))

    # Send to GPU + turning on eval
    model.eval()
    model.to(device)

    # Test Batch
    logging.info(f"Transform Type: {eval_transform}")
    if test_batch:
        imgs = torch.rand((2, 3, 224, 224), device=device)
        with torch.no_grad():
            features = model(imgs)
        logging.info(f'Test batch successful, feature dimension: {features.size(1)}')
        del imgs, features

    return model, eval_transform
