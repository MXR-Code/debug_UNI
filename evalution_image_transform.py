from torchvision import transforms

def evalution_image_transform(constant_config: dict = None,
                              image_norm_type: str = 'imagenet',
                              image_resize: int = 224,
                              center_crop: bool = False):
    if constant_config is None:
        constant_config = {'imagenet': {'mean': (0.485, 0.456, 0.406),
                                        'std': (0.229, 0.224, 0.225)},
                           'ctranspath': {'mean': (0.485, 0.456, 0.406),
                                          'std': (0.229, 0.224, 0.225)},
                           'openai_clip': {'mean': (0.48145466, 0.4578275, 0.40821073),
                                           'std': (0.26862954, 0.26130258, 0.27577711)},
                           'uniform': {'mean': (0.5, 0.5, 0.5),
                                       'std': (0.5, 0.5, 0.5)}}

    image_transform = []
    if image_resize > 0:
        image_transform.append(transforms.Resize(image_resize))
        if center_crop:
            image_transform.append(transforms.CenterCrop(image_resize))

    constant = constant_config[image_norm_type]
    mean = constant.get('mean')
    std = constant.get('std')

    image_transform.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    image_transform = transforms.Compose(image_transform)

    return image_transform