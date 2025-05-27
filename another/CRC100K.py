# 导入依赖库
import time  # 用于计时，计算代码执行时间
import torch
import torchvision
import os
import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm



def CRC100K(train_image_transform,
            test_image_transform,
            train_dataset_path='NCT-CRC-HE-100K-NONORM',
            test_dataset_path='CRC-VAL-HE-7K'):
    torch.multiprocessing.set_sharing_strategy("file_system")
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_image_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=16)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_image_transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)

    return train_dataloader, test_dataloader










