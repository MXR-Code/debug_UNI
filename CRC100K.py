# 导入依赖库
import time  # 用于计时，计算代码执行时间
import torch
import torchvision
import os
import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")


@torch.no_grad()
def extract_patch_features_from_dataloader(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters())[0].device

    for batch_idx, (batch, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(batch.type())
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu()[:remaining, :].cpu()
            labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    asset_dict = {"embeddings": np.vstack(all_embeddings).astype(np.float32),
                  "labels": np.concatenate(all_labels),
                  }

    return asset_dict





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


class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train(self):
        pass

    @torch.no_grad()
    def test(self, model, dataloader):
        """Uses model to extract features+labels from images iterated over the dataloader.

        Args:
            model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
            dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

        Returns:
            dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

        """
        model.eval()

        all_embed, all_label = [], []
        batch_size = dataloader.batch_size
        device = next(model.parameters())[0].device

        for batch_index, (batch_image, image_label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch_image = batch_image.to(device)

            with torch.inference_mode():
                image_embed = model.forward(batch_image)
                image_label = image_label.numpy()

            all_embed.append(image_embed)
            all_label.append(image_label)

        asset_dict = {"embeddings": np.vstack(all_embed).astype(np.float32),
                      "labels": np.concatenate(all_label),
                      }

        return asset_dict





# test
from uni2h import UNI2H

model = UNI2H(is_pretrain_model=True)

train_features = extract_patch_features_from_dataloader(model=model, train_dataloader)
test_features = extract_patch_features_from_dataloader(model, test_dataloader)

train_feats = torch.Tensor(train_features['embeddings'])  # 训练集特征张量（形状：[样本数, 特征维度]）
train_labels = torch.Tensor(train_features['labels']).type(torch.long)  # 标签转换为整型（分类任务要求）

test_feats = torch.Tensor(test_features['embeddings'])  # 测试集特征张量
test_labels = torch.Tensor(test_features['labels']).type(torch.long)

# 计算并输出总耗时
elapsed = time.time() - start  # 计算总执行时间（秒）
print(f'Took {elapsed:.03f} seconds')  # 格式化输出（保留3位小数）
