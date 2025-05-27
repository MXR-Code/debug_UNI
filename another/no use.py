# 导入依赖库
import time  # 用于计时，计算代码执行时间
import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics
from uni.downstream.utils import concat_images
from uni import get_encoder
from huggingface_hub import login

# 初始化计时和数据路径
start = time.time()  # 记录代码开始执行的时间戳
dataroot = './assets/data/CRC100K/'  # 数据集根目录，指向结直肠癌病理图像数据集（CRC100K）

# 定义路径拼接函数
def j_(root, sub):
    return os.path.join(root, sub)

# 定义图像预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),  # 调整图像大小
    torchvision.transforms.ToTensor(),  # 转换为张量
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 创建训练集和测试集数据集
train_dataset = torchvision.datasets.ImageFolder(
    j_(dataroot, 'NCT-CRC-HE-100K-NONORM'),  # 训练集路径：dataroot/NCT-CRC-HE-100K-NONORM
    transform=transform  # 图像预处理
)
test_dataset = torchvision.datasets.ImageFolder(
    j_(dataroot, 'CRC-VAL-HE-7K'),  # 测试集路径：dataroot/CRC-VAL-HE-7K
    transform=transform  # 使用与训练集相同的预处理
)

# 创建数据加载器（DataLoader）
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,  # 每批次加载256张图像
    shuffle=False,  # 不随机打乱数据（保持顺序一致，可能用于后续特征对齐）
    num_workers=16  # 使用16个子进程加速数据加载
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=256,  # 测试集同样批量处理256张
    shuffle=False,  # 测试集通常无需打乱
    num_workers=16
)

# 提取图像块特征
# 调用自定义函数提取特征
train_features = extract_patch_features_from_dataloader(model, train_dataloader)  # 返回字典：{"embeddings": [...], "labels": [...]}
test_features = extract_patch_features_from_dataloader(model, test_dataloader)

# 转换为PyTorch张量
train_feats = torch.Tensor(train_features['embeddings'])  # 训练集特征张量（形状：[样本数, 特征维度]）
train_labels = torch.Tensor(train_features['labels']).type(torch.long)  # 标签转换为整型（分类任务要求）

test_feats = torch.Tensor(test_features['embeddings'])  # 测试集特征张量
test_labels = torch.Tensor(test_features['labels']).type(torch.long)

# 计算并输出总耗时
elapsed = time.time() - start  # 计算总执行时间（秒）
print(f'Took {elapsed:.03f} seconds')  # 格式化输出（保留3位小数）

# 加载预训练的UNI2-h模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
login(token="your_huggingface_token")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = get_encoder(enc_name='uni2-h', device=device)

# 线性模型训练和评估
linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
    train_feats=train_feats,  # 训练集特征张量（来自前一步的特征提取）
    train_labels=train_labels,  # 训练集标签张量
    valid_feats=None,  # 未提供验证集特征（可能直接使用训练集训练）
    valid_labels=None,  # 未提供验证集标签
    test_feats=test_feats,  # 测试集特征张量
    test_labels=test_labels,  # 测试集标签张量
    max_iter=1000,  # 线性模型的最大训练迭代次数（防止不收敛）
    verbose=True,  # 输出训练过程的详细信息（如损失变化）
)

# 打印评估指标
print_metrics(linprobe_eval_metrics)  # 自定义函数，格式化输出评估结果

# KNN和ProtoNet评估
knn_eval_metrics, knn_dump, proto_eval_metrics, proto_dump = eval_knn(
    train_feats=train_feats,  # 训练集特征张量（形状：[样本数, 特征维度]）
    train_labels=train_labels,  # 训练集标签张量（形状：[样本数]）
    test_feats=test_feats,  # 测试集特征张量
    test_labels=test_labels,  # 测试集标签张量
    center_feats=True,  # 对特征进行中心化（减去均值）
    normalize_feats=True,  # 对特征进行L2归一化（单位向量化）
    n_neighbors=20  # KNN中使用的近邻数量
)

# 打印评估结果
print_metrics(knn_eval_metrics)  # 输出KNN的评估指标
print_metrics(proto_eval_metrics)  # 输出ProtoNet的评估指标

# 小样本学习评估
fewshot_episodes, fewshot_dump = eval_fewshot(
    train_feats=train_feats,
    train_labels=train_labels,
    test_feats=test_feats,
    test_labels=test_labels,
    n_iter=100,  # 生成100个小样本任务（episodes）
    n_way=9,  # 每次任务使用全部9个类别
    n_shot=16,  # 每个类取16个样本作为支持集
    n_query=test_feats.shape[0],  # 查询集为全部测试样本
    center_feats=True,  # 对特征进行中心化（减去均值）
    normalize_feats=True,  # 对特征进行L2归一化
    average_feats=True,  # 平均支持集样本特征作为类原型
)

# 展示每个episode的详细评估结果
display(fewshot_episodes)  # 展示每个episode的详细评估结果（如准确率）
display(fewshot_dump)  # 显示汇总统计（如平均准确率、标准差）

# ProtoNet深入探索
proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
proto_clf.fit(train_feats, train_labels)
print('What our prototypes look like', proto_clf.prototype_embeddings.shape)

# 模型预测与评估
test_pred = proto_clf.predict(test_feats)
get_eval_metrics(test_labels, test_pred, get_report=False)

# 基于原型的检索（ROI Retrieval）
dist, topk_inds = proto_clf._get_topk_queries_inds(test_feats, topk=5)
print('label2idx correspondenes', test_dataset.class_to_idx)

# 构建测试集图片路径DataFrame
test_imgs_df = pd.DataFrame(test_dataset.imgs, columns=['path', 'label'])

# ADIPOSE类别的Top5样本可视化
print('Top-k ADIPOSE-like test samples to ADIPOSE prototype')
adi_topk_inds = topk_inds[0]  # 假设ADIPOSE原型的索引是0
adi_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][adi_topk_inds]], scale=0.5, gap=5)
display(adi_topk_imgs)

# 其他类别的可视化
print('Top-k LYMPHOCYTE-like test samples to LYMPHOCYTE prototype')
lym_topk_inds = topk_inds[3]
lym_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][lym_topk_inds]], scale=0.5, gap=5)
display(lym_topk_imgs)

print('Top-k TUMOR-like test samples to TUMOR prototype')
tum_topk_inds = topk_inds[8]
tum_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][tum_topk_inds]], scale=0.5, gap=5)
display(tum_topk_imgs)

# 其余类型的可视化分析代码列举如下，我不再展示
print('Top-k MUCOSA-like test samples to MUCOSA prototype')
muc_topk_inds = topk_inds[4]
muc_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][muc_topk_inds]], scale=0.5, gap=5)
display(muc_topk_imgs)

print('Top-k MUSCLE-like test samples to MUSCLE prototype')
mus_topk_inds = topk_inds[5]
mus_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][mus_topk_inds]], scale=0.5, gap=5)
display(mus_topk_imgs)

print('Top-k NORMAL-like test samples to NORMAL prototype')
norm_topk_inds = topk_inds[6]
norm_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][norm_topk_inds]], scale=0.5, gap=5)
display(norm_topk_imgs)

print('Top-k STROMA-like test samples to STROMA prototype')
str_topk_inds = topk_inds[7]
str_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][str_topk_inds]], scale=0.5, gap=5)
display(str_topk_imgs)