Nature Medicine复现
病理AI领域的基础模型
计算病理学的通用自监督模型UNI
通过在大规模组织图像数据集上的预训练
UNI在多种计算病理学任务中展现出优异性能。

随着数据量增加，UNI性能提升显著，在复杂癌症分类任务上超越其他模型。
在弱监督切片分类、ROI分类、检索和细胞类型分割等多种临床任务评估中，UNI也展现出优势，尤其在罕见病分类方面效果突出，其少样本学习能力能以少量样本实现高效分类。


Nature
Medicine病理AI汇总｜UNI：一种用于计算病理学的通用自监督基础模型·顶刊精析·24 - 10 - 31

文献中提到，UNI的小样本学习能力很出色，那么我也做了一个小小的测试——用一个结直肠癌10K和100K的数据集做了一个对比，测试集大小保持不变，结果发现10K数据集的某些参数更好！

这里只展示一个测试结果，推文中还有很多其他实验结果
这里只展示一个测试结果，推文中还有很多其他实验结果
这张图也告诉我们一个结论，虽然理论上数据量越大，模型的性能会越好，但是也不一定要一味的追求数据量，质量比数量更重要！那么问题来了，我们如何确定自己的数据集质量是否达标呢？




一、文献概述
研究背景
计算病理学（CPath）需要对组织图像进行定量评估，以支持病理学诊断。
然而，全切片图像（WSIs）的高分辨率和形态特征的变异性使得大规模数据注释变得困难，限制了模型的训练和性能。
当前方法通常依赖于从自然图像数据集或公开的组织病理学数据集进行迁移学习，但这些方法在不同组织类型和疾病类别的广泛应用中存在局限性。

研究方法
数据集构建：
构建Mass-100K数据集，包含超1亿个组织图像块，来自超10万张诊断性H & E染色WSIs，涵盖20种主要组织类型。
同时创建Mass-22K和Mass-1K子集，用于评估数据缩放定律。

模型架构与预训练：
基于视觉Transformer架构，使用DINOv2自监督学习方法在Mass-100K上预训练UNI模型。
对比其他自监督学习算法（如MoCoV3）和不同模型架构（ViT - Base和ViT - Large）。

评估设置：
将UNI与CPath领域常用的三个预训练编码器（ResNet - 50、CTransPath、REMEDIS ）对比，在34个临床任务上评估，包括弱监督切片分类、ROI分类、检索、分割和少样本学习等任务。

二、Download Model Parameter
访问HuggingFace模型页面(https://huggingface.co/MahmoodLab/UNI2-h)
登录HuggingFace账号（需先注册）
申请模型访问权限（需填写使用目的）


三、项目复现准备

import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np

加载预训练的UNI模型
from uni import get_encoder

批量提取图像特征
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

线性分类评估
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe

K近邻 / 小样本学习评估
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot

原型网络实现
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote

计算分类指标
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics
from uni.downstream.utils import concat_images


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


加载预训练的UNI2-h模型
pip install ipywidgets
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import login
login(token="……")
from uni import get_encoder  # 从uni模块导入get_encoder函数
model, transform = get_encoder(enc_name='uni2-h', device=device)



CRC-100 Dataset
K数据集处理
数据准备：加载病理图像数据集并分批次处理。
特征提取：用预训练模型提取图像块的高维特征（迁移学习常见操作）。
格式转换：将特征和标签转换为PyTorch张量，供后续训练分类器使用。
性能评估：统计耗时，优化数据处理效率。
适用于医学图像分类、检索等下游任务，避免在每次训练时重复计算特征。


下载数据集
这一部分指令在终端完成。

训练集 (100K图像)
nohup wget https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K-NONORM.zip?download=1 -O NCT-CRC-HE-100K-NONORM.zip > output.logcrc1 2>&1 &
测试集 (7.18K图像)
nohup wget https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1 -O CRC-VAL-HE-7K.zip > output.logcrc2 2>&1 &

# 创建目标目录
mkdir -p ./assets/data/CRC100K/
# 解压文件到目标目录
unzip NCT-CRC-HE-100K-NONORM.zip -d ./assets/data/CRC100K/
unzip CRC-VAL-HE-7K.zip -d ./assets/data/CRC100K/

定义数据预处理与加载
导入依赖库
import time  # 用于计时，计算代码执行时间
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader  # 从自定义模块导入特征提取函数

初始化计时和数据路径
start = time.time()  # 记录代码开始执行的时间戳
dataroot = './assets/data/CRC100K/'  # 数据集根目录，指向结直肠癌病理图像数据集（CRC100K）
创建训练集和测试集数据集
# 使用ImageFolder加载图像数据集（假设j_是路径拼接函数，类似os.path.join）
train_dataset = torchvision.datasets.ImageFolder(
    j_(dataroot, 'NCT-CRC-HE-100K-NONORM'),  # 训练集路径：dataroot/NCT-CRC-HE-100K-NONORM
    transform=transform  # 图像预处理（如归一化、裁剪等，假设transform已定义）
)
test_dataset = torchvision.datasets.ImageFolder(
    j_(dataroot, 'CRC-VAL-HE-7K'),  # 测试集路径：dataroot/CRC-VAL-HE-7K
    transform=transform  # 使用与训练集相同的预处理
)
创建数据加载器（DataLoader）
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
提取图像块特征
# 调用自定义函数提取特征
train_features = extract_patch_features_from_dataloader(model,
                                                        train_dataloader)  # 返回字典：{"embeddings": [...], "labels": [...]}
test_features = extract_patch_features_from_dataloader(model, test_dataloader)
转换为PyTorch张量
# 将特征和标签从NumPy数组转换为PyTorch张量
train_feats = torch.Tensor(train_features['embeddings'])  # 训练集特征张量（形状：[样本数, 特征维度]）
train_labels = torch.Tensor(train_features['labels']).type(torch.long)  # 标签转换为整型（分类任务要求）

test_feats = torch.Tensor(test_features['embeddings'])  # 测试集特征张量
test_labels = torch.Tensor(test_features['labels']).type(torch.long)
计算并输出总耗时
elapsed = time.time() - start  # 计算总执行时间（秒）
print(f'Took {elapsed:.03f} seconds')  # 格式化输出（保留3位小数）

数据集精简（可选）
如果你采用原始的数据集，不做任何改动，那么执行这一步的时候会看到一个进度条，如下所示。

import os
import shutil
import random


def sample_and_copy_tif_images(src_root, dst_root, sample_num=10):
    """
    从源路径的每个子文件夹中随机抽取指定数量的.tif文件，保持原结构复制到目标路径
    """
    # 确保目标根目录存在
    os.makedirs(dst_root, exist_ok=True)

    # 遍历源目录下的所有子文件夹
    for folder_name in os.listdir(src_root):
        src_folder = os.path.join(src_root, folder_name)

        # 跳过非目录文件
        ifnot
        os.path.isdir(src_folder):
        continue

    # 获取所有.tif文件
    tif_files = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith(".tif")
    ]

    # 检查文件数量
    if len(tif_files) < sample_num:
        raise ValueError(f"文件夹 {folder_name} 中只有 {len(tif_files)} 个.tif文件，不足要求的 {sample_num} 个")

    # 随机抽样
    selected_files = random.sample(tif_files, sample_num)

    # 创建目标文件夹
    dst_folder = os.path.join(dst_root, folder_name)
    os.makedirs(dst_folder, exist_ok=True)

    # 复制文件
    for file_name in selected_files:
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(dst_folder, file_name)
        shutil.copy(src_path, dst_path)
        print(f"已复制: {src_path} -> {dst_path}")


if __name__ == "__main__":
    # 原始路径（包含9个子文件夹）
    source_path = "/data2/data2_mailab015/reproject/2025/03/03-14/UNI/assets/data/CRC100K/NCT-CRC-HE-100K-NONORM"

    # 新路径（将自动创建）
    destination_path = "/data2/data2_mailab015/reproject/2025/03/03-14/UNI/assets/data2/CRC100K/NCT-CRC-HE-100K-NONORM"

    # 执行抽样和复制
    sample_and_copy_tif_images(
        src_root=source_path,
        dst_root=destination_path,
        sample_num=1000
    )
    
五、线性模型训练和评估
流程总结
输入：训练集和测试集的预提取特征及标签。
训练：在训练集上训练线性分类器（如逻辑回归）。
评估：在测试集上计算分类性能指标（如准确率）。
输出：格式化显示结果，验证特征的有效性。
该代码是迁移学习流程的最后一步，用于验证特征提取模型（如预训练的ResNet）在特定任务上的实用性。

ROI：在医学图像中通常指“感兴趣区域”（Region of Interest），例如肿瘤区域。
Linear Probe：通过训练线性分类器（如逻辑回归）评估特征的质量，测试其线性可分性。

导入评估函数
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
** eval_linear_probe **：从自定义模块导入的函数，用于在提取的特征上训练线性分类器并评估性能。

执行线性探测评估
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

输入参数
train_feats / test_feats：从预训练模型提取的特征，形状为[样本数, 特征维度]。
valid_feats：验证集未提供，可能直接使用测试集评估或跳过超参数调优。
max_iter = 1000：控制线性模型（如sklearn的LogisticRegression）的最大迭代次数。
verbose = True：输出训练日志，例如每个epoch的损失值或进度条。

输出结果
linprobe_eval_metrics：评估指标字典（如准确率、F1分数、AUC）。
linprobe_dump：可能包含模型权重、预测结果等详细数据（未在后续代码中使用）。
6 - 4：打印评估指标

print_metrics(linprobe_eval_metrics)  # 自定义函数，格式化输出评估结果

数据集形状：
训练集形状为torch.Size([100000, 1536]) ，表示训练集有100000个样本，每个样本特征维度为1536。
测试集形状为torch.Size([7180, 1536]) ，即测试集有7180个样本，特征维度同样为1536。

输入：已提取的训练集和测试集特征。
预处理：中心化和归一化特征。
KNN评估：基于最近邻投票分类。
ProtoNet评估：基于类别原型分类。
输出：两种方法的分类性能指标。

任务说明ROI

注释：表示这是针对感兴趣区域（ROI）的K近邻（KNN）和原型网络（ProtoNet）评估
KNN：基于特征空间中的最近邻分类，用于评估特征的可分性。
ProtoNet：原型网络（Prototypical Network），小样本学习的经典方法，通过计算类别原型进行分类。


导入评估函数
from uni.downstream.eval_patch_features.fewshot import eval_knn  # 从自定义模块导入评估函数

执行KNN和ProtoNet评估
knn_eval_metrics, knn_dump, proto_eval_metrics, proto_dump = eval_knn(
    train_feats=train_feats,  # 训练集特征张量（形状：[样本数, 特征维度]）
    train_labels=train_labels,  # 训练集标签张量（形状：[样本数]）
    test_feats=test_feats,  # 测试集特征张量
    test_labels=test_labels,  # 测试集标签张量
    center_feats=True,  # 对特征进行中心化（减去均值）
    normalize_feats=True,  # 对特征进行L2归一化（单位向量化）
    n_neighbors=20  # KNN中使用的近邻数量
)
输入参数：
center_feats = True：将特征中心化（使特征均值为0），消除全局偏差。
normalize_feats = True：对特征向量进行L2归一化（每个向量缩放到长度为1），使距离计算不受特征尺度影响。
n_neighbors = 20：KNN分类时考虑最近的20个邻居（值越大抗噪性越强，但可能模糊类别边界）。

输出结果：
knn_eval_metrics：KNN的评估指标（如准确率、召回率）。
knn_dump：可能包含KNN预测结果或中间数据。
proto_eval_metrics：ProtoNet的评估指标。
proto_dump：ProtoNet的详细输出（如原型向量、预测结果）。

打印评估结果
print_metrics(knn_eval_metrics)  # 输出KNN的评估指标
print_metrics(proto_eval_metrics)  # 输出ProtoNet的评估指标
100
K数据模型
准确率（acc）
平衡准确率（bacc）


7 - 1：导入评估函数

from uni.downstream.eval_patch_features.fewshot import eval_fewshot

从指定模块导入eval_fewshot函数，该函数用于执行小样本学习的评估流程。

7 - 2：调用评估函数

fewshot_episodes, fewshot_dump = eval_fewshot(
    train_feats=train_feats,
    train_labels=train_labels,
    test_feats=test_feats,
    test_labels=test_labels,
    n_iter=100,  # 生成100个小样本任务（episodes）
    n_way=9,  # 每次任务使用全部9个类别
    n_shot=16,  # 每个类取16个样本作为支持集（注释提到4，可能参数或注释不一致）
    n_query=test_feats.shape[0],  # 查询集为全部测试样本
    center_feats=True,  # 对特征进行中心化（减去均值）
    normalize_feats=True,  # 对特征进行L2归一化
    average_feats=True,  # 平均支持集样本特征作为类原型
)
输入数据：使用训练和测试集的特征及标签。
n_iter = 100：生成100个独立的小样本任务，以评估模型鲁棒性。
n_way = 9：每个任务包含全部9个类别（假设数据集共9类）。
n_shot = 16：每个类抽取16个训练样本构建支持集，但注释提到4，可能存在不一致。
n_query：使用所有测试样本作为查询集，评估整体性能。
特征处理：中心化、归一化后计算类原型（均值），以提高分类效果。
输出结果展示

display(fewshot_episodes)  # 展示每个episode的详细评估结果（如准确率）
display(fewshot_dump)  # 显示汇总统计（如平均准确率、标准差）
fewshot_episodes：列表形式存储每个小样本任务的结果，反映模型在不同支持集下的表现波动。
fewshot_dump：汇总统计信息，如平均准确率，用于评估整体性能。


八、ProtoNet深入探索
导入ProtoNet类并初始化模型

from uni.downstream.eval_patch_features.protonet import ProtoNet

proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
功能：从自定义路径导入ProtoNet类，并初始化模型。
参数解析：
metric = 'L2'：使用欧氏距离（L2距离）作为相似性度量。
center_feats = True：在训练时对特征进行中心化（减去均值）。
normalize_feats = True：对特征进行L2归一化。
作用：通过中心化和归一化，确保特征分布的一致性，提升原型学习的鲁棒性。

训练模型
proto_clf.fit(train_feats, train_labels)
print('What our prototypes look like', proto_clf.prototype_embeddings.shape)
功能：用训练数据拟合模型，生成每个类别的原型（prototype）。
原型生成逻辑：
对每个类别，计算其所有样本特征的均值作为原型。例如，若类别A有100个样本，则原型为这100个特征的均值向量。
输出示例：(num_classes, feature_dim)，表示原型数量为类别数，每个原型的维度与输入特征相同。

模型预测与评估
test_pred = proto_clf.predict(test_feats)
get_eval_metrics(test_labels, test_pred, get_report=False)
预测逻辑：对于每个测试样本，计算其与所有原型的距离，选择最近的原型对应的类别作为预测结果。
指标名称

获取TopK查询索引
dist, topk_inds = proto_clf._get_topk_queries_inds(test_feats, topk=5)
作用：以测试特征作为查询集，计算每个原型最近的topk个测试样本

输入：
test_feats：测试集的特征矩阵（形状：[n_test_samples, feature_dim]）
topk = 5：每个原型取前5个最近样本

输出：
dist：距离矩阵（形状：[n_prototypes, n_test_samples]）
topk_inds：索引矩阵（形状：[n_prototypes, topk]），每行对应一个原型的前5个最近测试样本的索引


打印类别 - 索引映射关系
print('label2idx correspondenes', test_dataset.class_to_idx)
作用：显示数据集中类别名称与数字索引的对应关系（例如：{'ADIPOSE': 0, 'LYMPHOCYTE': 3, ...}）
意义：解释后续代码中topk_inds[0]
为什么对应ADIPOSE原型

输出如下：
label2idx
correspondenes
{'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}

构建测试集图片路径DataFrame
test_imgs_df = pd.DataFrame(test_dataset.imgs, columns=['path', 'label'])

结构
test_dataset.imgs：Pytorch
Dataset的标准属性，包含元组列表[(图片路径1, 标签1), ...]
转换为DataFrame后有两列：
path：图片文件的绝对 / 相对路径
label：对应的数字标签（需通过class_to_idx映射为类别名）

ADIPOSE类别的Top5样本可视化
print('Top-k ADIPOSE-like test samples to ADIPOSE prototype')
adi_topk_inds = topk_inds[0]  # 假设ADIPOSE原型的索引是0
adi_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][adi_topk_inds]], scale=0.5,
                              gap=5)
display(adi_topk_imgs)

从topk_inds第0行获取ADIPOSE原型最近的5个测试样本索引
通过test_imgs_df['path'][adi_topk_inds]
提取这些样本的图片路径
使用PIL库的Image.open加载图片
concat_images：自定义函数（未显示），将多张图片水平拼接
scale = 0.5：图片缩放50 %
gap = 5：图片间隔5像素
display()：在Jupyter中渲染拼接后的图片
ADIPOSE
ADIPOSE
5.
其他类别的可视化
LYMPHOCYTE

print('Top-k LYMPHOCYTE-like test samples to LYMPHOCYTE prototype')
lym_topk_inds = topk_inds[3]
lym_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][lym_topk_inds]], scale=0.5,
                              gap=5)
display(lym_topk_imgs)
索引说明：不同类别在topk_inds中的行号由class_to_idx决定

例如：test_dataset.class_to_idx['LYMPHOCYTE']
返回3，因此取topk_inds[3]
LYMPHOCYTE
LYMPHOCYTE
TUMOR

print('Top-k TUMOR-like test samples to TUMOR prototype')
tum_topk_inds = topk_inds[8]
tum_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][tum_topk_inds]], scale=0.5,gap=5)
display(tum_topk_imgs)


print('Top-k MUCOSA-like test samples to MUCOSA prototype')
muc_topk_inds = topk_inds[4]
muc_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][muc_topk_inds]], scale=0.5,gap=5)
display(muc_topk_imgs)

print('Top-k MUSCLE-like test samples to MUSCLE prototype')
mus_topk_inds = topk_inds[5]
mus_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][mus_topk_inds]], scale=0.5,gap=5)
display(mus_topk_imgs)

print('Top-k NORMAL-like test samples to NORMAL prototype')
norm_topk_inds = topk_inds[6]
norm_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][norm_topk_inds]], scale=0.5,gap=5)
display(norm_topk_imgs)

print('Top-k STROMA-like test samples to STROMA prototype')
str_topk_inds = topk_inds[7]
str_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][str_topk_inds]], scale=0.5,gap=5)
display(str_topk_imgs)

