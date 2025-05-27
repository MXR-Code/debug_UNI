Nature
Medicine复现｜病理AI领域的基础模型不会用？这篇保姆级教程带你掌握数据处理全流程！ 已付费
小罗碎碎念
在医学AI领域，计算病理学对于疾病诊断意义重大，而UNI模型的出现为该领域带来了新突破。

今天这期带领大家复现一篇发表于Nature
Medicine的文章，介绍的是用于计算病理学的通用自监督模型UNI——通过在大规模组织图像数据集上的预训练，UNI在多种计算病理学任务中展现出优异性能，为该领域的人工智能模型发展提供了重要基础。

https: // doi.org / 10.1038 / s41591 - 024 - 02
857 - 3
https: // doi.org / 10.1038 / s41591 - 024 - 02
857 - 3
随着数据量增加，UNI性能提升显著，在复杂癌症分类任务上超越其他模型。在弱监督切片分类、ROI分类、检索和细胞类型分割等多种临床任务评估中，UNI也展现出优势，尤其在罕见病分类方面效果突出，其少样本学习能力能以少量样本实现高效分类。

UNI是去年发表的，经过一年的时间检验，现在已经成为了病理AI领域家喻户晓的模型。我已经在之前的推送中详细解读过UNI这篇文献，这期推送侧重于如何复现项目，所以不再解读文献，感兴趣的可以跳转链接阅读。

Nature
Medicine病理AI汇总｜UNI：一种用于计算病理学的通用自监督基础模型·顶刊精析·24 - 10 - 31

文献中提到，UNI的小样本学习能力很出色，那么我也做了一个小小的测试——用一个结直肠癌10K和100K的数据集做了一个对比，测试集大小保持不变，结果发现10K数据集的某些参数更好！

这里只展示一个测试结果，推文中还有很多其他实验结果
这里只展示一个测试结果，推文中还有很多其他实验结果
这张图也告诉我们一个结论，虽然理论上数据量越大，模型的性能会越好，但是也不一定要一味的追求数据量，质量比数量更重要！那么问题来了，我们如何确定自己的数据集质量是否达标呢？

解决上述问题的第一步，就是要学会分析你拥有的数据集，所以这期推送我也会手把手带大家分析一个数据集，看看数据集的组成情况。

例如这是一个9分类数据集，我们现在查看训练集和测试集的数据结构
例如这是一个9分类数据集，我们现在查看训练集和测试集的数据结构
分析完数据集的组成情况，我们接下来就要测试模型的性能。既然是测试，那肯定不能一开始就把所有的数据都丢给模型，我们要小批量的测试，这时候就会涉及到数据集的采样，这一部分也都会在推文中和大家分享！

最后就来到了大家最关心的模型性能对比和可视化分析部分，这期推送和大家分享三种测试方法——线性评估、K近邻以及原型网络，部分测试结果展示如下。

模型
准确率（acc）
平衡准确率（bacc）
Cohen
's kappa系数（kappa）
加权F1值（weighted_f1）
knn20
0.969
0.957
0.981
0.969
proto
0.884
0.854
0.910
0.876
可视化分析部分结果展示如下，有的老师 / 同学可能会疑惑，为啥展示patch，以及为啥数据集是patch形式的。
其实很简单，因为……UNI是patch级别的基础模型，哈哈，大家在看后续的推送时一定要牢记这句话！

类别预测展示
类别预测展示
本期项目复现推送架构

一、文献概述
二、环境配置
三、项目复现准备
四、CRC - 100
K数据集处理
五、线性模型训练和评估
六、ROI区域进行K近邻和原型网络的评估
七、基于原型网络（ProtoNet）的小样本学习性能评估
八、ProtoNet深入探索
交流群

欢迎大家加入【医学AI】交流群，本群设立的初衷是提供交流平台，方便大家后续课题合作。

目前小罗全平台关注量52, 000 +，交流群总成员1100 +，大部分来自国内外顶尖院校 / 医院，期待您的加入！！

由于近期入群推销人员较多，已开启入群验证，扫码添加我的联系方式，备注姓名 - 单位 - 科室 / 专业，即可邀您入群。

图片
知识星球

如需获取推文中提及的各种资料，欢迎加入我的知识星球！

图片
阅前必读

注意，由于编写项目复现系列教程需要花费大量时间，所以采取付费阅读的形式，绝对让你物超所值，你通过这一篇推送，可以节省大量自己摸索的时间。

【1】阅读方式1：知识星球（推荐）

已订阅星球用户可以前往知识星球获取pdf版本教程，并且可以在星球中提问，我会给出详细解答。此外，星球是按年付费，更划算！

【2】阅读方式2：微信推送付费阅读

如果只对单篇内容感兴趣，可以直接支付本篇文章费用。

一、文献概述
1 - 1：研究背景

计算病理学（CPath）需要对组织图像进行定量评估，以支持病理学诊断。然而，全切片图像（WSIs）的高分辨率和形态特征的变异性使得大规模数据注释变得困难，限制了模型的训练和性能。

当前方法通常依赖于从自然图像数据集或公开的组织病理学数据集进行迁移学习，但这些方法在不同组织类型和疾病类别的广泛应用中存在局限性。

1 - 2：研究方法

数据集构建：构建Mass - 100
K数据集，包含超1亿个组织图像块，来自超10万张诊断性H & E染色WSIs，涵盖20种主要组织类型。
同时创建Mass - 22
K和Mass - 1
K子集，用于评估数据缩放定律。
模型架构与预训练：基于视觉Transformer架构，使用DINOv2自监督学习方法在Mass - 100
K上预训练UNI模型。对比其他自监督学习算法（如MoCoV3）和不同模型架构（ViT - Base和ViT - Large）。
评估设置：将UNI与CPath领域常用的三个预训练编码器（ResNet - 50、CTransPath、REMEDIS ）对比，在34个临床任务上评估，包括弱监督切片分类、ROI分类、检索、分割和少样本学习等任务。

1 - 3：研究结果
预训练缩放定律：UNI展现出模型和数据缩放能力，随着预训练数据增加，在OT - 43
和OT - 108
任务上性能提升显著，且优于其他预训练编码器。
弱监督切片分类：在15个切片级分类任务中，UNI表现优异，尤其在罕见癌症类型或高诊断复杂性任务上优势明显，部分结果超过人类病理学家表现。
少样本学习：在少样本学习任务中，UNI标签效率高，在切片级和ROI级分类中，相比其他编码器，能用更少训练样本达到更高性能。
其他任务表现：在ROI分类、检索、细胞类型分割等任务中，UNI均优于多数基线模型，对高分辨率图像具有鲁棒性。

二、环境配置
2 - 1：环境配置

# 克隆仓库并进入目录
git
clone
https: // github.com / mahmoodlab / UNI.git
cd
UNI

# 创建conda环境并安装依赖
conda
create - n
UNI
python = 3.10 - y
conda
activate
UNI
pip
install - e.

# 安装额外依赖
pip
install
timm
huggingface_hub
torchvision

1、访问HuggingFace模型页面(https: // huggingface.co / MahmoodLab / UNI2 - h)

image - 20250314120025550
image - 20250314120025550
2、登录HuggingFace账号（需先注册）

3、申请模型访问权限（需填写使用目的）

4、配置Token

image - 20250314120406673
image - 20250314120406673
然后选择模型。

image - 20250314120456254
image - 20250314120456254
注意复制Token，后续要用到。

image - 20250314120549016
image - 20250314120549016

三、项目复现准备
3 - 1：导入依赖项

1.基础库导入
import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np

作用：导入深度学习框架和数据处理工具

关键点：
torch / torchvision：PyTorch核心库和视觉扩展
os：操作系统级文件路径操作
PIL.Image：图像读取与处理
pandas / numpy：结构化数据操作

2.UNI专用模块导入
from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics
from uni.downstream.utils import concat_images

核心组件：
模块 / 函数
功能
get_encoder
加载预训练的UNI模型
extract_patch_features_from_dataloader
批量提取图像特征
eval_linear_probe
线性分类评估
eval_knn/ eval_fewshot
K近邻 / 小样本学习评估
ProtoNet
原型网络实现
get_eval_metrics
计算分类指标

3.设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
作用：自动检测并配置计算设备

3 - 2：加载预训练的UNI2 - h模型

1.功能概述
下载模型权重：get_encoder
函数会自动从远程服务器下载UNI2 - h的预训练权重，保存到本地目录. / assets / ckpts /。
创建模型实例：加载下载的权重，构建并返回模型对象。
获取预处理方法：返回与模型匹配的图像预处理流程（transform），确保输入数据符合模型要求。

2.代码
pip install ipywidgets
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import login
login(token="……")

from uni import get_encoder  # 从uni模块导入get_encoder函数
model, transform = get_encoder(enc_name='uni2-h', device=device)

补充

如果你跟着上面的教程走，一直报错，也不要紧，新建一个脚本，然后把上面所有的代码都放进脚本里，然后在终端运行。

image - 20250314154827185
image - 20250314154827185
首先你需要输入一个token，然后再输入一个y，就可以正常运行了。

四、CRC - 100
K数据集处理
流程总结

数据准备：加载病理图像数据集并分批次处理。
特征提取：用预训练模型提取图像块的高维特征（迁移学习常见操作）。
格式转换：将特征和标签转换为PyTorch张量，供后续训练分类器使用。
性能评估：统计耗时，优化数据处理效率。
适用于医学图像分类、检索等下游任务，避免在每次训练时重复计算特征。

4 - 1：数据集预处理

1、下载数据集
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



3、定义数据预处理与加载
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
4 - 2：数据集精简（可选）

如果你采用原始的数据集，不做任何改动，那么执行这一步的时候会看到一个进度条，如下所示。

image - 20250317170432609
image - 20250317170432609
这里解释一下，CRC - 100
K数据集的训练集总共有100K，391 * 256 = 100, 0
96 ；测试集有7
.8
K，29
x256 = 7524。

但是我这个推送只作为演示教程，如果按照原数据集大小进行处理，耗时过长，所以我会对数据做一些精简。

正好接着这个机会，也和大家展示一下，拿到一个新的数据集，应该如何分析数据集的组成，并整理自己的实验数据。

首先我们写一段代码，提取原始数据集的文件列表，计算每个类别数据的数量，最终结果整理为下图的形式。

代码在知识星球的test.ipynb文件中获取
代码在知识星球的test.ipynb文件中获取
由于我们此处不需要考虑数据不平衡的问题，所以我简单粗暴的直接对100K数据下采样为10K，7
K数据不做变动，正好给大家展示一下训练集数量大小对模型性能的影响。

下采样的代码如下，大家记得对应修改创建数据集的路径。

路径修改
路径修改
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

5 - 1：任务说明

ROI
Linear
Probe
Evaluation.  # 注释：表示这是针对感兴趣区域（ROI）的线性探测评估任务
ROI：在医学图像中通常指“感兴趣区域”（Region
of
Interest），例如肿瘤区域。
Linear
Probe：通过训练线性分类器（如逻辑回归）评估特征的质量，测试其线性可分性。
5 - 2：导入评估函数

from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
** eval_linear_probe **：从自定义模块导入的函数，用于在提取的特征上训练线性分类器并评估性能。

5 - 3：执行线性探测评估

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
max_iter = 1000：控制线性模型（如
sklearn
的
LogisticRegression）的最大迭代次数。
verbose = True：输出训练日志，例如每个epoch的损失值或进度条。
输出结果

linprobe_eval_metrics：评估指标字典（如准确率、F1分数、AUC）。
linprobe_dump：可能包含模型权重、预测结果等详细数据（未在后续代码中使用）。
6 - 4：打印评估指标

print_metrics(linprobe_eval_metrics)  # 自定义函数，格式化输出评估结果
线性模型评估（Linear
Probe
Evaluation）结果
线性模型评估（Linear
Probe
Evaluation）结果
数据集形状：
训练集形状为torch.Size([100000, 1536]) ，表示训练集有100000个样本，每个样本特征维度为1536。
测试集形状为torch.Size([7180, 1536]) ，即测试集有7180个样本，特征维度同样为1536。
训练相关指标：
训练过程中的最佳成本（Best
cost）为138
.240。
训练前损失（Loss）为2
.197 ，训练后损失降低到0
.030，说明模型训练效果显著，很好地拟合了训练数据。
测试相关指标：
测试时间的形状与前面测试集形状一致。
线性探针评估总耗时4
.95（未明确时间单位，推测是秒）。
测试准确率（lin_acc）为0
.969。
测试平衡准确率（lin_bacc）为0
.957。
测试Cohen
's kappa系数（lin_kappa）为0.988 ，表示模型预测结果与实际结果的一致性很高。
测试加权F1值（lin_weighted_f1）为0
.969。
测试曲线下面积（lin_auroc）为0
.989 ，说明模型区分正负样本的能力很强。 总体来看，该模型在测试集上表现非常优秀。
5 - 5：10
K数据🆚100
K数据

前面的章节中，我提到我对数据做了一个精简，下面我们来对比一下效果。

image - 20250317203051859
image - 20250317203051859
六、ROI区域进行K近邻和原型网络的评估
流程总结

输入：已提取的训练集和测试集特征。
预处理：中心化和归一化特征。
KNN评估：基于最近邻投票分类。
ProtoNet评估：基于类别原型分类。
输出：两种方法的分类性能指标。
6 - 1：任务说明

ROI
KNN and ProtoNet
evaluation.  # 注释：表示这是针对感兴趣区域（ROI）的K近邻（KNN）和原型网络（ProtoNet）评估
KNN：基于特征空间中的最近邻分类，用于评估特征的可分性。
ProtoNet：原型网络（Prototypical
Network），小样本学习的经典方法，通过计算类别原型进行分类。
6 - 2：导入评估函数

from uni.downstream.eval_patch_features.fewshot import eval_knn  # 从自定义模块导入评估函数

6 - 3：执行KNN和ProtoNet评估

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
6 - 4：打印评估结果

print_metrics(knn_eval_metrics)  # 输出KNN的评估指标
print_metrics(proto_eval_metrics)  # 输出ProtoNet的评估指标
100
K数据

模型
准确率（acc）
平衡准确率（bacc）
Cohen
's kappa系数（kappa）
加权F1值（weighted_f1）
knn20
0.969
0.957
0.981
0.969
proto
0.884
0.854
0.910
0.876
10
K数据

模型
准确率（acc）
平衡准确率（bacc）
Cohen
's kappa系数（kappa）
加权F1值（weighted_f1）
knn20
0.969
0.956
0.984
0.969
proto
0.886
0.857
0.912
0.879
KNN(k=20)
相关指标（100
K）

Test
knn20_acc（KNN(k=20)
的准确率）为0
.969 。
Test
knn20_bacc（KNN(k=20)
的平衡准确率）为0
.957 。
Test
knn20_kappa（KNN(k=20)
的Cohen
's kappa系数）为0.981 ，表明预测结果与实际结果一致性很高。
Test
knn20_weighted_f1（KNN(k=20)
的加权F1值）为0
.969 。整体来看，KNN(k=20)
在测试集上表现较为出色。
Proto
相关指标（100
K）

Test
proto_acc（Proto方法的准确率）为0
.884 。
Test
proto_bacc（Proto方法的平衡准确率）为0
.854 。
Test
proto_kappa（Proto方法的Cohen
's kappa系数）为0.910 。
Test
proto_weighted_f1（Proto方法的加权F1值）为0
.876 。相比KNN(k=20) ，Proto方法的各项指标略低。
七、基于原型网络（ProtoNet）的小样本学习性能评估
这段代码用于评估基于原型网络（ProtoNet）的小样本学习性能，通过多次抽样构建不同的支持集，并统计模型在测试集上的表现。

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
image - 20250317211847691
image - 20250317211847691
八、ProtoNet深入探索
8 - 1：导入ProtoNet类并初始化模型

from uni.downstream.eval_patch_features.protonet import ProtoNet

proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
功能：从自定义路径导入ProtoNet类，并初始化模型。
参数解析：
metric = 'L2'：使用欧氏距离（L2距离）作为相似性度量。
center_feats = True：在训练时对特征进行中心化（减去均值）。
normalize_feats = True：对特征进行L2归一化。
作用：通过中心化和归一化，确保特征分布的一致性，提升原型学习的鲁棒性。
8 - 2：训练模型

proto_clf.fit(train_feats, train_labels)
print('What our prototypes look like', proto_clf.prototype_embeddings.shape)
功能：用训练数据拟合模型，生成每个类别的原型（prototype）。
原型生成逻辑：
对每个类别，计算其所有样本特征的均值作为原型。例如，若类别A有100个样本，则原型为这100个特征的均值向量。
输出示例：(num_classes, feature_dim)，表示原型数量为类别数，每个原型的维度与输入特征相同。
8 - 3：模型预测与评估

test_pred = proto_clf.predict(test_feats)
get_eval_metrics(test_labels, test_pred, get_report=False)
预测逻辑：对于每个测试样本，计算其与所有原型的距离，选择最近的原型对应的类别作为预测结果。
指标名称
10
K
100
K
准确率（acc）
0.8863509749303621
0.883844011
平衡准确率（bacc）
0.8569268574763045
0.853915962
Cohen
's kappa系数（kappa）
0.9115189269748553
0.910136068
加权F1值（weighted_f1）
0.8791473464186615
0.875913657
8 - 4：基于原型的检索（ROI
Retrieval）

1.
获取TopK查询索引
dist, topk_inds = proto_clf._get_topk_queries_inds(test_feats, topk=5)
作用：以测试特征作为查询集，计算每个原型最近的topk个测试样本
输入：
test_feats：测试集的特征矩阵（形状：[n_test_samples, feature_dim]）
topk = 5：每个原型取前5个最近样本
输出：
dist：距离矩阵（形状：[n_prototypes, n_test_samples]）
topk_inds：索引矩阵（形状：[n_prototypes, topk]），每行对应一个原型的前5个最近测试样本的索引
2.
打印类别 - 索引映射关系
print('label2idx correspondenes', test_dataset.class_to_idx)
作用：显示数据集中类别名称与数字索引的对应关系（例如：{'ADIPOSE': 0, 'LYMPHOCYTE': 3, ...}）
意义：解释后续代码中topk_inds[0]
为什么对应ADIPOSE原型
输出如下：
label2idx
correspondenes
{'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
3.
构建测试集图片路径DataFrame
test_imgs_df = pd.DataFrame(test_dataset.imgs, columns=['path', 'label'])
结构

test_dataset.imgs：Pytorch
Dataset的标准属性，包含元组列表[(图片路径1, 标签1), ...]
转换为DataFrame后有两列：
path：图片文件的绝对 / 相对路径
label：对应的数字标签（需通过class_to_idx映射为类别名）
4.
ADIPOSE类别的Top5样本可视化
print('Top-k ADIPOSE-like test samples to ADIPOSE prototype')
adi_topk_inds = topk_inds[0]  # 假设ADIPOSE原型的索引是0
adi_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][adi_topk_inds]], scale=0.5,
                              gap=5)
display(adi_topk_imgs)
流程

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
tum_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][tum_topk_inds]], scale=0.5,
                              gap=5)
display(tum_topk_imgs)
image - 20250317214336418
image - 20250317214336418
其余类型的可视化分析代码列举如下，我不再展示。

print('Top-k MUCOSA-like test samples to MUCOSA prototype')
muc_topk_inds = topk_inds[4]
muc_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][muc_topk_inds]], scale=0.5,
                              gap=5)
display(muc_topk_imgs)

print('Top-k MUSCLE-like test samples to MUSCLE prototype')
mus_topk_inds = topk_inds[5]
mus_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][mus_topk_inds]], scale=0.5,
                              gap=5)
display(mus_topk_imgs)

print('Top-k NORMAL-like test samples to NORMAL prototype')
norm_topk_inds = topk_inds[6]
norm_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][norm_topk_inds]], scale=0.5,
                               gap=5)
display(norm_topk_imgs)

print('Top-k STROMA-like test samples to STROMA prototype')
str_topk_inds = topk_inds[7]
str_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][str_topk_inds]], scale=0.5,
                              gap=5)
display(str_topk_imgs)
科研合作意向统计

为了更好的利用小罗搭建的交流平台，我决定发放一个长期有效的问卷，征集大家在科研方面的任何需求，并且定期整理汇总，方便大家课题合作，招收学生，联系导师……

图片
结束语

本期推文的内容就到这里啦，如果需要获取医学AI领域的最新发展动态，请关注小罗的推送！如需进一步深入研究，获取相关资料，欢迎加入我的知识星球！
