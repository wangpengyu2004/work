import yaml
from collections import deque
import random
import os
import logging
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
logger = logging.getLogger()


def set_random_seed(seed: int = 2024, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set random seed to {seed}.")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("The CUDNN is set to deterministic. This will increase reproducibility, "
                    "but may slow down the training considerably.")
def show_class(features,labels,name): 
    # 设置图形
    plt.figure(figsize=(8, 8))
    # 遍历每个批次
    for batch_idx, (batch_features, batch_labels) in enumerate(zip(features, labels)):
        # 使用 PCA 对当前批次的特征降维
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(batch_features)
        batch_labels = batch_labels.detach().numpy()
        
        # 在同一张图中绘制当前批次的样本
        for label in set(batch_labels):
            indices = batch_labels == label
            class_features = reduced_features[indices]
            
            # 为每个点添加随机偏移量 (jitter)，避免重叠
            jitter = np.random.normal(loc=0, scale=0.02, size=class_features.shape)  # 调整 scale 控制偏移量大小
            jittered_features = class_features + jitter

            plt.scatter(
                jittered_features[:, 0], 
                jittered_features[:, 1], 
                label=f'Class {label}' if batch_idx == 0 else None,  # 避免重复图例
                alpha=0.7
            )
    # 添加图例和标题
    plt.legend()
    plt.title("Incremental PCA Visualization for All Batches")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()
    plt.savefig(f"/home/wpy/wpy_workspace/work/clip4emo/{name}.png")
