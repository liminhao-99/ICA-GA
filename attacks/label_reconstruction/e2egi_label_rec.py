# --------------------------------------------------
# E2EGI: End-to-End Gradient Inversion in Federated Learning
# 标签重建 Label Reconstruction
# --------------------------------------------------

import torch
from utils import logger


def e2egi_lab_rec(dW: torch.Tensor, batch_size: int):
    """
    Args:
        grad_w (torch.Tensor): 最后一层全连接层的梯度张量。
                               形状应为 (N, K)，其中 N 是类别总数，K 是特征维度。
        batch_size (int): 训练批次的大小。

    Returns:
        torch.Tensor: 一个长度为 batch_size 的一维张量，包含了推理出的所有标签。

    https://github.com/zhaohuali/E2EGI/blob/e99d601fa610c5f5b089ba900f93161c6955f562/kernels/__init__.py#L35
    """
    num_classes = dW.shape[0]
    g = dW.sum(-1)
    # C = g[torch.where(g > 0)[0]].max()

    positive_g_indices = torch.where(g > 0)[0]
    if len(positive_g_indices) > 0:
        C = g[positive_g_indices].max()
    else:
        logger.warning(f"E2egi Label Reconstruction: 分类层梯度不存在>0的项！")
        C = 0.1

    m = num_classes * C / batch_size

    pred_label = []
    for i, gi in enumerate(g):
        if gi < 0:
            pred_label.append(i)
            g[i] += m
    while len(pred_label) < batch_size:
        idx = g.argmin().item()
        pred_label.append(idx)
        g[idx] += m

    return torch.as_tensor(pred_label)
