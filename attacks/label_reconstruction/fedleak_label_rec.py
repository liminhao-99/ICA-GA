# --------------------------------------------------
# Boosting Gradient Leakage Attacks: Data Reconstruction in Realistic FL Settings
# https://arxiv.org/abs/2506.08435v1
# FedLeak 标签推理 Label inference
# --------------------------------------------------

import torch
import numpy as np


def fedleak_lab_rec(dW: torch.Tensor, batch_size: int):
    """
    Args:
        grad_w (torch.Tensor): 最后一层全连接层的梯度张量。
                               形状应为 (N, K)，其中 N 是类别总数，K 是特征维度。
        batch_size (int): 训练批次的大小。

    Returns:
        torch.Tensor: 一个长度为 batch_size 的一维张量，包含了推理出的所有标签。
    """

    num_classes = dW.shape[0]

    # ------------------ 聚合梯度 ------------------
    # 沿着特征维度（K）对梯度进行求和，得到一个长度为 N (类别总数) 的梯度和向量。
    # 这个向量中每个元素的值，代表了对应类别的“梯度强度”。
    # 根据iDLG等工作的结论，真实标签对应的梯度和通常为负数。
    aggregated_grads = torch.sum(dW, dim=1)

    # 情况一: 批次大小 <= 类别总数
    # 这种情况下，算法假设批次中的标签大概率是不重复的。
    if batch_size <= num_classes:
        # 找到梯度强度最小（即最负）的 batch_size 个类别。
        # 使用 torch.argsort() 可以得到排序后的索引，默认是升序排列。
        # 因此，取前 batch_size 个索引，即为梯度最小的那些类别的索引。
        # 这与基础的iDLG思想一致。
        inferred_labels = torch.argsort(aggregated_grads)[:batch_size]

    # 情况二: 批次大小 > 类别总数。可以确定批次中必然存在重复的标签。
    # 算法的目标是估算出每个标签重复了多少次。
    else:
        # --- 计算基准线 max_W ---
        # 找到所有梯度和中的最大值。这个值理论上对应于“未在批次中出现的标签”的梯度。
        # 将这个值作为后续计算的“零点”基准。
        max_w = torch.max(aggregated_grads)

        # --- 计算每个类别的相对梯度强度 ---
        # 公式: numerator_j = Σ (ΔW[i, j] - max_W)
        # 这等价于 aggregated_grads[j] - max_W
        # 这个操作使得未出现标签的梯度强度接近0，而出现过的标签的梯度强度为一个较大的负数。
        relative_grads = aggregated_grads - max_w

        # --- 计算总的相对梯度强度 (分母) ---
        # 对所有类别的相对梯度强度求和，用于后续的归一化。
        # 添加一个极小值 epsilon 防止分母为零。
        total_relative_grad = torch.sum(relative_grads)
        epsilon = 1e-9

        # --- 按比例估算每个类别的数量 ---
        # 一个标签的计数值(count)，正比于它的相对梯度强度在总强度中的占比。
        # 将这个比例乘以总批次大小，即可估算出该标签的数量。
        # 这里的计算结果是浮点数，需要四舍五入为整数。
        counts = torch.round(
            batch_size * (relative_grads / (total_relative_grad - epsilon))
        )

        # 将counts转换为整数类型
        counts = counts.long()
        # 由于浮点数计算和取整的误差，可能需要处理负数计数值（理论上不应出现）
        counts[counts < 0] = 0

        # --- 迭代补充，凑够批次大小 ---
        # 由于估算和取整存在误差，当前所有标签的总数可能不等于 batch_size。
        # 这个循环的作用就是“凑数”。
        current_total_count = torch.sum(counts)
        while current_total_count < batch_size:
            # 找到当前梯度强度最低（最负）的那个类别。
            # 这被认为是“最有可能再次出现”的标签。
            idx_to_increment = torch.argmin(aggregated_grads)

            # 为该类别的计数值加一。
            counts[idx_to_increment] += 1

            # 更新当前总数。
            current_total_count += 1

        # --- 根据计数值构建最终的标签列表 ---
        inferred_labels = []
        for label_idx, count in enumerate(counts):
            for _ in range(count.item()):
                inferred_labels.append(label_idx)
        inferred_labels = torch.as_tensor(inferred_labels, dtype=torch.long)

        # 如果由于取整误差导致标签超出，则进行截断
        if len(inferred_labels) > batch_size:
            inferred_labels = inferred_labels[:batch_size]

    return inferred_labels
