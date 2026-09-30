"""以可读的数学步骤计算 next-token Cross Entropy Loss。"""

from __future__ import annotations

import torch


def cross_entropy_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """返回 [B, T, V] Logits 对 [B, T] 非 -100 targets 的平均损失。"""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [B, T]")
    if target_ids.dtype != torch.int64:
        raise TypeError("target_ids must use torch.int64")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits and target_ids must have matching B and T dimensions")
    if logits.shape[0] <= 0:
        raise ValueError("batch size must be greater than zero")
    if logits.shape[1] <= 0:
        raise ValueError("sequence length must be greater than zero")
    if logits.shape[2] <= 0:
        raise ValueError("vocabulary size must be greater than zero")

    # -100 是回答训练和右侧补齐使用的忽略标签；只有真实监督位置参与平均。
    supervised = target_ids != -100
    if not supervised.any():
        raise ValueError("targets must contain at least one supervised Token")

    # V = Vocabulary Size（词表大小）。log_softmax 沿 V 维把每个位置的
    # 原始 Logits 转换为对数概率；它比先 softmax 再 log 更稳定。
    log_probabilities = torch.log_softmax(logits, dim=-1)

    # target_ids [B, T] 增加一个大小为 1 的 V 维后得到 [B, T, 1]。
    # gather 为每个位置选出正确 Token 对应的一个对数概率。
    target_log_probabilities = log_probabilities.gather(
        dim=-1,
        index=target_ids.masked_fill(~supervised, 0).unsqueeze(dim=-1),
    ).squeeze(dim=-1)

    # Negative Log-Likelihood（负对数似然）对正确 Token 的 log probability
    # 取负，再对实际监督位置求平均，得到标量 Loss。
    return -target_log_probabilities[supervised].mean()
