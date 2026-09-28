"""以可读的数学步骤实现残差连接和层归一化。"""

from __future__ import annotations

import torch
from torch import nn


def add_residual(x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
    """将形状相同的输入和子层输出逐元素相加。"""

    if x.ndim != 3 or sublayer_output.ndim != 3:
        raise ValueError("x and sublayer_output must have shape [B, T, C]")
    if x.shape != sublayer_output.shape:
        raise ValueError("x and sublayer_output must have the same shape")

    # B = Batch Size（批次大小），T = Sequence Length（序列长度），
    # C = Embedding Dimension（嵌入维度）。两个 [B, T, C] 张量逐元素相加，
    # 不改变任何维度，也不混合不同 Batch、Token 或 Feature 的位置。
    return x + sublayer_output


class LayerNormalization(nn.Module):
    """对形状 [B, T, C] 中每个 Token 的 C 个特征独立归一化。"""

    def __init__(self, embedding_dim: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than zero")

        self.embedding_dim = embedding_dim
        self.epsilon = epsilon

        # C = Embedding Dimension（嵌入维度）。每个特征都有一个可学习的
        # scale（缩放）和 shift（平移），初始值让归一化结果保持不变。
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状仍为 [B, T, C] 的归一化结果。"""

        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, C]")
        if x.shape[1] <= 0:
            raise ValueError("sequence length must be greater than zero")
        if x.shape[2] != self.embedding_dim:
            raise ValueError("x must have embedding dimension C equal to embedding_dim")

        # dim=-1 只沿每个 Token 的 C 个特征求均值。keepdim=True 保留最后
        # 一个大小为 1 的维度，使 mean 能广播回原来的 [B, T, C]。
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean

        # variance 使用总体方差：每个中心化特征先平方，再沿 C 求平均。
        # epsilon 防止方差为 0 时除以 0，同时保持 shape 为 [B, T, 1]。
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.epsilon)

        # scale 和 shift 的形状都是 [C]。PyTorch 将它们广播到 [B, T, C]，
        # 让模型在训练中学习每个特征适合的缩放和平移。
        return self.scale * normalized + self.shift
