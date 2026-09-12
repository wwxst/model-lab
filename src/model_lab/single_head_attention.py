"""以可读的数学步骤实现单头因果自注意力。"""

from __future__ import annotations

import math

import torch
from torch import nn


class SingleHeadSelfAttention(nn.Module):
    """对形状 [B, T, C] 的输入计算单头因果自注意力。"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")

        self.embedding_dim = embedding_dim

        # C = Embedding Dimension（嵌入维度）。三个独立投影都把最后一维 C
        # 映射回 C：它们从同一个 X 产生 Query、Key、Value，因此称为 Self-Attention。
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回上下文表示和注意力权重。"""

        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, C]")
        if x.shape[1] <= 0:
            raise ValueError("sequence length must be greater than zero")

        # B = Batch Size（批次大小），T = Sequence Length（序列长度），
        # C = Hidden Dimension（隐藏/嵌入维度）。输入 X 的形状是 [B, T, C]。
        # 每个投影都对最后一维做线性变换，所以 Q、K、V 仍然是 [B, T, C]。
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # K 从 [B, T, C] 交换最后两个维度后得到 [B, C, T]。
        # [B, T, C] @ [B, C, T] = [B, T, T]：
        # scores[b, i, j] 是第 b 个样本中，位置 i 的 Query 与位置 j 的 Key 的匹配分数。
        scores = query @ key.transpose(-2, -1)

        # 当 C 较大时，点积的数值幅度容易变大，Softmax 会过于尖锐。
        # d_k 在当前单头实现中就是 C，因此按公式 QK^T / sqrt(C) 缩放。
        scores = scores / math.sqrt(self.embedding_dim)

        # causal_mask 为 [T, T] 的布尔矩阵；对角线以上（j > i）代表未来位置，
        # 这些位置不能被当前位置读取。它会沿 Batch 维广播到 [B, T, T]。
        sequence_length = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )

        # 必须在 Softmax 之前把未来分数设为 -inf；softmax(-inf) = 0，
        # 所以被遮住的 Attention Weight 严格为零，而不是先归一化后再手工清零。
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # dim=-1 沿每个 Query 的全部 Key 位置归一化，得到形状 [B, T, T] 的
        # 概率分布 attention_weights[b, i, :]，每行总和约为 1。
        attention_weights = torch.softmax(scores, dim=-1)

        # [B, T, T] @ [B, T, C] = [B, T, C]。
        # 每个位置用自己的权重把可见位置的 Value 向量加权求和，形成上下文表示。
        context = attention_weights @ value

        return context, attention_weights
