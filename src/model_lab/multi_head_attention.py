"""以可读的数学步骤实现多头因果自注意力。"""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """对形状 [B, T, C] 的输入计算多头因果自注意力。"""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # C = Embedding Dimension（嵌入维度）。Q、K、V 投影先保留完整的 C，
        # 后续再把 C 均分给 H 个 Head；四个投影都不使用 bias。
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回上下文表示和每个 Head 的注意力权重。"""

        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, C]")
        if x.shape[1] <= 0:
            raise ValueError("sequence length must be greater than zero")
        if x.shape[2] != self.embedding_dim:
            raise ValueError("x must have embedding dimension C equal to embedding_dim")

        # B = Batch Size（批次大小），T = Sequence Length（序列长度），
        # C = Embedding Dimension（嵌入维度），H = Head 数量，
        # D = Head Dimension（单个 Head 的维度），并且 C = H × D。
        batch_size, sequence_length, _ = x.shape

        # 每个投影都只变换最后一维，因此 X、Q、K、V 的形状都是 [B, T, C]。
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # [B, T, C] reshape 为 [B, T, H, D]，把完整特征维 C 拆成 H 个 Head。
        # transpose 再把 Head 提到序列维之前，得到 [B, H, T, D]，这样 H 会像
        # Batch 一样并行参与后续矩阵乘法，无需用 Python for-loop 逐 Head 计算。
        query = query.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # K^T 是 [B, H, D, T]；[B, H, T, D] @ [B, H, D, T]
        # 得到 [B, H, T, T]。最后两个 T 分别表示 Query 位置和 Key 位置，
        # 所以每个 Batch、每个 Head 都有一张独立的 Token 关系矩阵。
        scores = query @ key.transpose(-2, -1)

        # 每个 Head 的点积只沿 D 个特征求和，因此 scale 必须是 sqrt(D)，
        # 而不是 sqrt(C)。缩放可避免 D 增大时 Softmax 过早变得尖锐。
        scores = scores / math.sqrt(self.head_dim)

        # causal_mask 只需要保存一张 [T, T] 的位置关系表。它没有 Batch 和 Head
        # 维度，但 masked_fill 会将它广播到 [B, H, T, T]，让所有样本、所有
        # Head 同时屏蔽对角线以上的未来位置。
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # dim=-1 沿每个 Query 对应的全部 Key 位置归一化。
        # attention_weights[b, h, i, :] 的总和约为 1，形状保持 [B, H, T, T]。
        attention_weights = torch.softmax(scores, dim=-1)

        # [B, H, T, T] @ [B, H, T, D] = [B, H, T, D]：
        # 每个 Head 使用自己的权重独立聚合 Value。
        context = attention_weights @ value

        # transpose 将 [B, H, T, D] 还原为 [B, T, H, D]，reshape 再把相邻的
        # H 与 D 拼接回 C，得到 [B, T, C]。这一步就是 Concat Heads。
        context = context.transpose(1, 2).reshape(
            batch_size, sequence_length, self.embedding_dim
        )

        # Output Projection（输出投影）在 C 维上混合刚拼接的各 Head 信息，
        # 让输出位置能够组合不同 Head 学到的关系，同时保持形状 [B, T, C]。
        context = self.output_projection(context)

        return context, attention_weights
