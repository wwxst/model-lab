"""为序列位置提供可学习的连续向量表示。"""

from __future__ import annotations

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    """将序列中的绝对位置编号映射为可学习的位置向量。"""

    def __init__(self, max_sequence_length: int, embedding_dim: int) -> None:
        super().__init__()
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be greater than zero")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")

        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim

        # P = Maximum Sequence Length（最大序列长度）。
        # C = Embedding Dimension（嵌入维度）。
        # 每一行代表一个绝对位置的可学习向量，参数表形状为 [P, C]。
        self.embedding = nn.Embedding(max_sequence_length, embedding_dim)

    @property
    def weight(self) -> torch.Tensor:
        """返回形状为 [P, C] 的可学习位置参数表。"""

        return self.embedding.weight

    def forward(self, sequence_length: int) -> torch.Tensor:
        """生成 0 到 T-1 的 Position ID，并查表返回形状 [T, C] 的向量。"""

        if type(sequence_length) is not int:
            raise TypeError("sequence_length must be an integer")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")
        if sequence_length > self.max_sequence_length:
            raise ValueError("sequence_length must not exceed max_sequence_length")

        # T = Sequence Length（序列长度）。Position ID 与 Token ID 都是 int64，
        # 但 Token ID 表示“是什么”，Position ID 表示“在哪里”。
        position_ids = torch.arange(
            sequence_length,
            dtype=torch.int64,
            device=self.weight.device,
        )
        return self.embedding(position_ids)
