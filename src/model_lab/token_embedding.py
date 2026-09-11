"""把离散 Token ID 映射为可学习的连续向量。"""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """为词表中的每个 Token 保存一个可学习向量。"""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")

        # V = Vocabulary Size（词表大小），C = Embedding Dimension（嵌入维度）。
        # nn.Embedding 维护形状为 [V, C] 的参数表，每一行属于一个 Token ID。
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    @property
    def weight(self) -> torch.Tensor:
        """返回形状为 [V, C] 的可学习 Embedding Table（嵌入表）。"""

        return self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """用 Token ID 查表，在输入形状末尾增加连续特征维度 C。"""

        # 单序列 [T] 查表后得到 [T, C]；批量序列 [B, T] 得到 [B, T, C]。
        # B = Batch Size（批次大小），T = Sequence Length（序列长度）。
        # 查表不会改变 B 或 T，只为每个 Token ID 取出参数表中的一行向量。
        return self.embedding(token_ids)
