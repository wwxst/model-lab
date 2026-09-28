"""把 Decoder 隐藏状态投影为词表 Logits。"""

from __future__ import annotations

import torch
from torch import nn


class LanguageModelHead(nn.Module):
    """将形状 [B, T, C] 的隐藏状态投影为 [B, T, V] Logits。"""

    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero")

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # C = Embedding Dimension（嵌入维度），V = Vocabulary Size（词表大小）。
        # [V, C] 权重矩阵为每个词表 Token 保存一行可学习的输出方向。
        self.projection = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """返回未经过 Softmax 的 [B, T, V] Logits。"""

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, C]")
        if hidden_states.shape[1] <= 0:
            raise ValueError("sequence length must be greater than zero")
        if hidden_states.shape[2] != self.embedding_dim:
            raise ValueError(
                "hidden_states must have embedding dimension C equal to embedding_dim"
            )

        # Linear 只把最后一维从 C 变成 V，B 和 T 保持不变。每个位置得到 V 个
        # 原始分数，每个分数表示该位置的下一个 Token 取某个词表项的相对倾向。
        return self.projection(hidden_states)
