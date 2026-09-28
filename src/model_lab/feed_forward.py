"""以可读的形状变化实现 Transformer 中的前馈网络。"""

from __future__ import annotations

import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """对形状 [B, T, C] 中每个 Token 的特征独立执行两层变换。"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")

        self.embedding_dim = embedding_dim
        self.hidden_dim = 4 * embedding_dim

        # C = Embedding Dimension（嵌入维度）。第一层把每个 Token 的特征
        # 从 C 扩展到 4C，让网络能在更大的中间空间中组合特征。
        self.input_projection = nn.Linear(embedding_dim, self.hidden_dim)
        self.activation = nn.GELU()

        # 第二层再把 4C 收缩回 C，因此输出可以与输入保持相同 shape，
        # 并在后续 Transformer Block 中参与 Residual Connection（残差连接）。
        self.output_projection = nn.Linear(self.hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状仍为 [B, T, C] 的逐 Token 特征变换结果。"""

        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, C]")
        if x.shape[1] <= 0:
            raise ValueError("sequence length must be greater than zero")
        if x.shape[2] != self.embedding_dim:
            raise ValueError("x must have embedding dimension C equal to embedding_dim")

        # nn.Linear 只变换最后一维。B = Batch Size（批次大小），
        # T = Sequence Length（序列长度），所以 B 和 T 在整个过程中保持不变：
        # [B, T, C] -> [B, T, 4C] -> [B, T, 4C] -> [B, T, C]。
        hidden = self.input_projection(x)
        hidden = self.activation(hidden)
        return self.output_projection(hidden)
