"""顺序堆叠多个 Decoder Transformer Block。"""

from __future__ import annotations

import torch
from torch import nn

from model_lab.transformer_block import TransformerBlock


class DecoderStack(nn.Module):
    """让形状 [B, T, C] 的隐藏状态依次通过 N 个 Transformer Block。"""

    def __init__(
        self, embedding_dim: int, num_heads: int, num_layers: int
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")

        self.num_layers = num_layers

        # ModuleList 会注册每个 Block 及其参数。每一层结构相同，但拥有独立的
        # Attention、Feed-Forward 和 Layer Normalization 参数。
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回最后一个 Block 输出的 [B, T, C] 隐藏状态。"""

        # 第 l 层的输出就是第 l+1 层的输入。每个 Block 都保持 [B, T, C]，
        # 因此堆叠只增加计算深度，不改变 Batch、Sequence 或 Feature 维度。
        for block in self.blocks:
            x = block(x)
        return x
