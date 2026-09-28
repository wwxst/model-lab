"""组合 Pre-Norm Decoder Transformer Block。"""

from __future__ import annotations

import torch
from torch import nn

from model_lab.feed_forward import FeedForwardNetwork
from model_lab.multi_head_attention import MultiHeadSelfAttention
from model_lab.residual_normalization import LayerNormalization, add_residual


class TransformerBlock(nn.Module):
    """按 Pre-Norm 顺序组合因果自注意力、前馈网络和两条残差路径。"""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()

        # 两个子层分别拥有自己的 Layer Normalization 参数。Attention 负责
        # Token 之间的信息交互，Feed-Forward 负责逐 Token 变换特征。
        self.attention_normalization = LayerNormalization(embedding_dim)
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.feed_forward_normalization = LayerNormalization(embedding_dim)
        self.feed_forward = FeedForwardNetwork(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状仍为 [B, T, C] 的 Transformer Block 输出。"""

        # Pre-Norm 先归一化输入，再送入 Attention。归一化和 Attention 都保持
        # [B, T, C]，所以 Attention 输出可以与原始 x 逐元素相加。
        normalized_attention_input = self.attention_normalization(x)
        attention_output, _ = self.attention(normalized_attention_input)
        x = add_residual(x, attention_output)

        # 第二个子层同样先归一化。Feed-Forward 在 C -> 4C -> C 后恢复
        # [B, T, C]，因此可以通过第二条残差路径加回 Attention 后的 x。
        normalized_feed_forward_input = self.feed_forward_normalization(x)
        feed_forward_output = self.feed_forward(normalized_feed_forward_input)
        return add_residual(x, feed_forward_output)
