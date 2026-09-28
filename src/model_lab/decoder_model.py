"""组合完整的 Decoder-only Transformer Language Model。"""

from __future__ import annotations

import torch
from torch import nn

from model_lab.decoder_stack import DecoderStack
from model_lab.language_model_head import LanguageModelHead
from model_lab.position_embedding import PositionEmbedding
from model_lab.residual_normalization import LayerNormalization
from model_lab.token_embedding import TokenEmbedding


class DecoderOnlyLanguageModel(nn.Module):
    """将形状 [B, T] 的 Token IDs 转换为 [B, T, V] Logits。"""

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim

        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionEmbedding(
            max_sequence_length, embedding_dim
        )
        self.decoder_stack = DecoderStack(embedding_dim, num_heads, num_layers)
        self.final_normalization = LayerNormalization(embedding_dim)
        self.language_model_head = LanguageModelHead(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """返回每个输入位置预测下一个 Token 的 [B, T, V] Logits。"""

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, T]")
        if token_ids.dtype != torch.int64:
            raise TypeError("token_ids must use torch.int64")

        sequence_length = token_ids.shape[1]
        if sequence_length <= 0:
            raise ValueError("sequence length must be greater than zero")
        if sequence_length > self.max_sequence_length:
            raise ValueError("sequence length must not exceed max_sequence_length")

        # Token Embedding 把“是什么”映射为 [B, T, C]，Position Embedding 把
        # “在哪里”映射为 [T, C]。相加时位置表示沿 Batch 维广播。
        token_representations = self.token_embedding(token_ids)
        position_representations = self.position_embedding(sequence_length)
        hidden_states = token_representations + position_representations

        # Decoder Stack 逐层更新 [B, T, C] 隐藏状态。Pre-Norm Block 之后再做
        # 一次 Final Layer Normalization，然后 Head 将 C 投影为词表维 V。
        hidden_states = self.decoder_stack(hidden_states)
        hidden_states = self.final_normalization(hidden_states)
        return self.language_model_head(hidden_states)
