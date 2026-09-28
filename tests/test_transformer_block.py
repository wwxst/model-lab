import pytest
import torch
from torch import nn

from model_lab.feed_forward import FeedForwardNetwork
from model_lab.multi_head_attention import MultiHeadSelfAttention
from model_lab.residual_normalization import LayerNormalization, add_residual
from model_lab.transformer_block import TransformerBlock


def test_block_contains_two_normalizations_attention_and_feed_forward() -> None:
    block = TransformerBlock(embedding_dim=6, num_heads=3)

    assert isinstance(block.attention_normalization, LayerNormalization)
    assert isinstance(block.attention, MultiHeadSelfAttention)
    assert isinstance(block.feed_forward_normalization, LayerNormalization)
    assert isinstance(block.feed_forward, FeedForwardNetwork)
    assert block.attention_normalization is not block.feed_forward_normalization
    assert isinstance(block, nn.Module)


@pytest.mark.parametrize(
    ("embedding_dim", "num_heads", "message"),
    [
        (0, 1, "embedding_dim"),
        (4, 0, "num_heads"),
        (5, 2, "divisible"),
    ],
)
def test_constructor_rejects_invalid_attention_dimensions(
    embedding_dim: int, num_heads: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads)


def test_forward_preserves_batch_sequence_and_embedding_dimensions() -> None:
    block = TransformerBlock(embedding_dim=6, num_heads=3)

    output = block(torch.randn(2, 4, 6))

    assert output.shape == torch.Size([2, 4, 6])


def test_forward_matches_explicit_pre_norm_sequence() -> None:
    block = TransformerBlock(embedding_dim=4, num_heads=2)
    x = torch.randn(2, 3, 4)

    normalized_attention_input = block.attention_normalization(x)
    attention_output, _ = block.attention(normalized_attention_input)
    after_attention = add_residual(x, attention_output)
    normalized_feed_forward_input = block.feed_forward_normalization(after_attention)
    feed_forward_output = block.feed_forward(normalized_feed_forward_input)
    expected = add_residual(after_attention, feed_forward_output)

    assert torch.equal(block(x), expected)


def test_zero_sublayer_outputs_leave_residual_input_unchanged() -> None:
    block = TransformerBlock(embedding_dim=4, num_heads=2)
    with torch.no_grad():
        block.attention.output_projection.weight.zero_()
        block.feed_forward.output_projection.weight.zero_()
        block.feed_forward.output_projection.bias.zero_()
    x = torch.randn(2, 3, 4)

    output = block(x)

    assert torch.equal(output, x)


def test_future_token_does_not_change_earlier_block_outputs() -> None:
    block = TransformerBlock(embedding_dim=4, num_heads=2)
    x1 = torch.randn(1, 4, 4)
    x2 = x1.clone()
    x2[:, 3, :] = torch.tensor([100.0, -100.0, 50.0, -50.0])

    output1 = block(x1)
    output2 = block(x2)

    assert torch.allclose(output1[:, :3], output2[:, :3])


def test_forward_preserves_input_dtype_and_device() -> None:
    block = TransformerBlock(embedding_dim=4, num_heads=2).to(dtype=torch.float64)
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    output = block(x)

    assert output.dtype == x.dtype
    assert output.device == x.device


def test_backward_reaches_input_and_every_block_parameter() -> None:
    block = TransformerBlock(embedding_dim=4, num_heads=2)
    x = torch.randn(2, 3, 4, requires_grad=True)

    block(x).square().sum().backward()

    assert x.grad is not None
    for parameter in block.parameters():
        assert parameter.grad is not None


def test_forward_rejects_non_batched_input() -> None:
    block = TransformerBlock(embedding_dim=4, num_heads=2)

    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        block(torch.randn(3, 4))


def test_forward_rejects_empty_sequence() -> None:
    block = TransformerBlock(embedding_dim=4, num_heads=2)

    with pytest.raises(ValueError, match="sequence length"):
        block(torch.randn(2, 0, 4))


def test_forward_rejects_wrong_embedding_dimension() -> None:
    block = TransformerBlock(embedding_dim=4, num_heads=2)

    with pytest.raises(ValueError, match="embedding dimension"):
        block(torch.randn(2, 3, 6))
