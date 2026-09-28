import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from model_lab.feed_forward import FeedForwardNetwork
from model_lab.multi_head_attention import MultiHeadSelfAttention


def test_constructor_rejects_non_positive_embedding_dimension() -> None:
    with pytest.raises(ValueError, match="embedding_dim"):
        FeedForwardNetwork(embedding_dim=0)


def test_projection_shapes_expand_to_four_times_embedding_dimension() -> None:
    feed_forward = FeedForwardNetwork(embedding_dim=6)

    assert feed_forward.hidden_dim == 24
    assert feed_forward.input_projection.weight.shape == torch.Size([24, 6])
    assert feed_forward.input_projection.bias is not None
    assert feed_forward.input_projection.bias.shape == torch.Size([24])
    assert feed_forward.output_projection.weight.shape == torch.Size([6, 24])
    assert feed_forward.output_projection.bias is not None
    assert feed_forward.output_projection.bias.shape == torch.Size([6])


def test_forward_preserves_batch_sequence_and_embedding_dimensions() -> None:
    output = FeedForwardNetwork(embedding_dim=6)(torch.randn(2, 4, 6))

    assert output.shape == torch.Size([2, 4, 6])


def test_known_projections_match_manual_gelu_example() -> None:
    feed_forward = FeedForwardNetwork(embedding_dim=2)
    with torch.no_grad():
        feed_forward.input_projection.weight.zero_()
        feed_forward.input_projection.bias.zero_()
        feed_forward.input_projection.weight[0, 0] = 1.0
        feed_forward.input_projection.weight[1, 1] = 1.0

        feed_forward.output_projection.weight.zero_()
        feed_forward.output_projection.bias.copy_(torch.tensor([0.5, -0.5]))
        feed_forward.output_projection.weight[0, 0] = 1.0
        feed_forward.output_projection.weight[1, 1] = 2.0

    x = torch.tensor([[[1.0, -1.0], [2.0, 0.5]]])
    output = feed_forward(x)
    expected = torch.stack(
        (
            functional.gelu(x[..., 0]) + 0.5,
            2.0 * functional.gelu(x[..., 1]) - 0.5,
        ),
        dim=-1,
    )

    assert torch.allclose(output, expected)


def test_changing_one_token_does_not_change_other_token_outputs() -> None:
    feed_forward = FeedForwardNetwork(embedding_dim=4)
    x1 = torch.randn(1, 3, 4)
    x2 = x1.clone()
    x2[:, 2, :] = torch.tensor([100.0, -100.0, 50.0, -50.0])

    output1 = feed_forward(x1)
    output2 = feed_forward(x2)

    assert torch.equal(output1[:, :2], output2[:, :2])


def test_forward_preserves_input_dtype_and_device() -> None:
    feed_forward = FeedForwardNetwork(embedding_dim=4).to(dtype=torch.float64)
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    output = feed_forward(x)

    assert output.dtype == x.dtype
    assert output.device == x.device


def test_backward_computes_input_and_projection_gradients() -> None:
    feed_forward = FeedForwardNetwork(embedding_dim=4)
    x = torch.randn(2, 3, 4, requires_grad=True)

    feed_forward(x).square().sum().backward()

    assert x.grad is not None
    assert feed_forward.input_projection.weight.grad is not None
    assert feed_forward.input_projection.bias is not None
    assert feed_forward.input_projection.bias.grad is not None
    assert feed_forward.output_projection.weight.grad is not None
    assert feed_forward.output_projection.bias is not None
    assert feed_forward.output_projection.bias.grad is not None


def test_forward_rejects_non_batched_input() -> None:
    feed_forward = FeedForwardNetwork(embedding_dim=4)

    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        feed_forward(torch.randn(3, 4))


def test_forward_rejects_empty_sequence() -> None:
    feed_forward = FeedForwardNetwork(embedding_dim=4)

    with pytest.raises(ValueError, match="sequence length"):
        feed_forward(torch.randn(2, 0, 4))


def test_forward_rejects_wrong_embedding_dimension() -> None:
    feed_forward = FeedForwardNetwork(embedding_dim=4)

    with pytest.raises(ValueError, match="embedding dimension"):
        feed_forward(torch.randn(2, 3, 6))


def test_attention_context_connects_to_feed_forward_network() -> None:
    attention = MultiHeadSelfAttention(embedding_dim=4, num_heads=2)
    context, _ = attention(torch.randn(2, 3, 4))

    feed_forward = FeedForwardNetwork(embedding_dim=4)
    output = feed_forward(context)

    assert context.shape == torch.Size([2, 3, 4])
    assert output.shape == torch.Size([2, 3, 4])
    assert isinstance(feed_forward, nn.Module)
