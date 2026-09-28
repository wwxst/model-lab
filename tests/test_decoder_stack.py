import pytest
import torch
from torch import nn

from model_lab.decoder_stack import DecoderStack
from model_lab.transformer_block import TransformerBlock


def zero_sublayer_outputs(stack: DecoderStack) -> None:
    with torch.no_grad():
        for block in stack.blocks:
            assert isinstance(block, TransformerBlock)
            block.attention.output_projection.weight.zero_()
            block.feed_forward.output_projection.weight.zero_()
            block.feed_forward.output_projection.bias.zero_()


def test_constructor_rejects_non_positive_layer_count() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        DecoderStack(embedding_dim=4, num_heads=2, num_layers=0)


@pytest.mark.parametrize(
    ("embedding_dim", "num_heads", "message"),
    [
        (0, 1, "embedding_dim"),
        (4, 0, "num_heads"),
        (5, 2, "divisible"),
    ],
)
def test_constructor_rejects_invalid_block_dimensions(
    embedding_dim: int, num_heads: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DecoderStack(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=2,
        )


def test_stack_registers_requested_number_of_distinct_blocks() -> None:
    stack = DecoderStack(embedding_dim=6, num_heads=3, num_layers=3)

    assert isinstance(stack.blocks, nn.ModuleList)
    assert len(stack.blocks) == 3
    assert all(isinstance(block, TransformerBlock) for block in stack.blocks)
    assert len({id(block) for block in stack.blocks}) == 3
    assert stack.blocks[0].attention is not stack.blocks[1].attention


def test_forward_preserves_batch_sequence_and_embedding_dimensions() -> None:
    stack = DecoderStack(embedding_dim=6, num_heads=3, num_layers=3)

    output = stack(torch.randn(2, 4, 6))

    assert output.shape == torch.Size([2, 4, 6])


def test_forward_matches_explicit_sequential_block_execution() -> None:
    stack = DecoderStack(embedding_dim=4, num_heads=2, num_layers=3)
    x = torch.randn(2, 3, 4)

    expected = x
    for block in stack.blocks:
        expected = block(expected)

    assert torch.equal(stack(x), expected)


def test_zero_sublayers_in_every_block_leave_input_unchanged() -> None:
    stack = DecoderStack(embedding_dim=4, num_heads=2, num_layers=3)
    zero_sublayer_outputs(stack)
    x = torch.randn(2, 3, 4)

    output = stack(x)

    assert torch.equal(output, x)


def test_future_token_does_not_change_earlier_stack_outputs() -> None:
    stack = DecoderStack(embedding_dim=4, num_heads=2, num_layers=3)
    x1 = torch.randn(1, 4, 4)
    x2 = x1.clone()
    x2[:, 3, :] = torch.tensor([100.0, -100.0, 50.0, -50.0])

    output1 = stack(x1)
    output2 = stack(x2)

    assert torch.allclose(output1[:, :3], output2[:, :3])


def test_forward_preserves_input_dtype_and_device() -> None:
    stack = DecoderStack(embedding_dim=4, num_heads=2, num_layers=2).to(
        dtype=torch.float64
    )
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    output = stack(x)

    assert output.dtype == x.dtype
    assert output.device == x.device


def test_backward_reaches_input_and_every_block_parameter() -> None:
    stack = DecoderStack(embedding_dim=4, num_heads=2, num_layers=3)
    x = torch.randn(2, 3, 4, requires_grad=True)

    stack(x).square().sum().backward()

    assert x.grad is not None
    for block in stack.blocks:
        for parameter in block.parameters():
            assert parameter.grad is not None


def test_forward_rejects_non_batched_input() -> None:
    stack = DecoderStack(embedding_dim=4, num_heads=2, num_layers=2)

    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        stack(torch.randn(3, 4))


def test_forward_rejects_empty_sequence() -> None:
    stack = DecoderStack(embedding_dim=4, num_heads=2, num_layers=2)

    with pytest.raises(ValueError, match="sequence length"):
        stack(torch.randn(2, 0, 4))


def test_forward_rejects_wrong_embedding_dimension() -> None:
    stack = DecoderStack(embedding_dim=4, num_heads=2, num_layers=2)

    with pytest.raises(ValueError, match="embedding dimension"):
        stack(torch.randn(2, 3, 6))
