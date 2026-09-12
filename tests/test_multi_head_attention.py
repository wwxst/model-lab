import math

import pytest
import torch
from torch import nn

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.multi_head_attention import MultiHeadSelfAttention
from model_lab.position_embedding import PositionEmbedding
from model_lab.text_dataset import TextSequenceDataset
from model_lab.token_embedding import TokenEmbedding


def make_identity_attention(
    embedding_dim: int = 4, num_heads: int = 2
) -> MultiHeadSelfAttention:
    attention = MultiHeadSelfAttention(embedding_dim, num_heads)
    identity = torch.eye(embedding_dim)
    with torch.no_grad():
        attention.query_projection.weight.copy_(identity)
        attention.key_projection.weight.copy_(identity)
        attention.value_projection.weight.copy_(identity)
        attention.output_projection.weight.copy_(identity)
    return attention


@pytest.mark.parametrize(
    ("embedding_dim", "num_heads", "message"),
    [
        (0, 1, "embedding_dim"),
        (4, 0, "num_heads"),
        (5, 2, "divisible"),
    ],
)
def test_constructor_rejects_invalid_dimensions(
    embedding_dim: int, num_heads: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MultiHeadSelfAttention(embedding_dim, num_heads)


def test_projection_weights_are_square_and_bias_free() -> None:
    attention = MultiHeadSelfAttention(embedding_dim=6, num_heads=3)

    for projection in (
        attention.query_projection,
        attention.key_projection,
        attention.value_projection,
        attention.output_projection,
    ):
        assert projection.weight.shape == torch.Size([6, 6])
        assert projection.bias is None


def test_forward_returns_context_and_weights_with_expected_shapes() -> None:
    attention = MultiHeadSelfAttention(embedding_dim=6, num_heads=3)

    context, weights = attention(torch.randn(2, 4, 6))

    assert context.shape == torch.Size([2, 4, 6])
    assert weights.shape == torch.Size([2, 3, 4, 4])


def test_each_head_attention_rows_sum_to_one() -> None:
    _, weights = MultiHeadSelfAttention(embedding_dim=6, num_heads=3)(
        torch.randn(2, 4, 6)
    )

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 4))


def test_causal_mask_zeroes_future_weights_for_every_head() -> None:
    _, weights = MultiHeadSelfAttention(embedding_dim=6, num_heads=3)(
        torch.randn(2, 4, 6)
    )
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    assert torch.equal(weights[:, :, future], torch.zeros_like(weights[:, :, future]))


def test_first_token_can_only_attend_to_itself_in_every_head() -> None:
    _, weights = make_identity_attention()(torch.randn(2, 3, 4))
    expected = torch.tensor([1.0, 0.0, 0.0]).expand(2, 2, 3)

    assert torch.equal(weights[:, :, 0, :], expected)


def test_future_token_does_not_change_earlier_context() -> None:
    attention = make_identity_attention()
    x1 = torch.tensor(
        [[[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 0.0], [1.0, 1.0, 0.0, 2.0]]]
    )
    x2 = x1.clone()
    x2[:, 2, :] = torch.tensor([100.0, -100.0, -50.0, 50.0])

    context1, _ = attention(x1)
    context2, _ = attention(x2)

    assert torch.equal(context1[:, :2], context2[:, :2])


def test_identity_projections_match_manual_two_head_example() -> None:
    attention = make_identity_attention()
    x = torch.tensor([[[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 0.0]]])

    context, weights = attention(x)

    head_0_current = math.exp(1.0 / math.sqrt(2.0))
    head_0_past_weight = 1.0 / (1.0 + head_0_current)
    head_0_current_weight = 1.0 - head_0_past_weight
    head_1_past = math.exp(2.0 / math.sqrt(2.0))
    head_1_current = math.exp(4.0 / math.sqrt(2.0))
    head_1_past_weight = head_1_past / (head_1_past + head_1_current)
    head_1_current_weight = 1.0 - head_1_past_weight
    expected_weights = torch.tensor(
        [
            [
                [[1.0, 0.0], [head_0_past_weight, head_0_current_weight]],
                [[1.0, 0.0], [head_1_past_weight, head_1_current_weight]],
            ]
        ]
    )
    expected_context = torch.tensor(
        [
            [
                [1.0, 0.0, 1.0, 1.0],
                [
                    head_0_past_weight,
                    head_0_current_weight,
                    head_1_past_weight + 2.0 * head_1_current_weight,
                    head_1_past_weight,
                ],
            ]
        ]
    )

    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(context, expected_context)


def test_different_heads_can_produce_different_attention_patterns() -> None:
    attention = make_identity_attention()
    x = torch.tensor([[[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 0.0]]])

    _, weights = attention(x)

    assert not torch.allclose(weights[0, 0, 1], weights[0, 1, 1])


def test_scale_uses_head_dim_instead_of_embedding_dim() -> None:
    attention = make_identity_attention()
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])

    _, weights = attention(x)

    expected_current_weight = math.exp(1.0 / math.sqrt(2.0)) / (
        1.0 + math.exp(1.0 / math.sqrt(2.0))
    )
    wrong_current_weight = math.exp(1.0 / math.sqrt(4.0)) / (
        1.0 + math.exp(1.0 / math.sqrt(4.0))
    )
    actual_current_weight = weights[0, 0, 1, 1]

    assert torch.allclose(actual_current_weight, torch.tensor(expected_current_weight))
    assert not torch.allclose(actual_current_weight, torch.tensor(wrong_current_weight))


def test_backward_computes_all_projection_gradients() -> None:
    attention = MultiHeadSelfAttention(embedding_dim=6, num_heads=3)
    context, _ = attention(torch.randn(2, 4, 6))

    context.square().sum().backward()

    assert attention.query_projection.weight.grad is not None
    assert attention.key_projection.weight.grad is not None
    assert attention.value_projection.weight.grad is not None
    assert attention.output_projection.weight.grad is not None


def test_forward_rejects_non_batched_input() -> None:
    attention = MultiHeadSelfAttention(embedding_dim=4, num_heads=2)

    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        attention(torch.randn(3, 4))


def test_forward_rejects_empty_sequence() -> None:
    attention = MultiHeadSelfAttention(embedding_dim=4, num_heads=2)

    with pytest.raises(ValueError, match="sequence length"):
        attention(torch.randn(2, 0, 4))


def test_forward_rejects_wrong_embedding_dimension() -> None:
    attention = MultiHeadSelfAttention(embedding_dim=4, num_heads=2)

    with pytest.raises(ValueError, match="embedding dimension"):
        attention(torch.randn(2, 3, 6))


def test_embedding_pipeline_connects_to_multi_head_attention() -> None:
    tokenizer = CharacterTokenizer.from_text("hello")
    token_ids = tokenizer.encode("hello")
    dataset = TextSequenceDataset(token_ids, context_length=3)
    input_ids, _ = dataset[0]

    token_embedding = TokenEmbedding(tokenizer.vocab_size, embedding_dim=4)
    position_embedding = PositionEmbedding(max_sequence_length=3, embedding_dim=4)
    input_representation = token_embedding(input_ids) + position_embedding(3)

    attention = MultiHeadSelfAttention(embedding_dim=4, num_heads=2)
    context, weights = attention(input_representation.unsqueeze(0))

    assert input_representation.shape == torch.Size([3, 4])
    assert context.shape == torch.Size([1, 3, 4])
    assert weights.shape == torch.Size([1, 2, 3, 3])
    assert isinstance(attention, nn.Module)
