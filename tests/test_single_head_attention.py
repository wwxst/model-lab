import math

import pytest
import torch
from torch import nn

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.position_embedding import PositionEmbedding
from model_lab.single_head_attention import SingleHeadSelfAttention
from model_lab.text_dataset import TextSequenceDataset
from model_lab.token_embedding import TokenEmbedding


def make_identity_attention(embedding_dim: int) -> SingleHeadSelfAttention:
    attention = SingleHeadSelfAttention(embedding_dim)
    identity = torch.eye(embedding_dim)
    with torch.no_grad():
        attention.query_projection.weight.copy_(identity)
        attention.key_projection.weight.copy_(identity)
        attention.value_projection.weight.copy_(identity)
    return attention


def test_projection_weights_are_square_and_bias_free() -> None:
    attention = SingleHeadSelfAttention(embedding_dim=4)

    assert attention.query_projection.weight.shape == torch.Size([4, 4])
    assert attention.key_projection.weight.shape == torch.Size([4, 4])
    assert attention.value_projection.weight.shape == torch.Size([4, 4])
    assert attention.query_projection.bias is None
    assert attention.key_projection.bias is None
    assert attention.value_projection.bias is None


def test_forward_returns_context_and_weights_with_expected_shapes() -> None:
    attention = SingleHeadSelfAttention(embedding_dim=4)
    context, weights = attention(torch.randn(2, 3, 4))

    assert context.shape == torch.Size([2, 3, 4])
    assert weights.shape == torch.Size([2, 3, 3])


def test_attention_rows_sum_to_one() -> None:
    _, weights = SingleHeadSelfAttention(embedding_dim=3)(torch.randn(2, 4, 3))

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4))


def test_causal_mask_zeroes_all_future_attention_weights() -> None:
    _, weights = SingleHeadSelfAttention(embedding_dim=3)(torch.randn(2, 4, 3))
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    assert torch.equal(weights[:, future], torch.zeros_like(weights[:, future]))


def test_first_token_can_only_attend_to_itself() -> None:
    _, weights = make_identity_attention(2)(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))

    assert torch.allclose(weights[:, 0, :], torch.tensor([[1.0, 0.0]]))


def test_later_token_can_attend_to_current_and_past_tokens_only() -> None:
    _, weights = make_identity_attention(2)(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]])
    )

    assert torch.all(weights[:, 2, 3:] == 0)
    assert torch.all(weights[:, 2, :3] > 0)


def test_future_token_does_not_change_earlier_context() -> None:
    attention = make_identity_attention(2)
    x1 = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    x2 = x1.clone()
    x2[:, 2, :] = torch.tensor([100.0, -100.0])

    context1, _ = attention(x1)
    context2, _ = attention(x2)

    assert torch.equal(context1[:, :2], context2[:, :2])


def test_past_token_can_change_later_context() -> None:
    attention = make_identity_attention(2)
    x1 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    x2 = x1.clone()
    x2[:, 0, :] = torch.tensor([5.0, 0.0])

    context1, _ = attention(x1)
    context2, _ = attention(x2)

    assert not torch.equal(context1[:, 1], context2[:, 1])


def test_identity_projection_manual_two_token_example() -> None:
    attention = make_identity_attention(2)
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    context, weights = attention(x)

    first_score = 1.0 / math.sqrt(2.0)
    second_weight = math.exp(first_score) / (1.0 + math.exp(first_score))
    first_weight = 1.0 - second_weight
    expected_weights = torch.tensor([[[1.0, 0.0], [first_weight, second_weight]]])
    expected_context = torch.tensor([[[1.0, 0.0], [first_weight, second_weight]]])

    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(context, expected_context)


def test_backward_computes_projection_gradients() -> None:
    attention = SingleHeadSelfAttention(embedding_dim=3)
    context, _ = attention(torch.randn(2, 4, 3))

    context.sum().backward()

    assert attention.query_projection.weight.grad is not None
    assert attention.key_projection.weight.grad is not None
    assert attention.value_projection.weight.grad is not None


def test_forward_rejects_non_batched_input() -> None:
    attention = SingleHeadSelfAttention(embedding_dim=3)

    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        attention(torch.randn(4, 3))


def test_forward_rejects_empty_sequence() -> None:
    attention = SingleHeadSelfAttention(embedding_dim=3)

    with pytest.raises(ValueError, match="sequence length"):
        attention(torch.randn(2, 0, 3))


def test_embedding_pipeline_connects_to_attention() -> None:
    tokenizer = CharacterTokenizer.from_text("hello")
    token_ids = tokenizer.encode("hello")
    dataset = TextSequenceDataset(token_ids, context_length=3)
    input_ids, _ = dataset[0]

    token_embedding = TokenEmbedding(tokenizer.vocab_size, embedding_dim=4)
    position_embedding = PositionEmbedding(max_sequence_length=3, embedding_dim=4)
    input_representation = token_embedding(input_ids) + position_embedding(3)

    attention = SingleHeadSelfAttention(embedding_dim=4)
    context, weights = attention(input_representation.unsqueeze(0))

    assert input_representation.shape == torch.Size([3, 4])
    assert context.shape == torch.Size([1, 3, 4])
    assert weights.shape == torch.Size([1, 3, 3])
    assert isinstance(attention, nn.Module)
