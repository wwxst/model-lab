from typing import Any

import pytest
import torch
from torch import nn

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.position_embedding import PositionEmbedding
from model_lab.text_dataset import TextSequenceDataset
from model_lab.token_embedding import TokenEmbedding


def test_position_table_has_one_learnable_row_per_position() -> None:
    position_embedding = PositionEmbedding(
        max_sequence_length=8,
        embedding_dim=4,
    )

    assert position_embedding.weight.shape == torch.Size([8, 4])
    assert isinstance(position_embedding.weight, nn.Parameter)
    assert position_embedding.weight.requires_grad


def test_sequence_length_produces_position_vectors_with_matching_shape() -> None:
    position_embedding = PositionEmbedding(
        max_sequence_length=8,
        embedding_dim=4,
    )

    vectors = position_embedding(sequence_length=5)

    assert vectors.shape == torch.Size([5, 4])
    assert vectors.dtype == position_embedding.weight.dtype
    assert vectors.device.type == "cpu"


def test_position_ids_lookup_rows_in_sequence_order() -> None:
    position_embedding = PositionEmbedding(max_sequence_length=3, embedding_dim=2)
    expected_weight = torch.tensor(
        [[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]],
        dtype=position_embedding.weight.dtype,
    )
    with torch.no_grad():
        position_embedding.weight.copy_(expected_weight)

    vectors = position_embedding(sequence_length=3)

    assert torch.equal(vectors, expected_weight)


@pytest.mark.parametrize("sequence_length", [0, -1, 4])
def test_sequence_length_must_fit_position_table(sequence_length: int) -> None:
    position_embedding = PositionEmbedding(max_sequence_length=3, embedding_dim=2)

    with pytest.raises(ValueError):
        position_embedding(sequence_length=sequence_length)


@pytest.mark.parametrize("sequence_length", [1.5, True])
def test_sequence_length_rejects_non_integer_values(sequence_length: Any) -> None:
    position_embedding = PositionEmbedding(max_sequence_length=3, embedding_dim=2)

    with pytest.raises(TypeError):
        position_embedding(sequence_length=sequence_length)


@pytest.mark.parametrize(
    ("max_sequence_length", "embedding_dim"),
    [(0, 3), (-1, 3), (4, 0), (4, -1)],
)
def test_position_embedding_dimensions_must_be_positive(
    max_sequence_length: int,
    embedding_dim: int,
) -> None:
    with pytest.raises(ValueError):
        PositionEmbedding(max_sequence_length, embedding_dim)


def test_gradient_accumulates_only_for_used_position_rows() -> None:
    position_embedding = PositionEmbedding(max_sequence_length=5, embedding_dim=3)

    loss = position_embedding(sequence_length=3).sum()
    loss.backward()

    gradient = position_embedding.weight.grad
    assert gradient is not None
    assert torch.equal(gradient[0], torch.ones(3))
    assert torch.equal(gradient[1], torch.ones(3))
    assert torch.equal(gradient[2], torch.ones(3))
    assert torch.equal(gradient[3], torch.zeros(3))
    assert torch.equal(gradient[4], torch.zeros(3))


def test_token_and_position_vectors_are_added_for_one_sequence() -> None:
    token_embeddings = torch.ones((3, 2))
    position_embedding = PositionEmbedding(max_sequence_length=3, embedding_dim=2)
    with torch.no_grad():
        position_embedding.weight.copy_(
            torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        )

    input_representation = token_embeddings + position_embedding(3)

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert torch.equal(input_representation, expected)


def test_position_vectors_broadcast_across_batch_dimension() -> None:
    token_embeddings = torch.zeros((2, 3, 2))
    position_embedding = PositionEmbedding(max_sequence_length=3, embedding_dim=2)
    with torch.no_grad():
        position_embedding.weight.copy_(
            torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        )

    position_vectors = position_embedding(3).unsqueeze(0)
    input_representation = token_embeddings + position_vectors

    assert position_vectors.shape == torch.Size([1, 3, 2])
    assert input_representation.shape == torch.Size([2, 3, 2])
    assert torch.equal(input_representation[0], input_representation[1])


def test_same_token_at_different_positions_gets_different_representation() -> None:
    token_embedding = TokenEmbedding(vocab_size=2, embedding_dim=2)
    position_embedding = PositionEmbedding(max_sequence_length=2, embedding_dim=2)
    with torch.no_grad():
        token_embedding.weight.copy_(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
        position_embedding.weight.copy_(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))

    token_vectors = token_embedding(torch.tensor([1, 1], dtype=torch.int64))
    input_representation = token_vectors + position_embedding(2)

    assert torch.equal(token_vectors[0], token_vectors[1])
    assert not torch.equal(input_representation[0], input_representation[1])


def test_same_position_vector_is_shared_across_batch_items() -> None:
    position_embedding = PositionEmbedding(max_sequence_length=2, embedding_dim=2)
    with torch.no_grad():
        position_embedding.weight.copy_(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))

    position_vectors = position_embedding(2).unsqueeze(0).expand(2, -1, -1)

    assert torch.equal(position_vectors[0, 1], position_vectors[1, 1])


def test_tokenizer_dataset_and_position_embedding_connect() -> None:
    tokenizer = CharacterTokenizer.from_text("hello")
    token_ids = tokenizer.encode("hello")
    dataset = TextSequenceDataset(token_ids, context_length=3)
    input_ids, _ = dataset[0]
    token_embedding = TokenEmbedding(tokenizer.vocab_size, embedding_dim=4)
    position_embedding = PositionEmbedding(max_sequence_length=3, embedding_dim=4)

    input_representation = token_embedding(input_ids) + position_embedding(3)

    assert input_ids.shape == torch.Size([3])
    assert input_representation.shape == torch.Size([3, 4])
