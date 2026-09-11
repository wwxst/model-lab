import pytest
import torch
from torch import nn

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.text_dataset import TextSequenceDataset
from model_lab.token_embedding import TokenEmbedding


def test_embedding_table_has_one_learnable_row_per_token() -> None:
    embedding = TokenEmbedding(vocab_size=4, embedding_dim=3)

    assert embedding.weight.shape == torch.Size([4, 3])
    assert isinstance(embedding.weight, nn.Parameter)
    assert embedding.weight.requires_grad


def test_single_sequence_adds_embedding_dimension() -> None:
    embedding = TokenEmbedding(vocab_size=4, embedding_dim=3)
    token_ids = torch.tensor([1, 0, 2, 2, 3], dtype=torch.int64)

    vectors = embedding(token_ids)

    assert vectors.shape == torch.Size([5, 3])
    assert vectors.dtype == embedding.weight.dtype
    assert vectors.device.type == "cpu"


def test_batch_preserves_batch_and_sequence_dimensions() -> None:
    embedding = TokenEmbedding(vocab_size=4, embedding_dim=3)
    token_ids = torch.tensor([[0, 1, 2], [2, 3, 0]], dtype=torch.int64)

    vectors = embedding(token_ids)

    assert vectors.shape == torch.Size([2, 3, 3])


def test_same_token_id_returns_same_vector() -> None:
    embedding = TokenEmbedding(vocab_size=3, embedding_dim=2)

    vectors = embedding(torch.tensor([1, 2, 1], dtype=torch.int64))

    assert torch.equal(vectors[0], vectors[2])


def test_token_ids_lookup_corresponding_embedding_rows() -> None:
    embedding = TokenEmbedding(vocab_size=3, embedding_dim=2)
    expected_weight = torch.tensor(
        [[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]],
        dtype=embedding.weight.dtype,
    )
    with torch.no_grad():
        embedding.weight.copy_(expected_weight)

    vectors = embedding(torch.tensor([2, 0, 1], dtype=torch.int64))

    assert torch.equal(vectors, expected_weight[[2, 0, 1]])


def test_gradient_accumulates_only_for_used_rows_and_repeated_tokens() -> None:
    embedding = TokenEmbedding(vocab_size=4, embedding_dim=3)
    token_ids = torch.tensor([0, 2, 2], dtype=torch.int64)

    loss = embedding(token_ids).sum()
    loss.backward()

    gradient = embedding.weight.grad
    assert gradient is not None
    assert torch.equal(gradient[0], torch.ones(3))
    assert torch.equal(gradient[1], torch.zeros(3))
    assert torch.equal(gradient[2], torch.full((3,), 2.0))
    assert torch.equal(gradient[3], torch.zeros(3))


def test_character_tokenizer_output_can_be_embedded() -> None:
    text = "你好，AI\n"
    tokenizer = CharacterTokenizer.from_text(text)
    embedding = TokenEmbedding(vocab_size=tokenizer.vocab_size, embedding_dim=4)

    vectors = embedding(tokenizer.encode(text))

    assert vectors.shape == torch.Size([len(text), 4])


def test_dataset_input_window_can_be_embedded() -> None:
    tokenizer = CharacterTokenizer.from_text("hello")
    dataset = TextSequenceDataset(tokenizer.encode("hello"), context_length=3)
    input_ids, _ = dataset[0]
    embedding = TokenEmbedding(vocab_size=tokenizer.vocab_size, embedding_dim=2)

    input_vectors = embedding(input_ids)

    assert input_ids.shape == torch.Size([3])
    assert input_vectors.shape == torch.Size([3, 2])


@pytest.mark.parametrize(
    ("vocab_size", "embedding_dim"),
    [(0, 3), (-1, 3), (4, 0), (4, -1)],
)
def test_embedding_dimensions_must_be_positive(
    vocab_size: int,
    embedding_dim: int,
) -> None:
    with pytest.raises(ValueError):
        TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
