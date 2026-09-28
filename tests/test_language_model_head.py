import pytest
import torch
from torch import nn

from model_lab.decoder_stack import DecoderStack
from model_lab.language_model_head import LanguageModelHead


@pytest.mark.parametrize(
    ("embedding_dim", "vocab_size", "message"),
    [
        (0, 5, "embedding_dim"),
        (4, 0, "vocab_size"),
    ],
)
def test_constructor_rejects_invalid_dimensions(
    embedding_dim: int, vocab_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LanguageModelHead(embedding_dim=embedding_dim, vocab_size=vocab_size)


def test_projection_has_vocab_rows_and_no_bias() -> None:
    head = LanguageModelHead(embedding_dim=4, vocab_size=7)

    assert head.projection.weight.shape == torch.Size([7, 4])
    assert head.projection.bias is None
    assert isinstance(head, nn.Module)


def test_forward_replaces_embedding_dimension_with_vocabulary_dimension() -> None:
    head = LanguageModelHead(embedding_dim=4, vocab_size=7)

    logits = head(torch.randn(2, 3, 4))

    assert logits.shape == torch.Size([2, 3, 7])


def test_known_projection_produces_expected_raw_logits() -> None:
    head = LanguageModelHead(embedding_dim=2, vocab_size=3)
    with torch.no_grad():
        head.projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, -1.0],
                ]
            )
        )
    hidden_states = torch.tensor([[[2.0, 3.0], [-1.0, 4.0]]])

    logits = head(hidden_states)

    expected = torch.tensor([[[2.0, 3.0, -1.0], [-1.0, 4.0, -5.0]]])
    assert torch.equal(logits, expected)
    assert not torch.allclose(logits.sum(dim=-1), torch.ones(1, 2))


def test_changing_one_position_does_not_change_other_position_logits() -> None:
    head = LanguageModelHead(embedding_dim=4, vocab_size=7)
    hidden_states1 = torch.randn(1, 3, 4)
    hidden_states2 = hidden_states1.clone()
    hidden_states2[:, 2, :] = torch.tensor([100.0, -100.0, 50.0, -50.0])

    logits1 = head(hidden_states1)
    logits2 = head(hidden_states2)

    assert torch.equal(logits1[:, :2], logits2[:, :2])


def test_forward_preserves_input_dtype_and_device() -> None:
    head = LanguageModelHead(embedding_dim=4, vocab_size=7).to(dtype=torch.float64)
    hidden_states = torch.randn(2, 3, 4, dtype=torch.float64)

    logits = head(hidden_states)

    assert logits.dtype == hidden_states.dtype
    assert logits.device == hidden_states.device


def test_backward_reaches_hidden_states_and_projection_weight() -> None:
    head = LanguageModelHead(embedding_dim=4, vocab_size=7)
    hidden_states = torch.randn(2, 3, 4, requires_grad=True)

    head(hidden_states).square().sum().backward()

    assert hidden_states.grad is not None
    assert head.projection.weight.grad is not None


def test_forward_rejects_non_batched_hidden_states() -> None:
    head = LanguageModelHead(embedding_dim=4, vocab_size=7)

    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        head(torch.randn(3, 4))


def test_forward_rejects_empty_sequence() -> None:
    head = LanguageModelHead(embedding_dim=4, vocab_size=7)

    with pytest.raises(ValueError, match="sequence length"):
        head(torch.randn(2, 0, 4))


def test_forward_rejects_wrong_embedding_dimension() -> None:
    head = LanguageModelHead(embedding_dim=4, vocab_size=7)

    with pytest.raises(ValueError, match="embedding dimension"):
        head(torch.randn(2, 3, 6))


def test_decoder_stack_hidden_states_connect_to_language_model_head() -> None:
    stack = DecoderStack(embedding_dim=4, num_heads=2, num_layers=2)
    hidden_states = stack(torch.randn(2, 3, 4))

    head = LanguageModelHead(embedding_dim=4, vocab_size=7)
    logits = head(hidden_states)

    assert hidden_states.shape == torch.Size([2, 3, 4])
    assert logits.shape == torch.Size([2, 3, 7])
