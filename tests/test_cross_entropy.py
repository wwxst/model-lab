import pytest
import torch
import torch.nn.functional as functional

from model_lab.cross_entropy import cross_entropy_loss
from model_lab.decoder_model import DecoderOnlyLanguageModel


def test_loss_matches_pytorch_cross_entropy_reference() -> None:
    logits = torch.tensor(
        [
            [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
            [[1.0, 3.0, 2.0], [2.0, 0.0, 1.0]],
        ]
    )
    target_ids = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)

    loss = cross_entropy_loss(logits, target_ids)
    expected = functional.cross_entropy(logits.reshape(-1, 3), target_ids.reshape(-1))

    assert torch.allclose(loss, expected)


def test_loss_is_scalar_average_over_batch_and_sequence() -> None:
    logits = torch.zeros(2, 3, 4)
    target_ids = torch.zeros(2, 3, dtype=torch.int64)

    loss = cross_entropy_loss(logits, target_ids)

    assert loss.shape == torch.Size([])
    assert torch.allclose(loss, torch.log(torch.tensor(4.0)))


def test_higher_correct_token_logit_produces_lower_loss() -> None:
    target_ids = torch.tensor([[1]], dtype=torch.int64)
    weak_logits = torch.tensor([[[0.0, 0.0, 0.0]]])
    strong_logits = torch.tensor([[[0.0, 5.0, 0.0]]])

    weak_loss = cross_entropy_loss(weak_logits, target_ids)
    strong_loss = cross_entropy_loss(strong_logits, target_ids)

    assert strong_loss < weak_loss


def test_large_logits_remain_numerically_finite() -> None:
    logits = torch.tensor([[[1000.0, 1001.0, 999.0]]])
    target_ids = torch.tensor([[1]], dtype=torch.int64)

    loss = cross_entropy_loss(logits, target_ids)

    assert torch.isfinite(loss)


def test_gradient_matches_softmax_minus_one_hot_average() -> None:
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    target_ids = torch.tensor([[0, 2]], dtype=torch.int64)

    cross_entropy_loss(logits, target_ids).backward()

    expected = torch.tensor(
        [[[-1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 6.0, -1.0 / 3.0]]]
    )
    assert logits.grad is not None
    assert torch.allclose(logits.grad, expected)


def test_loss_preserves_logits_dtype_and_device() -> None:
    logits = torch.randn(2, 3, 4, dtype=torch.float64)
    target_ids = torch.zeros(2, 3, dtype=torch.int64)

    loss = cross_entropy_loss(logits, target_ids)

    assert loss.dtype == logits.dtype
    assert loss.device == logits.device


def test_complete_model_logits_connect_to_loss_and_backward() -> None:
    model = DecoderOnlyLanguageModel(
        vocab_size=7,
        max_sequence_length=4,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
    )
    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int64)
    target_ids = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 6]], dtype=torch.int64)

    logits = model(input_ids)
    loss = cross_entropy_loss(logits, target_ids)
    loss.backward()

    assert loss.shape == torch.Size([])
    for parameter in model.parameters():
        assert parameter.grad is not None


def test_loss_rejects_non_batched_logits() -> None:
    with pytest.raises(ValueError, match=r"\[B, T, V\]"):
        cross_entropy_loss(torch.randn(2, 3), torch.zeros(2, dtype=torch.int64))


def test_loss_rejects_non_batched_targets() -> None:
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        cross_entropy_loss(torch.randn(1, 2, 3), torch.zeros(2, dtype=torch.int64))


def test_loss_rejects_non_int64_targets() -> None:
    with pytest.raises(TypeError, match="int64"):
        cross_entropy_loss(torch.randn(1, 2, 3), torch.zeros(1, 2))


def test_loss_rejects_mismatched_batch_or_sequence_dimensions() -> None:
    with pytest.raises(ValueError, match="matching B and T"):
        cross_entropy_loss(
            torch.randn(2, 3, 4),
            torch.zeros(2, 2, dtype=torch.int64),
        )


def test_loss_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch size"):
        cross_entropy_loss(
            torch.empty(0, 3, 4),
            torch.empty(0, 3, dtype=torch.int64),
        )


def test_loss_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="sequence length"):
        cross_entropy_loss(
            torch.empty(2, 0, 4),
            torch.empty(2, 0, dtype=torch.int64),
        )


def test_loss_rejects_empty_vocabulary() -> None:
    with pytest.raises(ValueError, match="vocabulary size"):
        cross_entropy_loss(
            torch.empty(2, 3, 0),
            torch.zeros(2, 3, dtype=torch.int64),
        )


def test_ignored_targets_match_reference_and_have_zero_logit_gradient() -> None:
    logits = torch.tensor(
        [[[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[-100, 2, -100]])
    loss = cross_entropy_loss(logits, targets)
    expected = functional.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))

    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[:, [0, 2]], torch.zeros(1, 2, 3))
    assert not torch.equal(logits.grad[:, 1], torch.zeros(1, 3))


def test_loss_rejects_batch_without_supervised_targets() -> None:
    with pytest.raises(ValueError, match="supervised Token"):
        cross_entropy_loss(torch.zeros(1, 2, 3), torch.full((1, 2), -100))
