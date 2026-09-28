import pytest
import torch

from model_lab.residual_normalization import LayerNormalization, add_residual


@pytest.mark.parametrize(
    ("embedding_dim", "epsilon", "message"),
    [
        (0, 1e-5, "embedding_dim"),
        (4, 0.0, "epsilon"),
    ],
)
def test_layer_normalization_rejects_invalid_configuration(
    embedding_dim: int, epsilon: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LayerNormalization(embedding_dim=embedding_dim, epsilon=epsilon)


def test_scale_and_shift_have_expected_initial_values() -> None:
    normalization = LayerNormalization(embedding_dim=4)

    assert normalization.scale.shape == torch.Size([4])
    assert normalization.shift.shape == torch.Size([4])
    assert torch.equal(normalization.scale, torch.ones(4))
    assert torch.equal(normalization.shift, torch.zeros(4))


def test_layer_normalization_matches_manual_example() -> None:
    normalization = LayerNormalization(embedding_dim=4, epsilon=1e-5)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    output = normalization(x)

    mean = torch.tensor(2.5)
    centered = x - mean
    variance = centered.square().mean()
    expected = centered / torch.sqrt(variance + 1e-5)
    assert torch.allclose(output, expected)


def test_each_token_is_normalized_independently() -> None:
    normalization = LayerNormalization(embedding_dim=4, epsilon=1e-12).to(
        dtype=torch.float64
    )
    x = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [10.0, 14.0, 18.0, 22.0]]],
        dtype=torch.float64,
    )

    output = normalization(x)

    assert torch.allclose(output.mean(dim=-1), torch.zeros(1, 2, dtype=torch.float64))
    assert torch.allclose(
        output.square().mean(dim=-1),
        torch.ones(1, 2, dtype=torch.float64),
    )


def test_changing_one_token_does_not_change_other_normalized_tokens() -> None:
    normalization = LayerNormalization(embedding_dim=4)
    x1 = torch.randn(1, 3, 4)
    x2 = x1.clone()
    x2[:, 2, :] = torch.tensor([100.0, -100.0, 50.0, -50.0])

    output1 = normalization(x1)
    output2 = normalization(x2)

    assert torch.equal(output1[:, :2], output2[:, :2])


def test_learnable_scale_and_shift_adjust_normalized_features() -> None:
    normalization = LayerNormalization(embedding_dim=2, epsilon=1e-5)
    with torch.no_grad():
        normalization.scale.copy_(torch.tensor([2.0, 3.0]))
        normalization.shift.copy_(torch.tensor([0.5, -0.5]))
    x = torch.tensor([[[1.0, 3.0]]])

    output = normalization(x)

    normalized = torch.tensor([[[-1.0, 1.0]]]) / torch.sqrt(torch.tensor(1.0 + 1e-5))
    expected = normalized * torch.tensor([2.0, 3.0]) + torch.tensor([0.5, -0.5])
    assert torch.allclose(output, expected)


def test_layer_normalization_preserves_input_dtype_and_device() -> None:
    normalization = LayerNormalization(embedding_dim=4).to(dtype=torch.float64)
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    output = normalization(x)

    assert output.dtype == x.dtype
    assert output.device == x.device


def test_layer_normalization_backward_computes_input_and_parameter_gradients() -> None:
    normalization = LayerNormalization(embedding_dim=4)
    x = torch.randn(2, 3, 4, requires_grad=True)

    normalization(x).square().sum().backward()

    assert x.grad is not None
    assert normalization.scale.grad is not None
    assert normalization.shift.grad is not None


def test_layer_normalization_rejects_non_batched_input() -> None:
    normalization = LayerNormalization(embedding_dim=4)

    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        normalization(torch.randn(3, 4))


def test_layer_normalization_rejects_empty_sequence() -> None:
    normalization = LayerNormalization(embedding_dim=4)

    with pytest.raises(ValueError, match="sequence length"):
        normalization(torch.randn(2, 0, 4))


def test_layer_normalization_rejects_wrong_embedding_dimension() -> None:
    normalization = LayerNormalization(embedding_dim=4)

    with pytest.raises(ValueError, match="embedding dimension"):
        normalization(torch.randn(2, 3, 6))


def test_add_residual_performs_elementwise_addition() -> None:
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    sublayer_output = torch.tensor([[[0.5, -0.5], [2.0, 1.0]]])

    output = add_residual(x, sublayer_output)

    expected = torch.tensor([[[1.5, 1.5], [5.0, 5.0]]])
    assert torch.equal(output, expected)


def test_add_residual_backward_reaches_both_inputs() -> None:
    x = torch.randn(2, 3, 4, requires_grad=True)
    sublayer_output = torch.randn(2, 3, 4, requires_grad=True)

    add_residual(x, sublayer_output).sum().backward()

    assert x.grad is not None
    assert sublayer_output.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))
    assert torch.equal(sublayer_output.grad, torch.ones_like(sublayer_output))


def test_add_residual_rejects_non_batched_inputs() -> None:
    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        add_residual(torch.randn(3, 4), torch.randn(3, 4))


def test_add_residual_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        add_residual(torch.randn(2, 3, 4), torch.randn(2, 1, 4))
