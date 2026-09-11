import torch

from model_lab.tensor_fundamentals import (
    broadcast_features,
    change_batch_dimension,
    compute_basic_gradient,
    create_basic_tensors,
    elementwise_operations,
    inspect_sequence,
    matrix_multiply,
    reorder_dimensions,
    reshape_sequence,
    select_tensor_regions,
)


def test_basic_tensors_have_expected_shapes_and_dimensions() -> None:
    scalar, vector, matrix, sequence = create_basic_tensors()

    assert scalar.shape == torch.Size([])
    assert scalar.ndim == 0
    assert vector.shape == torch.Size([4])
    assert vector.ndim == 1
    assert matrix.shape == torch.Size([3, 4])
    assert matrix.ndim == 2
    assert sequence.shape == torch.Size([2, 3, 4])
    assert sequence.ndim == 3


def test_basic_tensors_use_expected_dtypes_and_cpu_device() -> None:
    scalar, token_ids, matrix, sequence = create_basic_tensors()

    assert scalar.dtype == torch.float32
    assert token_ids.dtype == torch.int64
    assert matrix.dtype == torch.float32
    assert sequence.dtype == torch.float32
    assert all(
        tensor.device.type == "cpu" for tensor in (scalar, token_ids, matrix, sequence)
    )


def test_sequence_inspection_uses_shape_ndim_and_size() -> None:
    sequence = torch.zeros((2, 3, 4), dtype=torch.float32)

    shape, dimensions, sequence_length = inspect_sequence(sequence)

    assert shape == torch.Size([2, 3, 4])
    assert dimensions == 3
    assert sequence_length == 3


def test_sequence_indexing_selects_batch_token_and_feature_axes() -> None:
    sequence = torch.arange(24).reshape(2, 3, 4)

    first_batch, first_token, first_feature, _ = select_tensor_regions(sequence)

    assert first_batch.shape == torch.Size([3, 4])
    assert torch.equal(first_batch, sequence[0])
    assert first_token.shape == torch.Size([2, 4])
    assert torch.equal(first_token, sequence[:, 0])
    assert first_feature.shape == torch.Size([2, 3])
    assert torch.equal(first_feature, sequence[:, :, 0])


def test_sequence_slicing_preserves_batch_and_feature_axes() -> None:
    sequence = torch.arange(24).reshape(2, 3, 4)

    _, _, _, sequence_prefix = select_tensor_regions(sequence)

    assert sequence_prefix.shape == torch.Size([2, 2, 4])
    assert torch.equal(sequence_prefix, sequence[:, :2, :])


def test_reshape_and_view_flatten_batch_and_sequence_without_changing_values() -> None:
    sequence = torch.arange(24).reshape(2, 3, 4)

    reshaped, viewed = reshape_sequence(sequence)

    assert reshaped.shape == torch.Size([6, 4])
    assert viewed.shape == torch.Size([6, 4])
    assert torch.equal(reshaped, viewed)
    assert torch.equal(reshaped[3], sequence[1, 0])


def test_transpose_and_permute_reorder_dimensions_and_preserve_values() -> None:
    sequence = torch.arange(24).reshape(2, 3, 4)
    grouped_features = torch.arange(48).reshape(2, 3, 2, 4)

    transposed, permuted = reorder_dimensions(sequence, grouped_features)

    assert transposed.shape == torch.Size([2, 4, 3])
    assert transposed[1, 3, 2] == sequence[1, 2, 3]
    assert permuted.shape == torch.Size([2, 2, 3, 4])
    assert permuted[1, 0, 2, 3] == grouped_features[1, 2, 0, 3]


def test_unsqueeze_and_squeeze_add_then_remove_batch_dimension() -> None:
    sequence = torch.arange(12).reshape(3, 4)

    batched, restored = change_batch_dimension(sequence)

    assert batched.shape == torch.Size([1, 3, 4])
    assert restored.shape == torch.Size([3, 4])
    assert torch.equal(restored, sequence)


def test_broadcasting_adds_one_feature_vector_to_every_position() -> None:
    sequence = torch.zeros((2, 3, 4), dtype=torch.float32)
    features = torch.tensor([1.0, 2.0, 3.0, 4.0])

    result = broadcast_features(sequence, features)

    assert result.shape == torch.Size([2, 3, 4])
    assert torch.equal(result[0, 0], features)
    assert torch.equal(result[1, 2], features)


def test_elementwise_operations_compute_each_position_independently() -> None:
    left = torch.tensor([8.0, 6.0])
    right = torch.tensor([2.0, 3.0])

    added, subtracted, multiplied, divided = elementwise_operations(left, right)

    assert torch.equal(added, torch.tensor([10.0, 9.0]))
    assert torch.equal(subtracted, torch.tensor([6.0, 3.0]))
    assert torch.equal(multiplied, torch.tensor([16.0, 18.0]))
    assert torch.equal(divided, torch.tensor([4.0, 2.0]))


def test_matrix_multiplication_has_expected_shape_and_values() -> None:
    sequence = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    projection = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    operator_result, matmul_result = matrix_multiply(sequence, projection)

    expected = torch.tensor([[4.0, 5.0], [10.0, 11.0]])
    assert operator_result.shape == torch.Size([2, 2])
    assert torch.equal(operator_result, expected)
    assert torch.equal(matmul_result, expected)


def test_matrix_multiplication_preserves_batch_dimension() -> None:
    sequence = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    projection = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    operator_result, matmul_result = matrix_multiply(sequence, projection)

    assert operator_result.shape == torch.Size([2, 2, 2])
    assert torch.equal(operator_result[0], torch.tensor([[4.0, 5.0], [10.0, 11.0]]))
    assert torch.equal(operator_result, matmul_result)


def test_basic_autograd_computes_gradient_with_chain_rule() -> None:
    gradient = compute_basic_gradient()

    assert gradient.dtype == torch.float32
    assert torch.equal(gradient, torch.tensor([2.0, 4.0, 6.0]))
