import pytest
import torch

from model_lab.text_dataset import TextSequenceDataset, split_sequence


def test_dataset_length_is_number_of_next_token_windows() -> None:
    data = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)

    dataset = TextSequenceDataset(data, context_length=3)

    assert len(dataset) == 3


def test_first_sample_contains_shifted_input_and_target() -> None:
    data = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
    dataset = TextSequenceDataset(data, context_length=3)

    inputs, targets = dataset[0]

    assert torch.equal(inputs, torch.tensor([10, 11, 12], dtype=torch.int64))
    assert torch.equal(targets, torch.tensor([11, 12, 13], dtype=torch.int64))


def test_later_samples_slide_one_position_at_a_time() -> None:
    data = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
    dataset = TextSequenceDataset(data, context_length=3)

    second_input, second_target = dataset[1]
    third_input, third_target = dataset[2]

    assert torch.equal(second_input, torch.tensor([11, 12, 13], dtype=torch.int64))
    assert torch.equal(second_target, torch.tensor([12, 13, 14], dtype=torch.int64))
    assert torch.equal(third_input, torch.tensor([12, 13, 14], dtype=torch.int64))
    assert torch.equal(third_target, torch.tensor([13, 14, 15], dtype=torch.int64))


def test_sample_shapes_and_dtypes_match_context_length() -> None:
    data = torch.arange(8, dtype=torch.int64)
    dataset = TextSequenceDataset(data, context_length=4)

    inputs, targets = dataset[0]

    assert inputs.shape == torch.Size([4])
    assert targets.shape == torch.Size([4])
    assert inputs.dtype == torch.int64
    assert targets.dtype == torch.int64


def test_sequence_split_preserves_order_and_does_not_overlap() -> None:
    data = torch.arange(10, dtype=torch.int64)

    train_data, validation_data = split_sequence(data, train_ratio=0.8)

    assert torch.equal(train_data, torch.arange(8, dtype=torch.int64))
    assert torch.equal(validation_data, torch.arange(8, 10, dtype=torch.int64))
    assert train_data[-1] != validation_data[0]


@pytest.mark.parametrize(
    ("data", "context_length", "error"),
    [
        (torch.arange(6, dtype=torch.int64).reshape(2, 3), 3, ValueError),
        (torch.arange(6, dtype=torch.float32), 3, TypeError),
        (torch.arange(6, dtype=torch.int64), 0, ValueError),
        (torch.arange(3, dtype=torch.int64), 3, ValueError),
    ],
)
def test_dataset_rejects_undefined_input_boundaries(
    data: torch.Tensor,
    context_length: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        TextSequenceDataset(data, context_length)


@pytest.mark.parametrize("train_ratio", [0.0, 1.0, -0.1, 1.1])
def test_sequence_split_requires_ratio_between_zero_and_one(
    train_ratio: float,
) -> None:
    with pytest.raises(ValueError):
        split_sequence(torch.arange(10, dtype=torch.int64), train_ratio)
