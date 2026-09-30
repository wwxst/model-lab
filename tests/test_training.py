import math

import pytest
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.cross_entropy import cross_entropy_loss
from model_lab.decoder_model import DecoderOnlyLanguageModel
from model_lab.text_dataset import TextSequenceDataset
from model_lab.training import train_epoch


class ConstantLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.0, 1.0, 2.0]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        return self.logits.expand(batch_size, sequence_length, -1)


def make_model() -> DecoderOnlyLanguageModel:
    return DecoderOnlyLanguageModel(
        vocab_size=3,
        max_sequence_length=2,
        embedding_dim=4,
        num_heads=2,
        num_layers=1,
    )


def test_train_epoch_updates_model_parameters() -> None:
    torch.manual_seed(0)
    model = make_model()
    optimizer = SGD(model.parameters(), lr=0.1)
    input_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    target_ids = torch.tensor([[1, 2], [2, 0]], dtype=torch.int64)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_epoch(model, [(input_ids, target_ids)], optimizer, device="cpu")

    assert math.isfinite(loss)
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters(), strict=True)
    )


def test_repeated_training_reduces_loss_on_tiny_repeated_batch() -> None:
    torch.manual_seed(0)
    model = make_model()
    optimizer = AdamW(model.parameters(), lr=0.05)
    input_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.int64)
    target_ids = torch.tensor([[1, 2], [1, 2], [1, 2], [1, 2]], dtype=torch.int64)
    initial_loss = cross_entropy_loss(model(input_ids), target_ids).item()

    for _ in range(20):
        train_epoch(model, [(input_ids, target_ids)], optimizer, device="cpu")

    final_loss = cross_entropy_loss(model(input_ids), target_ids).item()
    assert final_loss < initial_loss


def test_epoch_loss_is_weighted_by_number_of_target_tokens() -> None:
    model = ConstantLogitModel()
    optimizer = SGD(model.parameters(), lr=0.0)
    first_input = torch.zeros(1, 1, dtype=torch.int64)
    first_target = torch.tensor([[0]], dtype=torch.int64)
    second_input = torch.zeros(1, 3, dtype=torch.int64)
    second_target = torch.tensor([[2, 2, 2]], dtype=torch.int64)
    first_loss = cross_entropy_loss(model(first_input), first_target).item()
    second_loss = cross_entropy_loss(model(second_input), second_target).item()

    epoch_loss = train_epoch(
        model,
        [(first_input, first_target), (second_input, second_target)],
        optimizer,
        device="cpu",
    )

    expected = (first_loss * 1 + second_loss * 3) / 4
    assert epoch_loss == pytest.approx(expected)


def test_train_epoch_switches_model_to_training_mode() -> None:
    model = make_model()
    model.eval()
    optimizer = SGD(model.parameters(), lr=0.1)
    input_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    target_ids = torch.tensor([[1, 2]], dtype=torch.int64)

    train_epoch(model, [(input_ids, target_ids)], optimizer, device="cpu")

    assert model.training


def test_train_epoch_rejects_empty_batches() -> None:
    model = make_model()
    optimizer = SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="at least one target"):
        train_epoch(model, [], optimizer, device="cpu")


def test_tokenizer_dataset_dataloader_and_training_form_complete_pipeline() -> None:
    torch.manual_seed(0)
    tokenizer = CharacterTokenizer.from_text("hello world")
    token_ids = tokenizer.encode("hello world")
    dataset = TextSequenceDataset(token_ids, context_length=3)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = DecoderOnlyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=1,
    )
    optimizer = AdamW(model.parameters(), lr=0.01)

    loss = train_epoch(model, data_loader, optimizer, device="cpu")

    assert math.isfinite(loss)
    assert loss > 0


def test_epoch_loss_weights_only_supervised_tokens() -> None:
    model = ConstantLogitModel()
    optimizer = SGD(model.parameters(), lr=0.0)
    inputs = torch.zeros(1, 3, dtype=torch.int64)
    first_targets = torch.tensor([[-100, -100, 0]])
    second_targets = torch.tensor([[-100, 2, 2]])
    first_loss = cross_entropy_loss(model(inputs), first_targets).item()
    second_loss = cross_entropy_loss(model(inputs), second_targets).item()

    loss = train_epoch(
        model,
        [(inputs, first_targets), (inputs, second_targets)],
        optimizer,
        "cpu",
    )
    assert loss == pytest.approx((first_loss + 2 * second_loss) / 3)
