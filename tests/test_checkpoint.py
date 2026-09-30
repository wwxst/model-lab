from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from model_lab.checkpoint import (
    load_checkpoint,
    load_checkpoint_metadata,
    load_model_weights,
    save_checkpoint,
)
from model_lab.decoder_model import DecoderOnlyLanguageModel
from model_lab.training import train_epoch


def make_model() -> DecoderOnlyLanguageModel:
    return DecoderOnlyLanguageModel(
        vocab_size=5,
        max_sequence_length=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=1,
    )


def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.int64),
        torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int64),
    )


def assert_optimizer_states_equal(
    expected: dict[str, object], actual: dict[str, object]
) -> None:
    assert expected["param_groups"] == actual["param_groups"]
    expected_states = expected["state"]
    actual_states = actual["state"]
    assert isinstance(expected_states, dict)
    assert isinstance(actual_states, dict)
    assert expected_states.keys() == actual_states.keys()
    for parameter_id in expected_states:
        expected_state = expected_states[parameter_id]
        actual_state = actual_states[parameter_id]
        assert isinstance(expected_state, dict)
        assert isinstance(actual_state, dict)
        assert expected_state.keys() == actual_state.keys()
        for name in expected_state:
            expected_value = expected_state[name]
            actual_value = actual_state[name]
            if isinstance(expected_value, torch.Tensor):
                assert isinstance(actual_value, torch.Tensor)
                assert torch.equal(expected_value, actual_value)
            else:
                assert expected_value == actual_value


def test_save_checkpoint_writes_model_optimizer_and_epoch(tmp_path: Path) -> None:
    model = make_model()
    optimizer = AdamW(model.parameters(), lr=0.01)
    batch = make_batch()
    train_epoch(model, [batch], optimizer, device="cpu")
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, epoch=3)

    assert path.exists()
    saved = torch.load(path, weights_only=False)
    assert set(saved) == {"model_state_dict", "optimizer_state_dict", "epoch"}
    assert saved["epoch"] == 3
    assert saved["optimizer_state_dict"]["state"]


def test_load_checkpoint_restores_parameters_optimizer_state_and_epoch(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    source_model = make_model()
    source_optimizer = AdamW(source_model.parameters(), lr=0.01)
    batch = make_batch()
    train_epoch(source_model, [batch], source_optimizer, device="cpu")
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, source_model, source_optimizer, epoch=4)

    restored_model = make_model()
    restored_optimizer = AdamW(restored_model.parameters(), lr=0.01)
    restored_epoch = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        map_location="cpu",
    )

    assert restored_epoch == 4
    for source, restored in zip(
        source_model.parameters(), restored_model.parameters(), strict=True
    ):
        assert torch.equal(source, restored)
    assert_optimizer_states_equal(
        source_optimizer.state_dict(), restored_optimizer.state_dict()
    )


def test_restored_training_step_matches_source_training_step(tmp_path: Path) -> None:
    torch.manual_seed(0)
    source_model = make_model()
    source_optimizer = AdamW(source_model.parameters(), lr=0.01)
    batch = make_batch()
    train_epoch(source_model, [batch], source_optimizer, device="cpu")
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, source_model, source_optimizer, epoch=1)

    restored_model = make_model()
    restored_optimizer = AdamW(restored_model.parameters(), lr=0.01)
    load_checkpoint(path, restored_model, restored_optimizer, map_location="cpu")

    source_loss = train_epoch(source_model, [batch], source_optimizer, device="cpu")
    restored_loss = train_epoch(
        restored_model,
        [batch],
        restored_optimizer,
        device="cpu",
    )

    assert restored_loss == pytest.approx(source_loss)
    for source, restored in zip(
        source_model.parameters(), restored_model.parameters(), strict=True
    ):
        assert torch.equal(source, restored)


def test_load_model_weights_restores_predictions_without_optimizer(
    tmp_path: Path,
) -> None:
    source = make_model()
    optimizer = AdamW(source.parameters(), lr=0.01)
    batch = make_batch()
    train_epoch(source, [batch], optimizer, device="cpu")
    path = tmp_path / "pretrained.pt"
    save_checkpoint(path, source, optimizer, epoch=7)

    restored = make_model()
    load_model_weights(path, restored, map_location="cpu")

    for name, value in source.state_dict().items():
        actual = restored.state_dict()[name]
        assert torch.equal(value, actual)
        assert actual.dtype == value.dtype
        assert actual.device == value.device
    assert torch.equal(source(batch[0]), restored(batch[0]))
    assert not AdamW(restored.parameters()).state


def test_save_checkpoint_rejects_negative_epoch(tmp_path: Path) -> None:
    model = make_model()
    optimizer = AdamW(model.parameters())

    with pytest.raises(ValueError, match="epoch"):
        save_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, -1)


def test_load_checkpoint_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save({"epoch": 1}, path)
    model = make_model()
    optimizer = AdamW(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(path, model, optimizer)


def test_checkpoint_round_trips_optional_metadata(tmp_path: Path) -> None:
    model = make_model()
    optimizer = AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "checkpoint.pt"
    metadata: dict[str, object] = {
        "tokenizer_characters": ["a", "b"],
        "model": {"embedding_dim": 4},
    }

    save_checkpoint(path, model, optimizer, epoch=2, metadata=metadata)

    assert load_checkpoint_metadata(path, map_location="cpu") == metadata


def test_old_checkpoint_without_metadata_returns_none(tmp_path: Path) -> None:
    model = make_model()
    optimizer = AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1)

    assert load_checkpoint_metadata(path) is None
