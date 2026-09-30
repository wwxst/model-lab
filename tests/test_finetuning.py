import hashlib
import json
import runpy
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.checkpoint import save_checkpoint
from model_lab.decoder_model import DecoderOnlyLanguageModel
from model_lab.training import train_epoch

SCRIPT = Path(__file__).parents[1] / "scripts" / "train_chat.py"


@pytest.fixture
def pretrained(tmp_path: Path) -> Path:
    text = "用户：你好\n助手：你好。\n" * 4
    data = tmp_path / "original.txt"
    data.write_text(text, encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    model = DecoderOnlyLanguageModel(tokenizer.vocab_size, 16, 8, 2, 1)
    optimizer = AdamW(model.parameters(), lr=0.01)
    tokens = tokenizer.encode(text[:9])
    train_epoch(model, [(tokens[:-1][None], tokens[1:][None])], optimizer, "cpu")
    path = tmp_path / "pretrained.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=7,
        metadata={
            "format_version": 1,
            "model": {
                "context_length": 16,
                "embedding_dim": 8,
                "num_heads": 2,
                "num_layers": 1,
            },
            "tokenizer_characters": list(tokenizer.char_to_id),
            "data": {
                "path": str(data),
                "format": "text",
                "max_characters": 0,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
        },
    )
    return path


def run_training(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])
    runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.mark.parametrize("data_format", ["text", "qa"])
def test_finetuning_inherits_weights_and_can_resume_new_data(
    pretrained: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_format: str
) -> None:
    original_bytes = pretrained.read_bytes()
    source = torch.load(pretrained, weights_only=False)
    data = tmp_path / "new-data.txt"
    text = (
        json.dumps({"question": "你好", "answer": "你好。"}, ensure_ascii=False) + "\n"
        if data_format == "qa"
        else "助手：你好。\n用户：你好\n" * 3
    )
    data.write_text(text, encoding="utf-8")
    output = tmp_path / "finetuned.pt"
    inspected = False

    def inspect_initialization(
        model: nn.Module,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        optimizer: Optimizer,
        device: torch.device | str,
    ) -> float:
        nonlocal inspected
        inspected = True
        for name, value in model.state_dict().items():
            assert torch.equal(value.cpu(), source["model_state_dict"][name])
        assert not optimizer.state
        assert optimizer.param_groups[0]["lr"] == 0.007
        return train_epoch(model, batches, optimizer, device)

    with monkeypatch.context() as training_patch:
        training_patch.setattr("model_lab.training.train_epoch", inspect_initialization)
        run_training(
            training_patch,
            [
                "--finetune",
                str(pretrained),
                "--data",
                str(data),
                "--data-format",
                data_format,
                "--max-characters",
                "0",
                "--checkpoint",
                str(output),
                "--batch-size",
                "8",
                "--learning-rate",
                "0.007",
            ],
        )

    saved = torch.load(output, weights_only=False)
    assert inspected
    assert saved["epoch"] == 1
    assert saved["metadata"]["model"] == source["metadata"]["model"]
    assert (
        saved["metadata"]["tokenizer_characters"]
        == source["metadata"]["tokenizer_characters"]
    )
    assert saved["metadata"]["data"]["format"] == data_format
    assert (
        saved["metadata"]["data"]["text_sha256"]
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    assert saved["optimizer_state_dict"]["state"]
    assert any(
        not torch.equal(value, source["model_state_dict"][name])
        for name, value in saved["model_state_dict"].items()
    )
    assert pretrained.read_bytes() == original_bytes

    run_training(monkeypatch, ["--resume", str(output)])
    assert torch.load(output, weights_only=False)["epoch"] == 2
    assert pretrained.read_bytes() == original_bytes


@pytest.mark.parametrize("missing", ["data", "checkpoint"])
def test_finetuning_requires_explicit_new_data_and_output(
    pretrained: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    arguments = ["--finetune", str(pretrained)]
    if missing != "data":
        arguments += ["--data", str(tmp_path / "new.txt")]
    if missing != "checkpoint":
        arguments += ["--checkpoint", str(tmp_path / "new.pt")]
    with pytest.raises(ValueError, match="--data.*--checkpoint"):
        run_training(monkeypatch, arguments)


def test_resume_and_finetune_are_mutually_exclusive(
    pretrained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as error:
        run_training(
            monkeypatch, ["--resume", str(pretrained), "--finetune", str(pretrained)]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("destination", ["source", "existing"])
def test_finetuning_does_not_overwrite_existing_files(
    pretrained: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    output = pretrained if destination == "source" else tmp_path / "existing.pt"
    if destination == "existing":
        output.write_bytes(b"existing experiment")
    original_bytes = output.read_bytes()
    with pytest.raises(FileExistsError):
        run_training(
            monkeypatch,
            [
                "--finetune",
                str(pretrained),
                "--data",
                str(tmp_path / "new.txt"),
                "--checkpoint",
                str(output),
            ],
        )
    assert output.read_bytes() == original_bytes


def test_finetuning_rejects_checkpoint_without_vocabulary(
    pretrained: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = torch.load(pretrained, weights_only=False)
    del checkpoint["metadata"]
    torch.save(checkpoint, pretrained)
    with pytest.raises(ValueError, match="词表和模型配置"):
        run_training(
            monkeypatch,
            [
                "--finetune",
                str(pretrained),
                "--data",
                str(tmp_path / "new.txt"),
                "--checkpoint",
                str(tmp_path / "new.pt"),
            ],
        )


@pytest.mark.parametrize("data_format", ["text", "qa"])
def test_finetuning_rejects_unknown_characters_before_saving(
    pretrained: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_format: str
) -> None:
    data = tmp_path / "new.txt"
    data.write_text(
        json.dumps({"question": "你好", "answer": "未知"}, ensure_ascii=False)
        if data_format == "qa"
        else "未知字符",
        encoding="utf-8",
    )
    output = tmp_path / "new.pt"
    original_bytes = pretrained.read_bytes()
    with pytest.raises(ValueError, match="词表没有的字符"):
        run_training(
            monkeypatch,
            [
                "--finetune",
                str(pretrained),
                "--data",
                str(data),
                "--data-format",
                data_format,
                "--checkpoint",
                str(output),
            ],
        )
    assert not output.exists()
    assert pretrained.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "option", ["context-length", "embedding-dim", "num-heads", "num-layers"]
)
def test_finetuning_rejects_model_structure_changes(
    pretrained: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    with pytest.raises(ValueError, match=option):
        run_training(
            monkeypatch,
            [
                "--finetune",
                str(pretrained),
                "--data",
                str(tmp_path / "new.txt"),
                "--checkpoint",
                str(tmp_path / "new.pt"),
                f"--{option}",
                "99",
            ],
        )
