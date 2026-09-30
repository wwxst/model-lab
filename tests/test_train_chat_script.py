import subprocess
import sys
from pathlib import Path

import torch


def test_train_chat_script_completes_training_and_saves_checkpoint(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "chat.txt"
    data_path.write_text("用户：你好\n助手：你好。\n" * 8, encoding="utf-8")
    checkpoint_path = tmp_path / "chat-model.pt"
    repository_root = Path(__file__).parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_chat.py",
            "--data",
            str(data_path),
            "--checkpoint",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--max-characters",
            "64",
            "--context-length",
            "8",
            "--batch-size",
            "8",
            "--embedding-dim",
            "8",
            "--num-heads",
            "2",
            "--num-layers",
            "1",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Epoch 1/1" in result.stdout
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["metadata"]["data"]["max_characters"] == 64
    assert checkpoint["metadata"]["model"]["context_length"] == 8

    blocked_overwrite = subprocess.run(
        [
            sys.executable,
            "scripts/train_chat.py",
            "--data",
            str(data_path),
            "--checkpoint",
            str(checkpoint_path),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert blocked_overwrite.returncode != 0
    assert "FileExistsError" in blocked_overwrite.stderr

    resumed = subprocess.run(
        [
            sys.executable,
            "scripts/train_chat.py",
            "--resume",
            str(checkpoint_path),
            "--epochs",
            "1",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Epoch 2/2" in resumed.stdout
    resumed_checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert resumed_checkpoint["epoch"] == 2

    original_text = data_path.read_text(encoding="utf-8")
    data_path.write_text("问" + original_text[1:], encoding="utf-8")
    blocked_changed_data = subprocess.run(
        [
            sys.executable,
            "scripts/train_chat.py",
            "--resume",
            str(checkpoint_path),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert blocked_changed_data.returncode != 0
    assert "ValueError" in blocked_changed_data.stderr


def test_train_chat_script_upgrades_old_checkpoint_on_resume(tmp_path: Path) -> None:
    data_path = tmp_path / "chat.txt"
    data_path.write_text("用户：你好\n助手：你好。\n" * 8, encoding="utf-8")
    checkpoint_path = tmp_path / "old-checkpoint.pt"
    repository_root = Path(__file__).parents[1]

    subprocess.run(
        [
            sys.executable,
            "scripts/train_chat.py",
            "--data",
            str(data_path),
            "--checkpoint",
            str(checkpoint_path),
            "--max-characters",
            "64",
            "--context-length",
            "8",
            "--batch-size",
            "8",
            "--embedding-dim",
            "8",
            "--num-heads",
            "2",
            "--num-layers",
            "1",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )
    old_checkpoint = torch.load(checkpoint_path, weights_only=False)
    del old_checkpoint["metadata"]
    torch.save(old_checkpoint, checkpoint_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/train_chat.py",
            "--resume",
            str(checkpoint_path),
            "--data",
            str(data_path),
            "--max-characters",
            "64",
            "--context-length",
            "8",
            "--batch-size",
            "8",
            "--embedding-dim",
            "8",
            "--num-heads",
            "2",
            "--num-layers",
            "1",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )

    upgraded_checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert upgraded_checkpoint["epoch"] == 2
    assert upgraded_checkpoint["metadata"]["format_version"] == 1
