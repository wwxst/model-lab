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
