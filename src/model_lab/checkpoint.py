"""保存和恢复训练所需的最小 Checkpoint 状态。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
) -> None:
    """保存模型参数、优化器状态和已完成的 Epoch 编号。"""

    if epoch < 0:
        raise ValueError("epoch must not be negative")

    # state_dict 保存 Parameter（参数）和 Optimizer 的内部状态，而不是整个
    # Python 对象。epoch 表示已经完整完成的训练轮次，恢复后可从下一个轮次继续。
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    map_location: torch.device | str | None = None,
) -> int:
    """加载 Checkpoint 到已创建的模型和优化器，并返回已完成 Epoch。"""

    # weights_only=False 是当前项目保存的本地纯 Tensor/标量字典格式所需的
    # 明确加载模式；文件内容由调用者提供，不在这里隐藏加载异常。
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain model_state_dict")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain optimizer_state_dict")
    if "epoch" not in checkpoint:
        raise ValueError("checkpoint must contain epoch")
    if not isinstance(checkpoint["epoch"], int) or checkpoint["epoch"] < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
