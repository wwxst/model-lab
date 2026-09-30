"""用显式步骤执行一个 Epoch 的 next-token 训练。"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from model_lab.cross_entropy import cross_entropy_loss


def train_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    device: torch.device | str,
) -> float:
    """训练模型一次遍历全部 batches，并返回按 Token 平均的 Loss。"""

    model.train()

    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids in batches:
        # 调用者在创建 Optimizer 前已经把模型放到目标 Device。DataLoader 默认
        # 在 CPU 产生 [B, T] int64 Tensor，所以当前 Batch 也要移动到同一 Device。
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 上一个 Batch 的梯度已经用于更新参数，当前 Batch 开始前必须清零。
        optimizer.zero_grad()

        # Forward Pass: [B, T] Token IDs -> [B, T, V] Logits -> scalar Loss。
        logits = model(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        # Backward 沿计算图为全部可学习参数计算梯度，step 再根据 Optimizer
        # 的更新规则修改参数。这两步共同完成一次 Parameter Update。
        loss.backward()
        optimizer.step()

        # 每个 Batch 的 loss 是该 Batch 内监督 Token 的平均值。乘回 Token 数
        # 后累加，最后再除以总 Token 数，避免较小的末尾 Batch 获得相同权重。
        # 问答样本中的 -100 是不参与学习的问题位置或补齐位置。
        token_count = (target_ids != -100).sum().item()
        total_loss += loss.detach().item() * token_count
        total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("batches must provide at least one target Token")

    return total_loss / total_tokens
