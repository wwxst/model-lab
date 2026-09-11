"""用最小滑动窗口准备语言模型的 input 和 target 训练样本。"""

import torch
from torch.utils.data import Dataset


class TextSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """从一维整数序列中生成 next-token prediction（下一位置预测）样本。"""

    def __init__(self, data: torch.Tensor, context_length: int) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        if data.ndim != 1:
            raise ValueError("data must be a one-dimensional Tensor")
        if data.dtype != torch.int64:
            raise TypeError("data must use torch.int64")
        if not isinstance(context_length, int):
            raise TypeError("context_length must be an integer")
        if context_length <= 0:
            raise ValueError("context_length must be greater than zero")
        if data.numel() <= context_length:
            raise ValueError("data must contain at least one input and target window")

        self.data = data
        self.context_length = context_length

    def __len__(self) -> int:
        # N 个离散元素中，每个样本需要 T 个 input 元素和紧随其后的 1 个 target 元素。
        return self.data.numel() - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 每次窗口向后移动一个位置，因此 input 和 target 长度都等于 T。
        start = index
        end = start + self.context_length

        # x 读取当前位置窗口，y 整体向右错一位，表示每个位置的下一个离散元素。
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]

        return x, y


def split_sequence(
    sequence: torch.Tensor,
    train_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按原始顺序切分一维整数序列，返回 train 和 validation 两段。"""

    if not isinstance(sequence, torch.Tensor):
        raise TypeError("sequence must be a torch.Tensor")
    if sequence.ndim != 1:
        raise ValueError("sequence must be a one-dimensional Tensor")
    if sequence.dtype != torch.int64:
        raise TypeError("sequence must use torch.int64")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")

    split_index = int(sequence.numel() * train_ratio)
    train_data = sequence[:split_index]
    validation_data = sequence[split_index:]

    return train_data, validation_data
