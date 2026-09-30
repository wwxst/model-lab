"""按独立问答构造与自回归提问一致的回答预测样本。"""

from collections.abc import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from model_lab.character_tokenizer import CharacterTokenizer


class QuestionAnswerDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """每个样本预测一个回答字符，问题和此前回答只作为上下文。"""

    def __init__(
        self,
        records: Sequence[tuple[str, str]],
        tokenizer: CharacterTokenizer,
        context_length: int,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be greater than zero")
        if not records:
            raise ValueError("question-answer records must not be empty")

        self.context_length = context_length
        self.sequences: list[torch.Tensor] = []
        self.positions: list[tuple[int, int]] = []
        for record_index, (question, answer) in enumerate(records):
            if not question.strip() or not answer.strip():
                raise ValueError("question and answer must not be empty")
            prompt = f"用户：{question}\n助手："
            # 两个换行标记回答结束；训练和生成都能观察这个普通文本边界。
            sequence = tokenizer.encode(prompt + answer + "\n\n")
            self.sequences.append(sequence)
            for position in range(len(prompt), len(sequence)):
                self.positions.append((record_index, position))

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index, position = self.positions[index]
        sequence = self.sequences[record_index]
        # 输入 [T] 只取这条问答的已知前缀，绝不包含即将预测的字符或其他问答。
        # 长回答只保留最后 context_length 个字符，与 generate_greedy 一致。
        start = max(0, position - self.context_length)
        inputs = sequence[start:position]
        # -100 表示不计入 Loss。只在前缀最后位置监督“下一个回答字符”，
        # 问题、角色标签以及前面的输入位置均不承担本样本的预测误差。
        targets = torch.full_like(inputs, -100)
        targets[-1] = sequence[position]
        return inputs, targets


def collate_question_answers(
    samples: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """在右侧补齐不同长度的前缀，生成 [B, T] 的输入和监督标签。"""

    inputs, targets = zip(*samples, strict=True)
    # B = 批次大小，T = 本批最长前缀。右侧补齐位置不会被因果 Attention
    # 读到，也不参与 Loss，因此无需增加词表字符或修改模型 Attention。
    return (
        pad_sequence(list(inputs), batch_first=True, padding_value=0),
        pad_sequence(list(targets), batch_first=True, padding_value=-100),
    )
