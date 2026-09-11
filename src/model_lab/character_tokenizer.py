"""字符级分词器：在字符串与离散 Token ID 之间建立可逆映射。"""

from collections.abc import Sequence

import torch

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


class CharacterTokenizer:
    """把每个 Unicode 字符映射为一个确定的整数 Token ID。"""

    def __init__(self, characters: Sequence[str]) -> None:
        if not characters:
            raise ValueError("字符词表不能为空")

        vocabulary = tuple(characters)
        if any(not isinstance(character, str) for character in vocabulary):
            raise TypeError("词表中的每个元素都必须是字符串")
        if any(len(character) != 1 for character in vocabulary):
            raise ValueError("字符词表中的每个元素必须是一个 Unicode 字符")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("字符词表不能包含重复字符")

        # 两个方向的直接映射让 encode 和 decode 的教学过程保持清晰。
        self.char_to_id = {
            character: token_id for token_id, character in enumerate(vocabulary)
        }
        self.id_to_char = {
            token_id: character for token_id, character in enumerate(vocabulary)
        }

    @classmethod
    def from_text(cls, text: str) -> CharacterTokenizer:
        """从训练文本中按 Unicode 码点顺序建立唯一字符词表。"""

        if not isinstance(text, str):
            raise TypeError("训练文本必须是字符串")
        if not text:
            raise ValueError("训练文本不能为空")

        # set 去重后再排序，避免集合遍历顺序导致同一文本产生不同 ID。
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """返回词表中唯一字符的数量。"""

        return len(self.char_to_id)

    def encode(self, text: str) -> torch.Tensor:
        """将字符串转换为 CPU 上形状为 [T] 的 int64 Token ID Tensor。"""

        if not isinstance(text, str):
            raise TypeError("待编码文本必须是字符串")

        token_ids: list[int] = []
        for character in text:
            if character not in self.char_to_id:
                raise ValueError(f"文本包含未知字符 {character!r}")
            token_ids.append(self.char_to_id[character])

        # T 是当前文本中的字符数；空字符串自然得到形状 [0] 的 Tensor。
        return torch.tensor(token_ids, dtype=torch.int64, device="cpu")

    def decode(self, token_ids: torch.Tensor | Sequence[int]) -> str:
        """将一维整数 Token ID 序列还原为字符串。"""

        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim != 1:
                raise ValueError("Token ID 序列必须是一维 Tensor")
            if token_ids.dtype not in _INTEGER_DTYPES:
                raise TypeError("Token ID Tensor 必须使用整数 dtype")
            values = token_ids.tolist()
        elif isinstance(token_ids, Sequence) and not isinstance(
            token_ids, (str, bytes)
        ):
            values = list(token_ids)
        else:
            raise TypeError("Token IDs 必须是一维整数序列")

        characters: list[str] = []
        for token_id in values:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token ID 必须是整数")
            if token_id not in self.id_to_char:
                raise ValueError(f"非法 Token ID {token_id!r}")
            characters.append(self.id_to_char[token_id])

        return "".join(characters)
