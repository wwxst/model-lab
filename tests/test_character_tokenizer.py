from typing import Any

import pytest
import torch

from model_lab.character_tokenizer import CharacterTokenizer


def test_vocabulary_is_sorted_and_contains_each_character_once() -> None:
    tokenizer = CharacterTokenizer.from_text("hello")

    assert tokenizer.char_to_id == {"e": 0, "h": 1, "l": 2, "o": 3}
    assert tokenizer.id_to_char == {0: "e", 1: "h", 2: "l", 3: "o"}
    assert tokenizer.vocab_size == 4


def test_encode_returns_cpu_int64_token_ids() -> None:
    tokenizer = CharacterTokenizer.from_text("hello")

    token_ids = tokenizer.encode("hello")

    assert torch.equal(token_ids, torch.tensor([1, 0, 2, 2, 3], dtype=torch.int64))
    assert token_ids.shape == torch.Size([5])
    assert token_ids.dtype == torch.int64
    assert token_ids.device.type == "cpu"


def test_decode_returns_original_text() -> None:
    tokenizer = CharacterTokenizer.from_text("hello")

    assert tokenizer.decode(torch.tensor([1, 0, 2, 2, 3], dtype=torch.int64)) == "hello"


def test_encode_and_decode_round_trip_unicode_text() -> None:
    text = "你好，AI\n"
    tokenizer = CharacterTokenizer.from_text(text)

    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size == len(set(text))


def test_encode_unknown_character_fails_with_character_in_message() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="未知字符.*d"):
        tokenizer.encode("abd")


@pytest.mark.parametrize("token_ids", [torch.tensor([0, 3]), [0, 3]])
def test_decode_rejects_invalid_token_id(token_ids: Any) -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="非法 Token ID.*3"):
        tokenizer.decode(token_ids)


def test_empty_vocabulary_source_fails() -> None:
    with pytest.raises(ValueError, match="训练文本不能为空"):
        CharacterTokenizer.from_text("")


def test_empty_encode_and_decode_are_supported() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    token_ids = tokenizer.encode("")

    assert token_ids.shape == torch.Size([0])
    assert token_ids.dtype == torch.int64
    assert tokenizer.decode(token_ids) == ""


@pytest.mark.parametrize(
    ("token_ids", "error"),
    [
        (torch.tensor([[0]], dtype=torch.int64), ValueError),
        (torch.tensor([0.0]), TypeError),
        ("0", TypeError),
    ],
)
def test_decode_requires_one_dimensional_integer_sequence(
    token_ids: Any,
    error: type[Exception],
) -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    with pytest.raises(error):
        tokenizer.decode(token_ids)


def test_tokenizer_output_connects_to_text_sequence_dataset() -> None:
    from model_lab.text_dataset import TextSequenceDataset

    tokenizer = CharacterTokenizer.from_text("hello")
    dataset = TextSequenceDataset(tokenizer.encode("hello"), context_length=3)

    inputs, targets = dataset[0]

    assert torch.equal(inputs, torch.tensor([1, 0, 2], dtype=torch.int64))
    assert torch.equal(targets, torch.tensor([0, 2, 2], dtype=torch.int64))
