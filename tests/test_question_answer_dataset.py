import pytest
import torch

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.decoder_model import DecoderOnlyLanguageModel
from model_lab.question_answer_dataset import (
    QuestionAnswerDataset,
    collate_question_answers,
)


def test_each_answer_character_uses_only_its_known_prefix() -> None:
    prompt = "用户：问题\n助手："
    tokenizer = CharacterTokenizer.from_text(prompt + "答案\n\n")
    dataset = QuestionAnswerDataset([("问题", "答案")], tokenizer, 32)

    assert len(dataset) == 4
    for index, character in enumerate("答案\n\n"):
        inputs, targets = dataset[index]
        assert tokenizer.decode(inputs) == prompt + "答案\n\n"[:index]
        assert targets.dtype == inputs.dtype == torch.int64
        assert targets.device.type == inputs.device.type == "cpu"
        assert torch.all(targets[:-1] == -100)
        assert tokenizer.decode(targets[-1:]) == character


def test_records_do_not_share_context_and_long_prefixes_are_cropped() -> None:
    records = [("甲", "乙"), ("丙", "丁")]
    tokenizer = CharacterTokenizer.from_text("用户：甲丙\n助手：乙丁\n\n")
    dataset = QuestionAnswerDataset(records, tokenizer, 4)

    first_inputs, _ = dataset[0]
    second_inputs, _ = dataset[3]
    assert tokenizer.decode(first_inputs) == "\n助手："
    assert tokenizer.decode(second_inputs) == "\n助手："
    assert first_inputs.shape == second_inputs.shape == torch.Size([4])

    uncropped = QuestionAnswerDataset(records, tokenizer, 32)
    assert tokenizer.decode(uncropped[0][0]) == "用户：甲\n助手："
    assert tokenizer.decode(uncropped[3][0]) == "用户：丙\n助手："


def test_collation_pads_on_right_and_retains_one_supervised_target_per_sample() -> None:
    samples = [
        (torch.tensor([1, 2]), torch.tensor([-100, 3])),
        (torch.tensor([4, 5, 6]), torch.tensor([-100, -100, 7])),
    ]
    inputs, targets = collate_question_answers(samples)

    assert inputs.tolist() == [[1, 2, 0], [4, 5, 6]]
    assert targets.tolist() == [[-100, 3, -100], [-100, -100, 7]]
    assert inputs.shape == targets.shape == torch.Size([2, 3])


@pytest.mark.parametrize("records", [[], [("", "答案")], [("问题", " ")]])
def test_undefined_question_answer_records_are_rejected(
    records: list[tuple[str, str]],
) -> None:
    tokenizer = CharacterTokenizer.from_text("用户：问题\n助手：答案\n")
    with pytest.raises(ValueError):
        QuestionAnswerDataset(records, tokenizer, 32)


def test_right_padding_does_not_change_logits_at_supervised_position() -> None:
    torch.manual_seed(0)
    model = DecoderOnlyLanguageModel(8, 4, 8, 2, 1)
    samples = [
        (torch.tensor([1, 2]), torch.tensor([-100, 3])),
        (torch.tensor([4, 5, 6]), torch.tensor([-100, -100, 7])),
    ]
    padded_inputs, _ = collate_question_answers(samples)
    padded_logits = model(padded_inputs)
    single_logits = model(samples[0][0].unsqueeze(0))
    assert torch.allclose(padded_logits[0, 1], single_logits[0, 1], atol=1e-6)
