import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import pytest

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.question_answer_dataset import QuestionAnswerDataset

ROOT = Path(__file__).parents[1]
DATA = ROOT / "examples" / "foundation"
SCRIPT = ROOT / "scripts" / "prepare_expanded_qa.py"


def read_records(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA / name).read_text(encoding="utf-8").splitlines()
    ]


def normalize(question: str) -> str:
    return "".join(character for character in question if character.isalnum())


def test_expansion_keeps_original_records_and_adds_six_questions_per_fact() -> None:
    base = read_records("qa_train.jsonl")
    expanded = read_records("qa_train_expanded.jsonl")
    facts = json.loads((DATA / "facts.json").read_text(encoding="utf-8"))
    extra = json.loads(
        (DATA / "extra_train_questions.json").read_text(encoding="utf-8")
    )
    assert len(expanded) == 500
    assert expanded[:200] == base
    assert Counter(row["fact_id"] for row in expanded) == {
        fact["id"]: 10 for fact in facts
    }
    assert set(extra) == {fact["id"] for fact in facts}
    for fact in facts:
        assert len(extra[fact["id"]]) == 6
        rows = [row for row in expanded if row["fact_id"] == fact["id"]]
        assert [row["question"] for row in rows] == (
            fact["train_questions"] + extra[fact["id"]]
        )
        assert all(row["answer"] == fact["answer"] for row in rows)
        assert all(row["question"] == row["question"].strip() for row in rows)
        assert all("\n" not in row["question"] for row in rows)


def test_expanded_questions_are_unique_and_isolated_from_test_and_corpus() -> None:
    expanded = read_records("qa_train_expanded.jsonl")
    questions = {normalize(row["question"]) for row in expanded}
    testing = {normalize(row["question"]) for row in read_records("qa_test.jsonl")}
    corpus = normalize((DATA / "pretraining.txt").read_text(encoding="utf-8"))
    assert len(questions) == 500
    assert "" not in questions
    assert questions.isdisjoint(testing)
    assert all(question not in corpus for question in questions)


def test_expanded_records_fit_fixed_vocabulary_and_existing_qa_consumer() -> None:
    tokenizer = CharacterTokenizer.from_text(
        (DATA / "pretraining.txt").read_text(encoding="utf-8")
    )
    records = read_records("qa_train_expanded.jsonl")
    pairs = [(row["question"], row["answer"]) for row in records]
    dataset = QuestionAnswerDataset(pairs, tokenizer, context_length=64)
    assert len(dataset.sequences) == 500
    assert tokenizer.vocab_size == 2472
    assert len(dataset) == sum(len(answer) + 2 for _, answer in pairs)
    for row, sequence in zip(records, dataset.sequences, strict=True):
        text = f"用户：{row['question']}\n助手：{row['answer']}\n\n"
        assert tokenizer.decode(sequence) == text
        assert len(sequence) <= 64


def test_expansion_manifest_matches_all_inputs_and_preserves_base_hashes() -> None:
    manifest = json.loads((DATA / "qa_expansion.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    assert manifest["training_records"] == 500
    assert manifest["base_training_records"] == 200
    assert manifest["additional_training_records"] == 300
    assert manifest["knowledge_points"] == 50
    assert manifest["questions_per_fact"] == 10
    assert manifest["vocabulary_source"] == "pretraining.txt"
    assert manifest["vocabulary_size"] == 2472
    assert manifest["license_url"] == base_manifest["license"]["url"]
    for field in ("inputs_sha256", "files_sha256"):
        for name, digest in manifest[field].items():
            assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest
    for name, digest in base_manifest["files_sha256"].items():
        assert manifest["inputs_sha256"][name] == digest
    assert manifest["inputs_sha256"]["facts.json"] == base_manifest["facts_sha256"]


def test_offline_preparation_reproduces_snapshot_without_changing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runpy.run_path(str(SCRIPT))
    before = {path.name: path.read_bytes() for path in DATA.iterdir() if path.is_file()}
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--output-dir", str(tmp_path)])
    module["main"]()
    for name in ("qa_train_expanded.jsonl", "qa_expansion.json"):
        assert (tmp_path / name).read_bytes() == (DATA / name).read_bytes()
    assert before == {
        path.name: path.read_bytes() for path in DATA.iterdir() if path.is_file()
    }


@pytest.mark.parametrize("name", ["qa_train_expanded.jsonl", "qa_expansion.json"])
def test_offline_preparation_refuses_to_overwrite_either_output(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / name
    existing.write_text("do not overwrite", encoding="utf-8")
    module = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--output-dir", str(tmp_path)])
    with pytest.raises(FileExistsError, match="数据已存在"):
        module["main"]()
    assert existing.read_text(encoding="utf-8") == "do not overwrite"
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_fact", "覆盖原有的50个知识点"),
        ("wrong_count", "新增6种问法"),
        ("duplicate", "训练问法重复"),
        ("test_overlap", "与固定测试题重复"),
        ("corpus_overlap", "已出现在预训练文本"),
        ("unknown_character", "固定预训练词表没有的字符"),
        ("blank", "非空的单行文本"),
    ],
)
def test_invalid_extra_questions_are_rejected_before_writing(
    case: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = json.loads(
        (DATA / "extra_train_questions.json").read_text(encoding="utf-8")
    )
    key = "water_formula"
    if case == "missing_fact":
        del extra[key]
    elif case == "wrong_count":
        extra[key].pop()
    elif case == "duplicate":
        extra[key][0] = " 水的化学式是什么？ ".strip().replace("？", "。")
    elif case == "test_overlap":
        extra[key][0] = read_records("qa_test.jsonl")[0]["question"].replace("？", "。")
    elif case == "corpus_overlap":
        extra[key][0] = "基础知识阅读材料"
    elif case == "unknown_character":
        extra[key][0] += "\U0001f9ea"
    else:
        extra[key][0] = " "
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")
    module = runpy.run_path(str(SCRIPT))
    monkeypatch.setitem(module["main"].__globals__, "EXTRA_PATH", invalid_path)
    output_dir = tmp_path / "output"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--output-dir", str(output_dir)])
    with pytest.raises(ValueError, match=message):
        module["main"]()
    assert not output_dir.exists()
