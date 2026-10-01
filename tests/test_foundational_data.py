import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import pytest

from model_lab.character_tokenizer import CharacterTokenizer

ROOT = Path(__file__).parents[1]
DATA = ROOT / "examples" / "foundation"


def read_records(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA / name).read_text(encoding="utf-8").splitlines()
    ]


def normalize_question(question: str) -> str:
    return "".join(character for character in question if character.isalnum())


def test_fixed_knowledge_points_generate_200_training_and_50_test_questions() -> None:
    facts = json.loads((DATA / "facts.json").read_text(encoding="utf-8"))
    train, test = read_records("qa_train.jsonl"), read_records("qa_test.jsonl")
    ids = {fact["id"] for fact in facts}
    assert len(facts) == len(ids) == 50
    assert len(train) == 200
    assert len(test) == 50
    assert {fact["topic"] for fact in facts} == {"science", "geography", "arithmetic"}
    assert Counter(record["fact_id"] for record in train) == dict.fromkeys(ids, 4)
    assert Counter(record["fact_id"] for record in test) == dict.fromkeys(ids, 1)

    # 题目与答案只有 facts.json 一份手工定义，生成文件不能独立修改后漂移。
    for fact in facts:
        assert fact["answer"].rstrip("。") in fact["statement"]
        assert [
            record["question"] for record in train if record["fact_id"] == fact["id"]
        ] == fact["train_questions"]
        assert [
            record["question"] for record in test if record["fact_id"] == fact["id"]
        ] == [fact["test_question"]]
        for record in train + test:
            if record["fact_id"] == fact["id"]:
                assert record["answer"] == fact["answer"]
                assert record["question"].strip() == record["question"]
                assert record["question"] and record["answer"]


def test_training_and_test_questions_are_disjoint_without_punctuation() -> None:
    train, test = read_records("qa_train.jsonl"), read_records("qa_test.jsonl")
    training_questions = {normalize_question(record["question"]) for record in train}
    test_questions = {normalize_question(record["question"]) for record in test}
    assert len(training_questions) == len(train)
    assert len(test_questions) == len(test)
    assert training_questions.isdisjoint(test_questions)
    corpus = normalize_question((DATA / "pretraining.txt").read_text(encoding="utf-8"))
    for question in training_questions | test_questions:
        assert question not in corpus


def test_corpus_covers_qa_vocabulary_without_using_test_to_build_it() -> None:
    corpus = (DATA / "pretraining.txt").read_text(encoding="utf-8")
    assert 100_000 <= len(corpus) <= 110_000
    assert all(line.rstrip() == line for line in corpus.splitlines())
    assert "\\displaystyle" not in corpus
    tokenizer = CharacterTokenizer.from_text(corpus)
    for record in read_records("qa_train.jsonl") + read_records("qa_test.jsonl"):
        sequence = f"用户：{record['question']}\n助手：{record['answer']}\n\n"
        assert tokenizer.decode(tokenizer.encode(sequence)) == sequence


def test_provenance_license_and_snapshot_hashes_match_committed_files() -> None:
    manifest = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    facts = json.loads((DATA / "facts.json").read_text(encoding="utf-8"))
    assert (
        manifest["license"]["url"]
        == "https://creativecommons.org/licenses/by-sa/4.0/deed.zh"
    )
    assert (
        manifest["facts_sha256"]
        == hashlib.sha256((DATA / "facts.json").read_bytes()).hexdigest()
    )
    assert manifest["characters"] == len(
        (DATA / "pretraining.txt").read_text(encoding="utf-8")
    )
    assert manifest["knowledge_points"] == 50
    assert manifest["training_records"] == 200
    assert manifest["test_records"] == 50
    assert manifest["test_unknown_characters"] == []
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest
    articles = manifest["articles"]
    assert len({article["page_id"] for article in articles}) == len(articles)
    assert {fact["source_title"] for fact in facts} == {
        title for article in articles for title in article["requested_titles"]
    }
    for article in articles:
        assert article["revision_id"] > 0
        assert article["revision_url"].endswith(f"oldid={article['revision_id']}")
        assert article["history_url"].startswith("https://zh.wikipedia.org/")
        assert article["selected_characters"] > 0
        assert len(article["selected_sha256"]) == 64


def test_preparation_refuses_to_overwrite_frozen_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "prepare_foundational_data.py"))
    monkeypatch.setattr(
        "sys.argv", ["prepare_foundational_data.py", "--output-dir", str(DATA)]
    )
    with pytest.raises(FileExistsError, match="数据已存在"):
        module["main"]()


def test_extract_cleaning_removes_reference_tail_and_headings() -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "prepare_foundational_data.py"))
    text = "正文第一段。\n\n== 原理 ==\n\n正文第二段。\n\n== 参考文献 ==\n\n来源列表。"
    assert module["clean_extract"](text) == ["正文第一段。", "正文第二段。"]


def test_extract_cleaning_drops_formula_commands_and_line_end_whitespace() -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "prepare_foundational_data.py"))
    text = "正文。  \n  \n{\\displaystyle a^2}\n  \n另一段。  "
    assert module["clean_extract"](text) == ["正文。", "另一段。"]
