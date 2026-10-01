"""保留基础数据，用每个知识点的六条新增问法生成500条训练问答。"""

import argparse
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).parents[1] / "examples" / "foundation"
EXTRA_PATH = DATA / "extra_train_questions.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    output = args.output_dir / "qa_train_expanded.jsonl"
    manifest_path = args.output_dir / "qa_expansion.json"
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(f"数据已存在：{path}。请指定新的 --output-dir。")

    facts = json.loads((DATA / "facts.json").read_text(encoding="utf-8"))
    extra = json.loads(EXTRA_PATH.read_text(encoding="utf-8"))
    if len(facts) != 50 or set(extra) != {fact["id"] for fact in facts}:
        raise ValueError("新增问法必须覆盖原有的50个知识点")
    training = [
        json.loads(line)
        for line in (DATA / "qa_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(training) != 200:
        raise ValueError("基础训练文件必须包含200条原题")
    records = list(training)
    for fact in facts:
        questions = extra[fact["id"]]
        if len(questions) != 6:
            raise ValueError(f"每个知识点必须新增6种问法：{fact['id']}")
        records.extend(
            {"question": question, "answer": fact["answer"], "fact_id": fact["id"]}
            for question in questions
        )

    # 去掉标点和空格后也不能重复；测试题只用于隔离检查，不写入训练文件。
    def normalize(question: str) -> str:
        return "".join(character for character in question if character.isalnum())

    testing = [
        json.loads(line)
        for line in (DATA / "qa_test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test_questions = {normalize(record["question"]) for record in testing}
    seen: set[str] = set()
    corpus = (DATA / "pretraining.txt").read_text(encoding="utf-8")
    normalized_corpus = normalize(corpus)
    vocabulary = set(corpus)
    for record in records:
        question = record["question"]
        normalized = normalize(question)
        if not normalized or question != question.strip() or "\n" in question:
            raise ValueError(f"问法必须是非空的单行文本，且无首尾空白：{question!r}")
        if normalized in seen:
            raise ValueError(f"训练问法重复：{question}")
        if normalized in test_questions:
            raise ValueError(f"训练问法与固定测试题重复：{question}")
        if normalized in normalized_corpus:
            raise ValueError(f"训练问法已出现在预训练文本中：{question}")
        sequence = f"用户：{question}\n助手：{record['answer']}\n\n"
        unknown = sorted(set(sequence) - vocabulary)
        if unknown:
            raise ValueError(f"问答含固定预训练词表没有的字符：{question} {unknown}")
        seen.add(normalized)

    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    inputs = [
        DATA / name
        for name in (
            "facts.json",
            "qa_train.jsonl",
            "qa_test.jsonl",
            "pretraining.txt",
            "sources.json",
        )
    ] + [EXTRA_PATH]
    manifest = {
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/deed.zh",
        "provenance": (
            "Model Lab AI-assisted original question paraphrases for all 50 "
            "existing facts; answers and source attribution remain in facts.json "
            "and sources.json. No model training or test-answer evaluation."
        ),
        "knowledge_points": 50,
        "base_training_records": len(training),
        "additional_training_records": len(records) - len(training),
        "training_records": len(records),
        "questions_per_fact": 10,
        "vocabulary_source": "pretraining.txt",
        "vocabulary_size": len(vocabulary),
        "inputs_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
        },
        "files_sha256": {output.name: hashlib.sha256(text.encode("utf-8")).hexdigest()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"已准备：{len(records)}道训练题，原题200道，新增问法300道。")
    print("原数据与固定词表未修改，没有训练或评估模型。")
    print(f"训练文件：{output}")


if __name__ == "__main__":
    main()
