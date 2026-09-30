import json
from pathlib import Path


def test_paraphrase_evaluation_questions_are_held_out_and_encodable() -> None:
    examples = Path(__file__).parents[1] / "examples"
    train = [
        json.loads(line)
        for line in (examples / "qa_paraphrases_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    test = [
        json.loads(line)
        for line in (examples / "qa_paraphrases_test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    training_questions = {record["question"] for record in train}
    testing_questions = {record["question"] for record in test}
    assert training_questions.isdisjoint(testing_questions)
    assert len(training_questions) == len(train)
    assert len(testing_questions) == len(test)

    # 检验已学知识的换一种表达，不让测试因新字符或新答案变成另一种任务。
    training_answers = {record["answer"] for record in train}
    training_characters = set(
        "".join(record["question"] + record["answer"] for record in train)
    )
    for record in test:
        assert record["answer"] in training_answers
        assert set(record["question"]).issubset(training_characters)
