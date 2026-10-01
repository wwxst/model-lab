"""下载有来源记录的百科文本，生成固定的基础知识训练与测试数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

FACTS_PATH = Path(__file__).parents[1] / "examples" / "foundation" / "facts.json"
API_URL = "https://zh.wikipedia.org/w/api.php"
TARGET_CHARACTERS = 100_000
INTRODUCTION = (
    "基础知识阅读材料\n\n"
    "阅读时注意名称、数量和单位，也要注意事实成立的条件。"
    "用户可以向助手询问什么、怎么、多少、哪一个等问题。"
    "阅读时也可以问自己：这是什么？条件是什么？"
    "对话中可以使用“用户：”和“助手：”表示说话者。\n\n"
)


def normalized_question(question: str) -> str:
    """避免只改空格或标点就把原题伪装成未见问法。"""
    return "".join(character for character in question if character.isalnum())


def clean_extract(text: str) -> list[str]:
    """保留正文段落；去掉参考尾部、章节标题、公式命令和空白噪声。"""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.split(
        r"(?m)^={2,}\s*(?:参见|參見|参考文献|參考文獻|参考资料|參考資料|"
        r"外部链接|外部連結|注释|註釋|注釋|延伸阅读|延伸閱讀)\s*={2,}\s*$",
        text,
        maxsplit=1,
    )[0]
    text = re.sub(r"(?m)^={2,}.*?={2,}\s*$", "", text)
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip() and "\\displaystyle" not in paragraph
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="准备约10万字符和200/50条基础问答")
    parser.add_argument("--proxy", help="本次下载使用的HTTP代理，不写入系统配置")
    parser.add_argument("--output-dir", type=Path, default=FACTS_PATH.parent)
    args = parser.parse_args()
    paths = {
        name: args.output_dir / name
        for name in (
            "pretraining.txt",
            "qa_train.jsonl",
            "qa_test.jsonl",
            "sources.json",
        )
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"数据已存在：{path}。请用 --output-dir 指定新目录。")

    facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    train = [
        {"question": question, "answer": fact["answer"], "fact_id": fact["id"]}
        for fact in facts
        for question in fact["train_questions"]
    ]
    test = [
        {
            "question": fact["test_question"],
            "answer": fact["answer"],
            "fact_id": fact["id"],
        }
        for fact in facts
    ]
    train_questions = {normalized_question(record["question"]) for record in train}
    test_questions = {normalized_question(record["question"]) for record in test}
    if train_questions & test_questions:
        raise ValueError("训练题和测试题在去掉标点、空格后仍有重复")

    opener = (
        build_opener(ProxyHandler({"https": args.proxy, "http": args.proxy}))
        if args.proxy
        else build_opener()
    )

    def query(parameters: dict[str, str]) -> dict:
        url = (
            API_URL
            + "?"
            + urlencode({"format": "json", "formatversion": "2", **parameters})
        )
        request = Request(
            url,
            headers={
                "User-Agent": "ModelLabDataset/0.1 (https://github.com/wwxst/model-lab)"
            },
        )
        with opener.open(request, timeout=30) as response:
            result = json.load(response)
        if "error" in result:
            raise ValueError(f"Wikipedia API error: {result['error']}")
        return result

    rights = query({"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"})[
        "query"
    ]["rightsinfo"]
    if "/by-sa/4.0/" not in rights["url"]:
        raise ValueError(f"来源许可需要重新确认：{rights}")

    pages: dict[int, dict] = {}
    titles = sorted({fact["source_title"] for fact in facts})
    for index, title in enumerate(titles, start=1):
        response = query(
            {
                "action": "query",
                "prop": "extracts|info|revisions",
                "explaintext": "1",
                "inprop": "url",
                "rvprop": "ids|timestamp",
                "redirects": "1",
                "variant": "zh-cn",
                "titles": title,
            }
        )
        page = response["query"]["pages"][0]
        if page.get("missing") or not page.get("extract"):
            raise ValueError(f"条目没有可用正文：{title}")
        page_id = page["pageid"]
        if page_id in pages:
            pages[page_id]["requested_titles"].append(title)
        else:
            revision = page["revisions"][0]
            paragraphs = clean_extract(page["extract"])
            excluded = []
            if page["title"] == "水":
                # 这段覆盖比例与地球条目的约71%冲突，不把冲突数字喂给模型。
                excluded = [
                    paragraph for paragraph in paragraphs if "75%~78%" in paragraph
                ]
                paragraphs = [
                    paragraph for paragraph in paragraphs if paragraph not in excluded
                ]
            pages[page_id] = {
                "title": page["title"],
                "page_id": page_id,
                "requested_titles": [title],
                "revision_id": revision["revid"],
                "revision_timestamp": revision["timestamp"],
                "url": page["fullurl"],
                "revision_url": f"https://zh.wikipedia.org/w/index.php?oldid={revision['revid']}",
                "history_url": f"https://zh.wikipedia.org/w/index.php?curid={page_id}&action=history",
                "extract_sha256": hashlib.sha256(
                    page["extract"].encode("utf-8")
                ).hexdigest(),
                "excluded_paragraphs": [
                    {
                        "sha256": hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
                        "reason": (
                            "Contains Earth water coverage 75%~78%, inconsistent "
                            "with the Earth article's ocean coverage of about 71%."
                        ),
                    }
                    for paragraph in excluded
                ],
                "paragraphs": paragraphs,
                "selected": [],
            }
        print(f"已读取 {index}/{len(titles)}：{title}", flush=True)

    notes = (
        INTRODUCTION + "知识说明\n\n" + "\n\n".join(fact["statement"] for fact in facts)
    )
    notes += "\n\n百科正文节选\n\n"
    # 轮流选择各条目的完整段落，避免长条目独占全部预算。
    # 所有段落只使用一次；到达目标后停止，不截断句子，也不重复凑长度。
    total_characters = (
        len(notes) + sum(len(page["title"]) + 2 for page in pages.values()) - 1
    )
    positions = dict.fromkeys(pages, 0)
    seen_paragraphs: set[str] = set()
    while total_characters < TARGET_CHARACTERS:
        added = False
        for page_id, page in pages.items():
            while positions[page_id] < len(page["paragraphs"]):
                paragraph = page["paragraphs"][positions[page_id]]
                positions[page_id] += 1
                if paragraph in seen_paragraphs:
                    continue
                seen_paragraphs.add(paragraph)
                page["selected"].append(paragraph)
                total_characters += len(paragraph) + 2
                added = True
                break
            if total_characters >= TARGET_CHARACTERS:
                break
        if not added:
            raise ValueError("可用正文不足10万字符；不能通过重复段落补足")

    corpus = (
        notes
        + "\n\n".join(
            page["title"] + "\n\n" + "\n\n".join(page["selected"])
            for page in pages.values()
        )
        + "\n"
    )
    for record in train + test:
        if record["question"] in corpus:
            raise ValueError(f"问答题混入预训练正文：{record['question']}")
    vocabulary = set(corpus)
    for record in train:
        unknown = (
            set(f"用户：{record['question']}\n助手：{record['answer']}\n\n")
            - vocabulary
        )
        if unknown:
            raise ValueError(f"训练题含预训练词表没有的字符：{sorted(unknown)}")

    sources = []
    for page in pages.values():
        excerpt = "\n\n".join(page["selected"])
        sources.append(
            {
                key: value
                for key, value in page.items()
                if key not in {"paragraphs", "selected"}
            }
            | {
                "selected_characters": len(excerpt),
                "selected_sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
            }
        )
    contents = {
        "pretraining.txt": corpus,
        "qa_train.jsonl": "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in train
        ),
        "qa_test.jsonl": "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in test
        ),
    }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "license": rights,
        "source_api": API_URL,
        "facts_sha256": hashlib.sha256(FACTS_PATH.read_bytes()).hexdigest(),
        "original_content": (
            "Model Lab AI-assisted teaching notes and questions, reviewed against "
            "the listed concepts; not copied QA pairs."
        ),
        "selection": (
            "NFC plaintext, remove line-end whitespace, omit trailing references, "
            "headings and paragraphs containing LaTeX displaystyle commands; "
            "unique whole paragraphs selected round-robin "
            "until about 100000 characters."
        ),
        "characters": len(corpus),
        "vocabulary_size": len(vocabulary),
        "knowledge_points": len(facts),
        "training_records": len(train),
        "test_records": len(test),
        "test_unknown_characters": sorted(
            set("".join(record["question"] + record["answer"] for record in test))
            - vocabulary
        ),
        "files_sha256": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in contents.items()
        },
        "articles": sources,
    }
    contents["sources.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        paths[name].write_text(text, encoding="utf-8", newline="\n")
    print(f"已准备：{len(corpus):,}字符，{len(train)}道训练题，{len(test)}道测试题")
    print(f"测试题未知字符：{manifest['test_unknown_characters']}")


if __name__ == "__main__":
    main()
