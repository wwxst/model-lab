"""用本地对话文本启动一次最小的 Decoder-only 模型训练。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.checkpoint import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from model_lab.decoder_model import DecoderOnlyLanguageModel
from model_lab.question_answer_dataset import (
    QuestionAnswerDataset,
    collate_question_answers,
)
from model_lab.text_dataset import TextSequenceDataset
from model_lab.training import train_epoch

DEFAULT_DATA_PATH = Path("data/processed/alpaca_zh_chat_1000.txt")
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/chat-model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个最小中文对话语言模型")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="UTF-8 训练文本路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint 输出路径",
    )
    parser.add_argument("--resume", type=Path, help="从已有 Checkpoint 继续训练")
    parser.add_argument(
        "--data-format",
        choices=["text", "qa"],
        help="text 为连续文本；qa 为每行一条 question/answer 的 JSONL",
    )
    parser.add_argument("--epochs", type=int, default=1, help="本次新增训练轮数")
    parser.add_argument("--max-characters", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero")

    output_path = args.checkpoint or args.resume or DEFAULT_CHECKPOINT_PATH
    if args.resume is None and output_path.exists():
        raise FileExistsError(
            f"Checkpoint 已存在：{output_path}。请使用 --resume 继续训练，"
            "或用 --checkpoint 指定新文件。"
        )
    if args.resume is not None and output_path != args.resume and output_path.exists():
        raise FileExistsError(f"Checkpoint 输出文件已存在：{output_path}")

    saved_metadata = None
    if args.resume is not None:
        saved_metadata = load_checkpoint_metadata(args.resume, map_location="cpu")

    if saved_metadata is None:
        data_format = args.data_format or "text"
        data_path = args.data or DEFAULT_DATA_PATH
        if args.resume is not None and args.max_characters is None:
            raise ValueError(
                "旧格式 Checkpoint 没有训练配置；第一次继续训练时必须提供"
                " --max-characters，并保持原来的模型参数。"
            )
        max_characters = (
            (0 if data_format == "qa" else 10_000)
            if args.max_characters is None
            else args.max_characters
        )
        model_config = {
            "context_length": 64
            if args.context_length is None
            else args.context_length,
            "embedding_dim": 32 if args.embedding_dim is None else args.embedding_dim,
            "num_heads": 4 if args.num_heads is None else args.num_heads,
            "num_layers": 2 if args.num_layers is None else args.num_layers,
        }
        tokenizer_characters = None
    else:
        data_metadata = saved_metadata["data"]
        saved_model_config = saved_metadata["model"]
        if not isinstance(data_metadata, dict) or not isinstance(
            saved_model_config, dict
        ):
            raise ValueError("Checkpoint 训练元数据格式错误")

        data_path = args.data or Path(str(data_metadata["path"]))
        max_characters = int(data_metadata["max_characters"])
        data_format = data_metadata.get("format", "text")
        if args.data_format is not None and args.data_format != data_format:
            raise ValueError("继续训练不能修改 data-format")
        if args.max_characters is not None and args.max_characters != max_characters:
            raise ValueError("继续训练不能修改 max-characters")

        model_config = {
            "context_length": int(saved_model_config["context_length"]),
            "embedding_dim": int(saved_model_config["embedding_dim"]),
            "num_heads": int(saved_model_config["num_heads"]),
            "num_layers": int(saved_model_config["num_layers"]),
        }
        requested_model_config = {
            "context_length": args.context_length,
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
        }
        for name, requested_value in requested_model_config.items():
            if requested_value is not None and requested_value != model_config[name]:
                raise ValueError(f"继续训练不能修改 {name.replace('_', '-')}")

        tokenizer_characters = saved_metadata["tokenizer_characters"]
        if not isinstance(tokenizer_characters, list):
            raise ValueError("Checkpoint tokenizer_characters 格式错误")

    if max_characters < 0:
        raise ValueError("max-characters must not be negative")
    if data_format == "qa" and max_characters != 0:
        raise ValueError("qa 模式必须读取完整 JSONL，使用 --max-characters 0")

    text = data_path.read_text(encoding="utf-8")
    available_characters = len(text)
    if max_characters > 0:
        text = text[:max_characters]

    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if saved_metadata is not None:
        data_metadata = saved_metadata["data"]
        if not isinstance(data_metadata, dict):
            raise ValueError("Checkpoint data 元数据格式错误")
        if text_sha256 != data_metadata["text_sha256"]:
            raise ValueError("训练文本与 Checkpoint 保存的文本不一致")

    records: list[tuple[str, str]] = []
    if data_format == "qa":
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            question, answer = record["question"], record["answer"]
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError("JSONL question and answer must be strings")
            records.append((question, answer))
        vocabulary_text = "".join(
            f"用户：{question}\n助手：{answer}\n\n" for question, answer in records
        )
    else:
        vocabulary_text = text

    tokenizer = (
        CharacterTokenizer.from_text(vocabulary_text)
        if tokenizer_characters is None
        else CharacterTokenizer(tokenizer_characters)
    )
    torch.manual_seed(0)
    if data_format == "qa":
        qa_dataset = QuestionAnswerDataset(
            records, tokenizer, model_config["context_length"]
        )
        sample_count = len(qa_dataset)
        data_loader = DataLoader(
            qa_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_question_answers,
        )
    else:
        token_ids = tokenizer.encode(text)
        text_dataset = TextSequenceDataset(token_ids, model_config["context_length"])
        sample_count = len(text_dataset)
        data_loader = DataLoader(
            text_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=model_config["context_length"],
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    completed_epochs = 0
    if args.resume is not None:
        completed_epochs = load_checkpoint(
            args.resume,
            model,
            optimizer,
            map_location=device,
        )

    print(f"训练数据：{data_path}")
    print(f"使用字符：{len(text):,} / {available_characters:,}")
    print(f"词表大小：{tokenizer.vocab_size:,}")
    print(f"训练模式：{data_format}")
    print(f"训练样本：{sample_count:,}")
    if data_format == "qa":
        print(f"独立问答：{len(records):,}")
    print(f"模型参数：{sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"训练设备：{device}")
    if args.resume is not None:
        print(f"继续训练：已完成 {completed_epochs} 个 Epoch")

    metadata: dict[str, object] = {
        "format_version": 1,
        "data": {
            "path": str(data_path),
            "format": data_format,
            "max_characters": max_characters,
            "text_sha256": text_sha256,
        },
        "tokenizer_characters": [
            tokenizer.id_to_char[index] for index in range(tokenizer.vocab_size)
        ],
        "model": model_config,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_epoch = completed_epochs + args.epochs
    for added_epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data_loader, optimizer, device)
        current_epoch = completed_epochs + added_epoch
        save_checkpoint(
            output_path,
            model,
            optimizer,
            current_epoch,
            metadata=metadata,
        )
        print(
            f"Epoch {current_epoch}/{target_epoch} "
            f"| 本次 {added_epoch}/{args.epochs} | loss={loss:.4f}"
        )

    print(f"Checkpoint 已保存：{output_path}")


if __name__ == "__main__":
    main()
