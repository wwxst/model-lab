"""用本地对话文本启动一次最小的 Decoder-only 模型训练。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.checkpoint import save_checkpoint
from model_lab.decoder_model import DecoderOnlyLanguageModel
from model_lab.text_dataset import TextSequenceDataset
from model_lab.training import train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个最小中文对话语言模型")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/alpaca_zh_chat_1000.txt"),
        help="UTF-8 训练文本路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/chat-model.pt"),
        help="Checkpoint 输出路径",
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--max-characters", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if args.max_characters < 0:
        raise ValueError("max-characters must not be negative")

    text = args.data.read_text(encoding="utf-8")
    available_characters = len(text)
    if args.max_characters > 0:
        text = text[: args.max_characters]

    tokenizer = CharacterTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    dataset = TextSequenceDataset(token_ids, args.context_length)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=args.context_length,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    print(f"训练数据：{args.data}")
    print(f"使用字符：{len(text):,} / {available_characters:,}")
    print(f"词表大小：{tokenizer.vocab_size:,}")
    print(f"训练窗口：{len(dataset):,}")
    print(f"模型参数：{sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"训练设备：{device}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data_loader, optimizer, device)
        save_checkpoint(args.checkpoint, model, optimizer, epoch)
        print(f"Epoch {epoch}/{args.epochs} | loss={loss:.4f}")

    print(f"Checkpoint 已保存：{args.checkpoint}")


if __name__ == "__main__":
    main()
