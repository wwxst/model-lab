"""读取 Checkpoint 内的模型配置和词表，在终端独立提问。"""

import argparse
from pathlib import Path

import torch

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.decoder_model import DecoderOnlyLanguageModel
from model_lab.generation import generate_greedy


def main() -> None:
    parser = argparse.ArgumentParser(description="加载本地模型，生成问题的回答")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--question", help="仅回答一个问题后退出")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    config = metadata["model"]
    tokenizer = CharacterTokenizer(metadata["tokenizer_characters"])
    model = DecoderOnlyLanguageModel(
        tokenizer.vocab_size,
        config["context_length"],
        config["embedding_dim"],
        config["num_heads"],
        config["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"模型：{args.checkpoint}，已完成 {checkpoint['epoch']} 个 Epoch")
    if args.question is None:
        print("输入问题后按回车，输入 /exit 退出。每道问题独立回答。")

    while True:
        question = args.question if args.question is not None else input("\n你：")
        question = question.strip()
        if question == "/exit":
            break
        if question:
            prompt = f"用户：{question}\n助手："
            unknown = set(prompt) - set(tokenizer.char_to_id)
            if unknown:
                print("词表中没有这些字符：", "".join(sorted(unknown)))
            else:
                inputs = tokenizer.encode(prompt).unsqueeze(0)
                generated = generate_greedy(model, inputs, max_new_tokens=100)
                answer = tokenizer.decode(generated[0, inputs.shape[1] :])
                # 文本中连续两个换行表示本条回答结束，只显示这个边界之前的内容。
                print("模型：", answer.split("\n\n")[0].split("\n用户：")[0])
        if args.question is not None:
            break


if __name__ == "__main__":
    main()
