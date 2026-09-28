"""用 Greedy Autoregressive Generation 逐 Token 延长序列。"""

from __future__ import annotations

import torch

from model_lab.decoder_model import DecoderOnlyLanguageModel


def generate_greedy(
    model: DecoderOnlyLanguageModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """根据输入 Token IDs 追加最多 max_new_tokens 个贪心选择的 Token。"""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    if input_ids.dtype != torch.int64:
        raise TypeError("input_ids must use torch.int64")
    if input_ids.shape[1] <= 0:
        raise ValueError("sequence length must be greater than zero")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative")

    # 生成是推理过程，不需要建立训练计算图。保存并恢复原来的 training 状态，
    # 让调用者在生成结束后继续使用模型时不会被意外改变模式。
    was_training = model.training
    model.eval()
    generated = input_ids.clone()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 模型上下文有限时只保留最后 max_sequence_length 个 Token。这样
                # 每轮仍可用完整的合法窗口预测下一个位置。
                context = generated[:, -model.max_sequence_length :]
                logits = model(context)

                # 只读取当前窗口最后一个位置的 [V] Logits；argmax 选择分数最高
                # 的 Token ID，形成确定性的 Greedy（贪心）下一步预测。
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(was_training)

    return generated
