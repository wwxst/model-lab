# Autoregressive Generation｜自回归生成

`generate_greedy`（贪心生成）从一段已有 Token IDs 开始，一次预测并追加一个
Token，直到达到 `max_new_tokens`。每一步都把当前序列送入 Decoder-only Model，
只读取最后一个位置的 Logits，并选择分数最高的词表项。

```text
English Term            ｜中文术语          ｜中文解释
Autoregressive          ｜自回归            ｜使用已经生成的序列逐步预测下一个 Token 的过程
Generation              ｜生成              ｜从输入上下文计算并追加新 Token 的推理过程
Greedy Decoding         ｜贪心解码          ｜每一步选择当前 Logit 最大的 Token
Next Token              ｜下一个词元        ｜当前序列之后新增的一个 Token
Context Window          ｜上下文窗口        ｜模型单次前向传播能够看到的 Token 范围
Inference Mode          ｜推理模式          ｜不建立反向计算图、只执行预测的运行方式
```

## 1. 单步生成流程

给定当前序列 `generated`，每一步执行：

```text
generated [B,T]
↓ Decoder-only Model
logits [B,T,V]
↓ 选择最后位置 logits[:, -1, :]
[B,V]
↓ argmax(dim=-1)
next_token [B,1]
↓ cat 到序列末尾
generated [B,T+1]
```

只有最后一个位置需要用来预测下一个 Token。更早位置的 Logits 已经计算过，不能
直接代表追加 Token 后的新上下文结果，因此每一步都重新执行一次前向传播。

## 2. 为什么叫 Autoregressive

第 `n+1` 个 Token 的输入依赖前面已经存在的 Token：

```text
初始： [t₀, t₁]
第 1 步：使用 [t₀, t₁] 预测 t₂
第 2 步：使用 [t₀, t₁, t₂] 预测 t₃
第 3 步：使用 [t₀, t₁, t₂, t₃] 预测 t₄
```

这个循环与训练时的并行 next-token prediction 不同：训练可以一次计算整个目标
窗口，生成必须等待当前 Token 选出后才能进行下一步。

## 3. Greedy Decoding

当前解码规则非常直接：

```text
next_token = argmax(logits[:, -1, :])
```

它是确定性的：相同模型、相同输入和相同参数会产生相同序列。当前不加入
Temperature（温度）、Top-k、Top-p、随机采样或 Beam Search；这些是未来具有真实
需求时再单独引入的策略。

## 4. Context Window 裁剪

模型的位置 Embedding 只支持 `max_sequence_length` 个位置。当已生成序列超过这个
长度时，当前实现只保留最后窗口：

```python
context = generated[:, -model.max_sequence_length :]
```

模型因此始终接收合法的 `[B,T]` 输入。已经生成的完整序列仍会保留在返回值中，
裁剪只影响下一步模型前向传播看到的上下文。

## 5. Inference Mode 与状态恢复

生成不需要梯度，所以当前实现使用 `torch.no_grad()`，避免为每一步建立训练图。
函数暂时把模型切换到 `eval()`，完成后恢复调用前的 training/eval 状态，让生成
不会改变调用者后续训练或评估行为。

## 6. 当前 API

```python
import torch

from model_lab.generation import generate_greedy

prompt_ids = torch.tensor([[0, 1, 2]], dtype=torch.int64)
generated_ids = generate_greedy(
    model,
    prompt_ids,
    max_new_tokens=8,
)

assert generated_ids.shape == (1, 11)
```

当前函数只返回 Token ID Tensor，不负责 Tokenizer Decode、停止词判断、采样策略、
KV Cache 或流式输出。
