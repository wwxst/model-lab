# Complete Decoder-only Model｜完整仅解码器模型

`DecoderOnlyLanguageModel` 把此前独立实现的 Token Embedding、Position
Embedding、Decoder Stack、Final Layer Normalization 和 Language Model Head
连接成一条完整前向计算链路。模型接收 Token IDs，并为每个位置输出预测下一个
Token 的 Logits。

```text
English Term          ｜中文术语          ｜中文解释
Decoder-only Model    ｜仅解码器模型      ｜只使用因果 Decoder 结构预测后续 Token 的语言模型
Forward Pass          ｜前向传播          ｜输入依次经过模型组件并产生输出的计算过程
Final Normalization   ｜最终归一化        ｜Decoder Stack 之后、Language Model Head 之前的 Layer Normalization
Model Context         ｜模型上下文        ｜一次前向传播最多能够接收的 Token 序列范围
End-to-End Pipeline   ｜端到端链路        ｜从 Token IDs 输入到 Logits 输出的完整连接关系
```

## 1. 完整前向链路

模型的数据流是：

```text
Token IDs [B,T]
↓ Token Embedding
Token Representations [B,T,C]
+ Position Embedding [T,C]
↓
Input Hidden States [B,T,C]
↓ Decoder Stack: N × Transformer Block
Contextual Hidden States [B,T,C]
↓ Final Layer Normalization
Normalized Hidden States [B,T,C]
↓ Language Model Head: C → V
Logits [B,T,V]
```

```text
B = Batch Size             ｜批次大小
T = Sequence Length        ｜序列长度
C = Embedding Dimension    ｜嵌入维度
V = Vocabulary Size        ｜词表大小
N = Number of Layers       ｜Decoder Block 数量
```

## 2. Token 与 Position Representation 相加

Token Embedding 表示每个 Token“是什么”，Position Embedding 表示它“在哪里”：

```text
Token Representation    [B,T,C]
Position Representation   [T,C]
────────────────────────────────
Input Hidden State      [B,T,C]
```

Position Embedding 没有 Batch 维，PyTorch 会把相同的位置表示广播给 Batch 中的
每个样本。两种表示具有相同的 `C` 维，因此可以逐元素相加。

## 3. Decoder Stack 更新上下文

相加后的 Hidden State 依次通过 `N` 个 Pre-Norm Transformer Block。每个 Block
使用 Causal Self-Attention 汇聚自己和过去 Token 的信息，再用 Feed-Forward
Network 变换每个 Token 的特征。

所有 Block 都保持 `[B,T,C]`，并且每层使用 Causal Mask，所以修改未来 Token
不会改变较早位置最终得到的 Logits。

## 4. 为什么 Stack 后还有 Final Normalization

Pre-Norm Block 在每个子层之前执行 Layer Normalization，但最后一条 Residual
Connection 的输出不会再自动经过下一个 Block 的归一化。当 Decoder Stack 结束时，
Final Layer Normalization 对最终隐藏状态进行一次归一化，再交给 Language Model
Head。

它拥有独立的 scale 和 shift 参数，不与 Block 内部的 Layer Normalization 共享。

## 5. Language Model Head 产生 Logits

Final Normalization 之后，Language Model Head 把最后一维从 `C` 投影为 `V`：

```text
[B,T,C] → Linear(C, V, bias=False) → [B,T,V]
```

`logits[b,t,v]` 表示第 `b` 个样本在位置 `t` 对词表 Token `v` 的原始预测分数。
模型不在前向传播中执行 Softmax，也不计算 Loss。

## 6. 输入契约

`token_ids` 必须满足：

```text
shape = [B,T]
dtype = torch.int64
0 < T <= max_sequence_length
```

Token ID 是否位于 `[0,V)` 继续由底层 `nn.Embedding` 的真实索引边界检查，不增加
重复的 Token 范围校验。

## 7. 当前 API

```python
import torch

from model_lab.decoder_model import DecoderOnlyLanguageModel

model = DecoderOnlyLanguageModel(
    vocab_size=32,
    max_sequence_length=16,
    embedding_dim=8,
    num_heads=2,
    num_layers=3,
)
token_ids = torch.randint(0, 32, (2, 4), dtype=torch.int64)
logits = model(token_ids)

assert logits.shape == (2, 4, 32)
```

当前模型已完成从 Token IDs 到 Logits 的前向传播，但尚未实现 Cross Entropy
Loss、Optimizer 驱动的训练循环、Checkpoint 或 Autoregressive Generation。
