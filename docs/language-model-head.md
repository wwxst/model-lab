# Language Model Head｜语言模型输出头

`LanguageModelHead`（语言模型输出头）把 Decoder Stack 产生的 Hidden State
（隐藏状态）从 Embedding Dimension（嵌入维度）投影到 Vocabulary Dimension
（词表维度），为每个序列位置产生一组预测下一个 Token 的 Logits（未归一化分数）。

```text
English Term       ｜中文术语          ｜中文解释
Language Model Head｜语言模型输出头    ｜把隐藏状态投影为词表中每个 Token 的 Logits
Vocabulary Dimension｜词表维度        ｜输出最后一维的大小，等于词表中的 Token 数量
Output Projection  ｜输出投影          ｜把 C 维隐藏状态线性映射为 V 个词表分数的变换
Raw Score          ｜原始分数          ｜Softmax 之前可以为任意实数的模型输出
Weight Tying       ｜权重绑定          ｜让输出投影与 Token Embedding 共享同一参数表的做法
```

## 1. 从 Hidden State 到词表分数

Decoder Stack 输出 `[B,T,C]`：

```text
B = Batch Size          ｜批次大小
T = Sequence Length     ｜序列长度
C = Embedding Dimension ｜嵌入维度
V = Vocabulary Size     ｜词表大小
```

Language Model Head 使用一个 `C → V` Linear Projection（线性投影）：

```text
Hidden States [B,T,C]
↓ Linear(C, V, bias=False)
Logits [B,T,V]
```

`B` 和 `T` 不变，只有最后一维从内部特征数量 `C` 变为候选 Token 数量 `V`。

## 2. 投影权重表示什么

投影权重形状为 `[V,C]`。第 `v` 行属于词表中的第 `v` 个 Token。对某个位置的
隐藏向量 `h`，第 `v` 个 Logit 是：

```text
logit_v = weight_v · h
```

点积越大，表示当前隐藏状态与这个 Token 的输出方向越匹配。权重会在训练中通过
梯度更新。

当前投影不使用 bias，直接学习 Hidden State 与各词表 Token 之间的线性关系。

## 3. 为什么输出 Logits 而不是概率

Logits 是 Raw Score（原始分数），可以是任意正数、负数或 0，也不要求总和为 1：

```text
Logits = [2.0, 3.0, -1.0]
```

训练时，Cross Entropy Loss（交叉熵损失）会直接接收 Logits，并在数值稳定的计算
中完成归一化。生成时可以根据 Logits 选择下一个 Token。提前在 Head 中执行
Softmax 会重复职责，并使后续 Loss 的数值计算不必要地复杂。

## 4. 每个位置独立投影

Linear 只变换最后的 `C` 维，不沿 `T` 维混合位置：

```text
Hidden State at Token 0 [C] → Head → Logits at Token 0 [V]
Hidden State at Token 1 [C] → Head → Logits at Token 1 [V]
Hidden State at Token 2 [C] → Head → Logits at Token 2 [V]
```

不同位置共享同一个投影权重，但一个位置的 Head 计算不会读取其他位置。Token 之间
的信息关系已经由前面的 Causal Attention 写入 Hidden State。

## 5. 当前为什么不做 Weight Tying

Weight Tying（权重绑定）会让 Language Model Head 的 `[V,C]` 权重与 Token
Embedding 的 `[V,C]` 参数表共享存储。它可以减少参数，但当前 Head 还是独立组件，
尚未与完整模型组合。没有真实组合关系时提前绑定会越过当前 Commit 边界，因此当前
使用独立参数。

## 6. 当前 API

```python
import torch

from model_lab.language_model_head import LanguageModelHead

head = LanguageModelHead(embedding_dim=8, vocab_size=32)
hidden_states = torch.randn(2, 4, 8)
logits = head(hidden_states)

assert logits.shape == (2, 4, 32)
```

当前 Head 不包含 Token/Position Embedding、Decoder Stack、Softmax、Loss、训练或
生成逻辑，也不与 Token Embedding 共享权重。
