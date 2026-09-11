# Position Representation

`PositionEmbedding`（位置嵌入）为序列中的每个绝对位置提供一个可学习向量。Token Embedding 回答“这个 Token 是什么”，位置表示补充“这个 Token 在哪里”。两者相加后，才得到同时包含 Token 信息和位置信息的输入表示。

## 1. 为什么 Token Embedding 不够

Token Embedding 只根据 Token ID 查表。同一个 Token 出现在位置 0、位置 1 或位置 5，都会得到同一行 Token 向量。因此仅有 Token Embedding 时，模型无法从向量本身知道顺序。

例如，`我 爱 你` 与 `你 爱 我` 包含相同的 Token 集合，但顺序不同。位置表示为这些不同位置提供额外信息。

```text
Token ID     → “是什么”
Position ID  → “在哪里”
```

## 2. Position ID

长度为 `T` 的序列使用连续的绝对位置编号：

```text
T = 4
Position IDs = [0, 1, 2, 3]
```

当前实现用 `torch.arange(T)` 生成 Position IDs。它们是形状 `[T]`、dtype 为 `torch.int64` 的整数编号，与 Token ID 使用相同的索引 dtype，但表达的是不同概念。Position ID 不表示词表中的 Token，也不会替代 Token ID。

## 3. Position Embedding Table

创建位置嵌入只需要最大序列长度和向量维度：

```python
from model_lab.position_embedding import PositionEmbedding

position_embedding = PositionEmbedding(
    max_sequence_length=8,
    embedding_dim=4,
)
```

```text
P = Maximum Sequence Length ｜最大序列长度 ｜位置表支持的位置数量
C = Embedding Dimension     ｜嵌入维度     ｜每个位置向量的连续特征数量
```

内部的 Position Table（位置表）形状为 `[P, C]`。位置 0 查 `position_embedding.weight[0]`，位置 1 查 `position_embedding.weight[1]`，每一行都是一个可学习向量。`weight` 是 `torch.nn.Parameter`，会通过反向传播得到梯度；它不是手工固定的编码公式。

当前实现直接使用 `torch.nn.Embedding(P, C)` 的默认初始化，不实现 RoPE、Sinusoidal Position Encoding、ALiBi 或相对位置表示。

## 4. `[T] → [T, C]`

调用位置嵌入时只需要序列长度：

```python
position_vectors = position_embedding(sequence_length=5)
```

查表过程是：

```text
Position IDs       [T]
Position Table     [P, C]
Position Vectors   [T, C]
```

例如 `T = 5`、`C = 8` 时，输出形状为 `[5, 8]`。序列长度不变，只是在每个位置后增加连续特征维度 `C`。Position ID 从 0 开始，最后一个位置是 `T - 1`。

`sequence_length` 必须大于 0，且不能超过 `max_sequence_length`。当前实现会明确抛出 `ValueError`，不会截断、循环、扩展或自动调整位置表。

## 5. Token 与 Position 相加

Token Embedding 和 Position Embedding 的维度必须相同，才能逐元素相加：

```python
token_vectors = token_embedding(token_ids)  # [T, C]
position_vectors = position_embedding(T)  # [T, C]
input_representation = token_vectors + position_vectors
```

```text
Token Vectors       [T, C]
+ Position Vectors  [T, C]
= Input Representation [T, C]
```

相同 Token 在不同位置的 Token Vector 仍然相同，但 Position Vector 不同，因此最终 Input Representation 可以不同。这是位置表示真正补充的内容。

## 6. Batch 与 Broadcasting

批量 Token Embedding 的输出形状为 `[B, T, C]`。位置向量与 batch 无关，先把 `[T, C]` 增加一个大小为 1 的 batch 维度：

```python
token_vectors = token_embedding(batch_token_ids)  # [B, T, C]
position_vectors = position_embedding(T)  # [T, C]
position_vectors = position_vectors.unsqueeze(0)  # [1, T, C]
input_representation = token_vectors + position_vectors
```

```text
[B, T, C] + [1, T, C] → [B, T, C]
```

Broadcasting（广播）会把同一个位置向量提供给每个 batch item。每条序列都从 Position ID 0 开始，因此 batch 不拥有独立的位置编号体系；batch 中所有样本的位置 1 共享位置表的第 1 行。

## 7. 梯度学习

位置表是可学习参数。一个简单的求和损失经过 `backward()` 后，实际用到的位置行获得梯度，未用到的位置行保持为 0：

```text
sequence_length = 3
使用位置行：0、1、2
未使用位置行：3、4、...
```

`backward()` 负责计算或累积 gradient（梯度）；它不会直接更新参数。真正更新 Parameter（参数）需要 Optimizer（优化器），当前实现不包含优化器或训练循环。

## 8. 与 Tokenizer 和 Dataset 连接

真实文本经过 CharacterTokenizer，再经过 TextSequenceDataset 取出输入窗口，最后可以直接组合 Token Embedding 与 Position Embedding：

```python
from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.position_embedding import PositionEmbedding
from model_lab.text_dataset import TextSequenceDataset
from model_lab.token_embedding import TokenEmbedding

text = "hello"
tokenizer = CharacterTokenizer.from_text(text)
token_ids = tokenizer.encode(text)
dataset = TextSequenceDataset(token_ids, context_length=3)
input_ids, _ = dataset[0]

token_embedding = TokenEmbedding(tokenizer.vocab_size, embedding_dim=4)
position_embedding = PositionEmbedding(max_sequence_length=3, embedding_dim=4)
input_representation = token_embedding(input_ids) + position_embedding(3)
```

```text
Raw Text
    ↓ CharacterTokenizer
Token IDs [N]
    ↓ TextSequenceDataset
input_ids [T]
    ↓ TokenEmbedding
Token Vectors [T, C]
    + PositionEmbedding [T, C]
    ↓
Input Representation [T, C]
```

这只是三个现有模块的直接组合，没有额外 Pipeline 或模型输入层。

## 9. 当前方案的限制

当前实现是 Learnable Absolute Position Embedding（可学习绝对位置嵌入）：每个绝对位置拥有一个独立的可学习向量。位置表有固定的 `max_sequence_length`，不能自然外推到表外的新位置；这也是现代模型可能选择 RoPE 等其他位置方案的原因之一。

现在每个位置已经同时拥有 Token 信息和 Position 信息，但不同位置之间还没有交换上下文。下一阶段的 Single-Head Self-Attention（单头自注意力）会处理一个位置如何读取其他位置的信息。
