# Token Embedding

`TokenEmbedding`（词元嵌入）把离散 Token ID 映射为可以参与神经网络连续数学计算的向量。它连接文本预处理与后续模型计算，但不包含位置信息或其他模型结构。

## 1. Token ID 为什么不能直接作为数值特征

Token ID 只是词表中的离散编号。例如 `A → 0`、`B → 1`、`C → 2` 不表示 `C > B > A`，也不表示 `C` 是 `A` 的两倍。这些整数负责定位 Token，不表达连续的大小或距离关系。

Embedding（嵌入）为每个 Token ID 准备一个连续向量：

```text
Token ID 0 → [ 0.12, -0.41, 0.83]
Token ID 1 → [-0.27,  0.66, 0.14]
```

向量中的数值才会进入后续神经网络计算。

## 2. Embedding Table

创建 Token Embedding 只需要两个参数：

```python
from model_lab.token_embedding import TokenEmbedding

embedding = TokenEmbedding(vocab_size=4, embedding_dim=3)
```

```text
V = Vocabulary Size      ｜词表大小 ｜词表中可用 Token 的数量
C = Embedding Dimension  ｜嵌入维度 ｜每个 Token 向量包含的连续特征数量
```

内部的 Embedding Table（嵌入表）形状为 `[V, C]`。上面的例子得到 `[4, 3]`：4 行分别属于 4 个 Token ID，每行保存一个 3 维向量。

`torch.nn.Embedding` 的本质就是维护这张参数表，然后用 Token ID 做索引查表：

```text
Token ID 2
    ↓
embedding.weight[2]
    ↓
第 2 行连续向量
```

`TokenEmbedding.weight` 直接返回这张表，方便观察它的 shape、数值和梯度。当前实现使用 PyTorch 默认初始化，不自定义初始化规则。

## 3. 输入与输出 shape

单个序列的转换是：

```text
token_ids       [T]
embedding table [V, C]
output          [T, C]
```

```text
T = Sequence Length     ｜序列长度 ｜输入中的 Token 数量
C = Feature Dimension   ｜特征维度 ｜每个 Token 向量的连续特征数量
```

例如 `[5] → [5, 8]` 表示 5 个 Token 分别变成 8 维向量。Embedding 不改变序列长度，只在每个 Token ID 后增加特征维度 `C`。

`nn.Embedding` 也天然保留批量输入的前置维度：

```text
token_ids [B, T]
    ↓
output    [B, T, C]

B = Batch Size ｜批次大小 ｜一次处理的序列数量
```

这只是查表操作对 batch shape 的自然支持，当前实现没有额外的 batching 系统。

## 4. dtype 与 device

Token ID 输入使用 `torch.int64`，输出 dtype 与 `embedding.weight.dtype` 相同，默认通常为 `torch.float32`。CPU 是当前测试基线；Token ID 和 Embedding 参数位于同一 device 时即可计算。

当前实现不涉及 `float16`、`bfloat16`、mixed precision（混合精度）或 quantization（量化）。

## 5. Parameter 与梯度

`embedding.weight` 是 `torch.nn.Parameter`（可学习参数），会被 PyTorch 的 Autograd（自动求导）记录。向量不是手工写死的；训练时，损失产生的梯度会通过反向传播到参数表。

```text
Token IDs
    ↓
Embedding Lookup
    ↓
Token Vectors
    ↓
scalar loss
    ↓ backward
embedding.weight.grad
```

只有被查到的行参与本次计算，因此 dense gradient（稠密梯度）中，使用过的 Token 行获得梯度，未使用行保持为 0。同一个 Token 出现多次时，同一行参数在计算图中被多次使用，梯度会累积。

当前实现只展示参数和梯度，不包含 Optimizer（优化器）或参数更新步骤。

## 6. 同一个 Token 与不同 Token

参数不变时，同一个 Token ID 总是查到 Embedding Table 的同一行，因此得到完全相同的向量。不同 ID 查不同的行，但随机初始化并不保证它们已经表达有意义的语义关系；这些关系需要后续训练逐渐学习。

## 7. 与 CharacterTokenizer 连接

`CharacterTokenizer` 先把真实文本转换为 Token ID，`TokenEmbedding` 再把每个 ID 转换为连续向量：

```python
from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.token_embedding import TokenEmbedding

text = "hello"
tokenizer = CharacterTokenizer.from_text(text)
token_ids = tokenizer.encode(text)
embedding = TokenEmbedding(tokenizer.vocab_size, embedding_dim=4)
vectors = embedding(token_ids)
```

```text
Raw Text
    ↓ CharacterTokenizer
Token IDs [T]
    ↓ TokenEmbedding
Continuous Vectors [T, C]
```

不需要额外的 Pipeline 类。

## 8. 与 TextSequenceDataset 连接

`TextSequenceDataset` 从完整 Token ID 序列中取出固定长度的输入窗口。这个窗口可以直接进入 Token Embedding：

```python
from model_lab.text_dataset import TextSequenceDataset

dataset = TextSequenceDataset(token_ids, context_length=3)
input_ids, target_ids = dataset[0]
input_vectors = embedding(input_ids)
```

```text
Raw Text
    ↓ CharacterTokenizer
Token IDs [N]
    ↓ TextSequenceDataset
input_ids [T]
    ↓ TokenEmbedding
input_vectors [T, C]
```

Tokenizer、Dataset 和 Token Embedding 各自保持单一职责，当前实现没有训练数据流水线抽象。

## 9. Token Embedding 不包含位置

Token Embedding 只回答“这个 Token 是什么”。所有相同 ID 都使用同一行向量，它不回答“这个 Token 在序列的第几个位置”。

下一阶段的位置表示会为序列加入顺序信息。当前实现不包含 position ID、Position Embedding（位置嵌入）、sinusoidal position encoding（正弦位置编码）或 RoPE。
