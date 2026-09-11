# Character Tokenizer

`CharacterTokenizer`（字符级分词器）把人类可读的字符串转换成后续模型输入流程使用的离散 Token ID 序列，并且可以把这个序列还原成原始文本。Token ID 还需要经过 Embedding（嵌入），才能成为模型计算使用的连续向量表示。

## 1. 为什么需要 Tokenizer

模型的输入是 Tensor（张量）中的数值，不能直接读取 Python `str`。Tokenizer（分词器）负责把文本拆成 Token（词元），再为每个 Token 分配一个 Token ID（词元编号）。当前实现选择最容易观察的规则：把 Python `str` 迭代得到的每个 Unicode 码点作为一个 Token。

```text
Raw Text（原始文本）
    ↓
Character Vocabulary（字符词表）
    ↓
Character → Token ID（字符到编号）
    ↓
Token IDs（整数序列）
```

Token ID 只是离散编号，不是有大小关系的数值。例如 `A → 0`、`B → 1`、`C → 2` 不表示 `C` 比 `B`“更大”，也不表示 `C` 是 `A` 的两倍。下一阶段的 Token Embedding（词元嵌入）会把这些编号映射为可学习的连续向量。

## 2. Vocabulary 如何建立

使用训练文本创建 Tokenizer：

```python
from model_lab.character_tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer.from_text("hello")
```

实现使用 `sorted(set(text))`：先去掉重复字符，再按确定性顺序排序。因此 `"hello"` 的词表是：

```text
e → 0
h → 1
l → 2
o → 3
```

相同的训练文本会产生相同的映射，`tokenizer.vocab_size` 为 `4`。空训练文本没有字符可以建立词表，因此会抛出 `ValueError`。

## 3. encode

```python
token_ids = tokenizer.encode("hello")
# tensor([1, 0, 2, 2, 3])
```

返回值是 CPU 上形状为 `[T]` 的 `torch.int64` Tensor：

```text
T = Sequence Length ｜序列长度 ｜Python str 迭代得到的 Unicode 码点数量
```

每个 Unicode 码点占一个位置。已经创建好的 Tokenizer 编码空字符串时，返回形状为 `[0]` 的空 Tensor。如果文本包含不在词表中的码点，Tokenizer 会抛出明确的 `ValueError`，当前实现不会添加 `<UNK>` 或其他特殊 Token。

## 4. decode

```python
text = tokenizer.decode(token_ids)
# "hello"
```

`decode` 接受一维整数 Token ID 序列，可以是 Tensor 或 Python 整数序列。超过词表范围或为负数的 ID 都会抛出 `ValueError`；多维 Tensor 和非整数输入会明确失败。空序列会还原为空字符串。

## 5. Round Trip

对于词表中存在的字符，编码和解码互为逆操作：

```python
assert tokenizer.decode(tokenizer.encode("hello")) == "hello"
```

这条性质是当前实现最重要的数值行为：字符没有被丢弃、替换或重新排序。

## 6. Unicode 字符

Python `str` 按 Unicode 码点迭代，因此中文、英文、标点、空格和换行都可以直接进入词表。一个用户看到的完整字形可能由多个码点组成，当前实现会把这些码点分别作为 Token。例如：

```python
text = "你好，AI\n"
tokenizer = CharacterTokenizer.from_text(text)
assert tokenizer.decode(tokenizer.encode(text)) == text
```

当前实现不把字符串转换为 UTF-8 字节，也不实现 byte-level tokenizer（字节级分词器）。

## 7. 与 TextSequenceDataset 连接

`TextSequenceDataset` 接受一维 `torch.int64` 离散序列，因此可以直接接收真实文本经过 Tokenizer 编码后的 Token ID：

```python
from model_lab.text_dataset import TextSequenceDataset

token_ids = tokenizer.encode("hello")
dataset = TextSequenceDataset(token_ids, context_length=3)
```

对于 `[1, 0, 2, 2, 3]`，第一个样本是：

```text
x = [1, 0, 2]
y = [0, 2, 2]
```

Tokenizer 只负责文本与 Token ID 的转换，不负责 Dataset、batch、padding、truncation 或训练数据流水线。

## 8. 优点与局限

字符级分词器的逻辑很简单，词表容易检查，最适合学习文本如何变成整数输入。它的局限是序列可能很长：`hello` 需要 5 个 Token，`你好` 需要 2 个 Token，复杂文本的序列长度也会随 Unicode 码点数量增长。

现代大模型通常选择 Subword Tokenization（子词分词）来平衡 Vocabulary Size（词表大小）和 Sequence Length（序列长度）。当前实现不包含 BPE、WordPiece、SentencePiece 等算法；下一阶段的 Token Embedding 会使用这里产生的整数 Token ID。
