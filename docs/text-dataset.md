# Text Dataset｜文本数据与训练样本

本文用一个最小的 PyTorch `Dataset` 说明：语言模型如何把一维离散序列切成训练样本。这里的整数序列只是已经离散化的数据示例，不是本项目已经实现的 Tokenizer（分词器）输出。

## 1. Dataset 负责什么

Dataset（数据集）负责保存样本并按索引取出样本。本项目的 `TextSequenceDataset` 接收一个一维 `torch.int64` Tensor，使用固定长度窗口返回一对：

```text
input（输入）  ｜模型看到的当前位置序列
target（目标） ｜每个位置之后的下一元素序列
```

Dataset 只准备数据，不负责预测、计算 loss（损失）或更新参数。

## 2. 从文本到训练样本的位置

完整语言模型流程还会经过 Tokenizer（分词器），但 Tokenizer 属于后续内容。本阶段从已经离散化的一维序列开始：

```text
原始文本
   ↓  （未来由 Tokenizer 完成）
离散序列
   ↓  （当前 Dataset 完成）
固定长度窗口
   ↓
input / target
```

例如，下面的数字只代表离散序列中的元素：

```text
10 11 12 13 14 15
```

当前代码不把它们称为真实 Token ID，也不实现字符到整数的映射。

## 3. 为什么 input 和 target 要错一位

语言模型训练的目标是 next-token prediction（下一位置预测）。窗口长度为 `3` 时：

```text
input:  [10, 11, 12]
target: [11, 12, 13]
```

每个位置分别学习：

```text
input[0] → target[0]   10 → 11
input[1] → target[1]   11 → 12
input[2] → target[2]   12 → 13
```

`target` 不是一个代表整句话的单独分类标签，而是 `input` 整体向右移动一个位置后的序列。未来模型会用输出 logits（未归一化分数）与 target 计算 loss；本阶段不实现模型和 loss。

## 4. context_length 是什么

`context_length`（上下文长度）是每个样本中 input 的元素数量，也就是这里的 `T`：

```text
T = context_length
input.shape  = [T]
target.shape = [T]
```

本阶段只返回单个样本，不增加 batch 维度 `[B, T]`。未来需要批处理时，多个 `[T]` 样本才会组合成 `[B, T]`。

## 5. 滑动窗口如何产生多个样本

序列长度为 `N`、上下文长度为 `T` 时，窗口起点从 `0` 到 `N - T - 1`，所以：

```text
样本数量 = N - T
```

当 `N = 6`、`T = 3` 时有三个样本：

```text
sample 0: input [10, 11, 12]  target [11, 12, 13]
sample 1: input [11, 12, 13]  target [12, 13, 14]
sample 2: input [12, 13, 14]  target [13, 14, 15]
```

相邻样本只向后移动一个位置，因此连续序列中的每个位置都能参与预测关系。

## 6. `__len__` 做什么

`__len__` 返回 Dataset 能够提供的样本数量，而不是原始序列长度。每个样本需要 `T` 个 input 元素，以及紧随其后的一个 target 元素，因此可用窗口数量是：

```python
len(dataset) == data.numel() - context_length
```

如果序列不足以产生一个完整的 input/target 窗口，构造 Dataset 会直接报错。

## 7. `__getitem__` 做什么

给定样本索引 `index`，核心切片保持直接可读：

```python
start = index
end = start + context_length

x = data[start:end]
y = data[start + 1 : end + 1]
```

`x` 和 `y` 长度相同；`y` 比 `x` 向后移动一个位置。返回值是 `(x, y)`，两个 Tensor 的 shape 都是 `[T]`。

## 8. 为什么当前使用整数序列

后续 Tokenizer 会把文本转换为离散 Token ID，Token ID 通常用 `torch.int64`（也写作 `torch.long`）保存。为了先学习 Dataset，本阶段直接使用已经准备好的整数 Tensor 来模拟这个输入：

```python
data = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
```

当前没有 Vocabulary（词表）、字符收集、`encode` 或 `decode`。这些属于后续 Tokenizer 阶段。

## 9. Train / Validation 为什么要分开

Training Set（训练集）用于让模型学习，Validation Set（验证集）用于检查模型在未参与训练的连续数据上的表现。`split_sequence` 按比例把一维序列切成前后两段：

```python
train_data, validation_data = split_sequence(data, train_ratio=0.8)
```

对于长度为 `N` 的序列，训练段长度使用 `int(N * train_ratio)`，验证段从该位置开始，直到序列末尾。

## 10. 为什么连续文本按顺序切分

语言模型数据具有顺序关系。当前实现不 shuffle（打乱），而是保留原始顺序：前一段作为 train，后一段作为 validation。这样不会在切分阶段额外破坏连续文本结构，也能清楚展示每一段来自原序列的哪个位置。

## 11. Dataset 和未来 DataLoader 的区别

Dataset 描述“一个索引对应什么样本”；DataLoader（数据加载器）负责未来的批量组合、遍历和可能的打乱。本阶段只学习 Dataset，因此没有引入 `DataLoader`、batching（批处理）、worker 或 sampler（采样器）。

## 12. 当前实现与未来方向

当前已实现：

- 一维 `torch.int64` 序列的顺序切分。
- 固定 `context_length` 的滑动窗口。
- 对齐的 input/target next-token prediction 样本。
- 单样本 `[T]` shape 和必要的输入边界检查。

未来方向：

- Character Tokenizer（字符分词器）和 Vocabulary。
- 多个样本组成 `[B, T]` 的批处理。
- Embedding、Transformer、loss、训练循环和验证指标。

这些内容不由当前 Dataset 实现提前承担。

## 13. 与后续语言模型训练的关系

后续训练流程会把这里返回的样本接到模型：

```text
input token IDs
   ↓
Transformer
   ↓
logits
   ↓ 与 target token IDs 比较
loss
```

Dataset 的职责到返回 `(input, target)` 为止。理解窗口长度、shape 和一位偏移后，才能准确判断模型每个位置应该预测什么。
