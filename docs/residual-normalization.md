# Residual Connection + Normalization｜残差连接与归一化

Residual Connection（残差连接）保留子层输入并把它直接加回子层输出，
Layer Normalization（层归一化）则让每个 Token 的特征保持稳定的数值尺度。
这两个计算是 Transformer Block（Transformer 模块）的基础组成部分。

```text
English Term       ｜中文术语          ｜中文解释
Residual Connection｜残差连接          ｜把子层输入直接加到子层输出上的连接
Layer Normalization｜层归一化          ｜沿单个 Token 的特征维计算并调整数值分布
Mean               ｜均值              ｜一组数值之和除以数值数量
Variance           ｜方差              ｜各数值与均值之差的平方的平均值
Epsilon            ｜极小常数          ｜加在方差上以避免除以零的正数
Scale              ｜缩放参数          ｜归一化后对每个特征进行可学习缩放的参数
Shift              ｜平移参数          ｜归一化后对每个特征进行可学习平移的参数
```

## 1. Residual Connection

假设子层接收输入 `X` 并计算 `F(X)`，残差连接的公式是：

```text
Residual(X) = X + F(X)
```

`X` 和 `F(X)` 必须具有相同的 `[B,T,C]` shape，才能让相同 Batch、Token 和
Feature 位置的数值逐元素相加：

```text
X    [B,T,C]
F(X) [B,T,C]
────────────
Y    [B,T,C]
```

直接路径让信息和梯度不必完全经过子层变换才能继续向后传递。当前实现使用
`add_residual` 函数明确表达一次加法，不为这个操作增加无实际需要的类或状态。

## 2. Layer Normalization 的计算范围

输入 `X` 的形状是 `[B,T,C]`：

```text
B = Batch Size       ｜批次大小
T = Sequence Length  ｜序列长度
C = Embedding Dimension｜嵌入维度
```

Layer Normalization 只沿最后的 `C` 维计算。每个 Batch 中的每个 Token 都使用
自己的 `C` 个特征求 Mean（均值）和 Variance（方差），不会读取其他样本或其他
Token 的数值。

## 3. Mean、Variance 与 Normalize

对于单个 Token 的特征向量 `x = [x₁, x₂, ..., x꜀]`，先计算均值：

```text
mean = (x₁ + x₂ + ... + x꜀) / C
```

再把每个特征减去均值，并计算总体方差：

```text
centered = x - mean
variance = mean(centered²)
```

最后用标准差缩放中心化结果：

```text
normalized = centered / sqrt(variance + epsilon)
```

`epsilon` 当前默认为 `1e-5`。当一个 Token 的全部特征相同、方差为 0 时，
它可以避免除以 0。

## 4. Learnable Scale 与 Shift

纯归一化会固定特征分布。为了让模型能够根据训练目标调整各特征，
`LayerNormalization` 为每个特征维护两个形状为 `[C]` 的参数：

```text
output = scale * normalized + shift
```

`scale` 初始为 1，`shift` 初始为 0，因此初始输出就是归一化结果。训练时，
Autograd（自动求导）会为它们计算梯度。

## 5. 完整 shape 流程

```text
X [B,T,C]
↓ mean(dim=-1, keepdim=True)
Mean [B,T,1]
↓ X - Mean
Centered [B,T,C]
↓ mean(Centered², dim=-1, keepdim=True)
Variance [B,T,1]
↓ Centered / sqrt(Variance + Epsilon)
Normalized [B,T,C]
↓ Scale [C] 与 Shift [C] 广播
Output [B,T,C]
```

## 6. 当前 API

```python
import torch

from model_lab.residual_normalization import LayerNormalization, add_residual

x = torch.randn(2, 4, 8)
normalization = LayerNormalization(embedding_dim=8)
normalized = normalization(x)
output = add_residual(x, normalized)

assert normalized.shape == (2, 4, 8)
assert output.shape == (2, 4, 8)
```

当前只实现两个独立原语，尚未决定它们在 Transformer Block 中的组合顺序，
也不包含 Attention、Feed-Forward Network、Dropout 或完整模型结构。
