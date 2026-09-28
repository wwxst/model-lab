# Transformer Block｜Transformer 模块

`TransformerBlock` 把 Layer Normalization（层归一化）、Multi-Head Causal
Self-Attention（多头因果自注意力）、Feed-Forward Network（前馈网络）和
Residual Connection（残差连接）组合为一个可重复堆叠的 Decoder Block
（解码器模块）。

```text
English Term       ｜中文术语          ｜中文解释
Transformer Block  ｜Transformer 模块  ｜组合 Attention、MLP、Normalization 和 Residual 的基本计算单元
Sublayer           ｜子层              ｜Block 内承担一种计算职责的 Attention 或 Feed-Forward 组件
Pre-Norm           ｜前置归一化        ｜在输入进入每个子层之前执行 Layer Normalization
Residual Path      ｜残差路径          ｜让子层输入绕过子层并直接加到输出上的路径
Hidden State       ｜隐藏状态          ｜Token 在模型内部逐层更新的连续向量表示
```

## 1. Block 组合了什么

当前 Block 包含两个 Sublayer（子层）：

```text
Attention Sublayer    ｜在不同 Token 之间汇聚可见的上下文信息
Feed-Forward Sublayer ｜对每个 Token 的特征独立执行非线性变换
```

每个子层之前都有自己独立的 Layer Normalization，之后都有一条 Residual
Connection。两个 Normalization 不能共享 scale 和 shift，因为它们面对的是不同
计算阶段的隐藏状态。

## 2. Pre-Norm 顺序

当前实现采用 Pre-Norm（前置归一化）：先归一化，再执行子层，最后加回残差。

```text
X
├──────────────────────────────┐
↓ Layer Normalization          │ Residual Path
↓ Multi-Head Self-Attention    │
↓                              │
+ ←────────────────────────────┘
↓ X₁
├──────────────────────────────┐
↓ Layer Normalization          │ Residual Path
↓ Feed-Forward Network         │
↓                              │
+ ←────────────────────────────┘
↓ Output
```

对应公式是：

```text
X₁ = X + Attention(LayerNorm₁(X))
Y  = X₁ + FeedForward(LayerNorm₂(X₁))
```

Post-Norm（后置归一化）会在残差相加之后归一化。当前项目只实现 Pre-Norm，
不同时维护两套 Block 结构。

## 3. Attention 子层

输入 `X` 的形状是 `[B,T,C]`。第一个 Layer Normalization 只沿每个 Token 的
`C` 个特征计算，因此 shape 不变。Multi-Head Self-Attention 同样返回
`[B,T,C]`，可以与原始 `X` 逐元素相加：

```text
X [B,T,C]
↓ LayerNorm₁
[B,T,C]
↓ Causal Attention
[B,T,C]
↓ + X
X₁ [B,T,C]
```

Causal Mask（因果掩码）仍由 Attention 负责，所以当前位置不能读取未来 Token。

## 4. Feed-Forward 子层

第二个 Layer Normalization 接收 Attention 残差相加后的 `X₁`。前馈网络只沿
特征维执行 `C → 4C → GELU → C`，不混合 Token：

```text
X₁ [B,T,C]
↓ LayerNorm₂
[B,T,C]
↓ Feed-Forward
[B,T,C]
↓ + X₁
Output [B,T,C]
```

Attention 提供 Token 之间的信息交互，Feed-Forward 则变换每个 Token 内部的
特征。两者承担不同职责。

## 5. Residual Path 为什么保持信息

如果某个子层暂时输出全 0，残差连接会得到：

```text
X + 0 = X
```

因此 Block 至少保留一条从输入到输出的直接路径。反向传播时，梯度也可以沿这条
路径传回较早层，而不必完全依赖 Attention 或 Feed-Forward 的变换。

## 6. 完整 shape 流程

```text
X                              [B,T,C]
LayerNorm₁(X)                  [B,T,C]
Attention(LayerNorm₁(X))       [B,T,C]
X₁ = X + Attention Output      [B,T,C]
LayerNorm₂(X₁)                 [B,T,C]
FeedForward(LayerNorm₂(X₁))    [B,T,C]
Y = X₁ + Feed-Forward Output   [B,T,C]
```

## 7. 当前 API

```python
import torch

from model_lab.transformer_block import TransformerBlock

block = TransformerBlock(embedding_dim=8, num_heads=2)
x = torch.randn(2, 4, 8)
output = block(x)

assert output.shape == (2, 4, 8)
```

当前 Block 不包含 Dropout、Decoder Stack、最终 Layer Normalization、Language
Model Head、Loss、训练或生成逻辑。多个 Block 的顺序堆叠由后续 Decoder Stack
负责。
