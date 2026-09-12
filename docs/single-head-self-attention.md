# Single-Head Self-Attention｜单头自注意力

## 1. Self-Attention 解决什么问题

Token Embedding（词元嵌入）说明一个 Token 是什么，Position Embedding（位置嵌入）说明它在哪里。两者相加后，每个位置仍然只拥有自己的输入表示，还没有读取上下文中其他位置的信息。

Self-Attention（自注意力）让每个位置根据当前输入，动态决定应从允许访问的位置读取多少信息。当前项目面向 Decoder-only Language Model（仅解码器语言模型），因此实现的是 Causal Self-Attention（因果自注意力）：位置只能读取自己和过去，不能读取未来。

## 2. 为什么叫 Self-Attention

Query（查询）、Key（键）和 Value（值）都由同一个输入 `X` 经过不同的可学习线性变换得到：

```text
Q = XWq
K = XWk
V = XWv
```

因为三者来源都是当前序列自身，所以叫 Self-Attention。若 Query 来自另一个序列，才属于 Cross-Attention（交叉注意力）；当前未实现 Cross-Attention。

## 3. 输入 [B,T,C]

当前实现只接受三维输入：

```text
English Term ｜中文术语          ｜中文解释
Batch        ｜批次              ｜B，一次共同计算的样本数量
Sequence     ｜序列              ｜T，每个样本中的位置数量
Hidden Dimension｜隐藏维度       ｜C，每个位置的连续特征数量
```

因此输入 `X` 的 shape（形状）是 `[B,T,C]`。单条 `[T,C]` 序列需要先执行 `unsqueeze(0)`，变为 `[1,T,C]`。

## 4. Q / K / V 是什么

可以先用大白话理解：

```text
English Term ｜中文术语          ｜帮助理解的比喻
Query        ｜查询              ｜当前位置想找什么信息
Key          ｜键                ｜每个位置有什么信息可以被匹配
Value        ｜值                ｜匹配后真正被加权读取的内容
```

这个说法只是帮助理解的比喻。实现中，Q、K、V 都是输入向量经过三个不同参数矩阵得到的连续向量：

```text
X [B,T,C] → Wq [C,C] → Q [B,T,C]
X [B,T,C] → Wk [C,C] → K [B,T,C]
X [B,T,C] → Wv [C,C] → V [B,T,C]
```

`Wq`、`Wk`、`Wv` 是模型真正学习的 Parameter（参数）。

## 5. QKᵀ 为什么得到 [B,T,T]

为了让每个 Query 位置与所有 Key 位置计算点积，需要交换 K 的最后两个维度：

```text
Q                     [B,T,C]
K                     [B,T,C]
K.transpose(-2, -1)   [B,C,T]

[B,T,C] @ [B,C,T] = [B,T,T]
```

结果 `scores[b, i, j]` 表示：第 `b` 个样本中，第 `i` 个位置的 Query 与第 `j` 个位置的 Key 的匹配分数。

## 6. 为什么除以 √C

当 `C` 较大时，Q 与 K 的点积数值容易变大，使 Softmax 输出过于尖锐，较小的分数差异也可能变成极端权重。因此 Scaled Dot-Product Attention（缩放点积注意力）使用：

```text
scores = QKᵀ / √d_k
```

当前只有一个 Head（头），且 `d_k = C`，所以实现直接除以 `√C`。未来多头结构会使用每个 Head 自己的维度，但不属于当前实现。

## 7. Causal Mask 为什么不能看未来

语言模型要根据已有内容预测下一个 Token。以 `hello` 为例，模型处理 `hel` 时不能偷看未来的 `lo`，否则训练会发生 Information Leakage（信息泄漏），无法学习真实的逐位置预测过程。

当 `T = 4` 时，可见关系如下：

```text
      K0 K1 K2 K3
Q0    ✓  ✗  ✗  ✗
Q1    ✓  ✓  ✗  ✗
Q2    ✓  ✓  ✓  ✗
Q3    ✓  ✓  ✓  ✓
```

实现使用严格上三角布尔矩阵遮住 `j > i` 的位置。未来位置的 score 在 Softmax 之前被替换为 `-inf`：

```text
softmax(-inf) → 0
```

因此未来位置的 Attention Weight（注意力权重）为零。不能先做 Softmax 再简单乘零，因为那会破坏每行已经完成的概率归一化。

## 8. Softmax 在做什么

`torch.softmax(scores, dim=-1)` 沿最后一个维度，也就是所有 Key 位置执行归一化。对每个 Query 位置：

```text
attention_weights[b, i, :]
```

是在其所有可见 Key 位置上的概率分布，每行总和约为 `1`，未来位置的权重为 `0`。权重越大，代表这个 Query 位置在当前 forward 中从对应 Value 位置读取的信息越多。

## 9. Attention Weights @ V 如何产生 Context

```text
Attention Weights [B,T,T]
Value             [B,T,C]

[B,T,T] @ [B,T,C] = [B,T,C]
```

对位置 `i` 来说，这一步按照 `attention_weights[b, i, :]` 对所有可见位置的 Value 向量加权求和，得到 Context Representation（上下文表示）。输出仍为 `[B,T,C]`，但每个位置现在聚合了允许读取的上下文信息。

## 10. 完整 Shape 流程

```text
X
[B,T,C]

↓ Wq / Wk / Wv

Q K V
[B,T,C]

K.transpose(-2,-1)
[B,C,T]

Q @ Kᵀ
[B,T,T]

↓ scale: ÷ √C
[B,T,T]

↓ causal mask
[B,T,T]

↓ softmax(dim=-1)
[B,T,T]

Attention Weights @ V
[B,T,T] @ [B,T,C]

↓

Context
[B,T,C]
```

## 11. 一个 T=3 的直观例子

设有位置 `0、1、2`：

- 位置 0 只能读取位置 0，因此第一行权重是 `[1, 0, 0]`。
- 位置 1 可以在位置 0 和 1 之间分配权重，位置 2 的权重是 0。
- 位置 2 可以读取位置 0、1、2，其三个可见权重总和约为 1。

如果只修改位置 2 的输入，位置 0 和位置 1 的 Context 不会改变；这正是因果性测试验证的行为。

## 12. Attention Weight 不是记忆

Attention Weight：

```text
不是数据库查询
不是永久记忆
不是把信息存进模型
```

它只是一次 forward 中，根据当前输入动态算出的中间结果。换一份输入，Attention Weight 通常也会改变。模型通过训练长期学习的是 `Wq`、`Wk`、`Wv` 参数，而不是某次 forward 产生的权重矩阵。

## 13. 梯度如何经过 Q/K/V Projection

Context 由 Q、K、V 共同参与计算，PyTorch Autograd（自动求导）会沿这些运算记录反向路径。调用 `loss.backward()` 后，`Wq`、`Wk`、`Wv` 获得或累积梯度。

`backward` 只计算或累积梯度，不更新参数。只有未来引入 Optimizer（优化器）并执行更新步骤后，参数值才会根据梯度改变。

## 14. 当前实现的 API 与限制

```python
attention = SingleHeadSelfAttention(embedding_dim=8)
context, attention_weights = attention(x)
```

返回 Attention Weight 便于当前阶段直接观察和测试数学行为。当前实现只包含单头、因果 mask 和三个无 bias 的 Q/K/V 投影；没有 Multi-Head Attention、Output Projection、Dropout、Cross-Attention、KV Cache、Residual Connection、Normalization、MLP 或 Transformer Block。

Multi-Head Attention 能让不同 Head 在不同表示子空间中学习关系，是后续可能研究的方向；具体设计与实现顺序尚未确定。
