# Multi-Head Self-Attention｜多头自注意力

`MultiHeadSelfAttention`（多头自注意力）把同一份输入特征拆给多个 Attention Head（注意力头），让各 Head 并行计算自己的因果注意力，再把结果拼回完整的 Embedding Dimension（嵌入维度）。当前实现直接展示多头注意力的核心数学步骤，不依赖 PyTorch 的高级注意力模块。

```text
English Term         ｜中文术语          ｜中文解释
Multi-Head Attention ｜多头注意力        ｜并行使用多个 Head 计算注意力，再合并各 Head 的结果
Attention Head       ｜注意力头          ｜在部分特征维度上独立计算注意力的一条分支
Head Dimension       ｜头维度            ｜单个 Head 分到的特征数量，记作 D
Split Heads          ｜拆分多头          ｜把完整特征维 C 拆成 H 个大小为 D 的 Head
Concat Heads         ｜拼接多头          ｜把 H 个 Head 的结果重新拼接为完整特征维 C
Output Projection    ｜输出投影          ｜在拼接后混合不同 Head 信息的线性变换
```

## 1. Multi-Head 相比 Single-Head 多了什么

Single-Head Self-Attention（单头自注意力）使用完整的 `C` 维特征计算一张 `[B,T,T]` 注意力权重矩阵。Multi-Head Self-Attention 先把完整特征维分成多个较小的 Head：

```text
C = H × D

C = Embedding Dimension ｜嵌入维度
H = Number of Heads      ｜Head 数量
D = Head Dimension       ｜单个 Head 的维度
```

每个 Head 都独立产生一张 `[T,T]` 的注意力权重矩阵。把 Batch（批次）也计算在内，最终权重形状是 `[B,H,T,T]`，而不是单头实现的 `[B,T,T]`。

例如 `C = 8`、`H = 2` 时：

```text
D = C / H = 8 / 2 = 4
```

每个 Head 使用 4 个特征计算注意力。因为特征必须被均匀拆分，所以 `embedding_dim` 必须能被 `num_heads` 整除。

## 2. Q、K、V Projection

输入 `X` 的形状是 `[B,T,C]`。Query、Key、Value 使用三个独立的无偏置线性层：

```python
query_projection = nn.Linear(C, C, bias=False)
key_projection = nn.Linear(C, C, bias=False)
value_projection = nn.Linear(C, C, bias=False)
```

线性层只变换最后一个维度，所以投影之后：

```text
X [B,T,C]
├─ Query Projection ─→ Q [B,T,C]
├─ Key Projection   ─→ K [B,T,C]
└─ Value Projection ─→ V [B,T,C]
```

Q、K、V 都来自同一个输入 `X`，因此这是 Self-Attention（自注意力）。三个投影的参数彼此独立，使它们可以学习不同的匹配和聚合方式。

## 3. Split Heads

以 Q 为例，先把完整特征维 `C` 拆成 `H × D`：

```text
[B,T,C]
   ↓ reshape，C = H × D
[B,T,H,D]
   ↓ transpose(1, 2)
[B,H,T,D]
```

`reshape` 只是重新解释相同数量的元素；`transpose` 把 Head 维移动到 Sequence（序列）维之前。得到 `[B,H,T,D]` 后，PyTorch 可以把 `B` 和 `H` 都当作并行维度，在一次矩阵乘法中计算所有 Head，不需要 Python `for` 循环。

## 4. 每个 Head 独立计算 Attention

对每个 Head，Query 与转置后的 Key 做矩阵乘法：

```text
Q   [B,H,T,D]
Kᵀ  [B,H,D,T]
────────────────
QKᵀ [B,H,T,T]
```

结果的最后两个 `T` 分别表示 Query 位置和 Key 位置。因此 `scores[b,h,i,j]` 表示第 `b` 个样本、第 `h` 个 Head 中，位置 `i` 的 Query 与位置 `j` 的 Key 的匹配程度。

不同 Head 使用 Q、K、V 投影结果中的不同特征片段。训练时，这些片段可以形成不同的注意力分布，例如一个 Head 更关注临近 Token，另一个 Head 更关注更早的位置。当前 API 返回完整的 `[B,H,T,T]` 权重，因此可以直接观察这种差异。

## 5. 为什么 scale 使用 sqrt(D)

单个 Head 的点积沿 `D` 个特征求和，所以缩放公式是：

```text
Scaled Scores = QKᵀ / sqrt(D)
```

不能使用 `sqrt(C)`。虽然投影前的完整向量有 `C` 个特征，但每个 Head 实际只用 `D` 个特征计算点积。除以 `sqrt(D)` 可以在 `D` 增大时控制分数幅度，避免 Softmax 过早集中到极少数位置。

## 6. Causal Mask 如何广播

Causal Mask（因果掩码）只创建一次，形状为 `[T,T]`：

```text
可见位置为 False，未来位置为 True

[[False, True,  True ],
 [False, False, True ],
 [False, False, False]]
```

注意力分数是 `[B,H,T,T]`。`masked_fill` 会把 `[T,T]` 掩码沿 Batch 和 Head 两个维度广播，相当于让每个样本的每个 Head 都使用同一条因果规则：

```text
mask [T,T]
  ↓ broadcast
[B,H,T,T]
```

未来位置在 Softmax 之前被填为负无穷，因此对应权重为 0。随后 `softmax(dim=-1)` 沿每个 Query 的全部 Key 位置归一化，使每个 Head 的每一行权重总和约为 1。

## 7. 聚合 Value 与 Concat Heads

每个 Head 使用自己的注意力权重聚合 Value：

```text
Attention Weights [B,H,T,T]
Value             [B,H,T,D]
───────────────────────────
Head Context      [B,H,T,D]
```

然后把 Head 结果拼回完整嵌入维度：

```text
[B,H,T,D]
   ↓ transpose(1, 2)
[B,T,H,D]
   ↓ reshape，H × D = C
[B,T,C]
```

`transpose` 先让同一个 Token 的各 Head 回到相邻位置，`reshape` 再把 `H` 和 `D` 合并为 `C`。这一步称为 Concat Heads（拼接多头）。

## 8. 为什么需要 Output Projection

拼接后的 `[B,T,C]` 仍按 Head 分段排列。Output Projection（输出投影）使用一个无偏置的 `C → C` 线性层混合这些分段：

```python
output_projection = nn.Linear(C, C, bias=False)
```

它让最终输出的每个特征都能组合不同 Head 提取的信息，同时不改变 shape。输出仍然是 `[B,T,C]`，可以继续作为后续模型计算的输入。

## 9. 完整 shape 流程

```text
X [B,T,C]
↓ Q/K/V Projection
Q, K, V [B,T,C]
↓ reshape
[B,T,H,D]
↓ transpose
[B,H,T,D]
↓ Q @ Kᵀ
Scores [B,H,T,T]
↓ / sqrt(D)
Scaled Scores [B,H,T,T]
↓ causal mask [T,T] 广播到 [B,H,T,T]
Masked Scores [B,H,T,T]
↓ softmax(dim=-1)
Attention Weights [B,H,T,T]
↓ Attention Weights @ V
Head Context [B,H,T,D]
↓ transpose + reshape
Concatenated Context [B,T,C]
↓ Output Projection
Context [B,T,C]
```

## 10. 当前 API

```python
import torch

from model_lab.multi_head_attention import MultiHeadSelfAttention

attention = MultiHeadSelfAttention(embedding_dim=8, num_heads=2)
x = torch.randn(2, 4, 8)

context, attention_weights = attention(x)

assert context.shape == (2, 4, 8)
assert attention_weights.shape == (2, 2, 4, 4)
```

构造时要求 `embedding_dim > 0`、`num_heads > 0`，且 `embedding_dim` 能被 `num_heads` 整除。输入必须是非空序列 `[B,T,C]`，并且最后一维与构造时的 `embedding_dim` 一致。

当前模块只负责多头因果自注意力本身，不包含 Dropout、Residual、Normalization、MLP、位置旋转、KV Cache、训练或生成逻辑。
