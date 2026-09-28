# Feed-Forward Network｜前馈网络

`FeedForwardNetwork`（前馈网络）是 Transformer 中负责逐 Token 变换特征的
MLP（多层感知机）。Attention（注意力）在不同 Token 之间汇聚信息；前馈网络
不读取其他位置，只对每个 Token 当前的特征向量执行同一组变换。

```text
English Term        ｜中文术语          ｜中文解释
Feed-Forward Network｜前馈网络          ｜对每个 Token 的特征独立执行两层线性变换的网络
MLP                 ｜多层感知机        ｜由线性层和非线性激活函数组成的前馈结构
Expansion Dimension ｜扩展维度          ｜前馈网络中间层扩展后的特征数量，当前固定为 4C
GELU                ｜高斯误差线性单元  ｜在两层线性投影之间加入非线性表达能力的激活函数
Position-wise       ｜逐位置            ｜对每个序列位置独立应用相同计算，不混合不同 Token
```

## 1. 为什么需要前馈网络

Multi-Head Self-Attention（多头自注意力）根据 Token 之间的关系汇聚上下文，
但 Attention 的 Value 聚合本身主要是加权求和。前馈网络随后对每个 Token 已经
获得的上下文表示进行非线性变换，使模型能够组合和重构特征。

两个组件的职责不同：

```text
Attention       ｜沿 Sequence Dimension 混合不同 Token 的信息
Feed-Forward    ｜沿 Feature Dimension 变换单个 Token 的信息
```

## 2. C → 4C → C

输入 `X` 的形状为 `[B,T,C]`：

```text
B = Batch Size       ｜批次大小
T = Sequence Length  ｜序列长度
C = Embedding Dimension｜嵌入维度
```

第一层线性投影把最后一维从 `C` 扩展到 `4C`：

```text
[B,T,C] → Linear(C, 4C) → [B,T,4C]
```

更宽的中间维度为特征组合提供更大的表示空间。当前实现把 Expansion Factor
（扩展倍数）直接固定为 4，不增加尚无实际需要的配置接口。

第二层再把特征收缩回原来的 `C`：

```text
[B,T,4C] → Linear(4C, C) → [B,T,C]
```

输出与输入 shape 相同，未来可以直接参与 Residual Connection（残差连接）。

## 3. GELU 提供非线性

如果两层 Linear（线性层）之间没有激活函数，它们组合后仍等价于一个线性变换。
GELU（高斯误差线性单元）根据输入大小平滑地调节特征，让前馈网络能够表达非线性
关系：

```text
FeedForward(X) = Linear₂(GELU(Linear₁(X)))
```

当前实现直接使用 PyTorch 的 `nn.GELU` 数值运算，不使用更高级的 Transformer
或语言模型封装。

## 4. 为什么不会混合 Token

`nn.Linear` 只变换输入的最后一维。对于 `[B,T,C]`，它把每个 `[C]` 向量独立
投影为 `[4C]`，不会沿 `T` 维读取相邻位置：

```text
Token 0 [C] ─→ 相同的 FeedForward ─→ Token 0 [C]
Token 1 [C] ─→ 相同的 FeedForward ─→ Token 1 [C]
Token 2 [C] ─→ 相同的 FeedForward ─→ Token 2 [C]
```

因此修改一个 Token 的输入不会改变其他位置的输出。所有位置共享同一组参数，
但每个位置的数值计算彼此独立。

## 5. 完整 shape 流程

```text
X [B,T,C]
↓ Input Projection
Hidden [B,T,4C]
↓ GELU
Activated Hidden [B,T,4C]
↓ Output Projection
Output [B,T,C]
```

## 6. 当前 API

```python
import torch

from model_lab.feed_forward import FeedForwardNetwork

feed_forward = FeedForwardNetwork(embedding_dim=8)
x = torch.randn(2, 4, 8)
output = feed_forward(x)

assert output.shape == (2, 4, 8)
```

构造时要求 `embedding_dim > 0`。输入必须是非空序列 `[B,T,C]`，且最后一维
必须等于 `embedding_dim`。

当前模块只实现前馈网络本身，不包含 Residual Connection、Normalization
（归一化）、Dropout、Transformer Block、训练或生成逻辑。
