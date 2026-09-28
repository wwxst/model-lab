# Decoder Stack｜解码器堆叠

`DecoderStack`（解码器堆叠）把多个 `TransformerBlock` 按深度顺序连接起来。
每个 Block 接收上一层产生的 Hidden State（隐藏状态），继续结合上下文并变换
特征，然后把结果交给下一层。

```text
English Term       ｜中文术语          ｜中文解释
Decoder Stack      ｜解码器堆叠        ｜按顺序连接多个 Decoder Transformer Block 的结构
Layer              ｜层                ｜Stack 中一个具有独立参数的 Transformer Block
Depth              ｜深度              ｜Stack 包含的 Transformer Block 数量
ModuleList         ｜模块列表          ｜让 PyTorch 注册并管理一组有顺序的子模块
Sequential Execution｜顺序执行        ｜让前一层输出成为后一层输入的计算方式
```

## 1. 从一个 Block 到多个 Layer

单个 `TransformerBlock` 已经包含一次 Attention 和一次 Feed-Forward 计算。
Decoder Stack 把这个结构重复 `N` 次：

```text
Input X₀ [B,T,C]
↓ Transformer Block 0
Hidden State X₁ [B,T,C]
↓ Transformer Block 1
Hidden State X₂ [B,T,C]
↓ ...
↓ Transformer Block N-1
Output Xₙ [B,T,C]
```

第 `l` 层的计算可以写成：

```text
Xₗ₊₁ = Blockₗ(Xₗ)
```

每一层具有相同的结构，但参数彼此独立。它们不会共享 Attention 投影、
Feed-Forward 权重或 Layer Normalization 的 scale/shift。

## 2. 为什么使用 ModuleList

普通 Python `list` 可以保存对象，但 PyTorch 不会自动把其中的 Module 识别为
模型子模块。`nn.ModuleList` 同时保留顺序并注册内部 Block，使这些参数能够：

```text
- 出现在 model.parameters() 中
- 随模型移动到 CPU 或 GPU
- 参与 dtype 转换
- 保存进 state_dict
- 在 backward 时接收梯度
```

当前实现仍使用清晰的 Python `for` 循环逐层执行，没有增加通用容器或执行框架。

## 3. Depth 表示什么

`num_layers` 是 Stack 的 Depth（深度）。例如 `num_layers=3` 表示隐藏状态依次
经过三个不同的 Transformer Block。更多层意味着进行更多次上下文聚合和特征
变换，但当前 Commit 不讨论更深模型的性能或优化策略。

至少需要一个 Block，所以 `num_layers` 必须大于 0。空 Stack 没有真实的 Decoder
计算职责，因此当前不支持。

## 4. Shape 为什么保持不变

每个 Transformer Block 都保持 `[B,T,C]`，所以任意层数的 Stack 也保持相同
shape：

```text
[B,T,C]
↓ Block 0
[B,T,C]
↓ Block 1
[B,T,C]
↓ Block 2
[B,T,C]
```

Stack 增加的是计算深度，不增加或删除 Batch、Sequence、Embedding 维度。

## 5. 多层之后仍保持因果性

每个 Block 中的 Self-Attention 都使用 Causal Mask（因果掩码），Layer
Normalization 和 Feed-Forward 又只在单个 Token 内计算。因此每层的当前位置都
只能依赖自己和过去位置。多个因果 Block 顺序组合后，未来 Token 仍不能影响较早
位置的输出。

## 6. 完整执行过程

```python
for block in self.blocks:
    x = block(x)
return x
```

这段循环直接表达数据流：变量 `x` 始终保存当前深度的 Hidden State。

## 7. 当前 API

```python
import torch

from model_lab.decoder_stack import DecoderStack

stack = DecoderStack(
    embedding_dim=8,
    num_heads=2,
    num_layers=3,
)
x = torch.randn(2, 4, 8)
output = stack(x)

assert output.shape == (2, 4, 8)
```

当前 Stack 只负责顺序执行 Transformer Block，不包含 Token/Position Embedding、
最终 Layer Normalization、Language Model Head、Loss、训练或生成逻辑。
