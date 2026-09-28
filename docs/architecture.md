# Architecture

本文描述第一阶段的目标架构与当前实现边界。以下模型与训练链路是目标架构，尚未实现的部分不能视为当前能力。

## 最小文本语言模型目标

```text
Raw Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Embedding
    ↓
Decoder Transformer
    ↓
Logits
    ↓
Next Token Prediction
```

目标是将 Raw Text（原始文本）转换为 Token IDs（词元编号），通过 Embedding（嵌入）和 Decoder Transformer（仅解码器 Transformer）计算 Logits（未归一化分数），最终预测下一个 Token（词元）。

## 训练主链路目标

```text
Text
  ↓
Tokenizer
  ↓
Input / Target
  ↓
Model
  ↓
Logits
  ↓
Cross Entropy Loss
  ↓
Backward
  ↓
Optimizer
  ↓
Parameter Update
```

训练时，文本经过 Tokenizer 后形成 Input（输入）与 Target（目标）。Model（模型）输出 Logits，通过 Cross Entropy Loss（交叉熵损失）衡量预测误差；Backward（反向传播）计算梯度，Optimizer（优化器）据此完成 Parameter Update（参数更新）。

## 当前边界

当前仓库已实现并测试 Tensor 教学代码、顺序文本窗口 Dataset、字符级 Tokenizer、Token 与 Position Embedding、Single-Head 与 Multi-Head Causal Self-Attention、Feed-Forward Network、Residual Connection 和手写 Layer Normalization。这些模型组件已经按 Pre-Norm 顺序组合成单个 Decoder Transformer Block：

```text
X → LayerNorm → Causal Attention → Residual
  → LayerNorm → Feed-Forward     → Residual → Output
```

多个 Block 已通过 `DecoderStack` 顺序连接。每层结构相同但参数独立，前一层的 `[B,T,C]` 输出直接成为后一层输入，整个 Stack 仍保持相同 shape 和因果行为。

当前尚未把 Token/Position Embedding 与 Decoder Stack 组合，也未实现 Language Model Head、完整模型、Loss、Optimizer 驱动的训练循环、Checkpoint 或推理生成流程；这些未来能力没有占位模块。
