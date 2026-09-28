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

Language Model Head 能把 Decoder Stack 的 `[B,T,C]` Hidden State 线性投影为 `[B,T,V]` Logits，不执行 Softmax。Token/Position Embedding、Decoder Stack、Final Layer Normalization 与 Language Model Head 已组合成完整模型前向链路：

```text
Token IDs [B,T]
→ Token Embedding + Position Embedding [B,T,C]
→ Decoder Stack [B,T,C]
→ Final Layer Normalization [B,T,C]
→ Language Model Head [B,T,V]
→ Logits
```

Cross Entropy Loss 能使用模型的 `[B,T,V]` Logits 和 Dataset 的 `[B,T]` Target IDs 计算标量平均损失，并通过 Autograd 将梯度传回全部模型参数。`train_epoch` 已将 DataLoader、模型、Loss、Backward 和调用者提供的 PyTorch Optimizer 连接为可执行训练链路：

```text
Batch Input / Target
→ zero_grad
→ Model Forward
→ Cross Entropy Loss
→ Backward
→ Optimizer Step
→ Parameter Update
```

当前尚未实现 Checkpoint 或推理生成流程；这些未来能力没有占位模块。
Checkpoint 已能保存和恢复模型参数、Optimizer 状态和已完成 Epoch 编号。恢复时由调用者先创建结构相同的模型和 Optimizer，再加载状态。

当前尚未实现自动保存策略或推理生成流程；这些未来能力没有占位模块。
