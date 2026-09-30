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

Checkpoint 已能保存和恢复模型参数、Optimizer 状态和已完成 Epoch 编号。恢复时由调用者先创建结构相同的模型和 Optimizer，再加载状态。

Autoregressive Generation 已能在 `torch.no_grad()` 下逐步读取最后位置 Logits，使用 Greedy `argmax` 追加 Token，并在每步将上下文裁剪到模型最大长度。Checkpoint、训练和生成均为独立职责，生成不修改模型参数。

训练入口同时支持连续文本模式和独立问答模式。问答模式由
`QuestionAnswerDataset` 将各条问题/回答分别展开为回答前缀预测样本，批处理在
右侧补齐；Cross Entropy 与 Epoch 平均值忽略目标为 `-100` 的位置。
训练入口将数据模式、固定词表、模型配置和文本指纹写入 Checkpoint 元数据。
终端提问入口读取保存的词表和配置，调用现有模型与贪心生成函数。
问答样本的数学过程见 [`question-answer-training.md`](question-answer-training.md)。

训练初始化有三个入口：新模型使用随机参数；`--resume` 恢复模型、Optimizer 和
轮次并核对原数据指纹；`--finetune` 沿用保存的模型结构和词表，通过
`load_model_weights` 只加载模型参数，再创建新的 Optimizer，用新数据从第 1 轮
开始训练。微调结果写入新 Checkpoint，记录当前数据配置，可独立恢复训练。
