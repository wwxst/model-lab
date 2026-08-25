# Architecture

本文描述第一阶段的目标架构，而不是 Commit 1 已经完成的实现。当前仓库只建立项目基础，以下链路中的组件均尚未实现。

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

## Commit 1 边界

上述内容仅用于明确学习目标和未来数据流。Commit 1 没有实现 Tokenizer、Embedding、Decoder Transformer、Loss、Backward、Optimizer、训练循环或推理流程，也没有为这些内容创建占位模块。
