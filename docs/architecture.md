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

当前仓库具备最小 Python 包、PyTorch CPU 基础运行、CUDA 环境识别、测试和代码质量工具。当前尚未实现 Tensor 教学内容、Tokenizer、Embedding、Decoder Transformer、Loss、Backward、Optimizer、训练循环或推理流程，也没有为这些内容创建占位模块。
