# Roadmap

本路线图描述 Model Lab 的学习与研发方向。除当前明确列出的近期 Commit 外，后续内容只是方向，不代表技术设计已经完全确定。

## Phase 1: Tiny Text Language Model

目标：从基础张量操作开始，逐步实现并训练一个小型 Decoder-only Transformer Language Model（仅解码器 Transformer 语言模型）。

```text
Commit 1  Repository Foundation
Commit 2  Python & PyTorch Foundation
Commit 3  Tensor Fundamentals
Commit 4  Text Dataset
Commit 5  Character Tokenizer
Commit 6  Token Embedding
Commit 7  Position Representation
Commit 8  Single-Head Self-Attention
Commit 9  Multi-Head Self-Attention
Commit 10 MLP / Feed-Forward Network
Commit 11 Residual Connection + Normalization
Commit 12 Transformer Block
Commit 13 Decoder Stack
```

后续计划按以下顺序逐步组合完整的 Decoder-only Transformer，建立训练与推理能力。以下项目仍是未来方向，不代表当前已经实现；每个 Commit 的具体模块边界和验收标准仍在进入该 Commit 时确定。

```text
Commit 14 Language Model Head
Commit 15 Complete Decoder-only Model
Commit 16 Cross Entropy Loss
Commit 17 Training Loop
Commit 18 Checkpoint
Commit 19 Autoregressive Generation
```

## Phase 2: Modern LLM Architecture

方向：在掌握最小文本语言模型后，研究现代大语言模型常见的架构选择与工程取舍。具体范围由 Phase 1 的结果决定。

## Phase 3: Image Generation Foundations

方向：学习图像生成模型所需的基础表示、训练目标与生成过程。当前不选择具体架构，也不实现图像生成。

## Phase 4: Video Generation Foundations

方向：在文本与图像生成基础之上，研究视频生成所需的时空表示与训练问题。当前不选择具体架构，也不实现视频生成。

## Current Status

当前阶段为 Decoder Stack（解码器堆叠）。仓库已能注册并顺序执行多个具有独立参数的 Pre-Norm Decoder Transformer Block，同时保持 `[B,T,C]` shape 和因果行为；尚未实现 Language Model Head、完整模型、训练或生成能力。
