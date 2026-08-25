# Model Lab

Model Lab 是一个从零学习、实现和训练生成式模型的研发项目。项目以可读、可解释、可验证的实现为优先，逐步建立对生成式模型核心原理的直接理解。

## 长期目标

项目按以下方向逐步推进：

1. Text Model（文本模型）
2. Image Generation（图像生成）
3. Video Generation（视频生成）

这些方向代表长期路线，不表示后续阶段的技术方案已经确定。

## 当前阶段

当前只开发 Text Model。第一阶段目标是使用 Python 与 PyTorch，从零实现并训练一个 Decoder-only Transformer Language Model（仅解码器 Transformer 语言模型）。

“从零实现”是指模型核心算法由本项目直接编写和验证：

- 不调用 OpenAI、Qwen、DeepSeek 等第三方大模型 API 来实现模型核心。
- 不使用 Hugging Face Transformers 替代核心模型实现。
- 数值计算使用 PyTorch；PyTorch 的具体版本和开发工具链将在后续 Commit 中确定。

本项目不是 Agent、Harness、RAG 或第三方模型 API 集成项目。

## 项目状态

Repository Foundation（仓库基础）。当前 Commit 只建立仓库结构、开发规则和目标文档，尚未实现 Tokenizer（分词器）、Transformer、Attention（注意力）、训练循环或生成逻辑。
