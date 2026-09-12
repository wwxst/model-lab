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
- 数值计算使用 PyTorch 2.13.x；Python 和开发工具链的稳定版本由当前基线固定。

本项目不是 Agent、Harness、RAG 或第三方模型 API 集成项目。

## 开发环境基础

Python 是运行项目代码的编程语言和执行环境。当前项目要求 Python `>=3.14,<3.15`，并使用 `.venv` 隔离项目依赖。

PyTorch 是数值计算和自动求导库。在本项目中，它提供 Tensor（张量）、梯度计算以及 CPU/GPU 数值计算能力；它不提供已经训练好的本项目模型。

CPU 擅长通用、顺序性较强的计算；GPU 包含更多适合并行数值计算的处理单元。CUDA 是 PyTorch 在 NVIDIA GPU 上执行计算时使用的软件平台。没有 CUDA 或 GPU 时，项目仍然可以使用 CPU 完成基础验证。

我们没有在本项目中自己实现 GPU 矩阵计算，因为那属于底层数值计算和硬件驱动基础，不是当前学习模型算法的目标。使用 PyTorch 仍然属于从零实现模型：我们自己编写模型算法、结构和训练逻辑，只把 Tensor、自动求导和 CPU/GPU 计算交给 PyTorch；这不同于调用别人已经训练好的大模型或 API。

安装开发环境：

```text
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 项目状态

Single-Head Self-Attention（单头自注意力）。当前已具备 Python 与 PyTorch 环境基础、可执行且可测试的 Tensor 教学代码、顺序文本窗口 Dataset、字符级文本到 Token ID 的可逆转换、Token 与 Position Embedding（位置嵌入），以及可观察注意力权重的单头因果自注意力。详细内容见 [`docs/single-head-self-attention.md`](docs/single-head-self-attention.md)。Multi-Head Attention（多头注意力）、完整 Transformer、训练循环和生成逻辑尚未实现。
