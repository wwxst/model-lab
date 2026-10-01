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

## 第一次运行训练

先确认 Python、PyTorch 和当前计算设备：

```text
.venv\Scripts\python.exe scripts\check_environment.py
```

本地存在 `data/processed/alpaca_zh_chat_1000.txt` 后，执行：

```text
.venv\Scripts\python.exe scripts\train_chat.py
```

默认训练使用前 10,000 个字符、1 个 Epoch（完整遍历一次训练数据）、
64 个字符的 Context Length（上下文长度）和 CPU 或可用的 CUDA 设备。
训练完成后，模型和 Optimizer 状态保存在 `checkpoints/chat-model.pt`。

再次训练时不要重新执行上面的首次训练命令。使用 `--resume` 读取模型已经
学到的参数、Optimizer 状态、固定词表和训练配置，再增加指定数量的 Epoch：

```text
.venv\Scripts\python.exe scripts\train_chat.py --resume checkpoints\chat-model.pt --epochs 5
```

假如 Checkpoint 已经完成 10 个 Epoch，这条命令会继续完成第 11 至第 15 个
Epoch。每完成一个 Epoch，更新后的累计训练状态都会保存回同一个文件。

在本功能加入前生成的旧 Checkpoint 没有保存训练配置。第一次继续旧模型时，
需要明确提供它原来使用的字符数量。例如已有模型使用了前 50,000 个字符：

```text
.venv\Scripts\python.exe scripts\train_chat.py --resume checkpoints\chat-model.pt --max-characters 50000 --epochs 5
```

这次继续完成后，Checkpoint 会写入固定词表、模型结构、文本范围和文本指纹；
以后只需要使用 `--resume` 和本次希望增加的 `--epochs`。

为了避免丢失已有结果，不使用 `--resume` 时，如果输出文件已经存在，脚本会
停止并提示错误。开始另一个全新模型时应指定不同文件名：

```text
.venv\Scripts\python.exe scripts\train_chat.py --checkpoint checkpoints\another-model.pt
```

终端输出中的 `loss` 表示模型预测下一个字符时的平均误差。相同数据和参数下，
经过更多训练后 loss 总体下降，说明模型正在学习训练文本中的字符关系。

确认最小流程能够运行后，可以从零训练完整文本，并保存为另一个模型：

```text
.venv\Scripts\python.exe scripts\train_chat.py --data data\processed\alpaca_zh_chat_full.txt --max-characters 0 --epochs 10 --checkpoint checkpoints\chat-full.pt
```

`--max-characters 0` 表示使用文件中的全部文本。当前字符级模型会为相邻位置
创建训练窗口，在 CPU 上训练完整数据会明显更慢。运行以下命令可查看全部参数：

```text
.venv\Scripts\python.exe scripts\train_chat.py --help
```

## 独立问答实验

仓库提供 `examples/qa_basics.jsonl`，包含五个简单知识点及部分提问写法。
使用 `qa` 模式，从零训练一个单独的实验模型：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\train_chat.py --data-format qa --data examples\qa_basics.jsonl --epochs 200 --learning-rate 0.003 --context-length 32 --batch-size 16 --checkpoint checkpoints\qa-answer-only.pt
```

训练完成后提问：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\ask_chat.py --checkpoint checkpoints\qa-answer-only.pt
```

输入问题后按回车，输入 `/exit` 退出。每道问题独立回答，不保留多轮对话历史。
也可以一次只问一题：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\ask_chat.py --checkpoint checkpoints\qa-answer-only.pt --question "水的化学式是什么"
```

继续训练时使用 `--resume checkpoints\qa-answer-only.pt --epochs 20`。
已保存词表和模型配置的普通文本模型也可以通过同一个提问脚本加载，
恢复训练时仍保持原来的数据模式。
样本格式、回答损失和验收边界见 [`docs/question-answer-training.md`](docs/question-answer-training.md)。

## 换一种问法的实验

`examples/qa_paraphrases_train.jsonl` 为同一知识点提供不同表达，并加入没有明确
国家时请求补充信息的首都问题。`examples/qa_paraphrases_test.jsonl` 是独立的
未见问法测试集，不能用于这次实验的训练。

从零训练一个单独的模型：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\train_chat.py --data-format qa --data examples\qa_paraphrases_train.jsonl --epochs 200 --learning-rate 0.003 --context-length 32 --batch-size 16 --checkpoint checkpoints\qa-paraphrases.pt
```

训练后打开提问界面：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\ask_chat.py --checkpoint checkpoints\qa-paraphrases.pt
```

先检查训练文件中的问题，再检查测试文件中不同写法的问题，分别记录正确数量。
答案判定采用与预期答案完全一致的标准，不能只检查生成内容是否包含几个关键词。
这个实验只覆盖已有知识点的有限表达，不代表通用聊天能力。

## 先读文本，再用问答微调

`--finetune` 读取已有模型参数和固定词表，用新数据开始训练。它与 `--resume`
不同：微调会新建 Optimizer，轮次从 1 开始；恢复训练则沿用原数据和训练状态。
微调必须明确指定新数据和新的输出文件，不会覆盖原模型或其他已有文件。

先用仓库中的小文本演示 Pretraining（预训练）：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\train_chat.py --data examples\pretraining_demo.txt --max-characters 0 --epochs 3 --context-length 32 --checkpoint checkpoints\pretrained-demo.pt
```

然后继承这个模型，在问答数据上进行 Fine-tuning（微调）：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\train_chat.py --finetune checkpoints\pretrained-demo.pt --data-format qa --data examples\qa_basics.jsonl --epochs 200 --learning-rate 0.003 --batch-size 16 --checkpoint checkpoints\qa-finetuned-demo.pt
```

训练后提问：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\ask_chat.py --checkpoint checkpoints\qa-finetuned-demo.pt
```

中断微调后用 `--resume checkpoints\qa-finetuned-demo.pt --epochs 20` 接着训练，
不要再次执行 `--finetune`，否则会重新从原模型开始。
这组小数据仅演示两阶段训练流程，不代表预训练改善了问答能力或具备通用聊天能力。
当前微调保持原词表和模型结构，不支持新字符；新数据的字符必须已被原词表覆盖。
数据准备、参数继承和验收边界见
[`docs/question-answer-training.md`](docs/question-answer-training.md)。

## 基础知识学习材料

[`examples/foundation/`](examples/foundation/README.md) 提供约10万字符的预训练
文本、200条问答训练题和50条独立测试题，覆盖50个基础知识点。
来源、许可、分集方式和离线检查命令见数据包说明。测试题只用于检查结果，
不能放入训练数据；这个数据包本身不代表模型已经学会这些知识。

## 项目状态

Autoregressive Generation（自回归生成）。当前已能从 `[B,T]` Token IDs 开始，以 Greedy Decoding（贪心解码）逐步追加 Token，并在超过模型上下文时裁剪输入窗口。详细内容见 [`docs/autoregressive-generation.md`](docs/autoregressive-generation.md)。当前不包含采样策略、KV Cache 或流式输出。
