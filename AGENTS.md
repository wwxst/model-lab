# AGENTS.md

本文件定义 Model Lab 仓库的强制开发规则。仓库中的人工开发者和自动化编码 Agent 均须遵守。

## 1. Commit-driven Development

- 一个 Commit 只解决一个明确问题，并保持可独立审查。
- 禁止跨 Commit 提前实现路线图中的未来功能。
- 只创建当前 Commit 真正需要的文件、依赖、接口和抽象。
- 未经当前 Commit 明确要求，不得添加占位模块、空实现或推测性的扩展点。
- 完成当前 Commit 的验收后必须停止，不得自动开始下一个 Commit。

## 2. Learning-first Implementation

核心算法的第一版必须优先清晰、可读，并能够解释对应的数学过程。不得一开始就用高级封装隐藏关键步骤。

例如，未来 Attention（注意力）第一版应明确展示以下过程：

```text
Q
K
V
QK^T
scale
causal mask
softmax
attention × V
```

后续 Commit 可以在 readable implementation（可读实现）之外增加 optimized implementation（优化实现），但不得删除可读实现。优化版本必须有一致性测试，说明它与可读版本的行为关系。

## 3. Tests

- 后续所有核心模型组件都必须具有单元测试。
- 测试不能只验证代码“不报错”。
- 测试应根据组件能力逐步覆盖以下方面：

```text
shape                形状              输出各维度是否符合约定
dtype                数据类型          数值精度和类型是否符合约定
device               设备              CPU 或加速设备上的放置是否正确
numerical behavior   数值行为          给定输入是否得到可解释的数值结果
gradient/backward    梯度与反向传播    梯度能否正确计算并传回参数
causal behavior      因果行为          当前位置是否无法读取未来 Token
```

- 每个 Commit 的测试范围应与该 Commit 引入的行为一致，不得用未来测试推动未来实现进入当前 Commit。

## 4. Documentation

- 重要模型概念必须随对应实现同步进入 `docs/`。
- 英文术语第一次出现时，应尽量同时给出中文术语或中文解释。
- 关键术语使用固定宽度三列 `text` 代码块，格式为 English Term、中文术语、中文解释。

```text
English Term ｜中文术语          ｜中文解释
Token        ｜词元              ｜模型实际处理的离散文本单位
Embedding    ｜嵌入              ｜将离散 Token 映射为连续向量
Attention    ｜注意力            ｜计算序列中不同 Token 之间的信息关系
```

- 文档必须区分“当前已实现”“目标架构”和“未来方向”，不得把尚未实现的内容写成现状。
- 文档应面向第一次学习大模型的读者，准确、直接，避免不必要的学术化表达。

## 5. Scope

当前阶段禁止加入以下内容，除非后续对应 Commit 明确引入：

```text
Agent
Tool Calling
MCP
RAG
Function Calling
Web UI
Electron
多模态输入
图像生成
视频生成
RLHF
DPO
GRPO
MoE
DeepSpeed
Megatron
vLLM
分布式训练
多机多卡
自定义 CUDA Kernel
```

当前文本模型阶段也不得通过第三方大模型 API 或 Hugging Face Transformers 替代核心模型实现。PyTorch 仅作为数值计算基础；具体依赖和工具链由后续 Commit 决定。
