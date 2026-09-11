# AGENTS.md

本文件定义 Model Lab 仓库的强制开发规则。仓库中的人工开发者和自动化编码 Agent 均须遵守。

## 1. Commit-driven Development

- 一个 Commit 只解决一个明确问题，并保持可独立审查。
- 禁止跨 Commit 提前实现路线图中的未来功能。
- 完成当前 Commit 的验收后必须停止，不得自动开始下一个 Commit。

## 2. Review Gate｜审查门禁

### Git Workflow｜Git 工作流

每次开发必须使用独立功能分支，不允许直接修改 `main`。

流程：

1. 从最新 `main` 创建功能分支。
2. 在功能分支完成当前 Commit 范围内的开发。
3. 执行要求的测试和验证。
4. 创建 Commit。
5. push 到远端功能分支。
6. 可以创建 PR，但不得 merge。
7. 输出功能分支名称、Commit 哈希、PR 链接、修改摘要、测试结果和 `git diff --stat`。
8. 完成后立即停止，等待人工 Review。

禁止：

- 自动 merge PR。
- 自动修改 `main`。
- 自动 fast-forward `main`。
- 自动 squash 或 rebase 到 `main`。
- Review 前删除功能分支。

只有收到明确的“Review PASS，可以合并”指令后，才允许将该功能分支合并到 `main`。

合并后必须同步本地 `main` 与 `origin/main`，重新执行完整验证，输出 merge commit，并保持工作树干净。上述同步和验证完成后，删除远端功能分支和本地功能分支。

## 3. Minimal Implementation｜最小实现规范

- 只实现当前 Commit 真正需要的代码，并将修改范围保持在解决当前问题所需的最小范围。
- 不为未来可能的需求提前创建抽象。
- 不创建当前没有真实消费者的接口、状态、配置和扩展点。
- 不创建无用途的空模块、占位类和占位 API。
- 不因为“以后可能会用”增加复杂度。
- 优先减少代码、状态、依赖和维护成本。

除非当前真实需求明确要求，否则禁止提前创建：

```text
Manager
Factory
Provider
Adapter
Wrapper
RuntimeContext
Backend abstraction
Compatibility layer
Legacy implementation
V2 implementation
```

例如当前阶段不要创建 `DeviceManager`、`TensorManager`、`ModelFactory`、`BackendProvider`、`RuntimeContext`、`LegacyTensor` 或 `TensorV2`。

## 4. Learning-first Implementation

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
attention @ V
```

后续 Commit 只有在存在真实需求时，才可以在 Readable Implementation（可读实现）之外增加 Optimized Implementation（优化实现），但不得删除可读实现。优化版本必须有一致性测试，说明它与可读版本的行为关系。

## 5. Code Comment Standard｜代码注释规范

Model Lab 面向第一次系统学习大模型底层源码的读者。未来核心模型代码允许并要求较密集的中文教学注释，接近逐句教学，但不能把代码注释成噪音。

核心注释重点解释：

- Tensor（张量）当前的 shape（形状）以及每个维度的含义。
- 数学运算正在做什么，以及数据经过当前语句前后的变化。
- 为什么需要 `reshape`、`view`、`transpose` 或 `permute`。
- 参数的数学含义以及返回值代表什么。
- mask（掩码）、scale（缩放）和 normalization（归一化）存在的原因。
- 梯度、参数或状态发生变化时的原因。

例如遇到 `[B, T, C]` 时，应明确说明：

```text
B = Batch Size       ｜批次大小
T = Sequence Length  ｜序列长度
C = Hidden Dimension ｜隐藏维度
```

不要机械注释 `import`、括号、简单 `return`、明显赋值或基础 Python 语法。注释必须随代码行为同步更新。

## 6. Tests and Validation Discipline｜测试与验证纪律

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
- 不隐藏真实 error（错误）或 warning（警告）只为了让输出看起来干净。
- 不使用全局 warning suppression（警告抑制）让测试通过，也不使用宽泛 `try/except` 吞掉真实错误。
- 不为了让测试输出漂亮而修改真实行为。
- 测试只验证当前功能真正应该保证的职责。

遇到第三方 warning 时，先判断它是否影响当前功能：不影响时允许 warning 暴露；影响时解决根因。只有存在明确、局部、必要的理由时，才允许窄范围过滤。

## 7. Documentation｜文档规范

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

### Documentation Ownership｜文档职责

一个事实只保留一个权威位置：

```text
README.md            ｜项目是什么、当前阶段、如何开始
AGENTS.md            ｜开发规则、修改约束、验证纪律
docs/architecture.md ｜当前真实架构、当前模块关系、当前数据流
docs/roadmap.md      ｜当前阶段、后续实现顺序
docs/glossary.md     ｜术语定义
```

同一个事实不要在多个文档完整复制维护。其他文档需要提及时，只做简短引用或指向权威文档，避免重复维护造成内容漂移。

### Current-state Documentation｜当前状态文档

长期文档默认描述当前事实，不使用“Commit 1 时”“Commit 2 后”“以前”“后来”或“曾经”等措辞持续记录开发过程。历史信息由 Git history、commit message 或必要的 decision note（决策记录）承担，不把长期文档写成开发日志。

## 8. Pre-release Development Policy｜早期研发政策

当前项目处于 0.x 早期研发学习阶段。如果发现模型设计、命名、目录边界、API 设计或数据结构错误，优先直接修正为正确设计。

不要为了尚不存在的外部用户保留 `LegacyXXX`、`NewXXX`、`XXXV2`、deprecated wrapper（废弃包装层）、compatibility shim（兼容适配层）或“旧路径 + 新路径”双实现。正确地基优先于早期兼容性；历史由 Git history 和 commit message 保存，不需要把错误设计永久留在源码中。

## 9. Defensive Programming｜防御性编程

没有真实边界时，不要新增重复参数校验、多层兜底、fallback、retry、recovery、compensation、宽泛异常捕获或提前的错误包装层。

用户输入、数据一致性、文件或 checkpoint 损坏、数值计算必要约束、第三方明确不可靠边界以及并发问题允许采用必要保护，但每项保护都必须有真实原因。

## 10. Scope

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
