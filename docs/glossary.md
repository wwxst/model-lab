# Glossary

本术语表面向第一次学习大模型的读者。英文术语保留模型研发中的常用写法，中文解释优先说明它在本项目中的实际含义。

```text
English Term         ｜中文术语          ｜中文解释
Model                ｜模型              ｜根据输入计算输出的一组结构和参数
Parameter            ｜参数              ｜模型在训练中通过梯度更新的数值
Token                ｜词元              ｜模型实际读取和预测的离散文本单位
Vocabulary           ｜词表              ｜模型能够识别的全部 Token 集合
Tokenizer            ｜分词器            ｜在文本与 Token ID 序列之间进行转换的规则或程序
Embedding            ｜嵌入              ｜将离散 Token ID 映射为连续向量表示
Tensor               ｜张量              ｜用于保存和计算多维数值数据的基本结构
Dimension            ｜维度              ｜张量中一个方向或轴，例如 Batch 维或 Sequence 维
Shape                ｜形状              ｜张量每个维度大小组成的描述
Batch                ｜批次              ｜一次共同参与计算的一组样本
Sequence             ｜序列              ｜按顺序排列的一串 Token
Logits               ｜未归一化分数      ｜模型为各候选 Token 输出的原始分数
Probability          ｜概率              ｜归一化后表示各候选结果可能性的数值
Loss                 ｜损失              ｜衡量模型预测与正确目标之间差距的数值
Gradient             ｜梯度              ｜表示参数变化会如何影响损失的数值
Optimizer            ｜优化器            ｜根据梯度更新模型参数的算法
Checkpoint           ｜检查点            ｜训练过程中保存的模型参数及相关状态
Training             ｜训练              ｜使用数据和损失反复更新模型参数的过程
Inference            ｜推理              ｜使用训练后的模型计算预测结果的过程
Transformer          ｜Transformer 架构  ｜通过注意力等结构处理序列的一类神经网络架构
Attention            ｜注意力            ｜计算序列中不同 Token 之间信息关系的机制
Python               ｜Python            ｜运行项目代码的编程语言和执行环境
PyTorch              ｜PyTorch           ｜提供张量、自动求导和 CPU/GPU 数值计算的库
Runtime              ｜运行时            ｜程序实际执行时所使用的软件环境
Dependency           ｜依赖              ｜项目运行或开发所需要安装的其他软件包
Virtual Environment  ｜虚拟环境          ｜隔离单个项目依赖和解释器工具的目录
CPU                  ｜中央处理器        ｜擅长通用计算的处理器
GPU                  ｜图形处理器        ｜适合大量并行数值计算的处理器
CUDA                 ｜CUDA              ｜让 PyTorch 使用 NVIDIA GPU 计算的软件平台
Device               ｜设备              ｜执行张量计算的 CPU 或 GPU
```
