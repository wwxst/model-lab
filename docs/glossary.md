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
Scalar               ｜标量              ｜没有维度的单个数值
Vector               ｜向量              ｜沿一个维度排列的一组数值
Matrix               ｜矩阵              ｜沿两个维度排列的数值
Dimension            ｜维度              ｜张量中一个方向或轴，例如 Batch 维或 Sequence 维
Axis                 ｜轴                ｜在代码中按编号指定的某个维度
Shape                ｜形状              ｜张量每个维度大小组成的描述
Data Type            ｜数据类型          ｜张量中每个元素采用的数值类型，代码中常写作 dtype
Indexing             ｜索引              ｜按位置选择张量中的元素或区域
Slicing              ｜切片              ｜按范围截取张量中的一段数据
Reshape              ｜形状变换          ｜不改变元素值和数量，改变看待张量的形状
View                 ｜视图              ｜以兼容的连续内存布局查看同一组元素的新形状
Transpose            ｜维度交换          ｜交换张量中的两个维度
Permute              ｜维度重排          ｜按指定顺序重新排列张量的多个维度
Unsqueeze            ｜增加维度          ｜在指定位置增加一个大小为 1 的维度
Squeeze              ｜移除维度          ｜移除指定位置上大小为 1 的维度
Broadcasting         ｜广播              ｜让兼容的较小张量自动参与较大形状的运算
Element-wise Operation｜逐元素运算       ｜对输入张量相同位置的元素分别执行计算
Matrix Multiplication｜矩阵乘法          ｜沿匹配的内部维度执行乘法并求和
Autograd             ｜自动求导          ｜记录计算并在反向传播时自动计算梯度的机制
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
Dataset              ｜数据集            ｜按索引提供训练样本的数据集合
Sample               ｜样本              ｜一次训练所使用的一对输入和目标数据
Context Length       ｜上下文长度        ｜单个样本 input 包含的离散元素数量
Input                ｜输入              ｜模型在当前位置看到的离散序列
Target               ｜目标              ｜每个输入位置对应的下一元素序列
Next-token Prediction｜下一位置预测      ｜根据当前位置学习预测下一个离散元素
Training Set         ｜训练集            ｜用于学习模型参数的数据部分
Validation Set       ｜验证集            ｜用于检查模型表现的数据部分
Sliding Window       ｜滑动窗口          ｜沿序列逐位置移动的固定长度取样范围
```
