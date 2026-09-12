# Glossary

本术语表面向第一次学习大模型的读者。英文术语保留模型研发中的常用写法，中文解释优先说明它在本项目中的实际含义。

```text
English Term         ｜中文术语          ｜中文解释
Model                ｜模型              ｜根据输入计算输出的一组结构和参数
Parameter            ｜参数              ｜模型在训练中通过梯度更新的数值
Token                ｜词元              ｜模型实际读取和预测的离散文本单位
Vocabulary           ｜词表              ｜模型能够识别的全部 Token 集合
Tokenizer            ｜分词器            ｜在文本与 Token ID 序列之间进行转换的规则或程序
Token ID             ｜词元编号          ｜词表为每个 Token 分配的离散整数编号
Character-level Tokenizer｜字符级分词器   ｜把 Python str 迭代得到的每个 Unicode 码点作为一个 Token 的分词器
Encode               ｜编码              ｜把文本转换为 Token ID 序列的过程
Decode               ｜解码              ｜把 Token ID 序列还原为文本的过程
Vocabulary Size      ｜词表大小          ｜词表中唯一 Token 的数量
Unknown Character    ｜未知字符          ｜当前词表中没有、因而无法编码的字符
Round Trip           ｜往返转换          ｜先编码再解码后恢复原始文本的过程
Embedding            ｜嵌入              ｜将离散 Token ID 映射为连续向量表示
Token Embedding      ｜词元嵌入          ｜为每个 Token ID 查找可学习连续向量的表示层
Embedding Table      ｜嵌入表            ｜按 Token ID 分行保存可学习向量的参数表
Embedding Dimension  ｜嵌入维度          ｜每个 Token 向量包含的连续特征数量
Lookup               ｜查表              ｜使用 Token ID 选取 Embedding Table 对应行的操作
Continuous Representation｜连续表示      ｜用可参与连续数学计算的向量表示离散对象
Feature Dimension    ｜特征维度          ｜向量中用于承载不同连续特征的维度
Position             ｜位置              ｜Token 在序列中所在的绝对顺序位置
Position ID          ｜位置编号          ｜表示序列位置的离散整数编号
Position Embedding   ｜位置嵌入          ｜把位置编号映射为可学习连续向量的表示层
Position Representation｜位置表示       ｜向模型提供序列位置信息的表示方式；当前实现使用可学习绝对位置嵌入
Maximum Sequence Length｜最大序列长度  ｜位置表支持的最大序列长度
Absolute Position    ｜绝对位置          ｜从序列起点编号的固定位置
Learnable Position Embedding｜可学习位置嵌入｜通过梯度学习每个绝对位置向量的位置表示
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
Self-Attention       ｜自注意力          ｜从同一输入产生 Query、Key、Value 并动态聚合序列信息的注意力机制
Single-Head Attention｜单头注意力        ｜只使用一组 Query、Key、Value 投影计算关系的注意力
Query                ｜查询              ｜由输入经过可学习投影得到、用于匹配各位置 Key 的向量
Key                  ｜键                ｜由输入经过可学习投影得到、用于与 Query 计算匹配分数的向量
Value                ｜值                ｜由输入经过可学习投影得到、按照注意力权重被加权聚合的向量
Attention Score      ｜注意力分数        ｜Query 与 Key 点积后得到的未归一化匹配分数
Scaled Dot-Product Attention｜缩放点积注意力｜将 Query 与 Key 的点积除以键维度平方根后计算权重的注意力
Causal Mask          ｜因果掩码          ｜屏蔽未来位置、使当前位置只能读取自己和过去位置的掩码
Softmax              ｜Softmax 归一化    ｜把一组分数转换为总和为 1 的非负权重
Attention Weight     ｜注意力权重        ｜由当前输入动态计算的中间结果，表示各可见位置参与信息聚合的比例，不是模型参数
Context Representation｜上下文表示      ｜按照注意力权重对 Value 加权求和后得到的连续表示
Future Token         ｜未来词元          ｜位于当前处理位置之后、因因果约束而不能被读取的 Token
Information Leakage  ｜信息泄漏          ｜训练计算错误读取本应不可见的信息，导致学习目标与真实预测过程不一致
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
