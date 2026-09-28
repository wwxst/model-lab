# Glossary

本术语表面向第一次学习大模型的读者。英文术语保留模型研发中的常用写法，中文解释优先说明它在本项目中的实际含义。

```text
English Term         ｜中文术语          ｜中文解释
Model                ｜模型              ｜根据输入计算输出的一组结构和参数
Decoder-only Model   ｜仅解码器模型      ｜只使用因果 Decoder 结构预测后续 Token 的语言模型
Forward Pass         ｜前向传播          ｜输入依次经过模型组件并产生输出的计算过程
Final Normalization  ｜最终归一化        ｜Decoder Stack 之后、Language Model Head 之前的 Layer Normalization
Model Context        ｜模型上下文        ｜一次前向传播最多能够接收的 Token 序列范围
End-to-End Pipeline  ｜端到端链路        ｜从 Token IDs 输入到 Logits 输出的完整连接关系
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
Cross Entropy Loss   ｜交叉熵损失        ｜衡量模型预测分布与正确 Token 之间差距的损失
Log Probability      ｜对数概率          ｜概率取自然对数后的数值
Log Softmax          ｜对数 Softmax      ｜以数值稳定方式把 Logits 转换为对数概率
Negative Log-Likelihood｜负对数似然      ｜正确 Token 对数概率的负值
Gather               ｜按索引选取        ｜根据 Target ID 取得对应词表位置数值的操作
Reduction            ｜归约              ｜把多个位置的损失合并为一个标量的过程
Gradient             ｜梯度              ｜表示参数变化会如何影响损失的数值
Optimizer            ｜优化器            ｜根据梯度更新模型参数的算法
Checkpoint           ｜检查点            ｜训练过程中保存的模型参数及相关状态
State Dict           ｜状态字典          ｜PyTorch 用于保存模块或优化器状态的键值结构
Resume Training      ｜恢复训练          ｜从 Checkpoint 加载状态后继续更新参数
Map Location         ｜设备映射          ｜加载 Tensor 时指定目标 CPU 或 GPU 设备
Completed Epoch      ｜已完成轮次        ｜Checkpoint 创建前完整结束的训练轮次数量
Training             ｜训练              ｜使用数据和损失反复更新模型参数的过程
Training Loop        ｜训练循环          ｜重复执行前向、损失、反向和参数更新的过程
Epoch                ｜训练轮次          ｜完整遍历一次训练 DataLoader 的过程
Zero Gradient        ｜梯度清零          ｜在新 Batch 反向传播前清除上一次参数梯度
Optimizer Step       ｜优化器更新        ｜根据当前梯度修改模型参数的一次操作
Parameter Update     ｜参数更新          ｜训练中让模型参数朝降低 Loss 的方向变化
Token-weighted Mean  ｜按 Token 加权平均 ｜按各 Batch 的 Target Token 数合并平均 Loss
Inference            ｜推理              ｜使用训练后的模型计算预测结果的过程
Transformer          ｜Transformer 架构  ｜通过注意力等结构处理序列的一类神经网络架构
Transformer Block    ｜Transformer 模块  ｜组合 Attention、MLP、Normalization 和 Residual 的基本计算单元
Sublayer             ｜子层              ｜Block 内承担一种计算职责的 Attention 或 Feed-Forward 组件
Pre-Norm             ｜前置归一化        ｜在输入进入每个子层之前执行 Layer Normalization
Residual Path        ｜残差路径          ｜让子层输入绕过子层并直接加到输出上的路径
Hidden State         ｜隐藏状态          ｜Token 在模型内部逐层更新的连续向量表示
Language Model Head  ｜语言模型输出头    ｜把隐藏状态投影为词表中每个 Token 的 Logits
Vocabulary Dimension｜词表维度          ｜输出最后一维的大小，等于词表中的 Token 数量
Raw Score            ｜原始分数          ｜Softmax 之前可以为任意实数的模型输出
Weight Tying         ｜权重绑定          ｜让输出投影与 Token Embedding 共享同一参数表的做法
Decoder Stack        ｜解码器堆叠        ｜按顺序连接多个 Decoder Transformer Block 的结构
Layer                ｜层                ｜Stack 中一个具有独立参数的 Transformer Block
Depth                ｜深度              ｜Stack 包含的 Transformer Block 数量
ModuleList           ｜模块列表          ｜让 PyTorch 注册并管理一组有顺序的子模块
Sequential Execution ｜顺序执行          ｜让前一层输出成为后一层输入的计算方式
Feed-Forward Network ｜前馈网络          ｜对每个 Token 的特征独立执行两层线性变换的网络
MLP                  ｜多层感知机        ｜由线性层和非线性激活函数组成的前馈结构
Expansion Dimension  ｜扩展维度          ｜前馈网络中间层扩展后的特征数量，当前固定为嵌入维度的四倍
GELU                 ｜高斯误差线性单元  ｜在两层线性投影之间加入非线性表达能力的激活函数
Position-wise        ｜逐位置            ｜对每个序列位置独立应用相同计算，不混合不同 Token
Residual Connection  ｜残差连接          ｜把子层输入直接加到子层输出上的连接
Layer Normalization  ｜层归一化          ｜沿单个 Token 的特征维计算并调整数值分布
Mean                 ｜均值              ｜一组数值之和除以数值数量
Variance             ｜方差              ｜各数值与均值之差的平方的平均值
Epsilon              ｜极小常数          ｜加在方差上以避免除以零的正数
Scale                ｜缩放参数          ｜归一化后对每个特征进行可学习缩放的参数
Shift                ｜平移参数          ｜归一化后对每个特征进行可学习平移的参数
Attention            ｜注意力            ｜计算序列中不同 Token 之间信息关系的机制
Self-Attention       ｜自注意力          ｜从同一输入产生 Query、Key、Value 并动态聚合序列信息的注意力机制
Single-Head Attention｜单头注意力        ｜只使用一组 Query、Key、Value 投影计算关系的注意力
Multi-Head Attention ｜多头注意力        ｜并行使用多个 Head 计算注意力，再拼接和投影各 Head 的结果
Attention Head       ｜注意力头          ｜在部分特征维度上独立计算注意力的一条分支
Head Dimension       ｜头维度            ｜单个 Head 分到的特征数量，等于嵌入维度除以 Head 数量
Split Heads          ｜拆分多头          ｜把完整特征维拆成多个较小 Head 的形状变换
Concat Heads         ｜拼接多头          ｜把多个 Head 的上下文结果重新拼接为完整特征维
Output Projection    ｜输出投影          ｜在拼接后混合不同 Head 信息的线性变换
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
