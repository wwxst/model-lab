# Tensor Fundamentals｜张量基础

本文通过可执行的 PyTorch 示例解释后续文本模型开发中最常见的 Tensor（张量）基础。对应源码位于 `src/model_lab/tensor_fundamentals.py`，行为测试位于 `tests/test_tensor_fundamentals.py`。

```text
English Term          ｜中文术语          ｜中文解释
Tensor                ｜张量              ｜保存和计算多维数值数据的结构
Shape                 ｜形状              ｜Tensor 每个维度大小组成的描述
Dimension             ｜维度              ｜Tensor 中一个独立的方向
Axis                  ｜轴                ｜在代码中按编号指定的某个维度
Data Type             ｜数据类型          ｜Tensor 中每个元素采用的数值类型
Device                ｜设备              ｜Tensor 数据实际存放和计算的位置
Indexing              ｜索引              ｜按位置选择 Tensor 中的元素或区域
Slicing               ｜切片              ｜按范围截取 Tensor 中的一段数据
Broadcasting          ｜广播              ｜让兼容的较小 Tensor 自动参与较大 shape 的运算
Matrix Multiplication ｜矩阵乘法          ｜沿匹配的内部维度执行乘法并求和
Autograd              ｜自动求导          ｜记录计算并自动计算梯度的机制
```

## 1. Tensor 是什么

Tensor 本质上可以看作更高维的数值数组。一个 Tensor 可以只保存一个数，也可以沿多个维度组织大量数值。PyTorch 使用同一个 `torch.Tensor` 类型表示这些情况。

```text
Scalar    ｜标量      ｜没有维度的单个数
Vector    ｜向量      ｜沿一个维度排列的一组数
Matrix    ｜矩阵      ｜沿两个维度排列的数值
Tensor    ｜张量      ｜标量、向量、矩阵及更高维数值数据的统一表示
```

```python
scalar = torch.tensor(7.0)  # shape = []
vector = torch.tensor([0, 1, 2, 3])  # shape = [C]
matrix = torch.arange(12).reshape(3, 4)  # shape = [T, C]
tensor = torch.arange(24).reshape(2, 3, 4)  # shape = [B, T, C]
```

`scalar.ndim` 是 `0`，`vector.ndim` 是 `1`，`matrix.ndim` 是 `2`，三维 `tensor.ndim` 是 `3`。

## 2. Shape、Dimension 和 Axis

`shape` 给出所有维度的大小，`ndim` 给出维度数量，`size(axis)` 给出指定轴的大小。

```python
x = torch.zeros((2, 3, 4))

x.shape  # torch.Size([2, 3, 4])
x.ndim  # 3
x.size(1)  # 3
```

Dimension（维度）描述一个数据方向。Axis（轴）是代码访问某个维度时使用的编号，从 `0` 开始。对 `[B, T, C]` 而言，轴 `0` 是 B，轴 `1` 是 T，轴 `2` 是 C。

## 3. dtype

`dtype` 表示每个元素的数据类型。当前示例只使用两种基础类型：

```text
torch.int64    ｜64 位整数    ｜适合保存 Token ID 等离散编号
torch.float32  ｜32 位浮点数  ｜适合模型参数、激活值和连续数学运算
```

整数编号表达“第几个”，不需要小数。模型参数和激活值会参与乘法、求和等连续数值运算，因此通常使用浮点数。参数在反向传播得到梯度后由优化器更新；激活值参与前向计算并在反向传播中传递梯度，但不是优化器直接更新的参数。本阶段不讨论 `float16`、`bfloat16`、混合精度或量化。

## 4. device

`device` 表示 Tensor 的数据实际存放在哪个计算设备上。

```python
cpu_tensor = torch.tensor([1.0, 2.0], device="cpu")
cpu_tensor.device.type  # "cpu"

if torch.cuda.is_available():
    cuda_tensor = cpu_tensor.to("cuda")
    cuda_tensor.device.type  # "cuda"
```

CUDA 只有在本机的 PyTorch 与 NVIDIA 环境可用时才能使用。Commit 3 的所有核心示例和测试都可以在 CPU 上运行，不包含自动设备策略或 GPU 性能测试。

## 5. `[B, T, C]`

后续模型代码经常使用三维 shape：

```text
B = Batch Size        ｜批次大小      ｜一次共同计算的样本数量
T = Sequence Length   ｜序列长度      ｜每个样本包含的位置数量
C = Hidden Dimension  ｜隐藏/特征维度 ｜每个位置由多少个数表示
```

例如 `[B, T, C] = [2, 3, 4]` 表示：

```text
2 个样本
每个样本有 3 个位置
每个位置由 4 个数表示
```

当前还没有 Tokenizer（分词器）或 Embedding（嵌入）。这里的三维 Tensor 只是用于学习 shape 语义，不能称为当前模型的 Embedding。

## 6. Indexing 和 Slicing

给定 `x.shape == [B, T, C] == [2, 3, 4]`：

```python
x[0]  # [B, T, C] -> [T, C]：选择第一个完整样本
x[:, 0]  # [B, T, C] -> [B, C]：选择所有样本的第一个位置
x[:, :, 0]  # [B, T, C] -> [B, T]：选择所有位置的第一个特征
x[:, :2, :]  # [B, T, C] -> [B, 2, C]：截取序列的前两个位置
```

单个整数索引会固定并移除对应维度；范围切片会保留该维度，只改变它的长度。

## 7. reshape 和 view

`reshape` 与 `view` 都可以改变 Tensor 的 shape，而不改变元素本身。

```text
[B, T, C] = [2, 3, 4]
        ↓
[B * T, C] = [6, 4]
```

```python
flattened_with_reshape = x.reshape(6, 4)
flattened_with_view = x.view(6, 4)
```

变形前后都包含 `24` 个元素。shape 可以改变，但元素总数必须一致。`view` 要求输入具有兼容的连续内存布局；当前示例使用连续 Tensor，不继续展开内部存储机制。`reshape` 在无法直接共享布局时可以产生满足目标 shape 的结果。

## 8. transpose 和 permute

`transpose` 交换两个维度：

```python
x.shape  # [B, T, C]
x.transpose(1, 2).shape  # [B, C, T]
```

`permute` 一次指定所有维度的新顺序：

```python
y.shape  # [B, T, H, D]
y.permute(0, 2, 1, 3).shape  # [B, H, T, D]
```

这里的 H 和 D 只用于说明四维 shape。当前没有实现多头 Attention（注意力）。这些操作也没有让数值被“重新训练”，只是改变维度的读取顺序。

## 9. unsqueeze 和 squeeze

`unsqueeze(axis)` 在指定位置增加一个大小为 `1` 的维度。`squeeze(axis)` 移除指定位置上大小为 `1` 的维度。

```text
[T, C]
   ↓ unsqueeze(0)
[1, T, C]
   ↓ squeeze(0)
[T, C]
```

模型经常需要临时增加 Batch Dimension（批次维度），把单个样本转换为一批只有一个样本的数据。

## 10. Broadcasting｜广播

PyTorch 从最右侧维度开始匹配 shape。在维度相同或其中一个大小为 `1` 时，较小 Tensor 可以自动参与运算。

```text
[B, T, C]
+       [C]
-----------
[B, T, C]
```

```python
sequence = torch.zeros((2, 3, 4))
features = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = sequence + features
```

`[C]` 中的四个值会加到每个样本的每个位置上。这里使用 PyTorch 自带的广播行为，不实现自定义广播逻辑。

## 11. Element-wise Operations｜逐元素运算

`+`、`-`、`*`、`/` 对应位置逐个计算：

```python
left = torch.tensor([8.0, 6.0])
right = torch.tensor([2.0, 3.0])

left + right  # [10.0, 9.0]
left - right  # [6.0, 3.0]
left * right  # [16.0, 18.0]
left / right  # [4.0, 2.0]
```

`left * right` 是逐元素乘法，不会沿某个维度求和。

## 12. Matrix Multiplication｜矩阵乘法

矩阵乘法会沿匹配的内部维度执行乘法并求和：

```text
[T, C] @ [C, D] -> [T, D]
[B, T, C] @ [C, D] -> [B, T, D]
```

左侧最后一维 C 必须与右侧倒数第二维 C 相同。`@` 与 `torch.matmul` 在这些示例中表达相同运算：

```python
operator_result = values @ projection
matmul_result = torch.matmul(values, projection)
```

矩阵乘法是未来 Linear Layer（线性层）和 Attention 中的重要基础，但本阶段只学习 Tensor 数学，不实现这些模型组件。

## 13. Autograd｜自动求导

PyTorch 会记录 `requires_grad=True` 的 Tensor 参与的计算，并在 `backward()` 时根据链式法则计算梯度。

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x**2).sum()
loss.backward()
x.grad  # tensor([2.0, 4.0, 6.0])
```

过程可以概括为：

```text
x
↓ 每个元素平方
x²
↓ 所有元素求和
loss
↓ backward
x.grad = 2x
```

这里仅演示 `requires_grad=True`、`loss.backward()` 和 `x.grad`。`nn.Module`、Parameter、Optimizer、Training Loop、自定义 autograd 和梯度累积均不属于 Commit 3。

## 14. 为什么这些知识对大模型重要

文本模型的大部分计算都可以追踪为 Tensor 的 shape 变化与数值运算。理解 `[B, T, C]`、索引、维度变换、广播、矩阵乘法和梯度后，后续学习 Dataset、Tokenizer、Embedding 与 Attention 时，能够直接判断每一步接收什么数据、输出什么 shape，以及梯度如何回到参与计算的值。
