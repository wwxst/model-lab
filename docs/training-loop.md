# Training Loop｜训练循环

`train_epoch` 把 Dataset、DataLoader、完整 Decoder-only Model、Cross Entropy
Loss、Autograd 和 Optimizer 连接成一次完整训练遍历。它显式展示每个 Batch 如何
产生 Loss、计算梯度并更新模型参数。

```text
English Term       ｜中文术语          ｜中文解释
Training Loop      ｜训练循环          ｜重复执行前向、损失、反向和参数更新的过程
Epoch              ｜训练轮次          ｜完整遍历一次训练 DataLoader 的过程
Zero Gradient      ｜梯度清零          ｜在新 Batch 反向传播前清除上一次参数梯度
Optimizer Step     ｜优化器更新        ｜根据当前梯度修改模型参数的一次操作
Parameter Update   ｜参数更新          ｜训练中让模型参数朝降低 Loss 的方向变化
Token-weighted Mean｜按 Token 加权平均 ｜按各 Batch 的 Target Token 数合并平均 Loss
```

## 1. 一个 Batch 的完整训练步骤

每个 Batch 提供：

```text
Input IDs  [B,T]
Target IDs [B,T]
```

训练循环依次执行：

```text
optimizer.zero_grad()
↓
logits = model(input_ids)       [B,T,V]
↓
loss = cross_entropy(logits, targets)  scalar
↓
loss.backward()
↓
optimizer.step()
```

顺序不能交换：`backward()` 需要 Forward Pass（前向传播）建立的计算图，
`optimizer.step()` 需要 `backward()` 已经写入参数的梯度。

## 2. 为什么每个 Batch 前要清零梯度

PyTorch 默认会把新的梯度累加到 `parameter.grad`，而不是自动覆盖。如果不调用
`optimizer.zero_grad()`，第二个 Batch 的梯度会与第一个 Batch 残留的梯度相加：

```text
没有清零：grad = grad_batch_1 + grad_batch_2 + ...
当前训练：grad = grad_current_batch
```

当前 Training Loop 的职责是每个 Batch 独立更新一次，所以在 Forward Pass 前清零。

## 3. Backward 与 Optimizer Step

`loss.backward()` 从标量 Loss 沿计算图反向传播，为 Language Model Head、Decoder
Stack、Embedding 等全部可学习参数计算梯度。它只计算梯度，不直接修改参数。

`optimizer.step()` 随后读取这些梯度，并按照具体 Optimizer 的规则更新参数。当前
函数接收调用者创建的 Optimizer，不在内部固定 SGD 或 AdamW，因此同一条清晰训练
流程可以使用 PyTorch 的标准优化器。

## 4. Device 放置

模型参数、Input IDs 和 Target IDs 必须位于同一个 Device。模型需要在创建
Optimizer 之前由调用者移动；这是因为 Device 转换可能产生新的参数对象，Optimizer
必须持有转换后的参数。`train_epoch` 再把每个 Batch 的 Input 和 Target 移到同一
Device：

```text
model.to(device)             # 创建 Optimizer 之前
optimizer = Optimizer(model.parameters())
input_ids.to(device)
target_ids.to(device)
```

没有 CUDA 时可以传入 `"cpu"`，训练逻辑不变。

## 5. Epoch Loss 如何计算

Cross Entropy Loss 返回当前 Batch 内全部 `B × T` Token 的平均值。DataLoader 的
最后一个 Batch 可能更小，因此不能简单地让每个 Batch Loss 拥有相同权重。

当前实现先乘回各 Batch 的 Target Token 数：

```text
total_loss += batch_mean_loss × batch_token_count
```

遍历完成后再除以全部 Target Token 数：

```text
epoch_loss = total_loss / total_token_count
```

这样返回值表示整个 Epoch 中每个 Token 的平均 Loss。

## 6. 当前 API

```python
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model_lab.training import train_epoch

device = torch.device("cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-4)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

for _ in range(10):
    loss = train_epoch(model, data_loader, optimizer, device=device)
```

当前函数只执行一个训练 Epoch，不包含验证循环、日志系统、Learning Rate Scheduler
（学习率调度器）、Gradient Clipping（梯度裁剪）、Checkpoint 或自动恢复逻辑。
