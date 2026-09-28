# Cross Entropy Loss｜交叉熵损失

`cross_entropy_loss` 使用模型输出的 `[B,T,V]` Logits 和 Dataset 提供的
`[B,T]` Target Token IDs，计算所有 next-token prediction（下一位置预测）的平均
Cross Entropy Loss（交叉熵损失）。

```text
English Term          ｜中文术语          ｜中文解释
Cross Entropy Loss    ｜交叉熵损失        ｜衡量模型预测分布与正确 Token 之间差距的损失
Log Probability       ｜对数概率          ｜概率取自然对数后的数值
Log Softmax           ｜对数 Softmax      ｜以数值稳定方式把 Logits 转换为对数概率
Negative Log-Likelihood｜负对数似然      ｜正确 Token 对数概率的负值
Gather                ｜按索引选取        ｜根据 Target ID 取得对应词表位置数值的操作
Reduction             ｜归约              ｜把多个位置的损失合并为一个标量的过程
```

## 1. 输入与目标

完整模型为每个输入位置输出 `V` 个 Logits：

```text
Logits     [B,T,V]
Target IDs [B,T]
```

```text
B = Batch Size       ｜批次大小
T = Sequence Length  ｜序列长度
V = Vocabulary Size  ｜词表大小
```

`target_ids[b,t]` 是样本 `b` 在位置 `t` 应该预测出的下一个 Token ID。Dataset 已经
让 Target 相对 Input 向右移动一个位置，所以 Loss 只需要比较同一 `[b,t]` 位置。

## 2. Log Softmax

对单个位置的词表 Logits `z`，Softmax 概率是：

```text
probability_v = exp(z_v) / sum(exp(z_j))
```

Loss 需要正确 Token 的对数概率。当前实现直接使用 `log_softmax`：

```text
log_probability_v = z_v - log(sum(exp(z_j)))
```

`torch.log_softmax` 会以数值稳定方式完成计算，避免先计算很大的 `exp(z)` 再取
对数。因此即使 Logit 接近 1000，Loss 仍能保持有限数值。

## 3. Gather 正确 Token

`log_probabilities` 的 shape 是 `[B,T,V]`，但每个 `[b,t]` 只有一个正确 Token。
先把 Target 从 `[B,T]` 变为 `[B,T,1]`：

```text
Target IDs [B,T]
↓ unsqueeze(-1)
[B,T,1]
```

再用 `gather(dim=-1)` 沿词表维选出正确 Token 的 Log Probability：

```text
Log Probabilities [B,T,V]
Target Index       [B,T,1]
──────────────────────────
Selected           [B,T,1]
↓ squeeze(-1)
Target Log Probability [B,T]
```

## 4. Negative Log-Likelihood 与平均

正确 Token 的概率越接近 1，对数概率越接近 0；概率越小，对数概率越负。取负后，
更好的预测得到更小的 Loss：

```text
loss[b,t] = -target_log_probability[b,t]
```

最后对 Batch 和 Sequence 中的全部位置求平均：

```text
Cross Entropy Loss = mean(loss[b,t])
```

输出是零维标量 Tensor，可以直接调用 `backward()`。

## 5. 梯度代表什么

对单个位置，Cross Entropy 关于某个 Logit 的梯度是：

```text
gradient_v = predicted_probability_v - correct_one_hot_v
```

对正确 Token，梯度推动它的 Logit 增大；对其他 Token，梯度推动其 Logit 相对减小。
当前 Reduction 使用平均值，所以梯度还会除以参与计算的预测位置数量 `B × T`。

## 6. 当前 API

```python
import torch

from model_lab.cross_entropy import cross_entropy_loss

logits = torch.randn(2, 4, 32, requires_grad=True)
target_ids = torch.randint(0, 32, (2, 4), dtype=torch.int64)

loss = cross_entropy_loss(logits, target_ids)
loss.backward()

assert loss.shape == ()
```

当前函数不创建 Optimizer、不清零梯度、不更新参数，也不实现 Training Loop。
Target ID 的词表范围继续由 `gather` 的真实索引边界检查。
