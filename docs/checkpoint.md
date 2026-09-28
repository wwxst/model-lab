# Checkpoint｜检查点

Checkpoint（检查点）是训练过程中保存的状态快照。当前实现保存继续训练所需的
模型参数、Optimizer 状态和已完成 Epoch 编号，使新创建的模型和 Optimizer 能够
恢复到保存时的状态。

```text
English Term        ｜中文术语          ｜中文解释
Checkpoint          ｜检查点            ｜训练过程中保存的模型与优化器状态快照
State Dict           ｜状态字典          ｜PyTorch 用于保存模块或优化器状态的键值结构
Resume Training     ｜恢复训练          ｜从 Checkpoint 加载状态后继续更新参数
Map Location        ｜设备映射          ｜加载 Tensor 时指定目标 CPU 或 GPU 设备
Completed Epoch     ｜已完成轮次        ｜Checkpoint 创建前完整结束的训练轮次数量
```

## 1. 为什么只保存三个字段

当前训练链路需要三类状态：

```text
model_state_dict     ｜模型全部 Parameter 的数值
optimizer_state_dict ｜Optimizer 的动量、步数等内部状态
epoch                ｜已经完成的训练轮次
```

只保存模型参数会丢失 AdamW 等 Optimizer 的动量和 step 计数。恢复后虽然可以继续
计算，但更新轨迹会改变。保存 epoch 则让调用者知道下一个训练轮次从哪里开始。

## 2. 保存状态

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
}
torch.save(checkpoint, path)
```

`state_dict` 保存参数和状态数据，不保存整个 Python 类实例，因此加载时仍需由
调用者先创建结构相同的 Model 与 Optimizer。

## 3. 恢复状态

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
completed_epoch = checkpoint["epoch"]
```

`map_location` 允许把在 GPU 上保存的 Tensor 映射到 CPU，或映射到调用者指定的
设备。当前实现保留 PyTorch 加载错误，不用宽泛异常捕获隐藏损坏文件或结构不匹配。

## 4. Optimizer 为什么必须先创建

恢复流程要求先创建与保存时结构对应的 Model 和 Optimizer，再调用
`load_checkpoint`：

```text
创建 Model
→ 创建 Optimizer(model.parameters())
→ load_checkpoint(...)
→ train_epoch(...)
```

Optimizer 需要持有当前模型的参数对象，不能在加载后再替换模型结构或重新创建
Optimizer，否则已恢复的动量等状态可能失去对应关系。

## 5. 当前 API

```python
from model_lab.checkpoint import load_checkpoint, save_checkpoint

save_checkpoint("model.pt", model, optimizer, epoch=10)
completed_epoch = load_checkpoint(
    "model.pt",
    model,
    optimizer,
    map_location="cpu",
)
assert completed_epoch == 10
```

当前 Checkpoint 模块不负责自动保存频率、最佳模型选择、目录管理、版本迁移、
恢复训练循环或生成逻辑。
