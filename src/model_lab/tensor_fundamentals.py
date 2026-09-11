"""用可执行的小例子展示大模型开发中常见的 PyTorch Tensor 基础。"""

import torch


def create_basic_tensors() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """创建标量、向量、矩阵和一个具有 ``[B, T, C]`` 形状的三维张量。"""

    # 标量没有维度，因此 shape 是 []；这里用 float32 表示可参与连续数值计算的值。
    scalar = torch.tensor(7.0, dtype=torch.float32, device="cpu")

    # 向量只有一个维度。Token ID 是离散编号，所以通常使用 int64 整数类型。
    token_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cpu")

    # 矩阵有两个维度，这里的 [T, C] = [3, 4] 表示 3 个位置、每个位置 4 个特征。
    matrix = torch.arange(12, dtype=torch.float32, device="cpu").reshape(3, 4)

    # 三维张量使用后续模型中常见的 [B, T, C] 排列：
    # B = 2 个样本，T = 每个样本 3 个位置，C = 每个位置 4 个特征。
    sequence = torch.arange(24, dtype=torch.float32, device="cpu").reshape(2, 3, 4)

    return scalar, token_ids, matrix, sequence


def inspect_sequence(sequence: torch.Tensor) -> tuple[torch.Size, int, int]:
    """读取三维序列 Tensor 的整体形状、维度数量和序列长度。"""

    # shape 给出所有轴的大小；对于 [B, T, C]，结果依次对应批次、序列和特征。
    shape = sequence.shape

    # ndim 表示一共有多少个维度，而不是某个维度有多长。
    dimensions = sequence.ndim

    # size(1) 读取第 1 号轴 T 的长度。轴从 0 开始编号，所以 1 对应序列维度。
    sequence_length = sequence.size(1)

    return shape, dimensions, sequence_length


def select_tensor_regions(
    sequence: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """从 ``[B, T, C]`` Tensor 中选择 batch、位置、特征和序列片段。"""

    # [B, T, C] -> [T, C]：固定 B 轴的第 0 项，选择第一个完整样本。
    first_batch = sequence[0]

    # [B, T, C] -> [B, C]：保留所有样本，固定 T 轴的第 0 个位置。
    first_token = sequence[:, 0]

    # [B, T, C] -> [B, T]：保留所有样本和位置，只选择 C 轴的第 0 个特征。
    first_feature = sequence[:, :, 0]

    # [B, T, C] -> [B, 2, C]：在 T 轴截取前两个位置，其他轴保持完整。
    sequence_prefix = sequence[:, :2, :]

    return first_batch, first_token, first_feature, sequence_prefix


def reshape_sequence(sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """用 ``reshape`` 和 ``view`` 将 ``[B, T, C]`` 展平为 ``[B * T, C]``。"""

    batch_size, sequence_length, hidden_dimension = sequence.shape
    flattened_shape = (batch_size * sequence_length, hidden_dimension)

    # reshape 只改变看待数据的形状，不改变元素值；新旧 shape 的元素总数必须相同。
    reshaped = sequence.reshape(flattened_shape)

    # view 在这里得到相同结果，但它要求输入的内存布局连续；本示例的输入满足该条件。
    viewed = sequence.view(flattened_shape)

    return reshaped, viewed


def reorder_dimensions(
    sequence: torch.Tensor,
    grouped_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """分别使用 ``transpose`` 和 ``permute`` 改变 Tensor 的维度顺序。"""

    # [B, T, C] -> [B, C, T]：transpose 只交换 T、C 两个维度。
    transposed = sequence.transpose(1, 2)

    # [B, T, H, D] -> [B, H, T, D]：permute 可以一次重新排列全部维度。
    # H、D 在这里只是演示四维 shape；当前没有实现多头注意力。
    permuted = grouped_features.permute(0, 2, 1, 3)

    # 两种操作都没有“重新训练”数值，只是改变了读取这些数值时的维度顺序。
    return transposed, permuted


def change_batch_dimension(sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """为 ``[T, C]`` Tensor 增加批次维度，再移除这个大小为 1 的维度。"""

    # [T, C] -> [1, T, C]：在第 0 轴增加大小为 1 的 Batch Dimension（批次维度）。
    batched = sequence.unsqueeze(0)

    # [1, T, C] -> [T, C]：只移除第 0 轴这个大小为 1 的维度。
    restored = batched.squeeze(0)

    return batched, restored


def broadcast_features(
    sequence: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """将 ``[C]`` 特征向量广播到 ``[B, T, C]`` Tensor 的每个位置。"""

    # [B, T, C] + [C] -> [B, T, C]：PyTorch 从最右侧维度开始匹配 shape。
    # 较小的 [C] 会在 B、T 轴上自动扩展，因此每个样本的每个位置都加上同一组特征。
    return sequence + features


def elementwise_operations(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """对两个相同 shape 的 Tensor 执行逐元素四则运算。"""

    # 每个结果位置只使用输入中相同位置的两个值，不会把一个维度上的值累加起来。
    added = left + right
    subtracted = left - right
    multiplied = left * right
    divided = left / right

    return added, subtracted, multiplied, divided


def matrix_multiply(
    values: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """使用 ``@`` 和 ``torch.matmul`` 执行相同的矩阵乘法。"""

    # [T, C] @ [C, D] -> [T, D]；输入为 [B, T, C] 时，B 轴会保留。
    # 左侧末维 C 必须与右侧倒数第二维 C 相同，乘积会沿这个内部维度求和。
    operator_result = values @ projection
    matmul_result = torch.matmul(values, projection)

    # `*` 是逐元素乘法；这里的矩阵乘法会组合 C 维信息，两者语义不同。
    return operator_result, matmul_result


def compute_basic_gradient() -> torch.Tensor:
    """计算 ``loss = sum(x²)`` 对 ``x`` 的梯度。"""

    # requires_grad=True 告诉 PyTorch 记录由 x 参与构成的计算过程。
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)

    # 先逐元素平方，再求和得到只有一个数的 loss；标量可以直接调用 backward。
    loss = (x**2).sum()

    # backward 按链式法则反向计算梯度。对于 x²，导数是 2x。
    loss.backward()

    # backward 已为这个参与求导的叶子 Tensor 写入梯度；断言同时收窄 Optional 类型。
    gradient = x.grad
    assert gradient is not None
    return gradient
