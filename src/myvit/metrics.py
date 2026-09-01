"""图像分类训练中使用的指标。

训练循环会分批处理数据，因此不能直接平均每个 batch 的 loss/accuracy：
最后一个 batch 往往更小。这里按样本数累加，保证 epoch 指标不受 batch 大小影响。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EpochMetrics:
    """一个 epoch 结束后的汇总指标。

    ``loss`` 是每个样本的平均交叉熵；``top1`` 和 ``top5`` 使用百分数，
    因而 75.2 表示 75.2%，而不是 0.752。
    """

    loss: float
    top1: float
    top5: float
    samples: int
    learning_rate: float | None = None
    gradient_norm: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        """转换为可直接写入 checkpoint 或 JSON 日志的字典。"""

        return asdict(self)


def topk_correct_counts(
    logits: Tensor,
    targets: Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> list[int]:
    """返回各个 top-k 指标预测正确的样本数。

    当类别数小于 5 时，Top-5 会自动退化为 Top-C，其中 C 是类别总数。
    这使同一套代码既能处理 ImageNet-1K，也能处理小型调试数据集。
    """

    if logits.ndim != 2:
        raise ValueError(f"logits 应为 [B, C]，实际收到 {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets 应为 [B]，实际收到 {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits 与 targets 的 batch 大小不一致")
    if logits.shape[1] <= 0:
        raise ValueError("类别数量必须大于 0")
    if not topk or any(k <= 0 for k in topk):
        raise ValueError("topk 必须包含正整数")

    num_classes = logits.shape[1]
    max_k = min(max(topk), num_classes)

    # predicted 的形状为 [B, max_k]，每行按置信度从高到低排列类别下标。
    predicted = logits.topk(max_k, dim=1, largest=True, sorted=True).indices
    correct = predicted.eq(targets.unsqueeze(1))

    return [
        int(correct[:, : min(k, num_classes)].any(dim=1).sum().item())
        for k in topk
    ]


class ClassificationMetricTracker:
    """按样本累计 loss、Top-1 和 Top-5。"""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.top1_correct = 0
        self.top5_correct = 0
        self.samples = 0

    def update(self, loss: Tensor, logits: Tensor, targets: Tensor) -> None:
        """加入一个 batch 的结果。

        传入的 loss 应是 batch 内样本的平均值，因此累加前要乘以 batch 大小。
        """

        batch_size = targets.shape[0]
        top1_correct, top5_correct = topk_correct_counts(logits, targets)

        self.loss_sum += float(loss.detach().item()) * batch_size
        self.top1_correct += top1_correct
        self.top5_correct += top5_correct
        self.samples += batch_size

    def compute(
        self,
        *,
        learning_rate: float | None = None,
        gradient_norm: float | None = None,
    ) -> EpochMetrics:
        """计算当前累计结果；空 DataLoader 会被视为配置错误。"""

        if self.samples == 0:
            raise ValueError("无法从空 DataLoader 计算训练指标")

        return EpochMetrics(
            loss=self.loss_sum / self.samples,
            top1=100.0 * self.top1_correct / self.samples,
            top5=100.0 * self.top5_correct / self.samples,
            samples=self.samples,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
        )


def parameters_gradient_norm(parameters: Iterable[Tensor]) -> float:
    """计算所有参数梯度合在一起的 L2 范数。

    梯度范数适合发现训练爆炸、NaN 或学习率过高，但不同模型之间不能简单横向比较。
    ``parameters`` 通常传入 ``model.parameters()``。
    """

    squared_norm = torch.zeros((), dtype=torch.float64)
    has_gradient = False

    for parameter in parameters:
        if parameter.grad is None:
            continue
        has_gradient = True
        gradient = parameter.grad.detach()
        squared_norm += gradient.double().pow(2).sum().cpu()

    if not has_gradient:
        return 0.0

    return float(squared_norm.sqrt().item())
