"""验证集评估逻辑。"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from myvit.metrics import ClassificationMetricTracker, EpochMetrics


# inference_mode 比 no_grad 更彻底：除关闭梯度外，还省去版本计数等 autograd 开销。
# 它适合纯验证/推理，但函数内部不能执行需要反向传播的操作。
@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """在不记录梯度的情况下评估 loss、Top-1 和 Top-5。"""

    # eval() 关闭 Dropout/DropPath，并让 BatchNorm（若模型中存在）使用运行统计量。
    # eval() 本身不会关闭梯度，因此仍需要上面的 inference_mode。
    model.eval()
    tracker = ClassificationMetricTracker()

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"验证 loss 出现非有限值：{loss.item()}")

        # inference_mode 下 logits 本就不携带计算图，因此这里不需要再 detach。
        tracker.update(loss, logits, targets)

    return tracker.compute()
