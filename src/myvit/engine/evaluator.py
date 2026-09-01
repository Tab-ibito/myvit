"""验证集评估逻辑。"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from myvit.metrics import ClassificationMetricTracker, EpochMetrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """在不记录梯度的情况下评估 loss、Top-1 和 Top-5。"""

    model.eval()
    tracker = ClassificationMetricTracker()

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"验证 loss 出现非有限值：{loss.item()}")

        tracker.update(loss, logits, targets)

    return tracker.compute()

