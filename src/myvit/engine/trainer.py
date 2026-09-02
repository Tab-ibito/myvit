"""单个 epoch 的训练逻辑。"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from myvit.metrics import (
    ClassificationMetricTracker,
    EpochMetrics,
    parameters_gradient_norm,
)


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_gradient_norm: float | None = None,
) -> EpochMetrics:
    """训练一个 epoch，并返回按样本汇总的指标。

    Args:
        model: 待训练模型。
        data_loader: 产生 ``(images, targets)`` 的可迭代对象。
        criterion: 通常使用 ``nn.CrossEntropyLoss``。
        optimizer: 例如 AdamW。
        device: CPU 或 CUDA 设备。
        scaler: 使用 CUDA AMP 时传入同一个 GradScaler，并保存进 checkpoint。
        use_amp: 仅 CUDA 下启用 FP16 自动混合精度。
        max_gradient_norm: 若非空，在 optimizer.step 前裁剪全局梯度范数。
    """

    if use_amp and device.type != "cuda":
        raise ValueError("当前实现仅在 CUDA 设备上启用 AMP")
    if use_amp and scaler is None:
        raise ValueError("use_amp=True 时必须提供 GradScaler")
    if max_gradient_norm is not None and max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm 必须大于 0")

    # train() 会启用 Dropout/DropPath 等训练行为；它不会自动启用梯度，
    # 梯度是否记录由 PyTorch 当前上下文和 requires_grad 决定。
    model.train()
    tracker = ClassificationMetricTracker()
    gradient_norm_sum = 0.0
    steps = 0

    for images, targets in data_loader:
        # non_blocking=True 只有在 CUDA + pinned memory DataLoader 时才能真正异步；
        # CPU 调试时仍然安全，相当于普通的数据移动。
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # set_to_none=True 比填充为 0 更省内存，下一次 backward 会重新创建梯度。
        optimizer.zero_grad(set_to_none=True)

        # autocast 只改变适合半精度计算的算子 dtype，模型参数本身仍由优化器管理。
        # CPU 调试时 use_amp=False，因此这里保持标准 FP32。
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"训练 loss 出现非有限值：{loss.item()}")

        if scaler is not None and use_amp:
            # FP16 的数值范围较小。GradScaler 先放大 loss，避免很小的梯度下溢为 0。
            scaler.scale(loss).backward()

            # 梯度裁剪和梯度范数统计必须基于反缩放后的真实梯度。
            scaler.unscale_(optimizer)

            # 日志记录裁剪前的梯度，因此即使 max_gradient_norm=1.0，
            # 输出的 grad_norm 仍可能大于 1，这正是预期行为。
            gradient_norm = parameters_gradient_norm(model.parameters())
            if max_gradient_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

            # scaler.step 会在检测到 Inf/NaN 梯度时跳过本次参数更新；
            # update() 再根据最近的数值稳定性调整下一步缩放倍率。
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通 FP32 路径不需要 loss scaling，反向传播后可直接统计/裁剪梯度。
            loss.backward()
            gradient_norm = parameters_gradient_norm(model.parameters())
            if max_gradient_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()

        # 指标不参与训练，所以 detach logits，防止 tracker 延长计算图生命周期。
        tracker.update(loss, logits.detach(), targets)
        gradient_norm_sum += gradient_norm
        steps += 1

    # 当前项目只有一个参数组。若以后为 backbone/head 设置不同学习率，
    # 日志也应相应扩展为记录所有 param_groups。
    learning_rate = float(optimizer.param_groups[0]["lr"])

    # 当前记录每个 step 梯度范数的算术平均，便于观察 epoch 级趋势。
    average_gradient_norm = gradient_norm_sum / steps if steps else None
    return tracker.compute(
        learning_rate=learning_rate,
        gradient_norm=average_gradient_norm,
    )
