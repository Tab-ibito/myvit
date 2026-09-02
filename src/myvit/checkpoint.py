"""训练 checkpoint 的保存与恢复。

checkpoint 不只保存模型参数，还应保存所有会影响下一步训练的状态：优化器动量、
学习率调度器位置和 AMP GradScaler。否则虽然模型权重恢复了，训练轨迹却会改变。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointMetadata:
    """恢复 checkpoint 后返回给训练脚本的轻量信息。"""

    epoch: int
    metrics: dict[str, Any]
    best_metric: float | None
    extra: dict[str, Any]


def inspect_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """读取 checkpoint 概要，但不创建模型或恢复训练。

    这个结果适合直接打印或转为 JSON，用于确认保存到了哪一轮、最佳指标是多少，
    以及优化器/调度器状态是否齐全。只应检查来源可信的 checkpoint 文件。
    """

    # inspect 只读取由张量和基础 Python 类型组成的状态字典，不执行模型构造。
    # weights_only=True 缩小了反序列化范围，但仍只应打开可信来源的文件。
    payload = torch.load(path, map_location=map_location, weights_only=True)
    return {
        "format_version": payload.get("format_version"),
        "epoch": payload.get("epoch"),
        "best_metric": payload.get("best_metric"),
        "metrics": payload.get("metrics", {}),
        "extra": payload.get("extra", {}),
        "has_optimizer_state": payload.get("optimizer_state") is not None,
        "has_scheduler_state": payload.get("scheduler_state") is not None,
        "has_scaler_state": payload.get("scaler_state") is not None,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    best_metric: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """原子性地保存训练状态。

    先写临时文件，再使用 ``os.replace`` 替换目标文件，可降低程序中断时留下
    半个 checkpoint 的概率。同一文件系统中的替换操作是原子的。
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    # state_dict 只描述状态，不保存 Python 模型类本身。因此加载时仍需先创建同结构模型。
    # format_version 用于未来 checkpoint 字段发生变化时做兼容判断。
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": (
            scheduler.state_dict() if scheduler is not None else None  # type: ignore[union-attr]
        ),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "best_metric": best_metric,
        "extra": extra or {},
    }

    try:
        # 不直接写 checkpoint_path：若进程在写入中途被杀，旧 checkpoint 仍然完好。
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        # torch.save 失败时清理残留临时文件，已成功 replace 时该路径不存在。
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CheckpointMetadata:
    """恢复 checkpoint，并返回 epoch、最佳指标等元数据。

    若需要继续训练，应同时传入 optimizer、scheduler 和 scaler；如果只做推理，
    只传 model 即可。恢复后下一轮通常从 ``metadata.epoch + 1`` 开始。
    """

    checkpoint_path = Path(path)
    # map_location 决定状态张量首先加载到哪里。CPU 推理通常使用 "cpu"；
    # GPU 续训可直接传训练 device，避免 optimizer 状态留在 CPU。
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    required_keys = {"format_version", "epoch", "model_state", "metrics"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise ValueError(f"checkpoint 缺少必要字段：{sorted(missing_keys)}")
    if payload["format_version"] != 1:
        raise ValueError(f"不支持的 checkpoint 版本：{payload['format_version']}")

    # strict=True 要求 checkpoint 与当前模型的参数名完全一致，最适合可靠续训。
    # 迁移学习时如果更换分类头，可以显式选择 strict=False，并检查缺失/多余参数。
    model.load_state_dict(payload["model_state"], strict=strict)

    # 只做推理时调用方不会传 optimizer/scheduler/scaler，因此只恢复模型即可。
    # 继续训练时三者应与模型一起恢复，否则学习率和 AdamW 动量会从头开始。
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])  # type: ignore[union-attr]
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])

    return CheckpointMetadata(
        epoch=int(payload["epoch"]),
        metrics=dict(payload["metrics"]),
        best_metric=payload.get("best_metric"),
        extra=dict(payload.get("extra", {})),
    )
