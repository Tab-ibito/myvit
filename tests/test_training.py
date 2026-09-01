"""训练、验证和 checkpoint 闭环测试。"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from myvit.checkpoint import inspect_checkpoint, load_checkpoint, save_checkpoint
from myvit.debug_training import create_debug_model, make_pattern_dataset
from myvit.engine import evaluate, train_one_epoch


def test_training_reports_finite_metrics_and_updates_parameters() -> None:
    torch.manual_seed(0)
    model = create_debug_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loader = DataLoader(make_pattern_dataset(16, seed=0), batch_size=8)
    before = model.head.weight.detach().clone()

    metrics = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        max_gradient_norm=1.0,
    )

    assert metrics.samples == 16
    assert metrics.loss > 0
    assert 0 <= metrics.top1 <= 100
    assert metrics.learning_rate == 3e-3
    assert metrics.gradient_norm is not None and metrics.gradient_norm > 0
    assert not torch.equal(before, model.head.weight.detach())


def test_evaluate_does_not_create_gradients() -> None:
    model = create_debug_model()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(make_pattern_dataset(8, seed=1), batch_size=4)

    metrics = evaluate(model, loader, criterion, torch.device("cpu"))

    assert metrics.samples == 8
    assert metrics.learning_rate is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_checkpoint_round_trip_restores_model_and_training_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    model = create_debug_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 先更新一次，确保 checkpoint 中包含非空的 AdamW 动量状态。
    images, targets = next(iter(DataLoader(make_pattern_dataset(8), batch_size=8)))
    loss = nn.CrossEntropyLoss()(model(images), targets)
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        metrics={"validation": {"top1": 75.0}},
        best_metric=75.0,
        extra={"experiment": "unit-test"},
    )

    restored_model = create_debug_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=9e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer,
        T_max=10,
    )
    metadata = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert metadata.epoch == 3
    assert metadata.best_metric == 75.0
    assert metadata.extra["experiment"] == "unit-test"
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert restored_scheduler.last_epoch == scheduler.last_epoch

    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, actual)

    summary = inspect_checkpoint(checkpoint_path)
    assert summary["epoch"] == 3
    assert summary["has_optimizer_state"] is True
    assert summary["has_scheduler_state"] is True
    assert summary["has_scaler_state"] is False
