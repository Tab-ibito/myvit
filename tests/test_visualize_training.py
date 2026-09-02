"""训练可视化的数据读取和 checkpoint 汇总测试。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from myvit.checkpoint import save_checkpoint
from myvit.debug_training import create_debug_model
from myvit.visualize_training import (
    format_checkpoint_table,
    inspect_checkpoints,
    load_metrics_history,
)


def test_load_metrics_history_sorts_epochs(tmp_path: Path) -> None:
    history_path = tmp_path / "metrics.jsonl"
    records = [
        {"epoch": 2, "train": {"loss": 0.5}},
        {"epoch": 0, "train": {"loss": 1.0}},
    ]
    history_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    history = load_metrics_history(history_path)

    assert [record["epoch"] for record in history] == [0, 2]


def test_inspect_checkpoints_reports_training_state(tmp_path: Path) -> None:
    model = create_debug_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    save_checkpoint(
        tmp_path / "best.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        metrics={"validation": {"loss": 0.25, "top1": 87.5}},
        best_metric=87.5,
    )

    summaries = inspect_checkpoints(tmp_path)

    assert len(summaries) == 1
    assert summaries[0].name == "best.pt"
    assert summaries[0].epoch == 2
    assert summaries[0].validation_loss == 0.25
    assert summaries[0].validation_top1 == 87.5
    assert summaries[0].optimizer is True
    assert summaries[0].scheduler is True
    assert summaries[0].scaler is False
    assert "best.pt" in format_checkpoint_table(summaries)
