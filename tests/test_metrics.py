"""分类指标测试。"""

import pytest
import torch

from myvit.metrics import ClassificationMetricTracker, topk_correct_counts


def test_topk_correct_counts() -> None:
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0],  # 预测类别 0，正确
            [1.0, 8.0, 2.0],  # 预测类别 1，目标类别 2：Top-1 错、Top-3 对
        ]
    )
    targets = torch.tensor([0, 2])

    assert topk_correct_counts(logits, targets, topk=(1, 3)) == [1, 2]


def test_metric_tracker_weights_batches_by_sample_count() -> None:
    tracker = ClassificationMetricTracker()

    tracker.update(
        torch.tensor(2.0),
        torch.tensor([[3.0, 0.0], [3.0, 0.0]]),
        torch.tensor([0, 0]),
    )
    tracker.update(
        torch.tensor(5.0),
        torch.tensor([[0.0, 3.0]]),
        torch.tensor([1]),
    )

    metrics = tracker.compute()

    assert metrics.loss == pytest.approx(3.0)
    assert metrics.top1 == pytest.approx(100.0)
    assert metrics.samples == 3

