"""把训练历史和 checkpoint 状态整理成曲线图与终端摘要。

训练脚本每个 epoch 会向 ``metrics.jsonl`` 追加一行 JSON。本模块读取这些记录，
绘制 Loss、Top-1/Top-5、学习率和梯度范数曲线，并检查同目录下的 checkpoint。
默认只保存 PNG，不弹出窗口，因此在本机、远程服务器和无桌面环境中都能运行。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from myvit.checkpoint import inspect_checkpoint


@dataclass(frozen=True)
class CheckpointSummary:
    """一个 checkpoint 在终端表格中需要展示的信息。"""

    name: str
    epoch: int | None
    validation_loss: float | None
    validation_top1: float | None
    best_top1: float | None
    optimizer: bool
    scheduler: bool
    scaler: bool
    size_mib: float
    error: str | None = None


def load_metrics_history(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 训练历史，并按 epoch 排序。

    JSONL 每一行都是一个完整的 epoch 记录。逐行解析可以在某一行损坏时准确指出
    行号，也避免把整个文件当成一个 JSON 数组来读取。
    """

    history_path = Path(path)
    if not history_path.is_file():
        raise FileNotFoundError(f"找不到训练指标文件：{history_path}")

    records: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    with history_path.open("r", encoding="utf-8") as history_file:
        for line_number, line in enumerate(history_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{history_path} 第 {line_number} 行不是合法 JSON"
                ) from error

            if not isinstance(record, dict) or "epoch" not in record:
                raise ValueError(f"{history_path} 第 {line_number} 行缺少 epoch")
            epoch = int(record["epoch"])
            if epoch in seen_epochs:
                raise ValueError(f"{history_path} 中 epoch={epoch} 重复")
            seen_epochs.add(epoch)
            records.append(record)

    if not records:
        raise ValueError(f"训练指标文件为空：{history_path}")
    return sorted(records, key=lambda record: int(record["epoch"]))


def inspect_checkpoints(directory: str | Path) -> list[CheckpointSummary]:
    """检查目录中的 ``.pt``/``.pth`` 文件，不加载模型结构。"""

    checkpoint_directory = Path(directory)
    checkpoint_paths = sorted(
        {
            *checkpoint_directory.glob("*.pt"),
            *checkpoint_directory.glob("*.pth"),
        }
    )

    summaries: list[CheckpointSummary] = []
    for checkpoint_path in checkpoint_paths:
        size_mib = checkpoint_path.stat().st_size / (1024**2)
        try:
            payload = inspect_checkpoint(checkpoint_path)
            validation = payload.get("metrics", {}).get("validation", {})
            summaries.append(
                CheckpointSummary(
                    name=checkpoint_path.name,
                    epoch=_optional_int(payload.get("epoch")),
                    validation_loss=_optional_float(validation.get("loss")),
                    validation_top1=_optional_float(validation.get("top1")),
                    best_top1=_optional_float(payload.get("best_metric")),
                    optimizer=bool(payload.get("has_optimizer_state")),
                    scheduler=bool(payload.get("has_scheduler_state")),
                    scaler=bool(payload.get("has_scaler_state")),
                    size_mib=size_mib,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            # 单个文件损坏时仍展示其他 checkpoint；错误会在表格里明确标出。
            summaries.append(
                CheckpointSummary(
                    name=checkpoint_path.name,
                    epoch=None,
                    validation_loss=None,
                    validation_top1=None,
                    best_top1=None,
                    optimizer=False,
                    scheduler=False,
                    scaler=False,
                    size_mib=size_mib,
                    error=str(error),
                )
            )
    return summaries


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _nested_metric(
    records: list[dict[str, Any]],
    section: str,
    metric: str,
) -> list[float | None]:
    """提取嵌套指标；缺失值保留为 None，让 Matplotlib 自动断开曲线。"""

    return [
        _optional_float(record.get(section, {}).get(metric))
        for record in records
    ]


def save_training_figure(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "ViT training metrics",
    show: bool = False,
) -> Path:
    """将训练曲线保存成一张 PNG，并可选地弹出交互窗口。"""

    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError(
            "缺少 matplotlib；请先安装项目的 visualization 可选依赖"
        ) from error

    # 不需要弹窗时使用 Agg 后端，保证 SSH、CI 和训练服务器也能生成图片。
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(record["epoch"]) for record in records]
    train_loss = _nested_metric(records, "train", "loss")
    validation_loss = _nested_metric(records, "validation", "loss")
    train_top1 = _nested_metric(records, "train", "top1")
    validation_top1 = _nested_metric(records, "validation", "top1")
    train_top5 = _nested_metric(records, "train", "top5")
    validation_top5 = _nested_metric(records, "validation", "top5")
    learning_rate = _nested_metric(records, "train", "learning_rate")
    gradient_norm = _nested_metric(records, "train", "gradient_norm")

    # 2x2 布局把最常看的训练信号放在同一张图中，方便对齐同一个 epoch 分析。
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle(title, fontsize=15)

    loss_axis = axes[0, 0]
    loss_axis.plot(epochs, train_loss, marker="o", label="Train loss")
    loss_axis.plot(epochs, validation_loss, marker="o", label="Validation loss")
    _finish_axis(loss_axis, "Loss", "Cross-entropy loss")

    accuracy_axis = axes[0, 1]
    accuracy_axis.plot(epochs, train_top1, marker="o", label="Train Top-1")
    accuracy_axis.plot(epochs, validation_top1, marker="o", label="Validation Top-1")
    accuracy_axis.plot(
        epochs,
        train_top5,
        linestyle="--",
        alpha=0.75,
        label="Train Top-5",
    )
    accuracy_axis.plot(
        epochs,
        validation_top5,
        linestyle="--",
        alpha=0.75,
        label="Validation Top-5",
    )
    accuracy_axis.set_ylim(0, 100)
    _finish_axis(accuracy_axis, "Accuracy", "Accuracy (%)")

    learning_rate_axis = axes[1, 0]
    learning_rate_axis.plot(epochs, learning_rate, marker="o", color="tab:purple")
    _finish_axis(learning_rate_axis, "Learning rate", "Learning rate", legend=False)

    gradient_axis = axes[1, 1]
    gradient_axis.plot(epochs, gradient_norm, marker="o", color="tab:orange")
    _finish_axis(gradient_axis, "Gradient norm", "L2 norm", legend=False)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output.resolve()


def _finish_axis(axis: Any, title: str, ylabel: str, *, legend: bool = True) -> None:
    """统一四张子图的坐标轴样式。"""

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.xaxis.get_major_locator().set_params(integer=True)
    if legend:
        axis.legend()


def format_checkpoint_table(summaries: list[CheckpointSummary]) -> str:
    """生成不依赖第三方表格库的 checkpoint 终端表格。"""

    if not summaries:
        return "未发现 .pt 或 .pth checkpoint。"

    headers = (
        "checkpoint",
        "epoch",
        "val_loss",
        "val_top1",
        "best_top1",
        "opt/sched/amp",
        "size(MiB)",
        "status",
    )
    rows = [
        (
            item.name,
            _format_value(item.epoch),
            _format_value(item.validation_loss, 4),
            _format_value(item.validation_top1, 2),
            _format_value(item.best_top1, 2),
            f"{_yes_no(item.optimizer)}/{_yes_no(item.scheduler)}/{_yes_no(item.scaler)}",
            f"{item.size_mib:.2f}",
            "OK" if item.error is None else f"ERROR: {item.error}",
        )
        for item in summaries
    ]
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render_row(row: tuple[object, ...]) -> str:
        return "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def _format_value(value: int | float | None, decimals: int | None = None) -> str:
    if value is None:
        return "-"
    if decimals is None:
        return str(value)
    return f"{value:.{decimals}f}"


def _yes_no(value: bool) -> str:
    return "Y" if value else "N"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="绘制 ViT 训练曲线并检查 checkpoint")
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        type=Path,
        default=Path("outputs/debug"),
        help="包含 metrics.jsonl 和 checkpoint 的实验目录",
    )
    parser.add_argument("--history", type=Path, help="自定义 metrics.jsonl 路径")
    parser.add_argument("--output", type=Path, help="输出 PNG 路径")
    parser.add_argument("--title", default="ViT training metrics", help="图表标题")
    parser.add_argument("--show", action="store_true", help="保存后弹出 Matplotlib 窗口")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    history_path = args.history or args.experiment_dir / "metrics.jsonl"
    output_path = args.output or args.experiment_dir / "training_curves.png"

    records = load_metrics_history(history_path)
    summaries = inspect_checkpoints(args.experiment_dir)
    saved_path = save_training_figure(
        records,
        output_path,
        title=args.title,
        show=args.show,
    )

    print(f"epochs: {records[0]['epoch']} -> {records[-1]['epoch']} ({len(records)} records)")
    print(f"figure: {saved_path}")
    print("\ncheckpoints:")
    print(format_checkpoint_table(summaries))


if __name__ == "__main__":
    main()
