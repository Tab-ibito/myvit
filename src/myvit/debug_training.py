"""不依赖真实数据集的最小训练闭环。

数据中的类别由图片颜色和亮区位置决定，因此模型不仅能死记随机标签，也可以在
独立验证集上学习到规律。本脚本用于检查训练引擎和指标，不用于衡量 ImageNet 性能。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from myvit.checkpoint import load_checkpoint, save_checkpoint
from myvit.engine import evaluate, train_one_epoch
from myvit.metrics import EpochMetrics
from myvit.models import VisionTransformer


def set_random_seed(seed: int) -> None:
    """固定 Python 与 PyTorch 随机数，方便复现实验。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_pattern_dataset(
    samples: int,
    *,
    image_size: int = 32,
    num_classes: int = 4,
    seed: int = 0,
) -> TensorDataset:
    """生成带有明确空间模式的彩色图片。

    四个类别分别在左上、右上、左下、右下放置亮色方块，并添加少量噪声。
    该任务很简单，若模型和训练循环正确，验证准确率应迅速上升。
    """

    if num_classes != 4:
        raise ValueError("当前调试数据固定为 4 个象限类别")

    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(samples, 3, image_size, image_size, generator=generator) * 0.05
    targets = torch.arange(samples) % num_classes

    square_size = image_size // 3
    offsets = (
        (2, 2),
        (2, image_size - square_size - 2),
        (image_size - square_size - 2, 2),
        (image_size - square_size - 2, image_size - square_size - 2),
    )

    for index, target in enumerate(targets.tolist()):
        top, left = offsets[target]
        channel = target % 3
        images[index, channel, top : top + square_size, left : left + square_size] += 1.0

    return TensorDataset(images, targets.long())


def create_debug_model() -> VisionTransformer:
    """创建快速调试用的迷你 ViT；模块结构与 ViT-Tiny 完全相同。"""

    return VisionTransformer(
        image_size=32,
        patch_size=8,
        num_classes=4,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )


def format_epoch_line(
    epoch: int,
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    best_top1: float,
) -> str:
    """生成适合终端阅读的一行训练日志。"""

    return (
        f"epoch={epoch:03d} "
        f"train_loss={train_metrics.loss:.4f} train_top1={train_metrics.top1:6.2f}% "
        f"val_loss={validation_metrics.loss:.4f} val_top1={validation_metrics.top1:6.2f}% "
        f"lr={train_metrics.learning_rate:.6f} "
        f"grad_norm={train_metrics.gradient_norm:.3f} "
        f"best_val_top1={best_top1:6.2f}%"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 ViT 最小训练闭环")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/debug"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="模拟训练中断：保存该 epoch 后正常退出；总 epochs 配置保持不变",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_debug_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    train_loader = DataLoader(
        make_pattern_dataset(64, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        make_pattern_dataset(32, seed=args.seed + 1),
        batch_size=args.batch_size,
    )

    start_epoch = 0
    best_top1 = float("-inf")
    if args.resume is not None:
        metadata = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = metadata.epoch + 1
        best_top1 = (
            metadata.best_metric
            if metadata.best_metric is not None
            else float("-inf")
        )
        print(f"resumed_from={args.resume} next_epoch={start_epoch}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "metrics.jsonl"
    if start_epoch == 0:
        # 新实验覆盖旧历史；只有 --resume 时才会继续追加。
        history_path.write_text("", encoding="utf-8")

    end_epoch = args.epochs
    if args.stop_after_epoch is not None:
        end_epoch = min(end_epoch, args.stop_after_epoch + 1)

    for epoch in range(start_epoch, end_epoch):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_gradient_norm=1.0,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        is_best = validation_metrics.top1 > best_top1
        best_top1 = max(best_top1, validation_metrics.top1)
        metrics = {
            "train": train_metrics.to_dict(),
            "validation": validation_metrics.to_dict(),
        }

        # 先推进调度器再保存，保证恢复后的学习率与不中断训练完全一致。
        scheduler.step()

        # last.pt 每轮覆盖，用于意外中断后继续；best.pt 只在验证集刷新纪录时更新。
        save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            best_metric=best_top1,
            extra={"seed": args.seed},
        )
        if is_best:
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                best_metric=best_top1,
                extra={"seed": args.seed},
            )

        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(
                json.dumps(
                    {"epoch": epoch, **metrics},
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(format_epoch_line(epoch, train_metrics, validation_metrics, best_top1))

    print(f"device={device} checkpoints={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
