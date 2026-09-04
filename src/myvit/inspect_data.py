"""检查 Imagenette 目录和第一个预处理 batch。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from myvit.data import IMAGENETTE_CLASS_NAMES, create_imagenette_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 Imagenette 数据集")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("data/imagenette2-320"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    train_loader, validation_loader = create_imagenette_loaders(
        args.root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0,
    )
    images, targets = next(iter(train_loader))
    train_dataset = train_loader.dataset
    validation_dataset = validation_loader.dataset

    class_names = [
        IMAGENETTE_CLASS_NAMES[class_id]
        for class_id in train_dataset.classes  # type: ignore[attr-defined]
    ]
    summary = {
        "root": str(args.root.resolve()),
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "classes": class_names,
        "batch_images_shape": list(images.shape),
        "batch_targets_shape": list(targets.shape),
        "batch_min": float(images.min().item()),
        "batch_max": float(images.max().item()),
        "batch_mean": float(images.mean().item()),
        "batch_std": float(images.std().item()),
        "first_targets": targets[: min(8, len(targets))].tolist(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

