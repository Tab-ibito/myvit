"""Imagenette 目录、类别映射和图像预处理测试。"""

from pathlib import Path

import torch
from PIL import Image

from myvit.data.imagenette import (
    IMAGENETTE_CLASS_NAMES,
    build_train_transform,
    build_validation_transform,
    create_imagenette_datasets,
)


def test_imagenette_transforms_produce_model_input_shape() -> None:
    """无论训练随机裁剪还是验证中心裁剪，都应输出标准 CHW Tensor。"""

    image = Image.new("RGB", (360, 320), color=(64, 128, 192))

    train_tensor = build_train_transform(224)(image)
    validation_tensor = build_validation_transform(224)(image)

    assert train_tensor.shape == (3, 224, 224)
    assert validation_tensor.shape == (3, 224, 224)
    assert torch.isfinite(train_tensor).all()
    assert torch.isfinite(validation_tensor).all()


def test_create_imagenette_datasets_keeps_train_val_class_mapping(
    tmp_path: Path,
) -> None:
    """ImageFolder 对 train/val 必须产生完全相同的类别下标。"""

    for split in ("train", "val"):
        for class_index, class_id in enumerate(IMAGENETTE_CLASS_NAMES):
            class_directory = tmp_path / split / class_id
            class_directory.mkdir(parents=True)
            Image.new(
                "RGB",
                (320, 320),
                color=(class_index * 20, 64, 128),
            ).save(class_directory / "sample.jpg")

    train_dataset, validation_dataset = create_imagenette_datasets(tmp_path)

    assert len(train_dataset) == 10
    assert len(validation_dataset) == 10
    assert train_dataset.class_to_idx == validation_dataset.class_to_idx
    image, target = validation_dataset[0]
    assert image.shape == (3, 224, 224)
    assert target == 0

