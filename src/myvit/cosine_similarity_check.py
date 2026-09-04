"""检查三 CLS 模型最终表征的相似度和可学习融合权重。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from myvit.data import create_imagenette_datasets
from myvit.models import UpdatedVisionTransformer
from myvit.updated_training import create_updated_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查最终三个 CLS 表征是否趋于正交")
    parser.add_argument("checkpoint", type=Path, help="三 CLS 模型 checkpoint")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/imagenette2-320"),
        help="Imagenette 数据目录",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--print-values",
        type=int,
        default=0,
        help="打印第一个样本中每个 CLS Token 的前 N 个数值；默认不打印",
    )
    return parser


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> UpdatedVisionTransformer:
    """加载新模型；旧版三 CLS checkpoint 会用等权重初始化新增参数。"""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = create_updated_model()
    incompatible = model.load_state_dict(payload["model_state"], strict=False)

    # 旧 checkpoint 没有可学习权重参数，此时全零 logits 对应原来的三等分平均。
    allowed_missing = {"cls_token_weight_logits"}
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "checkpoint 与三 CLS 模型结构不匹配："
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if "cls_token_weight_logits" in incompatible.missing_keys:
        print("checkpoint_note: legacy checkpoint, cls weights initialized to 1/3")

    return model.to(device).eval()


@torch.inference_mode()
def summarize_final_cls_tokens(
    model: UpdatedVisionTransformer,
    data_loader: DataLoader,
    device: torch.device,
    *,
    print_values: int = 0,
) -> tuple[Tensor, float]:
    """按样本平均最终三 CLS 表征的余弦 Gram 矩阵和正交损失。"""

    cosine_sum = torch.zeros(3, 3, dtype=torch.float64)
    orthogonality_loss_sum = 0.0
    samples = 0
    printed = False

    for images, _ in data_loader:
        images = images.to(device, non_blocking=True)
        cls_tokens = model.forward_cls_tokens(images)
        normalized = F.normalize(cls_tokens, p=2, dim=-1)
        cosine = normalized @ normalized.transpose(-1, -2)

        batch_size = images.shape[0]
        cosine_sum += cosine.sum(dim=0).double().cpu()
        orthogonality_loss_sum += (
            float(model.orthogonality_loss(cls_tokens).item())
            * batch_size
        )
        samples += batch_size

        if print_values > 0 and not printed:
            values = cls_tokens[0, :, :print_values].cpu()
            print(f"first_sample_cls_values (shape={tuple(values.shape)}):")
            print(values)
            printed = True

    if samples == 0:
        raise ValueError("验证集为空")
    return cosine_sum / samples, orthogonality_loss_sum / samples


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")
    if args.print_values < 0:
        raise ValueError("print-values 不能小于 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    _, validation_dataset = create_imagenette_datasets(args.data_dir)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
    )

    cosine_matrix, orthogonality_loss = summarize_final_cls_tokens(
        model,
        validation_loader,
        device,
        print_values=args.print_values,
    )
    weights = model.cls_token_weights().detach().cpu()

    print(f"device: {device}")
    print(f"cls_weights: {weights.tolist()} (sum={weights.sum().item():.6f})")
    print("mean_final_cls_cosine_matrix:")
    print(cosine_matrix)
    print(f"mean_orthogonality_loss: {orthogonality_loss:.6f}")


if __name__ == "__main__":
    main()
