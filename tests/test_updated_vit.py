"""三 CLS ViT、正交损失与可学习融合权重测试。"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from myvit.debug_training import make_pattern_dataset
from myvit.engine import train_one_epoch
from myvit.models import UpdatedVisionTransformer


def create_small_updated_model() -> UpdatedVisionTransformer:
    """创建适合 CPU 单元测试的小型三 CLS 模型。"""

    return UpdatedVisionTransformer(
        image_size=32,
        patch_size=8,
        num_classes=4,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
    )


def test_orthogonality_loss_for_orthogonal_and_identical_tokens() -> None:
    """正交向量损失应为 0，完全相同的单位向量损失应为 1。"""

    orthogonal_tokens = torch.eye(3).unsqueeze(0)
    identical_tokens = torch.tensor([[[1.0, 0.0]] * 3])

    assert UpdatedVisionTransformer.orthogonality_loss(
        orthogonal_tokens
    ).item() == pytest.approx(0.0, abs=1e-7)
    assert UpdatedVisionTransformer.orthogonality_loss(
        identical_tokens
    ).item() == pytest.approx(1.0, abs=1e-7)


def test_three_cls_weights_are_normalized_and_trainable() -> None:
    """融合权重初始均分，并能从分类目标接收到梯度。"""

    torch.manual_seed(0)
    model = create_small_updated_model()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])

    initial_weights = model.cls_token_weights()
    assert initial_weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    logits, orthogonality_loss = model.forward_with_orthogonality_loss(images)
    objective_loss = (
        nn.CrossEntropyLoss()(logits, targets)
        + model.orthogonality_lambda * orthogonality_loss
    )
    objective_loss.backward()

    assert logits.shape == (2, 4)
    assert model.cls_token_weight_logits.grad is not None
    assert torch.isfinite(model.cls_token_weight_logits.grad).all()
    assert model.cls_token_1.grad is not None


def test_training_adds_and_reports_orthogonality_penalty() -> None:
    """训练总目标应等于分类损失加 lambda 乘正交损失。"""

    torch.manual_seed(0)
    model = create_small_updated_model()
    loader = DataLoader(make_pattern_dataset(8, seed=0), batch_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weights_before = model.cls_token_weight_logits.detach().clone()

    metrics = train_one_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
    )

    assert metrics.orthogonality_loss is not None
    assert metrics.objective_loss is not None
    assert metrics.objective_loss == pytest.approx(
        metrics.loss + model.orthogonality_lambda * metrics.orthogonality_loss,
        rel=1e-5,
    )
    assert not torch.equal(weights_before, model.cls_token_weight_logits.detach())
