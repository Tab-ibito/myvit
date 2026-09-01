"""ViT 核心结构测试。"""

import pytest
import torch

from myvit.models.vit import (
    MultiHeadSelfAttention,
    PatchEmbedding,
    count_trainable_parameters,
    vit_tiny_patch16_224,
)


def test_patch_embedding_output_shape() -> None:
    """224 图片经过 16 大小的 Patch 后，应产生 14 x 14 个 Token。"""

    layer = PatchEmbedding(image_size=224, patch_size=16, embed_dim=192)
    images = torch.randn(2, 3, 224, 224)

    tokens = layer(images)

    assert tokens.shape == (2, 196, 192)
    assert layer.grid_size == (14, 14)
    assert layer.num_patches == 196


def test_patch_embedding_rejects_wrong_image_size() -> None:
    """固定位置编码版本应尽早拒绝错误的输入分辨率。"""

    layer = PatchEmbedding(image_size=224, patch_size=16, embed_dim=192)
    images = torch.randn(1, 3, 256, 256)

    with pytest.raises(ValueError, match="模型期望图片尺寸"):
        layer(images)


def test_attention_preserves_shape_and_supports_backward() -> None:
    """自注意力不能改变 Token 形状，并且必须能正常反向传播。"""

    attention = MultiHeadSelfAttention(embed_dim=192, num_heads=3)
    tokens = torch.randn(2, 197, 192, requires_grad=True)

    output = attention(tokens)
    output.mean().backward()

    assert output.shape == tokens.shape
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_vit_tiny_output_shape_parameter_count_and_backward() -> None:
    """完整 Tiny 模型应输出 1000 类 logits，并具有预期参数规模。"""

    model = vit_tiny_patch16_224()
    images = torch.randn(2, 3, 224, 224)

    logits = model(images)
    logits.mean().backward()

    assert logits.shape == (2, 1000)
    assert count_trainable_parameters(model) == 5_717_416
    assert model.cls_token.grad is not None


def test_num_classes_can_be_changed() -> None:
    """分类头应能适配不同数据集的类别数量。"""

    model = vit_tiny_patch16_224(num_classes=10)
    images = torch.randn(1, 3, 224, 224)

    logits = model(images)

    assert logits.shape == (1, 10)

