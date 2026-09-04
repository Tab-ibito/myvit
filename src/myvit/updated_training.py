from __future__ import annotations

from myvit.models import UpdatedVisionTransformer


# 调用改进过的 ViT
def create_updated_model(
    *,
    orthogonality_lambda: float = 0.01,
) -> UpdatedVisionTransformer:
    """创建三 CLS 实验模型；默认使用 0.01 的正交损失系数。"""

    return UpdatedVisionTransformer(
        image_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=10,
        orthogonality_lambda=orthogonality_lambda,
    )
