from __future__ import annotations

from myvit.models import VisionTransformer

# 调用一个小型的 Vision Transformer，参数数量大约570万
def create_tiny_model() -> VisionTransformer:
    return VisionTransformer (
        image_size=224,
        patch_size = 16,
        embed_dim = 192,
        depth = 12,
        num_heads = 3,
        num_classes = 10,
    )
