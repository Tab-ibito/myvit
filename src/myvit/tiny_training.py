from __future__ import annotations

from myvit.models import VisionTransformer
from myvit.data.imagenette import create_imagenette_loaders, create_imagenette_datasets

# 调用一个小型的 Vision Transformer，参数数量大约570万
def create_tiny_model() -> VisionTransformer:
    return VisionTransformer (
        image_size=224,
        patch_size = 16,
        embed_dim = 192,
        depth = 12,
        num_heads = 3,
        num_classes = 1000,
    )
