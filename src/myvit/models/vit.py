"""Vision Transformer 的教学版实现。

本文件从较基础的 PyTorch 模块搭建 ViT，重点展示以下数据流：

    图片 -> Patch Token -> 添加 CLS/位置编码 -> Transformer Blocks -> 分类头

默认工厂函数 ``vit_tiny_patch16_224`` 对应常见的 ViT-Tiny/16 配置：

* 输入分辨率：224 x 224
* Patch 大小：16 x 16
* Token 维度：192
* Transformer Block：12 层
* 注意力头数：3
* 参数量：约 5.7M（1000 个分类类别时）
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


Size2D = tuple[int, int]


def _to_2tuple(value: int | Sequence[int]) -> Size2D:
    """把整数或长度为 2 的序列规范化为 ``(height, width)``。"""

    if isinstance(value, int):
        return value, value

    if len(value) != 2:
        raise ValueError(f"期望长度为 2 的尺寸，实际收到：{value!r}")

    return int(value[0]), int(value[1])


class PatchEmbedding(nn.Module):
    """把二维图片切成 Patch，并将每个 Patch 映射为一个 Token。

    使用 ``Conv2d(kernel_size=patch_size, stride=patch_size)`` 同时完成：

    1. 将图片划分为互不重叠的 Patch；
    2. 将每个 Patch 从像素空间投影到 ``embed_dim`` 维特征空间。

    输入形状：``[batch, channels, image_height, image_width]``
    输出形状：``[batch, num_patches, embed_dim]``
    """

    def __init__(
        self,
        image_size: int | Sequence[int] = 224,
        patch_size: int | Sequence[int] = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()

        self.image_size = _to_2tuple(image_size)
        self.patch_size = _to_2tuple(patch_size)

        image_height, image_width = self.image_size
        patch_height, patch_width = self.patch_size

        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError(
                "图片尺寸必须能被 Patch 尺寸整除，"
                f"实际收到 image_size={self.image_size}, patch_size={self.patch_size}"
            )

        self.grid_size = (
            image_height // patch_height,
            image_width // patch_width,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 卷积输出的每一个空间位置，对应原图片中的一个 Patch Token。
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "PatchEmbedding 期望输入形状为 [B, C, H, W]，"
                f"实际收到 {tuple(images.shape)}"
            )

        actual_size = tuple(images.shape[-2:])
        if actual_size != self.image_size:
            raise ValueError(
                f"模型期望图片尺寸为 {self.image_size}，实际收到 {actual_size}。"
                "请先缩放图片，或使用对应分辨率的位置编码。"
            )

        # [B, C, H, W] -> [B, D, H/P, W/P]
        tokens = self.projection(images)

        # 展平二维 Patch 网格，再把特征维 D 移到最后：
        # [B, D, H/P, W/P] -> [B, D, N] -> [B, N, D]
        tokens = tokens.flatten(start_dim=2).transpose(1, 2)
        return tokens


class MultiHeadSelfAttention(nn.Module):
    """标准多头自注意力（Multi-Head Self-Attention）。

    每个 Token 同时生成 Query、Key 和 Value。多个注意力头在较小的
    ``head_dim`` 子空间中独立计算注意力，最后再拼接并线性投影。

    输入与输出形状相同，均为 ``[batch, num_tokens, embed_dim]``。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除"
            )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # 一次线性变换同时产生 Q、K、V，输出维度是 3 * embed_dim。
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, num_tokens, embed_dim = tokens.shape

        # [B, N, D] -> [B, N, 3, heads, head_dim]
        qkv = self.qkv(tokens).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )

        # 把 Q/K/V 放到最前面，便于拆分：
        # [B, N, 3, heads, head_dim] -> [3, B, heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        # 每个注意力头都会得到一个 [N, N] 的矩阵，表示 Token 两两之间的关系。
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        # [B, heads, N, head_dim] -> [B, N, heads, head_dim] -> [B, N, D]
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        output = self.projection(output)
        output = self.projection_dropout(output)
        return output


class MLP(nn.Module):
    """Transformer Block 中逐 Token 计算的前馈网络。"""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.layers(tokens)


class DropPath(nn.Module):
    """按样本随机丢弃完整残差分支，也称 Stochastic Depth。

    与普通 Dropout 丢弃单个元素不同，DropPath 会让某个样本的整条残差分支
    暂时失效。推理模式下不会丢弃任何分支。
    """

    def __init__(self, drop_probability: float = 0.0) -> None:
        super().__init__()

        if not 0.0 <= drop_probability < 1.0:
            raise ValueError("drop_probability 必须满足 0 <= p < 1")

        self.drop_probability = drop_probability

    def forward(self, inputs: Tensor) -> Tensor:
        if self.drop_probability == 0.0 or not self.training:
            return inputs

        keep_probability = 1.0 - self.drop_probability

        # 随机掩码只在 batch 维度变化，其余维度广播，从而丢弃整条残差分支。
        mask_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            mask_shape,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        binary_mask = random_tensor.floor()

        # 除以保留概率，使训练时输出的数学期望与推理时保持一致。
        return inputs.div(keep_probability) * binary_mask


class TransformerEncoderBlock(nn.Module):
    """采用 Pre-Norm 结构的 ViT Transformer Block。

    计算过程：

    ``x = x + DropPath(Attention(LayerNorm(x)))``

    ``x = x + DropPath(MLP(LayerNorm(x)))``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_probability: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path_probability)

        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.drop_path2 = DropPath(drop_path_probability)

    def forward(self, tokens: Tensor) -> Tensor:
        # Pre-Norm：先归一化，再进入注意力/MLP，通常比 Post-Norm 更易训练深层网络。
        tokens = tokens + self.drop_path1(self.attention(self.norm1(tokens)))
        tokens = tokens + self.drop_path2(self.mlp(self.norm2(tokens)))
        return tokens


class VisionTransformer(nn.Module):
    """用于图像分类的 Vision Transformer。"""

    def __init__(
        self,
        image_size: int | Sequence[int] = 224,
        patch_size: int | Sequence[int] = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if depth <= 0:
            raise ValueError("depth 必须大于 0")
        if num_classes < 0:
            raise ValueError("num_classes 不能小于 0")

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embedding.num_patches

        # CLS Token 用于汇聚整张图片的信息，最终送入分类头。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 额外的 1 对应 CLS Token，所以位置编码长度是 num_patches + 1。
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.position_dropout = nn.Dropout(dropout)

        # 越靠后的 Block 使用越高的 DropPath 概率，是常见的 ViT 训练策略。
        drop_path_probabilities = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_probability=drop_path_probabilities[index],
                )
                for index in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化位置编码、CLS Token 和各层权重。"""

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        """为线性层、卷积层和 LayerNorm 设置稳定的初始值。"""

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, images: Tensor) -> Tensor:
        """提取每张图片的 CLS 特征，返回形状 ``[B, embed_dim]``。"""

        # [B, C, H, W] -> [B, num_patches, embed_dim]
        tokens = self.patch_embedding(images)

        # CLS 参数只有一个模板，需要在 batch 维复制，但不会真正拷贝底层数据。
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        tokens = tokens + self.position_embedding
        tokens = self.position_dropout(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # 第 0 个 Token 是 CLS Token，作为整张图片的最终表示。
        return tokens[:, 0]

    def forward(self, images: Tensor) -> Tensor:
        features = self.forward_features(images)
        logits = self.head(features)
        return logits


def vit_tiny_patch16_224(
    *,
    num_classes: int = 1000,
    **kwargs: object,
) -> VisionTransformer:
    """创建 ViT-Tiny/16 模型。

    ``kwargs`` 可用于覆盖 dropout、drop_path_rate 等训练相关参数。
    模型在 ``num_classes=1000`` 时约有 5.7M 参数。
    """

    model_kwargs: dict[str, object] = {
        "image_size": 224,
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "num_classes": num_classes,
    }
    model_kwargs.update(kwargs)
    return VisionTransformer(**model_kwargs)


def count_trainable_parameters(model: nn.Module) -> int:
    """返回模型中需要梯度的参数数量，方便核对模型规模。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

