"""带三个可学习 CLS Token 的 ViT 实验模型。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .vit import PatchEmbedding, TransformerEncoderBlock


class UpdatedVisionTransformer(nn.Module):
    """使用三个 CLS Token 进行可学习加权融合的 Vision Transformer。

    三个 CLS Token 会经过同一组 Transformer Block。在训练阶段，可以对最终
    CLS 表征施加正交惩罚，鼓励它们提取互补信息。``orthogonality_lambda`` 只
    保存正交项的默认系数，实际损失由训练循环组合：

    ``total_loss = classification_loss + lambda * orthogonality_loss``。
    """

    num_cls_tokens = 3

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
        orthogonality_lambda: float = 0.01,
    ) -> None:
        super().__init__()

        if depth <= 0:
            raise ValueError("depth 必须大于 0")
        if num_classes < 0:
            raise ValueError("num_classes 不能小于 0")
        if orthogonality_lambda < 0:
            raise ValueError("orthogonality_lambda 不能小于 0")

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.orthogonality_lambda = float(orthogonality_lambda)

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embedding.num_patches

        # 三个独立参数是 CLS Token 的初始模板；经过 Transformer 后，它们会得到
        # 与当前图片相关的三个最终表示。
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_3 = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 保存未归一化权重（logits），前向时经过 softmax。这样权重始终为正且
        # 总和为 1；全零初始化对应初始权重各为 1/3，又允许训练中自动调整。
        self.cls_token_weight_logits = nn.Parameter(torch.zeros(self.num_cls_tokens))

        # 额外的 3 对应三个 CLS Token，所以位置编码长度是 num_patches + 3。
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + self.num_cls_tokens, embed_dim)
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
        nn.init.trunc_normal_(self.cls_token_1, std=0.02)
        nn.init.trunc_normal_(self.cls_token_2, std=0.02)
        nn.init.trunc_normal_(self.cls_token_3, std=0.02)
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

    def cls_token_weights(self) -> Tensor:
        """返回归一化后的三个融合权重，形状为 ``[3]``。"""

        return self.cls_token_weight_logits.softmax(dim=0)

    def forward_cls_tokens(self, images: Tensor) -> Tensor:
        """返回 Transformer 最后一层的三个 CLS 表征，形状为 ``[B, 3, D]``。"""

        # [B, C, H, W] -> [B, num_patches, embed_dim]
        tokens = self.patch_embedding(images)

        # CLS 参数只有一个模板，需要在 batch 维复制，但不会真正拷贝底层数据。
        cls_tokens_1 = self.cls_token_1.expand(tokens.shape[0], -1, -1)
        cls_tokens_2 = self.cls_token_2.expand(tokens.shape[0], -1, -1)
        cls_tokens_3 = self.cls_token_3.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens_1, cls_tokens_2, cls_tokens_3, tokens), dim=1)

        tokens = tokens + self.position_embedding
        tokens = self.position_dropout(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # 这里返回的是与图片内容交互后的表示，而不是模型中保存的初始 CLS 参数。
        return tokens[:, : self.num_cls_tokens]

    @staticmethod
    def orthogonality_loss(cls_tokens: Tensor) -> Tensor:
        """计算三个 CLS 表征两两余弦相似度的平方均值。

        先把每个 CLS 表征归一化，再计算 ``[3, 3]`` Gram 矩阵。对角线恒为 1，
        因此只惩罚非对角项。使用平方后，正相关和负相关都会受到惩罚；完全正交
        时损失为 0，三个完全同向的非零向量对应损失为 1。
        """

        if cls_tokens.ndim != 3:
            raise ValueError(
                "cls_tokens 应为 [batch, num_cls_tokens, embed_dim]，"
                f"实际收到 {tuple(cls_tokens.shape)}"
            )
        num_cls_tokens = cls_tokens.shape[1]
        if num_cls_tokens < 2:
            raise ValueError("计算正交损失至少需要两个 CLS Token")

        normalized_tokens = F.normalize(cls_tokens, p=2, dim=-1)
        gram = normalized_tokens @ normalized_tokens.transpose(-1, -2)
        identity = torch.eye(
            num_cls_tokens,
            device=cls_tokens.device,
            dtype=cls_tokens.dtype,
        ).unsqueeze(0)
        off_diagonal = gram - identity

        # 分母是有序的非对角元素数量，例如三个 Token 时为 3 * 2 = 6。
        return off_diagonal.square().sum(dim=(-2, -1)).mean() / (
            num_cls_tokens * (num_cls_tokens - 1)
        )

    def forward_features_with_cls_tokens(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """同时返回加权图片特征和正交损失需要的三个 CLS 表征。"""

        cls_tokens = self.forward_cls_tokens(images)
        weights = self.cls_token_weights().view(1, self.num_cls_tokens, 1)
        features = (cls_tokens * weights).sum(dim=1)
        return features, cls_tokens

    def forward_features(self, images: Tensor) -> Tensor:
        """提取可学习加权后的图片特征，形状为 ``[B, embed_dim]``。"""

        features, _ = self.forward_features_with_cls_tokens(images)
        return features

    def forward_with_orthogonality_loss(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """单次前向同时返回分类 logits 和未乘 lambda 的正交损失。"""

        features, cls_tokens = self.forward_features_with_cls_tokens(images)
        logits = self.head(features)
        return logits, self.orthogonality_loss(cls_tokens)

    def forward(self, images: Tensor) -> Tensor:
        features = self.forward_features(images)
        logits = self.head(features)
        return logits
