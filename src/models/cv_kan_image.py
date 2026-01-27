"""CV-KAN Image Classifier.

A CV-KAN variant optimized for image classification using:
- Patch-based embedding (ViT-style)
- Optional 2D positional encoding
- Global aggregation for polarization
- Magnitude-based classification

Inherits from BaseCVKAN for log-magnitude centering and pooling.
"""

from typing import Literal

import torch
from torch import nn

from ..configs.model import CVKANConfig
from ..modules.pooling import AttentionPool2d
from ..modules.positional_encoding import (
    Complex2DPositionalEncoding,
    Learnable2DComplexPositionalEncoding,
)
from .base import build_classifier_head
from .cv_kan import CVKAN, CVKANBackbone


class LinearPatchEmbedding(nn.Module):
    """Convert image to patch embeddings in complex space.

    Similar to ViT patch embedding, but projects to complex numbers
    for CV-KAN processing.

    Args:
        img_size: Input image size (height, width) or single int for square
        patch_size: Patch size (height, width) or single int for square
        in_channels: Number of input channels (3 for RGB)
        d_complex: Output complex dimension
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_channels: int = 3,
        d_complex: int = 64,
    ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_h = img_size[0] // patch_size[0]
        self.n_patches_w = img_size[1] // patch_size[1]
        self.n_patches = self.n_patches_h * self.n_patches_w
        self.d_complex = d_complex

        patch_dim = in_channels * patch_size[0] * patch_size[1]

        # Project patches to real and imaginary parts
        self.proj_real = nn.Linear(patch_dim, d_complex)
        self.proj_imag = nn.Linear(patch_dim, d_complex)

        # Initialize for stable training
        nn.init.xavier_uniform_(self.proj_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_imag.weight, gain=0.5)
        nn.init.zeros_(self.proj_real.bias)
        nn.init.zeros_(self.proj_imag.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Convert image to complex patch embeddings.

        Args:
            x: Image tensor of shape (batch, channels, height, width)

        Returns:
            Tuple of:
                - Complex tensor of shape (batch, n_patches, d_complex)
                - Spatial shape tuple (n_patches_h, n_patches_w)
        """
        batch, c, h, w = x.shape
        # print(f"DEBUG: LinearPatchEmbedding input: {x.shape}")

        # Extract patches: (B, C, H, W) -> (B, n_patches, patch_dim)
        # Using unfold (already correct logic theoretically, but dimensions must match)

        # Check simple case: H, W divisible by patch size?
        if h % self.patch_size[0] != 0 or w % self.patch_size[1] != 0:
            print(f"ERROR: Image size {h}x{w} not divisible by patch size {self.patch_size}")

        # Original:
        # x = x.unfold(2, self.patch_size[0], self.patch_size[0])  # (B, C, nH, W, pH)
        # x = x.unfold(3, self.patch_size[1], self.patch_size[1])  # (B, C, nH, nW, pH, pW)
        # x = x.permute(0, 2, 3, 1, 4, 5)  # (B, nH, nW, C, pH, pW)
        # x = x.reshape(batch, self.n_patches, -1)  # (B, n_patches, patch_dim)

        # Retrying to locate error with prints? No, error is "shape '[64, 196, -1]' is invalid for input of size 196608"
        # 196608 is the size of the tensor BEFORE reshape.
        # Target shape: [64, 196, -1]
        # 64 * 196 = 12544
        # 196608 / 12544 = 15.67... NOT INTEGER.

        # Input 'x' to these unfolds is 196608 elements?
        # If x is [64, 3, 32, 32], numel is 196608.
        # 32x32 image, patch size 16?
        # n_patches_h = 32 // 16 = 2
        # n_patches_w = 32 // 16 = 2
        # n_patches = 4.
        # But error says "shape [64, 196, -1]".
        # 196 patches => 14x14 grid.
        # This implies the model thinks img_size is 224 (224/16 = 14).
        # BUT input x comes from CIFAR-10 which is 32x32!

        # ROOT CAUSE FOUND: CVKANImageClassifier default img_size=224, but used on CIFAR (32x32).
        # Need to pass correct img_size to model constructor.

        x = x.unfold(2, self.patch_size[0], self.patch_size[0])
        x = x.unfold(3, self.patch_size[1], self.patch_size[1])
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch, self.n_patches, -1)

        # Project to complex space
        real = self.proj_real(x)
        imag = self.proj_imag(x)
        z = torch.complex(real, imag)

        return z, (self.n_patches_h, self.n_patches_w)


class ConvPatchEmbedding(nn.Module):
    """Deep convolutional patch embedding for better feature extraction.

    Uses a 2-layer convolutional stem instead of a single linear projection:
    1. Conv2d(3 -> d/2, 3x3, stride 2) -> BN -> ReLU
    2. Conv2d(d/2 -> d, 3x3, stride 2) -> BN -> ReLU

    Args:
        img_size: Input image size
        patch_size: Patch size (must be 4 for this implementation to match stride=4 total)
        in_channels: Input channels
        d_complex: Output complex dimension
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 32,
        patch_size: int | tuple[int, int] = 4,  # Designed for patch_size=4 (CIFAR)
        in_channels: int = 3,
        d_complex: int = 64,
    ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        # Determine strides based on patch size (assuming 2 layers)
        # For patch_size=4, we need total stride 4 = 2 * 2
        s1, s2 = 2, 2

        self.n_patches_h = img_size[0] // patch_size[0]
        self.n_patches_w = img_size[1] // patch_size[1]
        self.n_patches = self.n_patches_h * self.n_patches_w
        self.d_complex = d_complex

        dim1 = d_complex // 2
        dim2 = d_complex

        # Real branch stem
        self.conv_real = nn.Sequential(
            nn.Conv2d(in_channels, dim1, kernel_size=3, stride=s1, padding=1, bias=False),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, kernel_size=3, stride=s2, padding=1, bias=False),
            nn.BatchNorm2d(dim2),
            nn.ReLU(inplace=True),
        )

        # Imag branch stem (separate weights to learn phase info independently)
        self.conv_imag = nn.Sequential(
            nn.Conv2d(in_channels, dim1, kernel_size=3, stride=s1, padding=1, bias=False),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, kernel_size=3, stride=s2, padding=1, bias=False),
            nn.BatchNorm2d(dim2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Args:
            x: (B, C, H, W)

        Returns:
            z: (B, N, d_complex)
        """
        real = self.conv_real(x)  # (B, D, H/4, W/4)
        imag = self.conv_imag(x)

        z = torch.complex(real, imag)

        # Flatten spatial dims to patches
        B, D, H, W = z.shape
        z = z.flatten(2)  # (B, D, N) where N=H*W
        z = z.transpose(1, 2)  # (B, N, D)

        return z, (H, W)


class ImageEmbedding(nn.Module):
    """Composite embedding for images: Patch Embedding + Positional Encoding.
    Satisfies Embedding protocol.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        d_complex: int,
        embedding_type: str = "linear",
        pos_encoding: str = "sinusoidal",
    ):
        """Initialize ImageEmbedding."""
        super().__init__()

        # Patch embedding
        if embedding_type == "conv":
            self.patch_embed = ConvPatchEmbedding(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                d_complex=d_complex,
            )
        else:
            self.patch_embed = LinearPatchEmbedding(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                d_complex=d_complex,
            )

        n_patches_h = self.patch_embed.n_patches_h
        n_patches_w = self.patch_embed.n_patches_w

        # Positional encoding
        if pos_encoding == "sinusoidal":
            self.pos_encoding = Complex2DPositionalEncoding(
                d_complex=d_complex,
                max_height=n_patches_h,
                max_width=n_patches_w,
            )
        elif pos_encoding == "learnable":
            self.pos_encoding = Learnable2DComplexPositionalEncoding(
                d_complex=d_complex,
                max_height=n_patches_h,
                max_width=n_patches_w,
            )
        else:
            self.pos_encoding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        z, spatial_shape = self.patch_embed(x)  # (batch, n_patches, d_complex)

        # Apply positional encoding
        if self.pos_encoding is not None:
            z = self.pos_encoding(z, spatial_shape=spatial_shape)

        return z


class ImageClassificationHead(nn.Module):
    """Classification head for images.
    Supports special MLP-based AttentionPool2d.
    """

    def __init__(
        self,
        d_complex: int,
        n_classes: int,
        kan_hidden: int,
        pooling: str,
        dropout: float,
    ):
        """Initialize ImageClassificationHead."""
        super().__init__()
        self.pooling_type = pooling
        self.d_complex = d_complex

        if pooling == "attention":
            # Image-specific attention pool (MLP based)
            self.attention_pool = AttentionPool2d(in_channels=d_complex)
        else:
            # Fallback to standard pooling logic (placeholder)
            pass

        self.classifier = build_classifier_head(
            d_complex=d_complex,
            n_classes=n_classes,
            hidden_dim=kan_hidden * 2,  # Note: Image model used *2 in original code
            dropout=dropout,
        )

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "attention":
            # Apply AttentionPool2d to Real and Imag separately
            real = z.real
            imag = z.imag

            pooled_real = self.attention_pool(real)
            pooled_imag = self.attention_pool(imag)
            return torch.complex(pooled_real, pooled_imag)

        if self.pooling_type == "mean":
            return z.mean(dim=1)
        if self.pooling_type == "max":
            # Simple max magnitude
            mag = torch.abs(z)
            _, indices = mag.max(dim=1)
            batch_indices = torch.arange(z.size(0), device=z.device).unsqueeze(-1)
            dim_indices = torch.arange(z.size(2), device=z.device).unsqueeze(0)
            return z[batch_indices, indices, dim_indices]
        raise ValueError(f"Unknown pooling: {self.pooling_type}")

    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> dict:
        pooled = self._pool(z)
        features = torch.abs(pooled)
        logits = self.classifier(features)

        return {"logits": logits, "pooled": pooled, "features": features}


class CVKANImageClassifier(CVKAN):
    """CV-KAN model for image classification (Composition-based)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_complex: int = 64,
        n_layers: int = 6,
        n_classes: int = 1000,
        kan_hidden: int = 32,
        pos_encoding: Literal["sinusoidal", "learnable"] | None = "sinusoidal",
        pooling: Literal["mean", "max", "attention"] = "mean",
        embedding_type: Literal["linear", "conv"] = "conv",
        center_magnitudes: bool = True,
        dropout: float = 0.0,
        normalization: Literal["none", "layer", "batch", "rms"] = "none",
        **kwargs,  # Ignore extra args
    ):
        # 1. Config (for Backbone)
        config = CVKANConfig(
            d_complex=d_complex,
            n_layers=n_layers,
            kan_hidden=kan_hidden,
            center_magnitudes=center_magnitudes,
            dropout=dropout,
            input_type="real",  # Images are real
            normalization=normalization,
            **kwargs,
        )

        # 2. Components
        embedding = ImageEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_complex=d_complex,
            embedding_type=embedding_type,
            pos_encoding=pos_encoding,
        )

        backbone = CVKANBackbone(config)

        # Use specialized ImageClassificationHead if pooling is attention (for MLP capability)
        # or if we want to match legacy classifier width (kan_hidden * 2)
        head = ImageClassificationHead(
            d_complex=d_complex,
            n_classes=n_classes,
            kan_hidden=kan_hidden,
            pooling=pooling,
            dropout=dropout,
        )

        super().__init__(embedding, backbone, head)
