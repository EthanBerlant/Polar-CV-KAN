"""
CV-KAN Image Classifier.

A CV-KAN variant optimized for image classification using:
- Patch-based embedding (ViT-style)
- Optional 2D positional encoding
- Local or global aggregation
- Magnitude-based classification

The patch embedding converts image pixels to complex representations,
and the model leverages CV-KAN's phase mechanics for feature extraction.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

from ..modules.polarizing_block import PolarizingBlock
from ..modules.aggregation import GlobalMeanAggregation, LocalWindowAggregation
from ..modules.positional_encoding import (
    Complex2DPositionalEncoding,
    Learnable2DComplexPositionalEncoding,
)


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings in complex space.
    
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
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 16,
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert image to complex patch embeddings.
        
        Args:
            x: Image tensor of shape (batch, channels, height, width)
        
        Returns:
            Tuple of:
                - Complex tensor of shape (batch, n_patches, d_complex)
                - Spatial shape tuple (n_patches_h, n_patches_w)
        """
        batch, c, h, w = x.shape
        
        # Extract patches: (B, C, H, W) -> (B, n_patches, patch_dim)
        x = x.unfold(2, self.patch_size[0], self.patch_size[0])  # (B, C, nH, W, pH)
        x = x.unfold(3, self.patch_size[1], self.patch_size[1])  # (B, C, nH, nW, pH, pW)
        x = x.permute(0, 2, 3, 1, 4, 5)  # (B, nH, nW, C, pH, pW)
        x = x.reshape(batch, self.n_patches, -1)  # (B, n_patches, patch_dim)
        
        # Project to complex space
        real = self.proj_real(x)
        imag = self.proj_imag(x)
        z = torch.complex(real, imag)
        
        return z, (self.n_patches_h, self.n_patches_w)


class CVKANImageClassifier(nn.Module):
    """
    CV-KAN model for image classification.
    
    Architecture:
    1. Patch embedding: Image -> patches -> complex embeddings
    2. Optional 2D positional encoding
    3. Stack of CV-KAN layers with configurable aggregation
    4. Global pooling -> magnitude -> classifier
    
    Args:
        img_size: Input image size
        patch_size: Patch size for embedding
        in_channels: Number of input channels (3 for RGB)
        d_complex: Complex representation dimension
        n_layers: Number of CV-KAN layers
        n_classes: Number of output classes
        kan_hidden: Hidden size for KAN MLPs
        aggregation: 'global' or 'local' aggregation strategy
        local_kernel_size: Kernel size for local aggregation
        pos_encoding: 'sinusoidal', 'learnable', or None
        pooling: 'mean', 'max', or 'cls' (if using CLS token)
        use_cls_token: Whether to prepend a learnable CLS token
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_complex: int = 64,
        n_layers: int = 6,
        n_classes: int = 1000,
        kan_hidden: int = 32,
        aggregation: Literal['global', 'local'] = 'local',
        local_kernel_size: int = 3,
        pos_encoding: Optional[Literal['sinusoidal', 'learnable']] = 'sinusoidal',
        pooling: Literal['mean', 'max', 'cls'] = 'mean',
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.d_complex = d_complex
        self.pooling = pooling
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_complex=d_complex,
        )
        n_patches_h = self.patch_embed.n_patches_h
        n_patches_w = self.patch_embed.n_patches_w
        
        # CLS token (optional)
        if use_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, dtype=torch.cfloat) * 0.02
            ).expand(1, 1, d_complex).clone()
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, d_complex, dtype=torch.cfloat) * 0.02
            )
        
        # Positional encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoding = Complex2DPositionalEncoding(
                d_complex=d_complex,
                max_height=n_patches_h,
                max_width=n_patches_w,
            )
        elif pos_encoding == 'learnable':
            self.pos_encoding = Learnable2DComplexPositionalEncoding(
                d_complex=d_complex,
                max_height=n_patches_h,
                max_width=n_patches_w,
            )
        else:
            self.pos_encoding = None
        
        # Aggregation strategy
        if aggregation == 'local':
            self.aggregation = LocalWindowAggregation(
                kernel_size=local_kernel_size,
                stride=1,
            )
        else:
            self.aggregation = GlobalMeanAggregation()
        
        # CV-KAN layers
        self.layers = nn.ModuleList([
            PolarizingBlock(d_complex, kan_hidden, aggregation=self.aggregation)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_complex, kan_hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(kan_hidden * 2, n_classes),
        )
        
        self.spatial_shape = (n_patches_h, n_patches_w)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
        """
        Forward pass for image classification.
        
        Args:
            x: Image tensor of shape (batch, channels, height, width)
            return_features: If True, return intermediate features
        
        Returns:
            Dictionary with:
                - logits: Classification logits (batch, n_classes)
                - features: Pooled features (if return_features=True)
        """
        batch = x.shape[0]
        
        # Patch embedding
        z, spatial_shape = self.patch_embed(x)  # (batch, n_patches, d_complex)
        
        # Add CLS token if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch, -1, -1)
            z = torch.cat([cls_tokens, z], dim=1)
        
        # Apply positional encoding
        if self.pos_encoding is not None:
            if self.use_cls_token:
                # Apply pos encoding only to patch tokens
                z_patches = self.pos_encoding(
                    z[:, 1:], 
                    spatial_shape=spatial_shape
                )
                z = torch.cat([z[:, :1], z_patches], dim=1)
            else:
                z = self.pos_encoding(z, spatial_shape=spatial_shape)
        
        # Apply CV-KAN layers
        for layer in self.layers:
            z = layer(z)
        
        # Pool
        if self.pooling == 'cls' and self.use_cls_token:
            pooled = z[:, 0]  # CLS token
        elif self.pooling == 'mean':
            if self.use_cls_token:
                pooled = z[:, 1:].mean(dim=1)
            else:
                pooled = z.mean(dim=1)
        elif self.pooling == 'max':
            if self.use_cls_token:
                z_patches = z[:, 1:]
            else:
                z_patches = z
            mags = torch.abs(z_patches)
            max_idx = mags.sum(dim=-1).argmax(dim=1, keepdim=True)
            pooled = torch.gather(
                z_patches, 1, 
                max_idx.unsqueeze(-1).expand(-1, -1, self.d_complex)
            ).squeeze(1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify using magnitudes
        features = torch.abs(pooled)
        logits = self.classifier(features)
        
        output = {'logits': logits}
        if return_features:
            output['features'] = pooled
            output['feature_magnitudes'] = features
        
        return output
