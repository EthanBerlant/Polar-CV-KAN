"""
CV-KAN Image Classifier.

A CV-KAN variant optimized for image classification using:
- Patch-based embedding (ViT-style)
- Optional 2D positional encoding
- Global aggregation for polarization
- Magnitude-based classification

Inherits from BaseCVKAN for log-magnitude centering and pooling.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

from .base import BaseCVKAN, build_classifier_head
from ..modules.positional_encoding import (
    Complex2DPositionalEncoding,
    Learnable2DComplexPositionalEncoding,
)
from ..modules.pooling import AttentionPool2d


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


class DeepPatchEmbedding(nn.Module):
    """
    Deep convolutional patch embedding for better feature extraction.
    
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
        img_size: int | Tuple[int, int] = 32,
        patch_size: int | Tuple[int, int] = 4, # Designed for patch_size=4 (CIFAR)
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
            nn.ReLU(inplace=True)
        )
        
        # Imag branch stem (separate weights to learn phase info independently)
        self.conv_imag = nn.Sequential(
            nn.Conv2d(in_channels, dim1, kernel_size=3, stride=s1, padding=1, bias=False),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, kernel_size=3, stride=s2, padding=1, bias=False),
            nn.BatchNorm2d(dim2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            z: (B, N, d_complex)
        """
        real = self.conv_real(x) # (B, D, H/4, W/4)
        imag = self.conv_imag(x)
        
        z = torch.complex(real, imag)
        
        # Flatten spatial dims to patches
        B, D, H, W = z.shape
        z = z.flatten(2).transpose(1, 2) # (B, H*W, D)
        
        return z, (H, W)


class CVKANImageClassifier(BaseCVKAN):
    """
    CV-KAN model for image classification.
    
    Architecture:
    1. Patch embedding: Image -> patches -> complex embeddings
    2. Optional 2D positional encoding
    3. Stack of CV-KAN layers with global aggregation
    4. Log-magnitude centering (from BaseCVKAN)
    5. Global pooling -> magnitude features -> classifier
    
    Args:
        img_size: Input image size
        patch_size: Patch size for embedding
        in_channels: Number of input channels (3 for RGB)
        d_complex: Complex representation dimension
        n_layers: Number of CV-KAN layers
        n_classes: Number of output classes
        kan_hidden: Hidden size for KAN MLPs
        pos_encoding: 'sinusoidal', 'learnable', or None
        pooling: 'mean', 'max', or 'attention'
        center_magnitudes: Whether to center log-magnitudes (recommended)
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
        pos_encoding: Optional[Literal['sinusoidal', 'learnable']] = 'sinusoidal',
        pooling: Literal['mean', 'max', 'attention'] = 'mean',
        embedding_type: Literal['standard', 'deep'] = 'standard',
        center_magnitudes: bool = True,
    ):
        # Initialize base class
        super().__init__(
            d_complex=d_complex,
            n_layers=n_layers,
            kan_hidden=kan_hidden,
            pooling=pooling,
            center_magnitudes=center_magnitudes,
        )
        
        # Patch embedding
        if embedding_type == 'deep':
            self.patch_embed = DeepPatchEmbedding(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                d_complex=d_complex,
            )
        else:
            self.patch_embed = PatchEmbedding(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                d_complex=d_complex,
            )
        n_patches_h = self.patch_embed.n_patches_h
        n_patches_w = self.patch_embed.n_patches_w
        
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
        
        # Classification head
        self.classifier = build_classifier_head(
            d_complex=d_complex,
            n_classes=n_classes,
            hidden_dim=kan_hidden * 2,
        )
        
        # Pooling
        if pooling == 'attention':
            self.attention_pool = AttentionPool2d(in_channels=d_complex)
        
        self.spatial_shape = (n_patches_h, n_patches_w)
    
    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        """Override pooling to support attention pooling."""
        if self.pooling_strategy == 'attention':
            # z is (B, N, d_complex) - complex
            # We want to pool based on magnitude or learn attention over complex?
            # AttentionPool2d expects Real input.
            # Let's pool the Real and Imag parts separately with shared attention or operate on Magnitude?
            
            # Alternative: Attention over complex vectors using magnitude for scoring?
            # Or just separate attention for Real/Imag.
            
            # Let's assume we treat Real and Imag as 2*D features for attention scoring, 
            # but we want to return a Complex result.
            
            # For simplicity: Use AttentionPool2d on the concatenation of Real/Imag,
            # but that gives a Real output.
            
            # Better strategy: Apply AttentionPool2d to Real and Imag separately 
            # (which means different attention for real/imag components - flexible).
            
            real = z.real
            imag = z.imag
            
            # We need to share the attention weights if we want phase coherence?
            # But AttentionPool2d computes weights internally.
            
            # Let's use the provided AttentionPool2d on Real and Imag separately for now.
            pooled_real = self.attention_pool(real)
            pooled_imag = self.attention_pool(imag)
            return torch.complex(pooled_real, pooled_imag)
            
        return super()._pool(z)
    
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
                - features: Magnitude features (if return_features=True)
                - pooled: Pooled complex representation (if return_features=True)
        """
        # Patch embedding
        z, spatial_shape = self.patch_embed(x)  # (batch, n_patches, d_complex)
        
        # Apply positional encoding
        if self.pos_encoding is not None:
            z = self.pos_encoding(z, spatial_shape=spatial_shape)
        
        # Apply CV-KAN layers with magnitude centering (from base class)
        z = self._apply_layers(z)
        
        # Pool across patches (from base class)
        pooled = self._pool(z)
        
        # Extract magnitude features (from base class)
        features = self._extract_features(pooled)
        
        # Classify
        logits = self.classifier(features)
        
        output = {'logits': logits}
        if return_features:
            output['features'] = features
            output['pooled'] = pooled
        
        return output
