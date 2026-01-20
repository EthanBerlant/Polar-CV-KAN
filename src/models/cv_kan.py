"""
CV-KAN: Full model for sequence classification.

Architecture:
- Input embedding → complex projection
- Stack of PolarizingBlocks (no normalization - magnitudes encode attention)
- Log-magnitude centering to prevent drift
- Global pooling → classifier head

Note: Traditional normalization (LayerNorm, RMSNorm) is intentionally NOT used
because magnitudes in CV-KAN encode attention-like information. Normalizing
would destroy this signal. Instead, we use log-magnitude centering which
preserves relative magnitudes while preventing absolute scale drift.

Inherits from BaseCVKAN for shared functionality.
"""

from typing import Literal

import torch
import torch.nn as nn

from ..modules.multi_head import (
    EmergentHeadsPolarizing,
    FactoredHeadsPolarizing,
    PhaseOffsetPolarizing,
)
from ..modules.phase_attention import PhaseAttentionBlock
from .base import BaseCVKAN, ComplexEmbedding, build_classifier_head


class ComplexInputEmbedding(nn.Module):
    """
    Handle complex input directly (for the synthetic task).

    Optionally applies a learnable linear transform in complex space.

    Args:
        d_in: Input complex dimension
        d_out: Output complex dimension
        use_transform: Whether to apply a transform (if d_in == d_out, can skip)
    """

    def __init__(self, d_in: int, d_out: int, use_transform: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_transform = use_transform and (d_in != d_out)

        if self.use_transform:
            # Complex linear transform
            self.weight = nn.Parameter(torch.randn(d_in, d_out, dtype=torch.cfloat) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through or transform complex input."""
        if self.use_transform:
            return torch.einsum("...i,io->...o", x, self.weight)
        return x


class CVKAN(BaseCVKAN):
    """
    Complete CV-KAN model for sequence classification.

    This architecture uses complex-valued representations where magnitudes
    encode attention-like information ("polarization"). Traditional normalization
    is NOT used because it would destroy the magnitude signal.

    Inherits from BaseCVKAN for log-magnitude centering and pooling.

    Args:
        d_input: Input dimension (real if input_type='real', complex if 'complex')
        d_complex: Complex representation dimension
        n_layers: Number of layers
        n_classes: Number of output classes
        kan_hidden: Hidden dimension for KAN MLPs
        head_approach: 'emergent', 'offset', or 'factored' (for polarizing block)
        n_heads: Number of heads (for offset/factored/attention)
        input_type: 'real' or 'complex'
        pooling: 'mean', 'max', or 'attention'
        block_type: 'polarizing' or 'attention'
        center_magnitudes: Whether to center log-magnitudes to prevent drift
    """

    def __init__(
        self,
        d_input: int = 32,
        d_complex: int = 64,
        n_layers: int = 4,
        n_classes: int = 2,
        kan_hidden: int = 32,
        head_approach: Literal["emergent", "offset", "factored"] = "emergent",
        n_heads: int = 8,
        input_type: Literal["real", "complex"] = "complex",
        pooling: Literal["mean", "max", "attention"] = "mean",
        block_type: Literal["polarizing", "attention"] = "polarizing",
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

        self.head_approach = head_approach
        self.block_type = block_type

        # Input embedding
        if input_type == "real":
            self.embedding = ComplexEmbedding(d_input, d_complex)
        else:
            self.embedding = ComplexInputEmbedding(d_input, d_complex)

        # Override layers with configured block type
        self.layers = self._build_cvkan_layers(head_approach, block_type, n_heads)

        # Classification head (operates on magnitude)
        self.classifier = build_classifier_head(
            d_complex=d_complex,
            n_classes=n_classes,
            hidden_dim=kan_hidden,
            dropout=0.0,  # Keep original behavior
        )

    def _build_cvkan_layers(
        self, head_approach: str, block_type: str, n_heads: int
    ) -> nn.ModuleList:
        """Build layers based on head approach and block type."""
        layers = nn.ModuleList()

        for _ in range(self.n_layers):
            if block_type == "polarizing":
                if head_approach == "emergent":
                    layer = EmergentHeadsPolarizing(self.d_complex, self.kan_hidden)
                elif head_approach == "offset":
                    d_per_head = self.d_complex // n_heads
                    layer = PhaseOffsetPolarizing(n_heads, d_per_head, self.kan_hidden)
                elif head_approach == "factored":
                    d_per_head = self.d_complex // n_heads
                    layer = FactoredHeadsPolarizing(
                        n_heads, self.d_complex, d_per_head, self.kan_hidden
                    )
                else:
                    raise ValueError(f"Unknown head approach: {head_approach}")
            elif block_type == "attention":
                layer = PhaseAttentionBlock(self.d_complex, n_heads=n_heads)
            else:
                raise ValueError(f"Unknown block type: {block_type}")

            layers.append(layer)

        return layers

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_tokens, d_input)
            mask: Optional binary mask (batch, n_tokens)
            return_intermediates: If True, return intermediate representations

        Returns:
            Dictionary with:
                - logits: Classification logits (batch, n_classes)
                - pooled: Pooled representation (batch, d_complex) - complex
                - intermediates: List of intermediate Z values (if requested)
        """
        intermediates = []

        # Embed to complex space
        Z = self.embedding(x)
        if return_intermediates:
            intermediates.append(Z.clone())

        # Apply polarizing layers
        for layer in self.layers:
            Z = layer(Z, mask=mask)
            if return_intermediates:
                intermediates.append(Z.clone())

        # Center log-magnitudes (from base class)
        if self.center_magnitudes:
            Z = self._center_log_magnitudes(Z)

        # Pool across tokens (from base class)
        pooled = self._pool(Z, mask)

        # Classify based on magnitude (phase-invariant)
        features = self._extract_features(pooled)
        logits = self.classifier(features)

        output = {
            "logits": logits,
            "pooled": pooled,
        }

        if return_intermediates:
            output["intermediates"] = intermediates

        return output

    def get_phase_coherence(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute phase coherence across tokens.

        High coherence = tokens have similar phases = attention-like alignment.

        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_complex)

        Returns:
            Coherence per dimension of shape (batch, d_complex)
        """
        phases = torch.angle(Z)  # (batch, n_tokens, d)

        # Circular mean resultant length
        cos_sum = torch.cos(phases).sum(dim=1)
        sin_sum = torch.sin(phases).sum(dim=1)
        n = phases.shape[1]
        R = torch.sqrt(cos_sum**2 + sin_sum**2) / n

        return R  # [0, 1] per dimension, 1 = perfect coherence


class CVKANTokenClassifier(CVKAN):
    """
    CV-KAN variant for per-token classification (e.g., signal/noise detection).

    Instead of pooling, outputs logits per token.
    """

    def __init__(self, **kwargs):
        # Remove n_classes from kwargs temporarily
        n_classes = kwargs.pop("n_classes", 2)
        super().__init__(n_classes=n_classes, **kwargs)

        # Replace classifier with per-token version
        self.token_classifier = nn.Sequential(
            nn.Linear(self.d_complex, self.kan_hidden),
            nn.GELU(),
            nn.Linear(self.kan_hidden, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> dict:
        """
        Forward pass for token classification.

        Returns:
            Dictionary with:
                - token_logits: Per-token logits (batch, n_tokens, n_classes)
                - sequence_logits: Sequence-level logits (batch, n_classes)
                - Z: Final complex representation
        """
        intermediates = []

        # Embed
        Z = self.embedding(x)
        if return_intermediates:
            intermediates.append(Z.clone())

        # Apply layers
        for layer in self.layers:
            Z = layer(Z, mask=mask)
            if return_intermediates:
                intermediates.append(Z.clone())

        # Center log-magnitudes
        if self.center_magnitudes:
            Z = self._center_log_magnitudes(Z)

        # Per-token classification (on magnitudes)
        token_features = torch.abs(Z)  # (batch, n_tokens, d)
        token_logits = self.token_classifier(token_features)  # (batch, n_tokens, n_classes)

        # Sequence-level: pool token logits
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            sum_logits = (token_logits * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            sequence_logits = sum_logits / count
        else:
            sequence_logits = token_logits.mean(dim=1)

        output = {
            "token_logits": token_logits,
            "sequence_logits": sequence_logits,
            "Z": Z,
        }

        if return_intermediates:
            output["intermediates"] = intermediates

        return output
