"""Real-valued model with explicit polar structure.

This is an ablation model that uses 2D real vectors (r, θ) instead of
complex numbers, but maintains the polar decomposition and transformation
structure of the original architecture.
"""

import numpy as np
import torch
from torch import nn


class RealPolarizingBlock(nn.Module):
    """Polarizing block using real-valued polar coordinates.

    Instead of complex numbers, we represent each "complex" dimension
    as an explicit (magnitude, phase) pair. Operations mimic the original
    but use real arithmetic.
    """

    def __init__(self, d_polar: int, kan_hidden: int = 32):
        """Args:
        d_polar: Number of polar dimensions (equivalent to d_complex)
        kan_hidden: Hidden size for transform MLPs
        """
        super().__init__()
        self.d_polar = d_polar

        # Magnitude transform (operates on log-magnitude)
        self.mag_transform = nn.Sequential(
            nn.Linear(1, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 1),
        )

        # Phase transform (operates on sin/cos representation)
        self.phase_transform = nn.Sequential(
            nn.Linear(2, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 2),
        )

    def forward(self, r: torch.Tensor, theta: torch.Tensor) -> tuple:
        """Args:
            r: Magnitudes (batch, seq_len, d_polar)
            theta: Phases (batch, seq_len, d_polar)

        Returns:
            (r_out, theta_out) after polarizing operation
        """
        batch, seq_len, d = r.shape

        # 1. Aggregate: compute mean in Cartesian, convert back to polar
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, d)
        y_mean = y.mean(dim=1, keepdim=True)

        r_agg = torch.sqrt(x_mean**2 + y_mean**2 + 1e-8)
        theta_agg = torch.atan2(y_mean, x_mean)

        # 2. Transform magnitude (in log space)
        log_r = torch.log(r_agg + 1e-8)
        log_r_flat = log_r.view(-1, 1)
        log_r_transformed = self.mag_transform(log_r_flat)
        log_r_transformed = log_r_transformed.view(batch, 1, d)
        r_transformed = torch.exp(log_r_transformed)

        # 3. Transform phase (using sin/cos representation)
        sin_theta = torch.sin(theta_agg)
        cos_theta = torch.cos(theta_agg)
        phase_input = torch.stack([sin_theta, cos_theta], dim=-1)  # (batch, 1, d, 2)
        phase_input_flat = phase_input.view(-1, 2)
        phase_output_flat = self.phase_transform(phase_input_flat)
        phase_output = phase_output_flat.view(batch, 1, d, 2)

        # Renormalize to unit circle
        phase_norm = torch.sqrt(phase_output[..., 0] ** 2 + phase_output[..., 1] ** 2 + 1e-8)
        sin_transformed = phase_output[..., 0] / phase_norm
        cos_transformed = phase_output[..., 1] / phase_norm
        theta_transformed = torch.atan2(sin_transformed, cos_transformed)

        # 4. Convert transformed aggregate back to Cartesian
        x_transformed = r_transformed * torch.cos(theta_transformed)
        y_transformed = r_transformed * torch.sin(theta_transformed)

        # 5. Broadcast (residual in Cartesian space)
        x_out = x + x_transformed
        y_out = y + y_transformed

        # 6. Convert back to polar
        r_out = torch.sqrt(x_out**2 + y_out**2 + 1e-8)
        theta_out = torch.atan2(y_out, x_out)

        return r_out, theta_out


class RealPolarModel(nn.Module):
    """Complete model using real-valued polar representation."""

    def __init__(
        self,
        vocab_size: int,
        d_polar: int,
        n_layers: int,
        n_classes: int,
        kan_hidden: int = 32,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_polar = d_polar

        # Embedding (outputs 2 * d_polar for magnitude and phase init)
        self.embedding = nn.Embedding(vocab_size, d_polar * 2)

        # Polarizing blocks
        self.blocks = nn.ModuleList(
            [RealPolarizingBlock(d_polar, kan_hidden) for _ in range(n_layers)]
        )

        # Classifier (uses magnitude only, like original)
        self.classifier = nn.Sequential(
            nn.Linear(d_polar, kan_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(kan_hidden, n_classes),
        )

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Embed
        h = self.embedding(input_ids)  # (batch, seq, 2*d)

        # Split into initial magnitude and phase
        r = torch.abs(h[..., : self.d_polar]) + 0.1  # Ensure positive
        theta = h[..., self.d_polar :] * np.pi  # Scale to [-pi, pi]

        # Apply blocks
        for block in self.blocks:
            r, theta = block(r, theta)

        # Pool (mean magnitude)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            r_masked = r * mask_expanded
            r_pooled = r_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            r_pooled = r.mean(dim=1)

        # Classify
        logits = self.classifier(r_pooled)

        return logits

    @classmethod
    def from_config(cls, config: dict):
        """Create from config dict for compatibility."""
        return cls(
            vocab_size=config.get("vocab_size", 30522),
            d_polar=config.get("d_complex", 64),
            n_layers=config.get("n_layers", 2),
            n_classes=config.get("n_classes", 2),
            kan_hidden=config.get("kan_hidden", 32),
            max_seq_len=config.get("max_seq_len", 512),
            dropout=config.get("dropout", 0.1),
        )
