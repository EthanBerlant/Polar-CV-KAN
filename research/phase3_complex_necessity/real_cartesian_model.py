"""Real-valued model without polar structure.

This ablation model uses 2D real vectors but without any polar decomposition.
It's the "null hypothesis" - maybe 2x dimensions is all that matters.
"""

import torch
from torch import nn


class RealCartesianBlock(nn.Module):
    """Block using 2D real vectors with standard operations.

    No polar decomposition - just aggregate, transform, broadcast
    in Cartesian coordinates.
    """

    def __init__(self, d_model: int, kan_hidden: int = 32):
        """Args:
        d_model: Model dimension (2x d_complex equivalent)
        kan_hidden: Hidden size for transform MLP
        """
        super().__init__()
        self.d_model = d_model

        # Single transform on the full vector
        self.transform = nn.Sequential(
            nn.Linear(d_model, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor after aggregate-transform-broadcast
        """
        # 1. Aggregate (mean)
        x_agg = x.mean(dim=1, keepdim=True)  # (batch, 1, d)

        # 2. Transform
        batch, _, d = x_agg.shape
        x_agg_flat = x_agg.view(-1, d)
        x_transformed_flat = self.transform(x_agg_flat)
        x_transformed = x_transformed_flat.view(batch, 1, d)

        # 3. Broadcast (residual)
        x_out = x + x_transformed

        return x_out


class RealCartesianModel(nn.Module):
    """Complete model using real-valued Cartesian representation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,  # This should be 2 * d_complex for fair comparison
        n_layers: int,
        n_classes: int,
        kan_hidden: int = 32,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Blocks
        self.blocks = nn.ModuleList(
            [RealCartesianBlock(d_model, kan_hidden) for _ in range(n_layers)]
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, kan_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(kan_hidden, n_classes),
        )

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Embed
        x = self.embedding(input_ids)  # (batch, seq, d_model)

        # Apply blocks
        for block in self.blocks:
            x = block(x)

        # Pool
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            x_masked = x * mask_expanded
            x_pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x_pooled = x.mean(dim=1)

        # Classify
        logits = self.classifier(x_pooled)

        return logits

    @classmethod
    def from_config(cls, config: dict):
        """Create from config dict for compatibility."""
        # d_model should be 2 * d_complex for parameter fairness
        d_complex = config.get("d_complex", 64)
        return cls(
            vocab_size=config.get("vocab_size", 30522),
            d_model=d_complex * 2,  # 2x for fair comparison
            n_layers=config.get("n_layers", 2),
            n_classes=config.get("n_classes", 2),
            kan_hidden=config.get("kan_hidden", 32),
            max_seq_len=config.get("max_seq_len", 512),
            dropout=config.get("dropout", 0.1),
        )


class RealCartesianSplitBlock(nn.Module):
    """Variant: separate transforms for "real" and "imaginary" parts.

    This is still not polar, but acknowledges the 2-channel structure.
    """

    def __init__(self, d_half: int, kan_hidden: int = 32):
        super().__init__()
        self.d_half = d_half

        # Separate transforms for each "channel"
        self.transform_a = nn.Sequential(
            nn.Linear(d_half, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, d_half),
        )
        self.transform_b = nn.Sequential(
            nn.Linear(d_half, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, d_half),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split
        a = x[..., : self.d_half]
        b = x[..., self.d_half :]

        # Aggregate
        a_agg = a.mean(dim=1, keepdim=True)
        b_agg = b.mean(dim=1, keepdim=True)

        # Transform separately
        batch = a_agg.size(0)
        a_trans = self.transform_a(a_agg.view(-1, self.d_half)).view(batch, 1, -1)
        b_trans = self.transform_b(b_agg.view(-1, self.d_half)).view(batch, 1, -1)

        # Broadcast
        a_out = a + a_trans
        b_out = b + b_trans

        return torch.cat([a_out, b_out], dim=-1)


class RealCartesianSplitModel(nn.Module):
    """Model with split transforms but no polar structure."""

    def __init__(
        self,
        vocab_size: int,
        d_complex: int,  # Each "channel" has this dimension
        n_layers: int,
        n_classes: int,
        kan_hidden: int = 32,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_complex = d_complex

        self.embedding = nn.Embedding(vocab_size, d_complex * 2)

        self.blocks = nn.ModuleList(
            [RealCartesianSplitBlock(d_complex, kan_hidden) for _ in range(n_layers)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_complex * 2, kan_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(kan_hidden, n_classes),
        )

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            x_masked = x * mask_expanded
            x_pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x_pooled = x.mean(dim=1)

        return self.classifier(x_pooled)

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            vocab_size=config.get("vocab_size", 30522),
            d_complex=config.get("d_complex", 64),
            n_layers=config.get("n_layers", 2),
            n_classes=config.get("n_classes", 2),
            kan_hidden=config.get("kan_hidden", 32),
            max_seq_len=config.get("max_seq_len", 512),
            dropout=config.get("dropout", 0.1),
        )
