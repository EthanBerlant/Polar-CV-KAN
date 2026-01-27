from typing import Any

import torch
from torch import nn

from ..configs.model import CVKANConfig
from .backbone import CVKANBackbone
from .base import ComplexEmbedding, build_classifier_head
from .protocols import Backbone, Embedding, Head


class CVKAN(nn.Module):
    """Unified CV-KAN model using composition over inheritance.

    Structure:
        Input -> Embedding -> Backbone -> Head -> Output
    """

    def __init__(
        self,
        embedding: Embedding,
        backbone: Backbone,
        head: Head,
    ):
        super().__init__()
        self.embedding = embedding
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> dict[str, Any]:
        """Standard forward pass.
        Returns dict for flexibility (logits, auxiliary outputs from head).
        """
        # 1. Embed (Real/Index -> Complex Sequence)
        z = self.embedding(x)

        # 2. Backbone (Refine Complex Sequence)
        z = self.backbone(z, mask=mask)

        # 3. Head (Pool & Classify)
        return self.head(z, mask=mask)

    @classmethod
    def from_config(cls, config: CVKANConfig, input_dim: int, n_classes: int):
        """Factory method to create a standard Sequence Classification CVKAN."""
        # 1. Standard Embedding
        if config.input_type == "real":
            embedding = ComplexEmbedding(input_dim, config.d_complex)
        # Identity/Passthrough embedding for complex inputs (synthetic)
        elif input_dim != config.d_complex:
            embedding = ComplexProjection(input_dim, config.d_complex)
        else:

            class IdentityComplexEmbedding(nn.Module):
                def forward(self, x):
                    return x

            embedding = IdentityComplexEmbedding()

        # 2. Standard Backbone
        backbone = CVKANBackbone(config)

        # 3. Standard Classification Head
        # We need to wrap detailed pooling/classification logic into a Head component
        # Currently BaseCVKAN has _pool and _extract_features.
        # Let's create a reusable StandardHead
        head = StandardClassificationHead(
            d_complex=config.d_complex,
            n_classes=n_classes,
            kan_hidden=config.kan_hidden,
            pooling=config.pooling,
            dropout=config.dropout,
        )

        return cls(embedding, backbone, head)


class StandardClassificationHead(nn.Module):
    """Standard head: Pooling -> Magnitude -> Classifier."""

    def __init__(self, d_complex, n_classes, kan_hidden, pooling, dropout):
        super().__init__()
        self.d_complex = d_complex
        self.pooling_type = pooling

        # Pooling Query (for attention pooling)
        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_complex, dtype=torch.cfloat) * 0.02)

        self.classifier = build_classifier_head(
            d_complex=d_complex, n_classes=n_classes, hidden_dim=kan_hidden, dropout=dropout
        )

    def _pool(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.pooling_type == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                sum_z = (z * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1.0)
                return sum_z / count
            return z.mean(dim=1)

        if self.pooling_type == "max":
            mag = torch.abs(z)
            if mask is not None:
                mag = mag.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

            # Select token with max magnitude per dimension
            # (batch, d_complex) indices
            _, indices = mag.max(dim=1)

            # Gather complex values
            # This is tricky in PyTorch complex.
            # Simplified: just return max magnitude for features (phase lost in max pool usually)
            # But we need complex output for consistency?
            # Let's stick to returning the complex value at max mag index
            batch_indices = torch.arange(z.size(0), device=z.device).unsqueeze(-1)
            dim_indices = torch.arange(z.size(2), device=z.device).unsqueeze(0)
            return z[batch_indices, indices, dim_indices]

        if self.pooling_type == "attention":
            # Simple dot-product attention with learnable query
            # Query: (1, 1, d)
            # Keys: z (b, n, d)
            # Re{Query * conj(Keys)}

            scores = (self.pool_query * z.conj()).real  # (b, n, d)
            scores = scores.mean(dim=-1)  # (b, n) average agreement across dims

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (b, n, 1)
            return (z * attn).sum(dim=1)

        if self.pooling_type == "covariance":
            # Second-order pooling (Correlation between feature dimensions)
            # z: (B, N, D)
            # Output: (B, D) - Diagonal of Gram matrix? Or Flattened?
            # Standard Covariance pooling usually returns (B, D*(D+1)/2) but we need to stay in d_complex dimension?
            # Let's do a simplified version: Global Average Pooling of the Gram Matrix diagonal (Correlation magnitude)
            # Or: We project the Gram Matrix back to D dimensions?

            # Alternative: "Attentive Covariance" or just Gram Matrix diagonal.
            # Let's try: Centered Covariance -> Diagonal (Variance) as a starting point for "texture"
            # BUT variance is just Mean(Mag^2).
            # True covariance captures inter-channel deps.

            # Implementation:
            # 1. Center features: z_centered = z - mean(z)
            # 2. Gram: G = z_c.H @ z_c  (B, D, D)
            # 3. We typically flatten this for classification, but our head expects (B, D).
            # Constraint: Output must be (B, D) complex.
            # Strategy: Return the principal components or just the diagonal?
            # Let's go with: Mean + Variance encoded in Real/Imag?

            # Let's check literature/user intent: "Excellent for texture".
            # Usually requires a classifier that takes (D, D).
            # Given our architecture (Backbone -> Head(Linear(D))), we MUST output (B, D).
            # So Covariance *Pooling* in this specific pipeline might be "Global Variance Pooling".
            # z_mean = z.mean(dim=1)
            # z_var = ((z - z_mean.unsqueeze(1)) * (z - z_mean.unsqueeze(1)).conj()).mean(dim=1)
            # return z_mean + z_var  (Superposition of 1st and 2nd moments)

            if mask is not None:
                # Masking logic needed
                mask_expanded = mask.unsqueeze(-1)
                count = mask_expanded.sum(dim=1).clamp(min=1.0)
                mean = (z * mask_expanded).sum(dim=1) / count
                centered = (z - mean.unsqueeze(1)) * mask_expanded
                cov = (centered * centered.conj()).sum(dim=1) / count
            else:
                mean = z.mean(dim=1)
                cov = ((z - mean.unsqueeze(1)) * (z - mean.unsqueeze(1)).conj()).mean(dim=1)

            return mean + cov  # Encode both moments

        if self.pooling_type == "spectral":
            # FFT-based pooling (Low-pass filter in sequence dim)
            # z: (B, N, D)
            # FFT over N -> (B, N, D)
            # Take DC component (index 0) = Mean
            # Take low freq components?
            # If we simply take index 0, it's Mean Pooling.
            # Let's take the *magnitude* of the first K frequencies and average them?
            # Or: Return the component with highest total energy (Dominant Frequency Pooling).

            # Implementation: Top-1 Frequency (excluding DC) + DC
            z_fft = torch.fft.fft(z, dim=1)  # (B, N, D)

            # DC component (Mean)
            dc = z_fft[:, 0, :] / z.size(1)

            # Find dominant non-DC freq
            if z.size(1) > 1:
                amplitudes = torch.abs(z_fft[:, 1:, :]).sum(dim=-1)  # (B, N-1)
                best_idx = amplitudes.argmax(dim=1) + 1  # (B,)

                # Gather (B, 1, D)
                batch_idx = torch.arange(z.size(0), device=z.device)
                dominant = z_fft[batch_idx, best_idx, :] / z.size(1)
                return dc + dominant
            return dc

        raise ValueError(f"Unknown pooling: {self.pooling_type}")

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
        # Pool
        pooled = self._pool(z, mask)

        # Feature extraction (Magnitude)
        features = torch.abs(pooled)

        # Classify
        logits = self.classifier(features)

        return {"logits": logits, "pooled": pooled, "features": features}


class TokenClassificationHead(nn.Module):
    """Head for per-token classification.
    No pooling, just projection of each token's magnitude.
    """

    def __init__(self, d_complex, n_classes, kan_hidden, dropout):
        super().__init__()
        self.classifier = build_classifier_head(
            d_complex=d_complex, n_classes=n_classes, hidden_dim=kan_hidden, dropout=dropout
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
        # z: (B, L, D) checks
        # Features: (B, L, D) magnitude
        features = torch.abs(z)

        # Logits: (B, L, C)
        logits = self.classifier(features)

        return {"token_logits": logits, "features": features}


class ComplexProjection(nn.Module):
    """Project complex inputs to different dimension [Complex -> Complex].
    Uses complex-valued linear layer.
    """

    def __init__(self, d_input, d_output):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output, bias=False, dtype=torch.cfloat)

    def forward(self, x):
        return self.linear(x)


class CVKANTokenClassifier(CVKAN):
    """CV-KAN model for token classification tasks (e.g., NER, signal detection).
    Legacy wrapper around composed CVKAN with TokenClassificationHead.
    """

    def __init__(
        self,
        d_input: int = 32,
        d_complex: int = 64,
        n_layers: int = 4,
        n_classes: int = 2,
        kan_hidden: int = 32,
        head_approach: str = "emergent",
        n_heads: int = 8,
        input_type: str = "complex",
        block_type: str = "polarizing",
        center_magnitudes: bool = True,
        dropout: float = 0.0,
    ):
        # 1. Config
        config = CVKANConfig(
            d_complex=d_complex,
            n_layers=n_layers,
            kan_hidden=kan_hidden,
            head_approach=head_approach,
            block_type=block_type,
            n_heads=n_heads,
            center_magnitudes=center_magnitudes,
            dropout=dropout,
            input_type=input_type,
        )

        # 2. Embedding
        if input_type == "real":
            embedding = ComplexEmbedding(d_input, d_complex)
        elif d_input != d_complex:
            embedding = ComplexProjection(d_input, d_complex)
        else:

            class IdentityComplexEmbedding(nn.Module):
                def forward(self, x):
                    return x

            embedding = IdentityComplexEmbedding()

        # 3. Backbone
        backbone = CVKANBackbone(config)

        # 4. Head
        head = TokenClassificationHead(
            d_complex=d_complex, n_classes=n_classes, kan_hidden=kan_hidden, dropout=dropout
        )

        super().__init__(embedding, backbone, head)
