import numpy as np
import torch
from torch import nn


class RealPolarBlock(nn.Module):
    """Polarizing block using real-valued polar coordinates."""

    def __init__(self, d_polar: int, kan_hidden: int = 32):
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

    def forward(self, r: torch.Tensor, theta: torch.Tensor, mask=None) -> tuple:
        batch, seq_len, d = r.shape

        # 1. Aggregate: compute mean in Cartesian, convert back to polar
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        if mask is not None:
            # mask is (B, L)
            mask_expanded = mask.unsqueeze(-1).float()
            x_mean = (x * mask_expanded).sum(dim=1, keepdim=True) / mask_expanded.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
            y_mean = (y * mask_expanded).sum(dim=1, keepdim=True) / mask_expanded.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        else:
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


class RealPolarNLP(nn.Module):
    """Real-valued Polar Model for NLP with Attention Pooling."""

    def __init__(
        self,
        vocab_size: int,
        d_polar: int = 32,
        n_layers: int = 3,
        n_classes: int = 2,
        kan_hidden: int = 32,
        pooling: str = "attention",
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.d_polar = d_polar
        self.pooling = pooling

        # Embedding: Projects to r, theta (2 * d_polar)
        self.embedding = nn.Embedding(vocab_size, d_polar * 2)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_polar * 2) * 0.02)

        # Blocks
        self.blocks = nn.ModuleList([RealPolarBlock(d_polar, kan_hidden) for _ in range(n_layers)])

        # Pooling Query
        if pooling == "attention":
            # Query for (r, theta) -> Cartesian query (2 * d_polar)?
            # Or separate queries?
            # Let's use a Cartesian query vector
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_polar * 2) * 0.02)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_polar, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, n_classes),
        )

    def forward(self, x, mask=None, **kwargs):
        # x: (B, L)
        B, L = x.shape

        # Embed
        h = self.embedding(x)  # (B, L, 2*D)

        # Pos encoding
        if self.pos_embedding.shape[1] >= L:
            h = h + self.pos_embedding[:, :L, :]

        # Split (interpret first half as r, second as theta for consistency)
        # Note: Phase 3 model did: r = abs(h[:d]), theta = h[d:]*pi
        # Let's stick to that to be "RealPolar"
        embedding_dim = self.d_polar
        r = torch.abs(h[..., :embedding_dim]) + 0.1
        theta = h[..., embedding_dim:] * np.pi

        # Layers
        for block in self.blocks:
            r, theta = block(r, theta, mask=mask)

        # Feature Extraction & Pooling

        # Convert to Cartesian for pooling (easiest way to pool independent dimensions correctly)
        x_cart = r * torch.cos(theta)
        y_cart = r * torch.sin(theta)
        z_cart = torch.cat([x_cart, y_cart], dim=-1)  # (B, L, 2*D)

        if self.pooling == "attention":
            # Attention Mechanism
            # Query: (1, 1, 2*D)
            # Keys: z_cart (B, L, 2*D)
            query = self.pool_query.expand(B, -1, -1)

            # Simple dot product attention
            attn_scores = torch.bmm(z_cart, query.transpose(1, 2))  # (B, L, 1)

            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

            attn_weights = torch.softmax(attn_scores, dim=1)

            # Weighted average
            pooled_cart = (z_cart * attn_weights).sum(dim=1)  # (B, 2*D)

            # Extract features: Just Magnitude part?
            # CV-KAN extracts magnitude of the pooled complex vector.
            # Here pooled_cart is [x_pool, y_pool]
            x_p = pooled_cart[:, : self.d_polar]
            y_p = pooled_cart[:, self.d_polar :]
            r_pooled = torch.sqrt(x_p**2 + y_p**2 + 1e-8)

        elif self.pooling == "weighted":
            # Magnitude Weighted Average
            # Weights w_i = r_i / sum(r_j)
            # Apply to Cartesian coords
            # z_cart: (B, L, 2*D)
            # r: (B, L, D) => we need magnitude of the 2D "complex" vector at each dimension?
            # No, r is already that magnitude.

            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                # r_masked: (B, L, D)
                r_masked = r * mask_expanded
                # We want global weight per token? Or per dimension?
                # Usually weighted average is per-dimension if dimensions are independent.
                # w_i,d = r_i,d / sum_j(r_j,d)

                sum_r = r_masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
                weights = r_masked / sum_r  # (B, L, D)

                # Apply weights to Cartesian
                # x_pool = sum(x * w)
                # y_pool = sum(y * w)

                # Expand weights for 2*D
                # x is first D, y is second D
                weights_expanded = torch.cat([weights, weights], dim=-1)

                pooled_cart = (z_cart * weights_expanded).sum(dim=1)

            else:
                sum_r = r.sum(dim=1, keepdim=True).clamp(min=1e-8)
                weights = r / sum_r
                weights_expanded = torch.cat([weights, weights], dim=-1)
                pooled_cart = (z_cart * weights_expanded).sum(dim=1)

            x_p = pooled_cart[:, : self.d_polar]
            y_p = pooled_cart[:, self.d_polar :]
            r_pooled = torch.sqrt(x_p**2 + y_p**2 + 1e-8)

        elif self.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                z_cart = z_cart * mask_expanded
                pooled_cart = z_cart.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled_cart = z_cart.mean(dim=1)

            x_p = pooled_cart[:, : self.d_polar]
            y_p = pooled_cart[:, self.d_polar :]
            r_pooled = torch.sqrt(x_p**2 + y_p**2 + 1e-8)

        return {
            "logits": self.classifier(r_pooled),
            "features": r_pooled,
            "pooled": torch.complex(x_p, y_p),
        }
