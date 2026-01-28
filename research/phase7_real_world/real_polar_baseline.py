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

    def forward(self, r: torch.Tensor, theta: torch.Tensor) -> tuple:
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


class RealPolarImageClassifier(nn.Module):
    """Real-valued Polar Model for Images."""

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        d_polar: int = 32,
        n_layers: int = 3,
        n_classes: int = 10,
        kan_hidden: int = 32,
    ):
        super().__init__()
        self.d_polar = d_polar
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Embedding: Projects patch to r, theta (2 * d_polar)
        self.patch_embed = nn.Linear(patch_dim, d_polar * 2)

        # Blocks
        self.blocks = nn.ModuleList([RealPolarBlock(d_polar, kan_hidden) for _ in range(n_layers)])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_polar, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, n_classes),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Patchify
        # Unfold to patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = (
            x.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(B, -1, C * self.patch_size * self.patch_size)
        )

        # Embed
        h = self.patch_embed(x)  # (B, N, 2*D)

        # Split
        r = torch.abs(h[..., : self.d_polar]) + 0.1
        theta = h[..., self.d_polar :]

        # Layers
        for block in self.blocks:
            r, theta = block(r, theta)

        # Global Pool (Magnitude Weighted Average of r?)
        # Or just mean r. Original RealPolar used mean r.
        # To be consistent with the requested "Magnitude Weighted" test,
        # let's stick to the baseline behavior (Mean) unless specified.
        # But wait, the user asked "What aggregation are you using for polar-cv-kan? It should be magnitude weighted."
        # This implies the RealPolar baseline should probably just use whatever it used before (Mean) to serve as a baseline.

        r_pooled = r.mean(dim=1)
        x_cart = r * torch.cos(theta)
        y_cart = r * torch.sin(theta)
        xp = x_cart.mean(dim=1)
        yp = y_cart.mean(dim=1)
        pooled_complex = torch.complex(xp, yp)

        return {"logits": self.classifier(r_pooled), "features": r_pooled, "pooled": pooled_complex}
