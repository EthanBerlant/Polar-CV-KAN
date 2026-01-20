import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool2d(nn.Module):
    """
    Learned attention pooling for 2D feature maps.
    Replaces Global Average Pooling with a weighted sum of spatial tokens.

    Args:
        in_channels (int): Number of input channels/features
        hidden_dim (int): Hidden dimension for the attention MLP (default: in_channels // 2)
    """

    def __init__(self, in_channels: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_channels // 2

        self.attn_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, N, C)

        Returns:
            Pooled tensor of shape (B, C)
        """
        # Handle (B, C, H, W) input
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).permute(0, 2, 1)  # -> (B, N, C) where N=H*W

        # x is now (B, N, C)

        # Calculate attention scores
        # attn_scores: (B, N, 1)
        attn_scores = self.attn_mlp(x)

        # Softmax over spatial dimension (N)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum: (B, N, 1) * (B, N, C) -> (B, N, C) -> sum -> (B, C)
        pooled = torch.sum(attn_weights * x, dim=1)

        return pooled
