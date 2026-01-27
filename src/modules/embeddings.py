import torch
from torch import nn

from src.modules.positional_encoding import (
    Complex2DPositionalEncoding,
    Learnable2DComplexPositionalEncoding,
)
from src.registry import EMBEDDING_REGISTRY


@EMBEDDING_REGISTRY.register("linear")
class ComplexLinearEmbedding(nn.Module):
    """Project real-valued input to complex space.

    Used for tabular data or pre-featurized inputs.
    """

    def __init__(self, input_dim: int, d_complex: int) -> None:
        """Initialize."""
        super().__init__()
        self.d_complex = d_complex
        self.proj_real = nn.Linear(input_dim, d_complex)
        self.proj_imag = nn.Linear(input_dim, d_complex)

        # Init
        nn.init.xavier_uniform_(self.proj_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_imag.weight, gain=0.5)
        nn.init.zeros_(self.proj_real.bias)
        nn.init.zeros_(self.proj_imag.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (..., input_dim).

        Returns:
            Complex tensor (..., d_complex).
        """
        real = self.proj_real(x)
        imag = self.proj_imag(x)
        return torch.complex(real, imag)


class LinearPatchEmbedding(nn.Module):
    """Internal helper: Linear patch projection.

    Splits image into flattened patches and projects them.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        d_complex: int,
    ) -> None:
        """Initialize."""
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.patch_size = patch_size
        self.n_patches_h = img_size[0] // patch_size[0]
        self.n_patches_w = img_size[1] // patch_size[1]
        self.n_patches = self.n_patches_h * self.n_patches_w

        patch_dim = in_channels * patch_size[0] * patch_size[1]
        self.proj_real = nn.Linear(patch_dim, d_complex)
        self.proj_imag = nn.Linear(patch_dim, d_complex)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Forward pass."""
        # x: (B, C, H, W)
        batch = x.shape[0]
        # Unfold to patches
        x = x.unfold(2, self.patch_size[0], self.patch_size[0])
        x = x.unfold(3, self.patch_size[1], self.patch_size[1])
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch, self.n_patches, -1)

        real = self.proj_real(x)
        imag = self.proj_imag(x)
        return torch.complex(real, imag), (self.n_patches_h, self.n_patches_w)


@EMBEDDING_REGISTRY.register("image_patch")
class ImageEmbedding(nn.Module):
    """Image patch embedding with positional encoding.

    Available args in config:
     - img_size: int
     - patch_size: int
     - in_channels: int
     - pos_encoding: str
    """

    def __init__(
        self,
        d_complex: int,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        pos_encoding: str = "sinusoidal",
        embedding_type: str = "linear",  # linear vs conv
    ) -> None:
        """Initialize."""
        super().__init__()
        # Simplified: defaulting to linear patch embedding for now
        # Could add ConvPatchEmbedding support here too

        self.patch_embed = LinearPatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, d_complex=d_complex
        )

        n_h, n_w = self.patch_embed.n_patches_h, self.patch_embed.n_patches_w

        if pos_encoding == "sinusoidal":
            self.pos_encoding = Complex2DPositionalEncoding(d_complex, n_h, n_w)
        elif pos_encoding == "learnable":
            self.pos_encoding = Learnable2DComplexPositionalEncoding(d_complex, n_h, n_w)
        else:
            self.pos_encoding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds images into complex vectors."""
        z, spatial_shape = self.patch_embed(x)
        if self.pos_encoding:
            z = self.pos_encoding(z, spatial_shape=spatial_shape)
        return z


@EMBEDDING_REGISTRY.register("token")
class TokenEmbedding(nn.Module):
    """Learned embedding for token indices (NLP)."""

    def __init__(
        self, d_complex: int, vocab_size: int, max_seq_len: int = 512, dropout: float = 0.1
    ) -> None:
        """Initialize."""
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_complex * 2)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_complex * 2) * 0.02)
        self.emb_dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x: (B, L) long
        B, L = x.shape
        x_emb = self.token_embedding(x)

        # Pos encoding
        if self.max_seq_len >= L:
            x_emb = x_emb + self.pos_embedding[:, :L, :]
        else:
            x_emb = x_emb + self.pos_embedding[:, : min(L, self.max_seq_len), :]

        x_emb = self.emb_dropout(x_emb)
        real, imag = torch.chunk(x_emb, 2, dim=-1)
        return torch.complex(real, imag)
