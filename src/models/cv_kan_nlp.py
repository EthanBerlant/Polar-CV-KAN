from typing import Any, Literal

import torch
import torch.nn as nn

from .base import BaseCVKAN, build_classifier_head


class CVKANNLP(BaseCVKAN):
    """
    Complex-Valued KAN for Natural Language Processing/Text Classification.

    Treats text as a 1D sequence of complex embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_complex: int = 128,
        n_layers: int = 4,
        n_classes: int = 2,
        max_seq_len: int = 512,  # Renamed from max_len to match config/other models
        dropout: float = 0.1,
        pooling: Literal["mean", "max", "attention"] = "attention",
        kan_hidden: int = 64,
        block_type: str = "polarizing",
        **kwargs,
    ):
        # Initialize BaseCVKAN
        super().__init__(
            d_complex=d_complex,
            n_layers=n_layers,
            kan_hidden=kan_hidden,
            pooling=pooling,
            center_magnitudes=True,
            use_multi_head=False,
        )

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes
        self.block_type = block_type

        # 1. Complex Embedding: [Batch, Seq, d_complex]
        # We need a custom embedding layer since BaseCVKAN's ComplexEmbedding expects real inputs (float)
        # But here we have token indices (long).
        self.token_embedding = nn.Embedding(vocab_size, d_complex * 2)

        # 2. Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_complex * 2) * 0.02)

        self.emb_dropout = nn.Dropout(dropout)

        # 3. Classifier Head
        self.classifier = build_classifier_head(
            d_complex=d_complex, n_classes=n_classes, hidden_dim=kan_hidden, dropout=dropout
        )

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to complex embeddings with positional encoding.
        """
        B, L = x.shape

        # Embedding
        x_emb = self.token_embedding(x)

        # Positional Encoding
        if L <= self.max_seq_len:
            x_emb = x_emb + self.pos_embedding[:, :L, :]
        else:
            # Handle sequences longer than pos encoding (truncate pos encoding or input? usually input is truncated by loader)
            # Safe fallback:
            x_emb = x_emb + self.pos_embedding[:, : min(L, self.max_seq_len), :]

        x_emb = self.emb_dropout(x_emb)

        # Convert to complex: [B, L, D]
        real, imag = torch.chunk(x_emb, 2, dim=-1)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> dict[str, Any]:
        """
        Args:
            x: [Batch, Seq] long tensor
            mask: [Batch, Seq] bool/float mask
        """
        # 1. Embed
        z = self._embed(x)

        # 2. Apply Layers (BaseCVKAN helper)
        z = self._apply_layers(z, mask=mask)

        # 3. Pool (BaseCVKAN helper)
        pooled_complex = self._pool(z, mask=mask)

        # 4. Features (Magnitude)
        features = self._extract_features(pooled_complex)

        # 5. Classify
        logits = self.classifier(features)

        return {"logits": logits, "pooled": pooled_complex, "features": features}
