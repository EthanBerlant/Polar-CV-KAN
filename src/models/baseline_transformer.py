"""
Baseline Transformer for fair comparison with CV-KAN.

A simple Transformer encoder with:
- Configurable layers/heads/dimensions
- Same embedding approach (random init, no pretrained)
- Mean pooling + classifier head
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding. x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaselineTransformer(nn.Module):
    """
    Simple Transformer encoder for sequence classification.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_layers: Number of encoder layers
        n_heads: Number of attention heads
        d_ff: Feedforward dimension (defaults to 4 * d_model)
        n_classes: Number of output classes
        dropout: Dropout probability
        max_len: Maximum sequence length
        pooling: 'mean', 'first', or 'max'
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        n_classes: int = 2,
        dropout: float = 0.1,
        max_len: int = 512,
        pooling: str = 'mean',
    ):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights similar to BERT."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices (batch, seq_len)
            mask: Attention mask, 1 for valid, 0 for padding (batch, seq_len)
            return_intermediates: If True, return hidden states
        
        Returns:
            dict with 'logits' and optionally 'intermediates'
        """
        # Embed
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        intermediates = [x] if return_intermediates else None
        
        # Create attention mask (Transformer uses inverted mask: True = ignore)
        if mask is not None:
            attn_mask = ~mask.bool()  # Invert: True means "ignore this position"
        else:
            attn_mask = None
        
        # Encode
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        
        if return_intermediates:
            intermediates.append(x)
        
        # Pool
        if self.pooling == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                sum_x = (x * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1.0)
                pooled = sum_x / count
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == 'first':
            pooled = x[:, 0]
        elif self.pooling == 'max':
            if mask is not None:
                x_masked = x.masked_fill(~mask.bool().unsqueeze(-1), -1e9)
            else:
                x_masked = x
            pooled = x_masked.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify
        logits = self.classifier(pooled)
        
        output = {'logits': logits, 'pooled': pooled}
        if return_intermediates:
            output['intermediates'] = intermediates
        
        return output


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_matched_transformer(vocab_size: int, target_params: int, n_classes: int = 2):
    """
    Create a Transformer with approximately the same parameter count.
    
    Uses binary search to find d_model that gives similar param count.
    """
    def params_for_dim(d_model):
        model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=2,
            n_heads=max(1, d_model // 16),
            n_classes=n_classes,
        )
        return count_parameters(model)
    
    # Binary search for matching dimension
    low, high = 16, 256
    best_dim = 64
    best_diff = float('inf')
    
    while low <= high:
        mid = (low + high) // 2
        params = params_for_dim(mid)
        diff = abs(params - target_params)
        
        if diff < best_diff:
            best_diff = diff
            best_dim = mid
        
        if params < target_params:
            low = mid + 1
        else:
            high = mid - 1
    
    return BaselineTransformer(
        vocab_size=vocab_size,
        d_model=best_dim,
        n_layers=2,
        n_heads=max(1, best_dim // 16),
        n_classes=n_classes,
    )
