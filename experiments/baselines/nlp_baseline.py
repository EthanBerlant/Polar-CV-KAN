import torch
import torch.nn as nn


class BiLSTMBaseline(nn.Module):
    """
    Bidirectional LSTM Baseline.
    Standard efficient sequence model.
    """

    def __init__(
        self, vocab_size, embed_dim=128, hidden_dim=64, n_layers=2, n_classes=2, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Seq]
        embedded = self.dropout(self.embedding(x))
        # lstm_out: [Batch, Seq, Hidden*2]
        # hidden: [Layers*2, Batch, Hidden]
        _, (hidden, _) = self.lstm(embedded)

        # Concat the last hidden state of forward and backward LSTM
        # Hidden structure: [Layer 1 Fwd, Layer 1 Bwd, Layer 2 Fwd, Layer 2 Bwd...]
        # We take the last layer
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        cat = torch.cat((hidden_fwd, hidden_bwd), dim=1)

        return self.fc(self.dropout(cat))


class TinyTransformerBaseline(nn.Module):
    """
    Minimally sized Transformer Encoder for fair comparison.
    Not BERT-sized (110M), but "PC-sized" (0.1M - 5M params).
    """

    def __init__(
        self, vocab_size, embed_dim=64, n_heads=4, n_layers=3, n_classes=2, max_len=512, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc = nn.Linear(embed_dim, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x):
        B, L = x.shape
        x = self.embedding(x)

        # Add position encoding
        if L <= self.max_len:
            x = x + self.pos_encoder[:, :L, :]
        else:
            x = x + self.pos_encoder[:, :L, :]

        x = self.dropout(x)
        x = self.transformer(x)

        # Mean pooling
        x = x.mean(dim=1)
        return self.fc(x)
