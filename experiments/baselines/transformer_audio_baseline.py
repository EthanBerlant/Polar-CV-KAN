"""
Transformer Baseline for Audio Classification.

A standard Transformer encoder on mel-spectrograms.
Sized to match CV-KAN parameter count (~200-500k params).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.audio_data import (
    TORCHAUDIO_AVAILABLE,
    create_audio_dataloader,
    create_esc50_dataloader,
    create_urbansound8k_dataloader,
)

if TORCHAUDIO_AVAILABLE:
    import torchaudio


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AudioTransformer(nn.Module):
    """
    Transformer encoder for audio classification on spectrograms.
    """

    def __init__(
        self,
        n_mels=64,
        num_classes=35,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # Spectrogram frontend
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=n_mels
        )

        # Project mel bands to d_model
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, waveform):
        # waveform: (B, 1, T)
        # Compute mel spectrogram
        with torch.no_grad():
            spec = self.mel_spec(waveform.squeeze(1))  # (B, n_mels, T')
            spec = torch.log(spec + 1e-9)

        # Transpose to (B, T', n_mels) and project
        x = spec.transpose(1, 2)  # (B, T', n_mels)
        x = self.input_proj(x)  # (B, T', d_model)
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)  # (B, T', d_model)

        # Global average pooling
        x = x.mean(dim=1)  # (B, d_model)

        # Classify
        x = self.classifier(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Baseline for Audio Classification")
    parser.add_argument(
        "--dataset",
        type=str,
        default="speechcommands",
        choices=["speechcommands", "urbansound8k", "esc50"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_complex", type=int, default=128, help="Ignored, for compatibility")
    parser.add_argument("--n_layers", type=int, default=4, help="Ignored, for compatibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines/transformer_audio")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch_idx, (waveforms, labels) in enumerate(pbar):
        waveforms, labels = waveforms.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(waveforms)
                loss = F.cross_entropy(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(waveforms)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": total_loss / (batch_idx + 1), "acc": 100.0 * correct / total})

    scheduler.step()
    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def main():
    args = parse_args()

    if not TORCHAUDIO_AVAILABLE:
        print("torchaudio is required for audio classification.")
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    if args.dataset == "speechcommands":
        train_loader, val_loader, test_loader, num_classes = create_audio_dataloader(
            batch_size=args.batch_size, subset_size=args.subset_size
        )
    elif args.dataset == "urbansound8k":
        train_loader, val_loader, test_loader, num_classes = create_urbansound8k_dataloader(
            batch_size=args.batch_size, subset_size=args.subset_size
        )
    else:  # esc50
        train_loader, val_loader, test_loader, num_classes = create_esc50_dataloader(
            batch_size=args.batch_size, subset_size=args.subset_size
        )

    # Create model
    model = AudioTransformer(
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    # Training
    best_val_acc = 0
    patience_counter = 0

    run_name = args.run_name or f"transformer_audio_{args.dataset}_s{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time

    # Test evaluation
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, device)

    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Save results
    results = {
        "dataset": args.dataset,
        "model": "Transformer-Audio",
        "n_params": n_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_time_seconds": train_time,
        "epochs_trained": epoch,
        "args": vars(args),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
