"""
LSTM Baseline for Audio Classification.

Bidirectional LSTM on mel-spectrograms.
Sized to match CV-KAN parameter count (~200-500k params).
"""

import argparse
import json
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


class AudioLSTM(nn.Module):
    """
    Bidirectional LSTM for audio classification on spectrograms.
    """

    def __init__(
        self,
        n_mels=64,
        num_classes=35,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
    ):
        super().__init__()

        # Spectrogram frontend
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=n_mels
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Classification head
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, waveform):
        # waveform: (B, 1, T)
        # Compute mel spectrogram
        with torch.no_grad():
            spec = self.mel_spec(waveform.squeeze(1))  # (B, n_mels, T')
            spec = torch.log(spec + 1e-9)

        # Transpose to (B, T', n_mels)
        x = spec.transpose(1, 2)  # (B, T', n_mels)

        # LSTM
        x, _ = self.lstm(x)  # (B, T', hidden_dim * 2)

        # Use last timestep or mean pooling
        x = x.mean(dim=1)  # (B, hidden_dim * 2)

        # Classify
        x = self.classifier(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM Baseline for Audio Classification")
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
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_complex", type=int, default=128, help="Ignored, for compatibility")
    parser.add_argument("--n_layers", type=int, default=4, help="Ignored, for compatibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines/lstm_audio")
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
    model = AudioLSTM(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
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

    run_name = args.run_name or f"lstm_audio_{args.dataset}_s{args.seed}"
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
        "model": "LSTM-Audio",
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
