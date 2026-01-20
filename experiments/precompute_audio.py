"""
Precompute STFT spectrograms for Speech Commands dataset.

Saves complex spectrograms to disk for fast loading during training.
This eliminates the STFT computation bottleneck.

Usage:
    python experiments/precompute_audio.py
"""

import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_data import TORCHAUDIO_AVAILABLE, SubsetSC


def precompute_spectrograms(
    root: str = "./data/speech_commands",
    output_dir: str = "./data/speech_commands_stft",
    n_fft: int = 512,
    hop_length: int = 128,
    target_length: int = 16000,  # 1 second at 16kHz
):
    """
    Precompute STFT spectrograms for all Speech Commands samples.

    Saves:
    - Complex spectrograms as .pt files (real + imag stacked)
    - Label indices
    """
    if not TORCHAUDIO_AVAILABLE:
        print("ERROR: torchaudio required")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Create STFT transform
    window = torch.hann_window(n_fft)

    def compute_stft(waveform):
        """Compute STFT and return as stacked real/imag."""
        # Pad or trim to target length
        if waveform.shape[-1] < target_length:
            padding = target_length - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[..., :target_length]

        # Compute STFT
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )  # (n_freq, time)

        # Transpose to (time, freq) and stack real/imag
        stft = stft.T  # (time, n_freq)
        return torch.stack([stft.real, stft.imag], dim=0)  # (2, time, n_freq)

    for subset_name in ["training", "validation", "testing"]:
        print(f"\nProcessing {subset_name}...")

        try:
            dataset = SubsetSC(subset_name, root=root, download=True)
        except Exception as e:
            print(f"Failed to load {subset_name}: {e}")
            continue

        # Get labels
        labels = sorted(list(set(item[2] for item in dataset)))
        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        # Save label mapping
        if subset_name == "training":
            torch.save(label_to_idx, os.path.join(output_dir, "label_map.pt"))
            print(f"  Labels: {len(labels)}")

        # Create subset output directory
        subset_dir = os.path.join(output_dir, subset_name)
        os.makedirs(subset_dir, exist_ok=True)

        # Process all samples
        spectrograms = []
        labels_list = []

        for i, item in enumerate(tqdm(dataset, desc="  Computing STFTs")):
            waveform, sample_rate, label, *_ = item

            # Compute STFT
            stft = compute_stft(waveform)
            spectrograms.append(stft)
            labels_list.append(label_to_idx[label])

            # Save in batches of 1000 to avoid memory issues
            if len(spectrograms) >= 1000:
                batch_idx = i // 1000
                torch.save(
                    {
                        "spectrograms": torch.stack(spectrograms),
                        "labels": torch.tensor(labels_list),
                    },
                    os.path.join(subset_dir, f"batch_{batch_idx:04d}.pt"),
                )
                spectrograms = []
                labels_list = []

        # Save remaining
        if spectrograms:
            batch_idx = len(dataset) // 1000
            torch.save(
                {
                    "spectrograms": torch.stack(spectrograms),
                    "labels": torch.tensor(labels_list),
                },
                os.path.join(subset_dir, f"batch_{batch_idx:04d}.pt"),
            )

        print(f"  Saved {len(dataset)} samples to {subset_dir}")

    print(f"\nPrecomputation complete! Output: {output_dir}")
    print("Update train_audio.py to use precomputed spectrograms for faster training.")


if __name__ == "__main__":
    precompute_spectrograms()
