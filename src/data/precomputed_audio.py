from __future__ import annotations

import os
from bisect import bisect_right
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class PrecomputedSpectrogramDataset(Dataset):
    """Dataset for precomputed spectrogram batches saved as .pt files."""

    def __init__(self, root: str, subset: str) -> None:
        self.root = Path(root)
        self.subset = subset
        self.subset_dir = self.root / subset

        if not self.subset_dir.exists():
            raise FileNotFoundError(f"Missing precomputed subset directory: {self.subset_dir}")

        self.batch_files = sorted(self.subset_dir.glob("batch_*.pt"))
        if not self.batch_files:
            raise FileNotFoundError(f"No precomputed batches found in {self.subset_dir}")

        self._lengths = []
        self._cumulative = []
        total = 0
        for batch_file in self.batch_files:
            payload = torch.load(batch_file, map_location="cpu")
            batch_len = payload["spectrograms"].shape[0]
            self._lengths.append(batch_len)
            total += batch_len
            self._cumulative.append(total)

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def _locate(self, idx: int) -> tuple[Path, int]:
        file_idx = bisect_right(self._cumulative, idx)
        prev_total = 0 if file_idx == 0 else self._cumulative[file_idx - 1]
        return self.batch_files[file_idx], idx - prev_total

    def __getitem__(self, idx: int):
        batch_file, local_idx = self._locate(idx)
        payload = torch.load(batch_file, map_location="cpu")
        return payload["spectrograms"][local_idx], payload["labels"][local_idx]


def create_precomputed_audio_dataloader(
    root: str = "./data/speech_commands_stft",
    batch_size: int = 64,
    num_workers: int = 0,
):
    root_path = Path(root)
    label_map_path = root_path / "label_map.pt"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Missing label map at {label_map_path}. Run precompute_audio.py first."
        )

    label_map = torch.load(label_map_path, map_location="cpu")
    n_classes = len(label_map)

    train_set = PrecomputedSpectrogramDataset(root, "training")
    val_set = PrecomputedSpectrogramDataset(root, "validation")
    test_set = PrecomputedSpectrogramDataset(root, "testing")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, n_classes
