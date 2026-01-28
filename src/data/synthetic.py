from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class SignalNoiseDataset(Dataset):
    """Synthetic dataset with signal tokens embedded in noise."""

    def __init__(
        self,
        n_samples: int = 10_000,
        n_tokens: int = 16,
        k_signal: int = 4,
        d_complex: int = 32,
        signal_phase_std: float = 0.5,
        seed: int | None = 42,
    ) -> None:
        self.n_samples = n_samples
        self.n_tokens = n_tokens
        self.k_signal = k_signal
        self.d_complex = d_complex
        self.signal_phase_std = signal_phase_std
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def _generator(self, idx: int) -> torch.Generator:
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(self.seed + idx)
        return gen

    def _sample(self, idx: int) -> dict[str, Any]:
        gen = self._generator(idx)
        token_labels = torch.zeros(self.n_tokens, dtype=torch.long)
        signal_indices = torch.randperm(self.n_tokens, generator=gen)[: self.k_signal]
        token_labels[signal_indices] = 1

        phases = torch.empty(self.n_tokens, self.d_complex)

        signal_mask = token_labels == 1
        noise_mask = ~signal_mask

        phases[signal_mask] = torch.randn(
            signal_mask.sum(), self.d_complex, generator=gen
        ) * self.signal_phase_std
        phases[noise_mask] = (torch.rand(noise_mask.sum(), self.d_complex, generator=gen) * 2 - 1) * torch.pi

        magnitudes = torch.ones(self.n_tokens, self.d_complex)
        sequence = torch.polar(magnitudes, phases).to(torch.cfloat)

        return {"sequence": sequence, "token_labels": token_labels}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._sample(idx)


def create_signal_noise_dataloader(
    n_samples: int = 10_000,
    n_tokens: int = 16,
    k_signal: int = 4,
    d_complex: int = 32,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int | None = 42,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SignalNoiseDataset(
        n_samples=n_samples,
        n_tokens=n_tokens,
        k_signal=k_signal,
        d_complex=d_complex,
        seed=seed,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
