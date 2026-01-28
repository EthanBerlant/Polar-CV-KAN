from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from .text import load_agnews, load_imdb, load_sst2, pad_collate


class NLPDataLoader:
    """Utility wrapper for NLP dataset loading."""

    def __init__(
        self,
        dataset_name: str = "sst2",
        batch_size: int = 32,
        max_seq_len: int = 128,
        root: str = "./data",
        subset_size: int | None = None,
        num_workers: int = 0,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.root = root
        self.subset_size = subset_size
        self.num_workers = num_workers

    def _maybe_subset(self, dataset):
        if self.subset_size is None:
            return dataset
        indices = torch.randperm(len(dataset))[: self.subset_size].tolist()
        return Subset(dataset, indices)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader, int, int]:
        if self.dataset_name in {"sst2", "sst-2"}:
            train, test, vocab = load_sst2(
                root_dir=f"{self.root}/SST-2",
                max_len=self.max_seq_len,
            )
            n_classes = 2
        elif self.dataset_name == "imdb":
            train, test, vocab = load_imdb(
                root_dir=f"{self.root}/imdb",
                max_len=self.max_seq_len,
            )
            n_classes = 2
        elif self.dataset_name in {"agnews", "ag_news"}:
            train, test, vocab = load_agnews(
                root_dir=f"{self.root}/agnews",
                max_len=self.max_seq_len,
            )
            n_classes = 4
        else:
            raise ValueError(f"Unsupported NLP dataset '{self.dataset_name}'.")

        train = self._maybe_subset(train)
        test = self._maybe_subset(test)

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=pad_collate,
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=pad_collate,
        )

        return train_loader, test_loader, n_classes, len(vocab)
