from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class DataAdapter(ABC):
    """Abstract base class for domain-specific data adapters."""

    @abstractmethod
    def adapt(self, raw_sample: Any) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Convert raw sample to (input_features, mask, label).

        Returns:
            input_features: (L, D) or (D, H, W) depending on model expectation.
            mask: (L,) or None.
            label: scalar tensor.
        """


class VisionAdapter(DataAdapter):
    """Adapter for Image Data (e.g., MNIST)."""

    def adapt(
        self, raw_sample: tuple[torch.Tensor, int]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        img, label = raw_sample
        # img is (C, H, W).
        # Should we flatten to tokens here, or let model handle it?
        # CV-KAN expects (C, H, W) and patches internally, or (L, D).
        # The existing CVKANImageClassifier takes (B, C, H, W).
        # So we leave it as is.
        # Mask is None (typically no padding in images).
        return img, None, torch.tensor(label, dtype=torch.long)


class NLPAdapter(DataAdapter):
    """Adapter for Text Data (e.g., IMDB)."""

    def __init__(self, vocab_size: int = 20000, max_len: int = 256):
        self.vocab_size = vocab_size
        self.max_len = max_len
        # In a real scenario, this would hold a tokenizer.
        # For our pre-processed NLP loader, samples are already indices.

    def adapt(self, raw_sample: dict) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        # raw_sample from our NLPDataLoader is usually a dict batch if using the loader directly.
        # But if we wrap a dataset...
        # Let's assume raw_sample is token_ids tensor and label.
        input_ids, label = raw_sample

        # Create mask (non-zero entries)
        mask = (input_ids != 0).long()

        return input_ids, mask, torch.tensor(label, dtype=torch.long)


class UniversalDataset(Dataset):
    """Wrapper that applies a DataAdapter to any underlying Dataset."""

    _adapters: dict[str, type[DataAdapter]] = {}

    def __init__(self, dataset: Dataset, domain: str, **adapter_kwargs):
        self.dataset = dataset
        self.domain = domain

        if domain not in self._adapters:
            raise ValueError(f"Unknown domain: {domain}. Registered: {list(self._adapters.keys())}")

        self.adapter = self._adapters[domain](**adapter_kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_sample = self.dataset[idx]
        return self.adapter.adapt(raw_sample)

    @classmethod
    def register(cls, domain: str, adapter_cls: type[DataAdapter]):
        cls._adapters[domain] = adapter_cls


# Register defaults
UniversalDataset.register("vision", VisionAdapter)
UniversalDataset.register("nlp", NLPAdapter)
# Future: AudioAdapter, TimeseriesAdapter
