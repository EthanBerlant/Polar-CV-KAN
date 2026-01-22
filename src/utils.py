"""Shared utilities for CV-KAN training and evaluation."""

import gc

import torch


def cleanup_gpu() -> None:
    """Force GPU memory cleanup.

    Call this:
    - Before creating models
    - After each epoch during training
    - At the end of training runs
    - When switching between train/eval modes with large batches
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
