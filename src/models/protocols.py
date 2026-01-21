from typing import Any, Protocol

import torch


class Embedding(Protocol):
    """Protocol for input embeddings (Real/Index -> Complex)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to complex representation."""
        ...


class Backbone(Protocol):
    """Protocol for CV-KAN layer stacks (Complex -> Complex)."""

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Process complex sequence. Returns refined complex sequence."""
        ...


class Head(Protocol):
    """Protocol for task heads (Complex Sequence -> Output)."""

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
        """Produce task output from complex sequence."""
        ...
