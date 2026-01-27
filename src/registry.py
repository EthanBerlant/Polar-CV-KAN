from collections.abc import Callable
from typing import Any


class Registry:
    """A simple registry to map names to classes/functions."""

    def __init__(self, name: str) -> None:
        """Initialize the registry.

        Args:
            name: Name of the registry.
        """
        self.name = name
        self._registry: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable:
        """Decorator to register a class or function.

        Args:
            name: Optional name for the object. If None, uses class/func name.

        Returns:
            Decorator function.
        """

        def _register_wrapper(obj: Any) -> Any:
            key = name if name is not None else obj.__name__.lower()
            if key in self._registry:
                raise ValueError(f"'{key}' is already registered in {self.name}")
            self._registry[key] = obj
            return obj

        return _register_wrapper

    def get(self, name: str) -> Any:
        """Retrieve an object by name.

        Args:
            name: Key to lookup.

        Returns:
            The registered object (class or function).

        Raises:
            KeyError: If name is not found.
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self.name}. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list_available(self) -> list[str]:
        """List all registered names."""
        return list(self._registry.keys())


# Global Registries
NORMALIZATION_REGISTRY = Registry("normalization")
AGGREGATION_REGISTRY = Registry("aggregation")
EMBEDDING_REGISTRY = Registry("embedding")
BACKBONE_REGISTRY = Registry("backbone")
DATASET_REGISTRY = Registry("dataset")
MODEL_REGISTRY = Registry("model")
