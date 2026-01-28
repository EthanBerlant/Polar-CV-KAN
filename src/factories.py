import contextlib
from argparse import Namespace
from dataclasses import asdict
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from .configs.model import ExperimentConfig, ModelConfig
from .models.backbone import CVKANBackbone
from .models.cv_kan import CVKAN, StandardClassificationHead, TokenClassificationHead
from .models.cv_kan_image import ImageClassificationHead
from .registry import DATASET_REGISTRY, EMBEDDING_REGISTRY, MODEL_REGISTRY


class ModelFactory:
    """Factory for creating CV-KAN models from configuration.

    Centralizes all model creation logic, ensuring consistent use of registries.
    """

    @staticmethod
    def create(config: ExperimentConfig, meta: dict[str, Any] | None = None) -> nn.Module:
        """Create a model based on experiment configuration."""
        # 1. Check Model Registry first (for custom/legacy models)
        # However, standard CV-KAN is compositional, so we usually build it here.
        model_type = getattr(config.model, "model_type", "cv-kan")

        if model_type != "cv-kan":
            # Lookup in registry
            with contextlib.suppress(KeyError):
                model_cls = MODEL_REGISTRY.get(model_type)
                return model_cls(config.model)

        # 2. Build CV-KAN Components
        # Embedding
        embedding = ModelFactory._create_embedding(config.model, meta)

        # Backbone
        # Note: Backbone contains the stack of Polarizing Blocks
        backbone = CVKANBackbone(config.model)

        # Head
        head = ModelFactory._create_head(config.model, meta, config.model.d_complex)

        # Assemble
        return CVKAN(embedding=embedding, backbone=backbone, head=head)

    @staticmethod
    def _create_embedding(config: ModelConfig, meta: dict | None) -> nn.Module:
        # Determine embedding type based on data meta or config
        # Heuristic: if meta has 'img_size', usage ImageEmbedding
        embedding_type = "linear"  # default
        if meta and ("img_size" in meta or "image_size" in meta):
            embedding_type = "image_patch"
        elif meta and "vocab_size" in meta:
            embedding_type = "token"

        # Allow config override
        requested_patch_type = None
        if getattr(config, "embedding_type", None):
            embedding_type = config.embedding_type
            if embedding_type == "conv":
                requested_patch_type = "conv"
                embedding_type = "image_patch"

        try:
            emb_cls = EMBEDDING_REGISTRY.get(embedding_type)
        except KeyError as err:
            raise ValueError(f"Embedding type '{embedding_type}' not found.") from err

        # Extract relevant args from config and meta
        # This is where a Config -> Args mapping is needed.
        # For now, we pass what we can.
        kwargs = {"d_complex": config.d_complex}

        if embedding_type == "image_patch" and meta:
            img_size = meta.get("img_size", meta.get("image_size", 32))
            in_channels = meta.get("in_channels", 3)
            patch_size = meta.get("patch_size", getattr(config, "patch_size", 4))
            kwargs.update(
                {
                    "img_size": int(img_size) if img_size else 32,
                    "in_channels": in_channels,
                    "patch_size": patch_size,
                }
            )
            if requested_patch_type:
                kwargs["embedding_type"] = requested_patch_type

        if embedding_type == "token" and meta:
            if "vocab_size" not in meta:
                raise ValueError("Token embeddings require 'vocab_size' in dataset metadata.")
            kwargs.update({"vocab_size": meta["vocab_size"]})

        if embedding_type == "linear" and meta:
            kwargs.update({"input_dim": meta.get("input_dim", meta.get("n_features", 10))})

        return emb_cls(**kwargs)

    @staticmethod
    def _create_head(config: ModelConfig, meta: dict | None, d_complex: int) -> nn.Module:
        n_classes = meta.get("n_classes", 10) if meta else 10
        task_type = meta.get("task_type", "classification") if meta else "classification"
        kan_hidden = getattr(config, "kan_hidden", 32)
        pooling = getattr(config, "pooling", "mean")
        dropout = getattr(config, "dropout", 0.0)

        if task_type == "token_classification":
            return TokenClassificationHead(d_complex, n_classes, kan_hidden, dropout)
        if task_type == "image_classification":
            return ImageClassificationHead(d_complex, n_classes, kan_hidden, pooling, dropout)
        return StandardClassificationHead(d_complex, n_classes, kan_hidden, pooling, dropout)


class DataFactory:
    """Factory for dataset loading."""

    @staticmethod
    def create_dataloaders(
        config: ExperimentConfig,
    ) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
        """Load data using dataset registry strategies."""
        ds_name = config.data.dataset_name

        # 1. Check Registry
        # Use contextlib.suppress here? No, logic flow is a bit complex
        # because we don't assign variable in suppress block easily for later use
        # But here we just check availability.

        # Heuristic: Check available registries
        if ds_name in DATASET_REGISTRY.list_available():
            loader_fn = DATASET_REGISTRY.get(ds_name)
            return loader_fn(config.data)

        # Fallback to explicit module mapping for legacy loaders.
        legacy_image_loaders = {
            "cifar10": "create_cifar10_dataloader",
            "cifar100": "create_cifar100_dataloader",
            "fashion_mnist": "create_fashionmnist_dataloader",
        }

        if ds_name in legacy_image_loaders:
            from .data import image_data as domain_module

            loader_name = legacy_image_loaders[ds_name]
            loader_fn = getattr(domain_module, loader_name, None)
            if loader_fn is None:
                raise ValueError(f"Legacy loader '{loader_name}' missing for dataset '{ds_name}'.")

            train_loader, val_loader, test_loader, meta = loader_fn(
                root=config.data.data_dir,
                batch_size=config.data.batch_size,
                image_size=config.data.img_size,
                num_workers=config.data.num_workers,
                subset_size=config.data.subset_size,
            )
            if not isinstance(meta, dict):
                meta = {
                    "n_classes": int(meta),
                    "task_type": "image_classification",
                    "img_size": config.data.img_size,
                    "in_channels": 3,
                }
            return train_loader, val_loader, test_loader, meta

        raise ValueError(f"Unknown dataset '{ds_name}'. Register it in DATASET_REGISTRY.")
