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
            kwargs.update(
                {
                    "img_size": int(img_size) if img_size else 32,
                    "in_channels": in_channels,
                    "patch_size": 4,  # Config? or default
                }
            )
            if requested_patch_type:
                kwargs["embedding_type"] = requested_patch_type

        if embedding_type == "token" and meta:
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

        # Fallback to existing domain logic if not in registry?
        # Or wrap existing logic into registry.
        # Let's rely on 'src.data.factories' or similar.
        # For now, let's look up specific domain modules based on name convention
        domain_name = "image"  # default
        if ds_name in ["cifar10", "mnist", "fashion_mnist"]:
            domain_name = "image"
        elif ds_name in ["imdb", "ag_news"]:
            domain_name = "text"

        try:
            if domain_name == "image":
                from .data.domains import image as domain_module
            elif domain_name == "text":
                from .data.domains import text as domain_module
            elif domain_name == "audio":
                from .data.domains import audio as domain_module
            else:
                raise ValueError(f"Unknown domain for dataset {ds_name}")

            # Adapt Config to what legacy modules expect
            # Legacy expected a single Namespace or Dict with img_size, batch_size etc.
            # It expects (model_config, train_config)
            # We can pass our new config objects if they are compatible attributes
            # Or convert to SimpleNamespaces

            # Adapt Model Config
            legacy_model_config = Namespace(**asdict(config.model))
            if hasattr(config.data, "img_size"):
                legacy_model_config.img_size = getattr(config.data, "img_size", 32)
            else:
                legacy_model_config.img_size = 32

            legacy_train_config = Namespace(**asdict(config.data), **asdict(config.trainer))
            return domain_module.create_dataloaders(legacy_model_config, legacy_train_config)
        except ImportError as err:
            raise ValueError(
                f"Could not load data for dataset '{ds_name}' (domain: {domain_name})"
            ) from err
