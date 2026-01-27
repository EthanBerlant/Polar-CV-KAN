import pytest

from src.configs.schema import ExperimentConfig
from src.factories import ModelFactory
from src.registry import DATASET_REGISTRY, Registry


def test_registry_basics():
    # Test internal registry mechanism
    reg = Registry("test")

    @reg.register("foo")
    def foo():
        pass

    assert reg.get("foo") == foo
    with pytest.raises(KeyError):
        reg.get("bar")


def test_config_defaults():
    config = ExperimentConfig()
    assert config.model.d_complex == 64
    assert config.data.dataset_name == "cifar10"


def test_dataset_factory_cifar10():
    config = ExperimentConfig()
    config.data.dataset_name = "cifar10"
    # Mock data dir to avoid download?
    # Just check if factory resolves it.

    # We can inspect the registry directly to see if it's there
    assert "cifar10" in DATASET_REGISTRY.list_available()


def test_model_factory_instantiation():
    config = ExperimentConfig()
    config.model.normalization = "batch"
    config.model.aggregation = "mean"
    meta = {"input_dim": 3, "n_classes": 10}

    model = ModelFactory.create(config, meta)
    assert model is not None
    # Check if normalization is correct type
    # model.backbone.layers ...
    # This involves inspecting the module structure


def test_model_factory_unknown_component():
    config = ExperimentConfig()
    config.model.normalization = "non_existent_norm"
    meta = {"input_dim": 3}

    with pytest.raises(ValueError):
        ModelFactory.create(config, meta)
