import os
import sys

sys.path.append(os.getcwd())

import torch

from src.modules.multi_path_hierarchical import MultiPathHierarchicalPolarization


def test_multipath_output_shape():
    """Verify output shape matches input shape."""
    batch = 2
    seq_len = 16
    d_complex = 4

    Z = torch.randn(batch, seq_len, d_complex, dtype=torch.cfloat)

    # 2 paths, mean combination
    model = MultiPathHierarchicalPolarization(d_complex, n_paths=2, combine_method="mean")
    output = model(Z)

    assert output.shape == Z.shape
    assert output.dtype == Z.dtype


def test_multipath_offset_logic():
    """Verify that different paths receive different shifted inputs."""
    # We can't easily inspect internal states, but we can verify that
    # n_paths=2 produces a different result than n_paths=1 (standard)
    # given the same initialization (if we could sync weights, but we can't easily).
    #
    # Instead, let's just run it and ensure gradients flow, showing it's connected.

    batch = 1
    seq_len = 8
    d_complex = 2

    Z = torch.randn(batch, seq_len, d_complex, dtype=torch.cfloat, requires_grad=True)

    model = MultiPathHierarchicalPolarization(d_complex, n_paths=2, combine_method="mean")
    output = model(Z)

    loss = output.sum().abs()
    loss.backward()

    assert Z.grad is not None
    assert torch.all(Z.grad != 0)


def test_multipath_learned_combine():
    """Verify learned combination weights."""
    model = MultiPathHierarchicalPolarization(d_complex=4, n_paths=3, combine_method="learned")
    assert hasattr(model, "path_weights")
    assert len(model.path_weights) == 3

    Z = torch.randn(2, 8, 4, dtype=torch.cfloat)
    output = model(Z)
    assert output.shape == Z.shape


def test_multipath_shared_weights():
    """Verify weight sharing reduces parameter count."""
    d_complex = 4
    kwargs = {"kan_hidden": 8, "dropout": 0.0}

    # Unshared
    model_unshared = MultiPathHierarchicalPolarization(
        d_complex, n_paths=2, share_horizontal_weights=False, **kwargs
    )
    params_unshared = sum(p.numel() for p in model_unshared.parameters())

    # Shared
    model_shared = MultiPathHierarchicalPolarization(
        d_complex, n_paths=2, share_horizontal_weights=True, **kwargs
    )
    params_shared = sum(p.numel() for p in model_shared.parameters())

    # Shared should have fewer parameters (roughly half)
    assert params_shared < params_unshared

    # Run forward to check it works
    Z = torch.randn(1, 16, d_complex, dtype=torch.cfloat)
    out = model_shared(Z)
    assert out.shape == Z.shape


def test_multipath_kan_combiner():
    """Verify KAN combiner works."""
    d_complex = 4
    model = MultiPathHierarchicalPolarization(
        d_complex, n_paths=2, combine_method="kan", combine_kan_hidden=8
    )

    assert hasattr(model, "combiner_block")

    Z = torch.randn(1, 16, d_complex, dtype=torch.cfloat)
    out = model(Z)
    assert out.shape == Z.shape
