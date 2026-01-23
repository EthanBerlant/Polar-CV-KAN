"""
Unit tests for HierarchicalPolarization.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.hierarchical import HierarchicalPolarization


class TestHierarchicalPolarization:
    """Tests for HierarchicalPolarization."""

    def test_forward_shape_power_of_2(self):
        """Output shape should match input for power-of-2 lengths."""
        model = HierarchicalPolarization(d_complex=32)
        Z = torch.randn(4, 16, 32, dtype=torch.cfloat)
        out = model(Z)
        assert out.shape == Z.shape
        assert out.dtype == torch.cfloat

    def test_forward_shape_non_power_of_2(self):
        """Should handle non-power-of-2 sequence lengths."""
        model = HierarchicalPolarization(d_complex=32)
        Z = torch.randn(4, 13, 32, dtype=torch.cfloat)  # 13 is not power of 2
        out = model(Z)
        assert out.shape == Z.shape

    def test_gradient_flow(self):
        """Gradients should flow through all levels."""
        model = HierarchicalPolarization(d_complex=32, max_levels=3)
        Z = torch.randn(4, 8, 32, dtype=torch.cfloat, requires_grad=True)
        out = model(Z)
        loss = torch.abs(out).sum()
        loss.backward()

        assert Z.grad is not None
        assert not torch.isnan(Z.grad).any()

    def test_weight_sharing_shared(self):
        """Shared mode should have only one transform."""
        model = HierarchicalPolarization(d_complex=32, weight_sharing="shared")
        assert len(model.up_transforms) == 1

    def test_weight_sharing_per_level(self):
        """Per-level mode should have multiple transforms."""
        model = HierarchicalPolarization(d_complex=32, weight_sharing="per_level")
        assert len(model.up_transforms) > 1

    def test_top_down_none(self):
        """No top-down should have no down transforms."""
        model = HierarchicalPolarization(d_complex=32, top_down="none")
        assert model.down_transforms is None

    def test_top_down_learned(self):
        """Learned top-down should have separate transforms."""
        model = HierarchicalPolarization(d_complex=32, top_down="learned")
        assert model.down_transforms is not None
        assert len(model.down_transforms) > 0

    def test_top_down_mirror(self):
        """Mirror top-down should reuse up transforms."""
        model = HierarchicalPolarization(d_complex=32, top_down="mirror")
        Z = torch.randn(4, 8, 32, dtype=torch.cfloat)
        out = model(Z)  # Should not error
        assert out.shape == Z.shape

    def test_aggregation_mean(self):
        """Mean aggregation should work."""
        model = HierarchicalPolarization(d_complex=32, aggregation="mean")
        Z = torch.randn(4, 8, 32, dtype=torch.cfloat)
        out = model(Z)
        assert out.shape == Z.shape

    def test_aggregation_magnitude_weighted(self):
        """Magnitude-weighted aggregation should work."""
        model = HierarchicalPolarization(d_complex=32, aggregation="magnitude_weighted")
        Z = torch.randn(4, 8, 32, dtype=torch.cfloat)
        out = model(Z)
        assert out.shape == Z.shape

    def test_max_levels_limit(self):
        """Max levels should limit recursion depth."""
        model = HierarchicalPolarization(d_complex=32, max_levels=2)
        Z = torch.randn(4, 64, 32, dtype=torch.cfloat)  # Would be 6 levels normally
        out = model(Z)
        assert out.shape == Z.shape

    def test_single_token(self):
        """Should handle single token sequences."""
        model = HierarchicalPolarization(d_complex=32)
        Z = torch.randn(4, 1, 32, dtype=torch.cfloat)
        out = model(Z)
        assert out.shape == Z.shape

    def test_get_config(self):
        """Config should be retrievable."""
        model = HierarchicalPolarization(
            d_complex=64,
            max_levels=4,
            weight_sharing="per_level",
            aggregation="magnitude_weighted",
            top_down="learned",
        )
        config = model.get_config()
        assert config["d_complex"] == 64
        assert config["max_levels"] == 4
        assert config["weight_sharing"] == "per_level"
        assert config["aggregation"] == "magnitude_weighted"
        assert config["top_down"] == "learned"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
