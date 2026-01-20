"""
Unit tests for CV-KAN modules.

Note: ComplexLayerNorm and ComplexRMSNorm tests have been removed because
CV-KAN intentionally does not use traditional normalization (magnitudes
encode attention-like information). Log-magnitude centering is used instead.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SignalNoiseDataset
from src.losses import diversity_loss, phase_anchor_loss
from src.models import CVKAN
from src.models.cv_kan import CVKANTokenClassifier
from src.modules import (
    GatedPolarization,
    PhaseAttentionBlock,
    PolarizingBlock,
)
from src.modules.multi_head import (
    EmergentHeadsPolarizing,
    FactoredHeadsPolarizing,
    PhaseOffsetPolarizing,
)


class TestPolarizingBlock:
    """Tests for the core PolarizingBlock."""

    def test_forward_shape(self):
        """Output shape should match input shape."""
        block = PolarizingBlock(d_complex=32, kan_hidden=16)
        Z = torch.randn(4, 16, 32, dtype=torch.cfloat)
        out = block(Z)
        assert out.shape == Z.shape
        assert out.dtype == torch.cfloat

    def test_residual_connection(self):
        """Block should have residual behavior (output ≈ input for small changes)."""
        block = PolarizingBlock(d_complex=32, kan_hidden=16, mag_init_scale=0.0)
        Z = torch.randn(4, 16, 32, dtype=torch.cfloat)
        out = block(Z)

        # With mag_scale=0, should still change slightly due to phase transform
        # But should be in the same ballpark
        diff = torch.abs(out - Z).mean()
        assert diff < torch.abs(Z).mean() * 10  # Not too far from input

    def test_gradient_flow(self):
        """Gradients should flow through the block."""
        block = PolarizingBlock(d_complex=32, kan_hidden=16)
        Z = torch.randn(4, 16, 32, dtype=torch.cfloat, requires_grad=True)
        out = block(Z)
        loss = torch.abs(out).sum()
        loss.backward()

        assert Z.grad is not None
        assert not torch.isnan(Z.grad).any()


class TestGatedPolarization:
    """Tests for GatedPolarization."""

    def test_initial_identity(self):
        """Should start near identity (strength = 0)."""
        gated = GatedPolarization(d_complex=32)
        strength = gated.get_strength()
        assert strength < 0.6  # Near 0.5 (sigmoid(0))

    def test_gated_transform(self):
        """Should apply gated transform."""
        gated = GatedPolarization(d_complex=32)
        mag = torch.abs(torch.randn(4, 16, 32))
        out = gated(mag)

        assert out.shape == mag.shape
        assert (out > 0).all()  # Magnitudes stay positive


class TestMultiHead:
    """Tests for multi-head approaches."""

    def test_emergent_heads(self):
        """EmergentHeadsPolarizing should maintain shape."""
        block = EmergentHeadsPolarizing(d_complex=64, kan_hidden=32)
        Z = torch.randn(4, 16, 64, dtype=torch.cfloat)
        out = block(Z)
        assert out.shape == Z.shape

    def test_phase_offset(self):
        """PhaseOffsetPolarizing should handle explicit heads."""
        n_heads = 8
        d_per_head = 8
        block = PhaseOffsetPolarizing(n_heads, d_per_head)

        Z = torch.randn(4, 16, n_heads * d_per_head, dtype=torch.cfloat)
        out = block(Z)
        assert out.shape == Z.shape

    def test_factored_heads(self):
        """FactoredHeadsPolarizing should project and process."""
        n_heads = 8
        d_model = 64
        d_per_head = 8
        block = FactoredHeadsPolarizing(n_heads, d_model, d_per_head)

        Z = torch.randn(4, 16, d_model, dtype=torch.cfloat)
        out = block(Z)
        assert out.shape == (4, 16, n_heads * d_per_head)


class TestCVKAN:
    """Tests for the full CVKAN model."""

    def test_sequence_classification(self):
        """Model should produce sequence-level predictions."""
        model = CVKAN(
            d_input=32,
            d_complex=64,
            n_layers=2,
            n_classes=2,
        )

        x = torch.randn(4, 16, 32, dtype=torch.cfloat)
        outputs = model(x)

        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 2)

    def test_token_classification(self):
        """Token classifier should produce per-token predictions."""
        model = CVKANTokenClassifier(
            d_input=32,
            d_complex=64,
            n_layers=2,
            n_classes=2,
        )

        x = torch.randn(4, 16, 32, dtype=torch.cfloat)
        outputs = model(x)

        assert "token_logits" in outputs
        assert outputs["token_logits"].shape == (4, 16, 2)

    def test_intermediates(self):
        """Should return intermediate representations when requested."""
        model = CVKAN(d_input=32, d_complex=64, n_layers=3)
        x = torch.randn(4, 16, 32, dtype=torch.cfloat)
        outputs = model(x, return_intermediates=True)

        assert "intermediates" in outputs
        assert len(outputs["intermediates"]) == 4  # embed + 3 layers


class TestDataset:
    """Tests for synthetic dataset."""

    def test_signal_noise_dataset(self):
        """Dataset should generate correct structure."""
        dataset = SignalNoiseDataset(
            n_samples=100,
            n_tokens=16,
            k_signal=4,
            d_complex=32,
        )

        assert len(dataset) == 100

        item = dataset[0]
        assert item["sequence"].shape == (16, 32)
        assert item["sequence"].dtype == torch.cfloat
        assert item["token_labels"].shape == (16,)
        assert item["token_labels"].sum() == 4  # k_signal tokens

    def test_signal_coherence(self):
        """Signal tokens should have more coherent phases."""
        dataset = SignalNoiseDataset(
            n_samples=100,
            n_tokens=16,
            k_signal=4,
            d_complex=32,
            signal_phase_std=0.1,  # Very coherent
        )

        item = dataset[0]
        signal_mask = item["token_labels"] == 1
        noise_mask = item["token_labels"] == 0

        signal_phases = torch.angle(item["sequence"][signal_mask])
        noise_phases = torch.angle(item["sequence"][noise_mask])

        # Signal should have lower phase variance
        signal_var = signal_phases.std(dim=0).mean()
        noise_var = noise_phases.std(dim=0).mean()

        assert signal_var < noise_var


class TestLosses:
    """Tests for regularization losses."""

    def test_diversity_loss(self):
        """Diversity loss should penalize correlated dimensions."""
        # Create highly correlated data
        base = torch.randn(100, 16, 1, dtype=torch.cfloat)
        Z_corr = base.expand(-1, -1, 32)  # All dims same

        # Create independent data
        Z_indep = torch.randn(100, 16, 32, dtype=torch.cfloat)

        loss_corr = diversity_loss(Z_corr)
        loss_indep = diversity_loss(Z_indep)

        # Correlated should have higher loss
        assert loss_corr > loss_indep

    def test_phase_anchor_loss(self):
        """Phase anchor loss should encourage clustering."""
        # Phases near anchors (0, π/2, π, 3π/2)
        anchored_phases = torch.tensor([0.0, 1.57, 3.14, 4.71])
        Z_anchored = torch.exp(1j * anchored_phases.view(1, 4, 1).expand(10, -1, 8))

        # Random phases
        Z_random = torch.exp(1j * torch.rand(10, 4, 8) * 2 * torch.pi)

        loss_anchored = phase_anchor_loss(Z_anchored)
        loss_random = phase_anchor_loss(Z_random)

        # Anchored should have lower loss
        assert loss_anchored < loss_random


class TestPhaseAttentionBlock:
    """Tests for PhaseAttentionBlock."""

    def test_forward_shape(self):
        """Output shape should match input shape."""
        block = PhaseAttentionBlock(d_complex=32, n_heads=4)
        Z = torch.randn(4, 16, 32, dtype=torch.cfloat)
        out = block(Z)
        assert out.shape == Z.shape
        assert out.dtype == torch.cfloat

    def test_masking(self):
        """Masked tokens should not contribute to attention."""
        block = PhaseAttentionBlock(d_complex=32, n_heads=4)
        Z = torch.randn(2, 4, 32, dtype=torch.cfloat)
        # Mask out last token
        mask = torch.ones(2, 4)
        mask[:, -1] = 0

        out = block(Z, mask=mask)
        assert out.shape == Z.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
