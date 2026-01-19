"""Quick test to verify all modules work."""
import torch
import sys
sys.path.insert(0, '.')

from src.modules import PolarizingBlock, GatedPolarization
from src.modules.multi_head import (
    EmergentHeadsPolarizing, 
    PhaseOffsetPolarizing, 
    FactoredHeadsPolarizing
)
from src.models import CVKAN
from src.models.cv_kan import CVKANTokenClassifier
from src.data import SignalNoiseDataset
from src.losses import diversity_loss, phase_anchor_loss

print("All imports successful!")

# Test PolarizingBlock
Z = torch.randn(4, 16, 32, dtype=torch.cfloat)
block = PolarizingBlock(32)
out = block(Z)
print(f"PolarizingBlock: {Z.shape} -> {out.shape}")

# Test multi-head approaches
emergent = EmergentHeadsPolarizing(64)
Z64 = torch.randn(4, 16, 64, dtype=torch.cfloat)
out_e = emergent(Z64)
print(f"EmergentHeads: {Z64.shape} -> {out_e.shape}")

offset = PhaseOffsetPolarizing(n_heads=8, d_per_head=8)
out_o = offset(Z64)
print(f"PhaseOffset: {Z64.shape} -> {out_o.shape}")

factored = FactoredHeadsPolarizing(n_heads=8, d_model=64, d_per_head=8)
out_f = factored(Z64)
print(f"FactoredHeads: {Z64.shape} -> {out_f.shape}")

# Test full model
model = CVKANTokenClassifier(
    d_input=32, 
    d_complex=32, 
    n_layers=2, 
    n_classes=2
)
outputs = model(Z)
print(f"CVKANTokenClassifier: token_logits {outputs['token_logits'].shape}")

# Test dataset
dataset = SignalNoiseDataset(n_samples=100, n_tokens=16, k_signal=4, d_complex=32)
item = dataset[0]
print(f"Dataset: sequence {item['sequence'].shape}, labels {item['token_labels'].shape}")

# Test losses
div_loss = diversity_loss(Z)
anchor_loss = phase_anchor_loss(Z)
print(f"Losses: diversity={div_loss.item():.4f}, anchor={anchor_loss.item():.4f}")

print("\n✓ All tests passed!")
