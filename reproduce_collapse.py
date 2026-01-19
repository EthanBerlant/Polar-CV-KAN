
import torch
import torch.nn as nn
from src.models.cv_kan_image import CVKANImageClassifier

def test_global_collapse():
    print("Testing Global Aggregation Collapse...")
    # Create model with global aggregation
    model = CVKANImageClassifier(
        img_size=32,
        patch_size=4,
        d_complex=32,
        n_layers=2,
        aggregation='global'
    )
    
    # Create two different random inputs
    x = torch.randn(2, 3, 32, 32)
    
    # Get patch embeddings (z0)
    z0, _ = model.patch_embed(x)
    
    # Calculate initial relative differences between first two tokens of first image
    diff_0 = z0[0, 0] - z0[0, 1]
    print(f"Initial diff between token 0 and 1: {diff_0.abs().mean().item():.6f}")
    
    # Run through layers manually to inspect
    z = z0.clone()
    for i, layer in enumerate(model.layers):
        z = layer(z)
        diff_i = z[0, 0] - z[0, 1]
        print(f"Layer {i} diff between token 0 and 1: {diff_i.abs().mean().item():.6f}")
        
        # Check if difference is preserved
        err = (diff_i - diff_0).abs().max()
        assert err < 1e-5, "Relative difference should be preserved with global aggregation!"

    print("Confirmed: Global aggregation preserves relative differences.")

def test_local_mixing():
    print("\nTesting Local Aggregation Mixing...")
    # Create model with local aggregation
    model = CVKANImageClassifier(
        img_size=32,
        patch_size=4,
        d_complex=32,
        n_layers=2,
        aggregation='local',
        local_kernel_size=3
    )
    
    x = torch.randn(2, 3, 32, 32)
    z0, spatial_shape = model.patch_embed(x)
    print(f"Spatial shape: {spatial_shape}")
    
    # Apply one layer
    z = z0.clone()
    
    # Calculate initial diff
    diff_0 = z0[0, 0] - z0[0, 1]
    
    print("Running layers...")
    for i, layer in enumerate(model.layers):
        # Note: PolarizingBlock doesn't take spatial_shape, depends on dynamic check in LocalWindowAggregation
        z = layer(z)
        diff_i = z[0, 0] - z[0, 1]
        print(f"Layer {i} diff between token 0 and 1: {diff_i.abs().mean().item():.6f}")
        
        # Check if difference changed
        err = (diff_i - diff_0).abs().max()
        print(f"  Change relative to initial: {err.item():.6e}")
        
        if err > 1e-5:
            print(f"  -> Layer {i} mixed information successfully!")
    
    if (z[0, 0] - z[0, 1] - diff_0).abs().max() > 1e-5:
        print("Success: Local aggregation changes relative structure.")
    else:
        print("FAILURE: Local aggregation did NOT change relative structure.")

if __name__ == "__main__":
    test_global_collapse()
    test_local_mixing()
