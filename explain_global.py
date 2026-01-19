
import torch
import torch.nn as nn
from src.models.cv_kan_image import CVKANImageClassifier

def demonstrate_global_limitations():
    print("Demonstrating Global Aggregation Limitations...")
    # Create model with global aggregation (now correctly passed after fix)
    model = CVKANImageClassifier(
        img_size=32,
        patch_size=4,
        d_complex=32,
        n_layers=2,
        aggregation='global'
    )
    
    # Create an image with a pattern in the top-left
    x1 = torch.zeros(1, 3, 32, 32)
    x1[:, :, 0:8, 0:8] = 1.0
    
    # Create an image with the SAME pattern in the bottom-right
    x2 = torch.zeros(1, 3, 32, 32)
    x2[:, :, 24:32, 24:32] = 1.0
    
    # Run both through the model
    # Global aggregation sums/averages everything, so it is "invariant" to position too early
    # But usually we want equivariance in early layers (detect "corner" at top-left vs bottom-right)
    
    # Let's inspect the features after layer 1
    z1, _ = model.patch_embed(x1)
    z2, _ = model.patch_embed(x2)
    
    # Apply global aggregation layer
    out1 = model.layers[0](z1)
    out2 = model.layers[0](z2)
    
    # Check if the OUTPUT features preserve the location difference
    # With global aggregation, the *update* (A_new) is identical if the "sum of features" is identical
    # Since x1 and x2 have same pixel values just shifted, their global mean is IDENTICAL.
    # So the contextual update A_new will be IDENTICAL.
    
    # Check aggregating signal
    agg = model.layers[0].aggregation
    A1 = agg(z1)
    A2 = agg(z2)
    
    print(f"Aggregated signal difference: {(A1 - A2).abs().sum().item():.6f}")
    
    if (A1 - A2).abs().sum().item() < 1e-5:
        print("CONFIRMED: Global aggregation produces IDENTICAL context for spatially shifted images.")
        print("This means the model cannot distinguish WHERE features are, only THAT they exist.")
        print("For ImageNet/CIFAR, spatial layout matters (sky is up, grass is down).")
        print("Conclusion: Global aggregation is too destructive for early layers.")
    else:
        print("Surprise! Global aggregation differed.")

if __name__ == "__main__":
    demonstrate_global_limitations()
