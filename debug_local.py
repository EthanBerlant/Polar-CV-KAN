
import torch
import torch.nn as nn
from src.modules.aggregation import LocalWindowAggregation, GlobalMeanAggregation
from src.modules.polarizing_block import PolarizingBlock

def debug_local_agg():
    print("Debugging Local Aggregation...")
    
    # Setup
    batch, h, w, d = 2, 8, 8, 32
    n = h * w
    z = torch.randn(batch, n, d, dtype=torch.cfloat)
    
    # 1. Test Aggregation directly
    agg = LocalWindowAggregation(kernel_size=3, stride=1)
    # It infers shape from n=64 -> 8x8
    a = agg(z)
    
    print(f"Input Z shape: {z.shape}")
    print(f"Agg A shape: {a.shape}")
    
    # Check if A varies spatially
    # Reshape A to grid to check neighbors
    a_grid = a.view(batch, h, w, d)
    diff = (a_grid[:, 0, 0] - a_grid[:, 0, 1]).abs().mean()
    print(f"Agg A diff between (0,0) and (0,1): {diff.item():.6f}")
    
    if diff.item() < 1e-5:
        print("CRITICAL: Aggregation output is spatially CONSTANT!")
    else:
        print("Aggregation output is spatially varying (Good).")
        
    # 2. Test PolarizingBlock with this Aggregation
    block = PolarizingBlock(d, aggregation=agg)
    z_out = block(z)
    
    # Check output
    z_out_grid = z_out.view(batch, h, w, d)
    diff_out = (z_out_grid[:, 0, 0] - z_out_grid[:, 0, 1]).abs().mean()
    print(f"Block Output diff between (0,0) and (0,1): {diff_out.item():.6f}")
    
    z_grid = z.view(batch, h, w, d)
    diff_in = (z_grid[:, 0, 0] - z_grid[:, 0, 1]).abs().mean()
    print(f"Input diff between (0,0) and (0,1): {diff_in.item():.6f}")
    
    change = (diff_out - diff_in).abs()
    print(f"Change in diff: {change.item():.6e}")

    if change.item() < 1e-5:
        print("CRITICAL: Block output preserves exact relative structure (Collapse).")
        
        # Dig deeper: why is A_new effectively not contributing relative changes?
        # A_new = f(A). A varies. So A_new should vary.
        # Z_out = Z + A_new.
        # diff_out = Z(p1) + A_new(p1) - (Z(p2) + A_new(p2))
        #          = (Z(p1) - Z(p2)) + (A_new(p1) - A_new(p2))
        # diff_in = Z(p1) - Z(p2)
        # So change = | A_new(p1) - A_new(p2) |
        # If change is 0, implying A_new(p1) == A_new(p2).
        # But A(p1) != A(p2).
        # So f(A) must be collapsing spatial info? Unlikely for MLP.
        pass
    else:
        print("Block works as expected.")

if __name__ == "__main__":
    debug_local_agg()
