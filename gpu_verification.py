import torch
import sys
import traceback

sys.path.insert(0, '.')

from src.modules import PolarizingBlock, GatedPolarization
from src.modules.multi_head import (
    EmergentHeadsPolarizing, 
    PhaseOffsetPolarizing, 
    FactoredHeadsPolarizing
)
from src.models.cv_kan import CVKANTokenClassifier
from src.losses import diversity_loss, phase_anchor_loss

def testing_log(msg):
    print(f"[GPU TEST] {msg}")

def run_gpu_verification():
    if not torch.cuda.is_available():
        print("CUDA not available! Cannot run GPU verification.")
        sys.exit(1)
    
    device = torch.device("cuda")
    testing_log(f"Running on device: {device}")
    
    try:
        # 1. Test PolarizingBlock
        testing_log("Testing PolarizingBlock...")
        Z = torch.randn(4, 16, 32, dtype=torch.cfloat).to(device)
        block = PolarizingBlock(32).to(device)
        out = block(Z)
        assert out.is_cuda, "Output should be on GPU"
        testing_log("PolarizingBlock passed.")

        # 2. Test Multi-Head Modules
        testing_log("Testing EmergentHeadsPolarizing...")
        Z64 = torch.randn(4, 16, 64, dtype=torch.cfloat).to(device)
        emergent = EmergentHeadsPolarizing(64).to(device)
        out_e = emergent(Z64)
        assert out_e.is_cuda
        testing_log("EmergentHeadsPolarizing passed.")

        testing_log("Testing PhaseOffsetPolarizing...")
        offset = PhaseOffsetPolarizing(n_heads=8, d_per_head=8).to(device)
        out_o = offset(Z64)
        assert out_o.is_cuda
        testing_log("PhaseOffsetPolarizing passed.")

        testing_log("Testing FactoredHeadsPolarizing...")
        # Note: check init args in file if this fails
        factored = FactoredHeadsPolarizing(n_heads=8, d_model=64, d_per_head=8).to(device)
        out_f = factored(Z64)
        assert out_f.is_cuda
        testing_log("FactoredHeadsPolarizing passed.")

        # 3. Test Full Model
        testing_log("Testing CVKANTokenClassifier...")
        model = CVKANTokenClassifier(
            d_input=32, 
            d_complex=32, 
            n_layers=2, 
            n_classes=2
        ).to(device)
        outputs = model(Z)
        # Check dictionary outputs
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                assert v.is_cuda, f"Output {k} not on GPU"
        testing_log("CVKANTokenClassifier passed.")

        # 4. Test Losses
        testing_log("Testing Losses...")
        # diversity_loss and phase_anchor_loss usually take tensors
        # Assuming they operate on the tensor provided (which is on GPU)
        div = diversity_loss(Z)
        anchor = phase_anchor_loss(Z)
        assert div.is_cuda, "Diversity loss should return tensor on same device (or scalar on device)"
        assert anchor.is_cuda, "Anchor loss should return tensor on device"
        testing_log("Losses passed.")
        
        print("\n\nSUCCESS: All modules verified on GPU!")

    except Exception:
        print("\n\nFAILURE: GPU Verification Failed")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_gpu_verification()
