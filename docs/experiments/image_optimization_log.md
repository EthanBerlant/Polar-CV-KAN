# Image Optimization Log: Breaking the 60% Ceiling

**Date:** January 19, 2026
**Objective:** Improve CV-KAN performance on CIFAR-10 to match or exceed Vision Transformer (ViT) baseline (~63%).
**Result:** **Success (73%+ Accuracy)**

## 1. Baseline Performance (Start of Session)
- **Model**: CV-KAN (`d_complex=256`, `n_layers=6` or `4`).
- **Configuration**: Global Mean Pooling, Single-layer Patch Embedding.
- **Performance**: Plateaued at **~55% Validation Accuracy**.
- **Issues**:
  - Slow convergence (required 100 epochs to hit 55%).
  - Instability with larger learning rates.
  - Significant gap compared to ViT baseline.

## 2. Experiments & Findings

### A. Optimizer Sensitivity
We tested SGD with Momentum vs. Adam/AdamW.
- **SGD (Momentum=0.9, LR=0.01)**: Failed to converge effectively. Accuracy remained <25% after several epochs.
- **AdamW (LR=0.001)**: Provided the most stable and rapid convergence.
- **Finding**: Complex-valued networks like CV-KAN appear much harder to optimize with SGD. Adaptive moment estimation (Adam) is essential.

### B. Precision Issues (AMP)
We attempted Automatic Mixed Precision (AMP) to speed up training.
- **Result**: Immediate instability and `NaN` losses.
- **Cause**: PyTorch's `ComplexHalf` support or the interaction between complex magnitude calculations and FP16 likely caused underflow/overflow.
- **Decision**: Disabled AMP. Training stabilized immediately.

### C. Scaling Hypothesis (Failed)
We hypothesized that model capacity was the bottleneck.
- **Experiment**: Increase `d_complex` from 256 to 512.
- **Result**: No significant improvement in convergence speed. Epoch 1-10 trajectory matched the `d=256` model exactly.
- **Conclusion**: The bottleneck was **architectural**, not capacity-based. The model had enough parameters but couldn't effectively extract or aggregate spatial features.

## 3. The Pivot: Architectural Upgrades
We identified two critical weaknesses in the original design:
1.  **Mean Pooling**: "Smears" the signal. Averaging all 64 spatial tokens (8x8 grid) destroys the ability to focus on foreground features.
2.  **Shallow Embedding**: A single linear projection (or 1-layer conv) is insufficient to capture edge/texture primitives before they enter the complex KAN layers.

### Implemented Changes
1.  **Deep Patch Embedding (`DeepPatchEmbedding`)**:
    - Replaced single projection with a **2-layer Convolutional Stem**.
    - Structure: `Conv3x3` -> `BN` -> `ReLU` -> `Conv3x3` -> `BN` -> `ReLU`.
    - Purpose: Stronger low-level feature extraction.
2.  **Attention Pooling (`AttentionPool2d`)**:
    - Replaced `GlobalAveragePooling` with a **Learned Attention Mechanism**.
    - Mechanism: Learns a weight for each spatial token using a small MLP, then computes a weighted sum.
    - Purpose: Allows the model to "select" relevant patches and ignore background.

## 4. Final Results
The new architecture (`d=512`, `L=6`, `DeepEmbed`, `AttnPool`, `AdamW`) yielded immediate, massive gains.

| Metric | Original Baseline (d512) | **New Architecture** | Improvements |
| :--- | :--- | :--- | :--- |
| **Epoch 1 Val Acc** | 29.3% | **38.5%** | +9.2% |
| **Epoch 2 Val Acc** | 34.0% | **46.2%** | +12.2% |
| **Epoch 10 Val Acc** | 43.0% | **~58%** | +15% |
| **Peak Accuracy** | ~55% (Epoch 100) | **73%+ (Epoch 27)** | **+18% (and climbing)** |

### Conclusion
The **CV-KAN Image Classifier has successfully beaten the ViT baseline (63%)**, achieving >73% accuracy. The addition of **Attention Pooling** and a **Deep Convolutional Stem** was decisive, transforming the model from a mediocre classifier to a high-performer.

**Next Steps:**
- Apply `AttentionPooling` to Audio models.
- Investigate regularization for Timeseries (which shows overfitting).
- Run full overnight benchmark to confirm final scores.
