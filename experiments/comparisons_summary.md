# Summary of Comparison Experiments

This document summarizes the methodology and findings from a series of one-off comparison scripts used to evaluate the Polar-CV-KAN architecture. These scripts have been removed from the codebase to reduce clutter.

## 1. Capacity vs. Structure (`compare_capacity.py`)
**Goal**: Test whether vertical weight sharing (structure) can match the performance of independent weights (capacity) or wider shared models.
**Methodology**:
- Compared three Vertical configurations on SST2, FashionMNIST, and CIFAR10:
    1.  **Independent (Base)**: No vertical sharing.
    2.  **Shared (Compressed)**: Vertical sharing enabled (significantly fewer parameters).
    3.  **Shared (Wide/Iso-Param)**: Vertical sharing with increased width (`d_complex`) to match the parameter count of the Independent model.
**Key Insight**: Determine if "deep" parametrizing is necessary or if a repeated recursive structure (sharing) is sufficient when scaled.

## 2. Hierarchical vs. Flat (`compare_hierarchical.py`)
**Goal**: Assess if the recursive "Polarizing" structure is superior to a flat stack of blocks.
**Methodology**:
- Task: Synthetic Signal/Noise classification.
- Models:
    - **Flat**: Standard stack of `PolarizingBlock`s.
    - **Hierarchical**: `HierarchicalPolarization` block with modifications:
        - Weight Sharing: Shared vs. Per-Level.
        - Aggregation: Mean vs. Magnitude-Weighted.
        - Top-Down: Optional feedback paths.

## 3. Hybrid Weight Sharing (`compare_hybrid.py`)
**Goal**: Find a middle ground between "Independent" (flexible but heavy) and "Shared" (efficient but rigid).
**Methodology**:
- **Strategy**: Use shared weights for deep/high-frequency levels (where patterns might be self-similar) and independent weights for early/low-frequency levels.
- **Implementation**: `hierarchical_sharing="hybrid"` with a split index (e.g., share levels > 2).

## 4. Phase Shifting (`compare_phase.py`)
**Goal**: Evaluate if learnable phase shifts in the complex domain improve performance.
**Methodology**:
- Toggled `hierarchical_phase_shifting=True/False` in the Hybrid model.
- Tested on FashionMNIST.
- **Hypothesis**: Phase information in complex numbers can capture shift-invariant or relational features better than magnitude alone.

## 5. Pointwise vs. Broadcast Interaction (`compare_pointwise.py`)
**Goal**: Determine the best way to combine signals in the KAN (Kolmogorov-Arnold Network) nodes.
**Methodology**:
- **Broadcast**: Standard interaction where one path modulates the other broadly.
- **Pointwise**: Element-wise interaction, potentially offering finer control but less global context.
- Tested on FashionMNIST.

## 6. Domain-Specific Weight Sharing (`compare_sharing_{audio,image,sst2}.py`)
**Goal**: Verify if the "Structure vs. Capacity" trade-offs hold across different modalities.
**Methodology**:
- **Audio (`compare_sharing_audio.py`)**: SpeechCommands dataset. Used STFT frontend.
- **Image (`compare_sharing_image.py`)**: CIFAR-10. Used Patch Embeddings.
- **Text (`compare_sharing_sst2.py`)**: SST-2. Used Embedding + Projection.
- **Configurations Tested**:
    1.  **Independent**: No sharing (maximal params).
    2.  **Fully Shared**: Horizontal + Vertical sharing (minimal params).
    3.  **Horizontal Only**: Sharing across parallel paths but unique weights per depth level (ensemble-like).

---
*Note: These experiments were exploratory. Their successful concepts have been integrated into the main `MultiPathHierarchicalPolarization` module and the canonical training scripts.*
