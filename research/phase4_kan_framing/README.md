# Phase 4: KAN Framing

## Goal

Assess whether the Kolmogorov-Arnold Network framing is justified or merely nominal.

## Key Questions

1. Does the K-A theorem apply to complex-valued functions?
2. What does the "KAN" paper actually prove vs. claim?
3. Should the architecture be renamed?

## Contents

### `literature_notes.md`
Analysis of the K-A theorem, the KAN paper, and how CV-KAN relates to both.

**Key finding**: The K-A theorem is proven for ℝⁿ → ℝ only. Its extension to ℂⁿ → ℂ is an open question. The CV-KAN architecture learns univariate transforms but this doesn't actually invoke K-A.

## Recommended Action

Either:
1. Add caveat: "K-A theorem applies to real functions; complex extension is open"
2. Rename: "Polar Decomposition Network" or similar
3. Prove something: Attempt a complex K-A theorem (hard, potential contribution)

## No Code Required

This phase is literature review only. The output is the analysis document.
