---
description: How to log qualitative research findings and lessons learned
---

# Logging Research Findings

## Purpose
Document qualitative insights, unexpected behaviors, and lessons learned during experimentation in a structured way for future reference.

## Findings Log Location
All findings go in: `docs/research_log.md`

## Steps

1. **Open or create the research log**:
   ```
   docs/research_log.md
   ```

2. **Add a new entry** with this format:
   ```markdown
   ## [YYYY-MM-DD] [Short Title]

   **Category**: [Training | Architecture | Data | Tooling | Performance]
   **Status**: [Open | Resolved | Workaround]
   **Related**: [file paths, experiment IDs, or MLflow run IDs]

   ### Finding
   [What you observed]

   ### Impact
   [How this affects the project]

   ### Action Items
   - [ ] [Specific next step]
   - [ ] [Another step if needed]
   ```

3. **Example entry**:
   ```markdown
   ## 2026-01-20 AMP Breaks Complex Number Operations

   **Category**: Training
   **Status**: Workaround
   **Related**: `experiments/train.py`, MLflow run `abc123`

   ### Finding
   Automatic Mixed Precision (AMP) causes NaN gradients when used with
   complex-valued tensors. The `torch.amp.autocast` context manager
   casts intermediate results to float16, which loses precision in
   complex magnitude calculations.

   ### Impact
   Cannot use AMP for training speedup. Training is ~2x slower than
   comparable real-valued models.

   ### Action Items
   - [ ] Investigate selective AMP (only for real-valued layers)
   - [ ] Test with bfloat16 instead of float16
   - [x] Disable AMP as workaround (--amp flag removed from defaults)
   ```

4. **Commit the update**:
   // turbo
   ```powershell
   git add docs/research_log.md
   git commit -m "docs: log finding - [short title]"
   ```

## Categories

| Category | Use For |
|----------|---------|
| **Training** | Optimizer, loss, convergence issues |
| **Architecture** | Module behavior, design insights |
| **Data** | Dataset quirks, preprocessing |
| **Tooling** | MLflow, pytest, dependencies |
| **Performance** | Speed, memory, hardware issues |

## When to Log

- Unexpected behavior that took time to debug
- Design decisions with non-obvious rationale
- Hardware/environment-specific issues
- Performance cliffs or optimization insights
- Things you wish you'd known earlier
