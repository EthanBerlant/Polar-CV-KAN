---
description: How to validate code quality and architecture compliance
---

# Code Quality Validation

## Quick Quality Check

```powershell
# Run linting with auto-fix
// turbo
ruff check --fix .

# Format code
// turbo
ruff format .

# Run full pre-commit checks
// turbo
pre-commit run --all-files
```

## Architecture Validation

### Check for Duplicate Training Loops

All training should flow through one of these entry points:
- `experiments/train.py` - CV-KAN training
- `experiments/baselines/base_trainer.py` - Baseline training

```powershell
# Find potential duplicate training loops (should only be in sanctioned files)
// turbo
rg -l "def train_epoch" --glob "*.py" | Select-String -NotMatch "base_trainer|trainer"
```

### Check for Orphaned Scripts

All experiment scripts should use the domain adapter pattern:

```powershell
# Find scripts that directly create dataloaders (should use domains/)
// turbo
rg -l "create_.*_dataloader" experiments/*.py --glob "!train.py"
```

### Verify Import Structure

```powershell
# Check for proper sys.path usage (should be consistent)
// turbo
rg "sys.path.insert" --glob "*.py" -c
```

## Pre-commit Setup

```powershell
# Install pre-commit hooks (one-time)
// turbo
pip install pre-commit
pre-commit install

# Update hook versions
// turbo
pre-commit autoupdate
```

## Code Structure Rules

### Allowed Structure:
```
experiments/
├── train.py                    # SINGLE CV-KAN training entrypoint
├── run_benchmark.py            # SINGLE benchmark orchestrator
├── domains/                    # Domain-specific factories
│   └── *.py                    # Must export: create_model, create_dataloaders
├── baselines/
│   ├── base_trainer.py         # SINGLE baseline training infrastructure
│   └── *_baseline.py           # Model definitions only, use base_trainer
└── analysis/
    └── *.py                    # Post-hoc analysis scripts
```

### Forbidden Patterns:
- ❌ Multiple benchmark runners (only `run_benchmark.py`)
- ❌ Training loops in baseline files (use `base_trainer.py`)
- ❌ Shell scripts wrapping Python scripts
- ❌ Duplicate documentation (one source of truth per topic)
- ❌ Standalone training scripts outside `train.py` pattern

## Lint Error Categories

| Severity | Action | Examples |
|----------|--------|----------|
| Error | Must fix before commit | Syntax errors, undefined names |
| Warning | Should fix | Unused imports, complexity > 15 |
| Info | Nice to fix | Missing docstrings, style |
