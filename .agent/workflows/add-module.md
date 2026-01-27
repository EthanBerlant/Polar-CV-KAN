---
description: How to add a new module to the CV-KAN architecture
---

# Adding a New Module

## Module Types

| Type | Location | Naming |
|------|----------|--------|
| Layer/Block | `src/modules/` | `*Block`, `*Layer` |
| Aggregation | `src/modules/aggregation.py` | `*Aggregation` |
| Full Model | `src/models/` | `CVKAN*` |

## Steps

1. **Create the module** in appropriate location:
   - For new layer types: `src/modules/your_module.py`
   - For aggregation: Add to `src/modules/aggregation.py`
   - For model: `src/models/cv_kan_yourname.py`

2. **Implement with Registry**:
   - Import the corresponding registry from `src.registry`
   - Decorate your class with `@REGISTRY.register("name")`
   
   ```python
   from ..registry import AGGREGATION_REGISTRY
   from torch import nn
   
   @AGGREGATION_REGISTRY.register("my_new_agg")
   class MyAggregation(nn.Module):
       ...
   ```

3. **Follow the patterns**:
   - Use `torch.cfloat` dtype for complex operations
   - Include residual connections for stability
   - Document with docstrings

4. **Verify Architecture Compliance**:
   Run the linter to ensure you didn't bypass the factories:
   // turbo
   ```powershell
   python scripts/lint_architecture.py
   ```

5. **Update ARCHITECTURE.md**:
   - Add entry to appropriate table
   - Mark with 🧪 (experimental) initially

6. **Add tests** to `tests/test_modules.py` or a new test file:
   ```python
   def test_forward_shape():
       # Ideally use the Factory to create it if possible, or direct instantiation for unit tests
       pass
   ```

6. **Run tests**:
   // turbo
   ```powershell
   pytest tests/test_modules.py -v
   ```

## Stability Markers

When adding to ARCHITECTURE.md:
- 🧪 **Experimental**: New, untested, API may change
- 🔒 **Stable**: Proven, API frozen (upgrade after validation)
