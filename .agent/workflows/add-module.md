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

2. **Follow the patterns**:
   - Use `torch.cfloat` dtype for complex operations
   - Include residual connections for stability
   - Document with docstrings

3. **Export from package** - add to `src/modules/__init__.py`:
   ```python
   from .your_module import YourNewBlock
   ```

4. **Update ARCHITECTURE.md**:
   - Add entry to appropriate table
   - Mark with 🧪 (experimental) initially

5. **Add tests** to `tests/test_modules.py`:
   ```python
   class TestYourNewBlock:
       def test_forward_shape(self):
           block = YourNewBlock(d_complex=32)
           Z = torch.randn(4, 16, 32, dtype=torch.cfloat)
           out = block(Z)
           assert out.shape == Z.shape
           assert out.dtype == torch.cfloat
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
