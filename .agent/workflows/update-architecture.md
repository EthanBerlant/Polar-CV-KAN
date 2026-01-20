---
description: How to update architecture documentation after changes
---

# Updating Architecture

## When to Update

Update `ARCHITECTURE.md` when:
- Adding a new module or model
- Changing stability status (🧪 → 🔒 or vice versa)
- Discovering new design patterns/decisions
- Deprecating functionality

## Steps

1. **Edit ARCHITECTURE.md** with your changes:
   - Module taxonomy tables
   - Stability markers
   - Design decisions section

2. **If promoting to stable (🧪 → 🔒)**:
   - Ensure comprehensive test coverage
   - Document the API contract
   - Get validation on at least one real task

3. **If deprecating (→ ⚠️)**:
   - Note the replacement approach
   - Keep code for reference but mark deprecated
   - Update any imports in other files

4. **Verify nothing broke**:
   // turbo
   ```powershell
   pytest tests/ -v
   ```

5. **Update README.md** if the change affects user-facing features.

## Stability Transition Criteria

### 🧪 → 🔒 (Experimental → Stable)
- [ ] Used successfully in ≥2 experiments
- [ ] Has unit tests with >80% coverage
- [ ] API reviewed and finalized
- [ ] Documented in ARCHITECTURE.md

### 🔒 → ⚠️ (Stable → Deprecated)
- [ ] Replacement exists and is documented
- [ ] Migration path documented
- [ ] Kept for backward compatibility (at least 1 version)
