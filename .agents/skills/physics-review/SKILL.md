---
name: physics-review
description: Review physics changes in cxr_mc. Use when modifying or adding physics equations, scattering models, transport algorithms, derivation docstrings, validation markers, or physics-validation ledger entries.
---

# Physics Review

## Checklist

- Verify units.
- Check coordinate systems and handedness.
- Verify normalization.
- Check limiting cases: zero thickness, low/high energy, and vanishing scattering.
- Compare against analytic results or published figures if available.
- Add a `Validation: <id>` marker and a row in `docs/physics-validation-ledger.md`.
