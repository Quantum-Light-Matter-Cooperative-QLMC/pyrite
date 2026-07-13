---
name: physics-review
description: Use when reviewing modified or new physics equations, scattering models, transport algorithms, derivation docstrings, validation markers, or ledger entries in cxr-mc.
---

# Physics Review

## Checklist

- Verify units.
- Check coordinate systems and handedness.
- Verify normalization.
- Check limiting cases: zero thickness, low/high energy, and vanishing scattering.
- Compare against analytic results or published figures if available.
- Add a `Validation: <id>` marker and a row in `docs/physics-validation-ledger.md`.
