---
name: physics-review
description: Use when changed cxr-mc physics models, equations, derivation docs, validation markers, or ledger coverage need implementation-context review; not independent validation.
---

# Physics Review

Review changed physics in implementation context; do not claim independent
validation or mark ledger entries signed off.

- Verify units.
- Check coordinate systems and handedness.
- Verify normalization.
- Check limiting cases: zero thickness, low/high energy, and vanishing scattering.
- Compare against analytic results or published figures if available.
- Require a source equation, assumptions, limiting case, `Validation: <id>`,
  ledger row, and derivation record; report omissions to the implementation owner.
- Invoke `physics-validation` for the fresh-context source-to-code check.
