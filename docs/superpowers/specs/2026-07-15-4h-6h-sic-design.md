# 4H- and 6H-SiC basal crystal presets

**Date:** 2026-07-15

## Goal

Register the two common hexagonal silicon-carbide polytypes as bulk,
single-crystal scan presets:

- 4H-SiC on the basal `(0004)` cut;
- 6H-SiC on the basal `(0006)` cut.

Both presets retain the project's existing default scan grid. This is symmetric
cut support only: the slab normal and the selected reciprocal vector are
parallel, so no reciprocal-miscut model is needed.

## Catalog data and geometry

Add one bundled CIF for each polytype and load both through the declarative
material catalog. The CIFs must represent the named hexagonal polytype,
contain a complete, symmetry-expandable Si/C basis, and state their
crystallographic provenance in comments or adjacent catalog metadata.

| Catalog key | Display label | Beam axis | Pinned reflection family |
| --- | --- | --- | --- |
| `4h_sic` | `4H-SiC (0004)` | `[0, 0, 1]` | `+-(0004)` |
| `6h_sic` | `6H-SiC (0006)` | `[0, 0, 1]` | `+-(0006)` |

Pinning the basal families prevents automatic reflection ranking from choosing
an asymmetric plane and preserves `g || n`. The existing catalog defaults
continue to supply slab thickness, Debye-Waller factor, mosaic model, incident
energies, and tilt/azimuth grids.

## Scope and validation

Silicon and carbon already have atomic and transport support, so this work adds
no transport constants or physics algorithms. Existing CIF parsing, catalog
validation, composition normalization, and sweep construction remain the sole
implementation path.

Refresh the serialized catalog golden fixture and add focused assertions for:

- catalog membership, expected lattice geometry, and Si:C stoichiometry;
- finite, nonzero structure factors and optical quantities for the pinned
  `(0004)` and `(0006)` reflections;
- the `[0, 0, 1]` beam axes and exact pinned reflection lists reaching sweep
  cases.

Add validation-ledger rows that identify the entries as crystallographic and
catalog validation only. They must not claim experimental PXR/CBS agreement;
only a human may later mark an applicable claim signed off.

## Non-goals

This work does not add cubic 3C-SiC, non-basal cuts, new scan-grid controls,
transport-model changes, or experimental-performance claims.
