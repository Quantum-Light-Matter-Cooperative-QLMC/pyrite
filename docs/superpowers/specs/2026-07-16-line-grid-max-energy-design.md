# Empirically-Bounded Line-Grid Upper Energies

## Goal

Replace the standard profile's current `E_grid_line_by_energy` upper bounds
(`stop` per beam energy in `src/cxr_mc/data/materials.toml`) with values
derived from the actual simulated coherent-line intensity, rather than the
undocumented scaling currently in place. Each beam energy's `stop` must be
wide enough that the line grid captures at least 99% of the cumulative
coherent-line intensity radiated by any standard-profile catalog material,
at any point in the swept tilt/azimuth geometry, at that beam energy.

This is a data-only recalibration of an existing, validated kernel
(`Validation: line-energy-dispersion` in
`src/cxr_mc/montecarlo/geometry.py`). It does not change the resonance
physics, only the sampling window used to observe it, so no new derivation
docstring or ledger row is required — the existing marker already covers the
`E_res` relationship this bound is measured against.

## Background

`mc_spectrum` (`src/cxr_mc/montecarlo/spectrum.py`) computes a per-trajectory
resonance energy `E_res = HBARC_EV_ANG * (v·g) / (1 - v·n)` for the coherent
PXR+CBS line spectrum, then keeps only segments whose `E_res` falls within
the configured `E_grid_line` (plus a 20%-of-span pad). A `stop` that is too
low silently discards real radiated intensity rather than merely losing
discretization resolution. An earlier uncommitted change in this worktree
capped `stop` at 3000 eV for all higher beam energies; empirical checking
showed this discarded 25-65% of coherent-line intensity for diamond and
silicon at moderate-to-large tilt and higher beam energy. That change is
discarded; this design replaces it with a bound derived from simulation
rather than guessed.

`tilted_geometry` establishes that `v0·g` (and hence `E_res`) is maximized
near `tilt_polar ≈ 0`, since `v0·g ∝ cos(tilt_polar)` while the ray's
denominator term is comparatively tilt-invariant. The standard polar sweep is
the 10 endpoint-inclusive values of `linspace(0, 89, 10)`, so the two
smallest polar values (0° and ≈9.89°) are the primary candidates for
maximum `E_res`; larger tilts are spot-checked to confirm they don't exceed
this bound in some material.

## Approach

Add a new committed dev script, `scripts/analyze_line_grid_bounds.py`, that
empirically measures the 99%-cumulative-intensity energy per beam energy
across the standard profile's catalog materials and geometry, in two
stages:

**Coarse stage.** For each of the 7 standard beam energies, run `mc_spectrum`
for every standard-profile catalog material at the two smallest standard
polar values (0° and ≈9.89°) across all 10 standard azimuthal values, using
a wide diagnostic `E_grid_line` (e.g. 10-12000 eV, fine spacing) and a small
electron count (~500) for speed. Separately, spot-check a handful of
materials at 2-3 larger polar values to confirm the tilt≈0 prior holds; if
any spot-check exceeds the near-zero-tilt candidates, report it and fold
that geometry into the top candidates for the refine stage.

**Refine stage.** For each beam energy, take the top few
(material, tilt, azimuth) candidates ranked by 99%-cumulative-intensity
energy, and rerun those at a higher electron count (~5000) for stable
statistics. The refined 99% energy across these candidates, maximized over
the set, becomes that beam energy's raw bound.

**Margin and rounding.** Apply a fixed +15% safety margin to the raw bound,
then round up to the nearest 100 eV to produce the final `stop`. Recompute
`num` for each beam energy's grid to preserve the ~3 eV endpoint-inclusive
spacing convention already used in `materials.toml` (matching the existing
"endpoint inclusion takes priority over exact 3 eV spacing" rule from the
original per-beam grid design).

The script prints one results table (beam energy, raw 99% energy, margined
`stop`, `num`, driving material/tilt/azimuth) to stdout. This table both
documents the derivation for the ledger/TODO and becomes the literal source
for the `materials.toml` edit — no other document duplicates these numbers.

### Alternatives considered

- **Per-material bounds** instead of a shared profile-level bound: rejected
  per prior discussion — the schema stays profile-level/shared, so a single
  worst-case-driven bound is required rather than a tighter per-material fit.
- **Analytic bound from `E_res`'s closed form** instead of simulation: would
  need to also model the multiple-scattering-driven spread of `v` through
  the slab (which broadens `E_res` beyond the single-scattering estimate),
  effectively re-deriving what the Monte Carlo already computes. Rejected in
  favor of measuring the quantity that actually matters (simulated
  intensity), which is also cheap to script here.

## Data Flow

The script performs the coarse and refine stages purely in-memory using the
existing `mc_spectrum`/catalog/sweep machinery from `src/cxr_mc/`; it does
not write intermediate files. Its stdout table is the handoff artifact:

1. Run `scripts/analyze_line_grid_bounds.py`, capture the results table.
2. Manually transcribe the resulting `stop`/`num` pairs into
   `src/cxr_mc/data/materials.toml`'s `[profiles.standard].E_grid_line_by_energy`
   (only `stop` and `num` change; `start` is untouched).
3. Regenerate `tests/data/material_catalog_golden.json` and update the
   `expected_bounds` mapping (and spacing tolerance, if needed) in
   `tests/test_material_catalog.py::test_standard_profile_uses_requested_angles_energies_and_line_grids`.

No changes are needed to `sweep.py`, `catalog.py`, or `spectrum.py` — this is
purely a data recalibration consumed by existing, unmodified code paths.

## Error Handling

If the coarse stage finds that exact `tilt_polar = 0°` yields vanishing or
degenerate line intensity for some material (a possible geometric
degeneracy distinct from ordinary Monte Carlo noise), the script reports
this explicitly per material rather than silently treating it as the
dominant (zero-intensity) candidate, and falls back to the next-smallest
polar value as that material's primary candidate. This does not change
production behavior — `tilt_deg = 0` remains a valid, unrelated point in the
production angular sweep; it only affects which geometry the analysis script
treats as the driver of the bound.

If a beam energy's refined bound is driven by a spot-checked larger-tilt
geometry rather than one of the two near-zero candidates, the script flags
this loudly in its output table so it gets attention during review rather
than being silently accepted.

## Testing

- The analysis script is a dev tool, not a pytest target; it will be
  smoke-tested by running it end-to-end and confirming it produces a
  complete 7-row results table with no exceptions.
- `test_standard_profile_uses_requested_angles_energies_and_line_grids` is
  updated so its `expected_bounds` dict and spacing tolerance match the new
  `stop`/`num` values; it continues to assert exact start/stop and ~3 eV
  spacing per beam energy.
- `tests/data/material_catalog_golden.json` is regenerated so golden
  hash/shape checks reflect the new grids.
- Acceptance gate is `uv run python scripts/dev.py verify` (lint, typecheck,
  full test suite) passing cleanly, run with sandbox disabled per this
  worktree's established `uv` cache-write requirement.

## Branch TODO

Once implemented, this branch's `TODO.md` will be overwritten (per the
`todo-sync` convention) with a single scoped item summarizing: the problem
(undocumented/unsafe line-grid caps), the empirical method used, and a
pointer to `scripts/analyze_line_grid_bounds.py` and this spec as the
implementation record.
