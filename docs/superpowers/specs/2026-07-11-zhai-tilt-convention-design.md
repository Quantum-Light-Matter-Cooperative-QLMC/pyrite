# Design: adopt Zhai's tilt-angle convention canonically

**Date:** 2026-07-11
**Status:** approved (design); implementation pending
**Author:** Alex Amador (with Claude)

## Problem

Zhai et al. report sample-tilt angles in a convention cxr-mc does not currently
present:

- **Polar (θ):** Zhai reports **positive** when the reciprocal (inverse-lattice)
  vector is tilted **toward the detector**. cxr's grids report **negative**.
- **Azimuthal (φ):** Zhai's φ is a roll about the electron-beam `+z` axis,
  positive = CCW, reported in **[0°, 180°]** (values above 90° occur). cxr's
  grids scan **negative** φ (`−80…0`).

The goal is to make cxr speak Zhai's convention everywhere so results compare
directly against the paper.

## Key finding: this is a grid + validation change, not a math change

cxr's lab-frame sample normal is
`normal = [sinθ·cosφ, sinθ·sinφ, cosθ]` (`tilted_geometry`,
`src/cxr_mc/montecarlo/geometry.py`). This is **already** Zhai's spherical
`(θ, φ)` parametrization:

- **Polar sign already matches.** `tilt_deg` flows into `tilted_geometry` with
  **no negation** at any call site (`detector.py:99`, `runner.py:244,326`). At
  `θ=+10°, φ=0`, the reciprocal vector `g ∥ normal = [0.174, 0, 0.985]` has dot
  `−0.326` with the detector direction `[sin119°, 0, cos119°]`, vs `−0.630` at
  `θ=−10°` — i.e. **positive `tilt_deg` already means "reciprocal vector toward
  detector," exactly Zhai's positive θ.** The convention was never opposite; the
  grids merely populated the negative (away-facing) half of the range.
- **Azimuth already matches.** As `φ` increases from 0, the normal's azimuth
  sweeps `+x → +y`, a CCW roll about the beam `+z` — Zhai's positive-φ sense.
  Confirmed by the user: **φ=0 places the tilt in the scattering `x–z` plane.**

There is therefore **no converter to build and no rotation-math to edit**. The
overhaul is: flip the grids to Zhai's positive ranges, fix the docstrings/labels,
add one forward-looking hook, and regenerate + re-validate the affected outputs.

## Physics consequence (verified)

Flipping the polar grids from negative (away) to positive (toward) is a **real
correction**, not cosmetic. Empirical check (WSe₂, 55 nm, θ_obs=119°, same
transport seed, `+10°` vs `−10°`):

- **Line energy: unchanged** (peak at 981.5 eV both ways — the line-energy
  denominator `1 − v0·n̂` is even in θ at φ=0).
- **Intensity: differs 2×** (peak ratio 0.505; integrated flux ~40%).

So cxr's existing negative-tilt outputs simulated the **mirror** (reciprocal away
from detector) at ~half the peak intensity. Adopting Zhai's convention changes
simulated **intensities** (not line energies), so intensity/enhancement
validations must be redone; line-energy validations are unaffected. This resolves
the open azimuth question in `docs/validation/zhai-supplementary.md`.

## Scope: canonical, repo-wide

One convention everywhere. No compatibility shim (repo is solo, non-public).

### 1. Grids and defaults → Zhai's positive ranges

Rule for polar: replace `np.linspace(-a, 0, n, endpoint=...)` with
`np.linspace(0, a, n, endpoint=...)` (swap bounds, drop the sign). Rule for
azimuth: replace the `−…0` span with `np.linspace(0, 180, n, endpoint=True)`
(same point count, full Zhai range; note the azimuthal step roughly doubles —
acceptable, flagged).

- `src/cxr_mc/materials.py`: every `tilt_deg` grid (`−85…0`, and hbn `−89…−80`
  → `80…89`) and every `tilt_azim_deg` grid (`−80…0`, hopg `−85…0`) → positive /
  `0…180`. Fix the stale comment (`materials.py:~113`,
  "negative = entrance-toward-detector") to state the Zhai convention.
- `src/cxr_mc/config.py` `default_sweep` (`~line 119`):
  `np.linspace(-span, 0.0, count)` → `np.linspace(0.0, span, count)`.
- `src/cxr_mc/sweep.py`: dataclass defaults `tilt_deg = -30.0` → `30.0`; update
  the docstring examples (`~lines 12–13, 201–202`) to positive.
- `src/cxr_mc/scan.py` demo sweep (`~lines 77–78`): `tilt_deg` and
  `tilt_azim_deg` → positive.
- `checks/anchor_figures.py`: `polar_tilts_deg = (−10, −15, −17.5, −20)` →
  `(10, 15, 17.5, 20)`. (The transport-seed formula `(tilt_deg + 30)*10` stays
  valid and distinct for positive tilts.)

Reporting surfaces (`results/scoring.py` `polar`/`azim`, `sweep.py` summary
table, plot/selection labels) round the **stored** value, so they show Zhai's
positive numbers automatically once the grids are positive — no per-surface sign
logic to add. Audit them only to confirm no hidden `abs()`/negation.

### 2. Documentation of the convention

Rewrite the `tilted_geometry` and `detector_directions` docstrings + the
geometry.py header comment to state explicitly: *positive θ = reciprocal vector
(∥ slab normal) tilted toward the detector; positive φ = CCW roll about the beam
`+z`, range `[0°, 180°]`, with φ=0 placing the tilt in the scattering `x–z`
plane; matches Zhai SI.* Add `docs/tilt-convention.md` capturing this and the
`n`/`g` split hook's intended future meaning.

### 3. Optional `n`/`g` split hook (implemented, but unused by any grid)

Add an optional parameter (default `None` ≡ `g ∥ normal`, today's behavior
**bit-for-bit**) that lets the reciprocal vector `g` tilt independently of the
physical slab normal `n` (a crystal miscut). Threaded through the case dict to
the reciprocal-vector orientation path (`_orientation_R` / `mc_spectrum`); a
strict no-op when `None`. Implement it as functional (an extra rotation applied
only to reciprocal vectors, not to `n̂`/`beam_dir`), with a unit test proving
(a) `None` reproduces current numerics exactly and (b) a nonzero value rotates
`g` while leaving the slab normal fixed. **Do not wire it into any grid or study
yet** — it exists as a plumbed, tested, but unused parameter.

### 4. Tests

Audit `tests/` for assertions that hardcode negative tilt values or the old
convention; update them to the positive convention. Add the `n`/`g`-split test
(§3) and a small test asserting `tilted_geometry`'s documented convention
(positive θ → reciprocal toward detector; φ=0 tilt in x–z plane).

### 5. Regeneration + cache clearing (post-code)

Old outputs encode the mirror configuration and are stale.

- **Local:** clear `checkpoints/zhai_reproduction/*.pkl` (Zhai reproduction
  cache — keyed on `model_coherent_spectra` source, so it orphans anyway).
  Regenerate material checkpoints via `cxr scan` / `cxr remote`.
- **Remote (`qlmc`):** clear the box's `checkpoints/zhai_reproduction/` cache and
  any cached supplementary pkls (via `cxr remote` if available, else `ssh qlmc
  'rm -f <remote_repo>/checkpoints/zhai_reproduction/*.pkl'`), then regenerate.
  If ssh is unreachable from the dev box, surface the exact command for the user.
- **Validation:** re-run the validation app / `cxr check --export`; update
  `docs/validation/zhai-supplementary.md` (resolve the open azimuth question,
  update the tilt table to positive, note line-energy invariance +
  intensity correction) and the affected rows in
  `docs/physics-validation-ledger.md` (intensity claims → need re-verification;
  physics changed, so no row is `signed-off` without a fresh-context check).

## Non-goals

- No backward-compat shim / dual convention.
- No change to the rotation math in `tilted_geometry` (formula already correct).
- No wiring of the `n`/`g` split into production grids.

## Risks

- **Stale outputs mistaken for valid.** Mitigation: clear caches (local +
  remote) and regenerate before any comparison; treat intensity ledger rows as
  unverified until re-checked in fresh context.
- **φ reference offset.** Assumed φ=0 ≡ tilt in `x–z` plane — **confirmed by
  user**, but add the convention test (§4) so it can't silently drift.
- **Asymmetric reflections.** θ-via-normal is exact only for `g ∥ normal`; Zhai's
  series (and cxr's (00l) TMD/HOPG reflections) satisfy this. The `n`/`g` hook
  (§3) is the future escape hatch if asymmetric `g` is ever needed.
