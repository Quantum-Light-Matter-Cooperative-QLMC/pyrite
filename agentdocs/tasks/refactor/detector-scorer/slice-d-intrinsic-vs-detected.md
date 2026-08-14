# Slice D prerequisite — where the intrinsic/detected boundary actually sits

The task doc lists this as an open question and says slice D must first
establish it as a **fact**, not decide it. Established below by reading every
site. Line numbers against `main` @ `7094130`.

## The task doc's premise needs one correction

The doc says "`store_result` applies it today, which means the stored record is
already detector-convolved in some paths and not others."

That is not what the code does. **`store_result` never convolves anything.**
Read `src/pyrite/results/store.py:133-168`: the record stores `out["spec"]`,
`out["brem"]`, `out["brem_wide"]` exactly as the transport produced them. What
`store_result` adds is two *scalars* derived from detector geometry:

| Stored key | Kind | Detector-derived? |
| --- | --- | --- |
| `E_grid`, `E_grid_brem` | array | no — binning |
| `spec`, `brem`, `brem_wide`, `spec_coherent` | array | **no — intrinsic, as emitted** |
| `fwhm` | scalar | **yes** — `line_fwhm_eV:116-129`, EDS + aperture (+ capped mosaic) in quadrature |
| `scale` | scalar | **yes** — `case["domega_sr"] * PER_NA` |
| `E_pk`, `eta`, `hit_frac`, `source_current_na` | scalar | no |

So the boundary is clean, and it is drawn in a better place than the doc
assumes: **every stored array is intrinsic; the detector enters the record only
as two scalars that parameterize a transform applied later.** Nothing stored is
detector-convolved, in any path.

This is stated deliberately in the design already — `profiles.py:45-48` excludes
`domega_sr` from the per-case content key precisely because "`store_result`
folds it into the derived `scale` scalar; the spec/brem arrays are untouched",
which is what makes a cached blob reusable at any solid angle.

## Where the transform is actually applied

At **read time**, in two independent families that never meet:

**Family 1 — the legacy analytic EDS/SDD window.** Gated by
`Settings.apply_detector_qe` (default `False`) and `Settings.convolve_with_det`
(default `False`); applies `detector_efficiency(E)` then
`convolve_detector(E, y, r["fwhm"])`. Four sites, all the same two lines:

| Site | Note |
| --- | --- |
| `results/store.py:171-187` `detected_background` | brem on the line grid |
| `results/store.py:190-215` `_detected_background_wide` | brem on the wide grid |
| `plots/_common.py:116-127` `_line_brem` | the line half + delegates brem to the above |
| `plots/mpl/spectra.py:156,252-264`, `plots/mpl/interactive.py:232` | open-coded repeats for the wide-grid brem |

`Settings.apply_detector_qe`'s own comment (`store.py:80-85`) says this window is
off by default *because* the real detectors carry their own QE, so the
"intrinsic" spectra are genuinely what leaves the sample.

**Family 2 — the real forward models.** `TimepixResponse` / `EagleResponse`,
applied only in the plotting layer and the anchor-figures app, and ignoring the
Family-1 settings entirely:

| Site | Shape |
| --- | --- |
| `plots/mpl/detectors.py:95-102` `_tpx_detected` | `incident = (spec + brem) * scale`; `tpx.get_response(E_grid, ...).apply(incident)` |
| `plots/mpl/detectors.py:321-324` `_eag_detected` | same shape via `eag.get_response` |
| `plots/altair/detectors.py:157-164,406` | same |
| `apps/anchor_figures.py:581,597,643,1291` | Family 1, open-coded |

Both families take the identical input — an intrinsic density on `E_grid`, times
`scale` — and return a detected density. That congruence is what makes one code
path possible.

## What this means for slice D

The unification is a **read-time** one. `store_result` needs no change at all,
which is exactly why "stored record layout for a one-detector run unchanged"
and "reproduces its stored spectrum bit for bit" are cheap to hold: the slice
does not touch the write path.

Design that follows from the fact:

- `Detector.score(E, intrinsic_density, *, fwhm, scale)` is the single entry
  point, returning a detected density.
- `response=None` → the intrinsic spectrum, i.e. identity. This is what the
  current default `Settings` already produces, since both legacy flags default
  to `False`.
- The legacy EDS window becomes a `Response` implementation wrapping
  `detector_efficiency` + `convolve_detector` at `r["fwhm"]`, so
  `Settings.apply_detector_qe` / `convolve_with_det` keep working through the
  new path for their support window rather than being reimplemented.
- `Timepix3` / `EagleXO` wrap `tpx.get_response` / `eag.get_response`
  unchanged — **no response physics moves**, per the hard non-goal.
- The four Family-1 sites and the four Family-2 sites all delegate. The
  open-coded repeats in `mpl/spectra.py`, `mpl/interactive.py` and
  `anchor_figures.py` are the ones that most need it; they are where the two
  families have drifted.

Bit-for-bit risk is confined to floating-point *ordering*: `_line_brem` computes
`spec * qe` then convolves, while `detected_background` computes `brem * qe`,
convolves, then multiplies by `scale`. The unified path must preserve each
site's existing operation order rather than normalizing it, or a stored
spectrum will differ in the last ulp.
