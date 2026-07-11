# Validation status: Zhai supplementary coherent-emission studies

**Scope.** `checks/anchor_figures.py::ZHAI_SUPPLEMENTARY_STUDIES` — the
WSe₂/MoSe₂/h-BN coherent-only reproductions (`figure_supplementary_tmd`,
`figure_supplementary_hbn`), driven by `model_coherent_spectra` and rendered
by the validation app's "Zhai supplementary" section and `cxr check --export`.

This is a **provenance and open-question record, not a physics
re-derivation write-up** (contrast `docs/validation/hbn-structure.md`) — none
of these study *inputs* carry a ledger `id` of their own; they parameterize
existing ledgered claims (`line-energy-dispersion`, `coherent-line-spectrum`,
`electron-transport`). Promoting anything here past this record requires a
fresh-context re-derivation per `docs/validation/README.md`, and only a human
signs off.

## What's encoded, and where it came from

| material | thicknesses (nm) | energy window (eV) | polar tilts (deg) | beam energy |
|----------|-------------------|---------------------|--------------------|-------------|
| WSe₂ | 42, 55, 75 | 800–1200 | +10, +15, +17.5, +20 | 200 keV |
| MoSe₂ | 47, 112, 147 | 800–1200 | +10, +15, +17.5, +20 | 200 keV |
| h-BN | 921 | 600–1200 | +10, +15, +17.5, +20 | 200 keV |

(Tilts flipped from negative to positive 2026-07-11 — see
[Open question: azimuthal angle](#open-question-azimuthal-angle-resolved)
below and `docs/tilt-convention.md`. The table above reflects the corrected,
canonical-convention grid; the transcribed `TODO.md` note below predates the
flip and is left as-is for provenance.)

These match the values transcribed into `TODO.md`'s original user-added note
("Zhai supplementary coherent-emission figures ... all 200 keV, at polar
tilts −10/−15/−17.5/−20 deg") — the code and that note agree, which is as far
as a provenance check without the source SI in-repo can go. Confirming these
numbers directly against the published SI figures/tables is unclaimed here.

## Open question: azimuthal angle (resolved)

`model_coherent_spectra` (`checks/anchor_figures.py`) builds each tilt's
geometry via:

```python
beam_dir, n_hat = tilted_geometry(study.theta_obs_rad, float(np.deg2rad(tilt_deg)))
```

`tilted_geometry` (`src/cxr_mc/montecarlo/geometry.py`) takes only a polar
tilt — there is no azimuthal parameter in this call at all, so every
supplementary panel is implicitly computed at **azimuth = 0**, i.e. `φ = 0`
in the canonical convention (`docs/tilt-convention.md`): the tilt lies in the
scattering `x–z` plane. This part of the original open question is settled —
azimuth = 0 is correct, it is the SI's own reference plane, not an
unconfirmed guess.

What *was* wrong was the sign of `tilt_deg` itself. `tilted_geometry`'s
`normal = [sinθ·cosφ, sinθ·sinφ, cosθ]` already matches Zhai's positive-θ =
"reciprocal vector toward detector" convention (`docs/tilt-convention.md`),
but the series above were generated with **negative** `tilt_deg`
(`checks/anchor_figures.py`'s old `polar_tilts_deg = (−10, −15, −17.5, −20)`)
— the mirror configuration, reciprocal vector tilted **away** from the
detector. Confirmed empirically (WSe₂, 55 nm, `θ_obs = 119°`, same transport
seed): flipping `+10°` vs `−10°` leaves the line energy **unchanged** (981.5
eV both ways — the line-energy denominator `1 − v0·n̂` is even in θ at
`φ = 0`), but changes peak **intensity** by 2× (ratio 0.505) and integrated
flux by ~40%.

**Resolution:** the grids in the table above are now generated at positive
`tilt_deg`. The underlying lineshape physics was never wrong (as anticipated
in the original open-question note), and the reported *line energies* in any
prior figure/comparison are unaffected by the flip. Reported *intensities*
(peak heights, integrated flux, any bulk-vs-film enhancement ratio computed
from these studies) were computed at the mirror configuration and need
re-generation + re-comparison against the published SI before being trusted.

## Status

Provenance: internally consistent (code ≡ transcribed note, modulo the sign
correction above). Azimuth assumption: **resolved** — `φ = 0` is the correct,
canonical reference plane. Polar sign: **corrected** 2026-07-11 (positive =
Zhai's toward-detector convention); line energies unaffected, intensities
require fresh regeneration.

**Not signed off.** This write-up documents what changed and why; it is not
an independent re-derivation. Per `docs/validation/README.md`, any
intensity/enhancement claim exercised through these studies needs a
fresh-context re-verification against the regenerated (positive-tilt)
checkpoints before advancing past its current ledger status — see
`docs/physics-validation-ledger.md`.
