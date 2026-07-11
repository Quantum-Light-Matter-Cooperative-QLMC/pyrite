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
| WSe₂ | 42, 55, 75 | 800–1200 | −10, −15, −17.5, −20 | 200 keV |
| MoSe₂ | 47, 112, 147 | 800–1200 | −10, −15, −17.5, −20 | 200 keV |
| h-BN | 921 | 600–1200 | −10, −15, −17.5, −20 | 200 keV |

These match the values transcribed into `TODO.md`'s original user-added note
("Zhai supplementary coherent-emission figures ... all 200 keV, at polar
tilts −10/−15/−17.5/−20 deg") — the code and that note agree, which is as far
as a provenance check without the source SI in-repo can go. Confirming these
numbers directly against the published SI figures/tables is unclaimed here.

## Open question: azimuthal angle

`model_coherent_spectra` (`checks/anchor_figures.py`) builds each tilt's
geometry via:

```python
beam_dir, n_hat = tilted_geometry(study.theta_obs_rad, float(np.deg2rad(tilt_deg)))
```

`tilted_geometry` (`src/cxr_mc/montecarlo/geometry.py`) takes only a polar
tilt — there is no azimuthal parameter in this call at all, so every
supplementary panel is implicitly computed at **azimuth = 0** (the function's
own docstring: "azimuth 0. Tilting the sample so its normal points along...").

**Unconfirmed:** whether the Zhai SI's reported WSe₂/MoSe₂/h-BN polar-tilt
series were themselves measured/computed at azimuth = 0, or at some other
fixed azimuthal setting. If the SI's series varies azimuth (or fixes it at a
nonzero value), the current code reproduces the wrong slice of the parameter
space at every tilt in the table above — this would not show up as a
qualitative failure (the underlying lineshape physics is still correct), only
as a quantitative mismatch against the specific published panel.

**Next step (separate from this write-up):** a fresh-context reviewer with
access to the Zhai et al. SI should locate the exact figure/table describing
the polar-tilt series' azimuthal convention and confirm or refute azimuth = 0
against `docs/validation/README.md`'s re-derivation workflow. Until then this
stays an open flag, not a `discrepancy` (no check has actually failed — the
assumption is merely unverified) and not `rederived` (no independent
confirmation exists yet).

## Status

Provenance: internally consistent (code ≡ transcribed note). Azimuth
assumption: **unconfirmed**, open question above. No ledger row changes as a
result of this write-up.
