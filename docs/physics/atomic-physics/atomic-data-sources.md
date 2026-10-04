# Atomic data sources

This page states which external dataset supplies the atomic numbers behind the form factor. For the model those numbers feed (the definition of $F(g,E)$, its conventions, its domain, and its limits) see [Atomic form factors](atomic-form-factors.md). For the separate electron-transport constants, see [Elemental transport data](elemental-transport-data.md).

## Sources

`src/pyrite/materials/atomic.py` sources all atomic scattering data from [`xraydb`](https://xraypy.github.io/XrayDB/), behind one seam:

| Datum | Source |
| ----- | ------ |
| `Z_TABLE` | `xraydb.atomic_number` (any symbol, resolved lazily) |
| `cromer_mann_f0` f0(g) | Waasmaier–Kirfel parametrization (`xraydb.f0`) |
| `henke_dispersion` f′, f″ | Chantler/FFAST (`xraydb.f1_chantler`, `xraydb.f2_chantler`) |
| `load_henke` (E, f1, f2) | Chantler on `xraydb.chantler_energies` (edge-dense grid) |
| `_EDGE_PRONE` (in crystallography) | hand-flagged element set (policy, not data) |

The public API (`cromer_mann_f0`, `henke_dispersion`, `atomic_form_factor`, `Z_TABLE`, `load_henke`) is the single access point for atomic data. Downstream consumers (`structure_factor`, `chi_g`, `U_g`, `absorption_length_ang`) depend only on its signatures, shapes, and the NaN-out-of-range contract. The names `cromer_mann_f0` and `henke_dispersion` are retained although the data are Waasmaier–Kirfel and Chantler.

Any element xraydb knows is available for f0, f′, f″, and Z without per-element data files or hand-typed coefficients.

## Why xraydb

- **Pure Python and light to deploy.** No compiled extension is needed on remote boxes.
- **Vectorized.** Energy arrays are accepted directly, matching the array-based `henke_dispersion` API. The alternative C library (xraylib) exposes scalar calls that would have to be looped over the line and bremsstrahlung energy grids.
- **eV and element symbols.** These match the project's conventions; xraylib uses keV and atomic number and flips the sign of f″.
- **Chantler anomalous terms.** Chantler/FFAST covers 1 eV–966 keV, so bremsstrahlung self-absorption above 30 keV is evaluated rather than dropped. Chantler f″ is lower than Henke/CXRO in the soft band, which gives longer absorption lengths (for example graphite $L_{\rm abs}$ at 973 eV is 1.97 µm against 1.84 µm from Henke).

On the real amplitude $f_0+f'$ in the 1–4.5 keV line band, Chantler and Henke/CXRO differ by about 0.3–3 %, largest near edges. The thin-film regime is insensitive to this at the <1 % level; bulk results change by 5–7 %, almost entirely through the absorption length.

## Adapter behavior

- `f1_chantler` returns f′ directly, not the Henke-style $Z + f'$.
- `f0(el, q)` takes $q = g/4\pi$.
- `f1_chantler` can raise at the exact table endpoint, and the bremsstrahlung grid passes $E = 0$, so `henke_dispersion` masks to the strict interior of the table and returns NaN outside it.

The choice of dataset is recorded in [ADR-0015](../../adr/0015-xraydb-atomic-scattering-data.md).
