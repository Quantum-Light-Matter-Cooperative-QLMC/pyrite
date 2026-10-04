# 0015 — xraydb as the atomic scattering data source

- **Status:** Accepted
- **Date:** 2026-06-23

## Context

`materials/atomic.py` carried hand-typed Cromer–Mann coefficients, a hand-typed `Z_TABLE`, and per-element Henke/CXRO `.nff` CSVs. Adding an element touched about eight non-colocated registries, roughly four of them atomic data.

Candidates compared on Python 3.14:

| Library | f0(q) | f1/f2 | Outcome |
| --- | --- | --- | --- |
| `xraydb` 4.5.8 | Waasmaier–Kirfel | Chantler (FFAST) | Adopted: pure Python, vectorized, eV and symbols |
| `xraylib` 4.2.1 | Waasmaier | Cromer–Liberman | Rejected: compiled C+SWIG, scalar calls, keV and Z, older anomalous terms |
| `periodictable` | Cromer–Mann | Henke/CXRO | Not chosen: Henke-preserving hybrid |
| `scikit-beam` | wraps xraylib | wraps xraylib | Build fails on 3.14 |
| XATOM | n/a | n/a | Not pip-installable |

Agreement with the previous Henke baseline on $f_0+f'$ in the 1–4.5 keV band was comparable for xraydb and xraylib (about 0.3–3 %, largest near edges), so closeness to Henke did not discriminate; engineering fit did.

## Decision

`xraydb` supplies f0, f′, f″, and Z. The public API (`cromer_mann_f0`, `henke_dispersion`, `atomic_form_factor`, `Z_TABLE`, `load_henke`) keeps its signatures, shapes, and NaN-out-of-range contract, so downstream code is unchanged. The unused CXRO `.nff` files were removed under ADR-0014.

## Consequences

Numbers shift by a few percent because the anomalous terms are Chantler rather than Henke. Validation anchors re-run on CPU before and after:

| Anchor | Before | After | Shift |
| --- | --- | --- | --- |
| Feranchuk LiF (200) $\lvert\chi_g\rvert$ | 4.311e-05 | 4.306e-05 | −0.1 % |
| Feranchuk LiF model flux | 2.812e+04 ph/s | 2.817e+04 ph/s | +0.2 % |
| Feranchuk LiF $\lvert A_{\rm PXR}/A_{\rm CBS}\rvert$ | 0.6088 | 0.6068 | −0.3 % |
| Zhai graphite 29 nm A/B line ratio | 1.00 | 1.00 | unchanged |
| Zhai graphite 1 mm A/B line ratio | 2.54 | 2.57 | +1.2 % |
| graphite $L_{\rm abs}$ at 973 eV | 1.84 µm | 1.97 µm | +7 % |

- $f_0$ is interchangeable to ≤0.25 %; the thin-film regime is unchanged to <1 %.
- Bulk results move 5–7 %, entirely through the absorption length (Chantler f″ < Henke f″ at soft energies).
- Chantler covers 1 eV–966 keV against Henke's 10 eV–30 keV, so bremsstrahlung self-absorption above 30 keV is evaluated instead of dropped.
- New elements need no data files.
