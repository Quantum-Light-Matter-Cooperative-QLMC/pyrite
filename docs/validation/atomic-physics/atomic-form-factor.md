# Validation: `atomic-form-factor`

## Claim and sources

- Claim: `materials/atomic.py::atomic_form_factor` returns `F(g,E) = f0(g) + f'(E) + i f''(E)`.
- Sources: Waasmaier--Kirfel elastic form factors and Chantler/FFAST anomalous terms, exposed through xraydb; source mapping recorded in `docs/physics/atomic-physics/atomic-data-sources.md`.
- Signature: `atomic_form_factor(element, g, E_eV, on_out_of_range="nan")` with `g` in inverse angstrom and energy in eV.

## Independent derivation

Away from resonances, the single-atom elastic scattering amplitude is the Fourier transform of the bound-electron density, `f0(q)`. Causality adds a complex, energy-dependent anomalous correction:

$$
F(q,E)=f_0(q)+f'(E)+i f''(E).
$$

For crystallographic momentum transfer `g = 2 pi/d = 4 pi sin(theta)/lambda`, the tabulated xraydb/Waasmaier argument is therefore

$$
q=\frac{\sin\theta}{\lambda}=\frac{g}{4\pi}.
$$

The Chantler functions supply `f'` directly, not `Z+f'`, and positive `f''`. Thus neither an additional atomic number nor a sign flip belongs in the sum. The `+i f''` convention is consistent with the repository's structure-factor phase and, after the negative susceptibility prefactor, its passive-medium optical convention.

## Cheap filters

- Units: `f0`, `f'`, and `f''` are electron scattering amplitudes and are dimensionless in the code's convention; their sum is dimensionless.
- Limits: as `g -> 0`, `f0 -> Z`, so far from an edge `F -> Z+f'+i f''`; increasing `g` suppresses `f0` while anomalous terms remain energy-only; `f',f'' -> 0` recovers ordinary elastic scattering.
- Sign/convention: xraydb takes `q=g/(4 pi)` and returns positive `f2`; using `+i f''` avoids both common adapter errors (`q=g` and `-i f''`).

## Implementation comparison

Production delegates `f0` to `cromer_mann_f0`, which passes exactly `g/(4*pi)` to `xraydb.f0`. `henke_dispersion` returns direct `xraydb.f1_chantler` and `xraydb.f2_chantler` values. The final expression is

```text
(f0 + fp) + 1j * fpp
```

It matches the independent expression term-for-term. Direct xraydb comparisons at `(C, g=0, E=8000 eV)`, `(Mo, g=3 inverse angstrom, E=3000 eV)`, and `(Te, g=1 inverse angstrom, E=10000 eV)` agree exactly; maximum absolute difference is `0.0` in float64 evaluation. Existing tests also pin `f0(0)=Z`, monotonic decrease with `g`, positive finite `f''`, and out-of-range NaNs.

Follow-up applied after independent verification: the function docstring now contains `Validation: atomic-form-factor`.

## Verdict

`rederived`. Units, momentum-transfer conversion, anomalous-term convention, limits, and direct backend values match. Ledger status advanced from `filtered` to `rederived` after verification.
