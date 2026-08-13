# Validation: `xray-chi-zero`

## Claim and source

- Claim: `materials/crystal.py::chi_0` returns the g=0 (bulk, forward-scattering)
  unit-cell electric susceptibility
  `χ₀ = −rₑλ²/(πV_cell) · Σᵢ(f1ᵢ + i f2ᵢ)`, `f1 = Z + f′`, `f2 = f″`, summed
  over every atom `i` in the unit-cell basis.
- Source (per the ledger row and docstring): `chi_g` (Feranchuk–Spence 2000,
  Eq. (3)) evaluated at `g = 0`, where the Debye–Waller factor and every basis
  phase reduce to 1; forward form factors taken in the Henke/Chantler
  convention shared with `absorption-length` and `grazing-optical-constants`.
- Signature: `chi_0(crystal, photon_E_eV, use_henke=True)`.

## Independent derivation

Start from the already-verified general result for `chi_g` (`pxr-amplitude`,
independently re-derived from the driven free-electron response and matching
the implementation term-for-term):

```{math}
\chi_{\mathbf g}=-\frac{r_e\lambda^2}{\pi V_{cell}}\,S_{\mathbf g},
\qquad
S_{\mathbf g}=\sum_j f_j(\mathbf g,E)\,
  \exp(i\mathbf g\cdot\mathbf r_j)\,\exp(-W_j).
```

Set `g = 0`. Two things happen independent of any code inspection:

1. The basis phase `exp(i·0·r_j) = 1` for every site `j`.
2. The Debye–Waller exponent is `W_j = B_j s^2` with `s = |g|/(4π)`; at `g=0`,
   `s=0`, so `exp(-W_j) = 1` regardless of the atom's `B` factor.

So the structure factor collapses to the plain forward-scattering sum over
every atom in the cell,

```{math}
S(0)=\sum_j f_j(0,E),
```

and, using the standard atomic scattering-factor decomposition into a real
non-resonant part plus complex dispersion correction,
`f_j(0,E) = Z_j + f'_j(E) + i f''_j(E) \equiv f1_j + i f2_j`,

```{math}
\boxed{\chi_0=-\frac{r_e\lambda^2}{\pi V_{cell}}\sum_j\big(f1_j+i f2_j\big).}
```

This is exactly the claimed formula, obtained purely as the `g→0` limit of the
already-verified `chi_g` result — no new physical assumption beyond the ones
already checked for `pxr-amplitude`.

### Consistency with `optical_constants` normalization (not independent evidence)

Writing `n_j = N_j/V_{cell}` for the number density of atom type `j`'s sites
and grouping the sum by element, `Σⱼ f1_j = V_{cell} Σ_{type} n_{type} f1_{type}`.
Substituting into the boxed result and comparing with the
`grazing-optical-constants` definitions `δ = (r_e λ²/2π) Σ n_type f1_type`,
`β = (r_e λ²/2π) Σ n_type f2_type` gives `Re χ₀ = −2δ`, `Im χ₀ = −2β` exactly —
consistent with `n = √(1+χ₀) ≈ 1 + χ₀/2` and the module's `n = 1−δ−iβ`
convention. This reproduces the ledger's own caveat: it is a shared-prefactor
identity, not a second independent check, so it is reported only as a
consistency note, not as corroborating evidence for the verdict below.

### Thomson (`f′=f″=0`) limiting case

Setting the dispersion corrections to zero collapses the sum to the bare
atomic numbers:

```{math}
\chi_0\big|_{f'=f''=0}=-\frac{r_e\lambda^2}{\pi V_{cell}}\sum_j Z_j
=-\frac{r_e\lambda^2 Z_{cell}}{\pi V_{cell}},
```

purely real and negative. With `Z_cell/V_cell = n_e` (unit-cell electron
number density) this is `−r_e λ² n_e/π`, so `δ = −Re(χ₀)/2 = r_e λ² n_e/(2π)`,
the textbook Thomson-limit refractive decrement — matching the docstring's
stated limiting case.

## Cheap filters

- **Units.** `r_e` and `λ` in Å, `V_cell` in Å³, so `r_e λ²/V_cell` is
  dimensionless; `S(0)` is a dimensionless count of effective electrons.
  `χ₀` is dimensionless. ✓
- **Sign / passive medium.** `f1 = Z+f' > 0` and `f2 = f'' ≥ 0` off an
  absorption edge, so the leading minus sign gives `Re χ₀ < 0` (index
  decrement, `n<1`) and `Im χ₀ < 0` (absorptive, matching the `n=1−δ−iβ`,
  `β>0` convention). ✓
- **Limiting cases.** `f'=f''=0` gives the closed-form real Thomson result
  above; `λ→0` (E→∞) drives `χ₀→0` as `λ²` at fixed `f1,f2`, so `n→1`
  (vacuum) at high energy, consistent with `f1,f2` varying slowly relative to
  the explicit `λ²` prefactor away from edges. ✓
- **Multiplicity.** Doubling the cell (same site density, doubled `V_cell`
  and `S(0)` together) leaves `χ₀` unchanged, since it is an intensive
  (density-like) quantity — required for any bulk susceptibility. ✓

## Implementation comparison

`src/pyrite/materials/crystal.py::chi_0`:

```python
for el, _R in info["basis"]:
    if el in forward_F:
        continue
    if use_henke:
        fp, fpp = henke_dispersion(el, E)
        forward_F[el] = (Z_TABLE[el] + fp) + 1j * fpp
    else:
        forward_F[el] = np.full(E.shape, float(Z_TABLE[el]), dtype=complex)
S = np.zeros(E.shape, dtype=complex)
for el, _R in info["basis"]:
    S = S + forward_F[el]
with np.errstate(divide="ignore", invalid="ignore"):
    lam = HC_EV_ANG / E
    return -R_E_ANG * lam**2 / (np.pi * info["V_cell"]) * S
```

Term by term: `forward_F[el]` memoizes `(Z_TABLE[el] + f') + i f''` once per
*element* (an efficiency choice — `henke_dispersion` depends only on element
and energy, not on which site), but the second loop accumulates `S` once per
*basis site*, so `S = Σⱼ f_j(0,E)` correctly sums over every atom in the unit
cell, including repeated elements at distinct sites — exactly `S(0)` above. No
Debye–Waller or position phase is applied anywhere in this function, which is
correct because both are identically 1 at `g=0` and would be pure
implementation overhead to compute and then discard. `henke_dispersion`
(`materials/atomic.py`) returns `f'`, `f''` as the anomalous corrections
directly from Chantler/FFAST via `xraydb` — the same convention already
verified for `absorption-length` and `grazing-optical-constants` — so
`Z_TABLE[el] + fp` is exactly `f1 = Z + f'`. The final line is
`-R_E_ANG * lam**2 / (pi * V_cell) * S`, matching the boxed independent
result term-for-term, with no extra factor of `2`, `π`, or unit-cell
multiplicity.

The `use_henke=False` branch sets `forward_F[el] = Z_TABLE[el]` (real,
`f'=f''=0`), reproducing the Thomson closed form above exactly.

`chi_0` deliberately does not reuse `chi_g(hkl=(0,0,0))`, because `chi_g`'s
non-dispersive part comes from `cromer_mann_f0(el, 0)` — a Cromer–Mann fit
that only approximates `Z` at `s=0` — while `chi_0` uses the exact `Z_TABLE`
value. This is a documented, deliberate normalization choice (shared exactly
with `absorption-length`/`grazing-optical-constants`) rather than a
discrepancy from the `chi_g` formula; the `g=0` limit of `chi_g`'s *structure*
(sum of `f1+if2` over the cell, no phase, no DW factor) is what is being
checked here, and that structure is reproduced exactly.

## Numerical spot check (independent, not using `structure_factor`/`chi_g`)

Computed `S(0)` directly from `xraydb.f1_chantler`/`f2_chantler` and
`Z_TABLE`, bypassing `chi_0`'s own element-memoization and `structure_factor`
entirely, for three catalog crystals:

```text
silicon @ 5000 eV:  reference = -3.960101774020022e-05 -2.163590607052588e-06j
                     production = -3.960101774020022e-05 -2.163590607052588e-06j
                     abs diff = 0.0

hopg    @ 1500 eV:  reference = -4.334994209218367e-04 -1.915976786605724e-05j
                     production = -4.334994209218367e-04 -1.915976786605724e-05j
                     abs diff = 0.0

lif     @ 3890 eV:  reference = -6.809210032825659e-05 -1.259695312646882e-06j
                     production = -6.809210032825659e-05 -1.259695312646882e-06j
                     abs diff = 0.0
```

`use_henke=False` Thomson closed form (`-r_e λ² Z_cell/(π V_cell)`) checked
against production for the same three materials/energies: exact bit-for-bit
agreement (`abs diff = 0.0`) in every case.

Sign checks hold in all three: `Re χ₀ < 0`, `Im χ₀ < 0`.

## Anchors reviewed

- `tests/materials/test_crystallography.py::test_chi_0_linearization_is_optical_constants_exactly` —
  the shared-normalization consistency identity discussed above (`rel=1e-12`);
  correctly labeled in its own comment as a cross-check between two
  normalizations, not independent corroboration.
- `::test_chi_0_without_henke_is_real_thomson_limit` — matches the Thomson
  closed form derived independently above.
- `::test_chi_0_vanishes_and_index_tends_to_vacuum_at_high_energy` — consistent
  with the `λ→0` limiting case above.

None of these substitute for the numeric spot check above, which is built
from independent `xraydb` calls rather than any `chi_0`/`chi_g`/
`structure_factor` helper.

## Verdict

**`matches`.** The `g=0` limit of the already-verified `chi_g` (Feranchuk–Spence
2000 Eq. (3)) reproduces the claimed formula exactly, with the correct units,
sign for a passive medium, Thomson limiting case, and high-energy vanishing.
The implementation reproduces this term-for-term, including the deliberate
`Z_TABLE`-vs-`cromer_mann_f0(el,0)` normalization choice, and matches an
independent `xraydb`-based numeric recomputation to `abs diff = 0.0` at three
catalog materials/energies in both the dispersive and Thomson-limit branches.

Ledger status: `filtered` → `rederived`.
