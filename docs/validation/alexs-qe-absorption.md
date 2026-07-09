# `alexs-qe-absorption` — independent verification

**Code**: `src/cxr_mc/grating.py::qe_absorption`
**Ledger claim**: Beer-Lambert absorption-efficiency QE, `QE(E) = peak·(1 − exp(−t/L_abs(E)))`
**Source**: Beer-Lambert exponential attenuation + Henke f2 via `crystallography.absorption_length_ang`
(same absorption-length routine already covered by the `absorption-length` ledger entry).

## 1. Scope

- Signature: `qe_absorption(E_eV, active_um=None, peak=None) -> QE` (dimensionless, in `[0, peak]`).
- `t_um` = active (depleted) Si thickness [µm], default `ACTIVE_SI_UM` (placeholder, 30 µm).
- `L_ang` = Henke-f2 absorption length [Å] from `absorption_length_ang("Si", E, SI_N_PER_ANG3)`.
- `peak` = entrance-surface/dead-layer scale factor in `[0,1]`, default `ENTRANCE_QE_PEAK` (placeholder, 0.9).
- Assumption stated in the docstring: normal incidence, single characteristic absorption length,
  no back-surface dead layer (that's `eaglexo`'s job, not this detector's).

## 2. First-principles re-derivation (before reading the implementation body)

A photon beam of intensity `I0` normally incident on a homogeneous absorber attenuates via the
standard exponential (Poisson thinning) law:

```
dI/dx = -I/L_abs   =>   I(x) = I0 exp(-x/L_abs)
```

where `L_abs` is the 1/e absorption length (mean free path for absorption). The probability a
given photon survives (transmits) through thickness `t` is `I(t)/I0 = exp(-t/L_abs)`. The
complementary probability — absorbed somewhere within `[0,t]` — is

```
P_abs(t) = 1 - exp(-t/L_abs)
```

This is exactly the Beer-Lambert absorption-probability form and is the correct physical model for
"photon absorbed within a slab of thickness t given mean free path L_abs," under the stated
normal-incidence assumption (the path length equals the physical thickness only at normal
incidence; off-normal photons would need `t/cos(θ)`, but that's out of scope here — grazing angle
only enters the upstream grating-reflectivity stage, not the CCD).

`peak` is a bounded `[0,1]` prefactor for surface/dead-layer loss, applied multiplicatively — this
does not change the shape, only rescales the ceiling, and is dimensionally inert (unitless × unitless).

So my independently derived expression is:

```
QE(E) = peak * (1 - exp(-t_um*1e4 / L_abs_angstrom(E)))
```

with the `1e4` factor converting µm → Å (1 µm = 1e4 Å), matching the units `absorption_length_ang`
returns (confirmed from its own docstring/derivation: `mu` has units 1/Å, so `L_abs = 1/mu` is in Å).

### Limiting cases (independently derived, not copied from the docstring)

- **t → 0** (or `L_abs → ∞`, optically thin): Taylor-expand `1 - exp(-x) = x - x²/2 + ...` for
  small `x = t/L_abs`, giving `QE → peak·t/L_abs`, linear in thickness. This is the expected
  hard-photon/thin-sensor regime.
- **t → ∞** (or `L_abs → 0`, optically thick): `exp(-t/L_abs) → 0`, so `QE → peak`, full capture
  — the expected soft-photon/thick-sensor regime.
- Both limits, and the bound `0 ≤ QE ≤ peak` for all `t, L_abs > 0` (since `0 ≤ 1-exp(-x) ≤ 1` for
  `x ≥ 0`), follow directly from the monotonic, bounded nature of `1-exp(-x)`.

## 3. Diff against the implementation

```python
L_ang = absorption_length_ang("Si", np.clip(E, 1e-3, None), SI_N_PER_ANG3)
out = p * (1.0 - np.exp(-(t_um * 1.0e4) / L_ang))  # um -> Angstrom
```

This is term-for-term identical to my independent expression above: same `peak` prefactor, same
`1 - exp(-t/L_abs)` Beer-Lambert form, same µm→Å conversion factor (1e4), and it reuses the
already-separately-covered `absorption_length_ang` (see `absorption-length` ledger row) rather than
re-deriving Henke f2 physics inline. The `E → clip(E, 1e-3, None)` guard avoids a `1/E` singularity
in `lam = HC_EV_ANG/E` at `E=0`; combined with the `nan_to_num(..., posinf=0.0)` wrapper this maps
any pathological input to `QE=0` rather than propagating `NaN`/`inf`, which is a safe (if slightly
inelegant) numerical convention and does not affect physical values for `E > 0`.

This is also structurally identical (same formula, same units, same conversion factor) to the
existing, previously-reviewed `eaglexo_response.qe_absorption_model`, which strengthens confidence
that the pattern is applied correctly elsewhere in this codebase and this is not a one-off
transcription.

## 4. Numeric spot-check

Independently computed (script run against the code's own `absorption_length_ang`, not copying the
`qe_absorption` implementation) at `t=30 µm` (the `ACTIVE_SI_UM` placeholder), `peak=0.9`:

| E [eV] | L_abs [µm] | my QE | code QE | abs diff |
|---|---|---|---|---|
| 200  | 0.0645 | 0.900000 | 0.900000 | 0.0 |
| 900  | 2.234  | 0.899999 | 0.899999 | 0.0 |
| 1500 | 8.912  | 0.868931 | 0.868931 | 0.0 |
| 4000 | 9.715  | 0.858969 | 0.858969 | 0.0 |
| 8000 | 68.71  | 0.318396 | 0.318396 | 0.0 |

Max absolute diff over this sweep: **0.0** (exact bitwise agreement, as expected — same formula,
same underlying `absorption_length_ang` call).

Limiting-case numerics at E=900 eV, `L_abs=2.234 µm`:
- Thin (`t=1e-5 µm`): linear approx `t/L_abs = 4.476851e-6` vs. exact `1-exp(-t/L_abs) = 4.476841e-6`,
  relative difference `2.24e-6` — converges as expected (`O(x)` correction at `x~4.5e-6`).
- Thick (`t=1e5 µm`): `QE/peak → 1.0` to double precision.

Both match the closed-form Beer-Lambert limiting-case predictions.

## 5. Filters

- **Units**: `t/L_abs` is Å/Å = dimensionless; `exp(-t/L_abs)` and hence `1-exp(-t/L_abs)` are
  dimensionless. `t_um*1e4` correctly converts to Å, matching `L_ang`'s units. **Pass.**
- **Sign/bounds**: `0 ≤ QE ≤ peak ≤ 1` for `t, L_abs > 0`, confirmed both analytically (monotonic
  bounded function of a non-negative argument) and by the existing test
  `test_qe_absorption_bounded_and_thickness_limits`. **Pass.**
- **Limiting cases**: thin → linear (`peak·t/L_abs`), thick → `peak` (full capture), both confirmed
  analytically and numerically above, and both directly exercised by
  `test_qe_absorption_bounded_and_thickness_limits` (which checks `qe_thin ≈ t/L_abs` to `rel=1e-3`
  and `qe_thick ≈ 1.0` to `abs=1e-6`) and `test_qe_absorption_increases_with_thickness` (monotonicity
  sanity check, `thin < thick`). **Pass.**
- **Source correctness**: Beer-Lambert absorption probability `1-exp(-t/L_abs)` is the standard,
  correct first-principles result for photon-absorption probability in a slab under the
  normal-incidence assumption; the code and docstring correctly scope this assumption (CCD is
  treated as normal-incidence; grazing angle is confined to the upstream grating stage). **Pass.**

## 6. Verdict

**rederived.** The implementation is an exact term-for-term match to an independently-derived
Beer-Lambert absorption-probability expression, dimensionally consistent, correctly bounded, and
both limiting cases check out analytically and numerically (and are directly exercised by the
existing tests). No discrepancy found.

Caveat carried forward from the ledger (not a physics defect, a data-provenance one): `ACTIVE_SI_UM`
and `ENTRANCE_QE_PEAK` are placeholder device constants (`### FILL IN` in code) pending a real
greateyes ALEX-s datasheet/measurement — the *functional form* is verified here; the *device
parameter values* are not, and this function is the primary (not cross-check) QE model for that
detector. This does not block `filtered`/`rederived` status for the formula itself but should stay
flagged until real constants land.
