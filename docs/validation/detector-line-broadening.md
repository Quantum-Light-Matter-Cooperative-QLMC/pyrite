# Validation: `detector-line-broadening`

**Claim.** EDS polar-aperture line broadening, Zhai et al. 2025 SI Eq. (14):
`FWHM_Δθobs = (2√(2ln2)/3)·(∂Ep/∂θobs)·Δθobs`, with the paper's full second
line `∂Ep/∂θobs = −Ep·(−cosφ cosθ v_x − sinφ cosθ v_y + sinθ v_z)/(c − cosφ
sinθ v_x − sinφ sinθ v_y − cosθ v_z)`.

**Code.** `src/cxr_mc/montecarlo/detector.py::aperture_fwhm_eV` (post
2026-07-11 prefactor fix). **Anchor.** `tests/detectors/test_response.py`.
**Source.** Zhai et al. 2025 SI Eq. (14).
**Verifier context.** Independent session; did not author the implementation.

## Independent derivation

**Reduction to beam-axis geometry.** Set v = (0, 0, βc) (velocity along z,
θobs the polar angle of the observation vector from v). Every azimuthal term
in Eq. (14) carries v_x or v_y and vanishes identically. Numerator:
`−Ep·βc·sinθobs`; denominator: `c(1 − β cosθobs)`. Hence
`∂Ep/∂θobs = −Ep·β·sinθobs/(1 − β·cosθobs)` — magnitude
`Ep·β sinθ/(1 − β cosθ)`, matching the code's `dE_dth` term-for-term.

**Dispersion cross-check.** From `Ep = ħ(v·g)/(1 − β cosθobs)`: the numerator
v·g depends on beam direction and the crystal's g only, not on the observation
angle, so the θobs dependence sits entirely in the Doppler denominator and
`∂Ep/∂θobs = Ep·∂θ[−ln(1 − β cosθ)] = −Ep·β sinθ/(1 − β cosθ)`. Same result.
The negative sign (Ep falls as θobs opens away from v) is physical; a width
takes the magnitude, so the code's positive `dE_dth` is correct.

**Prefactor.** The code uses `2*sqrt(2*log(2))/3 ≈ 0.7849`, matching the
printed equation (numerically confirmed against an independent evaluation). A
variance match to a uniform span of full width Δθ (σ = Δθ/√12) would give
`√(2ln2/3) ≈ 0.6798` instead. Zhai's factor is reproduced exactly by
interpreting the full span as 3σ of a Gaussian (σ = Δθ/3, i.e. half-span =
1.5σ): FWHM = 2√(2ln2)·σ = (2√(2ln2)/3)·Δθ. That is a defensible ad-hoc
Gaussian-equivalencing convention (≈15% wider than the variance match), and
the code correctly follows the paper rather than rederiving it — flagged, not
a discrepancy.

## Filters (numeric, this session)

- **Units/sign:** eV out (linear in E_eV, angles in rad, dimensionless
  prefactor); positive for all 0 < θobs < π, β ∈ {0.1, 0.5, 0.9, 0.99}. ✓
- **Point check:** E = 400 eV, β = 0.55, θobs = 119°, Δθ = 16.6° →
  independent expression and code both give 34.5469 eV (rtol 1e-14). ✓
- **Limits:** FWHM = 0 exactly at Δθ→0 and at β→0. ✓
- **Monotonicity:** strictly increasing in β at θobs = 119° over
  β ∈ (1e-4, 0.999). ✓ (sinθ/(1−β cosθ) has ∂β > 0 for θ > 90°.)
- **Pinning tests:** `tests/detectors/test_response.py` — 2 passed.

## Adjudication

Filters pass; the independent reduction of the full Eq. (14) matches the
implementation exactly, corroborated by the dispersion-based derivative; the
prefactor matches the printed equation and admits a coherent (span = 3σ)
reading. Recommended status: **`rederived`** (upgrade from `filtered`).
Final `signed-off` is a human decision.
