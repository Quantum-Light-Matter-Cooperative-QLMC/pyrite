# Validation: `grazing-optical-constants`, `grazing-reflectivity`

**Claim 1 (`grazing-optical-constants`).** Complex refractive index `n = 1 − δ − iβ` of a material from its Henke/Chantler anomalous scattering factors: `δ(E) = (r_e λ²/2π)·n_atomic·f1(E)`, `f1 = Z + f'(E)`; `β(E) = (r_e λ²/2π)·n_atomic·f2(E)`.

**Claim 2 (`grazing-reflectivity`).** Small-angle grazing-incidence Fresnel reflectivity `r(θ) = (θ − √(θ² − 2δ − 2iβ)) / (θ + √(θ² − 2δ − 2iβ))`, `R(θ) = |r(θ)|²`, θ the grazing angle from the surface, treated polarization-independent.

**Code.** `src/pyrite/materials/crystal.py::optical_constants`, `src/pyrite/detectors/grating.py::Grating.reflectivity` (+ `Grating.throughput`, `disperse_spectrum(..., weight_by_throughput=)`). **Anchor.** `tests/materials/test_crystallography.py::test_optical_constants_beta_matches_absorption_length`, `::test_optical_constants_delta_positive_off_edge`; `tests/detectors/test_grating.py::test_reflectivity_total_external_reflection_below_critical_angle`, `::test_reflectivity_asymptotic_falloff_above_critical_angle`, `::test_reflectivity_bounded_and_decreases_away_from_critical_angle`. **Source.** Als-Nielsen & McMorrow, *Elements of Modern X-ray Physics* 2nd ed., Ch. 3; equivalently Attwood & Sakdinawat, *X-Rays and Extreme Ultraviolet Radiation* 2nd ed., Ch. 3. **Verifier context.** Independent session; did not author the implementation. Re-derived from Maxwell boundary conditions / the standard atomic-scattering result, then diffed against the code.

## Independent derivation

**δ, β.** The standard X-ray optical-constant result from the atomic scattering factor gives exactly `δ(E) = (r_e λ²/2π)·n_atomic·f1(E)`, `β(E) = (r_e λ²/2π)·n_atomic·f2(E)` with `f1 = Z + f'`. This matches the code term-for-term. `henke_dispersion` returns `f'`, `f''` (anomalous parts only, per its own docstring), so `f1 = Z + f'` correctly reconstructs the Henke convention inside `optical_constants`.

**β ↔ `absorption_length_ang` cross-check.** This is not an independent check of the physics — `optical_constants`'s `β` and `absorption_length_ang`'s internal `beta_idx` are literally the same expression applied to the same `f2`, so `test_optical_constants_beta_matches_absorption_length`'s `rel=1e-9` pass is a construction tautology, not corroboration. The code's own docstring says as much ("limiting-case consistency check, not an independent derivation") — noted here so the ledger doesn't overstate what that test buys.

**Independent numeric spot-check (new, not in the test suite).** Computed `θ_c = √(2δ)` from `optical_constants` at Cu Kα (8048 eV) for Si and Au — textbook benchmark values from this exact chapter:

| material | computed θ_c | textbook θ_c |
|---|---|---|
| Si | 0.2230° | ≈ 0.22° |
| Au | 0.5538° | ≈ 0.55° |

Exact match — independent corroboration beyond the self-consistency check above.

**Reflectivity.** From Maxwell/Snell at a vacuum(n₁=1)/medium(n₂=1−δ−iβ) interface, grazing angle θ from the surface: `k_z1 = k0 sinθ`, `k_z2 = k0√(n²−cos²θ)`, `r_s = (k_z1−k_z2)/(k_z1+k_z2)`. Small-angle: `n²−1 ≈ −2δ−2iβ`, `cos²θ ≈ 1−θ²`, so `n²−cos²θ ≈ θ²−2δ−2iβ`, giving exactly `r(θ) = (θ−√(θ²−2δ−2iβ))/(θ+√(θ²−2δ−2iβ))` — matches the code term-for-term, including its own in-docstring derivation. p/s polarization convergence at grazing incidence (`n²→1` makes `r_p→r_s`) is also correct, supporting the polarization-independence assumption.

## Limiting cases (derived independently)

- **θ ≪ θ_c** (β→0 idealization): the sqrt argument is negative real, so `r` is a numerator/denominator complex-conjugate pair → `|r| = 1`. The existing test uses θ = 0.001° vs θ_c(Au, 1500 eV) = 2.53° (ratio 4×10⁻⁴) — solidly in this regime. ✓
- **θ ≫ θ_c**: Taylor-expanding the code's own formula (not assuming small θ) gives `r ≈ (δ+iβ)/(2θ²)`, so `R → (δ²+β²)/(4θ⁴) = (θ_c/2θ)⁴·(1+(β/δ)²)`. Numerically, Au at 1500 eV has `β/δ ≈ 0.25` — not negligible — so the pure `(θ_c/2θ)⁴` idealization (β→0) only converges to <10% error for grazing angle ≳20° (8.1% at 20°, 13.6% at 10°, 41% at 5°). The existing test's choice of 60° (6.6% error against the idealization) is a deliberate, necessary choice given that convergence rate, not arbitrary slack.

**Caveat for future readers of that test:** 60° grazing is well outside the θ≪1 rad domain the small-angle Fresnel *reduction itself* assumes (`cos²(60°) = 0.25` vs the approximation's `1−θ² = −0.097` — badly broken as a model of true large-angle reflectivity there). What `test_reflectivity_asymptotic_falloff_above_critical_angle` actually verifies is that the code's `r(θ)` correctly reproduces **its own** algebraic θ≫θ_c asymptote — a Taylor-series identity valid for any θ as long as `δ,β ≪ θ²`, independent of θ's absolute size — not that the code matches true large-angle Fresnel physics (it was never meant to be evaluated there; real grating operation stays at few-degree grazing angles where θ≪1 holds throughout).

## Adjudication

Filters pass (units, sign, β↔absorption-length identity by construction); the independent Maxwell/Snell derivation of the reflectivity formula matches the code exactly, including the small-angle reduction algebra; the θ_c spot-check reproduces known Si/Au values at Cu Kα to 4 significant figures; both limiting cases hold as tested, with the θ≫θ_c test's regime choice now explained rather than assumed. Recommended status for both ids: **`rederived`** (independent derivation matches). Final `signed-off` is a human decision.
