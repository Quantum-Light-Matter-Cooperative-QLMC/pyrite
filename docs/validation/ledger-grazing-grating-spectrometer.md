# Grazing-incidence grating spectrometer

Part of the [physics validation ledger](physics-validation-ledger.md). See the [validation methodology](methodology.md) for the status lifecycle and the [domain inventories](domain-inventories.md) for a claim-by-claim index.

## `grazing-reflectivity`

- **Claim:** small-angle Fresnel reflectivity `R(θ) = \|r(θ)\|²`, `r(θ) = (θ − √(θ²−2δ−2iβ)) / (θ + √(θ²−2δ−2iβ))`, polarization-independent (grazing-incidence approximation)
- **Code:** `detectors/grating.py::Grating.reflectivity`
- **Source:** Als-Nielsen & McMorrow *Elements of Modern X-ray Physics* 2nd ed. Ch. 3; equiv. Attwood & Sakdinawat Ch. 3
- **Status:** rederived
- **Checks:** bounded in [0,1]; θ≪θc → R→1 (total external reflection); θ≫θc → R→(θc/2θ)⁴·(1+(β/δ)²) independently re-derived (Taylor expansion), test's 60° choice confirmed necessary (Au β/δ≈0.25 at 1500 eV, needs θ≳20° for <10% convergence to the β→0 idealization); monotonic decrease with grazing angle; independent Maxwell/Snell re-derivation matches code term-for-term
- **Anchor:** `tests/detectors/test_grating.py::test_reflectivity_total_external_reflection_below_critical_angle`, `::test_reflectivity_asymptotic_falloff_above_critical_angle`, `::test_reflectivity_bounded_and_decreases_away_from_critical_angle`
- **Notes:** coating element/density confirmed gold for McPherson 251MX, density/molar-mass values not independently re-verified here (`Grating.coating`); depends on `grazing-optical-constants`; write-up `detectors/grazing-reflectivity.md` — the 60°-grazing test checks the formula's own asymptotic self-consistency, not physical accuracy outside θ≪1 (real operation stays at few-degree grazing)

## `alexs-qe-absorption`

- **Claim:** Beer-Lambert absorption-efficiency QE, `QE(E) = peak·(1−exp(−t/L_abs(E)))`
- **Code:** `detectors/grating.py::qe_absorption`
- **Source:** Beer-Lambert / Henke f2 (via `crystallography.absorption_length_ang`, same as `absorption-length`)
- **Status:** rederived
- **Checks:** units (dimensionless, [0,1]) confirmed; thin/thick limiting cases (QE→peak·t/L_abs and QE→peak) independently re-derived from exponential attenuation and numerically spot-checked (thin-limit linear approx vs exact: rel diff 2.2e-6 at E=900 eV); independent from-scratch re-derivation bitwise-matches the code across E=200-8000 eV
- **Anchor:** `tests/detectors/test_grating.py::test_qe_absorption_bounded_and_thickness_limits`, `::test_qe_absorption_increases_with_thickness`
- **Notes:** primary QE model for the greateyes ALEX-s (no public digitized datasheet curve exists, unlike `detector-eaglexo`'s `eaglexo_qe.csv`); same functional form as `eaglexo_response.qe_absorption_model` (a cross-check there); `ACTIVE_SI_UM`/`ENTRANCE_QE_PEAK` device constants are placeholders, `### FILL IN` in code — functional form verified, not the hardware numbers, so this should not advance past `rederived` until a real datasheet lands; write-up `detectors/alexs-qe-absorption.md`

## `alexs-charge-diffusion`

- **Claim:** drift-diffusion charge-cloud spread, `σ² = 2(kT/q)·t·(t−min(L_abs(E),t))/v_dep`
- **Code:** `detectors/grating.py::charge_cloud_sigma_um`
- **Source:** Einstein relation (D=μkT/q) + drift-diffusion, standard back-illuminated-CCD treatment (e.g. Janesick, *Scientific Charge-Coupled Devices*, SPIE 2001, Ch. 4 — generic-treatment pointer, not page-verified)
- **Status:** rederived
- **Checks:** independent re-derivation from D=μkT/q + uniform-field drift confirms mobility μ cancels exactly (verified numerically with μ=1350 vs 450 cm²/(V·s): both give identical σ), reproducing the code's formula term-for-term; units (V·µm²/V → µm) confirmed; soft-photon (σ→max) and hard-photon (σ→0) limits independently re-derived and match tests; single-carrier (non-ambipolar) drift-diffusion assumed, appropriate for single-photon e-h counts but not stated explicitly in the docstring
- **Anchor:** `tests/detectors/test_grating.py::test_charge_cloud_sigma_soft_photon_is_maximum_blur`, `::test_charge_cloud_sigma_zero_when_absorbed_at_front`, `::test_charge_cloud_sigma_positive_and_bounded_by_soft_limit`, `::test_charge_cloud_sigma_decreases_with_energy_over_Si_absorption_band`
- **Notes:** `ACTIVE_SI_UM`/`DEPLETION_VOLTAGE_V`/`OPERATING_TEMP_C` device constants are placeholders, `### FILL IN` in code, pending a real greateyes datasheet/measurement; `z(E)≈min(L_abs(E),t)` is a single-number stand-in for the true depth-resolved exponential absorption profile; functional form verified, not the hardware numbers, so this should not advance past `rederived` until real ALEX-s numbers land; write-up `detectors/alexs-charge-diffusion.md`
