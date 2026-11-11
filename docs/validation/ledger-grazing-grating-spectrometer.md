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
- **Source:** Beer-Lambert / Henke `f₂` convention with Chantler/FFAST tabulation (via `crystallography.absorption_length_ang`, same as `absorption-length`)
- **Status:** rederived
- **Checks:** units (dimensionless, [0,1]) confirmed; thin/thick limiting cases (QE→peak·t/L_abs and QE→peak) independently re-derived from exponential attenuation and numerically spot-checked (thin-limit linear approx vs exact: rel diff 2.2e-6 at E=900 eV); independent from-scratch re-derivation bitwise-matches the code across E=200-8000 eV
- **Anchor:** `tests/detectors/test_grating.py::test_qe_absorption_bounded_and_thickness_limits`, `::test_qe_absorption_increases_with_thickness`
- **Notes:** primary QE model for the greateyes ALEX-s (no public digitized datasheet curve exists, unlike `detector-eaglexo`'s `eaglexo_qe.csv`); same functional form as `eaglexo_response.qe_absorption_model` (a cross-check there); `ACTIVE_SI_UM` is now the greateyes DS-ALEXs Rev05 epitaxial thickness, 15 µm (standard sensor; the write-up's 30 µm spot checks predate it and remain valid for the functional form); `ENTRANCE_QE_PEAK` remains a placeholder (`### FILL IN`), so this approximation is the `AlexS` fallback only until the user's digitized QE curve lands and should not advance past `rederived`; write-up `detectors/alexs-qe-absorption.md`

## `alexs-charge-diffusion`

- **Claim:** drift-diffusion charge-cloud spread, `σ² = 2(kT/q)·t·(t−min(L_abs(E),t))/v_dep`
- **Code:** `detectors/grating.py::charge_cloud_sigma_um`
- **Source:** Einstein relation (D=μkT/q) + drift-diffusion, standard back-illuminated-CCD treatment (e.g. Janesick, *Scientific Charge-Coupled Devices*, SPIE 2001, Ch. 4 — generic-treatment pointer, not page-verified)
- **Status:** rederived
- **Checks:** independent re-derivation from D=μkT/q + uniform-field drift confirms mobility μ cancels exactly (verified numerically with μ=1350 vs 450 cm²/(V·s): both give identical σ), reproducing the code's formula term-for-term; units (V·µm²/V → µm) confirmed; soft-photon (σ→max) and hard-photon (σ→0) limits independently re-derived and match tests; single-carrier (non-ambipolar) drift-diffusion assumed, appropriate for single-photon e-h counts but not stated explicitly in the docstring
- **Anchor:** `tests/detectors/test_grating.py::test_charge_cloud_sigma_soft_photon_is_maximum_blur`, `::test_charge_cloud_sigma_zero_when_absorbed_at_front`, `::test_charge_cloud_sigma_positive_and_bounded_by_soft_limit`, `::test_charge_cloud_sigma_decreases_with_energy_over_Si_absorption_band`
- **Notes:** `ACTIVE_SI_UM` (15 µm) and `OPERATING_TEMP_C` (−100 °C) now come from the greateyes DS-ALEXs Rev05 datasheet; `DEPLETION_VOLTAGE_V` is not in it and stays a `### FILL IN` placeholder; `z(E)≈min(L_abs(E),t)` is a single-number stand-in for the true depth-resolved exponential absorption profile; functional form verified, not the hardware numbers, so this should not advance past `rederived` until real ALEX-s numbers land; write-up `detectors/alexs-charge-diffusion.md`

## `flat-field-focal-plane`

- **Claim:** a ray diffracted at $\beta$ meets a flat focal plane with unit normal at $\beta_H$ and perpendicular distance $r_H'$ after path $r'(\beta)=r_H'/\cos(\beta-\beta_H)$, at dispersion-axis offset $s(\beta)=r_H'\tan(\beta-\beta_H)$ from the foot of the perpendicular ($s\to\mp r_H'\cot\beta$ for $\beta_H\to\pm90°$); a `"difference"`-convention record maps to this module's $\sin\beta=m\lambda/d-\sin\alpha$ by $\beta\to-\beta$ at fixed $m$
- **Code:** `detectors/grating.py::{FlatFieldMount.position_mm, FlatFieldMount.exit_arm_mm, Grating.from_catalog, Grating.mount_residuals}`
- **Source:** plane–line intersection in the dispersion plane; flat-field mount geometry of Kita, Harada, Nakano & Kuroda, *Appl. Opt.* 22, 512 (1983)
- **Status:** unverified
- **Checks:** ray endpoint lies on the plane and its tangential offset equals $s$; $\beta=\beta_H$ lands at the foot; $\beta_H\to90°$ reduces to $-r_H'\cot\beta$; synthetic catalog mount (both order conventions) reproduces tabulated band-end $\beta$ to < 0.05° and the focal-plane length to < 1 %
- **Anchor:** `tests/detectors/test_grating.py::test_flat_field_position_is_the_plane_intersection`, `::test_flat_field_reduces_to_cot_beta_for_a_plane_normal_to_the_grating`, `::test_from_catalog_builds_a_flat_field_grating`
- **Notes:** the real-grating check, `tests/detectors/test_grating.py::test_internal_grating_mount_matches_its_vendor_table`, runs only against the private catalog (`PYRITE_INTERNAL_USER_CATALOG`, `PYRITE_INTERNAL_GRATING`) and is skipped otherwise; no vendor mount number is in the repository. Point-source geometry only: no defocus, coma or VLS aberration. The existing explicit-distance map `L tan(β−β_ref)` is unchanged for existing callers.

## `grating-slit-image-width`

- **Claim:** a slit of width $w$ at entrance arm $r$ images at fixed $\lambda$ to $\Delta\beta=w\cos\alpha/(r\cos\beta)$, i.e. width $w'=w(\cos\alpha/\cos\beta)(r'/r)$ normal to the ray and $w'/\cos(\beta-\beta_H)$ on the focal plane; each column's point-slit flux is convolved with rect(width) ⊗ rect(pixel) ⊗ optional Gaussian of FWHM $|ds/d\lambda|\,\lambda/R$, conserving flux
- **Code:** `detectors/grating_image.py::{slit_image_width_mm, slit_image_pixels}`
- **Source:** differential of the grating equation $\cos\alpha\,d\alpha+\cos\beta\,d\beta=0$ (Samson & Ederer, *Vacuum Ultraviolet Spectroscopy*, Academic 2000); focal-plane map of `flat-field-focal-plane`; closed-form CDF of a rect ⊗ rect ⊗ Gaussian kernel
- **Status:** unverified
- **Checks:** width equals the anamorphic formula (with the plane-tilt factor, and exactly $w'$ at normal incidence on the plane) for the flat-field and explicit-distance maps; $w=0$ returns the point-slit dispersion exactly and $w\to0$ converges to it; flux conserved to $10^{-10}$ with and without aberration; a delta line through a wide slit images to the predicted rect width
- **Anchor:** `tests/detectors/test_grating_image.py::test_slit_image_width_matches_anamorphic_formula`, `::test_slit_image_width_explicit_distance_map`, `::test_zero_width_slit_is_the_point_slit_dispersion`, `::test_narrow_slit_converges_to_point_slit`, `::test_slit_convolution_conserves_flux_and_broadens`, `::test_rect_kernel_width_on_a_single_line`
- **Notes:** the kernel width is evaluated at each pixel column's focal-plane position and the source pixel is taken uniform across its pitch. Along-slit capture (`AlongSlitSpread`, `detected_image_2d`) is a normalization, not a separate claim: the image's row sum equals the 1D image times the capture fraction (`::test_2d_image_row_sum_is_1d_times_capture`). Finite-source and slit-plane kernels are #1076.

