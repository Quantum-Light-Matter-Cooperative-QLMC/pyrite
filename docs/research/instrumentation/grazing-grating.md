# Grazing-incidence soft X-ray grating spectrometer

**Status:** Exploratory research model; not wired into the production pipeline.

A measurement modality promoted to TODO P1 (was P3 #2): instead of reading the PXR+CBS spectrum as energy-vs-counts at a fixed take-off angle (EDS/Timepix), **disperse** it with a reflection grating at grazing incidence and read the resulting *spatial* image on a position-sensitive detector (Raptor Eagle XO CCD, or an "Alex"-type detector from Ultrafast Innovations). The grating's dispersion gives spectral resolution set by geometry and pixel size rather than by the detector's native energy resolution — potentially far better than the ~130 eV EDS line width that dominates the soft-X-ray band today.

---

## Status

`src/pyrite/detectors/grating.py` implements the **dispersion geometry**, the **grazing-incidence Fresnel reflectivity** of the grating's coating, a **simple CCD pixel grid** (geometry-only rebinning), a **combined forward-model entry** (`detected_image`, chaining the two above) that turns an `mc_cxr_spectrum` output directly into a detected image, and now a **physical CCD response** (`qe_absorption`, `charge_cloud_sigma_um`, `energy_fwhm_eV`, `detected_image_physical`) with real (if device-constant-placeholder) QE, charge-sharing, and energy-resolution physics — all as a standalone forward model — nothing in the sweep/plot pipeline imports it yet. Cross-checked in `tests/detectors/test_grating.py`.

| provided | meaning |
| --- | --- |
| `wavelength_angstrom(E)` / `groove_spacing_angstrom(ρ)` | unit conversions |
| `coating_number_density_per_ang3(element)` | atomic number density for a coating (Au/Pt/Ni) |
| `Grating(groove_density_per_mm, alpha_rad, order, coating, groove_efficiency)` | a grating in a fixed mount |
| `Grating.diffraction_angle_rad(E)` | the grating equation `sinβ = mλ/d − sinα` (NaN if no propagating order) |
| `Grating.angular_dispersion_rad_per_angstrom(E)` | `dβ/dλ = m/(d cosβ)` |
| `Grating.reflectivity(E)` | small-angle Fresnel reflectivity `R(E)` of the coating (grazing angle, polarization-independent approximation) |
| `Grating.throughput(E)` | `reflectivity(E) × groove_efficiency` (the latter a placeholder scalar, not a real groove-profile model) |
| `disperse_spectrum(E, spec, grating, distance_mm, weight_by_throughput=False)` | **flux-conserving** map of a spectrum to detector position (`∫I dx = ∫spec dE`); optionally throughput-weighted |
| `resolving_power(E, grating, distance_mm, pixel_mm)` | pixel-limited `λ/Δλ` |
| `ALEXS_SENSORS` / `SimpleCCD.from_alexs(variant)` | greateyes ALEX-s `1k256`/`2k512` pixel-format registry + a fixed pixel grid built from it |
| `bin_to_pixels(position_mm, intensity_per_mm, ccd, center_mm=None)` | rebin a dispersed profile onto `ccd`'s fixed pixels (each pixel integrates the polar-angle/position span it subtends); light outside the sensor is dropped |
| `detected_image(E_grid_eV, spec, grating, ccd, distance_mm, weight_by_throughput=True, ...)` | combined forward-model entry: `mc_cxr_spectrum` output → dispersed, detected image (counts vs pixel), chaining `disperse_spectrum` + `bin_to_pixels` |
| `qe_absorption(E, active_um=ACTIVE_SI_UM, peak=ENTRANCE_QE_PEAK)` | Beer-Lambert absorption-efficiency QE of the active silicon |
| `charge_cloud_sigma_um(E, active_um=..., v_dep=DEPLETION_VOLTAGE_V, temp_c=OPERATING_TEMP_C)` | lateral charge-cloud RMS spread [um] from drift-diffusion |
| `energy_fwhm_eV(E, n_pix=4, read_noise_e=READ_NOISE_E)` | Fano + read-noise single-photon energy resolution (photon-counting mode; not applied by default) |
| `detected_image_physical(E_grid_eV, spec, grating, ccd, distance_mm, ...)` | like `detected_image`, but weights by `qe_absorption` before binning and applies the `charge_cloud_sigma_um` Gaussian blur (at the detected spectrum's flux-weighted mean energy) after binning |
| `Grating(..., efficiency_curve=TabulatedEfficiency)` | absolute vendor efficiency `η(E)`; `throughput = η(E)` alone (no Fresnel factor), zero outside the tabulated band |
| `FlatFieldMount(r, r_H', β_H)` / `Grating(..., mount=...)` | flat focal plane at perpendicular distance `r_H'`, normal at `β_H`: `s(β) = r_H' tan(β − β_H)` (`≈ ∓r_H' cot β` for `β_H → ±90°`); used when `distance_mm` is `None` |
| `Grating.from_catalog(name)` / `Grating.mount_residuals()` | build grating, mount and curve from a private catalog `[gratings.NAME]`; compare the grating equation with the mount's tabulated band-end angles and focal-plane length |
| `grating_image.slit_image_width_mm` / `SlitImage` | slit image width on the focal plane, `w (cos α/cos β)(r'/r)/cos(β − β_H)` |
| `grating_image.slit_image_pixels` | point-slit image convolved per column with rect(slit image) ⊗ optional Gaussian aberration (vendor resolving power); flux-conserving |
| `grating_image.AlongSlitSpread` / `detected_image_2d` | along-slit uniform astigmatic spread `H(λ)` (or tabulated capture fraction) clipped to the sensor rows → `(rows, columns)` image whose row sum is the 1D image × capture |
| `AlexS`, `alexs_response.alexs_frame`, `grating_image.alexs_spectrometer_image` | ALEX-s charge-integrating response: `ADU = gain · min(Σ N·QE·E/3.66 eV + I_dark t, FW)` plus optional Poisson and read noise; the full spectrum → slit → grating → ALEX-s frame |

## Physics and conventions

Reflection grating equation, angles from the grating **normal**:

```
d (sin α + sin β) = m λ          →     sin β = m λ / d − sin α
```

- `α` = incidence angle, `β` = diffraction angle (from normal); `m = 0` is specular (`β = −α`).
- **Grazing** incidence (grazing angle `θ_g = 90° − α` of a few degrees) is a choice of `α`, not a different equation — it is what buys usable reflectivity for soft X-rays. In this regime first orders never run out of propagating solutions across 100 eV–2 keV (the NaN branch is only hit by pathological rulings; see the test).
- `λ = hc / E` with `hc = 12398.42 eV·Å` (`crystallography.HC_EV_ANG`).

Grazing-incidence Fresnel reflectivity of the coating, small-angle form:

```
n = 1 − δ − iβ                              (complex refractive index)
r(θ) = (θ − √(θ² − 2δ − 2iβ)) / (θ + √(θ² − 2δ − 2iβ))
R(θ) = |r(θ)|²
```

with `θ` the grazing angle (`Grating.grazing_angle_rad`) and `δ, β` from `crystallography.optical_constants` (Henke/Chantler `f1 = Z+f'`, `f2`). Treated as polarization-independent, the standard grazing-incidence approximation (source, assumptions, and limiting cases are in `Grating.reflectivity`'s docstring). `Grating.throughput(E) = reflectivity(E) × groove_efficiency` combines this with the placeholder groove-efficiency scalar; pass `weight_by_throughput=True` to `disperse_spectrum` to get a throughput-weighted dispersed profile instead of a purely geometric one.

## Hardware targets

The instrument is a **lab-built flat-field spectrometer**:

- **Grating: Shimadzu L2400-1-7** flat-field (varied-line-spacing, laminar) grazing-incidence grating. Its datasheet mount geometry, absolute efficiency curve and spot-diagram data are under restricted distribution, so none of them is in this repository: they live in the laboratory's private catalog layer (`gratings/`, `curves/`), and `Grating.from_catalog` reads them at run time. Tests use synthetic gratings; the real-grating consistency check (`tests/detectors/test_grating.py::test_internal_grating_mount_matches_its_vendor_table`) runs only when `PYRITE_INTERNAL_USER_CATALOG` and `PYRITE_INTERNAL_GRATING` point at that private layer.
- **Detector: greateyes ALEX-s 2k512 BI UV1**, from the public datasheet (greateyes DS-ALEXs Rev05, May 2025): 2048 × 515 usable pixels, 13.5 µm pitch, 27.6 × 6.9 mm active area; full well 100 ke⁻ (register 400 ke⁻); read noise 3.5 / 6.8 / 10.7 e⁻ at 50 kHz / 1 MHz / 3 MHz; dark current 0.00025 e⁻/px/s at −100 °C; gain 1 count/e⁻ (standard) or 0.34 count/e⁻ (high capacity); 3.66 eV per electron–hole pair; 15 µm epitaxial silicon (40 µm only for the DD variant). The QE curve will be digitized into the private catalog (`qe_curve`); until then `AlexS` uses `qe_absorption` with the 15 µm epitaxial layer, whose entrance-window peak is a placeholder.

**Surveyed background, no longer the target.** The McPherson 251MX flat-field SXR/EUV spectrograph (four interchangeable gold-coated gratings, 120–2400 g/mm, ~87° incidence, ~25 mm focal plane; [product page](https://www.mcphersoninc.com/spectrometers/xuhvvuvuv/model251mx.html), [flat-field grating press release](https://mcphersoninc.com/pressreleases/flatFieldGrating120.html)) was the earlier candidate; McPherson did not respond, so the spectrometer was built around the Shimadzu grating instead. Its product notes on laminar groove profiles remain useful background for a future groove-efficiency model.

## What is NOT modelled yet (and why)

Dispersion geometry, coating reflectivity or a tabulated absolute efficiency, the flat-field focal plane, an analytic slit-image kernel, along-slit capture and the ALEX-s charge-integrating response are modelled; deliberately still out of scope:

- **Groove-profile diffraction efficiency** from first principles — a vendor absolute efficiency curve (`Grating.efficiency_curve`) replaces it for a real grating; without one, the groove factor is a placeholder scalar (`Grating.groove_efficiency`). A predictive model needs a scalar or rigorous-coupled-wave (RCWA) treatment of the ruled profile.
- **Measured ALEX-s QE curve** — pending the user's digitization; the Beer-Lambert fallback uses the datasheet 15 µm epitaxial layer but a placeholder entrance-window peak (`ENTRANCE_QE_PEAK`) and no UV1 coating structure. `DEPLETION_VOLTAGE_V` is not in the datasheet and stays a placeholder for `charge_cloud_sigma_um`. Charge diffusion is not applied in `alexs_spectrometer_image`; `detected_image_physical`'s blur uses one sigma per frame. ADC saturation and register-well limits for on-chip binning are not modelled.
- **Ray tracing and VLS aberrations** — beyond the optional Gaussian width from the vendor resolving power, no defocus, coma or VLS aberration model. The along-slit astigmatic height `H(λ)` (or capture fraction) is user-supplied from private spot-diagram data through `AlongSlitSpread`; the catalog has no field for it yet, and within one column the along-slit spread uses that column's height.
- **Source size / beam divergence, finite slit geometry** — the slit is an analytic kernel, not a scene object, and slit acceptance (solid angle from the source) is supplied by the caller; the finite-slit/finite-source model and per-row spectra are issue #1076, the slit acceptance #1075. The grating stays a post-scoring forward model on a spectrum (ADR-0008), not a transport scene element.

## Phased plan to a real modality

1. **(done)** Dispersion geometry + resolving power, tested.
2. **(done)** Coating Fresnel reflectivity `R(E)` from optical constants (`crystallography.optical_constants`: `f', f''` → δ, β → Fresnel reflectivity at grazing angle, `Grating.reflectivity`), combined with a placeholder groove- efficiency scalar into `Grating.throughput`, wired into `disperse_spectrum` via `weight_by_throughput=True`. The groove-profile efficiency itself (vs. McPherson 251MX grating: 120/300/1200/2400 g/mm) is still a placeholder — see "What is NOT modelled yet".
3. **(done)** A **simple** CCD forward model: bin detected photons by the polar angle each pixel subtends (no QE/charge-sharing structure yet), sized to the ALEX-s 1k256/2k512 pixel formats above (`ALEXS_SENSORS`, `SimpleCCD`, `bin_to_pixels`). This is deliberately crude — it exists to get the full grating→image pipeline working end-to-end before adding detector physics.
4. **(done)** A forward-model entry (`detected_image`) that takes the model's `mc_cxr_spectrum` output + a `Grating` + the simple CCD from step 3 and returns the **dispersed, detected image** (counts vs pixel), so it slots in beside the Timepix / Eagle XO models. Pure composition of steps 2 and 3 (chains `disperse_spectrum` then `bin_to_pixels`) — no new physics, so no new ledger row.
5. **(done)** Replace the simple CCD with a more physical model: Beer-Lambert absorption QE (`qe_absorption`), Einstein-relation drift-diffusion charge sharing (`charge_cloud_sigma_um`), and Fano+read-noise energy resolution (`energy_fwhm_eV`), composed into `detected_image_physical` — same upgrade path `eaglexo_response.py` took, except no digitized greateyes ALEX-s datasheet exists (unlike the Eagle XO's `eaglexo_qe.csv`), so this is closed-form physics with `### FILL IN` device constants rather than a measured curve — see "What is NOT modelled yet" and the `alexs-qe-absorption`/`alexs-charge-diffusion` ledger rows.
6. **(preliminary survey done, no code yet)** Survey CCD/grating options beyond the McPherson 251MX + ALEX-s pairing for the broader ~10 eV-4 keV band (the 251MX's 2400 g/mm grating tops out near 1.24 keV), preferring broadband coverage. Findings so far (web/literature search, not yet cross-checked against primary datasheets in all cases — flagged where uncertain):
   - **Grating spectrometers**: Horiba/Jobin-Yvon PGM200 (grazing-incidence flat-field, interchangeable 1800/450 g/mm gratings, "varied groove depth" technology for a broadened single-grating band — exact range unconfirmed); Bestec GmbH custom plane-/spherical-grating monochromators (soft-to-hard X-ray, beamline-grade, custom builds not a catalog product); a published research instrument (Imazono et al., *Appl. Opt.* 57(27):7770, 2018, [PubMed](https://pubmed.ncbi.nlm.nih.gov/30462040/)) reaches 0.9-3.3 keV using an **aperiodic Ni/C multilayer** grating coating rather than a simple metal film — direct evidence that a multilayer coating (not gold) plus a shallower grazing angle is likely required to push toward the 4 keV goal.
   - **Detectors**: Andor iKon-M/L/XL "SO" series and Newton "SO" series (windowless, open-front, direct-detection CCDs, 13-26 µm pixels, up to 4.2 MP, peak QE ~95%, CF152 UHV flange) and Princeton Instruments PIXIS-XO / PI-MTE (back-illuminated, no AR coating, 10 eV-30 keV quoted, rotatable ConFlat flange, up to 2048×2048) are catalog alternatives to the greateyes ALEX-s family in the same windowless direct-detection class. Note the visually-similar Andor Newton **BEX2-DD** variant has a fused-silica *window* and is NOT usable here — only the "SO" line is windowless.
   - Not yet actionable as code; this is groundwork for a future hardware-survey writeup, not a `Grating`/CCD implementation choice.
7. **(done, #1081)** Lab-built flat-field spectrometer: ALEX-s 2k512 BI UV1 response with datasheet constants (`AlexS`, `alexs_response`, `PlanarDetector.alexs_2k512`, `response.kind = "alexs"` lowering), absolute grating efficiency and flat-field focal plane from the private catalog (`Grating.from_catalog`), and the analytic slit kernel with along-slit capture and 2D ADU image (`grating_image`). Ledger: `alexs-adu-conversion`, `flat-field-focal-plane`, `grating-slit-image-width`. Recipe: [model a private flat-field spectrometer](../../guides/flat-field-spectrometer.md).
8. Digitized ALEX-s QE curve in the private catalog (user); finite slit and per-row spectra (#1076); grating CLI (#1083).
9. Optionally expose grating parameters as `Sweep` knobs and add a `plots/` panel; validate against a measured grating-spectrometer dataset when available (data-dependent, like P1's other gated items).

Steps 1–5 and 7 are done; the flat-field path (step 7) reads its vendor data from the private catalog and remains a basic, non-ray-traced model. The digitized ALEX-s QE curve, the finite slit (#1076) and a measured-dataset comparison are next.
