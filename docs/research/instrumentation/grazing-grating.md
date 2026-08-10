# Grazing-incidence soft X-ray grating spectrometer

**Status:** Exploratory research model; not wired into the production pipeline.

A measurement modality promoted to TODO P1 (was P3 #2): instead of reading the
PXR+CBS spectrum as energy-vs-counts at a fixed take-off angle (EDS/Timepix),
**disperse** it with a reflection grating at grazing incidence and read the
resulting *spatial* image on a position-sensitive detector (Raptor Eagle XO CCD,
or an "Alex"-type detector from Ultrafast Innovations). The grating's dispersion
gives spectral resolution set by geometry and pixel size rather than by the
detector's intrinsic energy resolution — potentially far better than the ~130 eV
EDS line width that dominates the soft-X-ray band today.

---

## Status

`src/cxr_mc/detectors/grating.py` implements the **dispersion geometry**, the
**grazing-incidence Fresnel reflectivity** of the grating's coating, a
**simple CCD pixel grid** (geometry-only rebinning), a **combined
forward-model entry** (`detected_image`, chaining the two above) that turns an
`mc_spectrum` output directly into a detected image, and now a **physical CCD
response** (`qe_absorption`, `charge_cloud_sigma_um`, `energy_fwhm_eV`,
`detected_image_physical`) with real (if device-constant-placeholder) QE,
charge-sharing, and energy-resolution physics — all as a standalone forward
model — nothing in the sweep/plot pipeline imports it yet. Cross-checked in
`tests/detectors/test_grating.py`.

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
| `detected_image(E_grid_eV, spec, grating, ccd, distance_mm, weight_by_throughput=True, ...)` | combined forward-model entry: `mc_spectrum` output → dispersed, detected image (counts vs pixel), chaining `disperse_spectrum` + `bin_to_pixels` |
| `qe_absorption(E, active_um=ACTIVE_SI_UM, peak=ENTRANCE_QE_PEAK)` | Beer-Lambert absorption-efficiency QE of the active silicon |
| `charge_cloud_sigma_um(E, active_um=..., v_dep=DEPLETION_VOLTAGE_V, temp_c=OPERATING_TEMP_C)` | lateral charge-cloud RMS spread [um] from drift-diffusion |
| `energy_fwhm_eV(E, n_pix=4, read_noise_e=READ_NOISE_E)` | Fano + read-noise single-photon energy resolution (photon-counting mode; not applied by default) |
| `detected_image_physical(E_grid_eV, spec, grating, ccd, distance_mm, ...)` | like `detected_image`, but weights by `qe_absorption` before binning and applies the `charge_cloud_sigma_um` Gaussian blur (at the detected spectrum's flux-weighted mean energy) after binning |

## Physics and conventions

Reflection grating equation, angles from the grating **normal**:

```
d (sin α + sin β) = m λ          →     sin β = m λ / d − sin α
```

- `α` = incidence angle, `β` = diffraction angle (from normal); `m = 0` is
  specular (`β = −α`).
- **Grazing** incidence (grazing angle `θ_g = 90° − α` of a few degrees) is a
  choice of `α`, not a different equation — it is what buys usable reflectivity
  for soft X-rays. In this regime first orders never run out of propagating
  solutions across 100 eV–2 keV (the NaN branch is only hit by pathological
  rulings; see the test).
- `λ = hc / E` with `hc = 12398.42 eV·Å` (`crystallography.HC_EV_ANG`).

Grazing-incidence Fresnel reflectivity of the coating, small-angle form:

```
n = 1 − δ − iβ                              (complex refractive index)
r(θ) = (θ − √(θ² − 2δ − 2iβ)) / (θ + √(θ² − 2δ − 2iβ))
R(θ) = |r(θ)|²
```

with `θ` the grazing angle (`Grating.grazing_angle_rad`) and `δ, β` from
`crystallography.optical_constants` (Henke/Chantler `f1 = Z+f'`, `f2`). Treated
as polarization-independent, the standard grazing-incidence approximation
(source, assumptions, and limiting cases are in `Grating.reflectivity`'s
docstring). `Grating.throughput(E) = reflectivity(E) × groove_efficiency`
combines this with the placeholder groove-efficiency scalar; pass
`weight_by_throughput=True` to `disperse_spectrum` to get a throughput-weighted
dispersed profile instead of a purely geometric one.

## Hardware targets

Real instrument, not a hypothetical, for the reflectivity/CCD phases below:

- **Grating: McPherson 251MX** flat-field SXR/EUV spectrograph. Four interchangeable
  gratings — 120, 300, 1200, 2400 g/mm — covering roughly 6 eV-1.24 keV in total
  (each grating's own band is narrower); fixed ~87° incidence from normal (< 3°
  grazing), ~25 mm flat focal-plane length. This caps out well below the ~4 keV
  target range, so it is the *first* grating, not the only one — step 6 below
  surveys broadband options to extend upward.
  **Coating: gold**, confirmed for all four groove densities from McPherson's own
  product materials (press release + product page), matching `Grating`'s default
  `coating="Au"` — no longer just a placeholder guess ([McPherson 251MX product
  page](https://www.mcphersoninc.com/spectrometers/xuhvvuvuv/model251mx.html);
  [flat-field grating press release](https://mcphersoninc.com/pressreleases/flatFieldGrating120.html)).
  A related McPherson product line describes **laminar (square-wave, not blazed)
  groove profiles** and a steeper ~1.5° grazing option for higher-energy reach —
  useful groundwork for a future groove-efficiency model (step in "What is NOT
  modelled yet"), but **not confirmed** to be the exact 251MX groove profile, so
  treat as McPherson-family-typical rather than hardware-verified. No quantitative
  efficiency curve for the 251MX was found publicly — absolute throughput numbers
  remain unconfirmed pending a datasheet/measurement.
- **Detector: greateyes ALEX-s** deep-cooled CCD, two formats:
  - `1k256`: 1024×255 px, 26 µm pixels, 26.6×6.7 mm active area.
  - `2k512`: 2048×515 px, 13.5 µm pixels, 27.6×6.9 mm active area.
  Same family as the existing `eaglexo_response.py` (Raptor Eagle XO) model —
  windowless, back-illuminated, direct-detection CCD — so the QE/solid-angle
  pattern there is the template once real datasheet numbers are digitized.

## What is NOT modelled yet (and why)

Dispersion geometry, coating reflectivity, simple pixel binning, and now a
closed-form physical CCD response (QE, charge sharing, energy resolution) are
modelled; deliberately still out of scope:

- **Groove-profile diffraction efficiency** vs energy and angle — the coating's
  Fresnel reflectivity is modelled (`Grating.reflectivity`), but the
  groove-profile efficiency factor is a placeholder scalar
  (`Grating.groove_efficiency`, default 1.0), not a real model. That needs a
  scalar or rigorous-coupled-wave (RCWA) treatment of the ruled profile.
- **Measured ALEX-s QE/noise curves** — `qe_absorption`/`charge_cloud_sigma_um`/
  `energy_fwhm_eV` (step 5) are real closed-form physics (Beer-Lambert
  absorption, Einstein-relation drift-diffusion, Fano + read noise), but their
  device constants (`ACTIVE_SI_UM`, `DEPLETION_VOLTAGE_V`, `OPERATING_TEMP_C`,
  `READ_NOISE_E`, `ENTRANCE_QE_PEAK`) are placeholders — no public greateyes
  ALEX-s datasheet was found (unlike the Eagle XO's digitized
  `eaglexo_qe.csv`). `detected_image_physical`'s charge-cloud blur also uses a
  single characteristic sigma at the frame's flux-weighted mean energy, not a
  per-photon energy-dependent blur, and `energy_fwhm_eV` is exposed but not
  applied anywhere (this module models the ALEX-s as an imaging, not
  photon-counting, detector). No Poisson acquisition-noise draw yet either
  (cf. `eaglexo_response.poisson_counts`).
- **Aberrations / focusing** (spherical or VLS gratings, Rowland circle) — the
  flat-detector `x = L tan(β − β_ref)` map ignores defocus and coma.
- **Source size / beam divergence** — a real line image is the convolution of the
  dispersion with the source spot and slit; the pixel-limited `resolving_power`
  is an upper bound.

## Phased plan to a real modality

1. **(done)** Dispersion geometry + resolving power, tested.
2. **(done)** Coating Fresnel reflectivity `R(E)` from optical constants
   (`crystallography.optical_constants`: `f', f''` → δ, β → Fresnel reflectivity
   at grazing angle, `Grating.reflectivity`), combined with a placeholder groove-
   efficiency scalar into `Grating.throughput`, wired into `disperse_spectrum` via
   `weight_by_throughput=True`. The groove-profile efficiency itself (vs.
   McPherson 251MX grating: 120/300/1200/2400 g/mm) is still a placeholder — see
   "What is NOT modelled yet".
3. **(done)** A **simple** CCD forward model: bin detected photons by the polar
   angle each pixel subtends (no QE/charge-sharing structure yet), sized to the
   ALEX-s 1k256/2k512 pixel formats above (`ALEXS_SENSORS`, `SimpleCCD`,
   `bin_to_pixels`). This is deliberately crude — it exists to get the full
   grating→image pipeline working end-to-end before adding detector physics.
4. **(done)** A forward-model entry (`detected_image`) that takes the model's
   `mc_spectrum` output + a `Grating` + the simple CCD from step 3 and returns
   the **dispersed, detected image** (counts vs pixel), so it slots in beside
   the Timepix / Eagle XO models. Pure composition of steps 2 and 3 (chains
   `disperse_spectrum` then `bin_to_pixels`) — no new physics, so no new ledger
   row.
5. **(done)** Replace the simple CCD with a more physical model: Beer-Lambert
   absorption QE (`qe_absorption`), Einstein-relation drift-diffusion charge
   sharing (`charge_cloud_sigma_um`), and Fano+read-noise energy resolution
   (`energy_fwhm_eV`), composed into `detected_image_physical` — same upgrade
   path `eaglexo_response.py` took, except no digitized greateyes ALEX-s
   datasheet exists (unlike the Eagle XO's `eaglexo_qe.csv`), so this is
   closed-form physics with `### FILL IN` device constants rather than a
   measured curve — see "What is NOT modelled yet" and the
   `alexs-qe-absorption`/`alexs-charge-diffusion` ledger rows.
6. **(preliminary survey done, no code yet)** Survey CCD/grating options beyond
   the McPherson 251MX + ALEX-s pairing for the broader ~10 eV-4 keV band (the
   251MX's 2400 g/mm grating tops out near 1.24 keV), preferring broadband
   coverage. Findings so far (web/literature search, not yet cross-checked
   against primary datasheets in all cases — flagged where uncertain):
   - **Grating spectrometers**: Horiba/Jobin-Yvon PGM200 (grazing-incidence
     flat-field, interchangeable 1800/450 g/mm gratings, "varied groove depth"
     technology for a broadened single-grating band — exact range unconfirmed);
     Bestec GmbH custom plane-/spherical-grating monochromators (soft-to-hard
     X-ray, beamline-grade, custom builds not a catalog product); a published
     research instrument (Imazono et al., *Appl. Opt.* 57(27):7770, 2018,
     [PubMed](https://pubmed.ncbi.nlm.nih.gov/30462040/)) reaches 0.9-3.3 keV
     using an **aperiodic Ni/C multilayer** grating coating rather than a simple
     metal film — direct evidence that a multilayer coating (not gold) plus a
     shallower grazing angle is likely required to push toward the 4 keV goal.
   - **Detectors**: Andor iKon-M/L/XL "SO" series and Newton "SO" series
     (windowless, open-front, direct-detection CCDs, 13-26 µm pixels, up to
     4.2 MP, peak QE ~95%, CF152 UHV flange) and Princeton Instruments
     PIXIS-XO / PI-MTE (back-illuminated, no AR coating, 10 eV-30 keV quoted,
     rotatable ConFlat flange, up to 2048×2048) are catalog alternatives to the
     greateyes ALEX-s family in the same windowless direct-detection class. Note
     the visually-similar Andor Newton **BEX2-DD** variant has a fused-silica
     *window* and is NOT usable here — only the "SO" line is windowless.
   - Not yet actionable as code; this is groundwork for a future hardware-survey
     writeup, not a `Grating`/CCD implementation choice.
7. Optionally expose grating parameters as `Sweep` knobs and add a `plots/`
   panel; validate against a measured grating-spectrometer dataset when available
   (data-dependent, like P1's other gated items).

Coating reflectivity (step 2), simple pixel binning (step 3), the combined
`mc_spectrum` → `Grating` → `SimpleCCD` forward-model entry `detected_image`
(step 4), and a closed-form physical CCD response (`detected_image_physical`,
step 5) are done; groove-profile efficiency remains a placeholder scalar, and
the physical CCD's device constants (active thickness, depletion voltage,
temperature, read noise) are `### FILL IN` placeholders pending a real
greateyes datasheet, so the throughput- and QE-weighted, charge-cloud-blurred
image is closer to real flux/PSF than the purely geometric one but still not
final until a groove-efficiency model and real ALEX-s hardware numbers land.
Step 6 (the broadband hardware survey) is next to turn into code/a choice.
