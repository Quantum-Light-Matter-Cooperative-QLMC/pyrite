# Grazing-incidence soft X-ray grating spectrometer

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

`src/cxr_mc/grating.py` implements the **dispersion geometry** (the concrete,
testable physics) as a standalone forward model — nothing in the sweep/plot
pipeline imports it yet. Cross-checked in `tests/test_grating.py`.

| provided | meaning |
| --- | --- |
| `wavelength_angstrom(E)` / `groove_spacing_angstrom(ρ)` | unit conversions |
| `Grating(groove_density_per_mm, alpha_rad, order)` | a grating in a fixed mount |
| `Grating.diffraction_angle_rad(E)` | the grating equation `sinβ = mλ/d − sinα` (NaN if no propagating order) |
| `Grating.angular_dispersion_rad_per_angstrom(E)` | `dβ/dλ = m/(d cosβ)` |
| `disperse_spectrum(E, spec, grating, distance_mm)` | **flux-conserving** map of a spectrum to detector position (`∫I dx = ∫spec dE`) |
| `resolving_power(E, grating, distance_mm, pixel_mm)` | pixel-limited `λ/Δλ` |

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

## Hardware targets

Real instrument, not a hypothetical, for the reflectivity/CCD phases below:

- **Grating: McPherson 251MX** flat-field SXR/EUV spectrograph. Four interchangeable
  gratings — 120, 300, 1200, 2400 g/mm — covering roughly 6 eV-1.24 keV in total
  (each grating's own band is narrower); fixed ~87° incidence from normal (< 3°
  grazing), ~25 mm flat focal-plane length. This caps out well below the ~4 keV
  target range, so it is the *first* grating, not the only one — step 4 below
  surveys broadband options to extend upward.
- **Detector: greateyes ALEX-s** deep-cooled CCD, two formats:
  - `1k256`: 1024×255 px, 26 µm pixels, 26.6×6.7 mm active area.
  - `2k512`: 2048×515 px, 13.5 µm pixels, 27.6×6.9 mm active area.
  Same family as the existing `eaglexo_response.py` (Raptor Eagle XO) model —
  windowless, back-illuminated, direct-detection CCD — so the QE/solid-angle
  pattern there is the template once real datasheet numbers are digitized.

## What is NOT modelled yet (and why)

This is *dispersion geometry only*. Deliberately out of scope for the scaffold:

- **Grating reflectivity / groove efficiency** vs energy and angle — needs the
  coating optical constants and a groove-profile efficiency model (e.g. a scalar
  or rigorous-coupled-wave treatment). This sets the absolute throughput and the
  usable band, so it is the first thing to add before any flux comparison.
- **Aberrations / focusing** (spherical or VLS gratings, Rowland circle) — the
  flat-detector `x = L tan(β − β_ref)` map ignores defocus and coma.
- **Source size / beam divergence** — a real line image is the convolution of the
  dispersion with the source spot and slit; the pixel-limited `resolving_power`
  is an upper bound.

## Phased plan to a real modality

1. **(done)** Dispersion geometry + resolving power, tested.
2. Reflectivity/efficiency `R(E, α)` from coating optical constants (xraydb can
   supply `f', f''` → δ, β → Fresnel reflectivity at grazing angle) × a groove
   efficiency factor, parameterized per McPherson 251MX grating (120/300/1200/2400
   g/mm); multiply into `disperse_spectrum`.
3. A **simple** CCD forward model: bin detected photons by the polar angle each
   pixel subtends (no QE/charge-sharing structure yet), sized to the ALEX-s
   1k256/2k512 pixel formats above. This is deliberately crude — it exists to get
   the full grating→image pipeline working end-to-end before adding detector
   physics.
4. A forward-model entry that takes the model's `mc_spectrum` output + a
   `Grating` + the simple CCD from step 3 and returns the **dispersed, detected
   image** (counts vs pixel), so it slots in beside the Timepix / Eagle XO models.
5. Replace the simple CCD with a more physical model (QE(E), charge sharing,
   energy resolution) once step 4's pipeline is validated structurally — same
   upgrade path `eaglexo_response.py` took from geometry-only to a digitized QE
   curve.
6. Survey CCD/grating options beyond the McPherson 251MX + ALEX-s pairing for the
   broader ~10 eV-4 keV band (the 251MX's 2400 g/mm grating tops out near
   1.24 keV), preferring broadband coverage. Independent lookups/implementation
   here parallelize well across agents.
7. Optionally expose grating parameters as `Sweep` knobs and add a `plots.py`
   panel; validate against a measured grating-spectrometer dataset when available
   (data-dependent, like P1's other gated items).

Reflectivity (step 2) is the next high-value piece — until then the dispersed
profile is a *relative* spectrum, correct in position but not in throughput.
