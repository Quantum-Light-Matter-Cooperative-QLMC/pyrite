# TODO — feature/grazing-grating

This branch carries one backlog item; the full triaged backlog lives on `main`.

## Grazing-incidence soft X-ray diffraction grating (P1 — promoted from P3 #2)

*Dispersion scaffold IMPLEMENTED.*

With an Eagle XO or an Alex detector, like those from Ultrafast Innovations — a new
experimental modality: disperse the soft-X-ray spectrum with a grazing-incidence reflection
grating and read the spatial image, for spectral resolution beyond the ~130 eV EDS width.

`src/cxr_mc/grating.py` implements the dispersion geometry (grating equation
`sinβ = mλ/d − sinα`, angular dispersion, flux-conserving `disperse_spectrum`, pixel-limited
resolving power), cross-checked in `tests/test_grating.py`.

`src/cxr_mc/grating.py` also now implements grazing-incidence Fresnel
reflectivity of the coating (`Grating.reflectivity`, `Grating.throughput`,
`disperse_spectrum(..., weight_by_throughput=True)`), cross-checked in
`tests/test_grating.py` and `tests/test_crystallography.py`. Groove-profile
diffraction efficiency (`Grating.groove_efficiency`) is still a placeholder
scalar, not a real model.

`src/cxr_mc/grating.py` also now implements a simple CCD pixel grid
(`ALEXS_SENSORS`, `SimpleCCD`, `bin_to_pixels`) sized to the greateyes ALEX-s
1k256/2k512 formats: geometry-only rebinning of a dispersed profile onto fixed
pixels, no QE/charge-sharing/energy-resolution yet.

`src/cxr_mc/grating.py` also now implements the combined forward-model entry
`detected_image` (`mc_spectrum` output + `Grating` + `SimpleCCD` → dispersed,
detected image, counts vs pixel) — pure composition of the reflectivity and
pixel-binning pieces above, no new physics.

**Remaining — real CCD + grating + geometry buildout (promoted to P1):**

1. **(done)** Grating reflectivity `R(E,α)` (`crystallography.optical_constants`:
   `f',f''` → δ,β → Fresnel reflectivity at grazing angle, `Grating.reflectivity`)
   wired into `disperse_spectrum` so the dispersed profile can carry real
   throughput, not just position. Groove-profile efficiency (per McPherson
   251MX grating: 120/300/1200/2400 g/mm, ~87° incidence / <3° grazing,
   ~25 mm flat-field length) is still a placeholder scalar
   (`Grating.groove_efficiency`) pending a scalar/RCWA treatment.
2. **(done)** A **simple** CCD forward model: bin detected photons by the polar
   angle each pixel subtends (no QE/charge-sharing structure yet). Ground it in
   the greateyes ALEX-s 1k256 (1024×255 px, 26 µm px, 26.6×6.7 mm) and ALEX-s
   2k512 (2048×515 px, 13.5 µm px, 27.6×6.9 mm) formats
   (`ALEXS_SENSORS`/`SimpleCCD`/`bin_to_pixels`).
3. **(done)** A combined forward-model entry (`detected_image`) chaining the
   grating reflectivity + simple CCD binning above, so an `mc_spectrum` output
   can be turned into a detected image in one call, slotting in beside the
   existing Timepix / Eagle XO detector forward models.
4. Now that end-to-end path works, replace the simple binning with a more
   physical CCD model (QE(E), charge sharing, energy resolution).
5. **(preliminary survey done)** other CCD/grating options in the broader
   ~10 eV-4 keV band (the McPherson 251MX tops out near 1.24 keV at 2400 g/mm),
   preferring broadband coverage — see `docs/grazing-grating.md` phased-plan
   step 6 for findings (Horiba PGM200, Bestec custom builds, a published
   multilayer-coated grating reaching 3.3 keV; Andor "SO"-series and Princeton
   Instruments PIXIS-XO/PI-MTE as ALEX-s-class detector alternatives). Not yet
   turned into code/a hardware choice.
6. Validate vs measured data when available (data-dependent, like P1's other
   gated items).

Design + phased plan: [`docs/grazing-grating.md`](docs/grazing-grating.md).
