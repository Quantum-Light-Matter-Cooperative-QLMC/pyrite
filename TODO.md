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

**Remaining — real CCD + grating + geometry buildout (promoted to P1):**

1. Grating reflectivity / groove efficiency `R(E,α)` (xraydb `f',f''` → δ,β →
   Fresnel reflectivity at grazing angle, × groove efficiency) for the McPherson
   251MX's four gratings (120/300/1200/2400 g/mm, ~87° incidence / <3° grazing,
   ~25 mm flat-field length) — so the dispersed profile carries real throughput,
   not just position.
2. A **simple** CCD forward model first: bin detected photons by the polar angle
   each pixel subtends (no QE/charge-sharing structure yet). Ground it in the
   greateyes ALEX-s 1k256 (1024×255 px, 26 µm px, 26.6×6.7 mm) and ALEX-s 2k512
   (2048×515 px, 13.5 µm px, 27.6×6.9 mm) formats.
3. Once that end-to-end path works, replace the simple binning with a more
   physical CCD model (QE(E), charge sharing, energy resolution) beside the
   existing Timepix / Eagle XO detector forward models.
4. Survey other CCD/grating options in the broader ~10 eV-4 keV band (the McPherson
   251MX tops out near 1.24 keV at 2400 g/mm), preferring broadband coverage.
   Parallelize independent lookups/implementation steps across agents.
5. Validate vs measured data when available (data-dependent, like P1's other
   gated items).

Design + phased plan: [`docs/grazing-grating.md`](docs/grazing-grating.md).
