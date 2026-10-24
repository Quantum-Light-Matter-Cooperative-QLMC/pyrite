# Radiation diagnostics

Install the apps extra, then launch the Plotly optical diagnostic:

```bash
uv run pyrite app optical-diagnostics launch
uv run pyrite app optical-diagnostics launch --smoke
uv run pyrite app optical-diagnostics export
```

The initial page performs no numerical calculation. Select a view and click its
compute button. Cached numerical results are reused for repeated settings.
Static HTML contains the controls and explanation; use the live app to compute.
Tunnel mode uses port 2725. The app requires the finite-source and instrument
APIs introduced under #616/#617.

## Optical source and collection

The bounded reference is a 20 µm PEC disc (TR), or a 5 µm aperture in that disc
(DR), with a normal straight electron at β=0.95. The source band is 0.8–1.2 eV,
the FFT grid 128×128 at 0.5 µm pitch, and the collection radius 10 µm. Distance,
physical charge and Gaussian RMS duration are adjustable. Two identical,
offset-free numerical histories evaluate an analytic temporal-only Gaussian
law; they are not two physical bunch electrons or transported trajectories.

The spectrum is photons/(eV physical pulse); field intensity/phase maps show a
single electron's spectral electric field in V·s/m. A synthetic flat 25%
absorptance and 10¹² V/J pulse gain produce J/pulse and V. The 1 kHz repetition
rate scales average W only. Neither detector prediction nor inference applies
physical population a second time. This is not a measured calibration.

Use **Convergence** to compare independent pitch/window refinements, hemisphere
FFT bias and non-periodic direct-Green collection. A convergence flag covers
only those tests at 1 eV. Energy integration, angular/aperture integration,
source validity and spectral/delay convergence require separate studies.
Physical optics requires the screen/hole/edge scales to be much larger than the
wavelength; the app does not certify the approximations by displaying a plot.

## Interferograms and conditional inference

This is a separate synthetic far-IR exercise, not a continuation of the visible
PEC example. The two-port ideal Martin–Puplett uses 45° linear polarization,
a declared RMS voltage error and synthetic flat photon templates. Mechanical
mirror displacement is half the optical path difference. The detector response
and source provenance travel with the result.

The fit plots the existing profiled Δχ² objective and reports Gaussian RMS duration, conditional 95% interval, open endpoints,
multimodality, band-edge temporal factors and missing-band assumptions. A 200 fs
turnover is near 0.0033 eV. The high-band-only option demonstrates loss of
identifiability. No systematic covariance, measured response uncertainty,
nonparametric phase retrieval or unique bunch-profile reconstruction is implied.
Refine delay step/range and spectral integration before interpreting a fit.

**Saved outputs** reads existing planar-source, interferogram or Gaussian-fit
NPZ files through their owning loaders. It plots arrays and provenance without
running transport. Saved fit axes use scan index because the fit artifact does
not carry mechanical mirror coordinates. See the
[optical model and instrument reference](../physics/radiation-physics/optical-wavefront-and-detectors.md)
for artifact creation, APIs, assumptions and current validation status.

## Checkpoint components and DTR

Analysis and case comparison offer **TR/CTR** selection when plotting. Absent
components contribute zero; the radiation panel reports availability by case.
Deselect other components to isolate TR. Existing totals retain TR by default.
Plotly switches requested log-y plots to linear when signed values are present,
and identifies that change in the title. Arrays are never clipped.

The radiation panel integrates stored source densities over the line band,
without current, charge or acceptance scaling. It reports units, spectrum basis,
physical population, sampled histories and negative TR bins. The reusable
`pyrite.results.radiation.radiation_band_metrics` also accepts an explicit
in-grid band with interpolated endpoints. Existing map/ranking metrics include
all stored additive line components; compatibility names such as `coherent_flux`
do not identify pure coherent PXR/CBS.

For experimental two-beam cases, **Line** means signed joint-field correction,
**TR/CTR** means the TR baseline, and their sum is the joint total. Removing the
baseline is not a DTR-off counterfactual. The model status remains unverified.

The TR notebook (`pyrite app transition-radiation launch`) also has an explicitly
triggered Plotly DTR reference view. It displays Born component/cross-term
spectra, optionally with coupled two-beam source rows. All curves reuse the
maintained deterministic reference. See [experimental DTR](../physics/radiation-physics/diffracted-transition-radiation.md).

The validation app has a **Saved radiation evidence** section. It reads saved
omission and optical hardware records from a checkout, or `reference.json` from
a DTR export directory. Missing/malformed files remain visible. Reading a saved
pass does not promote a ledger claim; CUDA parity is not a physical anchor.
