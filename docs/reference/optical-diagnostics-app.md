# Optical diagnostics app controls and artifacts

Use the [radiation diagnostics guide](../guides/radiation-diagnostics.md) for a
worked far-IR example. Install the apps extra, then launch the Plotly optical
diagnostic:

```bash
uv run pyrite app optical-diagnostics launch
uv run pyrite app optical-diagnostics launch --smoke
uv run pyrite app optical-diagnostics export
```

The initial page performs no numerical calculation. Select a view and click its
compute button. Cached numerical results are reused for repeated settings.
Static HTML contains the controls and explanation; use the live app to compute.
Tunnel mode uses port 2725.

## Optical source and collection

The default reference is a 20 µm-radius PEC disc (TR; a nonzero hole radius gives an
aperture, DR) with a normal straight electron at β=0.95. The source band is
0.8–1.2 eV in 9 points, the FFT grid 128×128 at 0.5 µm pitch, and the collection
radius 10 µm at 100 µm. Every one of these, plus physical charge and Gaussian RMS
duration, is a form field bounded only by physical validation and a 4096² FFT
memory limit; the app reports the resulting window (samples × pitch), which
must exceed the screen. *Field device* offers CUDA on a
`pyrite app optical-diagnostics launch --backend cuda` session. Two identical,
offset-free numerical histories evaluate an analytic temporal-only Gaussian
law; they are not two physical bunch electrons or transported trajectories.

The spectrum is photons/(eV physical pulse); field intensity/phase maps show a
single electron's spectral electric field in V·s/m. A synthetic flat 25%
absorptance and 10¹² V/J pulse gain produce J/pulse and V. The 1 kHz repetition
rate scales average W only. Neither detector prediction nor inference applies
physical population a second time. This is not a measured calibration.

Use **Convergence** to compare independent pitch/window refinements, hemisphere
FFT bias and non-periodic direct-Green collection, at the submitted settings.
A convergence flag covers only those tests at the band's geometric-mean energy. Energy integration, angular/aperture integration,
source validity and spectral/delay convergence require separate studies.
Physical optics requires the screen/hole/edge scales to be much larger than the
wavelength; the app does not certify the approximations by displaying a plot.

## Interferograms and conditional inference

This is a separate synthetic far-IR exercise, not a continuation of the visible
PEC example. The two-port ideal Martin–Puplett uses 45° linear polarization,
a declared RMS voltage error and synthetic flat photon templates. Mechanical
mirror displacement is half the optical path difference. The detector response
and source provenance travel with the result.

The form sets the synthetic RMS duration, detector band, mirror travel, node
count and declared RMS error. The fit plots the existing profiled Δχ² objective
and tabulates Gaussian RMS duration, conditional 95% interval, open endpoints,
multimodality and band-edge temporal factors; assumptions and provenance sit in
a collapsed section. A 200 fs turnover is near 0.0033 eV; a band above it
demonstrates loss of identifiability. No systematic covariance, measured response uncertainty,
nonparametric phase retrieval or unique bunch-profile reconstruction is implied.
Refine delay step/range and spectral integration before interpreting a fit.

## Cherenkov cone

**Cherenkov cone** maps the bulk Frank–Tamm source
(`frank-tamm-homogeneous-dielectric`) of one electron in an unbounded,
lossless medium of constant refractive index: polar angle about the velocity
against electron kinetic energy, in bin-averaged photons/m/eV/sr. Sliders for
the index, photon energy, kinetic range and polar binning redraw live and
report the kinetic threshold and the saturated cone angle arccos(1/n). It is an
internal yield only: no interfaces, escape, total internal reflection,
multiple scattering or slowing down.

## Saved outputs

**Saved outputs** reads existing planar-source, interferogram or Gaussian-fit
NPZ files through their owning loaders. It plots arrays and provenance without
running transport. Saved fit axes use scan index because the fit artifact does
not carry mechanical mirror coordinates. See the
[optical model and instrument reference](../physics/radiation-physics/optical-wavefront-and-detectors.md)
for artifact creation, APIs, assumptions and current validation status.
