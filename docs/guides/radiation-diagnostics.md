# Radiation diagnostics

Explore when a far-IR interferogram constrains a Gaussian bunch duration.
These are **synthetic model outputs**, using the optical diagnostics app's
ideal instrument and response; they are not measured data or a validated
bunch-length diagnostic.

## Interferograms and conditional inference

From a source checkout:

```bash
uv sync --extra apps
uv run pyrite app optical-diagnostics launch
```

For an installed distribution, install `pyrite-mc[apps]`, then omit `uv run`.
Select **Instrument and inference** and click
**Compute interferogram and conditional fit** with the defaults:
**200 fs RMS**, **0.0005–0.025 eV**, **±1.2 mm mirror travel**, **61 nodes** and
**15 µV declared RMS error**. This CPU exercise needs no transport tables or GPU.

```{eval-rst}
.. plot::
   :filename-prefix: radiation-diagnostics-signal
   :alt: Two synthetic isolated-pulse port voltages and their fitted curves versus mechanical mirror displacement, with signed noise-free residuals below.
   :caption: Default two-port expectation (markers), fitted signal (lines) and signed residuals. Optical path difference is twice the mechanical displacement.

   from _tutorials import radiation_diagnostics
   radiation_diagnostics()
```

The input is a noise-free expectation with declared errors; no random noise was
added. Synthetic flat photon templates, 45° linear polarization, 25% absorptance
and a 10¹² V/J pulse gain set the signal. Population and acceptance are already
in the templates; no further charge scaling applies.

Change **Detector band low** to **0.02 eV** and submit again:

```{eval-rst}
.. plot::
   :filename-prefix: radiation-diagnostics-profile
   :alt: Conditional chi-square duration profiles for bands containing or excluding the 200 femtosecond Gaussian turnover, with the 95 percent threshold.
   :caption: App profile samples for two bands; lines join display nodes. The intervals below come from the fitter, rather than these coarse display crossings.

   from _tutorials import radiation_diagnostics
   radiation_diagnostics(view="profile")
```

| Photon band | Fitted RMS duration | Conditional 95% interval |
|---|---:|---|
| 0.0005–0.025 eV | 200.00 fs | 198.63–201.38 fs |
| 0.02–0.025 eV | 200.46 fs | Both endpoints open within 20–3000 fs search |

The 200 fs turnover is near **0.0033 eV**. Removing it from the band leaves the
duration unidentified: the profile is multimodal with open endpoints, even
though a numerical minimum still exists. The fit profiles a source scale with calibration fixed. Its interval assumes independent
Gaussian voltage errors; it excludes systematic covariance and response
uncertainty and implies no unique bunch-profile or phase reconstruction.
Refine spectral integration and delay step/range independently before interpreting
a fit. See the [model and validation limits](../physics/radiation-physics/optical-wavefront-and-detectors.md).

## Saved outputs

```bash
uv run pyrite app optical-diagnostics export --stem optical_diagnostics
```

Open `pyrite-output/results/optical_diagnostics.html` for the initial controls
and explanation. Computation requires the live app. The
{download}`cached guide output <../examples/data/radiation_diagnostics.json>`
records both scans, fits, assumptions and instrument/response hashes;
documentation builds only render these arrays. The maintained notebook is
{download}`optical_diagnostics_app.py <../../src/pyrite/apps/optical_diagnostics_app.py>`.
Existing NPZ inspection and launch modes are described in the
[app reference](../reference/optical-diagnostics-app.md#saved-outputs).

## Optical source and collection

The separate **Source and collection** exercise computes finite PEC TR/DR fields
and collected spectra. It does not supply the synthetic templates above.
See [settings, units and convergence limits](../reference/optical-diagnostics-app.md#optical-source-and-collection).

## Cherenkov cone

The separate **Cherenkov cone** view plots an internal bulk source in a lossless
medium. It does not predict escaped light or detector counts; see
[the source restrictions](../reference/optical-diagnostics-app.md#cherenkov-cone).

## Checkpoint components and DTR

For saved TR/CTR selection, signed source-band metrics and the **experimental,
unverified** DTR view, see the
[TR/CTR app reference](../reference/tr-ctr-app.md#checkpoint-components-and-dtr).
