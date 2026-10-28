# From X-rays to detector counts

[![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/06-detector-counts.py)

Configure an ideal pixel detector and compare expected counts with a seeded
Poisson realization from the same source calculation.


This page displays **captured outputs**; changing parameters requires Molab or a
local marimo session. The static page does not run PyRITE, download tables, or
need a GPU. The images are simulated observations of an ideal counter.

```{raw} html
<iframe src="../../_static/notebooks/06-detector-counts.html"
  title="From X-rays to detector counts: captured notebook"
  loading="lazy" style="width:100%; height:1750px; border:1px solid var(--pst-color-border); border-radius:6px;">
</iframe>
```

<a href="../../_static/notebooks/06-detector-counts.html">Open captured notebook full screen</a>.
{download}`Download canonical notebook <../06-detector-counts.py>`.


## Continue learning

Next: [Sampling, convergence and performance](07-sampling-convergence-performance.md).
Use the [detector-observation how-to](../../guides/detector-observations.md)
for an acquisition recipe and the [observations API](../../reference/python-api/observations.md)
for interfaces. [Detector response](../../physics/detectors/detector-response.md)
explains the physical model; [pixel angular interpolation evidence](../../validation/detectors/pixel-angular-interpolation.md)
records checks on pixel reconstruction from angular tiles.

## Canonical source

```{literalinclude} ../06-detector-counts.py
:language: python
```
