# Parameter sweeps

[![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/08-parameter-sweeps.py)

Map PXR/CBS band yield against beam energy and crystal thickness.
These are **captured outputs**; edit and recompute in Molab or local marimo.

```{raw} html
<iframe src="../../_static/notebooks/08-parameter-sweeps.html"
  title="Parameter sweeps: captured notebook"
  loading="lazy" style="width:100%; height:1250px; border:1px solid var(--pst-color-border); border-radius:6px;">
</iframe>
```

<a href="../../_static/notebooks/08-parameter-sweeps.html">Open captured notebook full screen</a>.
{download}`Download canonical notebook <../08-parameter-sweeps.py>`.

## Continue learning

Revisit [sampling](07-sampling-convergence-performance.md) before comparing
nearby yields. For reusable campaign settings, use
[Campaign configuration](../../guides/sweep-profiles.md); the
[scene and sweep API](../../reference/python-api/core.md) defines parameter studies.
[Spectral observables](../../physics/radiation-physics/spectral-observables.md)
explain band-yield normalization. Review the selected models'
[validation status](../../validation/status-summary.md) before interpreting a response surface.

## Canonical source

```{literalinclude} ../08-parameter-sweeps.py
:language: python
```
