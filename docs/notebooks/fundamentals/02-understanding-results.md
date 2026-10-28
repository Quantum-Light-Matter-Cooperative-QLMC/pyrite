# Understanding simulation results

[![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/02-understanding-results.py)

Convert native-grid source densities into energy-integrated yields and physical
photon rates, while keeping acquisition counts separate.


These are **captured outputs**. Recompute in Molab or locally after installing
the reference tables described in [notebook 01](01-first-spectrum.md).
The static page needs no Python kernel, GPU or simulation run.

```{raw} html
<iframe src="../../_static/notebooks/02-understanding-results.html"
  title="Understanding simulation results: captured notebook"
  loading="lazy" style="width:100%; height:1900px; border:1px solid var(--pst-color-border); border-radius:6px;">
</iframe>
```

<a href="../../_static/notebooks/02-understanding-results.html">Open captured notebook full screen</a>.
{download}`Download canonical notebook <../02-understanding-results.py>`.


## Continue learning

Next: [Predicting PXR line energies](03-pxr-line-energies.md).
Use [Working with results](../../guides/working-with-results.md) to inspect saved
calculations and the [results API](../../reference/python-api/results.md) for
analysis interfaces. [Spectral observables](../../physics/radiation-physics/spectral-observables.md)
and [energy-grid semantics](../../physics/radiation-physics/energy-grid-semantics.md)
explain the units and integration conventions used here.

## Canonical source

The source below is the exact notebook used for both captured output and Molab.

```{literalinclude} ../02-understanding-results.py
:language: python
```
