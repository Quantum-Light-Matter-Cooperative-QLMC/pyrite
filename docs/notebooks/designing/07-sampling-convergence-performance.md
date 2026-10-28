# Sampling, convergence and performance

[![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/07-sampling-convergence-performance.py)

Compare fixed numerical budgets and a bounded adaptive policy using one PXR/CBS
band yield, and observe the cost of increasing sampled electron histories.

This page displays **captured outputs**; changing parameters requires Molab or a
local marimo session. The static page does not run transport, download tables or
need a GPU. The plotted changes are **not** independent Monte Carlo error bars.

```{raw} html
<iframe src="../../_static/notebooks/07-sampling-convergence-performance.html"
  title="Sampling, convergence and performance: captured notebook"
  loading="lazy" style="width:100%; height:1750px; border:1px solid var(--pst-color-border); border-radius:6px;">
</iframe>
```

<a href="../../_static/notebooks/07-sampling-convergence-performance.html">Open captured notebook full screen</a>.
{download}`Download canonical notebook <../07-sampling-convergence-performance.py>`.


## Continue learning

Next: [Parameter sweeps](08-parameter-sweeps.md).
See [precision and tolerances](../../computation/precision-and-tolerances.md)
for stopping controls and the [configuration API](../../reference/python-api/config.md)
for their Python interfaces. [Statistical methods](../../computation/statistical-methods.md)
explain uncertainty; [adaptive-stopping evidence](../../validation/beam-transport/adaptive-sample-size-stopping.md)
records the estimator's assumptions and checks.

## Canonical source

```{literalinclude} ../07-sampling-convergence-performance.py
:language: python
```
