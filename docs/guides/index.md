# User Guides

Start with [installation and a first run](getting-started.md), then
[plot its saved results](working-with-results.md) or build a scene in Python.
Scientific tutorials show simulated outputs; each guide states its model and
sampling limits.

```{toctree}
:maxdepth: 1
:caption: Getting started

python-api-workflow
how-pyrite-works
```

```{toctree}
:maxdepth: 1
:caption: Simulations

configuration-cookbook
directional-source-heatmap
detector-observations
```

```{toctree}
:maxdepth: 1
:caption: Results

working-with-results
```

[Checkpoint and app contracts](../reference/results-and-apps.md) cover dataset
identity, comparison and archiving. [Saved electron histories](../reference/trajectory-workflows.md)
cover capture, replay and ParaView independently of spectral checkpoints.

```{toctree}
:maxdepth: 1
:caption: Advanced workflows

adding-materials
gpt-gdf-beams
running-on-a-cluster
optimization-studies
```

For execution tuning, use [performance profiling](../computation/performance-profile-analysis.md)
and [CUDA batch/chunk tuning](../computation/batch-chunk-tuning.md).
Scene construction and configuration contracts live in
[Reference](../reference/index.md).

```{toctree}
:maxdepth: 1
:caption: Specialized examples

experimental-planar-channeling
experimental-axial-channeling
transition-radiation-demo
radiation-diagnostics
```

Quantum/channeling examples are **experimental and not rigorously validated**.
The [experimental API reference](../reference/experimental-api.md) distinguishes
compatibility status from scientific validation.

```{toctree}
:maxdepth: 1
:caption: Help

troubleshooting
```
