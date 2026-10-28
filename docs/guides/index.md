# How-to Guides

Use these recipes when you have a specific task to complete. For a guided
introduction, begin with [Getting Started](getting-started.md) and the
[Interactive Tutorials](../notebooks/index.md). Scientific background belongs in
[Physics](../physics/index.md), including the
[simulation overview](../physics/how-pyrite-works.md); exact contracts belong in [Reference](../reference/index.md).

New to the Python API? Follow the [CLI quickstart](getting-started.md),
then run [Your first Python simulation](python-api-workflow.md).

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
optimization-studies
```

```{toctree}
:maxdepth: 1
:caption: Execution and performance

running-on-a-cluster
performance-profile-analysis
batch-chunk-tuning
```

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
