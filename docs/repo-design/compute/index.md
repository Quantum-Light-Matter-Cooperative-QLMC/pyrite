# Compute implementation and performance

Understand transport and scoring kernels, resource behavior and measured costs.
For execution tasks, use [profiling](../../guides/performance-profile-analysis.md)
or [CUDA tuning](../../guides/batch-chunk-tuning.md); exact runtime settings live
in [configuration reference](../../reference/configuration/index.md).

```{toctree}
:maxdepth: 1
:caption: Kernels and physics integration

transport-jit-kernel-walkthrough
gpu-transport-rawkernel
jit-spectrum-kernel-walkthrough
coherent-streaming-rawkernel
straggled-transport-integration
physics-preparation-adapters
emission-batch-adapters
extending-physics
photon-aperture
```

```{toctree}
:maxdepth: 1
:caption: Performance and resource measurements

startup-latency
slow-test-audit
phase-retention
transport-precision-experiment
physics-adapter-overhead
cxr-launch-benchmarks
```

Earlier optimization rounds and paired phase-retention measurements remain in
[Design and measurement history](../history.md).
