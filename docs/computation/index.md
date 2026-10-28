---
orphan: true
---

# Computation and Statistics

This compatibility landing page preserves the former top-level Computation URL.
The original subpage URLs, equations, and anchors remain unchanged, but their
primary navigation now follows their reader purpose:

* [Numerical and statistical methods](../physics/numerical-methods/index.md) — estimators, error bars, convergence, and comparisons between simulations.
* [Execution and runtime configuration](../reference/configuration/index.md) — backend and transport-core selection, memory and chunking, floating-point precision, and reproducible streams.

Direct links remain available to [execution and acceleration](execution-and-acceleration.md),
[memory, chunking, and scheduling](memory-and-scheduling.md),
[precision and tolerances](precision-and-tolerances.md),
[statistical methods](statistical-methods.md), and
[random number streams](random-streams.md).

For physical models see [Physics](../physics/index.md); for current implementation
and measurement records see [Development](../development/index.md).
The [validation methodology](../validation/methodology.md) describes the evidence
required for physics claims.

## Execution workflows and contracts

For tasks, use [cluster execution](../guides/running-on-a-cluster.md),
[performance profiling](../guides/performance-profile-analysis.md), or
[CUDA batch/chunk tuning](../guides/batch-chunk-tuning.md). Exact scheduler
configuration belongs in [cluster reference](../reference/cluster-execution.md);
worker publication and integrity belong in
[distributed worker artifacts](../reference/distributed-artifacts.md).

## Reference material

The measurement records behind the design choices described here live under the compute-optimization group of the [Dev. Reference](../repo-design/index.md):

* [GPU electron transport RawKernel](../repo-design/compute/gpu-transport-rawkernel.md) — CUDA transport kernel design, throughput tables, occupancy and divergence analysis;
* [streaming coherent line RawKernel stage](../repo-design/compute/coherent-streaming-rawkernel.md) — the streaming coherent-sum kernel and its memory behavior;
* [compute-performance optimization](../repo-design/compute/compute-performance-optimization.md) — the profiling and optimization history the current defaults were tuned against.

The ledger rows `gpu-transport-core`, `energy-step-convergence`, and `substep-radiation-invariance` in the [physics validation ledger](../validation/physics-validation-ledger.md) record the supporting evidence and validation status.
