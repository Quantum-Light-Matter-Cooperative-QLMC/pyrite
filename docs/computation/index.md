# Computation and statistics

These pages explain execution, numerical precision, and statistical uncertainty in PyRITE.

* [Execution and acceleration](execution-and-acceleration.md): backend and transport-core selection, CUDA kernels, lookup tables, and reproducibility.
* [Memory, chunking, and scheduling](memory-and-scheduling.md): memory budgets, chunk sizing, worker pools, and out-of-memory recovery.
* [Precision and tolerances](precision-and-tolerances.md): floating-point types and how to interpret numerical differences.
* [Statistical methods](statistical-methods.md): estimators, error bars, convergence, and comparisons between runs.
* [Random number streams](random-streams.md): stream assignment, seeding, and reproducibility contracts.

Physical models are documented in [Physics and simulation models](../physics/index.md). Implementation details and measurement records are in [Dev. Reference](../repo-design/index.md). The [validation methodology](../validation/methodology.md) defines the evidence required for physics claims.

```{toctree}
:maxdepth: 1

execution-and-acceleration
memory-and-scheduling
precision-and-tolerances
statistical-methods
random-streams
```

## Reference material

The measurement records behind the design choices described here live under the compute-optimization group of the [Dev. Reference](../repo-design/index.md):

* [GPU electron transport RawKernel](../repo-design/compute/gpu-transport-rawkernel.md) — CUDA transport kernel design, throughput tables, occupancy and divergence analysis;
* [streaming coherent line RawKernel stage](../repo-design/compute/coherent-streaming-rawkernel.md) — the streaming coherent-sum kernel and its memory behavior;
* [compute-performance optimization](../repo-design/compute/compute-performance-optimization.md) — the profiling and optimization history the current defaults were tuned against.

The ledger rows `gpu-transport-core`, `energy-step-convergence`, and `substep-radiation-invariance` in the [physics validation ledger](../validation/physics-validation-ledger.md) record the supporting evidence and validation status.
