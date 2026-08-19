# Computation and statistics

This section is for **how PyRITE computes**, as distinct from what it models. It
is the connective material between the physics and the implementation: the
numerical, statistical, and execution reasoning that the physical models depend
on but do not own.

Three questions organize it.

* **What does a run compute, and how fast?** Backend and core selection, the
  CUDA transport and coherent-streaming kernels, lookup-table construction, and
  what changes — and provably does not change — when a run moves to a device.
  See [execution and acceleration](execution-and-acceleration.md) and
  [memory, chunking, and scheduling](memory-and-scheduling.md).
* **How well does it compute it?** The floating-point policy, where `float32`
  is permitted and where it is not, and the tolerance vocabulary the tests and
  ledger use. See [precision and tolerances](precision-and-tolerances.md).
* **How much of the answer is noise?** The estimators the package forms, how
  their errors are quantified, and how the random streams that generate them are
  structured. See [statistical methods](statistical-methods.md) and
  [random number streams](random-streams.md).

Physical models stay in [Physics and simulation models](../physics/index.md),
even where their implementations are computationally involved. Concrete
implementation and design records — kernel-by-kernel measurements, profiling
history, storage schemas — stay in [Dev. Reference](../repo-design/index.md).
The evidence standard that governs physics claims is
[validation methodology](../validation/methodology.md); this section supplies
the statistical machinery that methodology assumes.

```{toctree}
:maxdepth: 1

execution-and-acceleration
memory-and-scheduling
precision-and-tolerances
statistical-methods
random-streams
```

## The one invariant worth stating first

Every page here is, in some form, about the same distinction: a **realization**
versus a **distribution**.

A realization is one concrete sample — this seed, this core, this device, this
draw order. A distribution is what the physics actually claims. Changing the
transport core, reordering the random streams, or moving a reduction to a GPU
changes the realization while leaving the distribution intact. That is not a
weakness of the implementation; it is a property of Monte Carlo, and stating it
precisely is what makes cross-core comparison meaningful.

The practical consequences run through everything below:

* a spectrum pinned on one core is **not** reproduced bit-for-bit by another
  (see [execution and acceleration](execution-and-acceleration.md));
* comparisons across cores, seeds, or refinement rungs are **statistical by
  construction**, with error bars, never realization by realization (see
  [statistical methods](statistical-methods.md));
* the places where bit-for-bit identity *is* claimed are narrow, deliberate, and
  individually tested — the zero-limit of every optional distribution, and the
  random generator itself across host and device (see
  [random number streams](random-streams.md)).

## Reference material

The measurement records behind the design choices described here live under
the compute-optimization group of the [Dev. Reference](../repo-design/index.md):

* [GPU electron transport RawKernel](../repo-design/compute/gpu-transport-rawkernel.md)
  — CUDA transport kernel design, throughput tables, occupancy and divergence
  analysis;
* [streaming coherent line RawKernel stage](../repo-design/compute/coherent-streaming-rawkernel.md)
  — the streaming coherent-sum kernel and its memory behavior;
* [compute-performance optimization](../repo-design/compute/compute-performance-optimization.md)
  — the profiling and optimization history the current defaults were tuned
  against.

The ledger rows `gpu-transport-core`, `energy-step-convergence`, and
`substep-radiation-invariance` in the
[physics validation ledger](../validation/physics-validation-ledger.md) carry
the statistical arguments in their condensed, adjudicated form.
