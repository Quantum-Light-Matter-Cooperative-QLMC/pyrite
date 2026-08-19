# Execution and acceleration

:::{note}
**Stub.** Scope is defined below; the content has not been written. Current
authoritative material is linked at the end.
:::

## Intended scope

How a case actually runs, once the physics is fixed.

* **Backend and core selection.** What `transport_core` and the array-backend
  resolution choose, when, and on what measured crossover. Why the choice is
  process-scoped and how it is pinned.
* **The GPU transport kernel.** One thread per electron, run to completion;
  control flow, output addressing, segment capacity and replay, launch geometry,
  and the batching policy that bounds memory without touching results.
* **The coherent streaming kernel.** Where the complex accumulation happens and
  why it streams.
* **Lookup tables.** The uniform-in-energy transport LUT: what it replaces, its
  interpolation error budget, and why the grid is uniform.
* **Device residency.** Keeping segments where they were produced, what that
  requires of the caller's spectrum backend, and its memory cost.
* **What acceleration does not change.** The distinction the ledger draws
  between a *realization* and a *distribution*: reordered random streams and
  a few ulp of libm difference give a different sample of the same physics, so
  cross-core comparison is statistical by construction, never bitwise.
* **Fail-closed boundaries.** The reductions that raise rather than silently
  falling back to a different algorithm on a non-host backend.

## Deliberately out of scope

Physical models, which stay under [physics](../physics/index.md). Statistical
estimators and error quantification, which are
[their own page](statistical-methods.md).

## Current sources

* [`repo-design/compute/gpu-transport-rawkernel.md`](../repo-design/compute/gpu-transport-rawkernel.md)
* [`repo-design/compute/coherent-streaming-rawkernel.md`](../repo-design/compute/coherent-streaming-rawkernel.md)
* [`repo-design/compute/compute-performance-optimization.md`](../repo-design/compute/compute-performance-optimization.md)
* ledger row `gpu-transport-core`
