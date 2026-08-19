# Computation and statistics

:::{note}
**Placeholder section.** The pages below are stubs that name their intended
scope and point at the material that exists today. They are not yet reference
documentation, and nothing here should be cited as a description of current
behavior until the stub marker is removed from the page in question.
:::

This section is for **how PyRITE computes**, as distinct from what it models.
Two bodies of material belong here:

* **Execution and acceleration** — backend selection, the CUDA transport and
  coherent-streaming kernels, lookup-table construction, batching and memory
  policy, and what changes (and provably does not change) when a run moves to a
  device.
* **Statistical technique** — the Monte Carlo estimators the package forms, how
  their errors are quantified, how random streams are structured and seeded, and
  the convergence protocol the validation records apply.

Physical models stay in [Physics and simulation models](../physics/index.md),
even where their implementations are computationally involved. Concrete
implementation and design records stay in
[Dev. Reference](../repo-design/index.md). This section is the connective
material: the numerical and statistical reasoning that both of those depend on
but neither owns.

```{toctree}
:maxdepth: 1

execution-and-acceleration
statistical-methods
random-streams
```

## Where the material lives today

Until these pages are written, the authoritative sources are:

* [`repo-design/compute/gpu-transport-rawkernel.md`](../repo-design/compute/gpu-transport-rawkernel.md)
  — CUDA transport kernel design and measurements;
* [`repo-design/compute/coherent-streaming-rawkernel.md`](../repo-design/compute/coherent-streaming-rawkernel.md)
  — streaming coherent-sum kernel;
* [`repo-design/compute/compute-performance-optimization.md`](../repo-design/compute/compute-performance-optimization.md)
  — profiling and optimization record;
* [`validation/methodology.md`](../validation/methodology.md) — the evidence
  standard the statistical pages must serve;
* the ledger rows `gpu-transport-core`, `energy-step-convergence`, and
  `substep-radiation-invariance`, which already contain most of the statistical
  argument in condensed form.
