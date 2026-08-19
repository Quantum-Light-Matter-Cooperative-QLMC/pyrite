# Random number streams

:::{note}
**Stub.** Scope is defined below; the content has not been written. Current
authoritative material is linked at the end.
:::

## Intended scope

Reproducibility is a design constraint here, not a convenience, and it is
achieved through stream structure rather than through a single global seed.

* **Stream topology.** The `SeedSequence` child tree: which physical input owns
  which child index, and why disjoint children are what make an inert
  distribution *provably* inert — enabling it takes no draw and leaves every
  other array bit-for-bit identical.
* **Counter-addressed streams.** SplitMix64 per-electron keys, how a
  run-to-completion core addresses its own stream without a shared generator,
  and why that is required for a device port.
* **Draw order as a contract.** Step-major versus electron-major consumption:
  same models, same draw semantics, different realization. What may be compared
  bitwise and what may only be compared statistically.
* **Seeding and provenance.** How a seed reaches a case, what a checkpoint
  records about it, and what changes a run's realization without changing its
  parameter hash.
* **Adding a new random input.** The rule that a new distribution takes a new
  child stream rather than sharing an existing one, and the test shape that pins
  the zero-limit bit-for-bit.

## Deliberately out of scope

The physical meaning of the sampled distributions, which belongs to the
[physics](../physics/index.md) pages that define them.

## Current sources

* ledger rows `gpu-transport-core`, `beam-phase-space-injection`,
  `longitudinal-bunch-sampling`
* [`repo-design/compute/gpu-transport-rawkernel.md`](../repo-design/compute/gpu-transport-rawkernel.md)
