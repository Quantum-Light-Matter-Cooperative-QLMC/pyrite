# Development Reference

This section documents the structure, interfaces, data models, workflows, and implementation decisions that shape the Pyrite codebase.

It is intended primarily for contributors and maintainers. User-facing instructions belong in [Guides](../guides/index.md), while physical-model documentation belongs in [Physics and simulation models](../physics/index.md).

```{toctree}
:maxdepth: 1
:caption: Development and repository structure

development-workspace
documentation
../validation/formatting-style
configuration-resolution
materials-catalog-schema
data-distribution-and-repository-size
core-architecture-rfc
../repo_map
```

```{toctree}
:maxdepth: 1
:caption: Compute optimization/GPU Acceleration

../guides/performance-profile-analysis
compute/coherent-streaming-rawkernel.md
compute/compute-performance-optimization.md
compute/gpu-transport-rawkernel.md
compute/jit-spectrum-kernel-walkthrough.md
compute/straggled-transport-integration.md
compute/transport-jit-kernel-walkthrough.md
```

```{toctree}
:maxdepth: 1
:caption: Storage and artifacts

storage/checkpoint-case-store
storage/dataset-identity-and-storage
storage/result-schema
../physics/beam-transport/transport-outputs
```

## Architecture decision records

Major architectural choices and their rationale are recorded separately as [architecture decision records](../adr/index.md).

Use an ADR when the important question is:

> Why did the project choose this approach over the alternatives?

Use a reference or design page in this section when the important question is:

> How does the system work now?

This distinction allows the reference documentation to evolve with the implementation while preserving the historical reasoning behind major decisions.

## Scope

Typical material for this section includes:

* repository and package organization,
* internal data and artifact formats,
* command-line interfaces,
* checkpointing and persistence,
* compute and execution architecture,
* development environment conventions,
* performance and scaling architecture,
* deprecation and compatibility policy.

Detailed descriptions of physical models should remain under the physics documentation even when their implementations are computationally complex.
