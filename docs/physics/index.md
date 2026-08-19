# Physical models

This section documents the physical models, assumptions, conventions, and numerical treatments used by the simulation.

These pages describe **how the simulated physical system is represented**. For evidence that a model or implementation reproduces analytic results, literature values, external calculations, or independent derivations, see the [physics validation](../validation/index.md) section.

```{toctree}
:maxdepth: 1
:caption: Atomic physics

atomic-physics/atomic-data-sources
```

```{toctree}
:maxdepth: 1
:caption: Beam and electron transport

beam-transport/beam-phase-space
beam-transport/longitudinal-structure
beam-transport/electron-transport
beam-transport/elastic-scattering
beam-transport/stopping-power
beam-transport/transport-outputs
```

```{toctree}
:maxdepth: 1
:caption: Radiation physics

radiation-physics/coherent-radiation
radiation-physics/coherent-emission
radiation-physics/bremsstrahlung
radiation-physics/photon-escape-and-dispersion
radiation-physics/spectral-observables
```

```{toctree}
:maxdepth: 1
:caption: Geometry and orientation

geometry/tilt-convention
geometry/transport-geometry
```

```{toctree}
:maxdepth: 1
:caption: Detectors and optics

detectors/detector-solid-angle
detectors/detector-response
```

```{toctree}
:maxdepth: 1
:caption: Materials and crystallography

materials/crystal-mosaicity
materials/multilayer-materials
```

## How to use this section

If you are trying to understand **what physical model Pyrite uses**, start here.

For example:

* **Beam and electron transport** covers the specification of the incident beam, its per-electron sampling, and the elastic-scattering and stopping models that propagate it through matter.
* **Radiation physics** covers the mechanisms by which radiation is produced and propagated.
* **Geometry and orientation** defines crystal, sample, and multilayer geometry, and the boundaries that constrain transport.
* **Atomic physics** documents the atomic quantities and external data on which higher-level models depend.
* **Materials and crystallography** documents material definitions and crystallographic data.
* **Detectors and optics** covers the physical models used after radiation leaves the source.

Validation documents are intentionally kept separate from these reference pages so that the current model description does not become mixed with the historical record of how the model was checked.
