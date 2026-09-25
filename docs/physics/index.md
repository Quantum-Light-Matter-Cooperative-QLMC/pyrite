# Physical models

This section documents the physical models, assumptions, conventions, and numerical treatments used by the simulation.

These pages describe **how the simulated physical system is represented**. For evidence that a model or implementation reproduces analytic results, literature values, external calculations, or independent derivations, see the [physics validation](../validation/index.md) section.

## Model at a glance

### Beam and electron transport

- Electron position, direction, energy, and arrival time are sampled at the target entrance. Upstream beamline transport and space charge are excluded; see [Beam phase space](beam-transport/beam-phase-space.md).
- Independent electrons follow piecewise-linear flights through slabs or layer stacks. Elastic collisions are explicit, while inelastic loss is condensed between them and is deterministic by default, with optional Urban straggling. Knock-on electrons are not transported; see [Electron transport](beam-transport/electron-transport.md).
- The default elastic model combines Browning total cross sections with NIST Mott-calibrated angular transport. Stopping uses a per-element Joy--Luo/Berger--Seltzer splice; see [Elastic scattering](beam-transport/elastic-scattering.md) and [Stopping power](beam-transport/stopping-power.md).

### Crystal and coherent emission

- Phase-specific crystal structures, complex atomic form factors, isotropic Debye--Waller factors, and selected or catalog-pinned reflection families define the reciprocal-space couplings. Optional Gaussian mosaicity uses either analytic broadening or incoherent orientation quadrature; see [Crystal structure](materials/crystal-structure.md), [Structure factor](materials/structure-factor.md), [Reflection selection](materials/reflection-selection.md), and [Crystal mosaicity](materials/crystal-mosaicity.md).
- The electron field couples to the crystal susceptibility as PXR; the periodic screened potential drives CBS. Their amplitudes interfere within each polarization, while distinct polarizations, reflections, mosaic orientations, and crystalline layers add as intensities; see [Coherent PXR and CBS radiation](radiation-physics/coherent-radiation.md).
- The line kernel is first-order and kinematic. Each numerical transport row uses one representative velocity and finite-flight sinc factor. CSDA may subdivide one collision-free flight to update energy and clock; those substep fields are grouped before the flight intensity is formed. Dynamical diffraction and electron channeling are excluded.
- Distinct physical flights and electrons add as intensities by default. This is a random-phase/independent-emission approximation, not collision-induced decoherence. The optional phased tracking policy preserves phase across each electron trajectory and blends inter-electron terms through sampled bunch form factors; see [Coherent-emission tracking](radiation-physics/coherent-emission.md).

### Incoherent emission

- The continuum is isotropic, unscreened Born Bethe--Heitler bremsstrahlung, evaluated with relativistic momenta and an Elwert Coulomb correction along the electron tracks; see [Bremsstrahlung](radiation-physics/bremsstrahlung.md).
- Characteristic radiation, fluorescence, and secondary-photon production are not modeled.

### Photon and detector transport

- Photons follow straight rays from segment midpoints and receive passive Beer--Lambert attenuation. Bulk refractive dispersion shifts coherent-line resonance and phase; interface optics, photon scattering, re-emission, and feedback on the electron are excluded. See [Photon escape and in-medium dispersion](radiation-physics/photon-escape-and-dispersion.md).
- The default detector treatment evaluates one far-field direction, applies flat solid-angle scaling and analytic aperture broadening, then treats quantum efficiency and measured-energy redistribution downstream. See [Detector solid-angle integration](detectors/detector-solid-angle.md) and [Detector and instrument response](detectors/detector-response.md).

## Reference pages

```{toctree}
:maxdepth: 1
:caption: Atomic physics

atomic-physics/atomic-form-factors
atomic-physics/elemental-transport-data
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
beam-transport/shell-soft-hard-transport
beam-transport/transport-outputs
```

```{toctree}
:maxdepth: 1
:caption: Radiation physics

radiation-physics/coherent-radiation
radiation-physics/coherent-emission
radiation-physics/bremsstrahlung
radiation-physics/characteristic-radiation
radiation-physics/photon-escape-and-dispersion
radiation-physics/spectral-observables
radiation-physics/energy-grid-semantics
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

materials/crystal-structure
materials/structure-factor
materials/reflection-selection
materials/material-composition
materials/crystal-mosaicity
materials/multilayer-materials
```
