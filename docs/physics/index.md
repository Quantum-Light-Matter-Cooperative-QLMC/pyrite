# Physical Models

This section documents the physical models, assumptions, conventions, and numerical treatments used by the simulation.

These pages describe **how the simulated physical system is represented**. For evidence that a model or implementation reproduces analytic results, literature values, external calculations, or independent derivations, see the [physics validation](../validation/index.md) section.

## Model at a glance

### Beam and electron transport

- Electron position, direction, energy, and arrival time are sampled at the target entrance. Upstream beamline transport and space charge are excluded; see [Beam phase space](beam-transport/beam-phase-space.md).
- Independent electrons follow piecewise-linear flights through slabs or layer stacks. Elastic collisions are explicit. By default, production runs sample hard inelastic collisions with the shell model wherever every layer has shell data, with optional PENELOPE soft-loss straggling and opt-in secondary tracking; elsewhere collision loss is continuous, with optional Urban straggling. See [Electron transport](beam-transport/electron-transport.md) and [Inelastic scattering](beam-transport/inelastic-scattering-events.md).
- The default elastic model samples ELSEPA Dirac partial-wave total and differential cross sections (muffin-tin tables for single-element crystals); the Browning/NIST Mott-calibrated model remains selectable. Collision stopping uses a material-level SBETHE table for each layer; see [Elastic scattering](beam-transport/elastic-scattering.md) and [Stopping power](beam-transport/stopping-power.md).

### Crystal and coherent emission

- Phase-specific crystal structures, complex atomic form factors, isotropic Debye--Waller factors, and selected or catalog-pinned reflection families define the reciprocal-space couplings. Optional Gaussian mosaicity uses either analytic broadening or incoherent orientation quadrature; see [Crystal structure](materials/crystal-structure.md), [Structure factor](materials/structure-factor.md), [Reflection selection](materials/reflection-selection.md), and [Crystal mosaicity](materials/crystal-mosaicity.md).
- The electron field couples to the crystal susceptibility as PXR; the periodic screened potential drives CBS. Their amplitudes interfere within each polarization, while distinct polarizations, reflections, mosaic orientations, and crystalline layers add as intensities; see [Coherent PXR and CBS radiation](radiation-physics/coherent-radiation.md).
- The line kernel is first-order and kinematic. Each numerical transport row uses one representative velocity and finite-flight sinc factor. CSDA may subdivide one collision-free flight to update energy and clock; those substep fields are grouped before the flight intensity is formed. Dynamical diffraction and electron channeling are excluded.
- Distinct physical flights and electrons add as intensities by default. This is a random-phase/independent-emission approximation, not collision-induced decoherence. The optional phased tracking policy preserves phase across each electron trajectory and blends inter-electron terms through sampled bunch form factors; see [Coherent-emission tracking](radiation-physics/coherent-emission.md). An opt-in profile key adds the line arrival-time intensity $I(t)=|E(t)|^2$; see [Temporal intensity profile](radiation-physics/temporal-intensity-profile.md).

### Incoherent emission

- The continuum defaults to the direction-resolved BremsLib backend when its tables are installed, and otherwise (with a warning) to isotropic EEDL bremsstrahlung — the ENDF-6 MF=23/MT=527 total cross section multiplied by the normalized MF=26/MT=527 photon-energy density. Unscreened Born Bethe--Heitler with an Elwert correction remains an optional/fallback backend; see [Bremsstrahlung](radiation-physics/bremsstrahlung.md).
- The default coupled BremsLib mode debits soft radiative loss continuously and samples hard photons that debit electron energy; see [Hard BremsLib photon events](radiation-physics/hard-bremsstrahlung-events.md).
- Characteristic radiation is modeled: electron-impact vacancies from EEDL subshell ionization cross sections relax through xraydb fluorescence yields with L-shell Coster--Kronig redistribution, and each line carries its natural-width Lorentzian as a separate incoherent component; see [Characteristic radiation](radiation-physics/characteristic-radiation.md). Secondary fluorescence and Auger-fed daughter vacancies are not modeled.

### Photon and detector transport

- Photons follow straight rays from segment midpoints and receive passive Beer--Lambert attenuation. Bulk refractive dispersion shifts coherent-line resonance and phase; interface optics, photon scattering, re-emission, and feedback on the electron are excluded. See [Photon escape and in-medium dispersion](radiation-physics/photon-escape-and-dispersion.md).
- The default detector treatment evaluates one far-field direction, applies flat solid-angle scaling and analytic aperture broadening, then treats quantum efficiency and measured-energy redistribution downstream. See [Detector solid-angle integration](detectors/detector-solid-angle.md) and [Detector and instrument response](detectors/detector-response.md).

## Reference pages

```{toctree}
:maxdepth: 1
:caption: Atomic physics

atomic-physics/atomic-form-factors
atomic-physics/elemental-transport-data
```

```{toctree}
:maxdepth: 1
:caption: Beam and electron transport

beam-transport/beam-phase-space
beam-transport/longitudinal-structure
beam-transport/electron-transport
beam-transport/elastic-scattering
beam-transport/atomic-electron-deflection
beam-transport/stopping-power
beam-transport/inelastic-scattering-events
beam-transport/shell-soft-hard-transport
```

```{toctree}
:maxdepth: 1
:caption: Radiation physics

radiation-physics/coherent-radiation
radiation-physics/coherent-emission
radiation-physics/temporal-intensity-profile
radiation-physics/bremsstrahlung
radiation-physics/hard-bremsstrahlung-events
radiation-physics/characteristic-radiation
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

materials/crystal-structure
materials/structure-factor
materials/reflection-selection
materials/material-composition
materials/crystal-mosaicity
materials/multilayer-materials
```
