# Quantum API map

Use `pyrite.quantum` for quantum specifications, configuration, runtime states,
and selected advanced diagnostics. It remains a module, with lazy exports:
importing it or its configuration/specification classes loads no solver,
GPU, plotting backend, scheduler, or material dataset. Requesting a runtime
factory or solver loads its existing scientific dependencies. Star imports
resolve the entire advanced surface and therefore load those dependencies.

API stability describes compatibility, not physical validation. Quantum
channeling remains experimental research software. The production adapter
admits only the existing planar Bloch density-matrix channeling recipe;
coherent, MCWF, axial, and other configurations still explicitly refuse in
`QuantumSetup.lower`. Direct diagnostic solvers do not imply production
admission. See the [planar](experimental-planar-channeling.md) and
[axial](experimental-axial-channeling.md) workflow restrictions.

## Imports and contracts

All names below are available from `pyrite.quantum`. This is the canonical
public import home; the implementation modules retain their existing types.

| Role | Names | API status and persistence |
| --- | --- | --- |
| Portable source | `TransverseStateSpec` | Advanced-stable; `to_dict`/`from_dict`, canonical JSON and content digest retain schema `pyrite.quantum.transverse-state.v1`. Sampled `.npy` content stays external and is checked on runtime conversion. |
| Configuration | `QuantumRegion`, `QuantumDynamics`, `QuantumNumerics`, `QuantumSetup` | Advanced-stable immutable values; existing constructors, pickle paths and production lowering are unchanged. These remain quantum-owned rather than duplicated in `pyrite.config`. |
| Runtime state types | `TransverseState`, `SampledState`, `Superposition`, `MixedState`, `Translated`, `EntryState` | Advanced-stable diagnostic interfaces; nonportable wavefunctions, arrays, or mixtures, with no portable JSON contract. |
| Runtime factories | `gaussian`, `hermite_gauss`, `laguerre_gauss`, `donut`, `bessel_gauss`, `airy`, `aperture_probe`, `from_array`, `from_abtem`, `superpose`, `mixture`, `translate` | Advanced-stable runtime interfaces; coherent superpositions add amplitudes, incoherent mixtures add weighted densities. `from_abtem` needs the caller's abTEM wave object. |
| Planar source bridge | `PlanarStateProjection`, `project_planar_source`, `propagate_planar_source`, `score_planar_source` | Experimental diagnostic projection, evolution and radiation scoring; does not construct a production case. |
| Planar solvers/results | `DensityMatrixSolver`, `DensityMatrixResult`, `MCWFSolver`, `MCWFResult` | Experimental direct solvers; results are runtime numerical histories, not portable source specifications or checkpoints. |
| Scattering inputs | `ScatteringKernel`, `tds_kernel`, `electronic_kernel` | Experimental solver-building interfaces; retain existing operator normalization and defaults. Profiles come from `pyrite.materials.channeling`. |
| Coherent real-space solving | `RealSpaceGrid`, `SplitStepPropagator` | Experimental periodic-grid diagnostics; finite grid loss is not physical dechanneling. |
| Axial solving/results | `AxialMCWF`, `AxialMCWFResult`, `AxialDiagnosticConfig`, `AxialDiagnosticResult`, `axial_diagnostic` | Experimental direct diagnostics; production axial admission is unchanged. The high-level diagnostic collects grid, escape, phase-space and optional emission summaries. |

Existing `pyrite` root aliases, `pyrite.quantum_state.TransverseStateSpec`,
`pyrite.states` factories, `pyrite.planar_source` bridge functions, and the
corresponding `pyrite.montecarlo.channeling` leaf imports remain the **same
objects**. No classes move, and pickle paths, case payloads, model markers,
random streams, and numerical defaults are unchanged. Other channeling engine
helpers are not implicitly part of this facade. The exact export/status list
is governed by `api-surface.toml`.

Portable recipes describe sources; runtime factories produce evaluated state
objects. Use `TransverseStateSpec.recipe(...)` and `.to_runtime()` when a
durable source description is needed. Production source injection remains
limited to its existing admitted subset, detailed in the planar guide.

## Coherent example: experimental diagnostic only

This bounded free-space example uses a periodic 2D envelope. Coordinates are
`(x, y)` in Å, with array axes in that order; carrier tilt is separate and in
radians. `grid.sample` periodizes the state. Normalize explicitly on the finite
grid, then inspect probability density in Å⁻². No absorptive potential is set,
so propagation preserves that grid norm. This is not a production transport
or physical escape calculation.

```python
import numpy as np
from pyrite.quantum import RealSpaceGrid, SplitStepPropagator, gaussian

grid = RealSpaceGrid(extent_ang=(16.0, 16.0), shape=(32, 32))
phi = grid.sample(gaussian(waist_ang=1.0, dimension=2))
phi /= np.sqrt(np.sum(np.abs(phi) ** 2) * grid.cell_area_ang2)
propagator = SplitStepPropagator(grid, kinetic_energy_eV=17e6)
history = list(propagator.trajectory(phi, [0.0, 10.0], step_ang=5.0))
coherent_density = np.abs(history[-1][1]) ** 2
coherent_norm = np.sum(coherent_density) * grid.cell_area_ang2
```

## Density-matrix example: direct planar diagnostic

The production `QuantumSetup` adapter supports planar Bloch density-matrix
channeling, subject to the existing transport restrictions. This example
instead runs the direct experimental solver, without transport or radiation
scoring. Its deliberately small basis is an API smoke example, not a converged
prediction. Depth and step are Å, energy is eV, incidence is the effective
refracted angle to the planes in radians. The default entry is a plane wave
with zero divergence. `trace` is retained subspace probability per incident
electron; it is not conditional on survival. The integral of the optional
cell density in Å⁻¹ equals `trace`. Flux fields book removed/entry population.
The explicit 5 eV margin is only a diagnostic choice; production defaults
remain unchanged.

```python
from pyrite.materials.channeling import planar_profile
from pyrite.quantum import DensityMatrixSolver, electronic_kernel, tds_kernel

profile = planar_profile("silicon", (1, 1, 0), 293.15)
kernel = tds_kernel(profile, 17e6, n_cells=2, max_order=2)
kernel += electronic_kernel(profile, 17e6, n_cells=2, max_order=2)
solver = DensityMatrixSolver(profile, 17e6, kernel, margin_eV=5.0)
x_ang = np.linspace(0.0, profile.period_ang, 16, endpoint=False)
density_result = solver.propagate(10.0, 5.0, spatial_x_ang=x_ang)
```

For portable-state projection use `QuantumSetup.project_state` or
`project_planar_source`, followed by `propagate_planar_source` or
`score_planar_source`. `impact="fixed"` retains positional coherence;
`impact="averaged"` averages over lattice position. Projection's retained,
free, and outside populations describe **entry**, not evolved escape. Scored
spectra retain photons/(incident electron eV sr), separate capture components,
and existing approximation limits. See the [planar source example](experimental-planar-channeling.md#specify-and-inspect-quantum-entrance-states).

## MCWF example: experimental diagnostic only

Continue with the same planar solver and scattering operators. The result is
an ensemble of stochastic quantum jumps, with populations per electron
launched, standard errors, and cumulative flux fractions. Sixteen trajectories
demonstrate the interface; they are insufficient for a quantitative comparison
with the density-matrix expectation. Seed and electron indices retain their
existing stream semantics. This direct MCWF run does not make
`QuantumNumerics(method="mcwf")` production-admitted.

```python
from pyrite.quantum import MCWFSolver

mcwf_result = MCWFSolver(solver).run(
    16, 10.0, 5.0, seed=7, record_every=1
)
```

For axial MCWF and its diagnostics, use the [axial guide](experimental-axial-channeling.md).
Grid survival, projection weight, and physical dechanneling remain distinct.
