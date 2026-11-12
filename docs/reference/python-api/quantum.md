# Quantum and channeling

Use `pyrite.quantum` for portable source specifications, the channeling section,
nonportable runtime states, and state factories. The API contract is distinct
from scientific validation: most direct channeling solvers are **experimental**,
and only the bounded planar Bloch density-matrix recipe is admitted by
`simulate`. See the [quantum API contracts](../quantum-api.md) for details.

Learn the interfaces in [Constructing transverse quantum states](../../notebooks/quantum/12-transverse-quantum-states.ipynb),
then continue to the planar and axial diagnostics.

The tables below link to generated signatures, docstrings, and source code.
Every name on this page, planar included, is experimental in `api-surface.toml`; exposing a factory here
does not make its outputs serializable or production-admitted.

## Portable sources and configuration

`TransverseStateSpec` is a portable recipe. `Channeling` is the single public
channeling section passed as `simulate(..., channeling=...)` or
`Scene(..., channeling=...)`; `ChannelingNumerics` holds its zone numerics.
Off is `channeling=None`.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.quantum.TransverseStateSpec
   pyrite.quantum.Channeling
   pyrite.quantum.ChannelingNumerics
```

## Runtime state objects

These experimental interfaces describe evaluated wavefunctions or mixtures.
They are **not** portable source specifications or persisted simulation
checkpoints; construct `TransverseStateSpec` when a serializable input is needed.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :template: autosummary/public-alias.rst
   :nosignatures:

   pyrite.quantum.TransverseState
   pyrite.quantum.SampledState
   pyrite.quantum.EntryState
   pyrite.quantum.Superposition
   pyrite.quantum.MixedState
   pyrite.quantum.Translated
```

## Runtime state factories

The following experimental factories construct states without admitting
them to production transport. Superpositions combine complex amplitudes;
mixtures combine populations. Array-based and abTEM-backed sources require the
caller's data (or installed external dependency). Dimensions and quantity
inputs follow the [quantum quantity contract](../quantum-api.md#runtime-quantity-inputs).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.quantum.gaussian
   pyrite.quantum.hermite_gauss
   pyrite.quantum.laguerre_gauss
   pyrite.quantum.bessel_gauss
   pyrite.quantum.airy
   pyrite.quantum.donut
   pyrite.quantum.aperture_probe
   pyrite.quantum.from_array
   pyrite.quantum.from_abtem
   pyrite.quantum.superpose
   pyrite.quantum.mixture
   pyrite.quantum.translate
```

## Additional solver diagnostics

These legacy leaf-module paths retain their generated object pages and link to
existing code owners. The same axial objects can be imported from the
`pyrite.quantum` facade; new experimental examples should use
`pyrite.experimental.quantum`, as described in the
[quantum API contracts](../quantum-api.md#imports-and-contracts).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.montecarlo.channeling.spatial.spatial_density
   pyrite.montecarlo.channeling.axial_propagation.AxialPropagationConfig
   pyrite.montecarlo.channeling.axial_propagation.AxialPropagationResult
   pyrite.montecarlo.channeling.axial_propagation.propagate_axial
```

## Solver diagnostic contracts

| Entry | Owner / status | Scientific contract and effects |
| --- | --- | --- |
| `spatial_density` | Quantum/channeling, experimental | Density matrix, basis eigenfunctions and spatial grid (angstrom); Born-rule probability density (1/angstrom), including retained survival weight, without conditioning or renormalization. Array compatibility errors propagate; no transport or artifact I/O. |
| `AxialPropagationConfig`, `AxialPropagationResult`, `propagate_axial` | Quantum/channeling, experimental | Bounded solver configuration (field-specific units), populations and incident-normalized spatial density on recorded depths; retained solver/configuration diagnostics. Explicit solver/allocation limits and invalid configurations raise. No standard axial transport integration. Ownership consolidation belongs to #704. |

For planar channeling, `DensityMatrixSolver.propagate(spatial_x_ang=...)`
records density per incident electron per Å on the supplied grid. Its cell
integral equals the surviving solver trace. The recorded array has shape
`(len(z_ang), len(x_ang))`, includes entrance and exit, and folds density
into one period in the instantaneous basis after energy reprojection. Without
position sampling, both the grid and density are `None`. See the
[spatial-density derivation](../../validation/channeling/cr-planar-spatial-density-reconstruction.md)
and [channeling demo](../../reference/results-and-apps.md#channeling-app).
The [experimental axial diagnostic](../../guides/experimental-axial-channeling.md) runs independently
of `simulate`.

## Experimental solvers and planar projection

These direct solvers and result records use the experimental namespace. Their
presence in the reference does not broaden production admission; consult the
[solver and source contracts](../quantum-api.md) before selecting a solver.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.experimental.quantum.DensityMatrixSolver
   pyrite.experimental.quantum.DensityMatrixResult
   pyrite.experimental.quantum.MCWFSolver
   pyrite.experimental.quantum.MCWFResult
   pyrite.experimental.quantum.AxialMCWF
   pyrite.experimental.quantum.AxialMCWFResult
   pyrite.experimental.quantum.RealSpaceGrid
   pyrite.experimental.quantum.SplitStepPropagator
   pyrite.experimental.quantum.PlanarStateProjection
   pyrite.experimental.quantum.project_planar_source
   pyrite.experimental.quantum.propagate_planar_source
   pyrite.experimental.quantum.score_planar_source
```

### Crystal profiles, Bloch states and emission helpers

Experimental channeling helpers (#1147): crystal potentials, beam-dependent
Bloch states and kinematics, line and spectrum diagnostics, exit-angle
distributions and the fibre research adapters. Import them from
`pyrite.quantum`; their implementation modules are internal.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.quantum.planar_profile
   pyrite.quantum.axial_profile
   pyrite.quantum.transverse_states
   pyrite.quantum.axial_states
   pyrite.quantum.AxialStates
   pyrite.quantum.entry_populations
   pyrite.quantum.longitudinal_kinematics
   pyrite.quantum.line_energy_eV
   pyrite.quantum.channeling_spectrum
   pyrite.quantum.axial_line_catalog
   pyrite.quantum.classical_exit_angles
   pyrite.quantum.coherent_exit_distribution
   pyrite.quantum.planar_fiber_inputs
   pyrite.quantum.unravel_fiber_outgoing
```

`M_E_EV` is the electron rest energy, a plain `float` in eV (no Quantity).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :template: autosummary/public-data.rst
   :nosignatures:

   pyrite.quantum.M_E_EV
```

### Scattering operators

These experimental operators remain at `pyrite.quantum`; they are not part of
the curated `pyrite.experimental.quantum` facade.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.quantum.ScatteringKernel
   pyrite.quantum.electronic_kernel
   pyrite.quantum.tds_kernel
```
