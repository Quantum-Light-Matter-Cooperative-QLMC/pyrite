# Quantum and channeling

Review supported portable quantum configurations and bounded experimental diagnostic entry points. See the [quantum API map](../quantum-api.md).

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.quantum.TransverseStateSpec
   pyrite.quantum.QuantumSetup
   pyrite.quantum.QuantumRegion
   pyrite.quantum.QuantumDynamics
   pyrite.quantum.QuantumNumerics
   pyrite.montecarlo.channeling.spatial.spatial_density
   pyrite.montecarlo.channeling.axial_diagnostic.AxialDiagnosticConfig
   pyrite.montecarlo.channeling.axial_diagnostic.AxialDiagnosticResult
   pyrite.montecarlo.channeling.axial_diagnostic.axial_diagnostic
```

## Solver diagnostic contracts

| Entry | Owner / status | Scientific contract and effects |
| --- | --- | --- |
| `spatial_density` | Quantum/channeling, advanced-stable | Density matrix, basis eigenfunctions and spatial grid (angstrom); Born-rule probability density (1/angstrom), including retained survival weight, without conditioning or renormalization. Array compatibility errors propagate; no transport or artifact I/O. |
| `AxialDiagnosticConfig`, `AxialDiagnosticResult`, `axial_diagnostic` | Quantum/channeling, experimental | Bounded solver configuration (field-specific units), populations and incident-normalized spatial density on recorded depths; retained solver/configuration diagnostics. Explicit solver/allocation limits and invalid configurations raise. No standard axial transport integration. Ownership consolidation belongs to #704. |

For planar channeling, `DensityMatrixSolver.propagate(spatial_x_ang=...)`
records density per incident electron per Å on the supplied grid. Its cell
integral equals the surviving solver trace. The recorded array has shape
`(len(z_ang), len(x_ang))`, includes entrance and exit, and folds density
into one period in the instantaneous basis after energy reprojection. Without
position sampling, both the grid and density are `None`. See the
[spatial-density derivation](../../validation/channeling/cr-planar-spatial-density-reconstruction.md)
and [channeling demo](../../reference/results-and-apps.md#planar-channeling-demo).
The [experimental axial diagnostic](../../guides/experimental-axial-channeling.md) runs independently
of `simulate`.
