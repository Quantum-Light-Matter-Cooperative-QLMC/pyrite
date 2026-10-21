# Axial periodic supercell and transverse flow (#625)

## Geometry and admission

The axial grid is **one finite, periodically repeated supercell** of projected
crystal cells. `cells=(cx,cy)` specifies how many lattice periods are inside
that supercell; `shape=(nx,ny)` counts FFT samples **across the entire
supercell**, not per crystal cell. Therefore `nx/cx` and `ny/cy` are the
resolutions per projected unit cell. A cropped app image is only a viewport
into that *same periodic solution*: cropping cannot convert periodic
boundaries into open boundaries. Grid images across opposite edges describe
the same periodized solution, not distinct independent electrons.

The axial diagnostic requires integral resolution per cell and at least eight
samples/cell/axis, and warns below 16. These are **admission guards**, not a
claim of physical accuracy; sharpen the grid until the required observables
converge. Increase supercell size at fixed samples per cell, e.g.

| `cells` | `shape` | Samples/cell |
| --- | --- | --- |
| `(1,1)` | `(24,24)` | `(24,24)` |
| `(2,2)` | `(48,48)` | `(24,24)` |
| `(3,3)` | `(72,72)` | `(24,24)` |

Increasing `cells` while leaving `shape` fixed **reduces spatial resolution**
and is not a valid isolated extent convergence test. Check depth-step
convergence independently. For a *localized* incident electron, the physical
beam and its images can overlap after significant spreading; compare the
central density, internal-cell population, current and flux on growing cells.
Extended Bloch-periodic states do not necessarily converge in the same sense.

## Current and boundary conventions

The solver propagates the periodic envelope `phi`, with a separately tracked
physical carrier `exp(i q·r)`. With `a = hbar^2/(2 gamma m)` and
`hbar_v = hbar v_z`, its paraxial transverse current is

`j = (2 a / hbar_v) [Im(phi* grad phi) + q |phi|^2]`.

The gradient is computed from the same FFT wave numbers as the kinetic
propagator. **The carrier term must not be omitted**, including after MCWF
electronic momentum kicks. `j_x` and `j_y` are in Å⁻² per incident electron;
`div j` is in Å⁻³. Dividing the recorded arrays by `surviving` is a separate
conditional operation; the default is **incident-electron normalized**.

The diagnostic reports current at grid-sample centres. At a face between
samples `i` and `i+1` it averages the two sample-centre currents, wrapping
at the supercell edge. Finite-volume divergence subtracts opposite face
currents divided by `dx` or `dy`. A rectangular `FluxContour` is defined by
`start=(ix,iy)` and `size=(width,height)` in **pixel units**; negative starts
wrap. Its four signed outward sides are `(-x,+x,-y,+y)` in Å⁻¹. The signed
sum is the **net outward rate per unit propagation depth**. Flux is evaluated
at sampled depths; it is not integrated into a loss counter.

The face/divergence/contour definitions satisfy the *discrete divergence
theorem*: sum of divergence times pixel area inside the contour equals sum
of its four side fluxes, including when the contour wraps. At fixed finite
FFT resolution, this *sample-centred continuum current* is not an exactly
locally conservative discretization of the nonlocal pseudospectral kinetic
operator: aliasing and face averaging produce continuity residuals. Refine
resolution and depth step and test `d_z |phi|^2 + div j`; do not silently
interpret any residual as a sink.

**Physical interpretation:**

- Flux through an *internal* cell boundary represents redistribution between
  cells, not dechanneling or loss.
- Flux through opposite *periodic supercell* boundaries re-enters from the
  other side. A full-supercell contour has zero net outward flux.
- Numerical loss is a failure of norm conservation in an otherwise unitary
  coherent run; it is assessed by the full supercell integral, **not** by
  current crossing a boundary.
- A non-Hermitian MCWF sink removes trajectory weight via the modeled jump
  rule and is explicitly recorded as `numerical_sink` by the existing
  diagnostic (legacy naming; model truncation/sink, not FFT drift). It does
  not arise from crossing a periodic boundary. Physical open-boundary
  escape / quantum-to-classical handoff requires separate validated physics;
  this diagnostic does **not** implement it.

## Recording and CUDA

`AxialDiagnosticConfig(record_current=True)` records both current components
at all requested depths. `record_current_divergence=True` additionally
records divergence; `current_contour=(x0,y0,width,height)` additionally
records four signed sides. The latter two require `record_current=True`.
Results are exposed as `transverse_current_per_ang2` of shape
`(depth,2,nx,ny)`, optional `transverse_divergence_per_ang3`, and optional
`contour_flux_per_ang` of shape `(depth,4)`. All outputs are online ensemble
reductions; individual trajectories are never stored. The existing composite
**128 MiB recording limit** accounts for density, current, divergence,
contour, spectral and coherence reductions together. CUDA callbacks reserve
FFT scratch arrays and use CuPy lazily. App runs still obey CPU/CUDA workload
limits, and solve caching is unchanged.

The app's optional contour is the periodic rectangle centered around string
zero with width one projected crystal cell. Current arrows show physical
carrier-inclusive direction. The displayed density and the flux are both
per incident electron; current crossing the drawn contour is not absorbed.
