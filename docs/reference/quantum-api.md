# Quantum API contracts

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
admission. See the [planar](../guides/experimental-planar-channeling.md) and
[axial](../guides/experimental-axial-channeling.md) workflow restrictions.

## Imports and contracts

All names below remain available from `pyrite.quantum`, the canonical domain
import home. The approved experimental solvers and planar bridge also have
identity aliases under `pyrite.experimental.quantum`; new experimental examples
use that facade to make their status visible. Specifications, configuration and
runtime factories stay under `pyrite.quantum`. See the
[experimental lifecycle](experimental-api.md).

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

## Runtime quantity inputs

Using the shared registry in `pyrite.units` (Pint is a core dependency), runtime factories through
`pyrite.quantum` (and the identical `pyrite.states` aliases) normalize
`waist_ang`, `scale_ang`, `center_ang`, `spacing_ang`, `origin_ang` and
`offset_ang` to Å; `k_radial_inv_ang` to Å⁻¹; and
`kinetic_energy_eV` to eV. Aperture semiangles use radians; `Cnm`
aberrations use Å and `phiNM` aberrations use radians. Sample arrays in
`from_array` remain unitless wavefunction amplitudes.

The planar diagnostic wrappers accept depth, sampling and geometric path
lengths (Å); stopping power (eV/Å); photon-energy grids and energy spread
(eV); and detector angles, divergence and aperture (rad). Layer endpoints
use Å and the surface normal is dimensionless. Quantity arrays and lists
are supported where the existing interface accepts array inputs. Bare
numbers retain their existing unit convention, and dimensionless Pint
quantities are refused for angular fields. Runtime state and solver values
remain ordinary numeric types; portable source JSON and digests are unchanged.

```python
import numpy as np
from pyrite.quantum import aperture_probe, from_array, gaussian
from pyrite.units import ureg

state = gaussian(waist_ang=1 * ureg.nm, center_ang=(0, 25 * ureg.pm))
sampled = from_array(np.ones(16), spacing_ang=0.1 * ureg.nm)
probe = aperture_probe(
    30 * ureg.keV,
    semiangle_rad=20 * ureg.mrad,
    aberrations={"C10": -3 * ureg.nm, "phi12": 45 * ureg.degree},
)
```

Direct experimental solver constructors, arbitrary low-level helpers, and
callback-defined phase plates are not part of this quantity boundary.

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
refracted angle to the planes in radians. Import NumPy as `np` for the coordinate grid. The default entry is a plane wave
with zero divergence. `trace` is retained subspace probability per incident
electron; it is not conditional on survival. The integral of the optional
cell density in Å⁻¹ equals `trace`. Flux fields book removed/entry population.
The explicit 5 eV margin is only a diagnostic choice; production defaults
remain unchanged.

```python
import numpy as np
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
and existing approximation limits. See the [planar source contracts](#specify-and-inspect-quantum-entrance-states)
and [runnable planar example](../guides/experimental-planar-channeling.md).

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

mcwf_result = MCWFSolver(solver).run(16, 10.0, 5.0, seed=7, record_every=1)
```

For axial MCWF and its diagnostics, use the [axial guide](../guides/experimental-axial-channeling.md).
Grid survival, projection weight, and physical dechanneling remain distinct.

## Planar transport and source contracts

Planar channeling evolves an electron's transverse density matrix in the Bloch
basis of a selected crystal-plane family. In supported `simulate` runs, a
channeling zone hands dechanneled electrons to ordinary Monte Carlo transport.
**Channeling radiation (CR)** is recorded separately from PXR/CBS,
characteristic radiation and bremsstrahlung; it is not Cherenkov radiation.

### Production planar channeling

`Physics(channeling_model="planar", channeling_plane=(h, k, l))` (CLI:
`pyrite physics set NAME --channeling-model planar --channeling-plane H K L`)
admits the planar zone in `simulate` and profile runs; the packaged
`channeling_demo` profile is a 17 MeV, 2 µm silicon example with a forward
detector. `Result.channeling` holds CR in
photons/(incident electron · eV · sr), or `None` when the zone is inactive;
`Result.combined_cxr()` includes it. Profile runs persist a `channeling.h5`
component that the analysis app can select. `channeling_basis_size`,
`channeling_bloch_cells`, `channeling_dz_ang` and `channeling_zone_max_ang`
control the zone's numerical representation and require independent
convergence checks. To run it, follow
[Add planar channeling to a transport run](../guides/experimental-planar-channeling.md#add-planar-channeling-to-a-transport-run).

`QuantumRegion(..., zone_max_length=Q_(2, "nm"))` and
`QuantumNumerics(step_length=Q_(10, "nm"))` take explicit positive scalar Pint
lengths for the zone cap and depth step. Each conflicts with its numeric
`zone_max_ang`/`step_ang` spelling, reads back as a fresh Å quantity
(`zone_max_length` is `None` when uncapped) and is replaced by
`with_zone_max_length` (which also accepts `None`) or `with_step_length`. Numeric
fields, pickles and the lowered `channeling_*` controls are unchanged;
`basis_size` and `bloch_cells` stay counts.

Diagnostic outputs keep their numeric fields. `PlanarStateProjection` adds
fresh `kinetic_energy`, `temperature`, `margin` and `incidence` quantities.
`ChannelingSpectrum.quantify()` returns a copied view whose `density` and
`capture_density` are node-sampled photons per energy and steradian per
incident electron; `to_energy("keV")` rescales the density inversely with the
energy axis, so integrated yields are unchanged. `line_yield` is a
dimensionless count per incident electron, `line_rates` is per Å of depth and
`theta`/`phi` are in radians. Density-matrix and MCWF results remain numeric.

### Specify and inspect quantum entrance states

`TransverseStateSpec` describes a transverse wavefunction or an **incoherent**
mixture separately from the classical beam spot and divergence. A coherent
superposition retains complex relative phases; a mixture adds *densities*, not
amplitudes. Specifications can be serialized with `to_dict()` and
`from_dict()`; sampled arrays reside separately in checked `.npy` files.
Lengths use Å, momenta Å⁻¹, energies eV and angles radians.

```python
import pyrite as pr

plus = pr.TransverseStateSpec.recipe("laguerre_gauss", ell=1, p=0, waist_ang=80)
minus = pr.TransverseStateSpec.recipe("laguerre_gauss", ell=-1, p=0, waist_ang=80)
superposition = pr.TransverseStateSpec.superpose([(1, plus), (1j, minus)])
mixture = pr.TransverseStateSpec.mixture([(3, plus), (1, minus)])
restored = pr.TransverseStateSpec.from_dict(superposition.to_dict())
```

`recipe`, `from_npy` and `translated` also take semantic Pint spellings of the
unit-bearing parameters: `waist`, `scale`, `center`, `spacing`, `origin`,
`offset`, `k_radial`, `kinetic_energy`, `semiangle` and `inner_semiangle`. A
semantic name requires a quantity and conflicts with its suffixed numeric name;
the stored parameters, JSON and digest match the equivalent numeric recipe.
`parameter_quantities` returns fresh quantities in Å, Å⁻¹, eV and rad keyed by
semantic name, and `with_parameters(...)` revalidates a replacement in either
spelling. Mode orders, selectors and aberration coefficients stay numeric.

```python
from pyrite.units import ureg

probe = pr.TransverseStateSpec.recipe(
    "aperture_probe", kinetic_energy=60 * ureg.keV, semiangle=20 * ureg.mrad
)
probe.parameter_quantities["semiangle"].to("mrad")
narrower = probe.with_parameters(semiangle=10 * ureg.mrad)
```

Other analytic recipes include Gaussian, Hermite–Gaussian, Bessel–Gaussian,
Airy and aperture-probe states. Most are **diagnostic-only**, not sources for
production `simulate` runs. The restricted experimental `Beam(state=...)`
path supports a centered one-dimensional Gaussian with lattice-averaged impact
for compatible planar scenes. Do not assume a source accepted by a standalone
diagnostic is accepted by a transport scene.

`pyrite.experimental.quantum.propagate_planar_source` and
`score_planar_source` evolve and score a projected entry without transport; see
[the planar guide](../guides/experimental-planar-channeling.md#project-a-portable-source-from-python).

`incidence_rad` is the effective incidence relative to the crystal planes,
not the laboratory target tilt. `impact="fixed"` retains position-dependent
coherence; `impact="averaged"` averages over the lattice position.
`retained_population`, `free_population` and `outside_population` describe
**initial basis projection**, not physical dechanneling or escape. The
`score_planar_source` spectrum is per incident electron and is not a calibrated
detector reading. Planar reduction of 2D states does not preserve full electron
or photon orbital-angular-momentum information.

### Supported and refused paths

Planar production currently requires a supported crystal/plane, a single
non-mosaic slab, fixed-count histories, and an appropriate bound-state energy
range. It refuses adaptive precision, trajectory capture, block transport,
coherent-emission combinations, secondary/coupled radiative transport, and
unsupported geometries or beam inputs. The channeling zone and source scoring
use the CPU; subsequent native transport may use CUDA.

The source assumes a mean electron energy and simplified photon attenuation;
finite side-face escape, accumulated in-plane angular diffusion and
instrument-resolution broadening are not fully modeled. Check sensitivity to
basis size, Bloch cells, step size, beam incidence, crystal data and photon
energy grid. Treat rates and linewidths as experimental predictions requiring
independent validation.

For postprocessing, see [Working with results](../guides/working-with-results.md) and
for model/evidence limits, see the [channeling validation ledger](../validation/ledger-channeling.md).

## Axial diagnostic contracts

Use `pyrite.experimental.quantum` for direct axial diagnostics and
`pyrite.quantum` for runtime entry states. Configuration fields use bare numeric
Å, eV and radians; the direct solver configuration is outside the runtime
factory Pint boundary. Axial diagnostics remain experimental and do not enable
`Physics(channeling_model="axial")` in production transport.

### Configuration and normalization

`axial_diagnostic(config, *, entry=None, observable=None, record_metrics=False)`
runs one `AxialDiagnosticConfig`; see
[the axial guide](../guides/experimental-axial-channeling.md#run-an-mcwf-diagnostic-from-python)
for a complete call.

`mode="coherent"` performs deterministic split-step propagation; `mode="mcwf"`
uses Monte Carlo wavefunctions with thermal-diffuse and electronic scattering.
The `seed` controls independent per-electron random streams. Depths must be
increasing, nonnegative multiples of `step_ang`. Results are normalized **per
incident electron**: integrating `density_per_ang2` over the periodic cell
gives `surviving`; `result.density(conditional=True)` instead conditions on
survival. A mixed MCWF ensemble has no single wavefunction phase.

Custom Gaussian, vortex or incoherent-mixture entry states from
`pyrite.quantum` can be supplied via `entry=`.

### Interpret the recorded observables

| Output or setting | Meaning and caution |
| --- | --- |
| `density_per_ang2`, `surviving` | Spatial density and retained incident weight; not a physical exit distribution. |
| `spectral_states`, `spectral_populations` | Projection onto orthonormal finite-grid transverse eigenstates. `energy_bins` compares changing momentum sectors after electronic kicks; `states` needs a fixed sector (disable electronic kicks). Omitted projector weight is not dechanneling. |
| `record_coherence`, coherence maps | Reference-point first-order mutual coherence `g1(r,r0)`, its magnitude and phase. Low-density and poorly resolved regions are masked; it is **not** an ensemble wavefunction phase. |
| `record_metrics=True` | Ensemble purity, basis-dependent coherence and transverse-energy moments. Small MCWF ensembles require independent seed checks. |
| `oam_string`, `oam_populations` | Local disk projection onto OAM modes ℓ = −8…8; not whole-cell OAM or photon OAM. |
| `record_current`, `transverse_current_per_ang2` | Carrier-inclusive transverse probability current, incident-normalized. |
| `record_current_divergence`, `current_contour` | Discrete current divergence and outward contour flux; these describe redistribution, not absorption or escape. |

Where a supported axial line catalog exists, the app additionally displays
channeling-radiation line populations, depth-dependent rates, helicity and
far-field dipole-pattern diagnostics. These are model observables, **not**
validated detected photon spectra; a local eigenstate population histogram
is not the same as a radiation-active line population.

### Supercell and boundaries

`cells=(cx, cy)` counts crystal periods in one **periodically repeated**
supercell, while `shape=(nx, ny)` is the FFT sampling grid over the *entire*
supercell. Keep samples per cell fixed when increasing physical extent, e.g.
`(cells, shape) = ((1,1),(24,24))`, then `((2,2),(48,48))`.
Raising `cells` without increasing `shape` reduces spatial resolution.

Flux crossing an internal cell boundary is redistribution. Flux leaving one
edge of the periodic supercell re-enters at the opposite edge. A cropped map
is only a viewport, not an open boundary. An
above-barrier finite-grid component or a recorded `numerical_sink` is **not**
a validated quantum-to-classical handoff.

For contour measurements, `current_contour=(x, y, width, height)` is in grid
pixels; the reported four signed sides are `(-x, +x, -y, +y)` and sum to net
outward flow per unit propagation depth. The local finite-grid continuity
residual is not guaranteed to vanish at fixed FFT resolution.

### CUDA execution

Install the `nvidia` extra and set `backend="cuda"` on the axial diagnostic
configuration for MCWF. CUDA requires available hardware and does not fall
back silently. Results return as NumPy arrays; the solver uses float64 and
complex128 arithmetic.

The lower-level `AxialMCWF(..., backend="cuda")` API additionally supports
`batch_size` and `memory_budget_bytes` limits on device allocations.
CPU and CUDA use different random
draw sequences, so compare ensemble statistics rather than individual tracks.
CUDA callbacks must explicitly support device arrays; an incompatible callback
fails rather than silently transferring wavefunctions to the host. Small jobs
may run faster on CPU. GPU acceleration changes neither the physical model nor
its validation status.

For larger runs, see [Running on a cluster](../guides/running-on-a-cluster.md). For
assumptions and checks, see the [channeling validation ledger](../validation/ledger-channeling.md).
