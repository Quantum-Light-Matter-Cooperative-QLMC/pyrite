# Experimental axial quantum channeling and diagnostics

:::{warning}
**Experimental — not rigorously validated.** Axial propagation, stochastic
scattering, state/coherence observables and radiation diagnostics are research
models. The axial solver is not integrated into standard transport runs:
`Physics(channeling_model="axial")` is refused. Numerical sinks and periodic
boundary crossings are **not** measured dechanneling or physical escape.
:::

Use the channeling app (`pyrite app channeling launch`, **Axial
(experimental)**) for interactive inspection, or call `axial_diagnostic`
directly for controlled studies. The API does not require the plotting extras.

## Run an axial diagnostic

```python
from pyrite.montecarlo.channeling.axial_diagnostic import (
    AxialDiagnosticConfig,
    axial_diagnostic,
)

config = AxialDiagnosticConfig(
    crystal="silicon",
    axis=(0, 0, 1),
    kinetic_energy_eV=5e6,
    cells=(2, 2),
    shape=(48, 48),
    depths_ang=(0, 250, 500),
    step_ang=5,
    mode="mcwf",
    n_electrons=16,
    seed=17,
    spectral_states=16,
    spectral_representation="energy_bins",
    record_coherence=True,
    record_current=True,
    record_current_divergence=True,
    current_contour=(0, 0, 12, 12),
)
result = axial_diagnostic(config, record_metrics=True)
print(result.surviving, result.density_per_ang2.shape)
```

`mode="coherent"` performs deterministic split-step propagation; `mode="mcwf"`
uses Monte Carlo wavefunctions with thermal-diffuse and electronic scattering.
The `seed` controls independent per-electron random streams. Depths must be
increasing, nonnegative multiples of `step_ang`. Results are normalized **per
incident electron**: integrating `density_per_ang2` over the periodic cell
gives `surviving`; `result.density(conditional=True)` instead conditions on
survival. A mixed MCWF ensemble has no single wavefunction phase.

Custom Gaussian, vortex or incoherent-mixture entry states can be supplied via
`entry=` using `pyrite.montecarlo.channeling.states`. For example:

```python
from pyrite.montecarlo.channeling.states import laguerre_gauss, mixture

entry = mixture(
    [
        (0.5, laguerre_gauss(1, waist_ang=0.5)),
        (0.5, laguerre_gauss(-1, waist_ang=0.5)),
    ]
)
result = axial_diagnostic(config, entry=entry, record_metrics=True)
```

## Interpret the recorded observables

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

## Supercell, boundaries and convergence

`cells=(cx, cy)` counts crystal periods in one **periodically repeated**
supercell, while `shape=(nx, ny)` is the FFT sampling grid over the *entire*
supercell. Keep samples per cell fixed when increasing physical extent, e.g.
`(cells, shape) = ((1,1),(24,24))`, then `((2,2),(48,48))`.
Raising `cells` without increasing `shape` reduces spatial resolution.

Flux crossing an internal cell boundary is redistribution. Flux leaving one
edge of the periodic supercell re-enters at the opposite edge. A cropped map
is only a viewport, not an open boundary. Check cell count, samples per cell,
propagation step, observation depths and ensemble size independently. An
above-barrier finite-grid component or a recorded `numerical_sink` is **not**
a validated quantum-to-classical handoff.

For contour measurements, `current_contour=(x, y, width, height)` is in grid
pixels; the reported four signed sides are `(-x, +x, -y, +y)` and sum to net
outward flow per unit propagation depth. The local finite-grid continuity
residual is not guaranteed to vanish at fixed FFT resolution.

## CUDA execution

Install the `nvidia` extra and set `backend="cuda"` on the axial diagnostic
configuration for MCWF. CUDA requires available hardware and does not fall
back silently. Results return as NumPy arrays; the solver uses float64 and
complex128 arithmetic.

For example, to repeat the diagnostic above on CUDA:

```python
from dataclasses import replace

gpu_result = axial_diagnostic(replace(config, backend="cuda"), record_metrics=True)
```

The lower-level `AxialMCWF(..., backend="cuda")` API additionally supports
`batch_size` and `memory_budget_bytes` limits on device allocations.
CPU and CUDA use different random
draw sequences, so compare ensemble statistics rather than individual tracks.
CUDA callbacks must explicitly support device arrays; an incompatible callback
fails rather than silently transferring wavefunctions to the host. Small jobs
may run faster on CPU. GPU acceleration changes neither the physical model nor
its validation status.

For larger runs, see [Running on a cluster](running-on-a-cluster.md). For
assumptions and checks, see the [channeling validation ledger](../validation/ledger-channeling.md).
