(experimental-axial-quantum-channeling-and-diagnostics)=
(experimental-axial-channeling)=
# Evolve an axial channeling state (experimental)

Evolve a 2D Gaussian in Si[001] at 5 MeV, then inspect its density and overlap
with the initial state. This coherent CPU diagnostic needs no transport tables
or GPU.

For a standalone public-API walkthrough, open [Axial channeling and wavefunction evolution](../notebooks/quantum/14-axial-channeling-evolution.ipynb).

:::{warning}
**Experimental — not rigorously validated.** These are finite-grid model
outputs, not experimental data or measured dechanneling. No convergence or
predictive-accuracy claim is made. Production transport refuses
`Physics(channeling_model="axial")`.
:::

(run-an-axial-diagnostic)=
## Evolve and plot

From a source checkout, run the maintained example or open its reactive notebook:

<!-- verify: skip (downloadable Python example and marimo editor; checked by notebook export) -->
```bash
uv sync --extra apps
uv run python docs/examples/experimental_axial_channeling.py
uv run marimo edit docs/examples/axial_channeling_notebook.py
```

The reactive example uses its shared script for computation and plots. Change
**Depth** or **Gaussian waist**, then select **Recompute diagnostic**; changes submit
together. The initial calculation uses 500 Å depth and a 0.5 Å waist.

(interpret-the-recorded-observables)=
The density uses **incident electron⁻¹ Å⁻²**. Integrate over the full periodic
grid to check norm, without conditioning on survival. Compare squared overlap
with the initial normalized state separately: changing density or overlap does
not imply population loss.

(supercell-boundaries-and-convergence)=
## Explore and interpret

`cells=(2, 2)` counts crystal periods; `shape=(64, 64)` samples the entire
supercell. The solver periodizes and normalizes the entry on that grid; the
cached output records the pre-normalization sampled norm. Grid normalization
does not establish an isolated-beam approximation or convergence. Refine supercell
extent, samples per cell and propagation step independently. Increasing cell
count without increasing grid shape reduces resolution.

One edge of the periodic supercell connects to its opposite edge. Boundary
crossings, omitted spectral-projector weight and numerical sinks are not
physical escape or validated quantum-to-classical handoff. This coherent example
has no scattering or absorption, so retained norm is a unitary-grid check,
not a scattering-survival probability. It uses the whole finite FFT grid,
without a retained eigenstate cutoff. Overlap is a pure-state diagnostic;
a mixed ensemble has no single wavefunction phase.

Download both {download}`the notebook <../examples/axial_channeling_notebook.py>`
and {download}`its shared example <../examples/experimental_axial_channeling.py>`
into one directory to run them outside the checkout with `pyrite-mc[apps]` installed.
The {download}`cached output <../examples/data/experimental_axial_channeling.json>`
records the source digest, configuration and generating revision. This fixed
output record remains available for reproducing the example.

## Run an MCWF diagnostic from Python

Add thermal-diffuse and electronic scattering with `mode="mcwf"` and record
spectral, coherence and current observables. Sixteen electrons demonstrate the
interface only; check results against independent seeds.

```python
from pyrite.experimental.quantum import (
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

To start from a vortex mixture instead of the default entry state:

```python
from pyrite.quantum import laguerre_gauss, mixture

entry = mixture(
    [
        (0.5, laguerre_gauss(1, waist_ang=0.5)),
        (0.5, laguerre_gauss(-1, waist_ang=0.5)),
    ]
)
result = axial_diagnostic(config, entry=entry, record_metrics=True)
```

(cuda-execution)=
With the `nvidia` extra and a GPU, repeat the MCWF run on CUDA. CPU and CUDA
use different random draws, so compare ensemble statistics, not tracks:

```python
from dataclasses import replace

gpu_result = axial_diagnostic(replace(config, backend="cuda"), record_metrics=True)
```

The [observable table](../reference/quantum-api.md#interpret-the-recorded-observables)
defines each recorded output.

For MCWF scattering, spectral/coherence/current observables and CUDA contracts,
use the [axial diagnostic reference](../reference/quantum-api.md#axial-diagnostic-contracts)
or `pyrite app channeling launch`, **Axial (experimental)**. Direct solver
configuration uses bare numeric Å, eV and radians; runtime state factories also
support the [quantity boundary](../reference/quantum-api.md#runtime-quantity-inputs).
See the [channeling evidence ledger](../validation/ledger-channeling.md)
before physical interpretation, and [cluster execution](running-on-a-cluster.md)
for larger calculations.
