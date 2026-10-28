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

The notebook shares this page's computation and plots. Change **Depth** or
**Gaussian waist**, then select **Recompute diagnostic**; changes submit
together. The initial calculation uses 500 Å depth and a 0.5 Å waist.

```{literalinclude} ../examples/experimental_axial_channeling.py
:language: python
:start-at: import numpy as np
:end-at:     return result, overlap
```

```{eval-rst}
.. plot::
   :filename-prefix: experimental-axial-density
   :alt: Three incident-normalized transverse probability-density maps of the same Gaussian at zero, 25 and 50 nanometers in periodic silicon.
   :caption: Coherent Si[001] evolution. Shared color scale; no scattering or absorption.

   from _tutorials import experimental_axial_channeling
   experimental_axial_channeling(view="density")
```

(interpret-the-recorded-observables)=
The density uses **incident electron⁻¹ Å⁻²**. Its integral over the full periodic
grid remains **1.0000** here, with no conditioning on survival. The squared
overlap with the initial
normalized state falls to **0.4469**. A changing density or overlap does not
imply population loss.

```{eval-rst}
.. plot::
   :filename-prefix: experimental-axial-evolution
   :alt: Retained periodic-grid norm stays at one while the squared initial-state overlap decreases to about 0.447 over 50 nanometers.
   :caption: Pure-state overlap and full-grid norm from the same wavefunction history.

   from _tutorials import experimental_axial_channeling
   experimental_axial_channeling(view="evolution")
```

(supercell-boundaries-and-convergence)=
## Explore and interpret

`cells=(2, 2)` counts crystal periods; `shape=(64, 64)` samples the entire
supercell. The solver periodizes and normalizes the entry on that grid; the
cached pre-normalization sampled norm is **1.0000**. Grid normalization does
not establish an isolated-beam approximation or convergence. Refine supercell
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
records the source digest, configuration and generating revision. Normal docs
builds render these arrays without solving.

(cuda-execution)=
For MCWF scattering, spectral/coherence/current observables and CUDA contracts,
use the [axial diagnostic reference](../reference/quantum-api.md#axial-diagnostic-contracts)
or `pyrite app channeling launch`, **Axial (experimental)**. Direct solver
configuration uses bare numeric Å, eV and radians; runtime state factories also
support the [quantity boundary](../reference/quantum-api.md#runtime-quantity-inputs).
See the [channeling evidence ledger](../validation/ledger-channeling.md)
before physical interpretation, and [cluster execution](running-on-a-cluster.md)
for larger calculations.
