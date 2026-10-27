# Experimental planar channeling

Project a 1D Gaussian into Si(110), evolve its transverse populations, and plot
forward channeling radiation at 17 MeV. This direct CPU diagnostic needs no
transport tables or GPU.

:::{warning}
**Experimental — not rigorously validated.** These are restricted model outputs,
not experimental data or detector predictions. The small basis and fixed grids
illustrate the workflow; no convergence or predictive-accuracy claim is made.
:::

(run-planar-channeling)=
(specify-and-inspect-quantum-entrance-states)=
## Project, evolve and score

From a source checkout, run the maintained example or open its reactive notebook:

<!-- verify: skip (downloadable Python example and marimo editor; checked by notebook export) -->
```bash
uv sync --extra apps
uv run python docs/examples/experimental_planar_channeling.py
uv run marimo edit docs/examples/planar_channeling_notebook.py
```

The notebook uses the same computation and plot functions as this page. Change
**Depth** or **Effective incidence**, then select **Recompute diagnostic**;
changes submit together. The initial calculation uses 2,000 Å depth and zero
effective refracted incidence to the planes. Incidence is not laboratory tilt.

```{literalinclude} ../examples/experimental_planar_channeling.py
:language: python
:start-at: import numpy as np
:end-at:     return source, entry, history, radiation
```

```{eval-rst}
.. plot::
   :filename-prefix: experimental-planar-populations-spectrum
   :alt: Incident-normalized retained, below-barrier and three band populations versus depth, followed by the forward total and capture radiation spectra.
   :caption: Fixed-impact Gaussian in Si(110), 0.2 µm depth. Population history and forward radiation from the same entry density; absorption disabled.

   from _tutorials import experimental_planar_channeling
   experimental_planar_channeling()
```

At the final depth, **0.9667** of the incident population remains in the retained
subspace and **0.7970** lies below the barrier. Populations are unconditioned on
survival; initial retained/free/outside projection weights are not evolved escape.
The spectrum includes capture and is normalized to **photons/(incident electron
· eV · sr)** at θ = 0. It has no aperture integration, instrument response or
charge scaling. Zero in-plane divergence and no stopping are used here.

(supported-and-refused-paths)=
## Explore and interpret

Refine basis size, Bloch cells, depth step and photon grid independently before
interpreting rates or linewidths. The source uses a mean electron energy;
simplified attenuation, finite side-face escape, accumulated in-plane diffusion
and instrument broadening limit the broader model. A planar reduction does not
retain full electron or photon OAM information.

Download both {download}`the notebook <../examples/planar_channeling_notebook.py>`
and {download}`its shared example <../examples/experimental_planar_channeling.py>`
into one directory to run them outside the checkout with `pyrite-mc[apps]` installed.
The {download}`cached output <../examples/data/experimental_planar_channeling.json>`
records the source digest, settings, entry weights and generating revision.
Normal docs builds render these arrays without running a solver.

For the separate production `simulate`/profile workflow, portable state recipes,
quantity inputs and refused combinations, use the
[quantum reference](../reference/quantum-api.md#planar-transport-and-source-contracts).
A direct diagnostic accepting a source does not imply production admission:
`Beam(state=...)` admits only the restricted centered 1D Gaussian with
lattice-averaged impact for compatible planar scenes. See the
[channeling evidence ledger](../validation/ledger-channeling.md) before physical interpretation.
