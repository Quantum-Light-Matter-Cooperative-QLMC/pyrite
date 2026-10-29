(experimental-planar-channeling)=
# Project and evolve a planar channeling state (experimental)

Project a 1D Gaussian into Si(110), evolve its transverse populations, and plot
forward channeling radiation at 17 MeV. This direct CPU diagnostic needs no
transport tables or GPU.

For a standalone public-API walkthrough, open [Planar channeling and radiation](../notebooks/quantum/13-planar-channeling-radiation.ipynb).

:::{warning}
**Experimental — not rigorously validated.** These are restricted model outputs,
not experimental data or detector predictions. The small basis and fixed grids
illustrate the workflow; no convergence or predictive-accuracy claim is made.
:::

## Project, evolve and score

From a source checkout, run the maintained example or open its reactive notebook:

<!-- verify: skip (downloadable Python example and marimo editor; checked by notebook export) -->
```bash
uv sync --extra apps
uv run python docs/examples/experimental_planar_channeling.py
uv run marimo edit docs/examples/planar_channeling_notebook.py
```

The reactive example uses its shared script for computation and plots. Change
**Depth** or **Effective incidence**, then select **Recompute diagnostic**;
changes submit together. The initial calculation uses 2,000 Å depth and zero
effective refracted incidence to the planes. Incidence is not laboratory tilt.

Interpret retained and below-barrier populations without conditioning on
survival; initial retained/free/outside projection weights are not evolved escape.
The spectrum includes capture and is normalized to **photons/(incident electron
· eV · sr)** at θ = 0. It has no aperture integration, instrument response or
charge scaling. Zero in-plane divergence and no stopping are used here.

(specify-and-inspect-quantum-entrance-states)=
## Project a portable source from Python

Without the example script, project, evolve and score a portable 1D source
directly. Units are Å, eV and radians; `incidence_rad` is relative to the
planes, not the laboratory tilt.

```python
import numpy as np
import pyrite as pr
from pyrite.experimental.quantum import propagate_planar_source, score_planar_source

source = pr.TransverseStateSpec.recipe("gaussian", waist_ang=1, dimension=1)
setup = pr.QuantumSetup(
    region=pr.QuantumRegion(kind="planar", plane_hkl=(1, 1, 0)),
    dynamics=pr.QuantumDynamics(kind="channeling"),
    numerics=pr.QuantumNumerics(basis_size=8, bloch_cells=4, step_ang=50),
)
entry = setup.project_state(
    source,
    crystal="silicon",
    kinetic_energy_eV=17e6,
    incidence_rad=0,
    impact="fixed",
)
evolution = propagate_planar_source(entry, depth_ang=500)
spectrum = score_planar_source(
    entry,
    depth_ang=500,
    photon_energy_eV=np.linspace(1_000, 50_000, 500),
    theta_rad=0,
    absorption=False,
)
print(entry.retained_population, evolution.trace[-1], spectrum.spectrum.shape)
```

Use `impact="averaged"` to average over lattice position. The
[entrance-state contract](../reference/quantum-api.md#specify-and-inspect-quantum-entrance-states)
defines the projection populations and spectrum normalization.

(run-planar-channeling)=
## Add planar channeling to a transport run

The production `simulate`/profile path is separate from the direct diagnostic
above. Install the tables, including SBETHE, and run the packaged 17 MeV,
2 µm silicon example:

```bash
pyrite tables fetch
pyrite tables fetch sbethe
pyrite profile show channeling_demo
pyrite run channeling_demo
pyrite app analysis launch silicon
```

To change the beam, target, detector or numerical settings, clone it:

```bash
pyrite profile create my_channeling --from channeling_demo
miller_negative=-1
pyrite physics set my_channeling \
  --channeling-model planar --channeling-plane 1 "$miller_negative" 0 \
  --radiative-model uncoupled --emission incoherent
pyrite run my_channeling
```

From Python, select the same zone in `Physics`:

```python
import numpy as np
import pyrite as pr

result = pr.simulate(
    pr.Beam(energy_keV=17_000, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
    pr.Slab(
        "silicon",
        thickness_ang=20_000,
        tilt_deg=3,
        tilt_azim_deg=45,
        footprint=pr.Footprint(5, 5),
    ),
    pr.Detector(
        0,
        acceptance=pr.SensorAcceptance(
            distance_mm=1_000,
            width_mm=3,
            height_mm=3,
        ),
    ),
    physics=pr.Physics(
        channeling_model="planar",
        channeling_plane=(1, -1, 0),
        radiative_model="uncoupled",
    ),
    numerics=pr.Numerics(
        cxr_trials=200,
        brem_trials=200,
        energy_bins=pr.EnergyBins(
            cxr=np.arange(1_000, 24_000, 50),
            brem=np.arange(1_000, 17_000_000, 500),
        ),
    ),
)
cr = result.channeling  # photons / (incident electron · eV · sr), or None
combined = result.combined_cxr()  # includes CR
```

Check convergence in `channeling_basis_size`, `channeling_bloch_cells`,
`channeling_dz_ang` and `channeling_zone_max_ang` before interpreting CR.
The [supported and refused paths](../reference/quantum-api.md#supported-and-refused-paths)
list the scenes and options the zone rejects.

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
This fixed output record remains available for reproducing the example.

For portable state recipes, quantity inputs and refused combinations, use the
[quantum reference](../reference/quantum-api.md#planar-transport-and-source-contracts).
A direct diagnostic accepting a source does not imply production admission:
`Beam(state=...)` admits only the restricted centered 1D Gaussian with
lattice-averaged impact for compatible planar scenes. See the
[channeling evidence ledger](../validation/ledger-channeling.md) before physical interpretation.
