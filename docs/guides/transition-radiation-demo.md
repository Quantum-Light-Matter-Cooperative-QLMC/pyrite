(tr-and-ctr-demo)=
# Compare straight-track and transported TR/CTR

Compute film transition radiation (TR) and bunch coherent transition radiation
(CTR) for straight tracks, then rerun both with transported electrons that
scatter and lose energy. The figures below are actual outputs of the packaged
marimo app, not measurements. To learn TR and CTR step by step through the
Python API, follow [Transition and coherent transition radiation](../notebooks/specialized/10-transition-radiation.ipynb).

## Compute the straight-track comparison

From a source checkout:

```bash
uv sync --extra apps
uv run pyrite app transition-radiation launch
```

For an installed distribution, install `pyrite-mc[apps]`, then use the same
commands without `uv run`. The initial analytic calculation needs no checkpoints,
transport tables or GPU.

The default source is a **30 keV beam** crossing a free-standing **20 nm silicon
film** tilted **5°**. The spectrum looks **150° from the beam**, over **200–1000 eV**.
An illustrative Gaussian bunch has **1 as RMS duration**, **1 nm FWHM spot** and
**0.0001 pC charge**, about **624 physical electrons**.

```{eval-rst}
.. plot::
   :filename-prefix: tr-ctr-analytic
   :alt: Analytic film TR and Gaussian CTR spectra at 150 degrees, followed by their 500 eV angular cuts from 100 to 175 degrees.
   :caption: Default straight-track calculation: film/interface spectra and a 500 eV angular cut.

   from _tutorials import transition_radiation_demo
   transition_radiation_demo()
```

The film curve includes entry/exit interference and absorption; its repeated
minima are physical fringes. The CTR curve also includes the Gaussian
time/position form factor: change the bunch inputs and submit
**Update comparison** to recompute it. The default 1 eV and 0.25° grids are
visualization samples; refine them before interpreting fringe minima or
integrating a yield.

## Add transported electrons

[Fetch the reference tables](getting-started.md#fetch-reference-data),
then click **Run transported comparison** in the notebook. It runs history TR
and bunch CTR with the same default scene, using **2048 line histories and 4096
continuum histories per run**. TR is scored from the continuum population.
The sampled trajectories include scattering and energy loss; bunch CTR samples
the time/position offsets rather than multiplying by an analytic form factor.

```{eval-rst}
.. plot::
   :filename-prefix: tr-ctr-transport
   :alt: Seeded transported history TR and signed bunch CTR spectra, with other CXR, combined CXR and dotted straight-track references on symmetric log axes.
   :caption: Actual CPU transport at seed 1: 2048 line and 4096 continuum histories per run, sampled every 1 eV. Dotted curves reproduce the analytic comparison above.

   from _tutorials import transition_radiation_demo
   transition_radiation_demo(view="transport")
```

## Save the cases as checkpoints

To keep the same default cases as campaign checkpoints:

```bash
uv run pyrite run tr_demo
uv run pyrite run ctr_demo
uv run pyrite app analysis launch silicon
```

Select the corresponding identity-qualified checkpoint in the analysis app.
`tr_demo` uses history coherence; `ctr_demo` uses bunch coherence. Clone a profile
before editing it. See [controls, table requirements and profile storage](../reference/tr-ctr-app.md).

## Check units before comparing

Every curve is **photons/(incident electron · eV · sr)** at one observation
direction. The app's **per bunch** option multiplies by physical population N,
independent of Monte Carlo history count. For identical tracks the per-electron
ensemble multiplier is `1 + (N - 1)|F|²`; an in-phase point bunch gives N² per
bunch. The transported CTR estimator is signed and can have negative bins when
the sample does not resolve the bunch excess. See the
[CTR estimator and validated limits](../physics/radiation-physics/transition-radiation.md#ctr-from-transported-crossings).

**Combined CXR** adds TR once to the other line-grid components. It excludes the
separately gridded bremsstrahlung continuum. This comparison uses incoherent
PXR/CBS; coherent TR–line interference and diffracted TR are outside it.

## Export a static page

```bash
uv run pyrite app transition-radiation export --stem tr_ctr_demo
```

Open `pyrite-output/results/tr_ctr_demo.html` for the initial analytic plots;
static HTML does not execute controls or transport. The
{download}`cached guide output <../examples/data/transition_radiation_demo.json>`
records arrays, settings, warnings and transport provenance. Documentation builds
only render those arrays. The maintained notebook source is
{download}`transition_radiation_app.py <../../src/pyrite/apps/transition_radiation_app.py>`.

The planar model assumes flat, sharp, laterally infinite interfaces and local
isotropic dielectric response; the independent-atom response is crude near plasma
energies and absorption edges. These are directional source densities, not
pixel counts or detector-integrated fringes. See the
[model assumptions](../physics/radiation-physics/transition-radiation.md#exact-planar-stack-tr),
[production restrictions](../physics/radiation-physics/transition-radiation.md#run-case-component)
and [detector integration](../physics/radiation-physics/transition-radiation.md#detector-channels-and-acceptance).
Optical/IR diagnostics and the separately triggered **experimental, unverified**
DTR view are covered in [Radiation diagnostics](radiation-diagnostics.md).
