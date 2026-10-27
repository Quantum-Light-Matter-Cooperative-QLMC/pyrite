(import-a-gpt-electron-beam)=
# Import and visualize an electron beam

Import correlated electron records, inspect their phase space and simulate
their radiation from a thin HOPG target. The input is a **synthetic
demonstration**, not experimental data or a GPT simulation.

Install `pyrite-mc[plots]` and [fetch the reference tables](getting-started.md#install-and-fetch-reference-data).
Save {download}`the runnable script <../examples/gpt_gdf_beams.py>` and
{download}`synthetic_beam.gdf <../examples/data/synthetic_beam.gdf>` in the same
directory, then run:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_MC_BACKEND=cpu python gpt_gdf_beams.py --input synthetic_beam.gdf
```

GPT is not required. The [fixture provenance and license](../examples/data/synthetic_beam.md)
describe the deterministic generator, available in the script with
`--make-fixture`. EasyGDF is already a PyRITE dependency; its
[raw block API](https://github.com/electronsandstuff/easygdf#quickstart)
supplies the arrays for plotting. PyRITE validates the electron records when
building the simulation.

```{literalinclude} ../examples/gpt_gdf_beams.py
:language: python
:start-after: tutorial-start
:end-before: tutorial-end
```

`calculate(path)` returns the input fields and the simulated result. The script
plots `x × 10⁶` against `1000 × Bx/Bz`, colors each point by beta-derived
kinetic energy, and weights the energy histogram by `nmacro/sum(nmacro)`.
It plots `result.cxr.incoherent` and `result.brem` on their native photon grids.

```{eval-rst}
.. plot::
   :alt: Synthetic lab-frame records show positive x-angle and x-energy correlations, a weighted electron-energy distribution around 30 keV, and actual simulated PXR/CBS and bremsstrahlung spectra.
   :caption: Original synthetic input and actual seeded CPU radiation output. Marker area represents relative record probability. The spectrum is a source density at 135°, without detector response or solid-angle integration.

   from _tutorials import gpt_gdf_beams
   gpt_gdf_beams()
```

The tilted cloud shows a position–angle correlation; its color gradient shows
the energy correlation. Import preserves these as **whole records**. Drawing
position, angle and energy independently would describe a different beam.

(choose-the-snapshot-and-coordinate-origin)=
## From lab snapshot to entrance

This snapshot is at lab `z = 0.1 m`, the chosen target-origin z coordinate.
The first two panels show the **lab snapshot**, before entrance projection.
`Bx`, `By`, `Bz` are velocities divided by c, **not magnetic fields**.
`Bx/Bz` is a slope; its value in mrad approximates the angle only at small
divergence.

The target is tilted by 30°. PyRITE subtracts the target origin, rotates both
positions and velocities to the sample frame, then intersects each ray with
entrance `z = 0`. Consequently these raw lab x coordinates are not the
transport entrance x coordinates. Only for an untilted target and a snapshot
already on its entrance would x/y be unchanged. This signed, field-free
extrapolation is not beamline or space-charge tracking; no transverse
recentering is inferred. See the [projection contract](../reference/gdf-beams.md#choose-the-snapshot-and-coordinate-origin)
and [independent validation](../validation/beam-transport/gpt-gdf-injection.md).

(configure-normalization)=
## Probabilities, charge and histories

The file has **96 records**, unequal positive `nmacro` weights and **1 pC**
total represented charge. With `gdf_charge` and 1 MHz repetition rate, the
physical average current is **1 µA**. PyRITE samples complete records with
replacement, using normalized `nmacro` probabilities; it does not multiply
each sampled contribution by `nmacro` again. The 64 histories per source
component are a numerical sample, distinct from record count and physical
electrons per bunch.

The displayed radiation remains **photons/(incident electron · eV · sr)**;
charge and repetition rate do not rescale these source densities. To obtain
rates, use the [current conversion](../physics/radiation-physics/spectral-observables.md)
once. The alternative `pyrite_current` mode uses configured charge/current
normalization while retaining the same relative record probabilities. See
[normalization modes](../reference/gdf-beams.md#configure-normalization).

(errors-and-limits)=
## Sampling and limits

The input generator uses seed `42`; this single-scene transport resolves seed
`1`, recorded with the [cached output and provenance](../examples/data/gpt_gdf_beams.json).
The example's small history count and two reflection families do not establish
convergence, uncertainty or beam quality. Refine sampling, photon grids and
reflection coverage before quantitative use. See
[capabilities](../physics/capabilities.md) for model and validation limits.

Imported beams currently support local execution, flat slab/stack entrances
and incoherent emission. The native energy distribution replaces the scalar
scene placeholder; `shape_only=True` instead preserves directions and weights
while assigning the selected energy. It preserves divergence, not momentum.
Do not combine imported phase space with analytic spot, energy-spread or
longitudinal-distribution settings.

(export-and-inspect)=
(catalog-example)=
(cli-example)=
(inspect-coordinates-and-select-a-screen)=
(use-only-the-imported-beam-shape)=
## Use your own output

Inspect a native time snapshot with `pyrite beam gdf inspect beam.gdf`; use
`--time-s` to select among snapshots, or `--screen-position-m` for a screen.
The plotting script above expects the supplied single-time fixture; arbitrary
files need explicit block selection. The [GDF input reference](../reference/gdf-beams.md)
owns required arrays/units, time and screen tolerances, coordinate placement,
catalog/CLI recipes, shape-only mode and error remedies.

Documentation builds render checked JSON only. Input and example SHA-256
checks reject stale output; explicit regeneration runs import and transport:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
MPLCONFIGDIR=/tmp/pyrite-matplotlib PYRITE_MC_BACKEND=cpu \
  uv run python docs/_tutorials.py --regenerate gpt_gdf_beams
```
