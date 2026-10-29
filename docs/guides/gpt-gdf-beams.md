(import-a-gpt-electron-beam)=
# Import and visualize an electron beam

Import correlated electron records while preserving their positions, directions,
energies and relative weights. For a guided phase-space visualization and source
simulation, open [Importing a realistic electron beam](../notebooks/specialized/09-importing-electron-beams.ipynb).

Install `pyrite-mc[plots]` and [fetch the reference tables](getting-started.md#fetch-reference-data)
when running transport. Inspect the input before choosing its snapshot:

```bash
pyrite beam gdf inspect beam.gdf
```

Use `--time-s` for a time snapshot or `--screen-position-m` for a screen.
The [GDF input reference](../reference/gdf-beams.md) owns required arrays, units,
selectors and the `Beam.from_gdf` interface. GPT is not required to read a file.

For a fixed reproduction, save {download}`the runnable script <../examples/gpt_gdf_beams.py>`
and {download}`synthetic_beam.gdf <../examples/data/synthetic_beam.gdf>` together:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_MC_BACKEND=cpu python gpt_gdf_beams.py --input synthetic_beam.gdf
```

This fixture is a **synthetic demonstration**, not experimental data or a GPT
simulation. Its [provenance and license](../examples/data/synthetic_beam.md)
describe the deterministic generator (`--make-fixture` in the script).
The script expects the supplied single-time fixture; arbitrary files need
explicit block selection. The notebook owns the guided plots and interpretation.

(choose-the-snapshot-and-coordinate-origin)=
## From lab snapshot to entrance

The fixture is a lab snapshot at `z = 0.1 m`, chosen to match the
target-origin coordinate. Its plotted lab phase space precedes entrance projection.
GPT's `Bx`, `By` and `Bz` are velocity components
divided by c, *not magnetic fields*; `Bx/Bz` approximates the x angle only
for small divergence.

For the 30°-tilted target, PyRITE translates to the specified origin,
rotates positions and velocities into the sample frame, then extrapolates each
ray to the entrance plane. The plotted lab x positions therefore need not equal
transport entrance x. This signed, field-free projection does not model
beamline or space-charge tracking and never infers transverse recentering.
For selectors, coordinate validation and screen output, see the
[GDF projection contract](../reference/gdf-beams.md#choose-the-snapshot-and-coordinate-origin).

(configure-normalization)=
## Preserve weights without double-counting charge

The fixture contains **96 correlated records** with unequal positive
`nmacro` weights and **1 pC** of represented charge. In `gdf_charge` mode,
1 MHz repetition rate therefore gives **1 µA** mean current. Each of the 64
sampled histories per source component selects a *whole record* with
replacement, using normalized `nmacro` probabilities; no second weight
multiplier is applied.

Source spectra remain **photons/(incident electron · eV · sr)**, not rates or
detected counts. Apply the [current conversion](../physics/radiation-physics/spectral-observables.md)
once when reporting physical rates. Alternatively,
`pyrite_current` uses explicitly configured charge and current while retaining
relative particle weights. See the [normalization contract](../reference/gdf-beams.md#configure-normalization)
for defaults, supported modes and identity behavior.

(errors-and-limits)=
## Check sampling and supported geometry

The synthetic generator uses seed `42`; the fixed script resolves transport
seed `1`, recorded in its {download}`cached provenance <../examples/data/gpt_gdf_beams.json>`.
The small history count and two reflection families do **not** demonstrate
convergence or beam quality. Refine sampling and the photon grid before
quantitative use; consult the [capability matrix](../physics/capabilities.md).

Imported records currently support local execution, flat slab/stack entrances
and incoherent emission. The imported energies replace the scalar scene
placeholder; `shape_only=True` instead assigns the selected energy while
retaining imported positions, directions and weights. Do not combine GDF
phase space with analytic spot, spread or longitudinal settings. The
[GDF error and restriction reference](../reference/gdf-beams.md#errors-and-limits)
owns input validation, field tolerances and unsupported combinations.

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
