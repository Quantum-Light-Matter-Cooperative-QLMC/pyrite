# Save and inspect electron trajectories

Keep the complete electron histories of a run, then list, visualize, rescore or
reduce them without transporting again. Checkpoints keep spectra only; a capture
is a separate, opt-in HDF5 artifact. File layouts, field roles, selection rules
and refusals are specified in the
[trajectory capture reference](../reference/trajectory-workflows.md).

## Capture a run

The bundled `trajectory_demo` profile runs 100 fixed histories of 30 keV
electrons in a 2 µm HOPG slab, tilted 45° at azimuth 180°, with a 1 µm FWHM
beam. Its counts suit visualization, not converged spectra.

```bash
pyrite run trajectory_demo --trajectories
pyrite trajectories ls
```

Each transported case writes one file under `pyrite-output/trajectories/`.
For another profile, first set fixed counts: adaptive counts cannot be
captured. See [capture restrictions](../reference/trajectory-workflows.md#save-electron-trajectories)
for cached cases, overwrites and remote runs.

## Describe or reopen a capture

```bash
pyrite trajectories info pyrite-output/trajectories/<stem>/<config>-<digest>/E0_30keV.h5
```

From Python, the same file returns the arrays `simulate_trajectories` produced:

```python
from pyrite.montecarlo.trajectories import read_trajectory_artifact

artifact = read_trajectory_artifact("pyrite-output/trajectories/<stem>/.../E0_30keV.h5")
segs = artifact.transport
artifact.units["r_mid"]  # "angstrom"
```

## View histories in ParaView

Export a few whole histories with their lab-frame scene:

```bash
pyrite trajectories export pyrite-output/trajectories/ --first 20 --scene --out-dir traj-vtk/
```

1. **File → Open** the close-up `<case>.vtm` (without `.instrument` in its
   name), then **Apply**. Use **Extract Block** to select `tracks`.
2. Colour by the **cell** array `E_start_keV` and rescale to the data range.
3. **Threshold** cell `generation` from `1` to isolate transported secondaries,
   `is_vacuum` to `1` for groove-gap flights, or `electron_id` for one history.
4. Open `<case>.instrument.vtm` in a separate render view: its coordinates are
   millimetres, the track scene's are angstrom.

To apply the same colouring and filters automatically, run the shipped preset
with ParaView's Python:

<!-- verify: skip (needs ParaView pvbatch, not a pyrite/pyrite-dev command) -->
```bash
SCRIPT=$(pyrite trajectories export --paraview-script)
pvbatch "$SCRIPT" case.vtm --color generation --threshold electron_id 0 9 --screenshot closeup.png
```

In the GUI, use **Macros > Import new macro...** with the same file. If each
shower looks tiny beside a millimetre-wide beam spot, export `--first 1` or
`--history ID` instead. For large captures, select `--sample N --seed S` or
`--track ID` so that ParaView reads only those rows.

## Rescore spectra after a spectrum-phase change

```bash
pyrite trajectories score pyrite-output/trajectories/
```

This replays only spectrum scoring into the capturing run's checkpoint stem;
add `--overwrite` to replace existing records. A transport change needs a new
run. Without a capture, use seeded `pyrite checkpoint recompute cxr|brem`.

## Summarize depth and energy profiles

```python
import numpy as np
from pyrite.montecarlo.trajectory_summary import summarize_trajectories
from pyrite.plots.altair.trajectory_summary import summary_charts

tables = summarize_trajectories(
    "capture.h5",
    depth_edges_ang=np.linspace(-1, 40001, 81),
    energy_edges_keV=np.linspace(0, 30, 61),
    max_rows=65536,
)
charts = summary_charts(tables)
```

Choose depth edges that cover the whole capture and include layer interfaces.
The [observable definitions](../physics/beam-transport/transport-outputs.md#saved-capture-observables)
explain what each table counts.

To browse a capture interactively, run `pyrite app viewer launch`, choose
**Saved capture**, enter the artifact path and history count, then select
**Open saved capture**.
