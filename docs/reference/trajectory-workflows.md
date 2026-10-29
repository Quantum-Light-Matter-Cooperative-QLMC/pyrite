# Saved trajectory workflows

Capture, inspect, reduce, replay and export electron histories independently of
spectral checkpoints. The [results guide](../guides/working-with-results.md)
covers source spectra; checkpoints alone cannot reconstruct these histories.
VTK/ParaView exports require no extra PyRITE viewer dependency. Interactive
PyRITE views require `pyrite-mc[apps]`.

## Save electron trajectories

Checkpoints keep spectra, not the electron histories behind them. To keep the complete transport result of a run for later analysis, opt in per run:

```bash
pyrite run trajectory_demo --trajectories
# Faster uncompressed writes, with a larger disk footprint:
pyrite run trajectory_demo --trajectories --trajectory-storage throughput
```

The bundled `trajectory_demo` profile runs one case: 100 fixed histories of 30 keV electrons in a 2 µm HOPG slab, tilted 45° at azimuth 180°, with a round 1 µm FWHM incident beam. Its compact beam spot keeps multiple histories visible at the same scale as their scattering paths. These counts are for visualization, not converged spectra. The profile needs neither `--quick` nor edits to `standard`.

Each case the run transports writes one HDF5 file, `pyrite-output/trajectories/<stem>/<config>-<digest>/E0_<energy>keV.h5`, where `<stem>` is the checkpoint stem and the digest keeps configuration names that sanitize alike apart. The file holds the exact mapping the case's spectrum phase consumed: every per-segment array, grooved runs' vacuum legs, the sampled incident phase space, the exit tallies, and whichever optional midpoint, shell, secondary, radiative, straggling, or diagnostic fields the run produced, with dtype, shape, and row order preserved. It also records the resolved case, the seed, the run identity, the resolved transport settings (core, per-electron cutoffs, `n_hat`), and each field's unit. Field meanings are in [transport outputs](../physics/beam-transport/transport-outputs.md).

A bare `--trajectories` writes under `pyrite-output/trajectories/`; `--trajectories DIR` selects an explicit directory. Put the profile before a bare flag.

Capture never changes a result. The file is written after transport and before the spectrum phase, from the arrays that phase then reads, so spectra and checkpoints are identical to the same seeded run without `--trajectories`. Without the option no trajectory file is written.

Things to know before enabling it:

- **Fixed counts required.** The default `standard` profile uses adaptive counts, which cannot capture trajectories. `trajectory_demo` already sets fixed counts. For another profile, set them with `pyrite numerics set NAME --cxr-trials N --brem-trials N`; reset any explicit precision policy first (see [adaptive electron counts](campaign-configuration.md#adaptive-electron-counts)).
- **Size and throughput.** A file holds every segment of every electron. The default `compact` policy uses LZF-compressed rows: a 30 keV silicon capture measured about 90 bytes per segment, against 208 for the older uncompressed layout. Use `--trajectory-storage throughput` alongside `--trajectories` for uncompressed schema-3 chunks in local or remote captures. Python callers can use `TrajectoryCapture(root=..., storage_policy="throughput")`. This trades disk space for faster writes and affects storage only. The default remains `compact`. See [trajectory artifact storage](../repo-design/storage/trajectory-artifacts.md) for measurements and limitations.
- **Cached cases are not captured.** Only cases this run transports get a file. Cases resumed from a checkpoint or replayed from the shared cache are never re-transported just to write one; the run reports how many and `--recompute` transports them again.
- **Existing files.** Before any transport the run checks the directory. A case that already has a file stops the run unless `--overwrite-trajectories` is given. A cached case whose file records different physics also stops the run: remove the file or pick another directory.
- **Interrupted runs.** Files are written as `<name>.partial` and renamed only once complete, so a crash or budget stop cannot leave a truncated file under the final name. A resumed run discards stale `.partial` files of the cases it transports.
- **Remote captures stay remote.** `pyrite run ... --remote --trajectories` writes on the compute host; transfer selected cases or export selected histories with `pyrite remote trajectories`. See [remote trajectory captures](cluster-execution.md#ssh-remote-helper-and-manual-slurm).

### Find and describe captures

```bash
pyrite trajectories ls
pyrite trajectories info pyrite-output/trajectories/<stem>/<config>-<digest>/E0_30keV.h5
```

`ls` lists every file under the capture root (or each `DIR` given): checkpoint stem, case, E0, schema, status, segment rows, histories with segments, highest secondary generation, and size, then totals per stem. Stray `.partial` files, incomplete artifacts, and HDF5 files that are not trajectory artifacts appear with their status rather than failing the listing. `info` prints one artifact's case, seed, provenance, whether a scene was recorded, history ID range, track count, segments per generation, and every `/transport` field with its role (per-segment, per-electron, nested, scalar), dtype, shape, and units; the per-segment names are the cell arrays an export carries. Both read headers, node metadata, and integer ID columns in bounded blocks only, never segment arrays. `-o json` emits one stable JSON envelope. `export` and `score` likewise skip non-artifact files in a directory with a warning.

Reopen a file from Python:

```python
from pyrite.montecarlo.trajectories import read_trajectory_artifact

artifact = read_trajectory_artifact("pyrite-output/trajectories/<stem>/.../E0_30keV.h5")
segs = artifact.transport  # same keys and arrays as simulate_trajectories
artifact.units["r_mid"]  # "angstrom"
artifact.case["seed"], artifact.settings["transport_core"]
```

The reader restores transported row order, dtypes and aliases from compressed,
electron-grouped storage. It accepts schemas 1–3 and rejects incomplete or
newer unsupported artifacts. Schema 2 adds the typed case and saved spectrum
inputs needed for replay. See [artifact storage](../repo-design/storage/trajectory-artifacts.md)
for the HDF5 layout and compression contract.

### Export segments for visualization

```bash
pyrite trajectories export pyrite-output/trajectories/
```

Every artifact becomes a VTK XML PolyData (`.vtp`) file beside it, readable by ParaView, VisIt, and PyVista/VTK. Each segment is a two-point line cell from `r_mid - L_ang v_hat / 2` to `r_mid + L_ang v_hat / 2` in the slab frame, in angstrom. All per-segment fields ride along as cell data, so thresholding on `electron_id` (or `track_id` with secondaries) isolates one history. Grooved vacuum legs are extra cells with `is_vacuum = 1`, where only `electron_id`, `E_start_keV`, `t_start_ang`, `t0_ang`, `L_ang`, and `v_hat` are defined and other fields are NaN or -1; `--no-vacuum` drops them.

VTK PolyData was chosen because it represents disconnected straight segments with arbitrary per-segment attributes and opens in the common scientific viewers without a plugin. It cannot carry the rest of the artifact, so the export omits the compatibility aliases, per-electron arrays (`initial_*`, `straggle_dE_keV`), tallies, nested metadata (`inelastic`, `radiative`, `secondaries`, `secondary_tracks`, `stopping_tables`, `transport_diagnostics`), the case, settings, provenance, and units. The HDF5 artifact stays the authoritative record.

### Score spectra from saved trajectories

```bash
pyrite trajectories score pyrite-output/trajectories/
```

This replays only the spectrum phase (line, characteristic, and bremsstrahlung spectra) of every captured case and stores each result as an ordinary record of the checkpoint stem and run identity the capturing run recorded, under `--checkpoint-dir` (default `pyrite-output/checkpoints/<stem>`). It never transports. Each new record carries `source_trajectory` with the artifact's path and SHA-256. Records that already exist are kept unless `--overwrite`, and records of cases without an artifact are never touched, so re-scoring a run's own checkpoint after a spectrum-phase change replaces exactly the captured cases. The shared per-case cache is neither read nor written.

Every artifact is checked before anything is written. The command refuses artifacts from before this schema (re-run the case with `--trajectories`), incomplete or foreign files named explicitly (a directory scan skips them with a warning), artifacts written outside a run, two runs' artifacts for one stem, and a target checkpoint written by a different run. Spectrum inputs that the case alone determines are recomputed and must match what the artifact stored.

Segments stream in whole-electron blocks, bounded by `--max-segments`
(default 1,048,576). One whole history may exceed the bound. Scored spectra
agree with live scoring to floating-point summation order. Coherent emission
or temporal profiles require all segments at once and are refused by this
bounded command; the Python `spectrum_from_artifact` path holds the complete
artifact in memory.

Reuse captures only when the change affects spectrum scoring. A transport
change requires new transport. Without a capture, use seeded
`pyrite checkpoint recompute cxr|brem`. See [artifact storage and measured
costs](../repo-design/storage/trajectory-artifacts.md) before retaining a large
campaign.

### Export scene geometry

```bash
pyrite trajectories export pyrite-output/trajectories/ --scene
```

In addition to each original slab-frame `.vtp`, `--scene` writes two independently viewable scenes:

- `<case>.vtm`: lab-frame tracks, a capture-fitted crystal crop, layer boundaries, grooves (up to 200 periods), and an entry-beam axis, in **angstrom**.
- `<case>.instrument.vtm`: the finite target footprint and round beam FWHM outline when their dimensions were recorded, ordered filter volumes, the physical detector face with pixel-grid metadata, and the incident beam axis, in **mm**.

Open these in separate views. Their coordinate scales differ by $10^7$; combining their raw coordinates in one view misrepresents physical distances. The sample entrance origin is the target reference point. Scene tracks and vectors use the recorded sample-to-lab rotation; the original `.vtp` retains its slab frame. Each scene block carries units, frame, material, case digest and transform in VTK FieldData. Grooved vacuum legs retain unknown `track_id`, `parent_id` and `generation` (`-1`); `electron_id` alone cannot recover their ancestry.

New capture-enabled profile runs snapshot downstream geometry in the HDF5 artifact, independently of whether counting acquisition is requested. Old captures remain readable and exportable. When downstream geometry was not captured, the scene omits filters and detector and marks them unavailable; it does not reconstruct them from today's catalog. The beam axis is a reference direction, not a reconstruction of individual incoming particles.

`--out-dir DIR` mirrors the input directory layout for tracks and manifests. `--overwrite` permits replacing their existing outputs. Each scene export owns an immutable `<case>.scene-<id>/` directory; all blocks are completed before publishing a manifest. Keep that directory with its manifests when copying or sharing a scene. Overwrite leaves older sidecar directories intact so previous manifests remain usable; delete an older directory only after verifying no retained manifest references it.

### Inspect trajectories in ParaView

1. **File → Open** the `.vtp` or one `.vtm`, then **Apply**. For a scene, use the block selector or **Extract Block** to select `tracks`. Use a separate render view for the instrument scene.
2. In the colouring menu choose the **cell** array `E_start_keV`; rescale to the data range. Each two-point cell represents a captured segment, so colour is constant on that segment.
3. To isolate transported secondaries, apply **Threshold** to cell `generation` with a lower bound of `1` and an upper bound at the recorded maximum. Primaries have generation `0`; unknown groove-vacuum ancestry is `-1`. A capture without secondary transport has no generation array.
4. To inspect groove-gap flights, threshold cell `is_vacuum` to `1`; threshold to `0` for material segments. The array is present when vacuum diagnostics are included; `--no-vacuum` omits those cells.
5. Threshold `electron_id` to select a complete primary history/shower, or `track_id` to select one electron trajectory. Use **Spreadsheet View** to inspect track, parent, energy and event attributes. Use **Clip** to inspect a depth range without changing the original artifact.

ParaView reads a whole `.vtp` into memory, so export only the histories you need from a large capture:

```bash
pyrite trajectories export trajectories/ --first 20
pyrite trajectories export case.h5 --sample 200 --seed 1 --scene
pyrite trajectories export case.h5 --history 7 --history 12
pyrite trajectories export case.h5 --track 4031
```

Selections take complete histories (a primary `electron_id` with every secondary of its shower) or complete tracks; `--track` intersects a history selection. `--first N` takes the first N histories that have segments, `--sample N` draws N at random (reproducibly from `--seed`, default 0). Only the selected rows are read, so export memory and the output size follow the selection, not the capture. A subset `.vtp` adds the cell array `segment_id`, each cell's row index in the full transported result, and a `selection` FieldData record of the request and the resolved history IDs. Grooved vacuum legs follow selected histories and are left out of track selections, whose ancestry they cannot express. An empty selection exports nothing and reports which history IDs have segments.

The same selection is available from Python:

```python
from pyrite.montecarlo.trajectory_selection import read_selection, select_trajectories

selection = select_trajectories("trajectories/<stem>/.../E0_30keV.h5", sample=50, seed=1)
segs = read_selection(selection, ["r_mid", "E_start_keV", "electron_id"])
segs.transport["segment_id"], segs.units["E_start_keV"]
```

#### ParaView preset

A ParaView script ships with PyRITE. It colours tracks by a cell array (`E_start_keV` by default), adds ready-to-enable Threshold filters for secondaries, material segments and one history, shows scene geometry as faint grey wireframe, and opens the instrument manifest in its own view. The close-up camera fits the visible tracks rather than the full crystal and reference axis. It needs ParaView's Python, not PyRITE's:

<!-- verify: skip (needs ParaView pvbatch, not a pyrite/pyrite-dev command) -->
```bash
SCRIPT=$(pyrite trajectories export --paraview-script)
pvbatch "$SCRIPT" case.vtp --screenshot case.png
pvbatch "$SCRIPT" case.vtm --color generation --threshold electron_id 0 9 --screenshot closeup.png
```

For a first GUI check with the bundled demo, export twenty whole histories:

```bash
pyrite trajectories export trajectories/ --first 20 --scene --out-dir traj-vtk/
```

Open the resulting close-up `.vtm` (without `.instrument` in its name), click **Apply**, select the reader in the Pipeline Browser, then run the imported macro. Other profiles can have millimetre-wide beam spots while each shower travels only micrometres: fitting twenty such histories can still make each shower tiny. Use `--first 1` or `--history ID` for a local shower from those captures, and the instrument view for the full footprint. The close-up crystal is a lateral crop around the selected tracks, retains the recorded thickness, and appears as wireframe so its faces do not hide them. Coordinates and track lengths are never stretched or recentered.

`--color` takes any cell array (`generation`, `layer`, `event_kind`, `t_start_ang`, ...); an unknown name lists the available ones. `--threshold FIELD LOW HIGH` repeats. Headless hosts need `pvbatch --force-offscreen-rendering` or an OSMesa/EGL ParaView build. In the GUI, use **Macros > Import new macro...** with the same file, select an opened export in the Pipeline Browser, and run the macro. For a remote capture, export a subset there and copy the `.vtp`, or connect the ParaView client to a `pvserver` on that host.

## Analyze saved trajectory captures

Schema-2/3 trajectory artifacts can be reduced without running transport:

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

Choose depth edges covering the entire capture, including surface nudges, and
include layer interfaces. Tables contain units in their column names. The
[observable definitions](../physics/beam-transport/transport-outputs.md#saved-capture-observables)
explain the primary-only fates/ranges and all-generation continuous-loss profile.
Histograms retain underflow/overflow counts; charts show finite bins only.
`iter_primary_summaries(path, max_rows=...)` yields bounded per-history tables
when individual values are needed. Consume those tables incrementally to keep
memory bounded. One whole primary shower may exceed `max_rows`; it is never
split. Reduction memory otherwise depends on the block and fixed histogram
sizes, rather than the capture's segment or history count.

For secondary-transport captures, compare the continuous-loss profile sum with
`capture_secondary_energy_balance(path)` from
`pyrite.montecarlo.trajectory_balance`. Its `term, energy_keV` table reuses the
transport energy ledger and includes cutoff, binding, pair and annihilation
terms. Pair metadata is filtered in bounded scans; this can require repeated
passes. `campaign.beam_metrics.capture_beam_metrics(path, beam,
max_particles=65536)` reuses sampled-beam diagnostics over a deterministic prefix
of `initial_*`, including missed entries, returning metrics plus sampled and
available counts. Increase the limit to the available count for full-bunch
metrics; its memory limit is separate from the segment-block limit.

Launch the trace viewer with `pyrite app viewer launch`. Choose **Saved capture**,
enter the artifact path and number of first histories, then **Open saved capture**.
The existing 3D, cross-section and survival views use those stored rows without
transport. Reads are capped at 100,000 material plus vacuum segments; reduce the
history count if the cap is exceeded. Saved-view fractions describe the selected
material-entering histories. The API also accepts `select_trajectories(...)`
whole-history selections directly in `trajectory_chart`,
`penetration_survival_chart`, `trajectory_volume_data` and
`trajectory_volume_figure`. Track-only selections are rejected because they
cannot describe complete primary survival.
