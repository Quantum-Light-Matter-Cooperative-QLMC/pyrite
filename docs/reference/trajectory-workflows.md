# Trajectory capture reference

Saved electron-history artifacts, their commands, and the contracts for
inspection, export, replay and reduction. Captures are independent of spectral
checkpoints, which cannot reconstruct these histories. For step-by-step tasks,
see [Save and inspect electron trajectories](../guides/saved-trajectories.md).
VTK/ParaView exports require no extra PyRITE viewer dependency. Interactive
PyRITE views require `pyrite-mc[apps]`.

## Save electron trajectories

Capture is opt-in per run with `pyrite run PROFILE --trajectories [DIR]`;
`--trajectory-storage compact|throughput` selects the storage policy. The
bundled `trajectory_demo` profile captures 100 fixed 30 keV histories in a 2 µm
HOPG slab and needs neither `--quick` nor edits to `standard`.

Each case the run transports writes one HDF5 file, `pyrite-output/trajectories/<stem>/<config>-<digest>/E0_<energy>keV.h5`, where `<stem>` is the checkpoint stem and the digest keeps configuration names that sanitize alike apart. The file holds the exact mapping the case's spectrum phase consumed: every per-segment array, grooved runs' vacuum legs, the sampled incident phase space, the exit tallies, and whichever optional midpoint, shell, secondary, radiative, straggling, or diagnostic fields the run produced, with dtype, shape, and row order preserved. It also records the resolved case, the seed, the run identity, the resolved transport settings (core, per-electron cutoffs, `n_hat`), and each field's unit. Field meanings are in [transport outputs](../physics/beam-transport/transport-outputs.md).

A bare `--trajectories` writes under `pyrite-output/trajectories/`; `--trajectories DIR` selects an explicit directory. Put the profile before a bare flag.

Capture never changes a result. The file is written after transport and before the spectrum phase, from the arrays that phase then reads, so spectra and checkpoints are identical to the same seeded run without `--trajectories`. Without the option no trajectory file is written.

Things to know before enabling it:

- **Fixed counts required.** The default `standard` profile uses adaptive counts, which cannot capture trajectories. `trajectory_demo` already sets fixed counts. For another profile, set them with `pyrite numerics set NAME --cxr-histories N --brem-histories N`; reset any explicit adaptive-sampling policy first (see [adaptive electron counts](campaign-configuration.md#adaptive-electron-counts)).
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

`pyrite.montecarlo.trajectories.read_trajectory_artifact(path)` returns the
same `transport` keys and arrays as `simulate_trajectories`, plus `units`,
`case` and `settings` mappings. The reader restores transported row order, dtypes and aliases from compressed,
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

### Select histories and tracks

Each two-point cell is one captured segment, so cell attributes are constant
along it. Primaries have `generation` `0`; unknown groove-vacuum ancestry is
`-1`, and a capture without secondary transport has no generation array.
`is_vacuum` is present when vacuum diagnostics are included. `electron_id`
identifies a complete primary history/shower and `track_id` one electron
trajectory. ParaView reads a whole `.vtp` into memory; export selections bound
what is read and written:

```bash
pyrite trajectories export trajectories/ --first 20
pyrite trajectories export case.h5 --sample 200 --seed 1 --scene
pyrite trajectories export case.h5 --history 7 --history 12
pyrite trajectories export case.h5 --track 4031
```

Selections take complete histories (a primary `electron_id` with every secondary of its shower) or complete tracks; `--track` intersects a history selection. `--first N` takes the first N histories that have segments, `--sample N` draws N at random (reproducibly from `--seed`, default 0). Only the selected rows are read, so export memory and the output size follow the selection, not the capture. A subset `.vtp` adds the cell array `segment_id`, each cell's row index in the full transported result, and a `selection` FieldData record of the request and the resolved history IDs. Grooved vacuum legs follow selected histories and are left out of track selections, whose ancestry they cannot express. An empty selection exports nothing and reports which history IDs have segments.

The same selection is available from Python as
`pyrite.montecarlo.trajectory_selection.select_trajectories(path, histories=..., tracks=..., first=..., sample=..., seed=0)`;
`read_selection(selection, fields)` reads only the named fields of the selected
rows, in transported order, with `segment_id` giving each row's transported index.

#### ParaView preset

A ParaView script ships with PyRITE. It colours tracks by a cell array (`E_start_keV` by default), adds ready-to-enable Threshold filters for secondaries, material segments and one history, shows scene geometry as faint grey wireframe, and opens the instrument manifest in its own view. The close-up camera fits the visible tracks rather than the full crystal and reference axis. It needs ParaView's Python, not PyRITE's:

<!-- verify: skip (needs ParaView pvbatch, not a pyrite/pyrite-dev command) -->
```bash
SCRIPT=$(pyrite trajectories export --paraview-script)
pvbatch "$SCRIPT" case.vtp --screenshot case.png
pvbatch "$SCRIPT" case.vtm --color generation --threshold electron_id 0 9 --screenshot closeup.png
```

The close-up crystal is a lateral crop around the selected tracks, retains the recorded thickness, and appears as wireframe so its faces do not hide them. Coordinates and track lengths are never stretched or recentered; with millimetre-wide beam spots, each micrometre-scale shower can appear tiny.

`--color` takes any cell array (`generation`, `layer`, `event_kind`, `t_start_ang`, ...); an unknown name lists the available ones. `--threshold FIELD LOW HIGH` repeats. Headless hosts need `pvbatch --force-offscreen-rendering` or an OSMesa/EGL ParaView build. In the GUI, use **Macros > Import new macro...** with the same file, select an opened export in the Pipeline Browser, and run the macro. For a remote capture, export a subset there and copy the `.vtp`, or connect the ParaView client to a `pvserver` on that host.

## Simulate once, score many times

`pyrite.simulate` transports electrons and scores their spectra in one call.
The two-step workflow splits those phases so that one fixed-seed transport can
be scored under several detectors, observation directions, line grids or
quadratures with no transport noise between them:

<!-- verify: skip (illustrative; see tests/campaign/test_score_spectra.py) -->
```python
import pyrite
from pyrite.detectors import Detector
from pyrite.transport import simulate_trajectories

trajectories = simulate_trajectories(beam, target, physics=physics, numerics=numerics)
front = pyrite.score_spectra(trajectories, Detector(observation_angle_deg=90.0))
side = pyrite.score_spectra(trajectories, Detector(observation_angle_deg=60.0))
```

{py:func}`pyrite.transport.simulate_trajectories` returns a
{py:class}`pyrite.transport.Trajectories`: the segment arrays
(`trajectories.segments`, angstrom and keV as in the
[transport outputs](../physics/beam-transport/transport-outputs.md)), the
resolved beam, target, capture detector, `Physics`, `Numerics` and case, and a
segment fingerprint. {py:func}`pyrite.score_spectra` runs only the production
spectrum phase on it (lines, characteristic, bremsstrahlung, transition
radiation and the automatic line grid) and returns the same
{py:class}`pyrite.Result` as `pyrite.simulate` for the same components: source
photon densities per incident electron per eV per sr, or observation averages
and spatial data for a planar detector. On one backend the two are bit for bit
equal. `score_spectra` never transports.

**What may change between scores.** Any detector, observation, acquisition and
filters, and the spectrum-phase settings of `physics=`/`numerics=`: line grids
and their policy, quadrature, reflections, emission mode, temporal output,
line roulette, spectral chunks and CXR launch controls. A detector-free
transport (`detector=None`) must be given a detector to score.

**What is refused.** Anything that would have changed the transport, checked by
comparing the lowered scoring case with the transported one field by field:
target, beam, transport physics or numerics, seed, history population, table
identity, transport kernel tuning, and explicit `E_cut_keV`/`max_steps`. A
grooved target's transport geometry depends on the observation angle, so its
detector angle is fixed; coherent and temporal scoring need the capture
detector's placement. Every refusal raises
{py:class}`pyrite.transport.ScoringRefusedError` (a `ValueError`) before any
spectrum work; its `field` names the input.

**Transport defaults.** Since #1162 `simulate_trajectories` uses the production
per-history cutoffs (5 keV for line histories, 1 keV for continuum histories,
the lower for shared ones) and the retrying step budget, exactly as
`pyrite.simulate`. Earlier releases stopped every primary at 5 keV. Pass
`E_cut_keV`/`max_steps` for a single explicit cutoff or limit; such
trajectories are for transport analysis and cannot be scored.

**Saving.** {py:func}`pyrite.storage.save_trajectories` writes the same HDF5
artifact as `pyrite run --trajectories`, plus the segment fingerprint.
{py:func}`pyrite.storage.load_trajectories` reopens either kind. Pass the
components the transport was made from to score any detector again; they must
lower to the stored case exactly, otherwise the first differing field is named:

<!-- verify: skip (illustrative) -->
```python
from pyrite.storage import load_trajectories, save_trajectories

path = save_trajectories(trajectories, "hopg.h5")
again = load_trajectories(path, beam, target, physics=physics, numerics=numerics)
recorded = load_trajectories("pyrite-output/trajectories/.../E0_30keV.h5")
pyrite.score_spectra(recorded)  # the recorded case and detector only
```

Without components, which is how run captures load, the result replays only
its recorded case and detector, with the inputs the run stored; this is the
computation `pyrite trajectories score` performs. That command scores into run
checkpoints and therefore accepts run captures only. `stream=True` leaves the
segments on disk and scores the recorded case in bounded-memory electron
blocks; streamed spectra equal in-memory ones to floating-point summation order
(rtol 1e-12 in tests), and coherent or temporal cases are refused. The
fingerprint of an in-memory load is verified before scoring.

**When to prefer two steps.** Transport dominates a case: #186 measured CPU
transport at 6–14× the CUDA spectrum phase for 40 keV hopg, hBN and MoS₂, and
an artifact read costs about 1.2 µs per segment against about 9.6 µs per
segment of CPU transport. Scoring a kept transport is therefore much cheaper
than transporting again, and comparisons between scores carry no transport
noise. Keep one-shot `pyrite.simulate` for a single score, adaptive sampling,
channeling, quantum entry states, streamed scoring and photon aperture
tallies, which `simulate_trajectories` does not admit. In memory, a
`Trajectories` holds every segment; save it and load with `stream=True` when
it does not fit.

## Analyze saved trajectory captures

Schema-2/3 trajectory artifacts can be reduced without running transport by
`pyrite.montecarlo.trajectory_summary.summarize_trajectories(path,
depth_edges_ang=..., energy_edges_keV=..., max_rows=...)`;
`pyrite.plots.altair.trajectory_summary.summary_charts` plots its tables.
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

The trace viewer (`pyrite app viewer launch`, **Saved capture**) opens the
first N histories of an artifact. The existing 3D, cross-section and survival views use those stored rows without
transport. Reads are capped at 100,000 material plus vacuum segments; reduce the
history count if the cap is exceeded. Saved-view fractions describe the selected
material-entering histories. The API also accepts `select_trajectories(...)`
whole-history selections directly in `trajectory_chart`,
`penetration_survival_chart`, `trajectory_volume_data` and
`trajectory_volume_figure`. Track-only selections are rejected because they
cannot describe complete primary survival.
