# Working with results

PyRITE stores simulation output as component checkpoints, then lets analysis and export commands consume those checkpoints without rerunning transport.

## Checkpoint layout and identity

Canonical full runs use `checkpoints/<material>/`. Survey runs and modified profiles use identity-qualified directories so incompatible parameter sets do not silently resume into one another. Each dataset records its resolved input payload and hash in component metadata.

Active datasets are directories beneath the effective `checkpoints/` root and are offered by the analysis app's dataset selector. `pyrite checkpoint list` does not list them; it lists labels in the long-term archive shelf:

```bash
uv run pyrite checkpoint list
```

The [sweep-profile guide](sweep-profiles.md) explains how fidelity and profile resolution affect dataset identity. The [checkpoint store design](../repo-design/storage/checkpoint-case-store.md) documents the internal persistence model.

## Analyze or export

Launch the checkpoint-driven analysis app for a material:

```bash
uv run pyrite app analysis launch hopg
```

The sidebar holds the **Material** and **Checkpoint** pickers, the **Emission** choice, and one shared slice: beam energy, polar tilt, azimuth, and crystal thickness. The Checkpoint menu lists every dataset for the material with its face, for example `standard · flat`, `standard · blazed`, or `high_energy (a7b2ce) · flat`. It opens on the standard flat run; when a material has none, it opens on **choose a checkpoint** and loads nothing until you pick one.

The app has three views, and each reads the sidebar slice:

- **Spectra** overlays narrowband and broadband spectra. **Vary** picks the dimension that changes across curves (beam energy, polar tilt, or azimuth); the other slice values stay pinned.
- **Map** draws one azimuth × polar-tilt heatmap at the slice's beam energy and thickness. Pick the plotted **quantity**, click a cell to plot its spectrum, and read the line/coherent-flux trends and the top-20 geometry ranking below it. A checkpoint without a 2-D angular sweep shows 1-D scan plots instead.
- **Detectors** shows the Eagle XO or Timepix3 response at the slice's polar tilt, azimuth, and thickness, with beam energy varying across curves.

A pinned thickness the beam never reached at a low beam energy falls back to that energy's thickest computed slab, and the view says so. The matplotlib efficiency, charge-map, and best-spectra figures are no longer in the app; call them from `pyrite.plots`.

To compare hand-picked cases across datasets (the case basket) or the best lines of every material with a checkpoint, launch the compare app:

```bash
uv run pyrite app compare launch hopg
```

Its Material and Checkpoint pickers select the dataset cases are added from; the basket persists while you switch either. The cross-material view ignores that picker and reads every catalog material's checkpoint.

For non-interactive use, inspect `pyrite app analysis export --help`; `pyrite app pixels export` and `pyrite app compare export` render the other two apps the same way. The generated [CLI reference](../repo-design/cli/cli-reference.md) is authoritative for accepted arguments and output formats.

Current full checkpoints contain three versioned HDF5 components:

- `line.h5` stores PXR/CBS line results without characteristic radiation;
- `brem.h5` stores the bremsstrahlung continuum;
- `characteristic.h5` stores `spec_characteristic` on the fine line grid.

The analysis app merges all available components and shows characteristic radiation by default. Clear **show characteristic radiation** to inspect the smaller PXR/CBS peaks without changing the checkpoint. Omitting `characteristic.h5` is supported: the dataset loads as line-only, which also makes intentional exclusion and transfer straightforward.

In memory, every record keeps `spec`, `spec_coherent`, `spec_characteristic`, and `brem` as separate arrays; none includes another. Code that needs a sum calls `pyrite._spectral_components.line_spectrum` or `incident_spectrum`, and `Result.line_total()` gives the same sum for API results.

Legacy `.pkl` component paths and plain, gzip, and zstd monoliths remain readable and migrate to HDF5 on the next normal save. Artifacts written before the separate-component contract stored `spec`/`spec_coherent` as totals including `spec_characteristic`; they are separated once on load. Use the [result schema](../repo-design/storage/result-schema.md) for independent inspection.

## Pixel-detector observations

A profile detector with an acquisition also stores one counting observation per case in `observations/<stem>/`, the sibling of that detector's `checkpoints/<stem>/`. A profile with several detectors has a separate ID-qualified stem for each. Observations hold factorized per-tile spectra, pixel solid angles, and filter paths rather than a pixel-by-energy cube, so reopening one never reruns transport. The [Python workflow](python-api-workflow.md#persist-reopen-and-rescore-an-observation) covers `ObservationStore` and rescoring; `pyrite.observations.observation_inventory(stem)` lists a stem's stored observations by case without opening their factors.

The pixel app reads one dataset's observations without loading its checkpoint. Pick the material and checkpoint that name the dataset:

```bash
uv run pyrite app pixels launch hopg
```

In the app:

- choose an observation, then an image: registered **total counts**, counts in an **energy window** bounded by configured reporting edges, primary **filter transmission** at a stored continuum-grid energy, or **filter coverage** (plates crossed by each pixel's centre ray);
- **Counts** shows deterministic expectations by default. A Poisson acquisition also offers its seeded realization, labelled as such; neither is ever substituted for the other;
- the image is drawn at full resolution up to 128 by 128 pixels, and larger grids draw summed (counts), averaged (transmission), or maximum (coverage) display cells whose tooltips give their pixel ranges;
- **row** and **column** select one pixel for its geometry (local and lab position, direction, polar/azimuth, solid angle, angular tile, filter paths), its true accepted spectra before response, and its measured reporting-bin histogram with underflow, overflow, and below-cut accounting.

Timepix3 counts are clustered photon events attributed to the incident-ray pixel; raw neighbouring-pixel triggers are not modelled. The first Timepix3 image in a session builds the response matrix and can take tens of seconds. Static HTML export renders the default centre-pixel selection only; interactive pixel selection needs a running app. Observations produced with `pyrite run --remote` stay on the remote host until remote transfer is implemented.

## Save electron trajectories

Checkpoints keep spectra, not the electron histories behind them. To keep the complete transport result of a run for later analysis, opt in per run:

```bash
uv run pyrite run standard -m hopg --trajectories trajectories/
```

Each case the run transports writes one HDF5 file, `trajectories/<stem>/<config>-<digest>/E0_<energy>keV.h5`, where `<stem>` is the checkpoint stem and the digest keeps configuration names that sanitize alike apart. The file holds the exact mapping the case's spectrum phase consumed: every per-segment array, grooved runs' vacuum legs, the sampled incident phase space, the exit tallies, and whichever optional midpoint, shell, secondary, radiative, straggling, or diagnostic fields the run produced, with dtype, shape, and row order preserved. It also records the resolved case, the seed, the run identity, the resolved transport settings (core, per-electron cutoffs, `n_hat`), and each field's unit. Field meanings are in [transport outputs](../physics/beam-transport/transport-outputs.md).

Capture never changes a result. The file is written after transport and before the spectrum phase, from the arrays that phase then reads, so spectra and checkpoints are identical to the same seeded run without `--trajectories`. Without the option no trajectory file is written.

Things to know before enabling it:

- **Size.** A file holds every segment of every electron: roughly 100 bytes per segment for the default frozen fields, more with midpoint or shell fields. That is usually far larger than the spectra; start with `-m` and `--quick` to gauge it.
- **Cached cases are not captured.** Only cases this run transports get a file. Cases resumed from a checkpoint or replayed from the shared cache are never re-transported just to write one; the run reports how many and `--recompute` transports them again.
- **Existing files.** Before any transport the run checks the directory. A case that already has a file stops the run unless `--overwrite-trajectories` is given. A cached case whose file records different physics also stops the run: remove the file or pick another directory.
- **Interrupted runs.** Files are written as `<name>.partial` and renamed only once complete, so a crash or budget stop cannot leave a truncated file under the final name. A resumed run discards stale `.partial` files of the cases it transports.
- **Local runs only.** `--trajectories` is rejected with `-R/--remote`.

Reopen a file from Python:

```python
from pyrite.montecarlo.trajectories import read_trajectory_artifact

artifact = read_trajectory_artifact("trajectories/hopg/.../E0_30keV.h5")
segs = artifact.transport  # same keys and arrays as simulate_trajectories
artifact.units["r_mid"]  # "angstrom"
artifact.case["seed"], artifact.settings["transport_core"]
```

The HDF5 layout is self-describing and readable with any HDF5 tool: `/transport` holds one node per field (aliases such as `E_keV` are hard links to their canonical field), and `/case`, `/settings`, and `/provenance` hold JSON attributes; `/case/payload` also stores the exact typed case. Per-segment fields (marked `pyrite_row`) are stored grouped by electron; when that differs from the transported order, `/transport_order` maps each stored row to its transported index, and PyRITE's reader restores the transported order. Readers reject files whose `complete` flag is unset or whose `schema_version` is newer than they support.

### Export segments for visualization

```bash
uv run pyrite checkpoint export-trajectories trajectories/hopg/
```

Every artifact becomes a VTK XML PolyData (`.vtp`) file beside it, readable by ParaView, VisIt, and PyVista/VTK. Each segment is a two-point line cell from `r_mid - L_ang v_hat / 2` to `r_mid + L_ang v_hat / 2` in the slab frame, in angstrom. All per-segment fields ride along as cell data, so thresholding on `electron_id` (or `track_id` with secondaries) isolates one history. Grooved vacuum legs are extra cells with `is_vacuum = 1`, where only `electron_id`, `E_start_keV`, `t_start_ang`, `t0_ang`, `L_ang`, and `v_hat` are defined and other fields are NaN or -1; `--no-vacuum` drops them.

VTK PolyData was chosen because it represents disconnected straight segments with arbitrary per-segment attributes and opens in the common scientific viewers without a plugin. It cannot carry the rest of the artifact, so the export omits the compatibility aliases, per-electron arrays (`initial_*`, `straggle_dE_keV`), tallies, nested metadata (`inelastic`, `radiative`, `secondaries`, `secondary_tracks`, `stopping_tables`, `transport_diagnostics`), the case, settings, provenance, and units. The HDF5 artifact stays the authoritative record.

### Score spectra from saved trajectories

```bash
uv run pyrite checkpoint score-trajectories trajectories/hopg/
```

This replays only the spectrum phase (line, characteristic, and bremsstrahlung spectra) of every captured case and stores each result as an ordinary record of the checkpoint stem and run identity the capturing run recorded, under `--checkpoint-dir` (default `checkpoints/<stem>`). It never transports. Each new record carries `source_trajectory` with the artifact's path and SHA-256. Records that already exist are kept unless `--overwrite`, and records of cases without an artifact are never touched, so re-scoring a run's own checkpoint after a spectrum-phase change replaces exactly the captured cases. The shared per-case cache is neither read nor written.

Every artifact is checked before anything is written. The command refuses artifacts from before this schema (re-run the case with `--trajectories`), incomplete or foreign files, artifacts written outside a run, two runs' artifacts for one stem, and a target checkpoint written by a different run. Spectrum inputs that the case alone determines are recomputed and must match what the artifact stored.

Segments are read in whole-electron blocks of at most `--max-segments` rows (default 1,048,576), so host memory does not grow with the artifact. On a 7.1-million-segment CPU artifact, the peak above the process baseline was about 1.1 GiB at the default, 0.7 GiB at 262,144 rows, and 0.4 GiB at 65,536 rows. Reading the whole artifact at once peaked at 4.9 GiB. At a fixed block size the peak does not grow with the artifact: 65,536-row blocks peaked at 0.4 GiB for both 2.4 and 7.1 million segments, while whole-artifact reads grew from 1.7 to 4.9 GiB. Smaller blocks were not slower on the CPU. Scored spectra equal the live run's to floating-point summation order (relative 1e-12). Cases with coherent emission or a temporal profile need every segment at once and are refused; score those in Python with `pyrite.montecarlo.runner.spectrum_from_artifact`, which matches the live run bit for bit but holds the whole artifact in memory.

Prefer artifact reuse when you redo only the spectrum phase of the same transported electrons, for example after a change to line, characteristic, or bremsstrahlung scoring. Prefer a seeded re-transport (`pyrite checkpoint recompute line|brem`) when no artifact exists or the change affects transport. Measured on the default 40 keV `standard` sweeps (CPU transport feeding CUDA spectra), transport took 6 to 14 times as long as the spectrum phase per case, and the GPU waited on transport 55 to 89 percent of the time. On the CPU an artifact holds about 216 bytes per segment and reads in about 1.2 microseconds per segment with a warm file cache, against about 9.6 microseconds per segment for lockstep transport. Artifacts cost disk, though: a 30,000-electron, 20 µm hopg case wrote 1.5 GB. CUDA-core transport and cold remote reads are not yet measured.

## Preserve or reduce data

Use checkpoint commands instead of manually editing checkpoint directories:

```bash
uv run pyrite checkpoint --help
uv run pyrite checkpoint archive --help
uv run pyrite checkpoint slim --help
```

`gc` removes cases that no longer match current profile resolution; `rm` targets selected datasets. Both provide previews and confirmation controls. Review their help before destructive use.

Remote runs can be pulled into the same local checkpoint layout. See [Running on a cluster](running-on-a-cluster.md) for submission and transfer.
