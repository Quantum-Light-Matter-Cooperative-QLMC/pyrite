# Trajectory scenes and viewer evaluation

Trajectory HDF5 is the authoritative transport snapshot. `checkpoint export-trajectories` preserves its original slab-frame, angstrom `.vtp` default. The additive `--scene` option exports two independently viewable `.vtm` files, with private sidecars:

| Manifest | Coordinates | Contents |
|---|---|---|
| `<case>.vtm` | lab axes, angstrom | captured tracks, crystal crop, layer boundaries, bounded groove relief, incident reference axis |
| `<case>.instrument.vtm` | lab axes, mm | recorded target footprint, supported beam outline, ordered filters, detector face/pixel metadata, incident reference axis |

The user workflow belongs in [Working with results](../../guides/working-with-results.md#export-scene-geometry).

## Geometry and capture ownership

`pyrite._scene_geometry` contains numeric box triangulation, groove ribbons, beam footprint projection and sample-to-lab rotations. Plotly retains only trace styling; the VTK exporter consumes the same numeric geometry. No plotting backend imports another backend. Groove relief delegates to the existing transport geometry; this work adds no transport model.

`instrument.scene.scene_payload` snapshots validated physical detector/filter poses and numeric corners into JSON-safe values. Scene coordinates are relative to the target reference point, identified explicitly as the sample entrance origin. The per-case rotation maps sample axes to lab axes. `sample_origin_lab_mm` records the translation (zero for current profile runs); a slab-centre shift is never inferred.

`TrajectoryCapture.scene` is optional. Schema-v1 HDF5 readers accept the additive `/scene` JSON group, and old files without it remain readable. The scene digest is separate from the transport case digest. Capture resume checks a requested scene against the stored scene digest, avoiding reuse of stale geometry. CLI profile capture includes physical geometry even without an acquisition. Library sweeps with an observation supply its scene when the capture did not already specify one.

The exporter streams tracks and fitted x/y extents in bounded row blocks; the VTK reader may still load the full dataset. Close-up crystal geometry is a crop fitted to captured endpoints, not a reconstruction of an unspecified lateral size. Instrument target footprints require both recorded dimensions. The FWHM contour currently supports a round analytic spot; non-round, Twiss and GPT footprints are omitted and marked unavailable instead of drawing an invented circle. The beam axis is a reference, not individual particle trajectories. Grooves exceeding 200 periods are omitted with metadata. Vacuum diagnostics preserve unknown ancestry as `-1`.

Every block has VTK FieldData for units, frame, scale group, material, case digest and sample transform. Track geometry and vector attributes rotate together; the original slab-frame `.vtp` is unchanged. Downstream geometry comes from the stored scene, never the current catalog. Missing geometry remains unavailable.

## Publication and compatibility

Before writing, the CLI checks collisions across every selected track and manifest destination, including collisions between different artifacts. Existing destinations require `--overwrite`. Output directories retain their existing mirroring behavior; default stdout remains one track-summary line per artifact. Scene mode additionally prints the two manifest paths.

Scene exports write all sidecars into a unique immutable directory, stage both manifests, and replace manifests individually. An interruption before publication removes the new directory. An interruption after one manifest publishes retains its complete sidecars. Older directories remain intact on overwrite, preserving previously copied manifests; cleanup is deliberate rather than implicit. Copy manifests together with referenced directories.

## Backend decision

The categorical rejection of PyVista based on marimo integration is withdrawn. A saved-capture viewer may be a separate application. PyVista supports native/Qt plotting and trame integration; trame server rendering retains geometry server-side and sends images, while client rendering transfers geometry to the browser. A live trame app can keep Python filters reactive. These choices have different deployment and resource costs. Sources: [PyVista Qt](https://docs.pyvista.org/api/plotting/qt_plotting.html), [PyVista trame](https://docs.pyvista.org/user-guide/jupyter/trame.html), [trame rendering](https://kitware.github.io/trame/guide/tutorial/vtk.html).

The existing small live preview remains supported. This backend evaluation preceded the [ParaView decision](#decision-paraview-is-the-saved-trajectory-viewer) below; the standalone PyVista prototype has been retired.

The two-point VTP layout uses 72 bytes per segment for coordinates, connectivity and offsets before attributes: about 720 MB for ten million segments. Readers, selected attributes, filtered copies and rendering buffers add memory. Neither a new backend nor image streaming provides out-of-core loading automatically. A standalone viewer should select whole histories/tracks and requested attributes before rendering; segment order and parent links must survive selection.

## Reproducible cost probe

`checks/trajectory_viewer_benchmark.py` compares tracks-only Plotly construction/image export with PyVista native rendering/filtering, using the same HDF5 arrays and generation styling. It reports preparation, construction, first render, repeated camera-image renders, filtering, PNG encoding/payload, versions and process peak RSS. Rendering is 800 × 600. Run each arm/trial in a fresh process and repeat; camera samples within one process are reported separately from first render.

```bash
# Four small CPU captures: eight primaries, seed 11, Si, cutoff 2 keV.
# Requires installed SBETHE tables and pinned shell configuration.
uv run python checks/trajectory_viewer_benchmark.py --generate-small /tmp/viewer-captures

uv run --with pyvista --with kaleido python checks/trajectory_viewer_benchmark.py \
  /tmp/viewer-captures/200-shell-soft-hard.h5 --backend plotly --output /tmp/plotly.json
uv run --with pyvista --with kaleido python checks/trajectory_viewer_benchmark.py \
  /tmp/viewer-captures/200-shell-soft-hard.h5 --backend pyvista --output /tmp/pyvista.json
```

Plotly image export uses Kaleido/Chrome, including its browser lifecycle per export. It does **not** measure camera interaction in an already-running browser. PyVista retains its render window between samples. Comparing these timings supports an animation/export investigation, not an interactive-frame-rate claim. Peak RSS covers the Python process and native libraries, not Chrome subprocesses. GPU memory, client latency, picking and trame network delivery require separate measurements.

`--replicate K` repeats entire showers with distinct history/track/parent identities. This exposes resource scaling but does not represent a newly transported larger or higher-energy sample. Heavy new transport generation remains remote-only. Small captures exercise the real path; their support does not establish production-scale reliability.

## Recorded results and choice

The [raw trial record](../../../checks/trajectory_viewer_results.json) retains input hashes, resolved cases, versions and individual measurements. Three sequential fresh processes per input/backend ran on Ubuntu 26.04, Python 3.14.4, 16 CPU threads and 30.5 GiB RAM, with Mesa 26.0.8 llvmpipe software rendering. Optional overlays used NumPy 2.5.3, Plotly 7.1.0, Kaleido 1.4.0, PyVista 0.49.0 and VTK 9.7.1. The four real captures used eight primaries each: 20 keV / 2 µm and 200 keV / 200 µm silicon, with continuous or shell-soft-hard transport. They are small correctness and cost probes, not representative production statistics.

Median measurements below report first PNG/screenshot time **after** preparation and construction, and process peak RSS. The second pair of columns includes native VTK initialization costs in its separate construction measurement; Chrome lifecycle remains in each Plotly image export.

| Capture | Segments / tracks | Plotly first PNG (s) | PyVista first screenshot (s) | Plotly RSS (MiB) | PyVista RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| 20 keV continuous | 1,961 / 8 | 3.269 | 0.539 | 375 | 648 |
| 20 keV shell-soft-hard | 2,561 / 17 | 3.258 | 0.749 | 378 | 663 |
| 200 keV continuous | 20,559 / 8 | 3.374 | 0.517 | 509 | 658 |
| 200 keV shell-soft-hard | 29,756 / 113 | 3.433 | 0.532 | 540 | 673 |
| Same shower ×4 | 119,024 / 452 | 5.590 | 0.542 | 622 | 716 |
| Same shower ×34 | 1,011,704 / 3,842 | 28.268 | 0.745 | 2,113 | 1,191 |
| Same shower ×336 | 9,998,016 / 37,968 | not attempted | 3.731 | not measured | 5,960 |

At 1,011,704 cells, median preparation was 0.130/0.127 s and construction 0.431/0.693 s for Plotly/PyVista. Plotly's JSON payload was 330 MB. Energy plus primary-generation selection took 0.014/0.055 s; these are selection costs, not time until a browser displays the filtered scene. PyVista's ten-million construction took 3.635 s and selection 0.685 s. Plotly's ten-million arm was not attempted given the observed growth; it has no comparative result at that size. All scaling inputs repeat identical spatial showers, so overdraw and geometry complexity differ from a real large capture.

ParaView 6.0.1 opened the same 29,756-cell VTP in three offscreen trials: median load 0.116 s, Show/reset setup 2.147 s, subsequent explicit Render 0.010 s, screenshot 0.043 s, generation threshold 0.014 s and process RSS 733 MiB. Show/reset may initialize rendering; the subsequent Render cannot be compared to another backend's first frame. Reproduce with:

```bash
pvpython --force-offscreen-rendering checks/trajectory_paraview_probe.py \
  tracks.vtp image.png report.json
```

The optional `checks/trajectory_trame_probe.py` smoke starts a live `VtkRemoteView` server, changes energy and secondary controls through state callbacks, renders and exports an image. On the same capture, energy ≥10 keV reduced cells from 29,756 to 23,504, disabling secondaries reduced them to 22,370, and restoring controls recovered all cells. No browser connected. Its event-loop-yield timings are deliberately **not** browser latency evidence. Time filtering, picking, clipping, VRAM and complete process-tree memory remain unmeasured.

**Initial choice (superseded):** an optional standalone PyVista/trame saved-capture app was prototyped in [#319](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/319).

## Decision: ParaView is the saved-trajectory viewer

The #319 prototype (branch commit `3bb81704`, raw records removed in the follow-up; recover them from that commit) selected whole histories/tracks from HDF5 before rendering and served native and trame image-streaming views. Its measurements: at 1.01M synthetic segments, native first headless PNG 0.70 s at about 1.5 GiB sampled process-tree RSS; connected-browser energy-control images 0.8–1.4 s and instrument scale switches 1.4–5.3 s across local and SSH deployments, with 1.4–1.7 GiB server and 1.1–1.2 GiB Chrome RSS; one exploratory 10M-segment trial at 8–9 GiB server RSS. User review of the browser application found most controls broken.

ParaView already provides colouring by any cell array, Threshold/Clip/Spreadsheet analysis, client–server remote rendering, `pvpython`/`pvbatch` scripting and routine 10^7-cell scenes. The prototype's only unique capability was selecting complete tracks before loading. That now lives in export: `montecarlo.trajectory_selection` resolves whole histories (bisection over electron-major rows) or tracks (bounded block scan) to stored-row runs, and `export-trajectories --history/--track/--first/--sample` streams only those runs, adding a `segment_id` cell array and a `selection` FieldData record. A bundled preset (`src/pyrite/data/paraview/pyrite_trajectories.py`, path printed by `--paraview-script`) sets up colouring, thresholds, scene context and headless PNG output; it was exercised with ParaView 6.2.0 `pvbatch`.

The trame server, native desktop mode, PyVista PNG/GIF export and their `trajectory-viewer` extra were removed. A custom viewer is justified only for physics-aware interaction, such as linking an electron to the emission it produced; that belongs in the marimo trace app, not a third rendering stack. The live Plotly preview is unchanged.

The preset frames the visible track bounds after thresholds, independently of crystal thickness and the beam reference axis. Scene context uses grey wireframe with inherited per-block colour/opacity overrides cleared; the close-up lateral crop follows selected endpoints without including the beam-axis origin. The bundled `trajectory_demo` profile uses one 30 keV / 2 µm HOPG case, 100 fixed histories and a 1 µm FWHM named beam to keep a multiple-history preview compact. A broad spot still spreads histories across their recorded landing positions, so use one-history exports for close-ups from those captures. Neither export nor the preset rescales tracks to fill the beam footprint.
