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

The existing small live preview remains supported. A standalone PyVista viewer is a candidate for saved showers; a measured supported range and browser interaction costs must accompany its implementation. ParaView remains an interoperable full-run consumer.

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

The probe also has an opt-in interactive mode:

```bash
uv run --with pyvista --with trame --with trame-vtk --with trame-vuetify \
  python checks/trajectory_trame_probe.py tracks.vtp image.png report.json \
  --serve --port 8765
```

Open `http://127.0.0.1:8765` in a browser. The energy slider and secondary switch share the smoke's filtering path; preset buttons reproduce its selections, and the footer reports selected cells. Drag the view to rotate the camera, then use **Save report** to write the current screenshot and JSON report. The report includes connection count, input hash/bytes, versions, initial/final camera, server callback costs and Python/native peak RSS. A client connection alone does not prove image delivery or correct controls: inspect the visible image and counts. Callback costs end at update submission and exclude browser delivery/display; the report does not measure first visible frame, camera latency, browser memory or VRAM. Empty energy selections clear the actor and can be restored. The automatic smoke verifies this recovery.

The raw trial record also includes `trame_continuation`: a fresh no-client smoke verifies the empty-selection recovery and records callback costs without an event-loop delay. Client-visible images and camera changes remain unverified; no connected-browser results are recorded.

**Choice:** implement an optional standalone PyVista/trame saved-capture app in [#319](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/319). The measured export and memory scaling justify that investment and a movie-export investigation. Retain the live Plotly preview while the new app demonstrates its connected-browser workflow. ParaView scene export is usable now. There is no supported production segment ceiling yet: native rendering succeeded up to roughly ten million replicated cells on this machine, while browser delivery and independent large captures still require validation. The follow-up starts with bounded whole-history/track and attribute selection, then measures those missing behaviors before advertising a workload range.

## Saved-viewer implementation

The optional saved-capture application is implemented under `trajectory_viewer/`, with `pyrite app trajectories inspect|launch|export` as its lazy CLI boundary. [The user guide](../../guides/saved-trajectory-viewer.md) owns installation, controls, output and deployment. Ordinary imports/help do not load PyVista/VTK/trame. No marimo or transport changes are required.

Selection scans only bounded HDF5 identity blocks, intersects complete history/track IDs, counts selected segments and estimates admission memory before loading coordinates. It reads only selected fields/row runs and restores original transport order with `/transport_order`. Every rendered segment retains its original `segment_id`, ancestry and available hard-event identifiers. Missing legacy track ancestry remains unknown. The full selection survives energy/generation/time and plane filters; picking reports partial-track status, and parent inspection can inspect a loaded parent hidden by display filters. Geometry-only scene export reuses the shared owner with selected extents; it never exports/loads the full source shower as an intermediate VTP.

The [new raw trial record](../../../checks/saved_trajectory_viewer_results.json) measures the actual viewer including selected attributes and shared scene context. It is not a controlled comparison with the earlier tracks-only probe. Runs used the same Ubuntu/Python host, Mesa llvmpipe software rendering, 1000 × 700 images, three sequential fresh processes per small real capture and 119k/1.01m synthetic selection, and one exploratory 10m trial. OS file caches were not cleared. Array preparation, scene build and first headless PNG are separate; none is a browser first-visible-frame measurement. Peak RSS samples the complete process tree every 20 ms and may miss brief peaks.

| Input | Segments | Trials | Scene build (s) | First headless PNG (s) | Sampled tree RSS (MiB) | Camera render/readback (s) |
|---|---:|---:|---:|---:|---:|---:|
| 20 keV continuous | 1,961 | 3 | 0.498 | 0.523 | 613 | 0.020 |
| 20 keV shell-soft-hard | 2,561 | 3 | 0.495 | 0.528 | 614 | 0.014 |
| 200 keV continuous | 20,559 | 3 | 0.502 | 0.535 | 628 | 0.025 |
| 200 keV shell-soft-hard | 29,756 | 3 | 0.504 | 0.537 | 633 | 0.020 |
| Same 200 keV shower ×4 | 119,024 | 3 | 0.559 | 0.566 | 717 | 0.042 |
| Same shower ×34 | 1,011,704 | 3 | 1.084 | 0.702 | 1,528 | 0.159 |
| Same shower ×336 | 9,998,016 | 1 | 6.220 | 2.361 | 9,235 | 1.334 |

Values are medians; camera medians include three successive renders per process. The 1.01m first PNG range was 0.696–0.706 s, scene build 1.047–1.116 s, and peak RSS 1,495–1,559 MiB. Median array loading was 0.093 s. Minimum-energy/primary-generation selection plus image readback took 0.549 s, time selection 0.432 s, and plane clipping 0.858 s. The 10m exploratory run took 1.040 s to load selected arrays, 5.557 s for energy/generation plus image readback, 4.956 s for time filtering, and 7.865 s for clipping. Ten-million admission estimates were increased after the measured peak exceeded the earlier estimate; neither the updated allowance nor one successful trial guarantees safe loading on another host.

**Measured native/headless range:** small real fixtures and approximately 119k–1.01m repeated whole-shower segments completed three fresh trials here. Approximately 10m completed one exploratory trial at about 9.0 GiB sampled RSS. Replication repeats spatial geometry and overdraw; it establishes neither independent production-shower complexity nor a production ceiling. Browser results are in the next section. Hardware VRAM is unmeasured; these runs used software rendering. Native desktop UI construction is exercised headlessly, while actual desktop interaction remains deployment-specific.

`checks/saved_trajectory_server_probe.py` starts the application's real trame `VtkRemoteView` server without a client. Energy ≥10 keV changes 29,756 cells to 23,504; restricting to primaries gives 22,370; time ≤0 fs gives eight; resetting restores every cell. It calls the server's pick handler directly, then clips and writes a rendered PNG. Because it bypasses the browser, it passed while browser picking was broken: the original view forwarded vtk.js `LeftButtonPress` call data, which contains functions and cannot be serialized, so no pick reached the server. The scale menu was also broken there, because trame reads a literal list attribute as a `(state, default)` pair.

### Connected-browser workflow

`checks/saved_trajectory_browser_probe.py` launches the CLI server and drives it from headless Chrome through Playwright with real mouse and keyboard input. It operates every slider, the clip switch and plane, picking, parent inspection, orbit, reset, both scales, and both downloads. It fails on browser serialization errors, a missing image update, or an unexpected control result. Picking uses the view's `click` picking mode, which sends render-window pixel coordinates. A click that follows a camera drag is ignored. Sliders commit on release. Browser exports render at the browser's current size, are returned to the client by a trame trigger, and keep a server copy and sidecar. `tests/montecarlo/test_saved_viewer.py` covers the same server contract, including parent navigation from a secondary, without a browser.

The additional deterministic `--fixture-secondary` browser regression requires picking a secondary and navigating to its loaded parent. It hides and restores the parent through the energy slider, preserving segment identity and refreshing the partial-track status; camera orbit and scale changes retain selection, and Clear removes it. Export checks decode the PNG and all 30 GIF frames, require rendered content and changing orbit frames, and verify server provenance sidecars. Replacing the export directory with a file then proves that an actual filesystem error produces feedback without an empty browser download. This inspection fixture simulates no transport and establishes no workload limit.

The [raw browser record](../../../checks/saved_trajectory_browser_results.json) has one fresh server and browser session per input, on the host above. It used Chrome 154 headless, co-located over loopback, with Mesa llvmpipe software rendering, a 1600 × 1000 page and a 1200 × 916 view. "Image" is the time from input until a new server image appears in the page. RSS is the server process tree; browser memory and VRAM are not measured.

| Input | Segments | Launch → first image (s) | Server RSS at start (MiB) | Slider/clip image (s) | Scale switch (s) | Reset (s) | GIF download (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20 keV shell-soft-hard | 2,561 | 2.62 | 711 | 0.15–0.77 | 0.54–0.55 | 0.55 | 2.8 |
| 200 keV shell-soft-hard | 29,756 | 2.61 | 731 | 0.15–0.70 | 0.59–0.65 | 0.56 | 3.0 |
| Same shower ×4 (synthetic) | 119,024 | 3.00 | 791 | 0.20–0.80 | 0.72–0.91 | 0.62 | 3.6 |
| Same shower ×34 (synthetic) | 1,011,704 | 5.74 | 1,384 | 0.29–1.18 | 1.77–3.26 | 0.70 | 7.0 |

Each control transferred about 20–240 KiB of image frames and state. At 1.01M the sampled server RSS peaked at 1,780 MiB, and the default admission limit refused that input without `--max-segments 2000000`. Before slider release-commit, each intermediate drag value refiltered the scene, and a 1M drag took 1.7 s to settle. Every click is also a server-rendered camera interaction, so rapid repeated clicking queues renders on large scenes.

These initial single trials establish co-located headless-browser behavior from small real captures up to 1.01M synthetic segments. The follow-up below adds repeated trials, remote image delivery, browser memory and verified hardware rendering; interactive human review and independent large captures remain open.

### Repeated workstation and remote measurements

The [follow-up raw measurements](../../../checks/saved_trajectory_measurement_results.json) retain three sequential fresh-process trials per workload for native/offscreen and connected-browser operation at 119,024 and 1,011,704 segments. The inputs are the same 200 keV silicon shower repeated ×4 and ×34, with distinct history/track identities; hashes match across hosts. No new transport was generated. The current application checkpoint is `f4518c82`; the remote snapshot was `ce2bfb30` before rebase, with viewer/geometry/CLI owners verified identical. All measured energy colors remain linear. The consolidated record includes 23 successful trials and one excluded failed probe. Raw browser/local-native records were rerun into durable paths after local temporary evidence disappeared; the complete remote 10M native report was recovered. Instrumentation lives in `checks/saved_trajectory_browser_worker.py`, `saved_trajectory_browser_probe.py`, `saved_trajectory_remote_worker.py`, and `saved_trajectory_remote_trials.py`.

Each browser trial exercises camera orbit/zoom, every filter and clipping control, picking, parent feedback, scale switching, reset, Clear and rendered PNG/GIF downloads. Local trials also inject a filesystem export failure. Remote trials retrieve the server copies and verify their hashes against actual downloads and their size/scale against provenance. The server worker measures selection scan, selected-array loading and scene/UI construction separately. Filter-call timings include the application filter and its render submission, exclude network/image delivery, and are retained without treating them as client latency.

The workstation uses Mesa llvmpipe software rendering; the remote host is WSL2, Python 3.14.6, 32 CPU threads and 41.1 GiB RAM. The local Chrome client stays on the workstation and connects through an actual localhost-only SSH tunnel. Default WSLg rendering selected `D3D12 (AMD Radeon(TM) Graphics)`. A separate three-trial 1.01M arm explicitly requested NVIDIA and verified `D3D12 (NVIDIA GeForce RTX 5080)`. The device requested by CUDA/SLURM does not by itself select the OpenGL adapter. Mesa's [`MESA_D3D12_DEFAULT_ADAPTER_NAME`](https://docs.mesa3d.org/drivers/d3d12.html) selects the D3D12 adapter by name substring.

Every server and browser starts fresh. OS, driver and shader caches are retained; the native workstation trials use a prepopulated matplotlib cache. The font-cache warmup is excluded and retained in the consolidated record. Earlier overlapping smoke and lost temporary reports do not contribute to the summarized measurement arms. These are deployment measurements, not a controlled GPU speedup comparison. Browser pages are 1600 × 1000 with a 1200 × 916 image view; native images are 1000 × 700.

Values below are median (minimum–maximum). Camera values pool three short orbit gestures per fresh trial; they measure gesture start until a new decoded image is observed after the gesture finishes, including the approximately 0.11 s input gesture. They are not FPS, network RTT, or proof of compositor paint. Control payload counts include all received websocket frame content (state, protocol and images), exclude TCP/SSH framing, and distinguish binary/text rather than assuming every binary frame is an image.

| Browser deployment | Segments | Trials | Start → first image (s) | Orbit image (s) | Energy control image (s) | Instrument scale image (s) | Peak server tree RSS (MiB) | Peak Chrome tree RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Workstation software / loopback | 119,024 | 3 | 5.81 (5.72–6.56) | 0.45 (0.36–0.51) | 0.81 (0.51–0.82) | 1.20 (1.19–1.21) | 982 (981–984) | 1,128 (1,128–1,135) |
| Workstation software / loopback | 1,011,704 | 3 | 11.96 (11.39–13.95) | 1.06 (0.89–1.36) | 1.37 (1.14–1.54) | 5.32 (5.05–5.52) | 1,723 (1,691–1,737) | 1,121 (1,104–1,122) |
| Remote WSLg AMD / SSH | 119,024 | 3 | 7.84 (7.83–10.88) | 0.14 (0.12–0.44) | 0.81 (0.27–0.83) | 0.57 (0.56–0.77) | 854 (854–856) | 1,197 (1,196–1,211) |
| Remote WSLg AMD / SSH | 1,011,704 | 3 | 5.86 (5.62–6.22) | 0.13 (0.11–0.19) | 0.85 (0.45–1.54) | 1.40 (1.34–1.41) | 1,436 (1,406–1,438) | 1,180 (1,173–1,182) |
| Remote WSLg RTX 5080 / SSH | 1,011,704 | 3 | 10.62 (7.13–11.76) | 0.19 (0.13–0.35) | 0.93 (0.54–1.31) | 1.38 (1.24–1.38) | 1,450 (1,440–1,458) | 1,184 (1,179–1,195) |
| Remote WSLg RTX 5080 / SSH (exploratory) | 9,998,016 | 1 | 16.30 (16.30–16.30) | 0.28 (0.17–0.33) | 2.45 (2.45–2.45) | 9.91 (9.91–9.91) | 8,092 (8,092–8,092) | 1,169 (1,169–1,169) |

Local startup runs from child-server launch; remote startup includes starting its SSH command, dependency check and server process. Local RSS is sampled every 50 ms; remote RSS targets 50 ms but GPU queries introduce measured gaps retained in the report. Summed RSS includes shared mappings repeatedly and may miss brief peaks. Chrome trees include browser, renderers, GPU and utility processes, rather than just JavaScript heap. NVIDIA device-wide used VRAM baselines were 1069–1069 MiB and sampled peaks were 1263–1263 MiB in the three 1.01M trials. This is an observed device-wide change, not an attributed per-process allocation; the NVIDIA readings do not measure the AMD adapter's memory.

**Measured browser range:** both synthetic selections completed repeated real-browser workflows locally and through this SSH deployment. At 1.01M, instrument scale switching had medians from 1.4 to 5.3 s across the measured deployments; image streaming still leaves roughly 1.4–1.7 GiB sampled server RSS and about 1.1–1.2 GiB summed Chrome RSS. Independent large captures, other network conditions, per-process VRAM and interactive human review remain open. These results do not establish a production ceiling or justify replacing the existing live preview.

Fresh native software trials give scene-build medians of 1.133 (1.067–1.190) s at 119k and 1.868 (1.859–2.753) s at 1.01M; first-PNG medians are 1.433 (1.422–2.884) s and 1.731 (1.684–1.779) s. Sampled native process-tree RSS is 709 (708–710) MiB and 1,545 (1,537–1,547) MiB. These native images and timings use a different view size from the browser arm and are not browser latency evidence. The native batch was interrupted between trials; the full completed reports and their spread remain in the raw record.

A bounded exploratory native trial on the NVIDIA lab deployment loaded 9,998,016 replicated segments with approximately 40 GiB available before admission. Estimated peak memory was 9.5 GiB, and sampled process-tree RSS reached 7,671 MiB. Selected-array loading took 1.358 s, scene construction 6.351 s, first PNG 0.869 s, energy/generation filtering plus readback 3.591 s, time filtering plus readback 1.941 s, and clipping plus readback 4.140 s. This is one synthetic trial, not a production ceiling. The capture repeats the 119k input ×84 (the original shower ×336); it generates no new histories through transport.

One exploratory 9,998,016-segment connected-browser trial also passed all controls, picking, PNG/GIF decoding and remote download/provenance checks. It used `--max-segments 10000000 --memory-mib 16384` and the verified NVIDIA renderer. SSH launch to first decoded image was 16.303 s; median short-orbit image observation was 0.281 s; instrument scale switching took 9.905 s. Sampled server RSS reached 8,092 MiB and Chrome RSS 1,169 MiB. Device-wide NVIDIA VRAM changed from a 1069 MiB baseline to a 1666 MiB sampled peak. One synthetic trial establishes an exploratory success only.

Reproduction:

```bash
LIBGL_ALWAYS_SOFTWARE=1 uv run --extra trajectory-viewer python checks/saved_trajectory_viewer_probe.py \
  capture.h5 --output trial.json --image frame.png
uv run --extra trajectory-viewer python checks/saved_trajectory_viewer_probe.py \
  capture.h5 --replicate 34 --scaled-output synthetic-million.h5
LIBGL_ALWAYS_SOFTWARE=1 uv run --extra trajectory-viewer python checks/saved_trajectory_server_probe.py \
  capture.h5 server.png server.json
LIBGL_ALWAYS_SOFTWARE=1 uv run --extra trajectory-viewer --with playwright \
  python checks/saved_trajectory_browser_probe.py --fixture-secondary --output browser-fixture.json
LIBGL_ALWAYS_SOFTWARE=1 uv run --extra trajectory-viewer --with playwright \
  python checks/saved_trajectory_browser_probe.py capture.h5 --output browser-trial.json
# Stage the same checkout/captures and install the optional extra in an isolated
# remote directory first; this command owns only the servers/tunnel it creates.
uv run --extra trajectory-viewer --with playwright python checks/saved_trajectory_remote_trials.py \
  --host HOST --root /tmp/REMOTE_RUNTIME --captures /tmp/LOCAL_CAPTURES --output /tmp/REMOTE_TRIALS
# Explicit WSLg NVIDIA arm; the recorded renderer must confirm the adapter.
uv run --extra trajectory-viewer --with playwright python checks/saved_trajectory_remote_trials.py \
  --host HOST --root /tmp/REMOTE_RUNTIME --captures /tmp/LOCAL_CAPTURES --output /tmp/NVIDIA_TRIALS \
  --scales x34 --adapter NVIDIA
```

The scaling command writes complete repeated histories with distinct track/parent identities and explicit synthetic provenance. It performs no Monte Carlo. Generate/reuse the four small source captures with the earlier probe; heavy new independent captures remain remote-only. Final connected-client measurements should retain the same distinctions between preparation, rendering, network delivery and visible UI response before revisiting live-preview replacement.
