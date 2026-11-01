# Repository map: CLI & packaging

Detail page of the [repository map](../repo_map.md). Start from the front page; module paths are relative to `src/pyrite/`.

## `cli/`
`pyrite` console-script dispatcher and CLI-specific helpers. Every subcommand's Click wiring lives in `cli/commands/`; pure domain logic stays in its domain module.
- Public: `main`.
- Deps: lazy per-command imports from `cli.commands.*` (see `_COMMANDS`), plus lazy checkpoint, app-validation, and remote owners reached through their canonical packages or retained root compatibility facades. Eager lightweight deps: `console.output`, `cli._groups`, `__version__`.

## `cli/_deprecations.py`
RFC D7 deprecation harness: frozen `Deprecation` registry keyed by command path, `DeprecatingGroup(click.Group)` that warns once per invocation in `resolve_command` (before callbacks and `--help`), and `scripts/generate_cli_deprecations.py` regenerates [CLI deprecations](../repo-design/cli/cli-deprecations.md) from the registry.
- Public: `DEPRECATIONS`, `Deprecation`, `DeprecatingGroup`, `message`, `warn`, `invocation_path`, `SUPPORT_WINDOW_MINORS`.
- Wired into `cli/_groups.py` (`LazyGroup(DeprecatingGroup)`), `cli/commands/profile.py`, `cli/commands/sweep.py`, `cli/commands/energy_grid.py`, `cli/commands/remote.py`.

## `cli/commands/`
One module per `pyrite` subcommand group, holding only the Click layer. `scan`/`blaze` command wiring split out of the fused `scan.py`/`blaze.py` drivers; `energy_grid` owns the whole `pyrite energy-grid` group; `profile` owns named campaign defaults and membership; `material` owns effective-range inspection and per-profile overrides; `beam` owns named `[beams.*]` objects; `sweep` is hidden compatibility aliases only. `job` owns the canonical remote asynchronous-job lifecycle and delegates transport/state operations to `remote/`. Shared validated atomic TOML helpers stay in `cli/_catalog_io.py`.

## `console/`
Terminal presentation primitives shared by the CLI and the domain packages, and the layer everything else may depend on without depending on `cli/`. It imports `paths`, `_env` and `_catalog_keys` and nothing else first-party.
- `output.py` — colour, Click parameter types, destructive-action confirmation, result/diagnostic emission, JSON envelopes, and `run()`, which preserves the pyrite exit contract around Click.
- `json.py` — stable JSON payload adapters for automation.
- `config.py` — the persistent context store (`profile.current`, `remote.*` target and SLURM profile, `workspace.root`, `catalog.path`, `xsgen.*`), its per-call > `PYRITE_*` > store > built-in precedence resolver, and workspace/catalog path helpers.
- `outputs.py` — lazy workspace output paths, ancestor discovery, CLI-only legacy migration (rename or verified cross-filesystem copy), and shared output-kind defaults; `config.py` exposes resolver wrappers.
- `dashboard/` — below.

`checkpoints`, `runs` and `remote` all emit results and render progress while they work. Sourcing that from `cli/` is what put `cli` in a bidirectional pair with each of them; the `domain-packages-stay-below-cli` and `console-stays-below-every-driver` contracts hold the split.

## `cli/_groups.py`
`LazyGroup(DeprecatingGroup)`: the Click group that resolves command modules on demand. Stays in `cli/` with the command tree and the deprecation registry it extends, while the primitives it uses moved to `console/output.py`.

## `cli/_options.py`
`-R/--remote`: the one shared option whose callback validates through `remote.config`, which `console/` sits below and cannot import.

## `_catalog_keys.py`
Offline `[materials|profiles|beams|detectors].*` key reads by listing the selected catalog directory's object files, without importing the scientific material modules. Shell completion, `runs.scan`'s material validation and checkpoint identity resolution all need catalog keys where a `materials` import would be far too expensive.

## `_catalog_layout.py`
On-disk catalog layout: a single TOML file or a one-object-per-file catalog directory (`catalog.toml` plus `<table>/<name>.toml`, optional `cifs/` and energy-grid artifacts). Owns selected-path lookup, layout validation, the plain `tomllib` assembly the loader and artifact GC read, the assembled `tomlkit` text CLI editors and `energy_grid.apply` mutate, the split write-back that rewrites only changed object files, and the artifact-store root (`catalog_root`). The bundled catalog is read with the user catalog layer (`catalog.user`; `profiles/` and energy-grid artifacts in the platform config dir): user profiles may not reuse bundled names, new profiles write to the layer, bundled profiles are read-only (`BundledProfileError`) except inside `bundled_profile_writes()` (`pyrite-dev profile`), and `catalog_root` of the bundled catalog is the layer. Import-light: no `pyrite.materials`; `tomlkit` only inside the text round trip.

## `_continuum_grid.py`
Photon-continuum node placement: the geometric baseline (`geometric_continuum_grid`), edge/endpoint refinement (`refined_continuum_grid`, `continuum_refinement_marks`) and the production default `automatic_continuum_grid` with its identity marker `CONTINUUM_GRID_MODEL` (#278). A root leaf so `campaign.sweep`/`campaign.profiles` can resolve an undeclared case `E_grid_brem` without importing `energy_grid` (which imports `campaign`); `energy_grid.refine` and `energy_grid.floor` re-export it. Deps: `_photon_continuum_floor`, `materials.attenuation`, `montecarlo.spectrum.cxr_seeds`.

## `console/dashboard/`
Shared live terminal dashboard package used by local scans and remote jobs. `render.py` owns frame composition, `state.py` owns progress parsing and terminal sanitization, and `poll.py` owns non-blocking keyboard input. `__init__.py` exports the fifteen names with callers outside the package and nothing else; anything absent from `__all__` is internal and imported from its owning module.
- Public: `KeyListener`, `render_frame`, `format_job_status`, `style_states`, `sanitize_terminal`, `marked_sections`, `parse_progress_records`, and the rest of `__all__`.
- Deps: `console.output` emitters, one deferred `materials.CATALOG` label read.

## `cli/commands/beam.py`
Canonical `pyrite beam list|show|create|set|rename|delete` Click layer over top-level `[beams.NAME]` catalog tables, attached to a profile by `beam = "NAME"`. `rename` cascades to every referencing profile and `delete` is blocked while a reference survives. This is the only CLI surface that writes beam distribution fields: `pyrite profile`'s inline beam flags were removed (issue #54), leaving `cli/commands/_beam_shared.py` to share option parsing, validation, and TOML writing between `create` and `set` alone. The reference resolves to values in `materials/catalog.py` before `profiles.py` hashes, so beam names never reach `parameter_sha256`. See [campaign configuration](../reference/campaign-configuration.md#named-beams-and-explicit-instruments) and [beam phase space](../physics/beam-transport/beam-phase-space.md).

## `cli/commands/detector.py`
Canonical `pyrite detector list|show|create|set|reset|rename|delete` Click layer over top-level `[detectors.NAME]` scalar/pixel objects. Profiles attach references through `detector = "NAME"` or ID collections; rename updates every reference and delete refuses while any survives. Named objects own reusable geometry and optional scorer/response/acquisition sections; `_profile_detectors.py` owns profile membership and inline edits. `_detector_shared.py` and `_pixel_detector_shared.py` share scalar/pixel options and updates. Names and display-only labels are removed during catalog resolution, so identity depends on resolved settings.

## Configuration editor services

`campaign/configuration_editor.py` adapts catalog drafts for the configuration
notebook: profile transactions, private beam copies, local detector slots,
validation/review and existing run-command generation. `apps/configuration_app.py`
owns reactive wiring; `configuration_controls.py`/`configuration_sections.py`
own typed forms and `configuration_exports.py` filters saved source-array
downloads. Launch: `pyrite app configuration launch` (port 2726). Guide:
[configuration notebook](../guides/configuration-notebook.md).

`catalog_edit.CatalogDraft` owns explicit-path in-memory drafts, canonical TOML,
validation, diffs, stale-edit checks and atomic selected-layer saving.
`detectors.editing` owns detector create/clone/patch and response/acquisition
merges, shared with profile edits and the typed detector CLI. Neither imports
Click command modules. `cli.commands.calculation` exposes API-owned
`physics`/`numerics` views and edits of existing profiles; it creates no preset
store. See [ADR-0019](../adr/0019-cli-configuration-ownership.md).

## `cli/commands/job.py`
Canonical `pyrite remote job list|status|logs|attach|stop` Click layer. Reuses the remote viewer, JSON envelope, completion, exit, confirmation, and lifecycle owners; normalizes persisted submitter names into the four public job kinds. The former root `job` alias was removed in 0.17.0.

## `cli/commands/checkpoint.py`
Canonical `pyrite checkpoint` group. Lazily routes `slim`, component `recompute {brem,cxr}`, `archive`, `restore`, `list`, `merge`, obsolete-record `gc`, and reachability-safe `rm` to existing checkpoint handlers.

## `checkpoints/checkpoint_cleanup.py`
Checkpoint reclamation drivers: `prune_checkpoints` (drops records no longer reproducible under the current catalog scan profiles, rewriting each dataset atomically) and `clear_checkpoints` (deletes explicitly selected datasets plus the CAS blobs they alone kept reachable). Selects by material, profile, or all; previews exact dataset paths and blobs; treats active and archived manifests as reachability roots; revalidates retained manifests before mutation and fails closed on malformed or changed state. Folded from the retired `prune.py`.

## `cli/commands/cleanup.py`
Click layer for `pyrite checkpoint gc` (`gc_command`) and `pyrite checkpoint rm` (`rm_command`), owning selector validation only. The hidden top-level `pyrite prune` alias resolves to `gc_command`.

## `perf/`
Performance sample collection and analysis. `performance_profile.py` owns the append-only runtime logger; `performance_analysis.py` owns artifact loading, summaries, and comparisons.
Opt-in cross-engine benchmark helpers (#510, driven from `checks/equal_precision/`): `precision.py` primary-history moments, `geant4.py` TestEm5 log parsing, `benchmark_matrix.py` frozen matrix expansion, `comparison.py` common per-primary comparison schema and cone acceptance quadrature.

## `cli/commands/performance.py`
Canonical local performance-artifact lifecycle: inventory, analysis, and preview-by-default profile deletion (`rm`; the retired `prune` spelling stays a hidden alias). `profile analyze` remains a hidden warning alias. Revalidates selected file signatures before deletion.

## `checkpoints/recompute.py`
Checkpoint repair and recompute owner for both component datasets. Per-record `repair_brem_wide` / `repair_cxr_spec`, checkpoint-level `repair_checkpoint` / `recxr_checkpoint`, and batch `rebrem_checkpoints` / `recxr_checkpoints` share one progress callback. Enumerates checkpoint stems and resolves per-material identity and fidelity defaults. Folded from the retired `rebrem.py`/`reline.py`.
Pure eligibility, target-grid validation and case-parameter planning live in `_repair_planning.py`; `_repair_progress.py` owns the shared deadline, callback and save cadence. Component execution stays in `recompute.py`, using the live runner scorers. Repair contracts live in `tests/checkpoint/test_recompute.py`.
- Deps: `checkpoints.persistence`, `checkpoints._checkpoint_store`, `checkpoints.recompute_defaults`, `_progress`, `energy_grid.encoding`, `montecarlo`; imports no run driver package.

## `cli/commands/recompute.py`
Click layer for `pyrite checkpoint recompute {brem,cxr}` (`brem_command`, `cxr_command`), including `--json` envelopes and the exclusive materials-or-`--all` selection contract.

## `checkpoints/recompute_defaults.py`
Fidelity-aware line/bremsstrahlung recompute defaults shared by local, grouped, and remote command paths. Resolves preset settings and material photon grids; keeps a compatibility fallback for older branches.

## `apps/analyze.py`
Notebook support for `src/pyrite/apps/analysis_app.py`: checkpoint loading, menu building, emission selection, and the initial-material precedence rule. Free of CLI imports so the notebook can load it on its own; the launcher lives in `cli/commands/app_analysis.py` and the persisted default both sides read lives in `_app_defaults.py`.
- Public: `material_menu`, `select_initial_material`, `face_menu`, `emission_menu`, `pick_spectrum`, `apply_emission`, `checkpoint_stem`, `analysis_checkpoint_manifest`, `load_analysis_checkpoint`, `cached_analysis`, `profile_menu`, `comparison_stem`, `initial_material`, `get_default_material`, `set_default_material`. `face_menu`/`profile_menu`/`checkpoint_stem` feed the apps' single **Checkpoint** dropdown (`analysis_ui/pickers.py`; blazed loads `<material>_blazed/` from `pyrite material blaze`). `emission_menu`/`apply_emission` back the app's **Emission** radio (Incoherent/Coherent), gated per checkpoint on the stored `spec`/`spec_coherent` and routing every spectrum read through the one `pick_spectrum`.

## `apps/viewer.py`
Notebook support for `src/pyrite/apps/trace_app.py`: the pure initial-material precedence rule. Free of CLI imports; the launcher lives in `cli/commands/app_viewer.py` and the persisted default both sides read lives in `_app_defaults.py`.
- Public: `initial_material`, `get_default_material`, `set_default_material` (the last two are thin re-spellings of the `_app_defaults` accessors).

## `apps/check.py`
Notebook support for `src/pyrite/apps/validation_app.py`: the persisted supplementary TMD azimuth plus optional remote Zhai-job launch/status/pull helpers driven from inside the notebook. Free of CLI imports; the launcher and cached-figure export live in `cli/commands/app_validation.py`.
- Public: `load_default_azimuth`, `save_default_azimuth`, `probe_remote_zhai`, `start_remote_zhai`, `remote_zhai_status`, `pull_remote_zhai`.
- Deps: `remote.config` (`remote_host`).

## Detector-free directional source

`directional.py::directional_source` lowers beam/target through a source-only
`SourceDirection` context, then `montecarlo/runner/directional_source.py` reuses
one transport for directional scoring of the selected components (PXR/CBS,
characteristic, bremsstrahlung, opt-in planar history TR; `directional_sources`).
`results/directional.py`
owns the typed component and mean-current scaling. The thin standalone
`apps/angular_heatmap_app.py` uses bounded cache/panel helpers in
`apps/angular_heatmap.py` (cached polar × azimuth orientations);
`apps/angular_cherenkov.py` supplies the separate internal bulk cone diagnostic,
per-bin normalization and export. `apps/angular_maps.py` owns the shared Plotly
angle × energy heatmap/profile figures, catalog material options, backend-aware
transport cores and form layout used by the heatmap, TR and optical apps.
`pyrite app angular-heatmap launch|export` uses the shared launcher on tunnel
port 2724; every shared-launcher app accepts `launch --backend`. No detector or
checkpoint required.

## `apps/transition_radiation.py` and `apps/transition_radiation_app.py`

The standalone TR/CTR demo lives in `apps/transition_radiation_app.py`, with
cached planar-solver/production adapters in `apps/transition_radiation.py`
(forward and backward straight-track maps, configurable medium/tilt/band).
`pyrite app transition-radiation launch|export` uses the shared `app_views`
launcher on tunnel port 2723; `tr_demo`/`ctr_demo` are its catalog references.

## Radiation diagnostic views

`apps/optical_diagnostics_app.py` is the bounded Plotly optical source,
collection/convergence, synthetic instrument/inference and saved-artifact app.
`apps/optical_diagnostics.py` caches existing solver adapters (`ScreenConfig`,
the constant-index Cherenkov cone map);
`apps/radiation_figures.py` renders stored arrays. The shared app launcher uses
`pyrite app optical-diagnostics launch|export` on tunnel port 2725.
`apps/transition_radiation.py::cached_dtr_reference` feeds an explicitly triggered
experimental DTR view. `apps/radiation_evidence.py` reads saved evidence for the
validation app without launching checks. `results/radiation.py` owns read-time
band yields and component semantics used by `apps/analysis_ui/radiation.py`.

## `apps/channeling.py` and `apps/channeling_app.py`

The channeling app also offers a bounded experimental axial mode through
`apps/axial_channeling.py` (entry presets, cached solves, stored-array plotting,
string-0 level densities and helicity-split line yields via `axial_radiation`).
`montecarlo/channeling/axial_diagnostic.py` assembles coherent/MCWF solvers with
typed configuration/results, incident-normalized density, optional string-centred
OAM and bounded ensemble metrics. `AxialMCWF` selects CPU or lazy CUDA execution;
`_axial_cuda*` owns device propagation, jumps and incident-addressed Philox draws.
`channeling/metrics.py` owns exact planar ensemble reductions and finite-grid
axial density-action sketches; `apps/channeling_metrics.py` renders stored
population/depth and evolution/accounting views without rerunning solvers.
There is no standard-run axial integration. Research helpers
`channeling/axial_phase_space.py` and `channeling/quantum_regression.py` supply
independently anchored finite-grid measurement and two-time correlation algebra,
with explicit allocation limits; physical escape and a persisted axial photon
source remain separate unimplemented contracts.
The phase-space reference accepts both pure states and mixed outgoing densities;
its mixed-state conditional sampler keeps coherences and incident trace without
constructing all joint outcomes or drawing a pure-state decomposition.
`channeling/axial_jump_instrument.py` supplies a bounded CPU absorbing-model
reference: one augmented exponential of the retained density and its
occupation integral retains recycling and separates outgoing and unresolved
tallies; a conditional sampler couples outgoing depth to the instantaneous
mixed phase-space gain. `channeling/axial_jump_unraveling.py` samples the same
law by terminal-event trajectories at retained-dimension cost. Both accept
supplied finite jump maps (`checks/axial_outgoing_maps.py` builds TDS maps).
`channeling/axial_fiber_unraveling.py` adds a classical Bloch-carrier label so
electronic kicks move trajectories between carrier fibres without wrapping,
with jump outputs measured on an outgoing order box;
`channeling/axial_tds_continuum.py` supplies its independent-string TDS jumps
(continuous carrier-moving transfers per string, exact displacement-mixture
rejection sampler, continuum loss); all of these work
without establishing a physical matching surface or native run admission.
`channeling/axial_outgoing_launch.py` maps fibre events (plus the surface
complement) under a declared matching cut to correlated research lab launches
(depth, position, CSDA energy, direction, clock, stream key) and per-incident
retained/physical-outgoing/unresolved fates; it is not a native `LaunchState`
adapter and does not supply the depth budget.
`channeling/axial_projection.py` provides the CPU research full/retained
no-jump matrix-exponential reference with native jump draws and step-resolved
jump times, used only by the opt-in return-flux comparison. It does not admit
standard axial transport.
`channeling/axial_qrt_source.py` adds a bounded global-grid eigensystem/current
factory and normalized on-axis finite RWA photon reference; it refuses oblique
directions and does not supply the actual material carrier-bath closure.
`apps/channeling.py::far_field_maps` draws fixed-basis circular far-field
intensity/phase of a planar or axial dipole line from `dipole.dipole_far_field`.

The planar channeling demo exposes eigenfunctions, entry populations, fate and
spectrum diagnostics, and sampled ensemble spatial density. The numerical
diagnostic `critical_angle_rad` uses the potential minimum and maximum. State,
depth and conditional-normalization controls only redraw cached solver output.
`montecarlo/channeling/spatial.py::spatial_density` owns the reusable Born-rule
reconstruction; `DensityMatrixSolver.propagate(spatial_x_ang=...)` optionally
records it at the existing recorded depths in the instantaneous basis.

## `cli/commands/app_analysis.py`
`pyrite app analysis launch` owner for `src/pyrite/apps/analysis_app.py`: persisted initial-material selection, smoke execution, edit/watch mode, ACP bridges, SSH-tunnel-friendly fixed-port launch. Imports no `apps` module — the persisted default comes from `_app_defaults.py` and the bridges from `_acp.py`, which is what takes `apps` out of the driver import cycle entirely (contract `nothing-imports-the-marimo-apps`).
- Public: `command`, `main`.

## `cli/commands/app_views.py`
`pyrite app pixels launch|export`, `pyrite app compare launch|export` and
`pyrite app channeling launch|export` share the `MarimoApp` launcher (headless
smoke, edit/watch, ACP bridges and static HTML export). Fixed tunnel ports are
2720/2721/2722. Pixels/compare resolve MATERIAL > the analysis app's persisted
default > `hopg`, passed as `--material` and `PYRITE_ANALYZE_INITIAL`. Channeling
targets `channeling_app.py` with no material argument. Imports no `apps` module.

## `cli/commands/app_viewer.py`
`pyrite app viewer launch|export` owner for `src/pyrite/apps/trace_app.py`: material-argument handling, smoke execution, static-HTML export, ACP bridges, fixed-port tunnel launch.
- Public: `command`, `export_command`, `main`.

## `cli/commands/app_validation.py`
`pyrite app validation launch|export` owner for `src/pyrite/apps/validation_app.py`. `pyrite check` is a hidden compatibility alias. `--export` skips marimo and renders cached figures through `validation.anchor_figures`.
- Public: `command`, `main`.

## `validation/_zhai.py`
Canonical maintained Zhai detector geometry and cache-schema provenance shared by anchor figures, headless reproduction, validation-app export, and remote SLURM metadata. Heavy calculations remain in `src/pyrite/validation/anchor_figures.py`; app and export consumers are cache-only and direct misses to `pyrite run --preset zhai --remote`.
- Internal: `ZHAI_DETECTOR`, `ZHAI_CACHE_SCHEMA`, `ZHAI_CACHE_FORMAT`, `detector_metadata`.
- Deps: `detectors.Detector`.

## `cli/commands/check_config.py`
`pyrite material validate` (hidden alias `pyrite check-config`): load the bundled material catalog or an explicit full catalog and print a one-line summary. No simulation, network access, or GPU probe. Body and Click wiring are both here; the body used to live in `validation/`, which had to reach up into `cli._core` (now `console.output`) for its output helpers — the whole of the `validation` ↔ `cli` import cycle.
- Public: `command`; internal `_run`.

## `remote/`
Optional SSH/SLURM orchestration for a configured remote host: sync, bounded and chunked submissions, progress/status/log viewers, checkpoint pulls, safe stop and clear, remote validation jobs.
- Public CLI: `add_subparser`, `main`.
- The public package contains the former facade and implementation modules (acyclic: `config` ◄ `transport` ◄ `scripts` ◄ `state` ◄ `lifecycle`/`viewer`; the Click wiring lives in `cli/commands/remote.py` and the frame rendering in `console/dashboard/`):
  - `config.py` — env-driven hosts/paths/SLURM constants.
  - `transport.py` — ssh/scp primitives, streamed downloads, hashing, code sync (per-sync-path rsync mirror when both ends have rsync, else generated-cache-filtered tar+scp; `PYRITE_SYNC_TRANSPORT`), selected external-catalog staging, material checks; one-round-trip immutable energy-grid hash inventory skips remote objects already present with matching content.
  - `scripts.py` — pure SLURM/shell string + command builders, job-id minting, per-allocation dependency-sync timing.
  - `state.py` — read-only job/reservation state queries over ssh; live-job discovery joins metadata against one scheduler snapshot.
  - `lifecycle.py` — submit/stage/stop/clear/pull job lifecycle; checkpoint pulls slim, stream, and clean up through one SSH session per stem; remote performance inventory and terminal-job-only pruning revalidate scheduler, job-state, and artifact inventory before deletion.
  - `viewer.py` — one-shot/attached status and logs rendering; attached status reuses one framed SSH stream across refreshes. Names re-export as import-time snapshots; internal cross-module calls resolve through the owning submodule, so tests patch the owner (e.g. `transport._ssh_capture`), not the facade.
  - `trajectories.py` — remote header inventory, case/energy-filtered rsync pulls with SHA-256 and completion checks, host subset VTK export, and SLURM spectrum replay. Click group: `cli.commands.remote_trajectories`; capture roots stay under the remote checkout and transfers are explicit.

## `cli/commands/export.py`
`pyrite app analysis export` subcommand — `marimo export html` of `src/pyrite/apps/analysis_app.py` → `pyrite-output/results/<stem>.html`. Body and Click wiring are both here.
- Public: `command`, `main`.

## `checkpoints/slim.py`
`pyrite checkpoint slim` subcommand — shrink checkpoint HDF5 for transfer (drop full-range brem arrays, downcast spectra to float32, filter configs; `--grid` keep only material's current-grid configs).
- Public: `slim_checkpoint`, `add_subparser`, `main`.
- Deps: `results` (`slim_results`, `_grid_names`).

## `checkpoints/archive.py`
`pyrite checkpoint archive store|restore|list|merge` subcommands (plus `checkpoint list` for active slots via `list_checkpoints`) — durable local checkpoint shelf. Copy active slot `checkpoints/<stem>/` (including resolved dataset identity in `meta.json`) to/from long-term `checkpoints/archive/<label>/` (atomic temp+replace, `--force` overwrite guards, label↔stem date-stamp inference). `merge` combines an archived checkpoint into active slot for same material and resolved dataset identity (compared via manifests when both have identity, then each store's `case["crystal"]`), live win on any overlapping (config name, E0) point; archive live checkpoint first by default (`--no-archive` to skip), leave source archive intact by default (`--delete-archive` to remove).
- Public: `archive_checkpoint`, `restore_checkpoint`, `list_archives`, `union_checkpoint`, `add_subparser`, `main`.

## `__init__.py`
Lightweight package root: expose `DATA_DIR`, `__version__`, and lazy supported scene/result objects plus `simulate`, without importing plotting or accelerator stacks during `import pyrite`.

## `devtools/`
Importable repository maintenance implementations behind `pyrite-dev`: package smoke, plot smoke, CLI reference/deprecation generation, and repository-map dependency generation. `complexity.py` owns the Radon CC report and high-CC ratchet against `docs/repo-design/complexity-baseline.json`; see [source complexity baseline](../repo-design/complexity.md). `startup.py` owns the fresh-process CPU import/README-simulation benchmark, with isolated cold Numba caches and a separate profile process; see [CPU startup latency](../repo-design/compute/startup-latency.md). Root `scripts/` files are thin compatibility wrappers or standalone checkout utilities; `checks/` contains developer-only physics anchors mapped to validation ids in `checks/README.md`.

## `apps/_compile_nb.py`
Internal notebook-compile helper; not public API.

## `montecarlo/gdf.py`

CPU-only EasyGDF adapter: raw GPT time/screen selection and coordinate inspection, electron validation,
correlated weighted resampling, and explicit lab-to-entrance projection.
`BeamSpec` resolves the source into hashed `Case.gdf_source` metadata; transport
receives numeric initial-state arrays through its existing backend boundary.

## Experimental DTR diagnostic

`montecarlo/spectrum/dtr_two_beam.py` adds an experimental CPU two-harmonic
Maxwell straight-passage diagnostic, with passive mode classification and
bounded face-referenced propagators. Its `dtr-two-beam-boundary` claim is
unverified. `spectrum/dtr_history.py` extends the boundary solve to finite
transport currents and moving lattice-force dipoles. `runner/dtr_emission.py`
owns the opt-in `Physics(transition_radiation="two-beam")` scorer, used by the
shared live/directions/replay/reline line seam. It returns the signed joint-field
correction to the separate crossing-TR component. Model/termination join identity
only when selected. The transported-current and CBS-phase claims remain
unverified; this CPU path supports one perfect crystal layer/reflection,
explicit node grids and equal line/continuum populations.

`montecarlo/spectrum/dtr.py` owns first-Born per-crossing diffraction of a
homogeneous stack's free TR waves and own-field diffraction on a straight
passage. `validation/dtr_reference.py` reads the shipped
`validation/reference_data/dtr_stack.toml` alternating SiO2/MoS2/SiO2 profile
and writes paired DTR-on/off spectra, signed interference, material/profile
provenance and a figure. Its opt-in `--two-beam` comparison appends repeated-
diffraction source-row densities, their signed cross term and the difference
from the Born total, with a separate model digest. No production case, CBS,
validated dynamical extinction or
validated high-energy transport is implied. Checks:
`tests/montecarlo/spectrum/test_transition_radiation_dtr*.py`. Derivation:
[experimental DTR](../physics/radiation-physics/diffracted-transition-radiation.md).

## Saved trajectory analysis

`montecarlo/trajectory_summary.py` owns bounded row-only histogram reductions and
per-primary summary streaming; `trajectory_balance.py` reuses the canonical
secondary energy ledger with local pair-event indices.
`plots/trajectory_capture.py` adapts bounded whole-history selections to the
existing shared trajectory projection; `plots/altair/trajectory_summary.py`
renders the summary tables. `apps/viewer.py::load_saved_capture` is the thin
trace-app loading owner. `campaign/beam_metrics.py::capture_beam_metrics` reads
a separately bounded prefix of initial sampled particles.

## Optical/IR diagnostic response

`materials/optical.py` owns explicit scalar local optical permittivity and the
pinned CC0 Rakić Au record under `data/optical/`. The source-only
`runner/transition_emission.py::case_layer_eps` keyword `layer_responses`
associates a provider with every layer without changing persisted cases.
`_component_support.py` (re-exported by `energy_grid/component_support.py`)
validates explicit medium/component bands
for node or midpoint-bin grids; it leaves the shared X-ray floor unchanged.
Independent derivation: `validation/radiation-physics/optical-material-response.md`.
Real-hardware check: `checks/optical_response_cuda.py`, run through remote staging
and a bounded CUDA allocation.
`OpticalResponse.to_record/from_record` preserve the data, supported segments,
provenance and response digest in a self-contained versioned JSON record;
restoration never fetches source data. Production Cherenkov remains proposed:
[integration contract](../research/physics/production-cherenkov.md).

Bounded optical TR/CTR (#615): `montecarlo/spectrum/optical_transition.py`
(physical-bunch source laws, weighted distinct-pair estimator, npz persistence),
`montecarlo/spectrum/optical_transition_validity.py` (formation and radiated-
energy refusals) and `runner/transition_emission.py::case_optical_transition`.
Physics: `physics/radiation-physics/optical-transition-radiation.md`; CUDA
parity: `checks/optical_transition_cuda.py`.

Bulk Cherenkov source (#587): `montecarlo/spectrum/cherenkov.py`
(`frank_tamm_yield`: Frank–Tamm yield and cone for an explicit weak-loss
permittivity; `BulkEmissionMechanism` extension point for interface-free
sources; `ConstantIndex` lossless nondispersive input). Standalone, not wired
into transport or scoring; the optical diagnostics app maps its cone. Physics:
`physics/radiation-physics/cherenkov-radiation.md`; Geant4 reference:
`checks/cherenkov_geant4`.

## `validation/inner_shell_experiment.py`

Loads cited thin-film K and Pb/Th L1–L3 measurements from
`validation/reference_data/inner_shell_ionization/`, evaluates EEDL and the
optional Bote–Salvat tabulated scoring backend at measured energies, and
summarizes model/measured ratios by overvoltage band. Developer report:
`checks/inner_shell_experiment_comparison.py`; evidence and limitations:
[backend validation](../validation/radiation-physics/bote-salvat-ionization-backend.md).

## Experimental physical photon event sources

`photon_events.py::simulate_photon_events` lowers a Scene and runs one retained
primary charged cascade, selecting either characteristic relaxation/Auger
emissions or recorded annihilation photons. It reuses
`runner/photon_emission.py::score_photon_sources` for ordinary aperture hits,
with optional photon-born charged feedback and separate observation identity.
`PhotonEventObservation` retains the primary relaxation budget when selected.
`photon_event_observation.py` saves/loads reduced event aperture tallies,
tracked/sub-cutoff source energies and optional signed primary relaxation
budgets as an explicit analysis artifact; no transport or table fetch on load.
No weighted-continuum summation, production Result/checkpoint/replay integration
or independent full-track validation is implied. Contract:
[CPU photon feedback](../research/physics/photon-secondary-feedback.md#scene-physical-event-source-boundary).
