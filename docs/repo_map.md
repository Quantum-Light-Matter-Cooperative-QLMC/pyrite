# Repository map

Navigation aid for `src/cxr_mc/` — importable package. Read before exploring
source. For *why* (physics, validation, provenance) see
[`README.md`](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/cxr-mc/blob/main/README.md)
and design notes in
[`docs/`](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/cxr-mc/tree/main/docs);
backlog in
[`TODO.md`](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/cxr-mc/blob/main/TODO.md).
Print current top-level directory inventory:
`uv run python scripts/dev.py repo-map`.

## Dependency layers (leaf → driver)

```
materials._transport_data      (transport-supported element constants; leaf)
materials._cif                 (crystals-backed CIF parsing/symmetry expansion; leaf)
materials.atomic               (xraydb-backed atomic data; no sibling deps)
        │
materials.catalog              (offline declarative catalog; CIF + transport validation)
        │
materials.crystal              (catalog projection, structure factor, χ_g/U_g, μ)
        ├── validation_oracles (optional external comparators; checks only)
        │
materials.attenuation          (composition and Beer–Lambert helpers)
        │
montecarlo                     (transport + radiation + detector helpers)
        │
sweep ─────────────┐
        │          │
results ◄── montecarlo, sweep
profiles ◄── results, sweep
config  ◄── profiles, results, sweep
run     ◄── montecarlo, results
scan    ◄── config, run, sweep
plots   ◄── montecarlo, results, detectors.timepix_response, detectors.eaglexo_response
cli     ◄── analyze, archive, check, check_config, export, remote, scan, slim
                                  (the `cxr` console script)
```

`detectors.timepix_response` / `detectors.eaglexo_response` depend only on `materials.crystal` plus
shared `_si_sensor` plumbing (Si constants, response caching, Poisson core).
Packaged data resolve via `cxr_mc.DATA_DIR` — imports work from any cwd.

## Entry points

- **`cxr` console script** → `cli:main` (`pyproject.toml [project.scripts]`),
  lazy Click dispatch for `scan`, `blaze`, `export`, `analyze`, `validate`,
  `catalog`, `checkpoint`, `remote`, and `line-grid`. Older flat checkpoint
  verbs, `check`, and `check-config` remain hidden compatibility aliases.
  Checked user-facing inventory:
  [`docs/cli-reference.md`](cli-reference.md).
- **`cxr catalog validate [catalog]`** → `check_config:_run`: validate bundled
  offline catalog or explicit complete catalog without starting simulation;
  hidden alias: `cxr check-config`.
- **`cxr checkpoint ...`** → `checkpoint_cli:command`: grouped local checkpoint
  shrink, component recompute, archive, restore, list, and merge operations.
- **`cxr scan <material> [--profile full|survey]`** → `scan:main` →
  `run.run_sweep` → write canonical `checkpoints/<material>/{line,brem}.pkl`
  or an identity-qualified variant directory. Root shim: `scan.py`.
- **Marimo apps**: `notebooks/scan_app.py` (sweep runner → checkpoint),
  `notebooks/analysis_app.py` (all figures, Altair + matplotlib, lazy tabbed
  layout), `notebooks/validation_app.py` (validation-study interface). Scan
  and analysis apps read per-material grids in `config.py`.
- **`cxr analyze [material]`** → `analyze:_cli`: launch or smoke-test analysis
  app with explicit or persisted initial material.
- **`cxr validate`** → `check:_cli`: launch validation app or export its
  literature-validation figures from cached results; hidden alias: `cxr check`.
- **`cxr remote ...`** → `remote:*`: optional SSH/SLURM lifecycle for lab GPU
  box: submit, attach/status/logs, pull, stop, validation jobs. Canonical
  detached submission and validation paths are `remote submit` and
  `remote validate`; `start` and `check` remain hidden aliases.
- **`cxr export [stem]`** → `export:main`: `marimo export html` of analysis
  app → `results/<stem>.html`.
- **`cxr slim <checkpoint-dir> [--grid]`** → `slim:slim_checkpoint` →
  `results.slim_results`: shrink checkpoint pickle for transfer (drop
  wide-brem / float32 / filter configs; `--grid` keep only material's
  current-grid configs).
- **`cxr archive`/`restore`/`archives`/`union`** → `archive:*`: local
  checkpoint shelf — copy active slot `checkpoints/<stem>/` to/from
  long-term `checkpoints/archive/<label>/`; `union` merge shelved
  checkpoint back into active slot for same material.
- **Sweep worker**: `montecarlo.run_case` (module-level so it pickle into
  `run_cases` process pool).

---

## Core physics

### `materials/` (package)
Material domain package. Narrow top-level API expose immutable
`CATALOG`, frozen record types (`MaterialCatalog`, `CrystalInfo`,
`CrystalSpec`, `MediumSpec`, `MaterialSpec`, `ScanSpec`, `LayerSpec`),
`load_material_catalog`, compatibility projections (`CRYSTALS`, `MATERIALS`,
`MATERIAL_LABELS`). Implementation helpers stay in submodules below.

### `materials/catalog.py`
Schema-version-1 loader for packaged `data/materials.toml`. Resolve only
phase-specific CIFs below packaged `data/cifs`, validate scan descriptors,
transport support, pinned-reflection policy, stacks, then return deeply
immutable typed records. `material_keys` preserve TOML declaration order.
- Public: `MaterialCatalog`, `MaterialConfigError`, `CrystalInfo`, `CrystalSpec`,
  `MediumSpec`, `MaterialSpec`, `ScanSpec`, `LayerSpec`, `load_material_catalog`.
- Deps: `materials._cif`, `materials._transport_data`, `materials._catalog_decode`,
  `DATA_DIR`.

### `materials/_catalog_decode.py`
Primitive decoders for schema-version-1 catalog grids and descriptors:
number/negative validation, immutable float64 grid construction,
`values`/`arange`/`linspace`/`logspace` line-grid kinds. `catalog.py` orchestrate
these, own record types and error aggregation.
- Internal: grid/descriptor decode helpers; `GridValue`, `LineGridByEnergy` aliases.
- Deps: NumPy.

### `materials/_cif.py`
Structural adapter around `crystals` 1.7: CIF parsing, symmetry expansion, cell
parameters, fractional sites, volume only. cxr-mc retain ownership of form
factors, structure factors, reflection selection, attenuation, transport.
- Internal: `crystals_crystal_to_crystal_info`, `load_crystal_from_cif`.
- Deps: external `crystals`, NumPy.

### `materials/crystal.py`
Catalog-backed crystal compatibility projection, structure factors, X-ray
optical constants — physics data layer under Monte Carlo.
- Public: `load_crystals`, `load_crystal_from_cif`,
  `crystals_crystal_to_crystal_info`, `reciprocal_g_vector`, `g_mag`,
  `debye_waller`, `structure_factor`, `chi_g`, `U_g`,
  `absorption_length_ang`, `dominant_reflections`, `beta_from_Ee`; `CRYSTALS`
  registry.
- Deps: `materials.atomic`, `materials.catalog`, `materials._cif`.

### `validation_oracles.py`
Optional validation-only adapters for external crystallography/scattering
comparators. First backend build or load `Dans_Diffraction` crystals, compare
lattice parameters, reciprocal-vector magnitudes, `|F_hkl|²`; production
physics stay in `materials/crystal.py`.
- Public: `build_dans_crystal_from_cxr`, `load_dans_crystal_from_cif`,
  `compare_lattice`, `compare_reflection_geometry`,
  `compare_structure_factor_magnitudes`; comparison dataclasses.
- Deps: `materials.crystal`; imports `Dans_Diffraction` lazy, only when check
  ask.

### `validation_background.py`
Analysis-only external bremsstrahlung comparison, weighted sideband
normalization, and experimental subtraction. External spectra stay in detected
units and enter through `montecarlo.load_external_brem`.
- Public: `BackgroundFit`, `BackgroundComparison`,
  `fit_external_background`, `subtract_external_background`,
  `compare_external_background`.
- Deps: `montecarlo`, NumPy.

### `materials/atomic.py`
Atomic scattering factors (Z, f0, f′, f″) from **xraydb** for any element — no
hand-maintained table.
- Public: `cromer_mann_f0`, `henke_dispersion`, `atomic_form_factor`,
  `load_henke`.
- Deps: none (leaf; external xraydb).

### `montecarlo/` (package)
Simulation core: electron transport, segment-sum PXR+CBS line spectrum,
bremsstrahlung, parallel case runner, detector-convolution helpers.
Split from single module into submodules; **every public and internal name
re-exported from package** — `from cxr_mc.montecarlo import X` unchanged
(`tests/test_montecarlo_exports.py` freeze export set).
- `_backend` — GPU/CPU array backend probe + banner: `xp`, `cp`, `REAL`,
  `_to_cpu`, `_GPU`.
- `materials.attenuation` — `_normalize_composition`, `_mu_total_inv_ang`, `_layer_dz`,
  `_stack_tau` (composition + cross-stack self-absorption). Deps: `_backend`,
  `materials.crystal`.
- `transport` — `simulate_trajectories` (multilayer-stack aware via `layers=`),
  `beta_from_keV`, scattering/stopping helpers; `TRANSPORT_ELEMENTS`
  registry. Pure NumPy. Deps: `materials.attenuation`, `DATA_DIR`.
- `geometry` — `tilted_geometry`, `detector_directions`, `_orientation_R`,
  `_small_tilt_R`, `_mosaic_quadrature`. Deps: `materials.crystal`.
- `spectrum` — `mc_spectrum` (PXR+CBS, cross-stack self-absorption, exact mosaic
  average), `mc_spectrum_solid_angle`, `mc_brem_spectrum`, `load_external_brem`.
  Deps: `_backend`, `materials.attenuation`, `transport`, `geometry`, `materials.crystal`.
- `detector` — `detector_efficiency`, `eds_fwhm_eV`, `aperture_fwhm_eV`,
  `mosaic_fwhm_eV`, `mosaic_psi_rad`, `convolve_detector`. Deps: `materials.attenuation`,
  `geometry`, `transport`, `materials.crystal`.
- `montecarlo/groove.py` — blazed sawtooth entrance-face grooves (escape-path engineering): closed-form entry/escape, `Sweep.groove_spacing_ang` knob.
- `runner` — `run_case`, `run_cases` (CPU transport pipelined behind one CUDA
  spectrum context, or memory-capped full-case CPU pool), `_transport_case`,
  `_spectrum_case`, `_worker_init`. Deps: `_backend`, `transport`, `geometry`,
  `spectrum`.
- Deps: `materials.crystal`, `materials.attenuation`, `DATA_DIR`.

## Sweep, config & drivers

### `sweep.py`
Turn `Sweep` definition into Cartesian product of `run_case` dicts.
- Public: `Sweep` (dataclass of all knobs), `LayerSpec` (one stack layer: material,
  thickness, orientation), `build_cases`, `crystal_params`,
  `substrate_composition`, `stack_layers`, `film_on_substrate_layers`,
  `layer_radiator`, `substrate_radiator`, `geometry_table`,
  `fmt_thickness`, `pm` (±hkl expansion); `MATERIAL_LABELS` registry.
- Deps: `materials` (`CATALOG`, `LayerSpec`), `materials.crystal`.

### `config.py`
Default settings/sweep builders shared by CLI and both notebooks; per-material
scan grids project from immutable `materials.CATALOG`.
- Public: `default_settings`, `material_grid`, `material_sweep`,
  `trajectory_sweep`; `MATERIALS` ordered tuple.
- Deps: `materials` (`CATALOG`, `MaterialSpec`), `results` (`Settings`),
  `profiles` (`get_profile`), `sweep` (`Sweep`).

### `profiles.py`
Named `full`/provisional `survey` policies, deterministic serialization of
resolved settings and sweeps, SHA-256 dataset identity, and variant checkpoint
stem selection.
- Public: `SweepProfile`, `PROFILE_NAMES`, `get_profile`, `dataset_identity`,
  `variant_stem`.
- Deps: `results` (`Settings`), `sweep` (`Sweep`), NumPy.

### `run.py`
Checkpointed, resumable sweep driver plus component checkpoint loaders/repair.
- Public: `run_sweep`, `load_checkpoint`, `checkpoint_path_for`,
  `cases_from_results`, `repair_brem_wide`, `repair_checkpoint`.
- Deps: `montecarlo` (`run_cases`), `results` (`store_result`).

### `_checkpoint_store.py`
Component storage adapter: active datasets live under
`checkpoints/<stem>/{line,brem}.pkl`, merge transparently into historical
in-memory result records, and migrate legacy `checkpoints/<stem>.pkl` stores on
next save.
- Internal: `discover`, `load`, `save`, `signature`, component/path helpers.
- Deps: `_checkpoint_io`, NumPy.

### `scan.py`
Headless sweep entry: parse args → build cases → `run_sweep` → checkpoint.
- Public: `main`, `run`, `add_subparser`.
- Deps: `config`, `run`, `sweep`.

### `blaze.py`
Headless blazed-crystal (sawtooth entrance face) sweep entry: `cxr blaze <material>
--energy E [E...] --spacing S [S...] [--angles A [A...]]`. Mirrors `scan.py`'s
parse args → build cases → `run_sweep` → checkpoint structure, but forces v1
groove geometry (`theta_obs=90`, `tilt_azim=180`, no substrate/stack/footprint)
per (energy, spacing) pair and writes to a dedicated
`checkpoints/<material>_blazed.pkl`, never the flat-face `<material>.pkl`.
- Public: `main`, `run`, `add_subparser`.
- Deps: `config`, `run`, `scan` (`validate_materials`, `_write_progress_record`), `sweep`.

## Results & plotting

### `results/` (package)
Result records, derived line metrics, ranking/selection. Split from single
module into submodules; **every public and internal name re-exported from
package** — `from cxr_mc.results import X` unchanged
(`tests/test_results_exports.py` freeze export set).
- `store` — `{config_name: {E0_keV: record}}` store: `Settings` (dataclass),
  `store_result`, `detected_background`, `PER_NA`.
- `selection` — subset/reduce store: `records`, `records_for_cases`,
  `filter_results`, `select_results`, `sweep_values`, `slim_results`,
  `best_azimuth`.
- `metrics` — per-record scalars for heatmaps: `line_index`, `line_quality`,
  `line_metrics`.
- `scoring` — geometry ranking: `SELECTION_MODES`, `selection_score`,
  `top_geometries`, `show_top`.
- `tables` — DataFrame views: `results_dataframe`, `summary_table`,
  `show_summary`.
- Deps: `montecarlo`, `sweep`.

### `plots/` (package)
All plotting — Matplotlib/Plotly. Split from single module into submodules by
figure type; **every public and internal name re-exported from package** —
`from cxr_mc.plots import X` unchanged (`tests/test_plots_exports.py`
freeze export set). Submodule DAG (leaf → driver):
`_style → _common → _frames → sweeps → {spectra, detectors, trajectories} → interactive`.
- `_style` — `COLORS`, `_ENERGY_PALETTE`, `energy_color` (per-energy colour map
  consistent across every figure). Leaf; no sibling deps.
- `_common` — shared figure plumbing: `_line_brem` (per-record line/brem split),
  `_per_tilt_figs` (one-figure-per-tilt loop), `_mode`, `_EFF_CACHE`. Deps:
  `montecarlo`, `results`.
- `_frames` — renderer-neutral tidy-data builders (`heatmap_frame`, `metric_vs_frame`,
  `scan_mode`, `pick_hue`, `_effective_x`/`_ndistinct`, axis/value-label registries
  `_AXIS_SPECS`/`_axis_disp`/`_value_label`/`_FLUX_GATED`): per-cell/per-point
  best-record reduction shared by matplotlib `sweeps.py` and `altair_sweeps.py`. Leaf-most
  of sweep-figure modules; no matplotlib/Altair imports. Deps: `_common`, `results`,
  `pandas`.
- `spectra` — `plot_by_energy`, `plot_full_spectrum`, `plot_peak_vs_tilt`,
  `plot_mosaic_comparison`, `plot_best_spectra`, `plot_material_comparison`,
  `plot_tilt_panel`, `_draw_*` spectral drawers. Deps: `_style`, `_common`,
  `montecarlo`, `results`.
- `sweeps` — `plot_heatmaps`, `facet_metric` (small-multiples over many knobs),
  `plot_metric_vs`, `plot_scan`; `_HEATMAP_QUANTITIES` / `_METRIC_LABELS`
  tables + axis helpers. Render from `_frames` tidy DataFrames, not re-derive
  best-record reduction inline. Deps: `_style`, `_frames`, `results`.
- `detectors` — `plot_timepix_efficiency` / `_detected` / `_poisson`,
  `plot_eaglexo_efficiency` / `_detected` / `_charge` / `_charge_map`. Deps:
  `_style`, `_common`, `sweeps`, `results`, `detectors.timepix_response`, `detectors.eaglexo_response`.
- `trajectories` — `plot_electron_trajectories`, `plot_trajectory_grid`,
  `plot_penetration_survival`. Deps: `_style`, `montecarlo`, `results`.
- `interactive` — `browse`, `browse_plotly`, `stream_chunk`, `plot_chunk`
  (slider/streaming drivers dispatching to `spectra`/`detectors` drawers).
  Top of DAG. Deps: `_style`, `_common`, `spectra`, `detectors`.
- `altair_*` — Altair/Vega-Lite renderers, interactive counterparts of
  matplotlib figures (marimo `analysis_app.py` use these first). Share exact
  data prep with matplotlib path (`_common._line_brem`, `_frames` builders,
  `detectors`/`trajectories` internals) — physics identical, only renderer
  differ. Intentionally **NOT** re-exported from package (frozen export
  guard) — import from submodule. Per-module guard tests: `tests/test_altair_*.py`.
  - `altair_spectra` — intrinsic spectra: `spectrum_chart`, `spectrum_frame`,
    `compare_spectrum_chart` (overlay one line per E0/tilt/azimuth, for
    Energy/Polar-angle/Azimuthal comparison notebook tabs). Deps: `_common`,
    `results`.
  - `altair_sweeps` — metric scans + parametric heatmaps: `metric_vs_chart`,
    `heatmap_chart`, `scan_charts` (auto heatmap-vs-lines, one shared metrics
    map across quantities). Deps: `_common`, `_frames`, `sweeps`, `results`.
  - `altair_detectors` — Timepix3/Eagle XO spectral views:
    `timepix_detected_chart`, `eaglexo_detected_chart`, `eaglexo_charge_chart`
    (+ their `*_frame` builders). Deps: `_common`, `altair_spectra`,
    `detectors`, `detectors.eaglexo_response`.
  - `altair_trajectories` — penetration views: `penetration_survival_chart`,
    `trajectory_chart` (+ `survival_frame`, `tracks_frame`,
    `track_segments_frame`); dense datashader raster stay on matplotlib.
    Deps: `sweeps`, `trajectories`.
- Deps: `montecarlo`, `results`, `detectors.timepix_response`, `detectors.eaglexo_response`.

## Detector forward models

### `detectors/`
Detector and detector-adjacent forward models. Deps: `materials.crystal`,
`DATA_DIR`.

#### `_si_sensor.py`
Internal shared silicon-sensor plumbing for both detector forward models:
fixed Si material constants, (grid → cached-response) keying pattern,
Poisson acquisition core, `.apply()` input-shape guard. Leaf; no
sibling deps.

#### `timepix_response.py`
Timepix3 charge-sensitive forward model (diffusion, absorption, energy
resolution, Poisson counts).
- Public: `TimepixResponse`, `build_response`, `get_response`,
  `absorption_efficiency`, `energy_fwhm_eV`, `sigma_diffusion_um`,
  `poisson_counts`.
- Deps: `materials.crystal`, `_si_sensor`.

#### `eaglexo_response.py`
Eagle XO detector forward model (geometry/solid angle, QE table, energy
resolution, Poisson counts).
- Public: `EagleResponse`, `geometry`, `sweep_geometry`, `get_response`, `qe`,
  `qe_absorption_model`, `load_qe_table`, `solid_angle_sr`, `energy_fwhm_eV`,
  `poisson_counts`.
- Deps: `materials.crystal`, `_si_sensor`.

#### `grating.py`
**Exploratory** grazing-incidence soft-X-ray grating spectrometer forward model
(dispersion geometry, coating reflectivity, simple CCD pixel grid; not
wired into pipeline). See [`docs/grazing-grating.md`](grazing-grating.md).
- Public: `Grating`, `wavelength_angstrom`, `groove_spacing_angstrom`,
  `coating_number_density_per_ang3`, `detector_position_mm`, `disperse_spectrum`,
  `resolving_power`, `ALEXS_SENSORS`, `SimpleCCD`, `bin_to_pixels`.
- Deps: `materials.crystal` (`HC_EV_ANG`, `optical_constants`).

## CLI & packaging

### `cli.py`
`cxr` console-script dispatcher.
- Public: `main`.
- Deps (lazy command imports): `analyze`, `blaze`, `catalog_cli`, `check`,
  `checkpoint_cli`, `export`, `line_grid`, `remote`, `scan`; hidden
  compatibility paths additionally dispatch to `archive`, `check_config`,
  `rebrem`, `reline`, and `slim`. Eager lightweight deps: `_cli_core`,
  `__version__`.

### `checkpoint_cli.py`
Canonical `cxr checkpoint` group. Lazily routes `slim`, component
`recompute {brem,line}`, `archive`, `restore`, `list`, and `merge` to existing
checkpoint handlers while root-level legacy paths remain hidden aliases.

### `recompute_defaults.py`
Profile-aware line/bremsstrahlung recompute defaults shared by local, grouped,
and remote command paths. Resolves profile settings and material photon grids;
keeps a compatibility fallback for branches predating named sweep profiles.

### `catalog_cli.py`
Canonical `cxr catalog` group. Lazily routes `catalog validate` to
`check_config.command` while `cxr check-config` remains a hidden alias.

### `analyze.py`
`cxr analyze` launcher for `notebooks/analysis_app.py`: persisted
initial-material selection, smoke execution, edit/watch mode, ACP bridges,
SSH-tunnel-friendly fixed-port launch.
- Public: `material_menu`, `select_initial_material`, `face_menu`,
  `checkpoint_stem`, `initial_material`, `get_default_material`,
  `set_default_material`, `command`, `main`. `face_menu`/`checkpoint_stem`
  back the app's flat/blazed **Face** dropdown (blazed loads
  `<material>_blazed.pkl` from `cxr blaze`).

### `check.py`
`cxr check` launcher for `notebooks/validation_app.py` plus cached validation
figure export, optional remote Zhai-job launch/status/pull helpers.
- Public: `load_default_azimuth`, `save_default_azimuth`, `probe_remote_zhai`,
  `start_remote_zhai`, `remote_zhai_status`, `pull_remote_zhai`,
  `add_subparser`, `main`.

### `check_config.py`
`cxr check-config` validate bundled material catalog or explicit full
catalog without importing GPU-heavy CLI modules.
- Public: `add_subparser`, `main`.

### `remote.py` (facade over `_remote/`)
Optional SSH/SLURM orchestration for configured lab box: sync, bounded and
chunked submissions, progress/status/log viewers, checkpoint pulls, safe stop
and clear, remote validation jobs.
- Public CLI: `add_subparser`, `main`.
- Thin re-export facade. Implementation split into `_remote/` submodules
  (acyclic: `config` ◄ `transport` ◄ `scripts` ◄ `state` ◄ `lifecycle`/`viewer`
  ◄ `cli`; plus `presentation`):
  - `config.py` — env-driven hosts/paths/SLURM constants.
  - `transport.py` — ssh/scp primitives, hashing, code-tar sync, material checks.
  - `scripts.py` — pure SLURM/shell string + command builders, job-id minting.
  - `state.py` — read-only job/reservation state queries over ssh.
  - `lifecycle.py` — submit/stage/stop/clear/pull job lifecycle.
  - `viewer.py` — live attach/status/logs rendering.
  - `cli.py` — argparse wiring and subcommand dispatch.
  Names re-export as import-time snapshots; internal cross-module calls resolve
  through the owning submodule, so tests patch the owner (e.g.
  `transport._ssh_capture`), not the facade.

### `export.py`
`cxr export` subcommand — `marimo export html` of `notebooks/analysis_app.py`
→ `results/<stem>.html` (replace retired nbconvert-PDF path).
- Public: `add_subparser`, `main`.

### `slim.py`
`cxr slim` subcommand — shrink checkpoint pickle for transfer (drop
full-range brem arrays, downcast spectra to float32, filter configs; `--grid`
keep only material's current-grid configs).
- Public: `slim_checkpoint`, `add_subparser`, `main`.
- Deps: `results` (`slim_results`, `_grid_names`).

### `archive.py`
`cxr archive`/`restore`/`archives`/`union` subcommands — durable local
checkpoint shelf. Copy active slot `checkpoints/<stem>/` (including resolved
dataset identity in `meta.json`) to/from
long-term `checkpoints/archive/<label>/` (atomic temp+replace, `--force`
overwrite guards, label↔stem date-stamp inference). `union` merge archived
checkpoint into active slot for same material and resolved dataset identity
(compared via manifests when both have identity, then each store's
`case["crystal"]`), live win on any overlapping (config name, E0) point;
archive live checkpoint first by default (`--no-archive` to skip), leave
source archive intact by default (`--delete-archive` to remove).
- Public: `archive_checkpoint`, `restore_checkpoint`, `list_archives`,
  `union_checkpoint`, `add_subparser`, `main`.

### `__init__.py`
Package root: expose `DATA_DIR` (packaged-data resolver) and `__version__`.

### `_compile_nb.py`
Internal notebook-compile helper; not public API.
