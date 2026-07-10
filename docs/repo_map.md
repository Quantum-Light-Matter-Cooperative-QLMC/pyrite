# Repository map

Navigation aid for `src/cxr_mc/` — the importable package. Read this before
exploring source. For *why* (physics, validation, provenance) see
[`README.md`](../README.md) and the design notes in [`docs/`](.); for the backlog
see [`TODO.md`](../TODO.md). Regenerate with the `docs:update-repo-map` command.

## Dependency layers (leaf → driver)

```
atomic_form_factors            (xraydb-backed atomic data; no sibling deps)
        │
crystallography                (crystal DB, structure factor, χ_g/U_g, μ)
        ├── validation_oracles (optional external comparators; checks only)
        │
montecarlo                     (transport + radiation + detector helpers)
        │
sweep ─────────────┐
        │          │
results ◄── montecarlo, sweep
config  ◄── results, sweep
run     ◄── montecarlo, results
scan    ◄── config, run, sweep
plots   ◄── montecarlo, results, timepix_response, eaglexo_response
cli     ◄── scan, export          (the `cxr` console script)
```

`timepix_response` / `eaglexo_response` depend only on `crystallography` plus
the shared `_si_sensor` plumbing (Si constants, response caching, Poisson core).
Packaged data resolves via `cxr_mc.DATA_DIR`, so imports work from any cwd.

## Entry points

- **`cxr` console script** → `cli:main` (`pyproject.toml [project.scripts]`),
  dispatching the `scan`, `export`, `slim`, `archive`, `restore`, `archives` and
  `union` subcommands.
- **`cxr scan <material>`** → `scan:main` → `run.run_sweep` → writes
  `checkpoints/<material>.pkl`. Root shim: `scan.py`.
- **Marimo apps**: `notebooks/scan_app.py` (sweep runner → checkpoint) →
  `notebooks/analysis_app.py` (all figures, Altair + matplotlib, lazy tabbed
  layout); both read the per-material grids in `config.py`.
- **`cxr export [stem]`** → `export:main`: `marimo export html` of the analysis
  app → `results/<stem>.html`.
- **`cxr slim <checkpoint> [--grid]`** → `slim:slim_checkpoint` →
  `results.slim_results`: shrink a checkpoint pickle for transfer (drop
  wide-brem / float32 / filter configs; `--grid` keeps only the material's
  current-grid configs).
- **`cxr archive`/`restore`/`archives`/`union`** → `archive:*`: the local
  checkpoint shelf — copy the active slot `checkpoints/<stem>.pkl` to/from the
  long-term `checkpoints/archive/<label>.pkl`; `union` merges a shelved
  checkpoint back into the active slot for the same material.
- **Sweep worker**: `montecarlo.run_case` (module-level so it pickles into the
  `run_cases` process pool).

---

## Core physics

### `crystallography.py`
Crystal database, structure factors, and X-ray optical constants — the physics
data layer under the Monte Carlo.
- Public: `load_crystals`, `load_crystal_from_cif`,
  `diffpy_structure_to_crystal_info`, `reciprocal_g_vector`, `g_mag`,
  `debye_waller`, `structure_factor`, `chi_g`, `U_g`,
  `absorption_length_ang`, `dominant_reflections`, `beta_from_Ee`; the
  `CRYSTALS` registry.
- Deps: `atomic_form_factors`, `DATA_DIR`.

### `validation_oracles.py`
Optional validation-only adapters for external crystallography/scattering
comparators. The first backend builds or loads `Dans_Diffraction` crystals and
compares lattice parameters, reciprocal-vector magnitudes, and `|F_hkl|²` while
leaving production physics in `crystallography.py`.
- Public: `build_dans_crystal_from_cxr`, `load_dans_crystal_from_cif`,
  `compare_lattice`, `compare_reflection_geometry`,
  `compare_structure_factor_magnitudes`; the comparison dataclasses.
- Deps: `crystallography`; imports `Dans_Diffraction` lazily only when a check
  asks for it.

### `atomic_form_factors.py`
Atomic scattering factors (Z, f0, f′, f″) from **xraydb** for any element — no
hand-maintained table.
- Public: `cromer_mann_f0`, `henke_dispersion`, `atomic_form_factor`,
  `load_henke`.
- Deps: none (leaf; external xraydb).

### `montecarlo/` (package)
The simulation core: electron transport, the segment-sum PXR+CBS line spectrum,
bremsstrahlung, the parallel case runner, and detector-convolution helpers.
Split from a single module into submodules; **every public and internal name is
re-exported from the package**, so `from cxr_mc.montecarlo import X` is unchanged
(`tests/test_montecarlo_exports.py` freezes the export set).
- `_backend` — GPU/CPU array backend probe + banner: `xp`, `cp`, `REAL`,
  `_to_cpu`, `_GPU`.
- `materials` — `_normalize_composition`, `_mu_total_inv_ang`, `_layer_dz`,
  `_stack_tau` (composition + cross-stack self-absorption). Deps: `_backend`,
  `crystallography`.
- `transport` — `simulate_trajectories` (multilayer-stack aware via `layers=`),
  `beta_from_keV`, scattering/stopping helpers; the `TRANSPORT_ELEMENTS`
  registry. Pure NumPy. Deps: `materials`, `DATA_DIR`.
- `geometry` — `tilted_geometry`, `detector_directions`, `_orientation_R`,
  `_small_tilt_R`, `_mosaic_quadrature`. Deps: `crystallography`.
- `spectrum` — `mc_spectrum` (PXR+CBS, cross-stack self-absorption, exact mosaic
  average), `mc_spectrum_solid_angle`, `mc_brem_spectrum`, `load_external_brem`.
  Deps: `_backend`, `materials`, `transport`, `geometry`, `crystallography`.
- `detector` — `detector_efficiency`, `eds_fwhm_eV`, `aperture_fwhm_eV`,
  `mosaic_fwhm_eV`, `mosaic_psi_rad`, `convolve_detector`. Deps: `materials`,
  `geometry`, `transport`, `crystallography`.
- `runner` — `run_case`, `run_cases` (GPU-serial / CPU-pooled), `_transport_case`,
  `_spectrum_case`, `_worker_init`. Deps: `_backend`, `transport`, `geometry`,
  `spectrum`.
- Deps: `crystallography`, `DATA_DIR`.

## Sweep, config & drivers

### `sweep.py`
Turns a `Sweep` definition into the Cartesian product of `run_case` dicts.
- Public: `Sweep` (dataclass of all knobs), `Layer` (one stack layer: material,
  thickness, orientation), `build_cases`, `crystal_params`,
  `substrate_composition`, `stack_layers`, `film_on_substrate_layers`,
  `layer_radiator`, `substrate_radiator`, `geometry_table`,
  `fmt_thickness`, `pm` (±hkl expansion); the `MATERIAL_LABELS` registry.
- Deps: `crystallography`.

### `config.py`
Per-material grids and the default settings/sweep builders shared by the CLI and
both notebooks.
- Public: `default_settings`, `material_grid`, `material_sweep`,
  `trajectory_sweep`, `MaterialGrid` (TypedDict); the `_MATERIAL_GRIDS` /
  `MATERIALS` registries.
- Deps: `results` (`Settings`), `sweep` (`Sweep`).

### `run.py`
Checkpointed, resumable sweep driver and checkpoint loaders/repair.
- Public: `run_sweep`, `load_checkpoint`, `checkpoint_path_for`,
  `cases_from_results`, `repair_brem_wide`, `repair_checkpoint`.
- Deps: `montecarlo` (`run_cases`), `results` (`store_result`).

### `scan.py`
Headless sweep entry: parse args → build cases → `run_sweep` → checkpoint.
- Public: `main`, `run`, `add_subparser`.
- Deps: `config`, `run`, `sweep`.

## Results & plotting

### `results/` (package)
Result records, derived line metrics, and ranking/selection. Split from a single
module into submodules; **every public and internal name is re-exported from the
package**, so `from cxr_mc.results import X` is unchanged
(`tests/test_results_exports.py` freezes the export set).
- `store` — the `{config_name: {E0_keV: record}}` store: `Settings` (dataclass),
  `store_result`, `detected_background`, `PER_NA`.
- `selection` — subsetting/reducing a store: `records`, `records_for_cases`,
  `filter_results`, `select_results`, `sweep_values`, `slim_results`,
  `best_azimuth`.
- `metrics` — per-record scalars for the heatmaps: `line_index`, `line_quality`,
  `line_metrics`.
- `scoring` — geometry ranking: `SELECTION_MODES`, `selection_score`,
  `top_geometries`, `show_top`.
- `tables` — DataFrame views: `results_dataframe`, `summary_table`,
  `show_summary`.
- Deps: `montecarlo`, `sweep`.

### `plots/` (package)
All plotting — Matplotlib/Plotly. Split from a single module into submodules by
figure type; **every public and internal name is re-exported from the package**,
so `from cxr_mc.plots import X` is unchanged (`tests/test_plots_exports.py`
freezes the export set). Submodule DAG (leaf → driver):
`_style → _common → _frames → sweeps → {spectra, detectors, trajectories} → interactive`.
- `_style` — `COLORS`, `_ENERGY_PALETTE`, `energy_color` (per-energy colour map
  consistent across every figure). Leaf; no sibling deps.
- `_common` — shared figure plumbing: `_line_brem` (per-record line/brem split),
  `_per_tilt_figs` (one-figure-per-tilt loop), `_mode`, `_EFF_CACHE`. Deps:
  `montecarlo`, `results`.
- `_frames` — renderer-neutral tidy-data builders (`heatmap_frame`, `metric_vs_frame`,
  `scan_mode`, `pick_hue`, `_effective_x`/`_ndistinct`, the axis/value-label registries
  `_AXIS_SPECS`/`_axis_disp`/`_value_label`/`_FLUX_GATED`): the per-cell/per-point
  best-record reduction shared by matplotlib `sweeps.py` and `altair_sweeps.py`. Leaf-most
  of the sweep-figure modules; no matplotlib/Altair imports. Deps: `_common`, `results`,
  `pandas`.
- `spectra` — `plot_by_energy`, `plot_full_spectrum`, `plot_peak_vs_tilt`,
  `plot_mosaic_comparison`, `plot_best_spectra`, `plot_material_comparison`,
  `plot_tilt_panel`, the `_draw_*` spectral drawers. Deps: `_style`, `_common`,
  `montecarlo`, `results`.
- `sweeps` — `plot_heatmaps`, `facet_metric` (small-multiples over many knobs),
  `plot_metric_vs`, `plot_scan`; the `_HEATMAP_QUANTITIES` / `_METRIC_LABELS`
  tables + axis helpers. Renders from `_frames`' tidy DataFrames rather than
  re-deriving the best-record reduction inline. Deps: `_style`, `_frames`, `results`.
- `detectors` — `plot_timepix_efficiency` / `_detected` / `_poisson`,
  `plot_eaglexo_efficiency` / `_detected` / `_charge` / `_charge_map`. Deps:
  `_style`, `_common`, `sweeps`, `results`, `timepix_response`, `eaglexo_response`.
- `trajectories` — `plot_electron_trajectories`, `plot_trajectory_grid`,
  `plot_penetration_survival`. Deps: `_style`, `montecarlo`, `results`.
- `interactive` — `browse`, `browse_plotly`, `stream_chunk`, `plot_chunk`
  (the slider/streaming drivers that dispatch to the `spectra`/`detectors`
  drawers). Top of the DAG. Deps: `_style`, `_common`, `spectra`, `detectors`.
- `altair_*` — Altair/Vega-Lite renderers, the interactive counterparts of the
  matplotlib figures (the marimo `analysis_app.py` uses these first). They share
  the exact data prep with the matplotlib path (`_common._line_brem`, the
  `_frames` builders, the `detectors`/`trajectories` internals), so the physics
  is identical — only the renderer differs. Intentionally **NOT** re-exported
  from the package (frozen export guard) — import from the submodule. Per-module
  guard tests: `tests/test_altair_*.py`.
  - `altair_spectra` — intrinsic spectra: `spectrum_chart`, `spectrum_frame`,
    `compare_spectrum_chart` (overlay one line per E0/tilt/azimuth, for the
    Energy/Polar-angle/Azimuthal comparison notebook tabs). Deps: `_common`,
    `results`.
  - `altair_sweeps` — metric scans + parametric heatmaps: `metric_vs_chart`,
    `heatmap_chart`, `scan_charts` (auto heatmap-vs-lines, one shared metrics
    map across quantities). Deps: `_common`, `_frames`, `sweeps`, `results`.
  - `altair_detectors` — Timepix3/Eagle XO spectral views:
    `timepix_detected_chart`, `eaglexo_detected_chart`, `eaglexo_charge_chart`
    (+ their `*_frame` builders). Deps: `_common`, `altair_spectra`,
    `detectors`, `eaglexo_response`.
  - `altair_trajectories` — penetration views: `penetration_survival_chart`,
    `trajectory_chart` (+ `survival_frame`, `tracks_frame`,
    `track_segments_frame`); the dense datashader raster stays on matplotlib.
    Deps: `sweeps`, `trajectories`.
- Deps: `montecarlo`, `results`, `timepix_response`, `eaglexo_response`.

## Detector forward models

### `_si_sensor.py`
Internal shared silicon-sensor plumbing for both detector forward models: the
fixed Si material constants, the (grid → cached-response) keying pattern, the
Poisson acquisition core, and the `.apply()` input-shape guard. Leaf; no
sibling deps.

### `timepix_response.py`
Timepix3 charge-sensitive forward model (diffusion, absorption, energy
resolution, Poisson counts).
- Public: `TimepixResponse`, `build_response`, `get_response`,
  `absorption_efficiency`, `energy_fwhm_eV`, `sigma_diffusion_um`,
  `poisson_counts`.
- Deps: `crystallography`, `_si_sensor`.

### `eaglexo_response.py`
Eagle XO detector forward model (geometry/solid angle, QE table, energy
resolution, Poisson counts).
- Public: `EagleResponse`, `geometry`, `sweep_geometry`, `get_response`, `qe`,
  `qe_absorption_model`, `load_qe_table`, `solid_angle_sr`, `energy_fwhm_eV`,
  `poisson_counts`.
- Deps: `crystallography`, `_si_sensor`.

### `grating.py`
**Exploratory** grazing-incidence soft-X-ray grating spectrometer forward model
(dispersion geometry, coating reflectivity, and a simple CCD pixel grid; not
wired into the pipeline). See [`docs/grazing-grating.md`](grazing-grating.md).
- Public: `Grating`, `wavelength_angstrom`, `groove_spacing_angstrom`,
  `coating_number_density_per_ang3`, `detector_position_mm`, `disperse_spectrum`,
  `resolving_power`, `ALEXS_SENSORS`, `SimpleCCD`, `bin_to_pixels`.
- Deps: `crystallography` (`HC_EV_ANG`, `optical_constants`).

## CLI & packaging

### `cli.py`
The `cxr` console-script dispatcher.
- Public: `main`.
- Deps: `scan`, `export`, `slim`, `archive`, `__version__`.

### `export.py`
`cxr export` subcommand — `marimo export html` of `notebooks/analysis_app.py`
→ `results/<stem>.html` (replaces the retired nbconvert-PDF path).
- Public: `add_subparser`, `main`.

### `slim.py`
`cxr slim` subcommand — shrink a checkpoint pickle for transfer (drop the
full-range brem arrays, downcast spectra to float32, filter configs; `--grid`
keeps only the material's current-grid configs).
- Public: `slim_checkpoint`, `add_subparser`, `main`.
- Deps: `results` (`slim_results`, `_grid_names`).

### `archive.py`
`cxr archive`/`restore`/`archives`/`union` subcommands — the durable local
checkpoint shelf. Copy the active slot `checkpoints/<stem>.pkl` to/from the
long-term `checkpoints/archive/<label>.pkl` (atomic temp+replace, `--force`
overwrite guards, label↔stem date-stamp inference). `union` merges an archived
checkpoint into the active slot for the SAME material (compared via each
store's `case["crystal"]`), live winning on any overlapping (config name, E0)
point; archives the live checkpoint first by default (`--no-archive` to skip)
and leaves the source archive intact by default (`--delete-archive` to remove
it).
- Public: `archive_checkpoint`, `restore_checkpoint`, `list_archives`,
  `union_checkpoint`, `add_subparser`, `main`.

### `__init__.py`
Package root: exposes `DATA_DIR` (packaged-data resolver) and `__version__`.

### `_compile_nb.py`
Internal notebook-compile helper; not part of the public API.
</content>
