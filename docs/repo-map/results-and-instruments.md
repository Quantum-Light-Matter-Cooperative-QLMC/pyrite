# Repository map: Results, plotting & downstream models

Detail page of the [repository map](../repo_map.md). Start from the front page; module paths are relative to `src/pyrite/`.

## Results & plotting

### `results/` (package)
Result records, derived line metrics, ranking/selection. Split from single module into submodules; **every public and internal name re-exported from package** — `from pyrite.results import X` unchanged (`tests/results/test_exports.py` freeze export set).
- `store` — `{config_name: {E0_keV: record}}` store: `Settings` (dataclass), `store_result`, `detected_background`, `PER_NA`.
- `selection` — subset/reduce store: `records`, `records_for_cases`, `filter_results`, `select_results`, `sweep_values`, `slim_results`, `best_azimuth`.
- `metrics` — per-record scalars for heatmaps: `cxr_index`, `cxr_quality`, `cxr_metrics`.
- `scoring` — geometry ranking: `SELECTION_MODES`, `selection_score`, `top_geometries`, `show_top`.
- `tables` — DataFrame views: `results_dataframe`, `summary_table`, `show_summary`.
- `model` — in-memory `Result` with `CxrSpectra`/`SpectralDensity` families; polars views `to_dataframe` (long, native grids) and `spectral_flux` (wide union grid, beam/acceptance-scaled `FLUX_UNITS`, `total` column); re-exports the `spatial` names.
- `provenance` — `Provenance`: typed read-only `Mapping` of `Result.provenance` (inputs, identity digests, physics-model markers, run diagnostics, environment); unknown keys kept in `extra`.
- `spatial` — factorized planar-detector data: `SpatialResult`, `PixelRayMap`, `SpectralFactors`, `PixelMetadata`, `COMPONENTS`; the `"characteristic"` component preserves the separately auditable EEDL line contribution. Selected pixel spectra and images materialize in bounded chunks and are not checkpoint schema fields.
- Deps: `montecarlo`, `sweep`.

### `observations/` (package)
Persisted factorized pixel-detector observations, separate from intrinsic checkpoints at `<workspace>/observations/<stem>/`. `store` owns `ObservationStore` (mutable `index.json` source content key → observation digests; immutable `objects/true/<true_spatial_digest>.h5` factor objects; `objects/obs/<observation_digest>.json` layer-payload records; atomic temp+`os.replace` writes; schema/checksum/digest verification), `StoredObservation` (reopen, count, and `rescore` response/acquisition/normalization over reused true factors), and `observation_from_result`. `plan` owns `PixelSampling` (pixel rays, angular tiles, sample-frame directions, factor assembly from runner directional output) and `ObservationPlan` (one counting observation of one source case: pre-transport `find_reusable`, post-transport `assemble`, resolving native acquisition requests across all component axes before identity/storage); `api.simulate`, `runs.observe`, and sweep production all go through it. Never stores a dense pixel-energy cube. Deps: `detectors`, `instrument`, `montecarlo.geometry`, `results.model`, `console.config`.
- `inventory` — `observation_inventory(stem)` lists a stem's indexed observations as `ObservationEntry` rows (case from record provenance, else the checkpoint `cases.json`, else source key; response, exposure, count mode) reading only `index.json` and records; `ObservationInventory.load` opens one. `ObservationStore.record` is the public small-record reader.

### `plots/` (package)
Renderer-neutral data preparation plus backend packages under `mpl/`, `altair/`, and `plotly/`. **Every public and internal package export remains re-exported from `plots`** — `from pyrite.plots import X` unchanged (`tests/plots/test_exports.py` freeze export set). Former flat submodule paths were removed in 0.5.0; use the backend-qualified paths. Matplotlib submodule DAG (leaf → driver): `_style → _common → _frames → mpl.sweeps → {mpl.spectra, mpl.detectors, mpl.trajectories} → mpl.interactive`.
- `_style` — `COLORS`, `_ENERGY_PALETTE`, `energy_color` (per-energy colour map consistent across every figure). Leaf; no sibling deps.
- `_common` — shared figure plumbing: `_cxr_brem` (per-record line/brem split scored through `Detector.score`), `_per_tilt_figs` (one-figure-per-tilt loop), `_mode`, `_EFF_CACHE`. Deps: `detectors`, `montecarlo`, `results`.
- `_frames` — renderer-neutral tidy-data builders (`heatmap_frame`, `metric_vs_frame`, `scan_mode`, `pick_hue`, `_effective_x`/`_ndistinct`, axis/value-label registries `_AXIS_SPECS`/`_axis_disp`/`_value_label`/`_FLUX_GATED`): per-cell/per-point best-record reduction shared by `mpl/sweeps.py` and `altair/sweeps.py`. Leaf-most of sweep-figure modules; no matplotlib/Altair imports. Deps: `_common`, `results`, `pandas`.
- `_pixel_frames` — pixel-observation frames: `observation_image` (total/window counts, continuum-node transmission, filter coverage) in bounded chunks, `counting_observation` (expected vs Poisson realization, never substituted), `detector_image_frame` (block-reduced display grid <= 128 cells per axis with exact pixel ranges), `pixel_spectrum_frame`, `histogram_frame`, `pixel_metadata_rows`. No plotting imports. Deps: `observations`, `results`, `pandas`.
- `mpl/spectra` — `plot_by_energy`, `plot_full_spectrum`, `plot_peak_vs_tilt`, `plot_mosaic_comparison`, `plot_best_spectra`, `plot_material_comparison`, `plot_tilt_panel`, `_draw_*` spectral drawers. Deps: `_style`, `_common`, `detectors`, `montecarlo`, `results`.
- `mpl/sweeps` — `plot_heatmaps`, `facet_metric` (small-multiples over many knobs), `plot_metric_vs`, `plot_scan`; `_HEATMAP_QUANTITIES` / `_METRIC_LABELS` tables + axis helpers. Render from `_frames` tidy DataFrames, not re-derive best-record reduction inline. Deps: `_style`, `_frames`, `results`.
- `mpl/detectors` — `plot_timepix_efficiency` / `_detected` / `_poisson`, `plot_eaglexo_efficiency` / `_detected` / `_charge` / `_charge_map`. Deps: `_style`, `_common`, `sweeps`, `results`, `detectors` response adapters.
- `mpl/trajectories` — `plot_electron_trajectories`, `plot_trajectory_grid`, `plot_penetration_survival`, and transport initial-state arrays used by trace beam diagnostics. Deps: `_style`, `montecarlo`, `results`.
- `mpl/interactive` — `browse`, `browse_plotly`, `stream_chunk`, `plot_chunk` (slider/streaming drivers dispatching to `spectra`/`detectors` drawers). Top of DAG. Deps: `_style`, `_common`, `spectra`, `detectors`.
- `altair/` — Altair/Vega-Lite renderers, interactive counterparts of matplotlib figures (marimo `analysis_app.py` use these first). Share exact data prep with matplotlib path (`_common._cxr_brem`, `_frames` builders, `detectors`/`trajectories` internals) — physics identical, only renderer differ. Intentionally **NOT** re-exported from package (frozen export guard) — import from submodule. Per-module guard tests: `tests/notebooks/altair/test_*.py`.
  - `spectra` — response-free source spectra: `spectrum_chart`, `spectrum_frame`, `compare_spectrum_chart` (overlay one line per E0/tilt/azimuth, for Energy/Polar-angle/Azimuthal comparison notebook tabs). Deps: `_common`, `results`.
  - `sweeps` — metric scans + parametric heatmaps: `metric_vs_chart`, `heatmap_chart`, `scan_charts` (auto heatmap-vs-lines, one shared metrics map across quantities). Deps: `_common`, `_frames`, `sweeps`, `results`.
  - `detectors` — Timepix3/Eagle XO spectral views: `timepix_detected_chart`, `eaglexo_detected_chart`, `eaglexo_charge_chart` (+ their `*_frame` builders). Deps: `_common`, `altair.spectra`, `detectors` response adapters.
  - `trajectories` — penetration views: `penetration_survival_chart`, `trajectory_chart` (+ `survival_frame`, `tracks_frame`, `track_segments_frame`); dense datashader raster stay on matplotlib. Deps: `mpl.sweeps`, `mpl.trajectories`.
  - `pixels` — `detector_image_chart`, `pixel_spectrum_chart`, `histogram_chart` over `_pixel_frames` frames. Deps: `altair._typing`.
- `plotly/` — Plotly-specific interactive and rendered trajectory views. `trajectories` builds 3D beam/crystal/track figures from shared `mpl.trajectories` geometry; `crystal_lattice` renders unit-cell figures; `render` owns cached animation export over the Plotly trajectory builder; `camera` turns azimuth/elevation/zoom or a named preset into a `scene.camera` dict, so a viewer's chosen view reaches snapshots and renders (marimo drops `scene.camera` relayout events, so a dragged one cannot).
- Deps: `detectors`, `montecarlo`, `results`.

## Downstream instrument and detector forward models

### `instrument/`
Closed, analytic post-emission photon geometry. `model.py` owns frozen `FilterPlate`, physical `PlanarDetector`, and the narrow `PixelScorer` request, re-exporting `PlanarPose` and `PixelGrid` from `_planar_geometry`; `geometry.py` owns `planar_detector_rays` (delegating to `_planar_geometry.planar_rays`), solid-angle weights, coarse angular partitions, and exact finite-box path lengths; `attenuation.py` owns stacked per-filter attenuation coefficients and primary-photon transmission. `Scene` owns ordered plates and accepts either the legacy scalar `Detector` or physical `PlanarDetector`; instrument objects never enter the electron-transport navigator. Deps: `_planar_geometry`, `detectors.spec`, `materials`.

### `detectors/`
Detector configuration, read-time scoring, and detector-adjacent forward models. `acceptance.py` owns frozen `AngularAcceptance` (any two of full polar/azimuthal spans and solid angle) and source-facing `SensorAcceptance` (distance and size); `Detector` resolves both onto the unchanged scalar case acceptance fields. `spec.py` owns `Detector`, re-exports the source-grid `EnergyBins` compatibility type from `_energy_bins.py`, and owns the `Timepix3`/`EagleXO`/`LegacyEDS` response adapters, and the deprecated `DetectorSpec` spelling. Stored arrays remain response-free; plots and result tables call `Detector.score`. Deps: `materials.crystal`, `montecarlo` (legacy analytic response only), `DATA_DIR`.

#### `_si_sensor.py`
Internal shared silicon-sensor plumbing for both detector forward models: fixed Si material constants, (grid → cached-response) keying pattern, Poisson acquisition core, `.apply()` input-shape guard. Leaf; no sibling deps.

#### `timepix_response.py`
Timepix3 charge-sensitive forward model (diffusion, absorption, energy resolution, Poisson counts).
- Public: `TimepixResponse`, `build_response`, `get_response`, `absorption_efficiency`, `energy_fwhm_eV`, `sigma_diffusion_um`, `poisson_counts`.
- Deps: `materials.crystal`, `_si_sensor`.

#### `eaglexo_response.py`
Eagle XO detector forward model (geometry/solid angle, QE table, energy resolution, Poisson counts).
- Public: `EagleResponse`, `geometry`, `sweep_geometry`, `get_response`, `qe`, `qe_absorption_model`, `load_qe_table`, `solid_angle_sr`, `energy_fwhm_eV`, `poisson_counts`.
- Deps: `materials.crystal`, `_si_sensor`.

#### `grating.py`
**Exploratory** grazing-incidence soft-X-ray grating spectrometer forward model (dispersion geometry, coating reflectivity, simple CCD pixel grid; not wired into pipeline). See [grazing-grating model](../research/instrumentation/grazing-grating.md).
- Public: `Grating`, `wavelength_angstrom`, `groove_spacing_angstrom`, `coating_number_density_per_ang3`, `detector_position_mm`, `disperse_spectrum`, `resolving_power`, `ALEXS_SENSORS`, `SimpleCCD`, `bin_to_pixels`.
- Deps: `materials.crystal` (`HC_EV_ANG`, `optical_constants`).
