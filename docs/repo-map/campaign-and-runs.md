# Repository map: Campaign, checkpoints, run drivers & optimization

Detail page of the [repository map](../repo_map.md). Start from the front page; module paths are relative to `src/pyrite/`.

## Campaign, checkpoints & run drivers

`campaign/` owns resolved campaign inputs; `checkpoints/` owns persistence, repair, archive, and lock lifecycles; `runs/` owns executable local drivers.

### `bodies.py`, `campaign/target.py`, `instrument/filter.py`

`bodies.py` owns frozen scalar `SubstanceRef` and `RectangularBody` descriptions,
with explicit substance namespaces and dimensions in mm. `campaign.target.Target`
lowers a crystalline rectangle to `Slab` with the catalog orientation or an
explicit `lattice_cut=CrystalCut(...)`, and an explicit footprint.
`campaign.geometry.CrystalCut` (flat `Slab.cut`) orients the lattice relative to
the entrance face (axis, roll, continuous miscut); `resolve_crystal_cut` is the
single resolver shared by `build_cases` and `campaign.front_matching`. `instrument.filter.Filter` lowers a placed rectangle
to `FilterPlate` for primary attenuation only. Namespace collisions/external
filter catalogs lower through inline `MediumSpec` when legacy strings cannot
retain their meaning. Legacy target/filter constructors remain unchanged.
`materials/_composition.py` owns homogeneous lookup with an explicit catalog;
legacy wrappers retain their case, namespace and validation rules.

### `campaign/model.py`

`Scene` retains body role wrappers for dotted sweep paths and exposes
`target_spec`/`filter_plates` as the lowered legacy geometry. Source lowering,
observation planning and trajectory consumers use those properties.

Supported scalar scene API: `Scene`, path-addressed `Sweep`, `Physics`, `Numerics`, nested `CxrReflections`, and presentation-only `Analysis`. `Scene` rejects implicit multi-value fields; `Sweep` applies ordered dotted/indexed paths such as `target.layers[1].thickness_ang`, expands Cartesian products, and lowers them through `api.build_sweep_cases`. This `Sweep` is the preferred public object; `campaign.sweep.Sweep` below remains the profile/CLI-compatible campaign implementation.

- Public: root-lazy `Scene`, `Sweep`, `Physics`, `Numerics`, `CxrReflections`, `Precision`, `Analysis`.
- Deps: `campaign.geometry`, `campaign.sweep.BeamSpec`, `detectors`.

`_numerics.py` owns frozen `Physics`, `Numerics`, and `CxrReflections`, plus
unchanged profile-schema validation. `_energy_bins.py` is the shared leaf for
sampled true-photon grids; `detectors.spec.EnergyBins` remains its public and
pickle-compatible re-export. `campaign/resolution.py` owns Scene/component
composition, run overrides, quantum adaptation and model/integration validation.
Its internal `ResolvedSimulation` separates requested legacy scene settings
from effective physics/numerics; `api.simulate`, observation planning and case
lowering consume it. Configured sweeps share its model validation while retaining
their historical expansion. `campaign/lowering.py` projects
`Numerics.energy_bins` onto the legacy campaign adapter, whose `energy_bins`
field feeds the existing case builder. Detectors own acceptance and response.

`_executor_contract.py` owns import-light executor admission/provenance shared by
composition and dispatch; `execution._simulation` preserves the legacy aliases.

`observations/spec.py::Observation` owns independent detector references,
response, acquisition and pixel sampling requests. Its binder validates physical
geometry and legacy conflicts before case preparation, normalizing into the
existing execution/identity representations without mutating the input scene.

### `_physics_contract.py`, `_physics_provenance.py`, and `campaign/physics_view.py`

Private in-tree extension metadata ([ADR-0017](../adr/0017-in-tree-physics-extension-contract.md)).
`_physics_contract.py` is a standard-library-only leaf: immutable descriptor
catalog, applicability/backend evidence, and canonical JSON/pickle-safe resolved
view. `_physics_provenance.resolved_physics_view` describes an already resolved
case through existing facade policy, retains historical mapping omissions, and
snapshots its parameters and existing model markers. Callers supply the per-case
inventory from `runner.case_tables.case_table_markers`, never a dataset union.
Construction does not load tables, resolve `auto`, probe hardware, dispatch
adapters, or feed a new payload into identity. `campaign.physics_view` re-exports
the helper; runtime consumers stay below campaign. API provenance and new
trajectory captures record the descriptive view, and replay checks model/data
compatibility before scoring or batch writes. Runtime preparation and
physical-domain verification remain separate checks.

`runner/case_tables.py` uses the elastic family IDs to select fixed immutable
Mott/ELSEPA preparation adapters through the existing kwargs/table-entry seams.
Native tables, model codes and exact/LUT dispatch remain in transport owners;
see [preparation boundaries](../repo-design/compute/physics-preparation-adapters.md).

`runner/emission_adapters.py` owns borrowed runtime batch contexts, existing-slot
outputs and immutable CXR/characteristic/continuum registration. Live scoring
uses the same staged segments/populations and façade scorer seams; continuum
wrappers delegate to separate EEDL, BremsLib and coupled producers while native
layer reductions remain in `runner/emission.py`. Replay, recompute, directional
and adaptive consumers use the same selections; see
[batch scoring](../repo-design/compute/emission-batch-adapters.md) and the
[extension guide](../repo-design/compute/extending-physics.md).

### `campaign/beam.py`

Public `Beam`: one sub-object per kind of property (`Spot`/`Point`/`TransverseDistribution`, `PointBunch`/`BunchLength`/`Offsets`/`LongitudinalDistribution`, optional `GdfSource` via `Beam.from_gdf`). `Beam.to_spec()` lowers to the flat `campaign.sweep.BeamSpec`, the internal record behind case payloads and profile files; `Beam.from_spec` is the inverse. `Scene` accepts either and stores a `Beam`.

- Public: root-lazy `Beam`, `Spot`, `Point`, `PointBunch`, `BunchLength`, `Offsets`, `GdfSource`.
- Deps: `campaign.sweep.BeamSpec`, `campaign.longitudinal`, `montecarlo.transverse`.

### `campaign/single_scene.py` and `cli/commands/_simulation.py`

The CLI singleton-profile resolver lives in `campaign/single_scene.py`, which lowers validated catalog fields without CLI dependencies or checkpoint I/O. `cli/commands/_simulation.py` owns catalog-boundary validation, simulation dispatch, JSON/presentation, and `.npz` serialization for `run --ephemeral`, which bypasses the checkpoint sweep driver.

### `api.py`

Filesystem-free simulation seam. `simulate` combines scalar beam, target, and detector components into a `Scene` and lowers exactly one typed `Case`. Scalar detectors call the unchanged `montecarlo.run_case`; physical detectors reuse one transport across coarse angular tiles, then attach exact pixel rays and finite-filter attenuation as a factorized `SpatialResult`. Source-case and observation identities remain separate. Combined line totals include atomic characteristic radiation, with the component exposed as `Result.characteristic` and in spatial factors. The API never reads or writes campaign checkpoints. `build_case`, `build_sweep_cases`, and legacy lowering helpers are implementation/compatibility seams rather than the root supported surface.

- Public: root-lazy `simulate`, whose experimental `progress=` routes `SimulationProgress` (`_simulation_progress.py`) from per-batch transport and per-stage spectrum sinks (`runner._TRANSPORT_PROGRESS`/`_SPECTRUM_PROGRESS`, installed by `runner.transport_options.run_progress`) for in-process runs.
- Deps: `campaign.model`, `campaign.sweep`, `campaign.profiles`, `detectors`, `montecarlo`, `results`.

### `phases.py`

Experimental scalar-detector phase split (#776): `simulate_transport` runs `runner._transport_case` (segments kept on the device for CUDA) into a `TransportedScene`; `score_spectra` runs only `_spectrum_with_recovery` on it, any number of times, with optional spectral chunk changes, and builds the same `Result` as `simulate` through `api._scalar_result`. Disk-backed replay stays `pyrite trajectories score`.

### `campaign/sweep.py`
Turn `Sweep` definition into Cartesian product of `run_case` dicts.
- Public: `BeamSpec` (incident phase space and pulse properties), `Sweep` (dataclass of simulation knobs), `beam_replace`, `LayerSpec` (one stack layer: material, thickness, orientation), `build_cases`, `geometry_table`, `pm` (±hkl expansion); `MATERIAL_LABELS` registry. Case names are built from the stable catalog key, never the display label: they key persisted checkpoint records and feed `checkpoints/checkpoint_cleanup._case_key`, so relabelling a material must not rename stored cases. Re-exports the `campaign/geometry.py` surface, so `pyrite.campaign.sweep` stays the stable import path for `crystal_params`, `fmt_thickness`, `stack_layers` and friends.
- `Sweep.target` is the canonical geometry spelling and the only geometry state on `Sweep`; `build_cases` reads geometry only through `target.lower()`. The flat constructor arguments (`thickness_ang`, `tilt_deg`, `groove_spacing_ang`, …) are `InitVar` aliases normalized onto `target` in `__post_init__`, mirroring the `detector` precedent — supplying both nested and flat is an error, and *reading* a retired flat name raises rather than returning stale state. `substrate=`/`substrate_thickness_ang=` are deprecated in favour of `Stack.on_substrate`.
- `Sweep.detector` owns acceptance and response. `Sweep.energy_bins` is the legacy campaign projection of `Numerics.energy_bins`, supplying fine line and wide bremsstrahlung source grids lowered into each case. Acquisition reporting edges are resolved separately over complete observation response outputs.
- `build_cases` resolves in fixed stages before the product loop: electron budget, energies and line/continuum grids, detector acceptance, orientation, beam keys (`campaign.beam_cases`), mosaic route, then target lowering; each case point only spreads those results, and "auto" models resolve on the assembled cases.
- Deps: `campaign.geometry`, `campaign.beam_cases`, `campaign.bremsstrahlung_cases`, `detectors` (`Detector`, `EnergyBins`), `materials` (`CATALOG`, `LayerSpec`), `materials.crystal`.

### `campaign/beam_cases.py`
Beam phase-space case keys for `build_cases`: `beam_case_keys` validates the bunch-front source and refuses declarative transverse/longitudinal policies mixed with legacy fields, then returns the flat beam keys that join a case only where they diverge from defaults (plus the GPT `gdf_source` record); `beam_distribution_keys` resolves the declarative policies at one geometry and beam energy.
- Deps: `materials` (`CATALOG`), `montecarlo.bunch_front`, `montecarlo.transverse`, `campaign.longitudinal`.

### `campaign/thick_target.py`
Thick-target screen run by `build_cases` after the kinematic screen: `warn_thick_target` emits `ThickTargetWarning` when a case above the pair threshold traverses more than `THICK_TARGET_X0_FRACTION` (0.05) radiation lengths, since photons are attenuated rather than transported (#341). `traversed_radiation_lengths` sums `t_i/X0_i` over transport layers at the case tilt. Changes no case key or identity (#439).
- Deps: `materials.radiation_length`.

### `campaign/bremsstrahlung_cases.py`
Per-case continuum source resolution for `build_cases`: `resolve_auto_bremsstrahlung` replaces `bremsstrahlung_model="auto"` with the source each case's elements resolve to, and `radiative_case_keys` validates the coupled radiative mode and returns its divergence-only case keys, which join only BremsLib-resolved cases (#172).
- Deps: `_numerics`, `montecarlo.case`, `xsgen.bremslib.tables` (lazy).

### Experimental planar overflow instrument

`montecarlo/channeling/overflow.py` contracts a supplied plane density with a
larger differential kernel into a nonwrapped outgoing momentum measure. It
reports unresolved loss and the changed in-band gain, and refuses sampling
until both close against the original solver overflow. The optional
`DensityMatrixSolver.propagate(jump_source_observer=...)` supplies the same
integrated source quadrature used for flux accounting.
`checks/channeling_overflow.py` measures full-depth fixed-energy marginals;
no production handoff selects this instrument. Claim:
`cr-planar-overflow-spectrum`.

### `campaign/channeling_cases.py`
Opt-in planar channeling selection for `lowering.build_case` (ADR-0018, #506): `resolve_channeling` attaches the resolved `Case.channeling` record (plane, temperature, declared dechanneling margin, zone numerics) only when `Physics.channeling_model="planar"` and applies the per-case gates: crystals without channeling thermal data and energies above 100 MeV are refused, energies above 30 MeV warn (#439), `n_b < 1` is refused, and a beam within a few Lindhard axial critical angles of a low-index axis warns. Off returns the cases unchanged.
Axial selection remains configuration-only (#624). The data-only `_channeling_admission` module implements production admission for raw records/selectors, re-exported through `montecarlo.case`; scan settings resolution, runner consumers, checkpoint repair planning and trajectory checkpoint planning apply it before work or mutation. Remote profile submissions use scan resolution before host probes; remote trajectory scoring preflights metadata before staging. Research details: [axial pre-integration](../research/physics/axial-preintegration-624.md).
- Deps: `montecarlo.case`; lazily `materials.channeling`, `materials.crystal`, `materials.atomic`, `montecarlo.geometry`, `montecarlo.channeling`.

### `campaign/geometry.py`

Internal `_geometry_values.py` owns scalar/sweep/index validation and copied
quantity read-back; `_geometry_table.py` owns numeric case summary tables;
`flat_geometry.py` owns the flat-vocabulary adapters (`target_from_flat`,
`target_flat_fields`, `target_replace`, `retired_flat_input`).
The original geometry imports (including those adapter names) and public class owners remain available.
The closed target variant set and its lowering to today's geometry case keys (`abs_layers`, `layer_radiators`, `crystal_width_mm`, `crystal_height_mm`, `groove_spacing_ang`, `mosaic`). Each variant owns its validity at construction, so an impossible target raises in the user's script rather than after profile resolution; arbitrary geometry is a recorded non-goal (ADR-0008), so there is no dispatch protocol or region algebra. `target_from_flat` / `target_flat_fields` / `target_replace` are the inverse pair between a `Target` and the flat geometry vocabulary that catalog data, profile overrides, and identity digests still speak. Sits below `campaign/sweep.py` and imports without it.
- Public: `Slab`, `Stack`, `Target` (the variant union), `Layer`, `Footprint`, `BlazedGrooves`, `LoweredTarget`, `Stack.on_substrate` (the sanctioned one-layer film-on-wafer stack), `DEFAULT_FOOTPRINT`, `DEFAULT_SUBSTRATE_THICKNESS_ANG`, `target_from_flat`, `target_flat_fields`, `target_replace`, `retired_flat_input`, `crystal_params`, `substrate_composition`, `stack_layers`, `film_on_substrate_layers`, `layer_radiator`, `substrate_radiator`, `fmt_thickness`, `ScalarOrSeq`.
- Deps: `detectors` (`Detector`, for the target×detector groove check), `materials` (`CATALOG`, `LayerSpec`), `materials.crystal`.

### `campaign/longitudinal.py`
Bunch-length policy on the input side of `BeamSpec`: named `long_shape` kinds resolved per case against the dominant basal reflection.
- Public: `LongitudinalDistribution`, `ResolvedLongitudinalDistribution`, `resolve_longitudinal_distribution`.
- Deps: `materials.crystal`, NumPy, SciPy constants.

### `campaign/beam_metrics.py`
Pure diagnostics over sampled initial phase-space arrays: per-plane RMS size, geometric and normalized emittance, Twiss parameters, longitudinal RMS size and emittance, bunch charge, Gaussian-equivalent peak current, and average current.
- Public: `PlaneMetrics`, `BeamMetrics`, `sampled_beam_metrics`, `initial_state_metrics`.
- Deps: NumPy, SciPy constants.

### `campaign/config.py`
Default settings/sweep builders shared by CLI and both notebooks; per-material scan grids project from immutable `materials.CATALOG`.
- Public: `default_settings`, `material_grid`, `material_sweep`, `trajectory_sweep`, `catalog_detector`; `MATERIALS` ordered tuple. `catalog_detector` applies the catalog's acceptance mapping onto `DEFAULT_CATALOG_DETECTOR` (response-free scalar geometry at 90 deg). The catalog speaks the flat geometry vocabulary, so both builders project their scan grid through `target_from_flat` and hand `Sweep` a built `target`. Catalog energy-grid artifacts resolve into the campaign source `EnergyBins` at this boundary.
- Deps: `materials` (`CATALOG`, `MaterialSpec`), `results` (`Settings`), `profiles` (`get_fidelity_preset`), `sweep` (`Sweep`), `geometry` (`target_from_flat`), `detectors` (`Detector`, `EnergyBins`).

### `campaign/profiles.py`
Named `full`/provisional `survey` fidelity presets, deterministic serialization of resolved settings and sweeps, SHA-256 dataset identity, and variant checkpoint stem selection. `FidelityPreset.emission` (`incoherent`/`coherent`/`both`, with a derived `coherent_emission`) is the preset-owned emission policy; `dataset_identity` adds an `emission` divergence key so the three modes get three distinct digests (incoherent unchanged, bit-for-bit). Detector response and campaign energy bins project back to the historical payload vocabulary, so read-time scoring does not perturb transport/checkpoint identity. The in-medium photon dispersion is unconditional (no setting, no divergence key); the payload instead hashes a constant `line_kinematics` generation marker, which orphaned every vacuum-era digest exactly once. The current SBETHE collision-stopping model has a constant `stopping_model` marker in both `dataset_identity` and `case_content_key`; the resolved SBETHE table digests also join identity, preventing reuse of results from the earlier Joy--Luo/Berger--Seltzer splice or a different table. The unconditional EEDL/xraydb characteristic model likewise hashes a constant `characteristic_model` marker into both identities, preventing reuse of pre-characteristic results. The resolved BremsLib or EEDL bremsstrahlung source has a `bremsstrahlung_model` marker in both identities; BremsLib table digests join the xsgen identity markers when that source is selected. The coupled radiative mode adds `radiative_model`/`radiative_cutoff_eV` to `transport_numerics` only when BremsLib resolves (#172). The PXR/CBS line route's escape model hashes a constant `line_escape_model` marker into both identities (segment-mean escape on the incoherent route since issue #181).
- `_identity_v1` assembles the hashed payload in sections: crystallography, flat target/detector/bin and beam projections, opt-in settings (`campaign.identity_options`), constant model-generation markers, the resolved bremsstrahlung source, and xsgen table digests.
- Public: `FidelityPreset`, `FIDELITY_NAMES`, `get_fidelity_preset`, `dataset_identity`, `case_content_key`, `variant_stem`.
- Deps: `results` (`Settings`), `sweep` (`Sweep`), `campaign.identity_options`, NumPy.

### `campaign/identity_options.py`
Run-option sections of the version-1 dataset identity: `pop_identity_options` strips run-affecting settings from the serialized settings, and `transport_numerics_payload` / `radiative_numerics_payload` return their divergence-only `transport_numerics` keys (auto inelastic resolution as the cases resolve it; coupled radiative keys only when BremsLib resolves on a flat face).
- Deps: `_numerics`, `campaign.inelastic_cases`.

### `campaign/numerics_resolution.py`
Immutable resolved numerics with value provenance and stable CLI/JSON grouping; re-exported by `campaign/profiles.py`, which owns fidelity resolution.
- Public: `NumericsResolution`, `NUMERICS_GROUPS`.
- Deps: `_numerics` (`CHANNELING_KEYS`), standard library.

### Profile authoring (`campaign/profile_*.py`)

Click-free operations over parsed catalog documents. Consumers import their
operation family directly. The `profile_edit.py` compatibility module was removed
in 0.17.0. CLI parsing,
prompts and persistence remain at their existing owners; `catalog_edit.CatalogDraft` provides service
validation, stale-edit checks and atomic saving.

| Owner | Responsibility |
|---|---|
| `_profile_document.py` | Shared document lookup, reference errors, range values and TOML copying |
| `profile_lifecycle.py` | Packaged creation templates, clone/set, ranges/emission, rename/delete |
| `profile_membership.py` | Explicit material membership and catalog ordering |
| `profile_numerics.py` | Numerical controls and coupled transport validation |
| `profile_precision.py` | Adaptive policy, fixed-count transitions and blockers |
| `profile_cxr_grid.py` | Line-grid selectors, conflicts and source inspection |
| `profile_filters.py` | Ordered filter queries and edits |
| `profile_detectors.py` | Detector storage, attachment, physical edits and explicit migration |
| `profile_summary.py` | Read-only summary assembly and named/inline detector display resolution |

`profile_detector_entries` selects collection/legacy/default IDs for editing,
summary and migration without changing storage. Pixel edits reuse
`detectors.editing.merge_pixel_detector`; summaries project geometry through
`observation.physical_detector_from_config`. Domain operations import no Click
command modules.

### `energy_grid/` (package)
Photon-energy-grid derivation, bounds analysis, and catalog application (user inspection nests under `pyrite material` and `pyrite profile`). The compact case-grid codec shared by campaign, checkpoints, and Monte Carlo lives below this driver package at `pyrite._energy_grid_encoding`; `encoding.py` is its public compatibility facade.
- The `pyrite energy-grid` Click group lives in `cli/commands/energy_grid.py` with every other CLI surface. This package is Click-free and imports nothing from `cli`. Monkeypatch seams for `remote`/`cli_json`/ `emit_json_result` live on the command module.
- The artifact store itself lives at `pyrite._energy_grid_artifacts` (below this package; see Core physics), because `materials.catalog` reads it too. `gc.py` resolves profile/campaign-lock reachability, out-of-band orphan ages, preview revalidation, and deletion.
- Also: `derive`, `bounds`, `apply`, `defaults`, `provenance`, `golden`, `job`.

### `checkpoints/campaign_lock.py`
Deterministic `cxr.campaign-lock.v1` writer/reader. Each successfully completed dataset records its resolved profile, material, dataset identity, explicit energy-grid artifact hash (or legacy marker), beside checkpoint metadata as `cxr.lock.json`; incomplete runs emit no lock. Active and archived locks remain artifact-GC roots.

### `runs/run.py`
Checkpointed, resumable sweep execution driver. Owns callback sequencing, trajectory preflight, deadlines and draining. `runs._sweep_plan` owns identity/metadata eligibility, per-case CAS seeding/replay and observation work selection; `runs._sweep_execution` owns progress/cost counters and config/group completion. Persistence, manifests, component loading, and repairs live in `checkpoints`. Compatibility re-exports preserve the former `pyrite.runs.run` helper imports.
- Public: `run_sweep`, `load_checkpoint`, `checkpoint_path_for`, `cases_from_results`, `repair_brem_wide`, `repair_checkpoint`.
- Deps: `runs._sweep_plan`, `runs._sweep_execution`, `checkpoints.persistence`, `checkpoints.recompute`, `montecarlo` (`run_cases`), `results`.

### `checkpoints/persistence.py`
Checkpoint persistence owner: paths, load/save, component loaders, thin `cases.json` manifests, legacy seeding, and the default checkpoint directory. `_SweepCheckpoint` owns per-material shard publication, incremental manifests and consolidation; it retains shards on pauses or publication failure and clears them only after component and manifest saves succeed. Keeps storage layout and serialization policy behind the checkpoint package.
- Public compatibility surface is re-exported from `runs.run` where required.
- Deps: `checkpoints._checkpoint_store`, `checkpoints._checkpoint_io`, `paths`, `results`.

### `checkpoints/_checkpoint_store.py`
Component storage adapter: active datasets live under `checkpoints/<stem>/{line,brem,characteristic}.h5`, merge transparently into historical in-memory result records, treat the characteristic and channeling companions as optional (`channeling.h5` is written only when a record carries `spec_channeling`), and migrate legacy `checkpoints/<stem>.pkl` stores on next save. Shared case blobs live at `checkpoints/<material>/<first2hex>/<content-key>.pkl` and use atomic writes. The `.pkl` suffix remains a layout token; current bytes use the versioned HDF5 schema.
- Internal: `discover`, `load`, `save`, `signature`, `cas_contains`, `cas_load`, `cas_save`, component/path helpers.
- Deps: `_checkpoint_io`, NumPy.

### `checkpoints/_checkpoint_io.py`
Result leaf codec. New writes use uncompressed `pyrite.result` HDF5 schema version 2 with content-addressed arrays. Reads sniff bytes and permanently dispatch across current and schema-version-1 HDF5 plus legacy zstd, gzip, and plain pickle generations. `dump_stream` stages seekable HDF5 output before zstd-framing it onto remote-pull pipes.
- Internal: `dump`, `dump_stream`, `load`; schema/version and compatibility constants.
- Deps: h5py, NumPy, stdlib gzip/pickle compatibility readers; stdlib `compression.zstd` on Python 3.14+, `backports.zstd` on Python 3.13.

### `runs/scan.py`
Headless sweep driver: build cases → `run_sweep` → checkpoint; owns `--no-cache`/`--recompute` and performance-run cache defaults, catalog-profile resolution, and nsys re-exec. Click wiring is in `cli/commands/scan.py`; `scan.command` stays available via a lazy `__getattr__` so the Monte Carlo hot path never imports Click.
- Public: `main`, `run`, `command` (lazy), `validate_materials`, `validate_catalog_profile`, `resolve_profile_materials`.
- Compatibility: `_ProgressTimer` and `_write_progress_record` re-export the neutral `pyrite._progress` implementations.
- Deps: `_progress`, `config`, `run`, `sweep`.
- `runs/_trajectory_capture.py` builds the opt-in `--trajectories` capture (root, overwrite, profile scene, provenance) for `scan.py`.
- `runs/_material_progress.py` holds `_run_material`'s `run_sweep` callback state (`MaterialProgress`: progress file, dashboard job record, performance-logger context) and starts the `--performance-profile` logger.

### `runs/observe.py`
Library producer for one persisted pixel-detector observation: `produce_observation(scene, numerics, store)` reuses a stored observation without transport when `api.observation_plan(...).find_reusable` matches its true-spatial identity, rescoring only response/acquisition/normalization; otherwise it runs `api.simulate` and stores the result.
- Public: `produce_observation`, `ObservationRun`.
- Deps: `api`, `observations`.

### `runs/blaze.py`
Headless blazed-crystal (sawtooth entrance face) sweep entry: `pyrite material blaze <material> --energy E [E...] --spacing S [S...] [--polar A [A...]]`. Mirrors `scan.py`'s parse args → build cases → `run_sweep` → checkpoint structure, but forces v1 groove geometry (`theta_obs=90`, `tilt_azim=180`, no substrate/stack/footprint) per (energy, spacing) pair and writes to a dedicated `checkpoints/<material>_blazed/`, never the flat-face `<material>/`. Click wiring is in `cli/commands/blaze.py`; `blaze.command` stays available via a lazy `__getattr__`.
- Public: `main`, `run`, `command` (lazy).
- Deps: `config`, `run`, `scan` (`validate_materials`, `_write_progress_record`), `sweep`.

## Optimization studies

`optimization/` owns physical parameter spaces, immutable Scene replacement,
study/trial identity and optional Optuna ask/tell persistence. `study.py` routes
each trial through Sweep/run_sweep with distinct seeds and the standard case
content key. `scoring.py::FixedWindowYield` selects one node-sampled incoherent
PXR/CBS record and delegates exact fixed-band integration to `results.radiation`;
it refuses bin means and joint-field corrections. `checks/pxr_optimization.py`
is the bounded TPE/random/grid demonstration and independent-finalist protocol.
Tests: `tests/optimization/`. Guide: `docs/guides/optimization-studies.md`.

`kernel_tuning.py::tune_kernels` (#776) tunes CUDA execution controls, not
physics: transport (`electron_batch`, `schedule`, `continuation_rows`) and
spectrum (`cxr_chunk`, `brem_chunk`) dimensions are independent Optuna studies
(exhaustive for small lattices), measured by one persistent
`_kernel_worker.py` process per tuning (JSON lines over stdin/stdout).
Transport candidates time `phases.simulate_transport` and compare a digest of
the transported arrays; spectrum candidates rescore one kept transport with
`phases.score_spectra`. The fastest candidate must survive interleaved
re-timing against the baseline (`min_speedup`). Tests:
`tests/optimization/test_kernel_tuning.py` (fake worker),
`tests/montecarlo/transport/test_transport_kernel_tuning_cuda.py` (device). Guide:
`docs/guides/batch-chunk-tuning.md`; tutorial 15.
