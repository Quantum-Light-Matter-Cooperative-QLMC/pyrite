# Repository map: Core physics

Detail page of the [repository map](../repo_map.md). Start from the front page; module paths are relative to `src/pyrite/`.

## `_progress.py`
Leaf progress-record codec and additive active-work timer shared by run and checkpoint drivers. Owns atomic compact-JSON replacement without importing a driver package.
- Internal: `_ProgressTimer`, `_write_progress_record`.
- Deps: none (stdlib only).

## `_app_defaults.py`
Persisted "which material does the app start on" state for the analysis and viewer marimo apps: one one-line file per app under `paths.state_dir()`. Sits below both sides that read it — the launchers in `cli/commands/` and the notebook helpers in `apps/` — so neither package has to import the other.
- Public: `get_analysis_default`, `set_analysis_default`, `get_viewer_default`, `set_viewer_default`; module-global `ANALYSIS_DEFAULT_FILE`, `VIEWER_DEFAULT_FILE` are the test seam.
- Deps: `paths`.

## `_acp.py`
Optional local ACP bridge lifecycle (Claude and Codex stdio-to-WebSocket adapters) for marimo developer sessions. Lives at the package root, not under `apps/`, because nothing in `apps/` uses it: the callers are the app launchers in `cli/commands/` and `pyrite-dev acp-up`/`acp-down`.
- Public: `running_acp`, `start_acp_servers`, `stop_acp_servers`, `ACP_SERVERS`.
- Deps: `paths`.

## `_precision.py`
Import-light `Precision` policy for adaptive sampling: validates the requested budget, observables, guards and bands; serializes the policy separately from realized statistics. Shared by `Numerics`, `Settings`, `Case`, and the adaptive runner, without importing a backend.

`montecarlo/runner/independent.py` owns opt-in brem-prefix stopping: separate scalar monitors, a frozen brem population, and live CXR rechecks. `Precision.brem_target_rse` selects it; `block_transport.py` redecides the prefix during realized-range replay, and the runner applies line-only cutoffs after the brem stop.

## `_backend.py`
Portable array backend shared by every device-dispatching kernel: NumPy, CUDA/ROCm CuPy, and Intel dpnp/SYCL adapters behind a deterministic `PYRITE_MC_BACKEND` selection. Exports `xp`, `cp`, `REAL`, `_to_cpu`, `_GPU`, `is_device_array`. Lives at the package root, not under `montecarlo/`, because `materials.attenuation` dispatches on it too — that is what keeps `materials` out of the `montecarlo` import cycle. Importing it runs the accelerator probe, so cheap-import callers (CLI, catalog parsing) reach for it inside the function that needs a device. Deps: `_env`.

## `_energy_grid_artifacts.py`
Immutable, content-addressed energy-grid artifact store: canonical `cxr.energy-grid-artifact.v1` identity bytes, SHA-256 naming, sharded storage, atomic dedup publication, and corruption checks. Imports nothing first-party. Read by `materials.catalog` when resolving `[profiles.*.energy_grid_refs]` at catalog-load time and written by `energy_grid.apply` / collected by `energy_grid.gc`, so it sits below both rather than inside `energy_grid/`. Deps: none (leaf; stdlib only).

## `_bremslib_table.py`
BremsLib table staging: `BremsLibBremsstrahlungTable`, `prepare_bremslib_table` (exact piecewise-linear solid-angle renormalization, geometric `T1` refinement, 30--100 MeV extension rows filled by `_bremslib_extension`), `solid_angle_integral`, `top_reduced_energy`, and `BREMSSTRAHLUNG_BREMSLIB_MODEL`. A leaf below `montecarlo` and `xsgen` so `xsgen.bremslib.tables` stages tables without importing Monte Carlo; `montecarlo.spectrum.brem_bremslib` re-exports every public name. Ledger `bremslib-angular-model`. Deps: `_bremslib_extension`; otherwise NumPy only.

## `_bremslib_extension.py`
Fills BremsLib's 30--100 MeV extension rows at staging (#439): `extend_above_complete` holds the SDCS tip ratio of the last complete row and builds `x > 0` angular shapes from the library `k = 0` shape times the Schiff (Koch–Motz 2BS) ratio and the complete-row residual; owns `schiff_density`, which `validation.brem_angular` reuses. Ledger `bremslib-high-energy-extension`. Deps: none (leaf; NumPy only).

## `_planar_geometry.py`
Planar pose and point-source ray geometry: `PlanarPose`, `PixelGrid`, `PixelRays`, `solid_angle_sr`, and `planar_rays(pose, pixels=, size_mm=)`. A leaf below `montecarlo`, `detectors`, and `instrument`: `montecarlo.geometry` tiles a detector face and `detectors.eaglexo_response` takes its solid angle without importing the instrument model. `instrument` and `instrument.geometry` re-export the public names; `instrument.geometry.planar_detector_rays` delegates to `planar_rays`. Ledger `positioned-filter-attenuation`. Deps: none (leaf; NumPy only).

## `_optical_coherency.py`
`OpticalSpectralCoherency`: Hermitian PSD 2x2 photon-density Jones matrix per energy node, photons/(eV pulse) (#617). A leaf below `montecarlo` and `detectors`: `montecarlo.spectrum.optical_interferometer_handoff` builds it from finite PEC screen fields and `detectors.optical_interferometer` (which re-exports it) and `detectors.optical_elements` (measured passive Jones elements) consume it. Ledger `optical-instrument-observable`. Deps: none (leaf; NumPy only).

## `datasets.py`
Hash-pinned upstream datasets (ADR-0014 class b): the EPICS2025 EEDL and EADL files and the PyRITE-derived EPDL photon table. Pins (ordered URLs: upstream then PyRITE's release mirror; SHA-256, and EEDL's declared final-CRLF transformation of the published bytes), install path `<data root>/datasets/<name>/<file>` (data root from `console.config.fetched_data_root`, shared with xsgen tables), `require_dataset`/`verify_dataset`, `fetch_dataset`, and the SBETHE `PDATCONF_SHA256` pin read by `montecarlo.shell_configuration` and `xsgen.verify` behind `pyrite tables fetch eedl|eadl|epdl`. Read by `montecarlo.eedl_ionization` and `montecarlo.eadl_relaxation`, verified by `xsgen.verify`, shipped by `remote.transport` sync. Deps: `console.config`, `xsgen` (fetch only, lazy).

## `materials/` (package)
Material domain package. Narrow top-level API expose immutable `CATALOG`, frozen record types (`MaterialCatalog`, `CrystalInfo`, `CrystalSpec`, `MediumSpec`, `MaterialIdentity`, `MaterialSpec`, `MaterialValidationSpec`, `ScanSpec`, `LayerSpec`), `load_material_catalog`, compatibility projections (`CRYSTALS`, `MATERIALS`, `MATERIAL_LABELS`). Implementation helpers stay in submodules below.

## `materials/catalog.py`
Thin schema-version-1 load/cache owner for the selected complete catalog (on-disk layout in `pyrite/_catalog_layout.py`; bundled `data/catalog/` by default). Owns `load_material_catalog`, the default `CATALOG` singleton, and compatibility exports for the frozen public record types. The exported types retain their historical `pyrite.materials.catalog` pickle identity even though their definitions live in `_schema.py`. `material_keys` preserve declaration order for a single-file catalog and file-name order for a catalog directory.
- Public: `MaterialCatalog`, `MaterialConfigError`, `CrystalInfo`, `CrystalSpec`, `MediumSpec`, `MaterialSpec`, `MaterialValidationSpec`, `ScanSpec`, `LayerSpec`, `load_material_catalog`.
- Deps: `materials._schema`, `materials._parse`, `DATA_DIR`. Imports no driver package, so `materials` remains a leaf component of the dependency DAG.

## `materials/_schema.py`
Frozen catalog record types, schema constants, catalog accessors, and grouped validation-error formatting. `catalog.py` re-exports the public types and pins their historical pickle module identity.
- Internal owner of: `MaterialCatalog`, `MaterialConfigError`, `CrystalInfo`, `CrystalSpec`, `MediumSpec`, `MaterialSpec`, `MaterialValidationSpec`, `ScanSpec`, `LayerSpec`.
- Deps: `materials._identity`, `materials._catalog_decode`, `_numerics`, NumPy.

## `materials/_parse.py`
Catalog validation and assembly for materials, crystals, media, profiles, stacks, scan descriptors, energy-grid artifacts, and transport support. Owns phase-specific CIF resolution below the selected catalog's `cifs/` with packaged `data/cifs` fallback, plus pinned-reflection policy; delegates beam/detector blocks to `_beam_detector_parse.py`.
- Deps: `materials._schema`, `materials._beam_detector_parse`, `materials._catalog_decode`, `materials._cif`, `materials._identity`, `materials._transport_data`, `_energy_grid_artifacts`, `_numerics`, `DATA_DIR`, NumPy.

## `materials/authoring.py`

Validated `build_medium` / `build_crystal` definitions and explicit `save_material`;
composition + g/cm³ conversion reuses transport weights, CIF import reuses `_parse`,
and safe publication targets the shared catalog layout. Medium definitions stay
in `media/`; crystal recipes reference `crystals/` from `materials/`. CLI adapters:
`cli/commands/material_add.py`, `material_inventory.py`; maintainer route:
`pyrite-dev material`. User objects and managed CIFs ship through
`remote/transport.py::_user_catalog_entries`. Workflow: `guides/adding-materials.md`.

## `materials/_beam_detector_parse.py`
Beam, detector, filter, and physical-detector catalog parsing. Detector blocks resolve to validated acceptance mappings rather than built `Detector` objects, keeping `materials` below `detectors`; `campaign.config.catalog_detector` constructs the detector. Copied bounds are pinned to `Detector.__post_init__` by `tests/materials/test_material_catalog.py::test_catalog_detector_bounds_match_the_detector_dataclass`.
- Deps: `materials._catalog_decode`; imports no driver or detector package.

## `materials/_identity.py`
Display identity for a material — chemical formula, crystal phase, crystal cut — and the label derived from them. Labels are never authored in the catalog; `[materials.*]` may only override the *name* segment via `display_name`, and the cut is always appended, so a label cannot contradict its crystal record. The cut is the declared slab normal reduced to its primitive representative, rendered `(hkl)` when the crystal declares `surface_hkl` and `[uvw]` when it declares `beam_uvw` — the two are mutually exclusive spellings of the same axis, and nothing is converted between them. Every packaged crystal declares `surface_hkl`, so every label reads `(hkl)`; a cell on hexagonal axes renders four Miller–Bravais indices instead — `(0001)`, not `(001)`.
- Public: `MaterialIdentity`, `reduce_indices`, `format_indices`, `bravais_indices`, `hexagonal_setting`; `CutFrame` alias.
- Deps: none (leaf).

## `materials/_catalog_decode.py`
Primitive decoders for schema-version-1 catalog grids and descriptors: number/negative validation, immutable float64 grid construction, `values`/`arange`/`linspace`/`logspace` line-grid kinds. `_parse.py` orchestrates validation; `_schema.py` owns record types and error aggregation.
- Internal: grid/descriptor decode helpers; `GridValue`, `CxrGridByEnergy` aliases.
- Deps: NumPy.

## `materials/_cif.py`
Structural adapter around Gemmi 0.7: CIF parsing, symmetry expansion, cell parameters, fractional sites, volume only. PyRITE retains ownership of form factors, structure factors, reflection selection, attenuation, transport.
- Internal: `gemmi_structure_to_crystal_info`, `load_crystal_from_cif`; `crystals_crystal_to_crystal_info` remains a dependency-free compatibility adapter for already expanded objects.
- Deps: external Gemmi, NumPy.

## `materials/crystal.py`
Catalog-backed crystal compatibility projection, structure factors, X-ray optical constants — physics data layer under Monte Carlo.
- Public: `load_crystals`, `load_crystal_from_cif`, `crystals_crystal_to_crystal_info`, `reciprocal_g_vector`, `g_mag`, `debye_waller`, `structure_factor`, `chi_g`, `U_g`, `absorption_length_ang`, `dominant_reflections`, `beta_from_Ee`; `CRYSTALS` registry.
- Deps: `materials.atomic`, `materials.catalog`, `materials._cif`.

## `materials/emission.py`
Supported vacuum photon-energy comparison API. Resolves catalogue entrance crystals and primary parallel reflections or explicit Miller triples, applies the catalogue crystal cut and sample tilt, and returns material-leading NumPy arrays in broadcast/grid modes. `materials.crystal._orientation_R` owns the shared cut rotation; `montecarlo.geometry._orientation_R` forwards to it for existing callers.
- Public: `photon_energies` (also lazy-exported from `pyrite`). Deps: `materials.catalog`, `materials.crystal`, NumPy; no Monte Carlo or backend imports.

## `materials/channeling/` (package)
Static, scattering-free quantum-channeling profile (ADR-0018, #503), in the plane-local frame. `thermal` interpolates phonon-DOS site `B(T)` from `data/channeling_thermal.toml` and extrapolates with the Debye model; crystals without rows are refused (`ChannelingDataError`). `potential` builds the planar continuum potential and nuclear/electron densities from Mott--Bethe `Z - f0` Fourier sums with thermal smearing, detecting extinctions to set the true period. `transverse` solves the Bloch plane-wave eigenproblem with mass `gamma m`, gives dipole matrix elements, entry populations versus incidence angle (with divergence), and the shared cancellation-free Doppler denominator. `entry` maps sample tilt and a #25 `CrystalCut` lattice rotation onto the plane-local incidence, with independent-atom mean-inner-potential refraction at the face. `axial` (#508, experimental) builds the 2D string potential of a lattice axis from every zone reflection, on the smallest rectangular projected cell, with per-site/anisotropic displacement tensors, atomic strings, band-limited grid synthesis and the 2D plane-wave eigenproblem `axial_states`. Depth evolution and emission belong to `montecarlo/channeling/` (#504, #505).
- Public: `planar_profile`, `PlanarProfile`, `plane_incidence`, `PlaneIncidence`, `mean_inner_potential_eV`, `transverse_states`, `TransverseStates`, `entry_populations`, `line_energy_eV`, `doppler_denominator`, `longitudinal_kinematics`, `thermal_b_factor`, `debye_b_factor`, `debye_integral`, `fit_debye_temperature`, `load_thermal_sites`, `ThermalSite`, `ChannelingDataError`. Deps: `materials.atomic`, `materials.crystal`, `DATA_DIR`, NumPy, SciPy; no Monte Carlo imports.

## `materials/recoil.py`
Provisional exact vacuum line-recoil reference: physical positive-mass-shell
photon roots, fixed-momentum energy mismatch and its stable derivative Jacobian.
Paired NumPy broadcasting. Read only by the record-only line coverage screen
(`spectrum/cxr/_coverage.py`); no production scorer or `photon_energies`
selection, medium/interface model, quantum amplitude or arbitrary-trajectory
coherence.
Ledger: `line-recoil-resonance`, `line-recoil-formation`, `line-recoil-validity-screen`.
- Entry points: `recoil_resonance_eV`, `recoil_shift_fraction`, `recoil_mismatch_inv_ang`, `recoil_mismatch_slope`.
- Deps: `materials.crystal` (shared constants), NumPy.

## `validation/validation_oracles.py`
Optional validation-only adapters for external crystallography/scattering comparators. Pinned `Dans_Diffraction` 3.4 backend builds or loads crystals, compares lattice parameters, reciprocal-vector magnitudes, and `|F_hkl|²`, then applies fail-closed geometry/non-resonant/dispersive thresholds; production physics stay in `materials/crystal.py`.
- Public: `build_dans_crystal_from_pyrite`, `load_dans_crystal_from_cif`, `compare_lattice`, `compare_reflection_geometry`, `compare_structure_factor_magnitudes`, `validate_dans_crystal`; comparison, tolerance, and report dataclasses.
- Deps: `materials.crystal`; imports `Dans_Diffraction` lazy, only when check ask.

## `validation/_pinned.py`
Shared on-demand SHA-256-verified download of external validation-only reference files; neither the files nor their fetch are part of production runs.
- Public: `read_pinned`. Deps: stdlib only.

## `validation/brem_angular.py`
Validation-only Schiff (Koch–Motz 2BS) photon angular density and enclosed-flux-angle comparison with the BremsLib double-differential tables. Driven by `checks/brem_angular_comparison.py`; ledger `bremslib-angular-schiff`.
- Public: `schiff_density`, `enclosed_angle`, `compare_angular_shape`. Deps: `montecarlo.spectrum.brem_bremslib`.

## `validation/shell_ionization.py`
Bote–Salvat EEDL comparison (parser and fit from `montecarlo/bote_salvat_ionization.py`), compared with pinned EEDL subshell rates and PyRITE characteristic-production yields. Pinned NIST `xione.jl` is fetched on demand, never packaged. Driven by `checks/shell_ionization_comparison.py`; ledger `eedl-shell-ionization-comparison`.
- Public: `BoteSalvatElement`, `parse_bote_salvat`, `load_bote_salvat`, `compare_shells`, `compare_production`.
- Deps: `validation._pinned`, `montecarlo.eedl_ionization`, `montecarlo.spectrum.characteristic`, `paths`.

## `validation/brem_sources.py`
Validation-only comparison of EEDL and BremsLib bremsstrahlung `dsigma/dk` with the Seltzer–Berger scaled tables (EGSnrc `nist_brems.data`, pinned SHA-256, fetched on demand into the user data directory, never packaged). Driven by `checks/brem_source_comparison.py`; ledger `brem-source-comparison`.
- Public: `SeltzerBergerTable`, `parse_seltzer_berger`, `load_seltzer_berger`, `model_chi`, `compare_sources`, `SourceComparison`.
- Deps: `validation._pinned`, `montecarlo.spectrum.brem` (production evaluation), `paths`.

## `validation/validation_background.py`
Analysis-only external bremsstrahlung comparison, weighted sideband normalization, and experimental subtraction. External spectra stay in detected units and enter through `montecarlo.load_external_brem`.
- Public: `BackgroundFit`, `BackgroundComparison`, `fit_external_background`, `subtract_external_background`, `compare_external_background`.
- Deps: `montecarlo`, NumPy.

## `materials/atomic.py`
Atomic scattering factors (Z, f0, f′, f″) from **xraydb** for any element — no hand-maintained table.
- Public: `cromer_mann_f0`, `henke_dispersion`, `atomic_form_factor`, `load_henke`.
- Deps: none (leaf; external xraydb).

## `materials/radiation_length.py`
Tsai radiation length (PDG Eq. 34.26 with the DBM Coulomb correction and Tsai's Z ≤ 4 logarithms) per element and for `(element, atoms/Å³)` compositions by number-density additivity. Screening scale only; no kernel reads it. Ledger `radiation-length-tsai`.
- Public: `radiation_cross_section_ang2`, `radiation_length_ang`. Deps: `materials.atomic`, `materials.crystal` (constants).

## `materials/photon_cross_sections.py`
Per-atom EPDL2025 photon cross sections (photoelectric, coherent, incoherent, nuclear- and electron-field pair) for Z = 1–100, 1 eV – 100 GeV, from the packaged derived table `data/photon_cross_sections/epdl2025_mf23.npz` (SHA-pinned; generator `scripts/release_epdl_table.py`). Source of every narrow-beam escape `μ` since #274; the refractive index and couplings keep Chantler `f₂`.
- Public: `photon_cross_sections_ang2`, `total_photon_cross_section_ang2`, `photoelectric_edges`, `PHOTON_CHANNELS`, `EPDL_E_MIN_EV`/`EPDL_E_MAX_EV`, `ATTENUATION_MODEL` (identity marker).
- Deps: `materials.atomic` (`Z_TABLE`), `DATA_DIR`.

## `materials/photon_differential.py`
EPDL2025 subshell photoionization (EADL designators and binding energies), coherent form factor `F(x, Z)` and incoherent scattering function `S(x, Z)` (`x = sin(θ/2)/λ`, Å⁻¹) for the opt-in photon transport of #341, from the fetched dataset `epdl-differential` (SHA-pinned; `scripts/release_epdl_table.py --table differential`). No transport caller yet.
- Public: `photoionization_subshells`, `subshell_photoionization_cross_sections_ang2`, `coherent_form_factor`, `incoherent_scattering_function`, `coherent_form_factor_table`, `incoherent_scattering_function_table` (read-only knots for bounded rejection), `PhotoionizationSubshell`, `X_MAX_INV_ANG`.
- Deps: `materials.photon_cross_sections`, `datasets`.

## `montecarlo/` (package)

`transport/photon_navigation.py` owns the standalone host all-channel photon
first-flight reference: sample-frame stack/prism ray intervals, vacuum entry
and gaps, and EPDL channel/element competition from explicit uniforms. It has
a production caller through `photon_histories`; the existing primary pair/annihilation route remains separate.

`transport/photon_cuda.py::scene_first_interactions_cuda` adds explicit CUDA
first flights: `_photon_cuda_kernel.py` owns fused binary64 clipping and stable
interval/channel selection, with the shared NumPy/CuPy array reference retained
as an explicit comparison path. Host inputs/rates/draw conversion feed device
navigation and sampling; repeated compositions reuse host rates and outputs
remain on device. `photon_scattering_cuda`, `photon_pair_cuda` and
`photon_photoelectric_cuda` own explicit-draw conditional samplers, backed by
private raw kernels and canonical `_photon_cuda_relaxation` tables.
`photon_cuda_events` admits bounded device event waves, groups conditional
collisions, retries identical Philox prefixes/capacities and reconstructs
physical products for the shared lineage/ledger queue. No CPU sampler fallback.
The production selector and execution identity live in `runner/photon_aperture`.

`transport/relaxation.py` owns the standalone host sampled EADL vacancy
cascade: explicit draws, depth-first transition ancestry, terminal-hole
multiplicity, and individual energy budgets with signed transition defects.
It consumes validated `eadl_relaxation` data; the existing expected cascade
and characteristic scorer retain their current paths. CPU photon histories call this sampler.

`transport/photon_scattering.py` owns standalone CPU Rayleigh (Thomson × F²)
and Compton (Klein–Nishina × S, without Doppler/binding corrections) events,
including recoil and sample-frame directions. `transport/photoelectric.py`
selects EPDL subshells, samples Sauter–Gavrila photoelectrons and calls the
sampled EADL cascade. `transport/photon_sampling.py` supplies explicit-draw
validation and stable angular rotations. CPU photon histories call these
references; integrated EPDL rates own channel/atom competition.

`transport/photon_histories.py` chains those references into a host photon
history queue with optional bounded CUDA event waves: lineage-keyed Philox streams, isotropic relaxation emission,
charged electron/positron handoff rows, an absorbing sample-frame aperture
for ordinary hits, and a per-history weighted energy ledger. Filter plates
navigate as oriented `PhotonBox`es (`photon_navigation.scene_first_interactions`);
lab-frame poses convert through `photon_box_from_pose`/`PhotonAperture.from_pose`.
This photon-only owner returns charged rows; the coupled owners below transport them.

`transport/photon_coupling.py` alternates that queue with a charged stage:
`track_coupled_histories` launches photon-born electrons above the stage
threshold inside the transport stack, `cascade_charged_stage` runs them
through the shell soft/hard cascade from an explicit `LaunchState`
(`secondaries.transport_secondary_cascade(launch=...)`), and their hard
bremsstrahlung photons re-enter the queue with lineage keys. Electron-only
stages keep positrons pending; per-history coupled ledger closes. Production aperture scoring uses this owner when feedback is enabled.

`transport/photon_feedback.py` adds `FeedbackChargedStage`: in-stack
positron transport/annihilation, sampled charged-impact EADL relaxation and
Auger re-entry. `transport/photon_event_sources.py` adapts unattenuated primary
characteristic/annihilation emissions, retaining sub-cutoff source energy.
`photon_transport.simulate_photon_continuum(charged_feedback=True)` connects
the stage to the experimental Scene boundary with enabled-only positron table
provenance. The production aperture path also uses the stage, forcing the CUDA
charged core on that backend. Production `_photon_scene_feedback` prepares
per-body catalog tables/pins for named filters; independent full-track validation
remains open. Contract:
[CPU secondary feedback](../research/physics/photon-secondary-feedback.md).

`transport/photon_charged_escape.py` retains terminal charged phase space;
`transport/photon_reentry.py` owns explicit CPU/CUDA scene feedback
scheduling across target slabs, vacuum gaps and filter boxes. Production
`_photon_scene_feedback.py` owns per-body table slices and enabled-only pins;
local transforms and scheduling stay on the host, ray clipping follows the
selected backend. The coupled
queue validates matching geometry and uses this stage's material ownership.

`photon_observation.py` owns explicit atomic save/load of the experimental
Scene continuum aperture tally, including identities and pinned provenance.
It stores reduced arrays; `PhotonContinuumObservation.save` is the standalone
entry point. `results.aperture` adapts the same checked payload into production
Result/checkpoints and replay. Finished CPU/CUDA payloads need no accelerator
or current scientific tables to read.

Simulation core: electron transport, segment-sum PXR+CBS and EEDL characteristic line spectra, bremsstrahlung, parallel case runner, and detector-convolution helpers. Split from single module into submodules; **every public and internal name re-exported from package** — `from pyrite.montecarlo import X` unchanged (`tests/montecarlo/test_exports.py` freeze export set).
- `case` — frozen, validating `Case` input schema. `build_cases` returns it; `to_dict()` reproduces the version-1 legacy mapping and `run_case` temporarily accepts either representation.
- `_resources` — execution resource-policy resolution and backend-neutral pre-allocation chunk admission. Small devices default conservative.
- `materials.attenuation` — `_normalize_composition`, `_mu_total_inv_ang` (EPDL2025 narrow-beam total), `_finite_mu_or_raise` (continuum out-of-domain policy), `_layer_dz`, `_stack_tau` (composition + cross-stack self-absorption). Deps: `pyrite._backend`, `materials.photon_cross_sections`.
- `transport/` — compatibility package for the former transport monolith; `__init__.py` re-exports its complete pre-split surface. `api.py` owns `simulate_trajectories` (multilayer-stack aware via `layers=`), which forwards an explicit `arguments.TransportInputs` payload to the secondary cascade or to `host_pass.run_transport_pass`, the staged host pass (validation, layer tables, entrance sampling, buffers, core dispatch, result assembly); `beam_entry.py` its initial entrance-face positions (spot, Twiss, groove phase) and `sample_primary_entry`, the primaries' complete entrance states composed from the same seed-pure draws without transport (#506), while `kinematics.py`, `scattering.py`, `stopping.py`, `straggling.py`, `lut.py`, `cores.py`, and `batching.py` own the corresponding transport stages. NumPy/Numba, plus a device core: `resolve_transport_core` picks the CUDA one above `CUDA_TRANSPORT_MIN_ELECTRONS` (1000) electrons on a CUDA box, pinnable with `PYRITE_MC_TRANSPORT_CORE`. `continuation.py` owns the opt-in bounded-continuation rounds of the exact CUDA core (`PerElectronTransportConfig.continuation`, #789): fixed active population, row-quantum yields, refill and canonical reassembly ([contract](../repo-design/compute/gpu-transport-rawkernel.md#bounded-continuation-automatic-production-schedule)). `energy_model` selects the propagation rule (`"frozen"` historical left-endpoint, `"midpoint"` default predictor--corrector); `max_dE_frac` splits a physical flight into numerical substeps under `"midpoint"` and adds the `E_end_keV`/`t_end_ang`/`E_repr_keV`, `flight_id`/`substep_id`, and `event_kind` row fields. `events.py` owns the row-end event codes (`SegmentEvent`, reserved hard-event codes included) and `check_segment_event_contract`, the flight/event invariants every CPU and CUDA core must satisfy; it imports no accelerator module so both core families read its constants. How the Urban sampler is applied inside the cores — the cutoff crossing rule under a random loss, what substepping still guarantees, and the per-core bookkeeping — is derived in [straggled transport integration](../repo-design/compute/straggled-transport-integration.md); `cores.py` carries only the resulting invariants at each site. Deps: `materials.attenuation`, `DATA_DIR`.
- `transport.inelastic` and `transport.dielectric` — host-only candidate soft/hard transfer distributions. These candidates are not selected by the production transport cores.
- `transport.shell_oscillators` → `shell_gos` → `shell_rates` → `shell_partition` → `shell_sampling` — host PENELOPE-2024 shell GOS model, stopping-closed and split at `W_c`. `transport.shell_transport` tabulates that partition on each layer's SBETHE nodes for the `simulate_trajectories(inelastic_model="shell-soft-hard")` mode (lazily imported; production default via `inelastic_model="auto"`, resolved per case by `campaign/inelastic_cases.py`); `transport.hard_inelastic` holds its Numba hard-transfer, recoil and soft-loss kernels, read by the `inelastic=True` CPU core specializations; `transport._jit_shell_device` holds their CUDA twins for the exact kernel. Grooved and CUDA-LUT transport reject the mode. `transport.secondaries` owns the opt-in `secondary_threshold_eV` cascade: harvesting launched secondaries from hard rows, their stream keys and explicit launch state, the generation loop over the existing cores, the joined track-identity result, and `secondary_energy_balance` (#94). See [shell soft/hard transport](../physics/beam-transport/shell-soft-hard-transport.md).
- `transport.pair_production` — opt-in (`pair_production_model`) pair conversion of coupled hard photons above 2mₑc²: host EPDL first-interaction sampling through the layer stack/footprint, PENELOPE-2024 pair energy/angle sampling, and per-photon keyed streams. `transport.secondaries` launches pair electrons as the cascade's `launch_kind=2` tracks and records positrons in `result["pair_production"]` (#275); opt-in `positron_transport` launches them as `launch_kind=3` tracks run per generation with positron SBETHE/ELSEPA tables, `F_p`-scaled BremsLib (`hard_radiative.positron_bremslib_tables`) and Bhabha shell tables (`shell_gos`/`hard_inelastic` positron branches, tagged `branch + 3`) (#276). `transport.annihilation` annihilates those positrons (#295): host Heitler in-flight thinning of the transported rows (`ANNIHILATION` event), at-rest 511 keV pairs, and one EPDL first-interaction step per photon. `transport._jit_annihilation` owns the CUDA cross section and device-resident batched emission sampler; CUDA cascades transfer its sampled photons only at the host EPDL scoring boundary. See [positron transport](../physics/beam-transport/positron-transport.md). See [hard bremsstrahlung events](../physics/radiation-physics/hard-bremsstrahlung-events.md#pair-conversion-of-hard-photons).
- `transport.hard_radiative` — host BremsLib soft/hard radiative partition, hard-photon energy/direction sampling, layer-table packing, and post-transport photon completion for coupled `simulate_trajectories`; `transport._jit_radiative` holds its Numba scalar twins, read by the `radiative=True` exact CPU core specializations, and `transport._jit_radiative_device` their CUDA twins for the exact kernel. Grooved and LUT transport use uncoupled scoring. `Physics`/`Settings`/`Case` carry it as divergence-only `radiative_model`/`radiative_cutoff_eV` keys; `runner.case_tables._case_radiative_kwargs` passes it to transport and `runner.emission._coupled_brem_from_segments` scores it (#172). See [bremsstrahlung](../physics/radiation-physics/hard-bremsstrahlung-events.md).
- `montecarlo/bote_salvat_ionization.py` — Bote–Salvat K/L/M fit and parser over the opt-in `bote-salvat` dataset; `spectrum/characteristic.py` `ionization_model="bote-salvat"` tabulates it for scoring and `spectrum/ionization_systematic.py` reports EEDL/Bote–Salvat line-production disagreement. Ledger `bote-salvat-ionization-backend`.
- `montecarlo/eedl_ionization.py` — checksum-pinned (fetched, `datasets`) EEDL MF=23 subshell parser shared by transport inputs and characteristic scoring; `montecarlo/shell_ionization.py` multiplies its shell cross sections by catalog element number densities for host-side vacancy-channel rates.
- `montecarlo/eadl_relaxation.py` — checksum-pinned EADL MF=28 relaxation parser/validator, the exact binding-energy-ordered vacancy cascade, and its energy budget; consumed by characteristic scoring.
- `transport._jit_device` / `transport._jit_shell_device` / `transport._jit_radiative_device` / `transport._jit_kernel` / `transport._jit_kernel_variants` / `transport._jit_lut_kernel` / `transport._jit_launch` — the `cupyx.jit` port of `transport._transport_core_ungrooved_perelectron`: one thread per electron, run to completion, output slots addressed by electron index. Same arithmetic as the CPU core, so the two stay one algorithm. Layered device helpers and constants (`_jit_device`, shell-mode twins in `_jit_shell_device`, radiative-mode twins in `_jit_radiative_device`) → the exact and LUT `__global__` kernels (`_jit_kernel`, `_jit_lut_kernel`) → host launchers and launch geometry (`_jit_launch`). The exact kernel is compiled per feature configuration: `_jit_kernel_variants.transport_kernel_variant` caches one kernel per `TransportKernelSwitches` tuple, and the launcher selects it (#827). All are kept out of `__init__.py`'s eager imports — `api.py` and `batching.py` reach `_jit_launch` only through deferred, function-local imports — so their module-scope `cupy` import cannot break CPU-only startup. See `docs/repo-design/compute/gpu-transport-rawkernel.md`.
- `geometry` — `tilted_geometry`, `detector_directions`, `_orientation_R`, `_small_tilt_R`, `_mosaic_quadrature`. Deps: `materials.crystal`.
- `spectrum/` — compatibility package for the former flat module. `lines/` owns `mc_cxr_spectrum` (PXR+CBS, cross-stack self-absorption, exact mosaic average) and `mc_cxr_spectrum_solid_angle`; `brem.py` parses the pinned EEDL MF=23/527 totals and MF=26/527 photon spectra for the EEDL fallback in `mc_brem_spectrum`, retains Bethe--Heitler as an optional/fallback backend, stages EEDL panels once per element/grid, and owns `load_external_brem`; `brem_blocks.py` scores a broad continuum grid in contiguous photon blocks (#819) so that staging scales with the block width, which `runner.chunking._brem_photon_block` sizes from the spectrum budget; `brem_unit_base.py` refines the decade-spaced EEDL photon panels by unit-base interpolation (32 sub-panels per decade) before staging; `brem_bremslib.py` stages and interpolates BremsLib SDCS/DDCS tables for the direction-resolved `cross_section_model="bremslib"` (tables are resolved by `xsgen/bremslib/tables.py` and passed in; the physics core never imports `xsgen`); `brem_jit_kernel.py` owns the fused CUDA EEDL/Bethe--Heitler reduction; `characteristic.py` re-exports the shared EEDL shell loader, joins xraydb direct-vacancy relaxation and natural level widths for emission, integrates Lorentzian line profiles over the fine grid, and owns `mc_characteristic_spectrum`; `coherent.py` groups the CUDA-only coherent-kernel surface (`coherent_jit_kernel.py` reduction, `coherent_stream_jit_kernel.py` streaming prologue/field/finalize, `coherent_grouped_jit_kernel.py` per-electron grouped intensities); `diagnostics.py` owns the opt-in, host-only radiation error estimators `cxr_endpoint_resonance_drift` and `brem_endpoint_quadrature_error` plus `subdivide_flights`, none of which sit on a default call path. `transition_radiation.py` is the host-only TR screening tool (#102): exact planar-stack transition radiation of a straight charge (`planar_transition_radiation`, and per crossing `interface_transition_radiation_amplitude`), the closed Ginzburg--Frank form, the catalog-medium permittivity, and the per-observable omission bound. `transition_radiation_crossings.py` (#103) reduces midpoint transport rows to the physical planar-face crossings TR is emitted at (`boundary_crossings`: primary entry, layer boundaries with a composition change, top/bottom exits; substeps and equal-composition boundaries never cross, side exits and grooves are excluded), with track, charge, slab-frame point on the plane, direction, energy, `t0 + age`, face normal and adjacent media. `crossing_amplitudes`/`crossing_transition_radiation` evaluate each crossing through the stack with `transition_radiation.interface_transition_radiation_amplitude` (the solve split per interface source) and sum one history's crossings coherently, histories incoherent, or across histories (CTR, `coherence="bunch"`, distinct-pair estimator). `runner.transition_emission` (#103) scores a `Case.transition_radiation` record as the separate `spec_transition` component on the line grid at the case and observation directions (continuum population `Ne_brem`, per-layer Chantler permittivity via `transition_radiation.composition_dielectric`, `TRANSITION_MODEL`); selected by `Physics.transition_radiation` (`campaign.transition_cases` lowers the record; `TRANSITION_MODEL` joins identity and provenance only when on), stored as the optional `transition.h5` checkpoint component, rescored by reline and artifact replay, and exposed as `Result.transition`/`SpatialResult.transition`; adaptive precision and electron-block transport refuse it (`montecarlo.case.refuse_transition_radiation`). `transition_radiation_jit_kernel.py` is its fused CUDA RawKernel (crossing amplitudes and per-history field sums in registers, deterministic block reduction; the same source builds as host C++ for CPU tests), used for CuPy arrays unless `transition_radiation_crossings._USE_JIT_TRANSITION` is off; `runner.transition_emission` scores on the CuPy backend when there is one. `transition_radiation_detector.py` (`detected_transition_radiation`) integrates that density over planar-detector pixels (`_planar_geometry` rays, transport tilt) and energy channels by adaptive trapezoid/sub-pixel doubling with an asymptotic error estimate; incident photons only. `coherent_windows.py` seeds the per-row coherent line-axis windows a windowed automatic line-grid policy uses for `coherent_emission` (#350), on the row kinematics `cxr_seeds.reflection_rows` shares with the kinematic seeds. Line, brem, and coherent CUDA kernels are colocated in this package and remain lazy on CPU. Deps: `pyrite._backend`, `materials.attenuation`, `transport`, `geometry`, `materials.crystal`.
- `spectrum/cxr/` — the line spectrum split by phase, bottom-up (issue #65). `_policy` owns the five device-kernel dispatch switches; `_kernels` the leaf array numerics (lineshape, interpolation, per-segment geometry, reflection tabulation); `_bin_quadrature` the opt-in `line_quadrature="bin-mean"` closed-form `sinc^2` bin masses (host, CuPy, and fused CUDA; issue #116); `_bin_tree` the sorted near/far line reduction for large CUDA bin-mean batches (issue #248); `_formation` the per-piece complex formation factor under absorption for the coherent and flight-grouped reductions, plus the escape-piece row split (issue #181); `_setup` the `SpectrumRequest` record and the one-shot `_prepare_spectrum`; `_per_hkl` the reference route that walks one reflection at a time (layered stacks, grooved escape, flight-grouped coherence); `_batched` the default route over stacked reflection tables; `_spectrum` the public entry points, route dispatch, and finalization. `__init__.py` re-exports the whole pre-split surface, so `pyrite.montecarlo.spectrum.cxr.<name>` still resolves — but those are bindings, not seams: a test that *replaces* one patches the owning submodule (`_policy` for the switches, `_kernels` for `chi_g`, `_setup` for `xp`).
- `detector` — `detector_efficiency`, `eds_fwhm_eV`, `aperture_fwhm_eV`, `mosaic_fwhm_eV`, `mosaic_psi_rad`, `convolve_detector`. Deps: `materials.attenuation`, `geometry`, `transport`, `materials.crystal`.
- `montecarlo/groove.py` — blazed sawtooth entrance-face grooves (escape-path engineering): closed-form entry/escape, `Sweep.groove_spacing_ang` knob.
- `montecarlo/channeling/` — open-system depth evolution and channeling-radiation emission of planar-channeled electrons (ADR-0018, #504/#505; CPU only). `operators` builds the Lindblad dissipator kernel `C(x, x')`: Einstein-model TDS phase covariance and the local-density electronic kernel from the `transport.shell_gos` PENELOPE oscillators resolved in `q_perp`. It stores them as Bloch-residue momentum blocks (`ScatteringKernel`), with the full-transfer diagonal giving the absorptive potential, and provides the cell-averaged decoherence rate `decoherence_rate`. `density` (`DensityMatrixSolver`) integrates the Bloch-blocked density matrix: exact eigenbasis phases plus non-Hermitian loss, a position-space jump product, and Strang splitting. Its collapse rule keeps `E_perp < barrier + margin` and books free, overflow, entry and reprojection flux. `rate_matrix` gives the Pauli rate-equation limit. `mcwf` (`MCWFSolver`) unravels the same operators into per-electron quantum jumps: waiting-time jumps, kernel eigen-channels per Bloch residue, the same collapse rule, and per-electron Philox streams keyed on `(seed, electron)`. `dipole` holds the photon transition primitives: the generic first-order operator `transition_operator`/`first_order_transition` and its dipole limit `dipole_rate_per_ang`, the boosted pattern and Stokes parameters. `emission` (#505) is the diagnostic channeling-radiation API built on them: the line catalog `emission_lines` (bound -> bound lines and free -> bound capture from tracked free states) with expected-value rates on `rho(z)` and Bloch-block coherence-transfer groups (`LineGroup`) for the line shapes, `channeling_spectrum` (photons/eV/sr/electron with a separate capture component, chained `DensityMatrixSolver.propagate`, complex-Voigt mode profiles, aperture and in-plane divergence quadrature, `_stack_tau` exit absorption), and the `capture_bound` diagnostic; `transition_radiation` holds the excluded-background `transition_radiation_bound`. `states` (#508) holds user-definable entry wavefunctions: Gaussian, Hermite/Laguerre–Gauss (vortex, donut), Bessel–Gauss, finite-energy Airy, aberrated/annular/vortex aperture probes with user phase plates, `from_array`, `from_abtem`, coherent `superpose`, incoherent `mixture`, `translate`. `planar_entry_density` projects them onto the planar Bloch basis (fixed or averaged impact) for `DensityMatrixSolver.entry_from_plane`, and `channeling_spectrum`/`MCWFSolver.run` accept `entry_state=`. `realspace` (#508, experimental axial tier) holds the periodic `RealSpaceGrid` (entry states sampled by periodization, carrier tilt factored out) and the Strang `SplitStepPropagator` (coherent `U` plus optional absorptive `W`). `axial_dissipator` carries the planar TDS/electronic Lindblad kernels onto that grid (per-atom phase-covariance jumps, out-of-band sink, local-density electronic jumps), `axial_mcwf` (`AxialMCWF`) unravels them into batched per-electron quantum jumps, and `vortex` (`CylindricalProjection`) gives `P_ℓ` and `⟨L_z⟩` about a chosen string (usable as an `AxialMCWF` observable). `axial_emission` builds the flat bound-level line catalog of one string and its helicity/`J_z`-resolved emission (`AxialLine.polarization`, `axis_stokes`, `photon_angular_momentum`), with `LevelDensity` as the MCWF observable. `runner.channeling_emission.case_channeling_spectrum` (#507) writes the expected-value source to `spec_channeling` on the native line grid, with a beam/plane frame, slab-normal attenuation and geometric entrance-path weights. Its constant-energy source and laterally infinite photon-absorption slab remain diagnostic-model approximations. `handoff` (#506) owns the `CHANNELING_MODEL` identity marker (v2 for persisted CR) and the cached `zone_solver` shared by source and transport, plus the zone handoff: `solve_zone` (one density solve per case over the mean path at constant `S(E0)`, cached), `sample_handoffs` (per-electron fate entry/free/overflow/bound, depth, `E_perp` and sign from counter-addressed SplitMix uniforms keyed on `(seed, electron)`) and `handoff_directions` (`psi = sqrt(2 E_perp / pv)` toward the plane normal, in-plane part kept; overflow at the band edge); `runner.channeling.transport_channeling_case` builds the zone rows (`EVENT_DECHANNEL` or an exit, tagged `channeling_zone`), launches the rest through the `transport.secondaries` launch seam with each primary's own stream key on the per-electron core, and joins them; `Case.channeling` cases are refused before work by trajectory capture, electron-block transport and adaptive precision (`montecarlo.case.refuse_channeling`); `runner.channeling_emission` also rescores the expected source for directions/observations, replay and line recompute, and native continuation runs on the per-electron CPU or CUDA core (#521). Consumes `materials/channeling/` profiles. Deps: `materials.channeling`, `materials.crystal`, `materials.attenuation`, `transport.shell_gos`/`shell_rates`/`inelastic`.
- `montecarlo/transverse.py` — transverse phase-space policy on the input side of `BeamSpec`: a Courant-Snyder triplet `(eps_n, beta, alpha)` per plane, resolved per case (normalized emittance is the stored input, geometric is derived as `eps_n / (beta*gamma)`) and sampled into per-electron offsets and slopes. The `y` plane mirrors `x` unless set. Zero first-party imports, so it is the destination `transport/api.py` samples from directly (`resolved_from_mapping`, `sample_transverse`) rather than reaching up into `campaign` (issue #69). See `docs/physics/beam-transport/beam-phase-space.md`. Public: `TransverseDistribution`, `ResolvedTransversePlane`, `ResolvedTransverseDistribution`, `resolve_transverse_distribution`, `sample_transverse`. Deps: NumPy, SciPy constants.
- `trajectories` — opt-in per-case transport artifacts (#159): `TrajectoryCapture` (picklable, crosses into transport workers), versioned HDF5 writer/reader (`write_trajectory_artifact`, `read_trajectory_artifact`; atomic `.partial` rename, alias hard links, bounded device-array download; schema 2 adds the typed case and `/spectrum_inputs` and stores row fields electron-major with a restoring `/transport_order`; schema 1 stays readable), `iter_electron_row_blocks` (bounded whole-electron row streaming), deterministic `artifact_relpath`, and `preflight_capture` overwrite/stale checks. `trajectory_selection` resolves whole histories (bisection over electron-major rows) or tracks to stored-row runs and reads them back in transported order with units (`select_trajectories`, `read_selection`). `trajectory_inventory` discovers artifacts (skipping `.partial`, incomplete and foreign files) and describes one from metadata and bounded ID-column scans (`discover_artifacts`, `describe_artifact`; #397). `trajectory_export` streams an artifact's segments, or a selection's, to VTK XML PolyData (`.vtp`). `runner._transport_case` writes when handed a capture; `run_cases`/`run_sweep` pass it only when set. Deps: `_backend`, `h5py` (lazy).
- `runner/` — compatibility package for the former flat module. `__init__.py` is the re-export facade and owns `_worker_init`, the `_TIMING`/progress state, and the Nsight helpers; `case_driver.py` owns per-case orchestration (`run_case`, `run_case_directions`, `_transport_case`, `_spectrum_case`, the `_cxr_*` family; patch it there, except the names it reads late through the package: `_TIMING`, `_is_gpu_oom`, `_nsys_range`, `_case_transport_core`, `_transport_case`); `scheduling.py` owns `run_cases` and runtime-plan selection; `chunking.py` owns spectrum chunk admission, including the larger portable EEDL working-set budget, and the bremsstrahlung photon-block width; `pool.py` owns host-memory and worker-pool sizing; `oom.py` owns accelerator OOM tags, release cadence, and pool limits. `step_budget.py` owns the per-electron transport step budget and `retry_step_budget`, the doubling retry after `TransportStepLimitError` shared by `_transport_case` and the penetration gate (#192, #315). `block_transport.py` owns the internal electron-block transport driver (`_transport_case(block_electrons=...)`): per-electron/CUDA transport over `[kB, (k+1)B)` joined bit-for-bit into the fixed-N result, the base for adaptive electron counts (#361). `adaptive.py` owns the adaptive electron count (`transport_case_adaptive` and the compatibility `run_case_adaptive`, #361): the sequential block-wise stopping rule on grid-independent per-electron line mass and brem band integrals, its heavy-tail guards, the realized fixed-N case, and optional per-bin batch means. `timing.py` owns the driver-side `PYRITE_MC_TIMING` phase aggregator and report. `case_tables.py` resolves each transport layer's SBETHE table and shell-mode catalog key. `artifacts.py` owns `_case_geometry` (case-determined grids, `n_hat`, groove) `spectrum_from_artifact`, which replays only the spectrum phase from a schema-2 trajectory artifact after checking its recorded inputs against the case, and `stream_spectrum_from_artifact`, its bounded-host-memory form over electron blocks (`STREAM_MAX_SEGMENTS`) (#186). `streamed_scoring.py` owns the shared additive `BlockSpectrumScorer` behind it and the opt-in live `run_case_streamed` (#611): electron-block transport through `_transport_case(block_consumer=...)`, each block scored then released, fixed-N normalization; `streaming_refusal` lists the cases it refuses before transport. `directions.py` evaluates physical-detector observation directions on the same transport as the scalar spectra (`run_case(observation_directions=...)`, `run_cases(observation_directions=...)`, `run_case_directions`) on a line grid that `cxr_grid.resolve_observation_cxr_grid` resolves jointly over every direction, separate from the scalar case grid. The backend classifier also recognizes delayed CUDA/ROCm runtime allocation failures. `_spectrum_case` always stores incoherent `spec` and, when requested, `spec_coherent` from the same transport, plus the shared characteristic component as the separate `spec_characteristic` (no array includes another; `_spectral_components` combines them); characteristic emission is evaluated directly on the fine line grid. `_cxr_pair_for_case` mirrors this for `pyrite reline`. Deps: `_backend`, `transport`, `geometry`, `spectrum`.
- Deps: `materials.crystal`, `materials.attenuation`, `DATA_DIR`.
