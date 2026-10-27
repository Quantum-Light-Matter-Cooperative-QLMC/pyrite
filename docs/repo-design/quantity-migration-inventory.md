# Physical-quantity migration inventory (#755)

This is the **review checkpoint** and is supplemented by node-sampled
`SpectralDensity.quantify()`, directional views, binned aperture views,
selected-pixel `SpatialResult.quantify_spectra()` and integrated
`SpatialResult.quantify_image()`, acquisition counts, semantic
`PixelMetadata` geometry read-back, beam dimensions/timing, target/body lengths
and sensor-acceptance lengths.
It records remaining migration boundaries for #755.
The existing [unit policy](physical-units.md) and
[result/storage contract](result-storage-contracts.md) remain authoritative for
shipped behavior. Canonical facades and stability follow `api-surface.toml`;
implementation paths below do not promote private symbols to public API.

## Reading the inventory

- **Numeric** means a documented raw magnitude/array (legacy bare numbers stay
  in the specified units). **Q input** means a compatible Pint value currently
  normalizes to such a magnitude; it does **not** imply quantity-valued read-back.
- **Semantic Q** requires an explicitly unit-bearing argument *and* returns a
  fresh unit-bearing value. `Beam.kinetic_energy` supplies this construction
  contract, as do `BunchLength.rms_duration` and `Offsets.arrival_offsets`;
  `PixelMetadata` supplies copied quantity-valued output properties.
- **Record** denotes the representation that must stay numeric and preserve its
  current fields, normalization, digests, and serialization.
- **Pending** names a public boundary still requiring its own quantity-valued
  read-back/replacement interface, or explicit approval to remain numeric. Do
  not interpret these entries as planned exact constructor/property spellings.

## Input boundaries

| Canonical entry / concrete owner | Physical controls; canonical numeric units | Accepted and read-back now | Record / consumer; migration disposition |
| --- | --- | --- | --- |
| `pyrite.beams.Beam` — `campaign/beam.py` | `energy_keV` keV; `rep_rate_hz` Hz; `bunch_charge_pc` pC; spread dimensionless | Q input -> numeric; `kinetic_energy` **Semantic Q** in keV; explicit Q-only replacement | `BeamSpec` / `Beam.to_spec`, sweeps, scene lowering and Case; retain numeric fields |
| `pyrite.beams.TransverseSpot` / `Spot` — `campaign/beam.py` | `fwhm_x_mm`, `fwhm_y_mm` mm | `fwhm_x`, `fwhm_y` explicit Q-only construction, copied canonical-mm effective-width read-back, `with_fwhm_x` / `with_fwhm_y` replacement; `None` means round y | numeric beam distribution specs / sweeps and `dataclasses.replace`; implemented |
| `pyrite.beams.BunchLength`, `Offsets` — `campaign/beam.py` | `rms_fs`, `offsets_fs` fs | `rms_duration` scalar and `arrival_offsets` 1D array: explicit Q construction, copied canonical-fs read-back and `with_rms_duration` / `with_arrival_offsets` replacement; numeric paths retained | `Beam.to_spec` / `from_spec`, numeric duration/offset records, nested numeric sweeps; implemented with shape/order preservation and overflow/underflow guards |
| `pyrite.beams.GdfSource`, bunch-front tilt — `campaign/beam.py`, `montecarlo/bunch_front.py` | source position m, time s, tolerances m/s; tilt angular convention | Selected Q input -> numeric; correlated GDF particle energies are **not** `Beam.kinetic_energy` | beam-source normalization and transport admission; independently audit tilt paths |
| `pyrite.geometry.Footprint` — `campaign/geometry.py` | full x/y widths mm, scalar or sweep grid | `width`/`height` explicit Q-only construction, copied canonical-mm read-back, `with_width`/`with_height` replacement; numeric width/height fields retained | existing Cartesian sweep/lowering, case keys and `dataclasses.replace`; implemented |
| `pyrite.geometry.Slab`, `Layer` — `campaign/geometry.py` | slab/film thickness Å, scalar or sweep; layer azimuth degrees and slab tilt degrees | `thickness` explicit Q-only construction, detached canonical-Å read-back and `with_thickness` replacement; numeric thickness and sweep/case specs preserved; angular fields still numeric | geometry lowering, `LayerSpec` and cases; thickness implemented, orientation pending |
| `pyrite.geometry.BlazedGrooves` — `campaign/geometry.py` | groove spacing Å | `spacing` explicit scalar Q-only construction, fresh canonical-Å read-back and `with_spacing` replacement | numeric case groove spacing and dataclass field retained; implemented |
| `pyrite.geometry.CrystalCut`, `Target` — `campaign/geometry.py`, `campaign/target.py` | roll/miscut/target orientation degrees; Miller indices integer | Selected Q input -> numeric; Miller/direction indices are not quantities | resolved orientation, target identity; pending |
| `pyrite.geometry.RectangularBody` — `bodies.py` | size/thickness mm | `size` (two components) and scalar `thickness` explicit Q-only construction, detached canonical-mm read-back and `with_size` / `with_thickness` replacement | numeric body fields, target/filter lowering, sweep paths and serialization retained; implemented |
| `pyrite.detectors.Detector`, `AngularAcceptance` — `detectors/spec.py`, `detectors/acceptance.py` | observation/polar/azimuth degrees; aperture sr | Q input -> numeric; angle/sr require explicit type guards | scalar acceptance and result normalizations; semantic interfaces pending |
| `pyrite.detectors.SensorAcceptance` — `detectors/acceptance.py` | radial distance, full azimuthal width and full polar height mm | `distance`/`width`/`height` Q-only scalar inputs, fresh canonical-mm read-back and `with_distance`/`with_width`/`with_height` replacement | numeric fields, degree/sr `resolve()` tuple and detector/case lowering retained; implemented |
| `pyrite.geometry.PlanarPose`, `PixelGrid` — `_planar_geometry.py` | positions, offsets, pitches, active sizes, distances mm; pose angles degrees | Q input on documented construction fields -> numeric arrays/tuples | pixel rays and observation store; preserve geometry shapes |
| `pyrite.detectors.PlanarDetector`, `FilterPlate`, response models — `instrument/model.py`, `detectors/spec.py` | detector/filter mm; sensor thickness µm, bias V, response energy steps eV | Selected Q input -> numeric; some response call arguments remain bare eV | response models and observation identities; audit each model separately |
| `pyrite.observations.Acquisition` — `instrument/observation.py` | exposure seconds; measured edges/threshold eV | Q input -> numeric edges; `native`/`uniform` constructors have separate edge policies | `ResolvedObservation` and digested observation store; preserve native/bin distinction |
| `pyrite.config.Physics`, `Numerics`, `EnergyBins` — `_numerics.py`, `_energy_bins.py` | energy thresholds/grids eV; channeling depths Å; lattice origin Å | Selected Q input -> numeric | Case/recipe lowering and scientific identity; counts, seeds and switches are not Q |
| `pyrite.quantum.QuantumRegion`, `QuantumNumerics`, `TransverseStateSpec` — `quantum.py`, `quantum_state.py` | zone/step/offset Å; recipe energies eV, angles rad, reciprocal length Å⁻¹ | Selected Q input -> numeric; state JSON schema stays numeric | runtime solver adapters and portable recipe digest; pending |
| `pyrite.quantum.project_planar_source`, `propagate_planar_source`, `score_planar_source` — `planar_source.py` | kinetic energy eV, T K, incidence/aperture rad, path/depth Å, stopping eV/Å, photon grids eV | Selected Q input -> numeric solver arguments; outputs currently numeric | experimental solver boundary; explicit temperature and density semantics required |
| `pyrite.quantum.axial_diagnostic` — `montecarlo/channeling/axial_diagnostic.py` | `AxialDiagnosticConfig`: electron/threshold energies eV, T K, depth/step/radius Å, tilt rad | Numeric public configuration; quantify supported Q coverage field-by-field before adding semantic names | experimental diagnostic, distinct from production transport |
| Catalog/profile entry — `materials/_catalog_decode.py`, `_beam_detector_parse.py`, `_config_presets.py` | field-specific suffixed TOML keys, grid descriptors, filter and acquisition values | numeric or explicit unit strings -> canonical magnitudes | material/profile digests and TOML schema; no silent new keys or alias mergers |
| Advanced transport — `pyrite.transport` facade, `montecarlo/trajectory_summary.py` | initial energy keV; segment depths Å; histogram edges keV/Å/rad | Numeric low-level Case and capture contracts; scoped public adapters need separate review | capture artifacts/summary DataFrames; do not inject Q into transport buffers |
| Advanced radiation — `pyrite.radiation` facade, `materials/emission`, `montecarlo/spectrum` | line photons eV; geometry degrees; solid-angle weights sr | Numeric deterministic spectrum/line-energy contracts; experimental directional sampling | detector-free spectra must not inherit detector acceptance |
| Optical response — `pyrite.detectors.optical_energy.CalibratedPulseEnergyDetector` | photon energy eV, gain V/J, noise/saturation V; pulse energy J, power W | Numeric calibration/reading; quantity-valued view not yet specified | optical instrument calibration/provenance; **not** photon counting |

**Exceptions requiring a deliberate numeric contract:** unit vectors, Miller
indices, charge-state/particle IDs, angle-grid indices, statistical weights,
probabilities, emission fractions, polarization labels, sample counts, seeds,
backend selectors and digests. A bare `dimensionless` Pint value is not an
explicit angle or steradian. `kelvin` and offset temperatures must be reviewed
per factory; no implicit energy↔wavelength/frequency or thermal equivalency.

**Coverage verification gate:** before marking this inventory complete, compare
all supported physical fields in `api-surface.toml` and
`docs/reference/experimental-api.md` against these owner rows, inspect public
material constants/atomic result functions and remaining deterministic/optical
outputs, and record each uncovered item or a justified numeric exception. This
first checkpoint does not assert that a helper import alone proves unit support
for every signature in a module.

## Material values and constants: field-level audit

The entries in this section were checked against the concrete definitions,
not inferred from a module's helper imports. `pyrite.materials` schema/authoring
objects and the registered atomic/crystal/attenuation functions are
**advanced-stable**. Their existing numeric representation remains supported;
quantity interfaces below are **pending**, not approved numeric exceptions.
No new scientific API or stability promotion follows from this inventory.

For these rows, dimensional Python inputs and read-back are numeric; there is
no explicit semantic quantity interface. Catalog unit-string parsing does not
imply that direct schema constructors or scientific helpers normalize Pint
inputs. Accidental NumPy coercion of a quantity is not supported unit conversion.
Existing dataclass replacement uses stored fields. Catalog records, CIF-derived
arrays and authoring hashes remain numeric; adapters must copy read-back and
preserve those identities rather than serialize registries.

| Canonical value / concrete owner | Physical dimension, shape and meaning | Consumer / evidence; migration obligation |
| --- | --- | --- |
| `CrystalInfo.lattice`; `CrystalSpec.lattice` — `materials/_schema.py` | Mapping: present `a`, `b`, `c` in Å; `alpha`, `beta`, `gamma` in degrees. `system` is a selector, not a quantity. Absent lengths/angles follow the crystal-system convention, not new independent fields. | Reciprocal geometry and CIF loader; `tests/materials/test_crystal_lattice.py`. Convert physical entries individually; do not attach one unit to the mixed mapping. |
| `CrystalInfo.V_cell`; `CrystalSpec.V_cell` | Scalar Å³ unit-cell volume. | Composition, structure factors; `tests/materials/test_crystallography.py`. Quantity read-back must retain volume rather than length units. |
| `CrystalInfo.composition`; `CrystalSpec.composition`; `MediumSpec.composition` | Ordered `(element, density)` pairs; scalar atoms/Å³ per element. | `SubstanceRef.composition`, body/filter lowering, transport; `tests/materials/test_composition.py`. Preserve element ordering and semantic atom number; not charge density. |
| `CrystalInfo.basis`; `CrystalSpec.basis` | Ordered element labels and three-component **fractional** cell coordinates, dimensionless. | Structure-factor phases, CIF expansion; `tests/materials/test_crystal_external_db.py`. Deliberate numeric coordinate contract: these are not Cartesian Å positions. |
| `CrystalInfo.mosaic_fwhm_deg`; `CrystalSpec.mosaic_fwhm_deg` | Optional scalar angular FWHM in degrees. | Catalog defaults and mosaic transport; `tests/materials/test_material_catalog.py`. Missing width stays `None`; not zero and not arbitrary dimensionless input. |
| `CrystalSpec.B_ang2` | Scalar isotropic Debye–Waller B in Å². | Structure/coupling calculations; `tests/materials/test_crystallography.py`. Do not confuse B with an RMS displacement or temperature. |
| `CrystalSpec.hkl_families`, `hkl_list`, `surface_hkl`, `beam_uvw`, `cut`; `MaterialIdentity` | Integer indices, identity and cut/phase labels; no dimensional values. | Catalog orientation/identity; `tests/materials/test_material_catalog.py`. Numeric indices and semantic labels are justified exceptions, not Q inputs. |
| `ScanSpec` — `materials/_schema.py` | 1D `thickness_ang` Å, `energy_keV` keV, `tilt_deg`/`tilt_azim_deg` degrees, `E_grid_line`/`E_grid_brem` eV. `E_grid_line_by_energy` has **keV keys and eV grid values**. Optional layer counts and particle counts are dimensionless counts. | Material/profile resolution, Cartesian sweeps; `tests/materials/test_unit_profile_inputs.py`, `test_line_grid_isolation.py`. Preserve optional grids, exact node ordering and both key/value units. |
| `LayerSpec.thickness_ang`, `azimuth_deg` | Scalar Å thickness and degree rotation; `beam_uvw` is an integer direction. | Stack serialization/lowering; `tests/scan/test_target_geometry.py`. Numeric record stays distinct from the quantity-aware public `Layer`. |
| `MaterialSpec`, `MaterialCatalog`, `CATALOG`, `load_material_catalog`, `save_material` | Containers/selectors for the physical records above; no additional scalar unit for the container. `MaterialValidationSpec` and `MaterialConfigError` carry status/errors, not physical quantities. | `tests/materials/test_material_catalog.py`, `test_material_authoring.py`. Reconstruct quantities from each physical record after loading; no quantity-valued catalog pickle state. |
| `build_medium(..., density_g_cm3=...)`; `MaterialDefinition.density_g_cm3`, `composition` — `materials/authoring.py` | Scalar bulk density g/cm³; returned per-element atoms/Å³. Formula/atomic ratios and mass fractions are dimensionless composition semantics. | Authoring validation and transport composition; `tests/materials/test_material_authoring.py`. Density assignment/read-back pending; preserve existing normalized definition/hash. |
| `build_crystal(..., B_ang2=..., mosaic_fwhm_deg=..., density_g_cm3=...)` | Scalar Å², angular degrees and optional density g/cm³; CIF supplies physical cell geometry. Existing-crystal overrides remain refused. | CIF/catalog authoring; `tests/materials/test_material_authoring.py`. Quantity support must retain the existing derived-density consistency check and optional-value rules. |

`materials.crystal.CRYSTALS` and `pyrite.materials.CRYSTALS` expose the existing
numeric compatibility projection of lattice, basis, volume, composition and
crystal settings. They are not separate authoritative property owners.
`MATERIALS`/`MATERIAL_LABELS` and `materials.atomic.Z_TABLE` expose keys, display
labels and integer atomic numbers, respectively; these are not quantities.
Those compatibility names are not registered as new advanced-stable exports by
this inventory.

The publicly importable constants below are also **unregistered numeric
exports**, rather than governed API contracts. Keep their current scalar numeric
values and engine consumers; a future quantity constant interface requires a
canonical owner decision. Do not replace them with updated constants as a unit
migration side effect.

| Existing constant (`pyrite.materials.crystal`) | Numeric unit / physical meaning | Evidence / disposition |
| --- | --- | --- |
| `HC_EV_ANG`, `HBARC_EV_ANG` | eV·Å; h c and ħ c | `tests/materials/test_emission.py`, `test_crystallography.py`; quantity read-back pending. |
| `M_E_EV` | eV; electron **rest energy**, not mass in kg | Transport/line-kinematics consumers; `tests/materials/test_emission.py`; semantic quantity must state rest energy. |
| `R_E_ANG` | Å; classical electron radius | Absorption/susceptibility; `tests/materials/test_crystallography.py`; length quantity pending. |
| `E2_EV_ANG` | eV·Å; Gaussian electrostatic coupling, not SI charge squared | `U_g` / coupling tables; `tests/materials/test_crystallography.py`; retain convention explicitly. |
| `ALPHA_FS` | Dimensionless fine-structure constant | `E2_EV_ANG` and coupling consumers; justified numeric physical constant, no dimensional conversion needed. |

## Deterministic scientific and instrument calls: field-level audit

These registered helpers are **advanced-stable**. They currently expose numeric
coordinates/results, with no explicit quantity normalization or quantity-aware
replacement API. Outputs are not serialized Pint objects. Pure deterministic
helpers have no independent persisted quantity state; their consumers save the
existing numeric arrays/tables. The pending adapters must specify copies,
complex values, shapes and unsupported regions at each owner. Electron-unit
form factors represent scattering amplitudes relative to one electron; Pint's
`electron` charge unit would give them the wrong physical meaning.

| Canonical call / owner | Numeric input and output contract | Concrete evidence / required distinction |
| --- | --- | --- |
| `materials.atomic.cromer_mann_f0(element, g)` | Scalar/array `g` in Å⁻¹ using **2π/d**; real electron-unit amplitudes with the input shape. | `tests/materials/test_form_factors.py`; reciprocal length is not a Miller index or 1/d coordinate. |
| `materials.atomic.henke_dispersion(element, E_eV)` | Scalar/array eV → read-only real `f_prime`, `f_double_prime` arrays with matching shape, in electron amplitude units. | `tests/materials/test_form_factors.py`; preserve table-range NaNs or strict refusal, and real/imaginary sign convention. |
| `materials.atomic.atomic_form_factor(element, g, E_eV)` | Broadcast Å⁻¹ and eV inputs → complex electron-unit amplitude on the broadcast shape. | `tests/materials/test_form_factors.py`; no intensity/count or electron-charge unit. |
| `materials.crystal.reciprocal_g_vector(hkl, lattice)` | Integer triple plus mixed cell mapping above → three-vector and scalar magnitude in Å⁻¹, 2π/d convention. | `tests/materials/test_crystal_lattice.py`; coordinate frame stays Cartesian in the existing lattice basis. |
| `materials.crystal.structure_factor(..., photon_E_eV, B_ang2)` | eV photon energy, Å² B → complex electron-unit structure amplitude plus reciprocal magnitude Å⁻¹. | `tests/materials/test_crystallography.py`; preserve scalar/vector behavior of each returned component rather than broadcasting a wrapper from tuple shape. |
| `materials.crystal.chi_g(..., photon_E_eV, B_ang2)` | Scalar/array eV, scalar Å² → complex dimensionless susceptibility on the energy shape. | `tests/materials/test_crystallography.py`; dimensionless susceptibility is not probability or count. |
| `materials.crystal.U_g(..., photon_E_eV, B_ang2)` | Scalar/array eV, scalar Å² → complex potential-energy Fourier component in eV. | `tests/materials/test_crystallography.py`; energy, not voltage; ignored anomalous-correction option retains its documented behavior. |
| `materials.crystal.absorption_length_ang(element, photon_E_eV, number_density_per_ang3)` | Scalar/array photon eV, elemental atoms/Å³ → intensity attenuation length Å on the energy shape. | `tests/materials/test_crystallography.py`; preserve unsupported-energy NaNs and divergent zero-attenuation limits. |
| `materials.crystal.dominant_reflections(...)` | `E_ref_eV` eV, `B_ang2` Å², `g_max_invang` Å⁻¹; `n_families` is an integer count → ordered integer Miller triples. | `tests/materials/test_crystallography.py`; ranking returns indices, not quantity vectors. |
| `materials.attenuation.linear_attenuation_inv_mm(material, energy_eV)` | Nonempty positive 1D eV nodes → read-only mm⁻¹ narrow-beam coefficients on those nodes. | `tests/materials/test_attenuation.py`, `tests/instrument/test_attenuation.py`; neither probability nor an absorbed photon tally. |
| `radiation.photon_energies` / root `photon_energies` — `materials/emission.py` | Electron kinetic eV and four degree-valued geometry inputs → photon eV. Broadcast mode retains leading material axis; grid mode has material, energy, tilt, tilt-azimuth, observation-polar, observation-azimuth axes. | `tests/materials/test_emission.py`; preserve empty axes, duplicates, singleton axes and kinematic NaNs/zeros. No automatic energy↔wavelength equivalency. |
| `detectors.response.detector_efficiency(E_eV, polymer_nm, al_nm, grid_open)` | Photon eV, two scalar nm thicknesses, dimensionless open fraction → dimensionless efficiency on energy shape. | `tests/detectors/test_response.py`; efficiency does not apply an exposure or per-primary normalization. |
| `detectors.response.convolve_detector(E_grid_eV, spec, fwhm_eV)` | Energy-node eV, samples and Gaussian FWHM eV → same-shape density in the input units; optional below/above mass has `spec × eV` units. | `tests/detectors/test_response.py`; distinguish returned density from integrated outside mass, including leading array dimensions. |
| `detectors.grating.Grating` | Scalar `groove_density_per_mm` mm⁻¹, `alpha_rad` rad; integer order and dimensionless groove efficiency. Read-back: `d_angstrom` Å, `grazing_angle_rad` rad; photon-eV methods give diffraction rad, angular dispersion rad/Å, reflectivity/throughput dimensionless. | `tests/detectors/test_grating.py`; groove order is not a length; propagating support must remain aligned. |
| `detectors.grating.SimpleCCD` | Integer `n_pix`, scalar `pixel_mm`; `width_mm`, `pixel_edges_mm(center_mm)`, `pixel_centers_mm(center_mm)` return mm lengths/axes, N+1 edges or N centers. | `tests/detectors/test_grating.py`; centers are nodes, edges bound pixels; no energy response or exposure is implied. |
| `detectors.grating.disperse_spectrum(...)` | 1D photon eV and density/eV, distance mm, optional detector-normal rad → propagating positions mm and density/mm. | `tests/detectors/test_grating.py`; **explicit Jacobian remap**, not a Pint unit relabel; dropped nonpropagating rows and optional throughput weighting stay explicit. |
| `detectors.grating.detected_image(...)` | Same energy spectrum/geometry plus CCD and optional `center_mm` → N centers mm and integrated values per pixel. | `tests/detectors/test_grating.py`; counts inherit the input integral's normalization, not automatically detector events/exposure. |

### Explicit remaining coverage work

This audit expands material/constants and deterministic-helper coverage; it
does **not** close the inventory gate. The following supported surfaces still
need their complete constructor/method field matrix, ownership and executable
unit/shape evidence before any milestone-wide completeness assertion:

- Runtime quantum states/factories, solver results, scattering kernels and
  real-space arrays: include wavefunction measures, conditional normalization,
  temperature offsets and backend ownership, not only portable recipe fields.
- `TimepixResponse` / `EagleResponse` method inputs/results and native measured
  spectra: distinguish response matrices, per-eV densities and bin-integrated
  events. `Timepix3` normalizes `thickness_um` and `bias_v` directly;
  `Timepix3ResponseNumerics` normalizes `dE_mc` and `dE_out`. The flat
  `Timepix3` energy-width fields do not perform that same constructor check.
  Neither path proves unit support in `get_response` or every response method.
- Photon transport/experimental observation controls and event energy ledgers:
  `edges_eV`, cutoffs/ceiling and per-primary/track budget arrays must be audited
  separately from the implemented tally-density views.
- Optimization parameters/objectives (`FixedWindowYield.lower_eV/upper_eV`,
  path-selected bounds and objective units), observation sampling/plans,
  beam-capture metrics and remaining result reducers: do not infer dimensions
  from an arbitrary parameter path or a metric's array shape.
- CLI/profile fields, plot/table/export metadata and optical pulse/response
  calibration: retain explicit owner-by-owner audits and consumer links;
  provisional optical contracts must not be promoted through this inventory.

## Existing output contracts (not all convertible by the same rule)

| Public value / owner | Axes, shape and current numeric measure | Important distinctions | Consumer / persisted owner |
| --- | --- | --- | --- |
| `Result.cxr`, `brem`, `characteristic`, `channeling`, `transition`; `SpectralDensity` — `results/model.py` | `photon_energy_eV` nodes; `density` photons/(incident electron·eV·sr) | `quantify().to_energy(...)` provides independent Pint node/density quantities and semantic metadata; CXR/extra lines share grid, brem owns wider grid | `Result.to_dataframe`, `spectral_flux`; Case and provenance unchanged |
| `Result.to_dataframe()` — `results/model.py` | long Polars table, component-native **node** coordinates and per-eV/sr density | neither resamples nor sums components | plots, notebook and downstream table exports |
| `Result.spectral_flux(units=...)` — `results/model.py` | union node grid; `photon_energy_eV` **and** `photon_energy_keV` axes; value columns **per eV** | `units` is a *normalization selector*, not an energy-axis unit switch; interpolation, acceptance and mean-current scaling | table/plot consumers, not persistent source values |
| `DirectionalSpectrum` — `results/directional.py` | `angle_deg` point samples × `energy_eV` nodes; density photons/(incident electron·eV·sr); `supported` mask | `quantify().to_energy(...)` preserves component, mask and native axes; grazing directions are NaN; angular samples are **not** sr bins; `rate()` adds current only | directional app, experimental facade |
| `SpatialResult` / `SpectralFactors` — `results/spatial.py` | tile × native energy; intrinsic density photons/(incident electron·eV·sr); filter attenuation mm⁻¹ | `spectra()` returns pixel-accepted **per-eV** density, with sr already multiplied; selected-pixel `quantify_spectra()` couples native energy and per-energy density without rescoring; `average_density()` divides by total acceptance | bounded pixel materialization, observation store |
| `SpatialResult.image()` — `results/spatial.py` | pixel image, integrated photons/incident electron | existing numeric integration unchanged; `quantify_image()` exposes detached Pint-dimensionless per-primary yields and convertible energy-window labels, no remaining per-eV or per-sr | spatial plots; not an acquisition count image |
| `PixelMetadata`, `PixelRayMap` — `results/spatial.py` | pixel mm, degrees, sr; filter-path mm and tile indices | `PixelMetadata.local_position`, `center`, `distance`, `path_length`, `polar`, `azimuth`, `solid_angle` provide copied quantities; directions/indices and `PixelRayMap` remain numeric geometry, **not** a photon-density grid | pixel navigation and stored true spatial factors; numeric fields and replacement retained |
| `PhotonContinuumTally`, `PhotonEventTally` — `photon_observation.py`, `photon_event_observation.py` | `edges_eV` (N+1); `*_per_primary_eV` (N) aperture-integrated per-eV bin **means**; separate detected/unbinned integrals | experimental `quantify().to_energy(...)` converts edges/widths/densities, preserves per-bin and unbinned counts; no /sr and no event-energy-budget changes | portable tally artifacts and production `Result.aperture` |
| `Result.aperture_dataframe()` — `results/aperture.py` | edge intervals; per-primary (or time/current) densities and bin-integrated `_counts` | width multiplication uses **edges**, not center interpolation; `current_nA` scales rate, not units of energy alone | Polars table and result plotting |
| `AcquisitionBatch` / `PixelCountResult` — `instrument/acquisition.py`, canonical facade `pyrite.results` | `measured_edges_eV` (N+1); pixel × bin `expected` float, `realized` integer; under/over/below-cut channels | experimental `quantify().to_energy(...)` exposes measured-energy edges and exposure counts, preserving RNG draws and all channels; not a rate or density | `Result.acquire`, `StoredObservation`, pixel plots |
| `results.store`, `results.tables`, `results.metrics` legacy reductions | dictionaries `E_grid`, `E_grid_brem`, density/rate scalars; unit-labelled summary columns | independent grids; current-normalized versus beam-current columns; metric peak differs from integrated line yield | campaigns/checkpoints, faceting and CSV/report users |
| Axial quantum diagnostic — `montecarlo/channeling/axial_diagnostic.py` | depth Å; potential/spectral energies eV; density Å⁻², current/divergence Å⁻²/Å⁻³; dimensionless populations | survival/conditional normalization and probability current are not photon flux | experimental diagnostics and plotting |
| Optical pulse readings — `detectors/optical_energy.py` | J per pulse, V, W average, detector efficiency dimensionless | calibrated energy-to-voltage response, not photon counts; spectral coherency has separate dimensions | optical diagnostics/plots |
| Advanced capture/trajectory reductions — `montecarlo/trajectory_summary.py` | physical histogram edges with per-bin depth loss/energy and fate fractions | keV/bin is **not** dose; no mass/volume normalization | saved capture readers, transport validation |

For the relevant persisted identities see [result/storage contracts](result-storage-contracts.md).
Unit views must be reconstructed from numeric values and known metadata *after*
loading; they must not enter a checkpoint, CAS payload, worker record or hash.

## Consumers and executable evidence to bind to subsequent slices

| Owner | Existing focused checks / downstream consumers | New acceptance assertion before migration |
| --- | --- | --- |
| Beam/geometry/observations | `tests/instrument/test_unit_inputs.py`, `tests/scan/`, `tests/observations/test_spec.py` | units, shape, external registry, replacement and numeric identity |
| Footprint dimension quantities | `tests/scan/test_footprint_quantities.py`, `tests/scan/test_target_geometry.py` | implemented: Q-only semantic widths, sweep shape/order, numeric lowering, aliases, input/read-back isolation, representability and lazy imports |
| Slab/Layer thickness quantities | `tests/scan/test_target_thickness_quantities.py`, `tests/scan/test_target_geometry.py` | Q-only scalar/sweep thicknesses, conflict refusal, numeric case/LayerSpec invariance, external registries, read-back isolation, storage and lazy Pint imports |
| Sensor-acceptance lengths | `tests/detectors/test_sensor_quantities.py`, `test_acceptance.py`, `tests/instrument/test_unit_inputs.py` | implemented: Q-only scalar dimensions, external registries, copies, conflicts/replacement, storage limits, exact resolved acceptance/case identity and lazy numeric paths |
| Bunch timing | `tests/instrument/test_beam_timing_quantities.py`, `tests/scan/test_beam_objects.py` | implemented: explicit units, scalar/vector shape, signed offsets, dual-key refusal, external registries, representability, copies, lowering/loading and replacement |
| Profile/catalog | `tests/materials/test_unit_profile_inputs.py`, CLI profile commands | no TOML key drift, correct error path, conflict/alias refusal |
| Source scalar spectra | `tests/results/test_result_views.py`, `docs/reference/python-scene-recipes.md` | per-energy inverse scaling, exact native nodes, untouched legacy table |
| Aperture/secondary events | `tests/montecarlo/test_photon_production.py`, `results/aperture.py` | edges/width conversion and unbinned conservation, no second acceptance |
| Spatial/measurement | `results/spatial.py`, `instrument/acquisition.py`, `plots/_pixel_frames.py` | pixel shape/solid angle, bins/counts, deterministic Poisson identity |
| Directional/quantum/optical | `results/directional.py`, `docs/reference/quantum-api.md`, optical response and angular apps | masks/survival/measure semantics, no implicit Jacobian or unitless density |
| Compatibility/persistence | `tests/dev/test_api_surface.py`, `tests/results/`, `observations/store.py` | `api check --imports`, schema/digest equality, lazy numeric imports |

### Migration order (review before implementation)

1. Approve the [output conversion contract](quantity-output-contract.md) and
   fill any public-surface omissions above; do not introduce a generic view yet.
2. Migrate one **input owner** at a time, retaining each numeric dataclass
   field/serializer and introducing explicit semantic Q read-back/replacement.
3. First output slice **implemented**: independent `SpectralDensity.quantify()` node-axis/density
   conversion with analytic invariance tests and **no** alteration of
   `spectral_flux`'s existing normalization selector. Directional spectra are
   **implemented** with independent angular coordinates and support masks.
   Binned aperture tallies, selected-pixel spectral quantities, integrated
   pixel-image yields and measured-energy acquisition counts are also
   **implemented** as distinct views. Selected-pixel geometry read-back is
   **implemented** directly on `PixelMetadata`; raw `PixelRayMap` scoring
   buffers remain numeric. Quantum and optical outputs remain separate.
4. Update CLI, plots, marimo and exports only with declared metadata and
   explicit backward-compatibility tests; schedule deprecations separately.

Any newly discovered ambiguity in units, probability measure or normalization
is a **blocking review item**, not an invitation to guess a Pint dimension.
