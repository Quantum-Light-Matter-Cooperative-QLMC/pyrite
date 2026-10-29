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
| `pyrite.beams.Beam` — `campaign/beam.py` | `energy_keV` keV; `rep_rate_hz` Hz; `bunch_charge_pc` pC; spread dimensionless | Q input -> numeric; `kinetic_energy` **Semantic Q** in keV; `repetition_rate` (Hz) and `bunch_charge` (pC) scalar nonnegative explicit Q construction, fresh read-back and `with_*` replacement | `BeamSpec` / `Beam.to_spec`, sweeps, scene lowering and Case; retain numeric fields; rate/charge implemented (`tests/instrument/test_beam_normalization_quantities.py`) |
| `pyrite.beams.TransverseSpot` / `Spot` — `campaign/beam.py` | `fwhm_x_mm`, `fwhm_y_mm` mm | `fwhm_x`, `fwhm_y` explicit Q-only construction, copied canonical-mm effective-width read-back, `with_fwhm_x` / `with_fwhm_y` replacement; `None` means round y | numeric beam distribution specs / sweeps and `dataclasses.replace`; implemented |
| `pyrite.beams.BunchLength`, `Offsets` — `campaign/beam.py` | `rms_fs`, `offsets_fs` fs | `rms_duration` scalar and `arrival_offsets` 1D array: explicit Q construction, copied canonical-fs read-back and `with_rms_duration` / `with_arrival_offsets` replacement; numeric paths retained | `Beam.to_spec` / `from_spec`, numeric duration/offset records, nested numeric sweeps; implemented with shape/order preservation and overflow/underflow guards |
| `pyrite.beams.GdfSource`, bunch-front tilt — `campaign/beam.py`, `montecarlo/bunch_front.py` | source position m, time s, tolerances m/s; tilt angular convention | Selected Q input -> numeric; correlated GDF particle energies are **not** `Beam.kinetic_energy` | beam-source normalization and transport admission; independently audit tilt paths |
| `pyrite.geometry.Footprint` — `campaign/geometry.py` | full x/y widths mm, scalar or sweep grid | `width`/`height` explicit Q-only construction, copied canonical-mm read-back, `with_width`/`with_height` replacement; numeric width/height fields retained | existing Cartesian sweep/lowering, case keys and `dataclasses.replace`; implemented |
| `pyrite.geometry.Slab`, `Stack`, `Layer` — `campaign/geometry.py` | slab/film thickness Å, scalar or sweep; layer azimuth degrees and slab tilt degrees | `thickness` explicit Q-only construction, detached canonical-Å read-back and `with_thickness` replacement; `Layer.azimuth` scalar explicit plane-angle Q construction, fresh degree read-back and `with_azimuth` replacement; `Slab`/`Stack` `tilt` and `tilt_azimuth` scalar/sweep plane-angle Q construction, fresh degree read-back and `with_tilt`/`with_tilt_azimuth` replacement; numeric fields and sweep/case specs preserved | geometry lowering, `LayerSpec` and cases; thickness, layer azimuth and target tilt implemented (`tests/scan/test_target_tilt_quantities.py`) |
| `pyrite.geometry.BlazedGrooves` — `campaign/geometry.py` | groove spacing Å | `spacing` explicit scalar Q-only construction, fresh canonical-Å read-back and `with_spacing` replacement | numeric case groove spacing and dataclass field retained; implemented |
| `pyrite.geometry.CrystalCut`, `Target` — `campaign/geometry.py`, `campaign/target.py` | roll/miscut/target orientation degrees; Miller indices integer | `CrystalCut.roll`/`miscut`/`miscut_azimuth` and `Target.tilt`/`tilt_azimuth` scalar explicit plane-angle Q construction, fresh degree read-back and `with_*` replacement; numeric `*_deg` fields retained; Miller/direction indices are not quantities | resolved orientation, target identity and `to_slab` lowering unchanged; implemented (`tests/scan/test_orientation_quantities.py`) |
| `pyrite.geometry.RectangularBody` — `bodies.py` | size/thickness mm | `size` (two components) and scalar `thickness` explicit Q-only construction, detached canonical-mm read-back and `with_size` / `with_thickness` replacement | numeric body fields, target/filter lowering, sweep paths and serialization retained; implemented |
| `pyrite.detectors.Detector`, `AngularAcceptance` — `detectors/spec.py`, `detectors/acceptance.py` | observation/polar/azimuth degrees; aperture sr | `Detector.observation_angle`, `AngularAcceptance.polar` / `azimuthal` / `solid_angle` Q-only construction, fresh degree/sr read-back and `with_*` replacement; the unspecified third acceptance member stays `None` | numeric detector fields, acceptance `resolve()`, scoring, response and case lowering preserved; implemented |
| `pyrite.detectors.SensorAcceptance` — `detectors/acceptance.py` | radial distance, full azimuthal width and full polar height mm | `distance`/`width`/`height` Q-only scalar inputs, fresh canonical-mm read-back and `with_distance`/`with_width`/`with_height` replacement | numeric fields, degree/sr `resolve()` tuple and detector/case lowering retained; implemented |
| `pyrite.geometry.PlanarPose`, `PixelGrid` — `_planar_geometry.py` | positions, offsets, pitches, active sizes, distances mm; pose angles degrees | Q input on documented construction fields -> numeric arrays/tuples; `PlanarPose.center` (3-component length), `PixelGrid.pitch` (2-component positive length) explicit Q construction, fresh mm read-back (`center`, `pitch`, `active_size`) and `with_center`/`with_pitch`; `from_observation` takes `distance`/`polar`/`azimuth`/`roll`/`offset` quantities. Normal/x-axis directions and pixel shape stay numeric | pixel rays and observation store; preserve geometry shapes; implemented (`tests/detectors/test_planar_quantities.py`) |
| `pyrite.detectors.PlanarDetector`, `FilterPlate`, response models — `instrument/model.py`, `detectors/spec.py` | detector/filter mm; sensor thickness µm, bias V, response energy steps eV | Selected Q input -> numeric; some response call arguments remain bare eV | response models and observation identities; audit each model separately |
| `pyrite.observations.Acquisition` — `instrument/observation.py` | exposure seconds; measured edges/threshold eV | Q input -> numeric edges; `exposure`, `measured_edges` and `hit_threshold` explicit Q construction (scalar positive time, 1-D energy edges, nonnegative scalar energy), fresh s/eV read-back and `with_*` replacement on `Acquisition` and `NativeAcquisition` (exposure/threshold); `uniform` adds `minimum`/`maximum`/`bin_width`; `native`/`uniform` keep separate edge policies | `ResolvedObservation` and digested observation store; preserve native/bin distinction; implemented (`tests/instrument/test_acquisition_quantities.py`), identities unchanged |
| `pyrite.config.Physics`, `Numerics`, `EnergyBins` — `_numerics.py`, `_energy_bins.py` | energy thresholds/grids eV; channeling depths Å; lattice origin Å | Selected Q input -> numeric | Case/recipe lowering and scientific identity; counts, seeds and switches are not Q |
| `pyrite.quantum.QuantumRegion`, `QuantumNumerics`, `TransverseStateSpec` — `quantum.py`, `quantum_state.py` | zone/step/offset Å; recipe energies eV, angles rad, reciprocal length Å⁻¹ | `QuantumRegion.zone_max_length` (optional) and `QuantumNumerics.step_length` scalar positive explicit length Q construction, fresh Å read-back and `with_*` replacement; numeric fields and lowering retained. `TransverseStateSpec` keeps Q input -> numeric; state JSON schema stays numeric | runtime solver adapters and portable recipe digest; region/numerics lengths implemented (`tests/scan/test_quantum_length_quantities.py`); recipe-parameter read-back pending (needs per-kind unit metadata) |
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
explicit angle or steradian. Absolute-temperature boundaries use
`normalize_unit(..., "kelvin", kind="temperature")`: K, degC, degF and degR
convert with offsets, while `delta_*` temperature differences are refused; no
implicit energy↔wavelength/frequency or thermal equivalency.

**Coverage verification gate (closed for accounting; see
[gate status](#coverage-gate-status)):** every supported physical field in
`api-surface.toml` and `docs/reference/experimental-api.md` has an owner row
below or an explicit justified numeric exception, including public material
constants, atomic/deterministic functions and optical outputs. A helper import
alone never proves unit support for every signature in a module; the field-level
audits record which signatures normalize Pint values and which are numeric-only.

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
| `build_medium(..., density_g_cm3=...)`; `MaterialDefinition.density_g_cm3`, `composition` — `materials/authoring.py` | Scalar bulk density g/cm³; returned per-element atoms/Å³. Formula/atomic ratios and mass fractions are dimensionless composition semantics. | Authoring validation and transport composition; `tests/materials/test_material_authoring.py`, `test_authoring_quantities.py`. Implemented: keyword-only `density` mass-density Q and `MaterialDefinition.density` g/cm³ read-back; canonical inputs keep definition/hash exact; numeric keyword refuses Q. |
| `build_crystal(..., B_ang2=..., mosaic_fwhm_deg=..., density_g_cm3=...)` | Scalar Å², angular degrees and optional density g/cm³; CIF supplies physical cell geometry. Existing-crystal overrides remain refused. | CIF/catalog authoring; `tests/materials/test_material_authoring.py`, `test_authoring_quantities.py`. Implemented: keyword-only `b_factor` (area, ≥0), `mosaic_fwhm` (angle, >0) and `density` Q inputs; derived-density check, existing-crystal override refusal and optional-value rules retained; numeric keywords refuse Q. |

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

## Remaining surfaces: field-level audit

These rows cover the remaining supported surfaces. Each row was checked
against the concrete definition. Unless a row says otherwise, read-back is the
stored numeric field or array (not necessarily a copy), there is no quantity-aware
replacement beyond `dataclasses.replace` on numeric fields, and nothing below is
persisted as a Pint object.

**Numeric-only** means the boundary has no `normalize_unit` call. Pint values
are then *unsupported*, whatever happens to them; the audit probes recorded as
**R**, **S** or **K** describe current accidental behavior, not a contract:
**R** refuses (Pint `DimensionalityError` from `float()` or a `Real` check),
**S** strips the magnitude through `np.asarray` with only a
`UnitStrippedWarning` (wrong whenever the unit is not the canonical one), and
**K** keeps the Pint object unconverted in a numeric field. **S** and **K**
cases were blocking review item B1; B1 is resolved by refusal (**E**: a
`TypeError` from `_units.reject_quantity` naming the field and its numeric unit),
so the B1 rows below record **E** with their regression tests. Other probe results
come from one-line audit calls against this branch, not committed tests; each
migration slice owes a regression test for the behavior it chooses.

### Runtime quantum states, solvers, kernels and real-space arrays

Canonical imports are `pyrite.quantum` (and `pyrite.experimental.quantum` for
experimental names). Runtime objects are **non-portable**: the portable,
digested form is `TransverseStateSpec` JSON; runtime arrays enter no checkpoint
or Case identity. The production planar recipe is the only admitted route. Units
throughout are Å, Å⁻¹, eV and rad (`states.py`, `realspace.py` module
contracts).

| Canonical surface · status — owner | Inputs: meaning, internal unit, accepted forms | Outputs: read-back, shape/axes, normalization | Evidence; disposition |
| --- | --- | --- | --- |
| `gaussian`, `hermite_gauss`, `laguerre_gauss`, `donut`, `bessel_gauss`, `airy`, `aperture_probe`, `from_array`, `translate` · advanced-stable — `montecarlo/channeling/states.py` | waist/scale/centre/spacing/origin/offset Å; `k_radial_inv_ang` Å⁻¹; `kinetic_energy_eV` eV; aperture semi-angles rad with the angle guard; aberration `C*` Å, `phi*` rad. Mode orders `m`, `n`, `ell`, `p`, `dimension` and Airy `truncation` are dimensionless selectors. **Q input** (scalar, array or tuple components) → numeric | Frozen state objects with numeric fields: `ApertureProbe.momentum`/`k_inner`/`k_outer` Å⁻¹, `center` Å, `SampledState.axes` Å, `SampledState.kinetic_energy_eV` eV metadata or `None`, `Translated.offset` Å. No semantic read-back or replacement | `tests/scan/test_quantum_runtime_units.py`, `tests/montecarlo/test_channeling_states.py`; Q input implemented, read-back pending |
| `from_abtem` · advanced-stable | Reads abTEM `Waves.sampling` (Å) and `.energy` (eV) as floats by abTEM's own convention; `origin_ang` normalizes through `from_array`. Leading ensemble axes become an equal-weight mixture | `SampledState` or `MixedState` | `tests/montecarlo/test_channeling_states.py`; external convention, numeric exception |
| `TransverseState.amplitude(kx, ky)`, `.wavefunction(x, y)`, `.centre`, `.k_extent`, `.k_resolution`; `SampledState`, `Superposition`, `Translated`, `MixedState`, `EntryState`; `superpose`, `mixture` · advanced-stable | Numeric-only tensor axes: `kx`, `ky` Å⁻¹; `x`, `y` Å. Superposition coefficients are complex dimensionless; mixture weights are nonnegative dimensionless and renormalized to sum 1 | `amplitude`: complex ψ̃ on the tensor grid `(nkx[, nky])` with `∫|ψ̃|² dᵈk/(2π)ᵈ = 1`, so dimension length^(d/2). `wavefunction`: ψ, length^(−d/2), copy of samples on matching axes. `centre` Å; `k_extent`/`k_resolution` Å⁻¹ per axis. `Superposition.norm` and mixture weights dimensionless | `tests/montecarlo/test_channeling_states.py`; future views must carry `d` and the `dᵈk/(2π)ᵈ` versus `dᵈx` measure, never infer dimension from array rank |
| `DensityMatrixSolver(profile, kinetic_energy_eV, kernel, margin_eV, n_grid)`, `.entry_lattice`, `.entry_state`, `.entry_from_plane`, `.propagate(...)` · experimental — `density.py` | Numeric-only: eV, eV margin; incidence/divergence rad; `thickness_ang`, `dz_ang`, `spatial_x_ang` Å; `stopping_eV_per_ang` eV/Å; `energy_edges_eV` transverse-energy edges eV; `record_every`, `n_grid` counts. Crystal temperature enters only through the numeric K argument of the planar profile builder | `DensityMatrixResult`: `z_ang` `(n_z,)` Å; `kinetic_energy_eV` `(n_z,)` eV; `band_populations` `(n_z, n_bands)`, `bound_population`, `trace`, cumulative `free/overflow/reprojection_flux` `(n_z,)` and scalar `entry_flux` are dimensionless fractions **per incident electron**; `free_flux_by_energy` is cumulative population per transverse-energy **bin** with out-of-range weight **clipped into the end bins**; `barrier_eV`, `cutoff_eV` eV; `final_density` `(n_cells, N, N)` complex dimensionless; optional `spatial_density_per_ang` `(n_z, n_x)` Å⁻¹, unconditioned, cell integral equals `trace` | `tests/montecarlo/test_channeling_density.py`, `test_channeling_spatial.py`; numeric exception for experimental solver; a binned view must flag clipped end bins |
| `MCWFSolver(solver).run(...)` · experimental — `mcwf.py` | Numeric-only as above plus `n_electrons`, `seed`, `first_electron` counts/seeds | `MCWFResult`: ensemble means **and SEMs** of populations `(n_z, n_bands)`; `trace` = fraction still channeled; cumulative flux fractions; `dechanneled_energy_eV` (transverse energy, NaN for overflow), `dechanneled_depth_ang` Å and `dechanneled_electron` index, aligned per event; `final_density` normalized per launched electron; `jumps` count | `tests/montecarlo/test_channeling_mcwf.py`; numeric exception; uncertainty must propagate with any future view |
| `AxialMCWF(dissipator, kinetic_energy_eV, potential_eV, backend)`, `.run(entry, depths_ang, n_electrons, step_ang, tilt_rad, ...)` · experimental — `axial_mcwf.py` | Numeric-only: eV; potential grid eV `(n_x, n_y)`; depths/step Å; tilt rad pair; `backend` `"cpu"`/`"cuda"` selector; batch/memory budgets are counts/bytes | `AxialMCWFResult`: `depths_ang` Å; `observables` `(n_depths, ...)` ensemble mean with removed electrons as zero; `surviving` fraction; `sink_depths_ang` Å; `jumps`, `jump_history` raw counts (÷`n_electrons` per incident electron). **Host arrays** returned for CUDA; callbacks own CuPy arrays only during the run | `tests/montecarlo/test_channeling_axial_cuda.py`, `test_channeling_axial_mcwf_return_flux.py`; numeric exception |
| `AxialDiagnosticConfig`, `axial_diagnostic`, `AxialDiagnosticResult` · experimental — `axial_diagnostic.py`, `axial_flux.py` | Numeric-only config: eV, K (`temperature_K`, absolute), Å depths/step/radius, rad tilt, `spectral_cutoff_eV`; counts, seeds, selectors (`mode`, `backend` CPU/CUDA, MCWF-only CUDA). **E** (B1): every unit-bearing or count field refuses Pint at construction, `tests/montecarlo/test_axial_numeric_only_inputs.py` | `density_per_ang2` Å⁻² per incident electron (integral = `surviving`); `density(conditional=True)` divides by survival and returns zero where nothing survives; `wavefunctions_per_ang` Å⁻¹ (coherent pure states only); `transverse_current_per_ang2` Å⁻² meaning probability per transverse length **per Å of depth**, divergence Å⁻³, contour side flux Å⁻¹; OAM and spectral populations are disk/projector populations per incident electron | `tests/montecarlo/test_axial_diagnostic.py`, `test_axial_state_diagnostics.py`, `test_axial_flux.py`; numeric exception; conditional and per-depth measures need explicit metadata (Pint would collapse Å⁻¹·Å⁻¹) |
| `RealSpaceGrid(extent_ang, shape)`, `SplitStepPropagator(grid, kinetic_energy_eV, potential_eV, absorptive_eV, tilt_rad)` · experimental — `realspace.py` | Numeric-only: supercell Å, sample counts, eV, eV potentials on the grid shape, rad tilt. **E** (B1): `RealSpaceGrid.extent_ang`, propagator energy/potentials and `tilt_rad` (including bare dimensionless and rad quantities) and `AxialMCWF.run` depths/step/tilt refuse Pint, `tests/montecarlo/test_axial_numeric_only_inputs.py` | `x`, `y` Å; `kx`, `ky` Å⁻¹ in FFT order; `area_ang2`, `cell_area_ang2` Å²; `sample`/`evolve`/`trajectory` return periodized φ in Å⁻¹ with `norm(φ) = ∫|ψ|² dA` (dropped Nyquist weight shows as norm < 1) | `tests/montecarlo/test_channeling_axial_dissipator.py`, `test_axial_flux.py`; numeric exception; B1 resolved (refusal) |
| `ScatteringKernel`, `tds_kernel`, `electronic_kernel` · experimental — `operators.py` | Numeric-only eV; `n_cells`, `max_order` counts; `beam_uvw`/`c_axis` integer/axis selectors. Temperature enters through the profile's Debye–Waller data | `period_ang` Å; `g_inv_ang`, `wavevectors()` Å⁻¹ `(n_cells, 2H+1)`; `momentum_kernel` `(n_cells, 2H+1, 2H+1)` Å⁻¹; `diagonal_coefficients`, `diagonal(x)`, `kept_diagonal(x)` Å⁻¹ (loss rate per depth); `absorptive_potential_eV(x)` eV; `hbar_v_eV_ang` eV·Å | `tests/montecarlo/test_channeling_operators.py`; numeric exception |
| `spatial_density(x_ang, density, coefficients, orders, kappa_inv_ang, period_ang)` · advanced-stable — `spatial.py` | Numeric-only Å, Å⁻¹, integer orders. **E** (B1): `x_ang`, `kappa_inv_ang`, `period_ang` quantities refuse instead of being read as Å | `(n_x,)` Å⁻¹ cell density **including survival weight**, no clipping/renormalization | `tests/montecarlo/test_channeling_spatial.py`; B1 resolved (refusal) |
| `PlanarStateProjection`, `project_planar_source`, `QuantumSetup.project_state`, `propagate_planar_source`, `score_planar_source` · experimental — `planar_source.py` | **Q input**: eV, K, rad, Å, eV/Å, photon eV grid, layer depths. `temperature_K` converts absolute K/degC/degF/degR with offsets (293.15 K tested) and refuses `delta_degC`/`delta_degF` differences (B2 resolved) | Projection fields numeric (eV, K, rad); `plane_density`/`subspace_density` complex dimensionless; `free_population` `(n_cells,)`, `outside_population` dimensionless entry flux; total population 1 within 2e-3. `propagate` → `DensityMatrixResult`; `score` → `ChannelingSpectrum`: photons/(eV sr electron) on `S + (n_E,)`, NaN Stokes where zero, `line_rates` Å⁻¹ | `tests/scan/test_planar_source_bridge.py`, `test_quantum_runtime_units.py`; implemented input boundary; output views pending |

### Detector responses and native measured spectra

Canonical imports: `pyrite.detectors` (`Timepix3`/`Timepix3Response`,
`EagleXO`/`EagleXOResponse`, `LegacyEDS`, `IdealPhotonCounter`,
`NativeSpectrum`, `NativeBinnedResponse`), `pyrite.config.Timepix3ResponseNumerics`,
and the module facades `pyrite.detectors.timepix_response` /
`eaglexo_response` (`TimepixResponse`, `EagleResponse`, `get_response`), all
advanced-stable. Three different measures must stay distinct: a
**response matrix** (dimensionless detected events per incident photon in an
input channel), a **per-eV density** (same flux units as the input) and
**bin-integrated event mass** (input density × eV).

| Canonical surface — owner | Inputs: meaning, internal unit, accepted forms | Outputs: read-back, shape/axes, normalization | Evidence; disposition |
| --- | --- | --- | --- |
| `Timepix3(thickness_um, bias_v, dE_mc, dE_out, n_mc, seed, numerics=...)` — `detectors/spec.py` | `thickness_um` µm and `bias_v` V: **Q input** → numeric. `numerics=Timepix3ResponseNumerics(...)`: `dE_mc`, `dE_out` **Q input** → eV, finite/positive checked; `n_mc`, `seed` integers. The flat `dE_mc`/`dE_out` fields are numeric-only: **E** (B1) `Timepix3(dE_mc=Q_(0.05, "keV"))` raises `TypeError` (use `numerics=`), and the flat fields get the grouped policy's finite/positive validation at construction; stored values keep their caller type | Numeric fields; `response_numerics` returns a fresh grouped policy. Flat fields form the response identity, pickle and checkpoint replay | `tests/observations/test_response_numerics_contract.py`; B1 resolved (refusal + validation) |
| `TimepixResponse(E_grid_eV, ...)`, `get_response(...)` — `timepix_response.py` | Numeric-only: fine node grid eV (finite, increasing, ≥ 0); `dE_mc`, `dE_out` eV; `thickness_um` µm, `bias_v` V; `n_mc`, `seed`. **E** (B1): quantity grids and hardware settings, and quantity energies to `detection_efficiency` and the module helpers, raise `TypeError` | Attributes: `E` fine nodes eV; `fine_edges`, `in_edges` (multiples of `dE_mc`), `out_edges_eV` eV; `E_in`, `E_out` bin centres eV; `R` `(n_out, n_in)` **response matrix**; `P_det`, `eps_abs` dimensionless on `E_in`; `mean_rec`, `fwhm_rec` eV; `sigma_diff_um` µm. `apply(spec)`: **per-eV density on the fine nodes**, input flux units, via coarse-bin integration then interpolation of the coarse density. `apply_native(spec)`: **event mass** per `out_edges_eV` bin, leading batch axes kept. `detection_efficiency(E)` dimensionless | `tests/detectors/test_timepix_native_rebin.py`, `tests/detectors/test_package.py`, `tests/detectors/test_response_numeric_only_inputs.py`; numeric exception. `get_response` returns a **shared cached instance** with mutable arrays, so any view must copy |
| `EagleXO(coating, resolve_energy, n_pix)`; `EagleResponse(E_grid_eV, ...)`, `get_response(...)` — `spec.py`, `eaglexo_response.py` | Numeric-only eV grid; coating selector; `n_pix` count. **E** (B1) as for Timepix grids | `qe`, `detection_efficiency` dimensionless on the grid; `apply`: per-eV density, same grid and flux units, optional `(below, above)` outside **mass** (density × eV); `charge_density`: input flux × collected **charge in units of e** per eV; `integrated_charge`: trapezoid over **nodes**, input flux × charge [e]; `FULL_WELL_E` also e. Numbers are charge in elementary charges (Pint `elementary_charge`/`e`, convertible to C/pC), not dimensionless counts | `tests/detectors/test_eag_wide_scaling_and_mosaic_fwhm.py`, `tests/detectors/test_response_numeric_only_inputs.py`; numeric exception; B6 resolved (docstrings; future read-back uses Pint `e`) |
| `LegacyEDS`, `IdealPhotonCounter`, `Detector.score(energy_eV, intrinsic_density, fwhm_eV, scale)` — `spec.py` | Numeric-only eV nodes, eV FWHM; `scale` dimensionless multiplier (acceptance or rate factor supplied by caller) | Scored **density** in the input's scaled units, same nodes | `tests/detectors/test_spec.py`, `test_response.py`; numeric exception |
| `NativeBinnedResponse.native_score`, `Detector.native_score`, `NativeSpectrum(edges_eV, events)` — `spec.py` | Numeric-only eV nodes and intrinsic per-eV density; `scale` | `NativeSpectrum`: read-only `edges_eV` `(N+1,)` half-open eV bins; read-only `events` `(..., N)` **event mass**, retaining the caller's normalization (e.g. per incident electron). Ideal counter uses midpoint cells clamped at 0 eV | `tests/instrument/test_acquisition_core.py`; consumed by `Acquisition`/`AcquisitionBatch` (already quantified downstream) |

Unregistered module functions — `timepix_response.sigma_diffusion_um`
(µm; `temperature_k` absolute K), `absorption_efficiency`, `energy_fwhm_eV`,
`build_response`, `poisson_counts`; `eaglexo_response.geometry` /
`sweep_geometry` (sr, degrees, m, mm), `qe`, `qe_absorption_model`,
`energy_fwhm_eV`, `poisson_counts` — are numeric module conveniences, not
governed APIs. They keep their documented units; a quantity interface requires
an owner decision first.

### Photon transport controls and event energy ledgers

Canonical imports: `pyrite.experimental.observations` (experimental) and
`pyrite.config.PhotonTransport` (experimental, through `Physics`).

| Canonical surface — owner | Inputs: meaning, internal unit, accepted forms | Outputs: read-back, shape/axes, normalization | Evidence; disposition |
| --- | --- | --- | --- |
| `PhotonTransport(edges_eV, source, n_samples, photon_cutoff_eV, electron_cutoff_eV, source_ceiling_eV, seed, charged_feedback)` — `_photon_config.py` | Numeric-only eV histogram edges (≥ 2, finite, nonnegative, increasing), cutoffs and ceiling (finite, positive, ceiling > photon cutoff for brem); `n_samples`, `seed` integers; `source` selector. Probe **R** for quantity edges or cutoffs | Normalized `edges_eV` tuple of float; numeric selection record embedded in the resolved Case `photon_transport` record and its digests | `tests/montecarlo/test_photon_production.py`; numeric exception (Case record) |
| `simulate_photon_continuum(scene, edges_eV, n_samples, photon_cutoff_eV, electron_cutoff_eV, source_ceiling_eV, seed, ...)`, `simulate_photon_events(scene, edges_eV, source, photon_cutoff_eV, electron_cutoff_eV, seed, ...)` — `photon_transport.py`, `photon_events.py` | Numeric-only. Cutoffs/ceiling: probe **R** (`np.isscalar` refusal). **`edges_eV`: E** (B1) — `edges_eV=Q_([1, 2], "keV")` (or a list of quantities) raises `TypeError` instead of being read as eV | `PhotonContinuumObservation`/`PhotonEventObservation.score` is the runner's numeric record: `edges_eV` `(N+1,)`, `*_per_primary_eV` `(N,)` photons/(incident electron eV) **aperture-integrated**, not per sr; `detected_per_primary` and `unbinned_detected_per_primary` per incident electron. Quantity read-back exists only after `save_*`/`load_*` as a tally `.quantify()` (output table above) | `tests/montecarlo/test_photon_scene_handoff.py`, `test_photon_event_observation.py`, `test_photon_production.py`; B1 resolved (refusal) |
| `PhotonEventTally.local_source_energy_eV`, `.tracked_source_energy_eV`; `PhotonRelaxationBudget` (`reserved_eV`, `escaped_eV`, `deposited_eV`, `binding_eV`, `emitted_eV`) — `photon_event_observation.py` | Loaded from artifacts, not user inputs | Each `(n_primaries,)` eV **per original primary** — absolute energy totals, not densities or detector energy. `tracked_source_energy_eV` sums `energy × weight`; `binding_eV` is a **signed** remainder; `emitted_eV` includes sub-cutoff photons; `reserved = escaped + deposited + binding + emitted`. Not converted by `quantify()` | `tests/montecarlo/test_photon_event_observation.py`; numeric record retained; a future ledger view must keep per-primary indexing and signs |

### Optimization, observation plans, beam metrics and result reducers

| Canonical surface · status — owner | Inputs: meaning, internal unit, accepted forms | Outputs: read-back, shape/axes, normalization | Evidence; disposition |
| --- | --- | --- | --- |
| `FixedWindowYield(lower_eV, upper_eV, name)` · experimental — `optimization/scoring.py` | eV window inside the stored node grid: plain eV numbers or **Q input** → eV at construction; `evaluation_id` hashes the lowered float bounds, so equivalent units give identical IDs | `{name: float}` window integral of the node-sampled PXR/CBS density; Pint unit `units = "1/sr"` (also `output_units`), counted normalization in the separate label `normalization = "photons per incident electron"`; no current, response or solid-angle factor | `tests/optimization/test_scoring.py`; B3 resolved |
| `ParameterSpace`, `Continuous`, `Integer`, `Categorical`, `apply_scene_parameters` · experimental (`Integer`, `ParameterSpace` registered) — `optimization/space.py` | Semantic quantity paths (e.g. `beam.kinetic_energy`, `target.thickness`) accept explicit quantity bounds, bound through `ParameterSpace.bind(scene)` to the owning property read-back unit; replacement delegates to `with_<name>`. Legacy suffixed paths retain numeric bounds/values and refuse quantities. Integer bounds/step must be exactly integral in the owner unit | Studies bind domains automatically; optimizers use `study.parameters` and persist plain numeric trials plus unit-bearing definition identity. Direct scene construction accepts numeric or compatible quantity candidates | `tests/optimization/test_optimization.py`, `test_persistence.py`, `test_units.py`; quantity parameter paths implemented |
| `Objective(name, direction, unit)`, `Constraint(name, relation, bound, unit)`, `OptimizationStudy`, `PyriteEvaluator`, `OptunaOptimizer` · experimental — `optimization/study.py` | `unit`: Pint unit expression or `Unit`, parsed at construction (unparseable refused; empty = dimensionless), caller's text stored so persisted definition digests are unchanged; `pint_unit` read-back. `Constraint.bound`: number in `unit` or **Q input** with matching root units → `unit`. `run` refuses objectives/constraints whose unit differs from the evaluator's declared `output_units` | Objective/constraint values are plain floats keyed by name | `tests/optimization/test_units.py`, `test_optimization.py`, `test_persistence.py` (numeric study resumes with equivalent quantity definition; pinned digest); B3 resolved |
| `ObservationPlan`, `PixelSampling`, `SweepObservation` · advanced-stable — `observations/plan.py` | `tilt_deg`, `tilt_azim_deg` degrees; `rep_rate_hz` Hz; `bunch_charge_pc` pC; numeric, copied from the resolved Case | `PixelSampling.rays` (`PixelRays`: `centers_mm` `(ny, nx, 3)` mm, `distance_mm`, `solid_angle_sr` `(ny, nx)`), `tile_index` `(ny, nx)` integer, `directions_lab` `(n_tile, 3)` solid-angle-weighted unit vectors; `directions_sample()` unit vectors. Read-only numeric geometry; quantity read-back is `PixelMetadata` | `tests/observations/test_sweep.py`, facade tests; numeric exception (internal scoring geometry) |
| `capture_beam_metrics(path, beam, max_particles)` · advanced-stable — `campaign/beam_metrics.py` | Trajectory artifact path; `beam.bunch_charge_pc` pC and `rep_rate_hz` Hz; count | `(BeamMetrics, sampled, available)`. `PlaneMetrics`: `position_sigma_mm` mm, `slope_sigma_rad`, geometric/normalized emittance mm·rad, `alpha`, `beta_mm_per_rad`, `gamma_rad_per_mm`; NaN Twiss for zero emittance. `BeamMetrics`: `sigma_t_fs` fs, `sigma_z_mm` mm, `sigma_delta` dimensionless, `longitudinal_emittance_fs` fs, `bunch_charge_pc` pC, peak and average current A (peak is a Gaussian-equivalent estimate, `inf` for zero duration) | `tests/montecarlo/test_beam_metrics.py`; B4 resolved: `slope_sigma_rad` is the RMS trace-space slope `u_x/u_z` reported in rad by accelerator convention (≈ angle only paraxially); emittance mm·rad, beta mm/rad, gamma rad/mm |
| `capture_secondary_energy_balance(path, max_rows)` · advanced-stable — `montecarlo/trajectory_balance.py` | Artifact path; row count | DataFrame `term`, `energy_keV`: ledger totals summed over every captured primary (not per primary, not dose) | `tests/montecarlo/test_trajectory_summary.py`; numeric table exception |
| `radiation_band_metrics(record, band_eV)` · advanced-stable — `results/radiation.py` | Numeric eV band inside the line grid | Window/full-grid integrals photons/(sr incident electron) per component (signed values retained); `transition_fraction`, `negative_transition_bins`; `physical_electrons` count from pC; `sampled_histories` | `tests/results/test_line_metrics_pinned.py`; numeric exception |
| `cxr_metrics(r, settings, ...)` · advanced-stable — `results/metrics.py` | Record arrays plus `Settings`; prominence and `n_fwhm` dimensionless | `*_per_na` values use `scale = domega_sr × electrons/(s nA)`: peak density photons/(eV s nA), integrated values photons/(s nA); unsuffixed values × case current → photons/(eV s) or photons/s; `line_eV`, `fwhm_eV`, `peak_sample_spacing_eV` eV; ratios/quality dimensionless. `peak_flux`/`peak_flux_per_na` are **deprecated** aliases of `peak_spectral_flux_density(_per_na)` (a density; removal scheduled in `_api_deprecations`); `coherent_flux*` are energy-integrated fluxes | `tests/results/test_line_metrics_pinned.py`, `test_results.py`; B5 resolved |
| `summary_table(recs, settings)` · advanced-stable — `results/tables.py` | Records plus `Settings` | pandas table; unit lives only in column labels (`[deg]`, `[keV]`, `[eV]`, `[Phs/eV/s/nA]`, `[Phs/eV/s]`, `[Phs/s/nA]`, `[Phs/s]`), rounded. Peak columns are `LegacyEDS`-scored densities; `line/brem/total [Phs/s]` integrals are source-side photons into the aperture (no QE/convolution), formerly mislabelled `[cts/s]` | `tests/results/test_results.py`; B5 resolved (label change) |
| `records`, `filter_results`, `select_results`, `best_azimuth` · advanced-stable — `results/selection.py` | `select_results` constraints compare numeric case fields in their **suffix units** (`tilt_deg`, `thickness_ang`, `E0_keV`) with float tolerance; Pint values unsupported | Record lists/stores, unchanged values | `tests/results/test_results.py`; numeric exception |
| `Settings` (`pyrite.results`), `Analysis` (root, deferred) · advanced-stable — `results/store.py`, `campaign/model.py` | `beam_current_na` nA fallback; `inelastic_cutoff_eV`, `secondary_threshold_eV`, `radiative_cutoff_eV` eV; `max_dE_frac` dimensionless; counts and selectors. No `normalize_unit` | Numeric fields used only at read time | `tests/results/test_results.py`; numeric compatibility exception (canonical inputs are `Physics`/`Numerics`) |
| `Precision`/`AdaptiveSampling`, `FixedSampling`, `TransportNumerics`, `RadiationNumerics`, `KernelTuning`, `CxrReflections` · advanced-stable — `_precision.py`, `_sampling.py`, `_numerics_groups.py`, `_numerics.py` | `band_eV`, `batch_means_band_eV` numeric eV pairs (probe **R**); `target_rse`, shares/fractions and `max_dE_frac` dimensionless; electron/block/family counts | Numeric fields in Case and precision identity | `tests/cli/test_profile_precision.py`; numeric exception for these config fields (completes the `Physics`/`Numerics` row) |

### CLI, profiles, plots, exports and optical calibration

| Owner | Physical fields and accepted forms | Read-back / metadata | Evidence; disposition |
| --- | --- | --- | --- |
| CLI options — `cli/commands/_beam_shared.py`, `_detector_shared.py`, `_pixel_detector_shared.py`, `_filter_shared.py`, `_profile_filters.py`, `_profile_precision.py`, `profile.py`, `energy_grid.py`, `blaze.py`, `tables.py`, `material_add.py`, `beam.py` | Bare numbers only (Click `float`/`FloatRange`); the unit is fixed by the option name or help: `-mm`, `-deg`, `-ev`, `-um`, `-v`, `-s`, `-m`, `-fs`, `-pc`, `-hz`, `-ang`, `--solid-angle` sr, `--energy` keV (profile/energy-grid) or eV (`tables`), `--t1-max` MeV, `--density` g/cm³, `--emittance` mm·mrad, `--twiss-beta` m, `blaze --spacing` m (converted to Å). No Pint or unit strings on the command line | Values are written to profile TOML in the canonical suffixed key or passed numerically; `docs/repo-design/cli/cli-reference.md` documents help | `tests/cli/`; numeric CLI exception — changing it is a CLI contract change outside this inventory |
| Profile/catalog TOML — `materials/_catalog_decode.py`, `materials/_beam_detector_parse.py`, `_config_presets.py` | Numbers in the key's unit, or explicit unit strings through `normalize_unit_input` → canonical magnitudes | Numeric profile records and digests | `tests/materials/test_unit_profile_inputs.py`; implemented (catalog/profile input row) |
| Plots — `pyrite.plots` (public domain, **unregistered** in `api-surface.toml`), `plots/_frames.py`, `_pixel_frames.py`, `_common.py` | Numeric record arrays and `Settings` | Units only in axis/legend label strings (e.g. `[mm]`, `[deg]`, `[sr]`, `[Phs/eV/s/nA]`); constants `C_ANG_PER_FS`, `SI_K_EDGE_EV` are unregistered numeric exports | `tests/plots/`; presentation-only numeric exception; labels follow the owning reducer |
| Exports — `Result.to_dataframe`, `aperture_dataframe`, `apps/configuration_exports.py`, `pyrite app analysis export` | Numeric tables with unit-suffixed column names; HTML export renders the app | Machine-readable unit metadata is the column naming convention only | `tests/results/test_cxr_api.py`, `test_quantity_aperture.py`, `tests/notebooks/test_configuration_app.py`; pending consumer/I/O migration |
| `CalibratedPulseEnergyDetector(energy_eV, efficiency, pulse_gain_v_per_j, calibration_id, noise_rms_v, saturation_v)`; `predict`, `predict_batch(energy_eV, collected_photons_per_eV_per_pulse, repetition_hz)` · provisional, unregistered — `detectors/optical_energy.py` | Numeric-only: knots eV (increasing), `efficiency` dimensionless **energy-absorption fraction** in [0, 1] (not photon QE), gain V/J, noise/saturation V, repetition Hz; photon density photons/(eV pulse) already collected | `OpticalPulseReading` host scalars / `OpticalPulseBatchReading` arrays on the input backend with shape `density.shape[:-1]`: incident/absorbed J per pulse, expected/reported V, noise RMS V, saturation flag, average W (only with repetition), `response_sha256` | `tests/detectors/test_optical_energy.py`, `tests/montecarlo/test_optical_cuda_parity.py`; numeric; provisional contract not promoted here |
| `OpticalResponse`, `gold_rakic_response`, `gold_johnson_christy_response`, `fused_silica_franta_response` · provisional — `materials/optical.py`; `optical_elements`, `optical_interferometer`, `optical_ctr_inference` module APIs | eV segments, dimensionless relative permittivity, `temperature_K` absolute K or `None`; Jones transfers dimensionless; coherency photons/(eV pulse); CTR RMS duration | Numeric records with digests (`to_record`/`from_record`) | `tests/detectors/test_optical_*.py`; numeric; provisional, not promoted |

### Blocking review items

Each item needs a human or owner-slice decision before the named surfaces gain
quantity read-back or refusal. None is resolved by guessing a Pint dimension.

- **B1 — silent acceptance at numeric-only boundaries. Resolved: refusal.**
  Decision: these boundaries error. `_units.reject_quantity` raises `TypeError`
  naming the field and numeric unit for `edges_eV` of
  `simulate_photon_continuum`/`simulate_photon_events`, Timepix/Eagle response
  grids, settings and module helpers, `spatial_density`, flat
  `Timepix3.dE_mc`/`dE_out` (now also finite/positive at construction),
  `RealSpaceGrid.extent_ang`, `SplitStepPropagator`/`AxialMCWF.run` tilts
  (bare dimensionless included) and `AxialDiagnosticConfig`. Plain numbers are
  unchanged. Tests: `tests/montecarlo/test_photon_scene_handoff.py`,
  `tests/detectors/test_response_numeric_only_inputs.py`,
  `tests/observations/test_response_numerics_contract.py`,
  `tests/montecarlo/test_channeling_spatial.py`,
  `tests/montecarlo/test_axial_numeric_only_inputs.py`. Angle/solid-angle
  `normalize_unit` kinds now classify by Pint root units (steradian is no
  longer accepted as a plane angle).
- **B2 — offset temperatures. Resolved: refuse differences.**
  `normalize_unit(..., kind="temperature")` (also via `normalize_unit_input`
  strings) raises `ValueError` for any Pint `delta_*` unit and converts
  absolute K/degC/degF/degR with offsets (20 degC → 293.15 K).
  `project_planar_source.temperature_K` (and `QuantumSetup.project_state`,
  which delegates) is the only quantity-accepting absolute-temperature
  boundary. Tests: `tests/instrument/test_unit_inputs.py`,
  `tests/scan/test_planar_source_bridge.py`.
- **B3 — optimization units. Resolved: quantized.** `Objective.unit`/
  `Constraint.unit` are parsed Pint units (unparseable text refused; empty is
  dimensionless; caller text kept for identity); `Constraint.bound` and
  `FixedWindowYield` bounds accept compatible quantities, lowered to the
  declared unit/eV so `evaluation_id` and persisted definitions are
  byte-identical. `FixedWindowYield.units == "1/sr"` with the semantic
  `normalization` label; `OptimizationStudy.run` refuses objective/constraint
  units that differ from the evaluator's `output_units`. Semantic parameter
  paths derive units from owner quantity properties and use their `with_*`
  methods; legacy numeric paths continue to refuse quantity bounds/values.
  Bound numeric domains and owner unit strings determine persisted identity.
  Tests: `tests/optimization/test_units.py`, `test_scoring.py`,
  `test_persistence.py`. Old free-text objective units (for example the former
  `"photons/(sr incident electron)"`) no longer construct, so studies persisted
  with them cannot be resumed.
- **B4 — trace-space slope. Resolved: convention documented.**
  `PlaneMetrics.slope_sigma_rad` is the RMS trace-space slope
  `x' = dx/dz = u_x/u_z` (`tan θ`, ≈ θ only paraxially), reported in rad by
  accelerator convention; emittance mm·rad, beta mm/rad and gamma rad/mm follow
  it. It is not an angular spread for strongly scattered/non-paraxial beams.
  Future quantity read-back: rad, mm·rad, mm/rad, rad/mm. Docstring only; no
  numeric or name change (`campaign/beam_metrics.py`).
- **B5 — reducer labels. Resolved: source-side photon labels.**
  `summary_table` integrals are relabelled `line/brem/total [Phs/s]` and
  `[Phs/s/nA]` (source-side photons into the aperture, no QE/convolution); its
  peak columns stay `LegacyEDS`-scored densities, as the docstring and
  `show_summary` footnote state. Numeric values are unchanged; readers keyed on
  the old `[cts/s]` column labels must update (advanced-stable table-label
  change). `cxr_metrics` comments now separate the deprecated density aliases
  `peak_flux`/`peak_flux_per_na` (replacements
  `peak_spectral_flux_density(_per_na)`, removal scheduled in
  `pyrite._api_deprecations`, docstring-documented because dict keys cannot
  warn) from the correctly named integrated `coherent_flux*`.
- **B6 — Eagle charge. Resolved: charge in units of e.**
  `charge_density`/`integrated_charge` and `FULL_WELL_E` are collected charge
  expressed in elementary charges e (Pint `elementary_charge`/`e`, convertible
  to C/pC), not dimensionless counts. Numeric returns are unchanged; docstrings
  in `detectors/eaglexo_response.py` state the unit, and any future quantity
  read-back uses Pint `e`.

### Coverage gate status

The inventory coverage gate is **closed for accounting**: every supported
physical input and output in `api-surface.toml` and the experimental API
reference has an owner row above or an explicit, justified numeric exception,
and unregistered numeric exports are named as such. B1–B6 are **resolved**
(decisions and tests above). Optimization parameter paths now use their
owning semantic quantity properties for units and replacement. Remaining Plan
work (semantic read-back/replacement, output views, consumer metadata) is
migration, not inventory.

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
| Optical pulse readings — `detectors/optical_energy.py` | J per pulse, V, W average; calibration `efficiency` is a dimensionless energy-absorption fraction, not photon QE | calibrated energy-to-voltage response, not photon counts; spectral coherency has separate dimensions | optical diagnostics/plots |
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
