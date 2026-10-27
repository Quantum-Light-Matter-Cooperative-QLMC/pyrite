# Physical-quantity migration inventory (#755)

This is the **review checkpoint**, not an implemented quantity-output API. It
records current representation owners and the migration boundaries for #755.
The existing [unit policy](physical-units.md) and
[result/storage contract](result-storage-contracts.md) remain authoritative for
shipped behavior. Canonical facades and stability follow `api-surface.toml`;
implementation paths below do not promote private symbols to public API.

## Reading the inventory

- **Numeric** means a documented raw magnitude/array (legacy bare numbers stay
  in the specified units). **Q input** means a compatible Pint value currently
  normalizes to such a magnitude; it does **not** imply quantity-valued read-back.
- **Semantic Q** requires an explicitly unit-bearing argument *and* returns a
  fresh unit-bearing value. Only `Beam.kinetic_energy` is shipped so far.
- **Record** denotes the representation that must stay numeric and preserve its
  current fields, normalization, digests, and serialization.
- **Pending** names a public boundary still requiring its own quantity-valued
  read-back/replacement interface, or explicit approval to remain numeric. Do
  not interpret these entries as planned exact constructor/property spellings.

## Input boundaries

| Canonical entry / concrete owner | Physical controls; canonical numeric units | Accepted and read-back now | Record / consumer; migration disposition |
| --- | --- | --- | --- |
| `pyrite.beams.Beam` — `campaign/beam.py` | `energy_keV` keV; `rep_rate_hz` Hz; `bunch_charge_pc` pC; spread dimensionless | Q input -> numeric; `kinetic_energy` **Semantic Q** in keV; explicit Q-only replacement | `BeamSpec` / `Beam.to_spec`, sweeps, scene lowering and Case; retain numeric fields |
| `pyrite.beams.Spot`, `BunchLength`, `Offsets` — `campaign/beam.py` | `fwhm_x_mm`, `fwhm_y_mm` mm; `rms_fs`, `offsets_fs` fs | Q input -> numeric; no semantic Q read-back | beam distribution specs / sweep paths; pending |
| `pyrite.beams.GdfSource`, bunch-front tilt — `campaign/beam.py`, `montecarlo/bunch_front.py` | source position m, time s, tolerances m/s; tilt angular convention | Selected Q input -> numeric; correlated GDF particle energies are **not** `Beam.kinetic_energy` | beam-source normalization and transport admission; independently audit tilt paths |
| `pyrite.geometry.Slab`, `Layer`, `Footprint`, `BlazedGrooves` — `campaign/geometry.py` | thickness/period Å; tilt/azimuth degrees; footprint mm | Selected Q input -> numeric or numeric sweep grid | target lowering, `LayerSpec`, case and catalogs; pending |
| `pyrite.geometry.CrystalCut`, `Target` — `campaign/geometry.py`, `campaign/target.py` | roll/miscut/target orientation degrees; Miller indices integer | Selected Q input -> numeric; Miller/direction indices are not quantities | resolved orientation, target identity; pending |
| `pyrite.geometry.RectangularBody` — `bodies.py` | size/thickness mm | Q input -> numeric | physical body serialization and scene composition; pending |
| `pyrite.detectors.Detector`, `AngularAcceptance`, `SensorAcceptance` — `detectors/spec.py`, `detectors/acceptance.py` | observation/polar/azimuth degrees; aperture sr; sensor mm | Q input -> numeric; angle/sr require explicit type guards | scalar acceptance and result normalizations; pending |
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

## Existing output contracts (not all convertible by the same rule)

| Public value / owner | Axes, shape and current numeric measure | Important distinctions | Consumer / persisted owner |
| --- | --- | --- | --- |
| `Result.cxr`, `brem`, `characteristic`, `channeling`, `transition`; `SpectralDensity` — `results/model.py` | `photon_energy_eV` nodes; `density` photons/(incident electron·eV·sr) | CXR/extra lines share grid, brem owns wider grid; `normalization` is `source` or `observation-average` | `Result.to_dataframe`, `spectral_flux`; Case and provenance unchanged |
| `Result.to_dataframe()` — `results/model.py` | long Polars table, component-native **node** coordinates and per-eV/sr density | neither resamples nor sums components | plots, notebook and downstream table exports |
| `Result.spectral_flux(units=...)` — `results/model.py` | union node grid; `photon_energy_eV` **and** `photon_energy_keV` axes; value columns **per eV** | `units` is a *normalization selector*, not an energy-axis unit switch; interpolation, acceptance and mean-current scaling | table/plot consumers, not persistent source values |
| `DirectionalSpectrum` — `results/directional.py` | `angle_deg` point samples × `energy_eV` nodes; density photons/(incident electron·eV·sr); `supported` mask | grazing directions are NaN; angular samples are **not** sr bins; `rate()` adds current only | directional app, experimental facade |
| `SpatialResult` / `SpectralFactors` — `results/spatial.py` | tile × native energy; intrinsic density photons/(incident electron·eV·sr); filter attenuation mm⁻¹ | `spectra()` returns pixel-accepted **per-eV** density, with sr already multiplied; `average_density()` divides by total acceptance | bounded pixel materialization, observation store |
| `SpatialResult.image()` — `results/spatial.py` | pixel image, integrated photons/incident electron | integrates selected **nodes** with trapezoid; no remaining per-eV or per-sr | spatial plots; not an acquisition count image |
| `PixelMetadata`, `PixelRayMap` — `results/spatial.py` | pixel mm, degrees, sr; filter-path mm and tile indices | positional/directional coordinates, **not** a photon-density grid | pixel navigation and stored true spatial factors |
| `PhotonContinuumTally`, `PhotonEventTally` — `photon_observation.py`, `photon_event_observation.py` | `edges_eV` (N+1); `*_per_primary_eV` (N) aperture-integrated per-eV bin **means**; separate detected/unbinned integrals | no /sr; out-of-range/unbinned detections not silently placed in bins; event energy budgets eV/primary | portable tally artifacts and production `Result.aperture` |
| `Result.aperture_dataframe()` — `results/aperture.py` | edge intervals; per-primary (or time/current) densities and bin-integrated `_counts` | width multiplication uses **edges**, not center interpolation; `current_nA` scales rate, not units of energy alone | Polars table and result plotting |
| `AcquisitionBatch` / `PixelCountResult` — `instrument/acquisition.py`, canonical facade `pyrite.results` | `measured_edges_eV` (N+1); pixel × bin `expected` float, `realized` integer; under/over/below-cut channels | registered counts **after** response and exposure; not spectral density and not a rate | `Result.acquire`, `StoredObservation`, pixel plots |
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
3. First output slice: independent spectrum energy-axis/density conversion,
   with analytic invariance tests and **no** alteration of `spectral_flux`'s
   existing normalization selector. Follow with directional spectra, then
   binned aperture; measured counts and quantum/optical outputs are separate.
4. Update CLI, plots, marimo and exports only with declared metadata and
   explicit backward-compatibility tests; schedule deprecations separately.

Any newly discovered ambiguity in units, probability measure or normalization
is a **blocking review item**, not an invitation to guess a Pint dimension.
