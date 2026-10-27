# Physical unit inputs

PyRITE accepts unit-bearing values at selected public configuration boundaries.
The boundary converts them to the field's existing numeric representation;
case payloads, transport buffers, result arrays, and checkpoint schemas remain
numeric. This keeps the established internal conventions while catching a
dimension mismatch before simulation.

## Public interface and dependency

Pint is a core runtime dependency. Use the one project registry:

```bash
pip install pyrite-mc
```

```python
from pyrite.units import Q_, ureg

energy = 60 * ureg.keV
duration = Q_(200, "fs")
```

`pyrite.units` exposes `ureg` and `Q_`; `import pyrite` does not import Pint.
The project registry is initialized once and not altered by PyRITE after
creation. Compatible quantities from another Pint registry are converted using
their own unit definitions. Pint quantities are accepted where a documented
physical field exists; arbitrary unit-bearing values are not stored in
configuration records.

## Library choice

Pint is the adapter dependency. Its supported unit registry and quantity model
cover scalar values, NumPy array magnitudes, common and compound units, custom
definitions, and explicit conversion. The implementation uses the documented
registry `.Quantity(...)`, quantity `.to(...)`, and `.magnitude` interfaces.
The supported range is `pint>=0.26,<0.27`; PyRITE's supported Python floor is
3.13. Pint ships with every installation, while imports remain lazy to keep
numeric configuration inexpensive. The empty `units` extra is retained for
compatibility with existing `pyrite-mc[units]` installation commands. Numerical
kernels and persisted records continue to contain ordinary magnitudes.

## Current normalization boundaries

| Input fields | Stored representation |
| --- | --- |
| `Beam.energy_keV`, `rep_rate_hz`, `bunch_charge_pc` | keV, Hz, pC; scalar or energy sequence |
| Beam spot widths, bunch duration/offsets, front angle, GDF time and position | Existing mm, fs, degrees, seconds, and meters |
| Target thickness/tilt, layer thickness/azimuth, groove period | Existing Å and degrees |
| `RectangularBody`, `Footprint`, planar detector poses/pixels, filters | Existing mm |
| Detector acceptance | Existing degrees and sr |
| `Physics` energy cutoffs, `Numerics` channeling depths, `EnergyBins` | Existing eV and Å |
| `QuantumRegion.zone_max_ang`, `QuantumNumerics.step_ang`, portable transverse-state lengths/energies/angles, planar source diagnostic inputs | Existing Å, eV, rad, K |
| `Acquisition` exposure, measured edges, threshold | Existing seconds and eV |
| Material scan grids and profile beam/detector/filter/response/acquisition fields | Existing field-specific units, resolved from numeric TOML values or explicit strings |

Bare numeric values, defaults, and serialized forms retain their prior values.
Quantity conversion happens once during construction or catalog parsing and
before case lowering. Same-field-equivalent values consequently feed the same
canonical numeric fields used by current scientific identity code. No rounded
identity canonicalization is applied.

The existing #680 quantum value objects and portable state specification accept
quantities at their current API owners; serialized transverse-state fields stay
ordinary JSON numbers and schema v1 is unchanged. This does not settle #704's
pending canonical import-map decision. No quantum recipes, solver admission, or
physical model changed.

The catalog parser accepts unit strings in physical grid descriptors and
profile fields. For example:

```toml
energy_keV = { values = ["60 keV", "0.08 MeV"] }
tilt_deg = "0.1 rad"

[profiles.demo.beam]
bunch_length_fs = "0.2 ps"
rep_rate_hz = "5 kHz"
bunch_charge_pc = "1 pC"

[profiles.demo.detectors.camera]
distance_mm = "10 cm"
polar_deg = "0.5 rad"
pitch_mm = ["55 um", "55 um"]

[profiles.demo.detectors.camera.acquisition]
exposure_s = "500 ms"
measured_edges_eV = ["0 eV", "0.1 keV", "200 eV"]

[physics.demo]
inelastic_model = "shell-soft-hard"
inelastic_cutoff_eV = "2 keV"

[numerics.demo]
channeling_dz_ang = "0.1 nm"
```

The TOML keys remain the catalog's existing unit-suffixed names; explicit
strings only choose a compatible input unit. Physics preset energy cutoffs and
lattice origins, plus Numerics channeling lengths, use the same parser and
canonical values as their Python configuration fields. Malformed strings and
dimension mismatches are reported at the offending catalog path. Numeric catalog
files do not import Pint.

## Serialization, results, and kernels

Quantities are normalized before scientific identities are constructed.
Checkpoint readers and writers do not require Pint, and quantities are never
added to low-level case payloads, Monte Carlo arrays, NumPy/Numba/CuPy kernels,
or distributed work items. PyRITE does not serialize Pint objects; this avoids
registry-dependent pickle state.

Numeric result schemas remain unchanged. Their unit contracts continue to be
described by the result types and API docs: source spectral densities are per
incident electron, eV, and sr; pixel/acquisition counts represent integrated
accepted events; rate and flux helpers apply their stated beam and geometry
normalizations. Angles and solid angles remain available through the existing
public degree/sr fields. No optional Pint wrapper is added to result arrays in
this slice. `Result.spectral_flux(units=...)` already provides named flux
normalizations (`detector`, `nA`, `sr`, and `electron`); the underlying source
components use distinct grids and spectral-density semantics, while pixel and
aperture results are integrated quantities. A generic quantity wrapper would
need a result-wide unit metadata map and could misrepresent those distinctions,
so this remains a separate result-contract design decision.

## Naming and representation policy

Choose a name from the value's representation, not the unit used by its caller:

| Representation | Naming rule | Example |
| --- | --- | --- |
| Public quantity carrying its unit | Semantic physical name; document whether energy is kinetic or total | `kinetic_energy`, `exposure` |
| Raw scalar/array with a prescribed unit | Explicit unit suffix | `energy_keV`, `exposure_s`, `thickness_ang` |
| Serialized magnitude | Unit suffix or mandatory, versioned unit metadata | Existing `energy_keV` keys |
| Dimensionless physical parameter | Semantic name and documented meaning | `energy_spread_frac` |

Quantity input support alone does not make a field quantity-valued. Today
`Beam.energy_keV` accepts compatible quantities but reads back a magnitude in
keV. Keep that name while accepting bare numbers. Do not rename it to `energy`
and silently retain an implicit keV convention. A semantic quantity interface
must specify both assignment and read-back, including array and missing-extra
behavior. Local equation symbols may follow mathematical notation when their
units are explicit in the routine's contract; boundary arguments and converted
intermediates should retain the information needed to distinguish conventions.

### Why Geant4 and PENELOPE use unsuffixed names

[Geant4's CLHEP unit definitions](https://github.com/Geant4/geant4/blob/master/source/externals/clhep/include/CLHEP/Units/SystemOfUnits.h)
are numeric scale constants: `millimeter = 1`, `centimeter = 10 * millimeter`,
`megaelectronvolt = 1`, and `kiloelectronvolt = 1.e-3 * megaelectronvolt`.
Long names have short aliases such as `mm`, `cm`, `MeV` and `keV`. Its
[unit guide](https://geant4.web.cern.ch/documentation/pipelines/master/bfad_html/ForApplicationDevelopers/Fundamentals/unitSystem.html)
prescribes explicit units when introducing dimensional inputs, converting
external data and formatting output. Existing internal values participate in
ordinary arithmetic without repeatedly attaching units. Internal bases include
mm, ns, MeV and positron charge, alongside temperature, amount, luminous
intensity and angular units.

The [particle gun](https://github.com/Geant4/geant4/blob/master/source/event/include/G4ParticleGun.hh)
stores `particle_energy`; the
[basic example](https://github.com/Geant4/geant4/blob/master/examples/basic/B1/src/PrimaryGeneratorAction.cc)
assigns `SetParticleEnergy(6. * MeV)`. These are ordinary doubles, so the
constants convert scale without enforcing dimensions at a C++ setter.

The [PENELOPE-2018 manual, §7.1.2/Table 7.3](https://www.oecd-nea.org/upload/docs/application/pdf/2020-10/penelope-2018__a_code_system_for_monte_carlo_simulation_of_electron_and_photon_transport.pdf#page=307)
lists double-precision track variables `E` in eV and `X`, `Y`, `Z` in cm;
`U`, `V`, `W` are direction cosines. This interface relies on fixed documented
units, without quantity objects or suffixes. This comparison concerns the 2018
core, not all subsequent versions or wrappers.

PyRITE should retain Pint's dimension checks at boundaries and numeric kernels.
Geant4's semantic naming is suitable for a quantity interface; adopting that
naming for PyRITE's mixed-unit magnitudes would hide their contract.

## Beam-energy quantity interface

The additive constructor keyword, property and replacement method retain `Beam`'s canonical import
from `pyrite.beams`, concrete owner in `campaign/beam.py`, and numeric
`energy_keV` dataclass field.

### Constructor and read-back contract

- Keyword-only `kinetic_energy` accepts a real Pint scalar quantity or a
  quantity with a one-dimensional real numeric array magnitude. Require explicit
  energy units: bare numbers, numeric arrays, strings, mixed component lists and
  dimensionless quantities are rejected. Lists remain available through the
  existing numeric `energy_keV` interface. Quantity inputs must be finite and
  positive, with nonempty arrays; scalar-scene versus sweep checks still apply
  at their current owners. Legacy numeric input validation is unchanged.
- An omitted-value sentinel distinguishes explicit `energy_keV` arguments. If
  neither argument is supplied, preserve the existing `(30, 45, 60)` keV default.
  If both are supplied, raise `TypeError` naming both arguments, even if their
  values are equivalent. An explicit legacy positional energy also conflicts.
  All other positional/keyword conventions and defaults are preserved.
- The supplied quantity normalizes once to the existing numeric keV field.
  The read-only `kinetic_energy` property returns a fresh quantity from the
  shared registry in keV. Thus read-back carries units but does not preserve
  the caller's original spelling, registry, or choice of MeV versus keV.
  For sequences, return a quantity with a copied one-dimensional array magnitude;
  mutating it must not mutate the beam. Scalar input reads back as a scalar.
- `energy_keV` stays numeric. `to_spec()` and `from_spec()` continue to use it;
  representation, dataclass fields, equality behavior and persisted records do
  not acquire a second energy value. The quantity property is not cached.
- Construction from external Pint registries uses the existing conversion
  boundary. Read-back uses PyRITE's unmodified registry. Numeric construction,
  lowering and record loading do not import Pint; requesting quantity read-back
  imports the core Pint dependency lazily. An incomplete installation missing
  Pint raises its normal import error; the property never falls back to a
  unitless value. No separate extra is required.

Examples:

```python
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.units import Q_, ureg

beam_options = dict(transverse=Spot(), longitudinal=PointBunch())
a = Beam(kinetic_energy=60 * ureg.keV, **beam_options)
b = Beam(kinetic_energy=0.06 * ureg.MeV, **beam_options)
assert a.energy_keV == b.energy_keV == 60
assert a.kinetic_energy.to("eV").magnitude == 60_000
grid = Beam(kinetic_energy=Q_([30, 45, 60], "keV"), **beam_options)

Beam(kinetic_energy=60, **beam_options)  # TypeError: units required
Beam(kinetic_energy=1 * ureg.mm, **beam_options)  # ValueError: expected energy
Beam(
    energy_keV=60, kinetic_energy=60 * ureg.keV, **beam_options
)  # TypeError: conflicting arguments
```

### Replacement, sweeps and serialization

Retaining the numeric dataclass field has an explicit consequence:
`dataclasses.replace(beam, energy_keV=80)` remains supported, while
`replace(beam, kinetic_energy=80 * ureg.keV)` is rejected. `replace` supplies the
existing `energy_keV` field as well as the new keyword, so the constructor must
report the conflict rather than guess which input wins. Unrelated replacements
and `replace(beam)` must preserve all existing values without loading Pint.

Use `beam.with_kinetic_energy(quantity)` for quantity-aware
replacement. It validates through the same new-input boundary, then replaces
only the numeric field; all other beam fields survive. An explicit numeric
alternative remains `replace(beam, energy_keV=quantity.to("keV").magnitude)`.
Do not add `kinetic_energy` as an `InitVar`: retaining or forwarding a second
energy input would make replacement ambiguous.

Array quantities describe the existing energy grid, not per-particle energies.
They lower to the current `BeamSpec.energy_keV` grid. Sweep path addresses
remain `beam.energy_keV`; scalar scene admission and GDF-derived correlated
energies retain their existing semantics. The property describes configured
energy, not a new override of sampled GDF particle energies.

`dataclasses.asdict`, beam/spec round-trips, observation records and checkpoint
cases retain their existing numeric keys and shapes. Do not serialize the
property, caller registry or supplied unit label. Quantity read-back may
allocate an array; it belongs in configuration/inspection, not kernel loops.
No new identity rounding is introduced: use tolerance-based comparisons for
general unit conversions and exact existing identity fixtures where already
guaranteed. Do not promise bitwise identity for all mathematically equivalent
floating-point expressions.

### Representative migration inventory

All current fields below accept compatible Pint inputs and return numeric
values. Strings are accepted by the catalog/preset parser, not generally by
Python constructors. The table identifies representation owners rather than
promoting every implementation module to public API.

| Canonical public input | Current accepted form and read-back | Numeric lowering/record owner | Pilot disposition |
| --- | --- | --- | --- |
| `pyrite.beams.Beam.energy_keV` | Scalar/sequence numbers or compatible quantities → keV | `campaign/beam.py::Beam.to_spec`, `campaign/sweep.py::BeamSpec`, `campaign/beam_cases.py` | Quantity keyword/property and explicit replacement method are available |
| `pyrite.beams.BunchLength.rms_fs` | Number or time quantity → fs | `campaign/beam.py::Beam.to_spec` | Keep numeric name; separate future quantity interface |
| `pyrite.geometry.RectangularBody.thickness_mm` | Number or length quantity → mm | `bodies.py`, `campaign/geometry.py` | Keep current body serialization and units |
| `pyrite.detectors.PlanarPose.center_mm`, `PixelGrid.pitch_mm` | Numeric components or compatible quantities, including array magnitudes → numeric mm components | `_planar_geometry.py`, persisted geometry in `observations/store.py` | Preserve coordinate and pixel representations |
| `pyrite.observations.Acquisition.exposure_s`, `measured_edges_eV` | Numbers/components or compatible quantities → s/eV | `instrument/observation.py`, persisted acquisition in `observations/store.py` | Preserve exposure and measured-bin meaning |
| `pyrite.config.Physics.inelastic_cutoff_eV`, `Numerics.channeling_dz_ang` | Number or compatible quantity → eV/Å | `_numerics.py`, preset parsing in `_config_presets.py` | Keep numeric controls and preset keys |
| `pyrite.quantum.TransverseStateSpec` recipe lengths/energies/angles | Numbers or compatible quantities → Å/eV/rad | `quantum_state.py` numeric JSON schema; runtime adapters in `montecarlo/channeling/states.py` | Preserve recipe-specific units and JSON identity |

Coverage remains selective. A helper import is not evidence that every physical
argument in its module accepts quantities. `BeamSpec` and low-level numerical
functions retain their documented numeric contracts; result arrays currently
have no generic Pint view. Any additional semantic field needs its own
assignment/read-back and serialization review.

Source anchors: {source}`src/pyrite/campaign/beam.py::Beam.to_spec`,
{source}`src/pyrite/_units.py::normalize_unit`,
{source}`src/pyrite/_planar_geometry.py`,
{source}`src/pyrite/instrument/observation.py`,
{source}`src/pyrite/observations/store.py`,
{source}`src/pyrite/_numerics.py` and
{source}`src/pyrite/quantum_state.py::TransverseStateSpec`.

Angle migration must retain explicit angular validation: for example current
`Detector(0.5 * ureg.radian)` converts to `observation_angle_deg`, whereas
`Detector(1 * ureg.dimensionless)` fails. Steradian inputs must remain distinct
from arbitrary dimensionless inputs. Do not add energy/frequency equivalencies
or change these guards as a naming side effect.

### Compatibility and verification

| Stage | Owner and deliverable | Required evidence |
| --- | --- | --- |
| Quantity boundary | `campaign/beam.py` | `tests/instrument/test_unit_inputs.py`: scalar/array equivalence, wrong dimensions, required units, conflicts, external registry, copied read-back arrays and blocked-Pint numeric subprocesses |
| Numeric consumers | `campaign/model.py`, `campaign/sweep.py`, `campaign/beam_cases.py`, `observations/spec.py` | Focused cases in `tests/scan/test_beam_objects.py`, `test_scene_model.py`, `test_sweep.py` and `tests/observations/test_spec.py`: replacement, unchanged defaults, numeric spec/record round-trips and existing identity fixtures |
| Dependency and public API | `pyproject.toml`, `uv.lock`, `docs/api.md`, beam docstrings, `api-surface.toml` | Core metadata in `tests/test_package_structure.py`; `pyrite-dev api check`, `api check --imports`, API tests in `tests/dev/test_api_surface.py`, and `pyrite-dev docs` |
| Further fields (future work) | Their existing geometry/observation/config/quantum owners | Review each new quantity-valued contract separately; quantum examples already have `tests/scan/test_quantum_runtime_units.py` and portable JSON checks |

The pilot keeps legacy numeric constructor arguments, properties, TOML keys and
sweep paths without warnings. It does not schedule their removal. If a later
review chooses to deprecate a spelling, register it in the existing API
deprecation machinery, announce it in the actual shipping release, and retain
the two-minor support window before removal. Release work owns version changes;
do not invent a parallel unit-migration schedule.

The bounded beam interface uses canonical keV read-back, lazy core Pint access
and the explicit `dataclasses.replace` restriction above. Expansion to further
fields requires a separate review;
preserving arbitrary input-unit labels would require a different storage and
compatibility proposal. A future result quantity view additionally needs
per-component grid/axis metadata, density versus integrated-measure semantics,
normalization and solid-angle contracts, and explicit conversion tests. It is
independent of this beam pilot.

## Measured overhead

A local Python 3.14.4 probe used seven fresh subprocesses for import timing and
peak RSS, and five batches of 5,000 `Beam` constructions for constructor
timing. Process startup is included in import wall time:

| Probe | Median |
| --- | ---: |
| `import pyrite` | 52.8 ms; 18.3 MiB peak RSS |
| `import pyrite; from pyrite.units import ureg` | 361.0 ms; 56.4 MiB peak RSS |
| Numeric `Beam` construction | 3.48 µs |
| Unit-bearing `Beam` construction (`0.06 * ureg.MeV`) | 44.67 µs |

These values describe this Python/container build, not a cross-platform
performance guarantee. The unit module is opt-in, and ordinary numeric
constructors retain the lazy fast path without importing Pint.

Scalar and array normalization were measured separately with
`uv run --extra units python checks/unit_conversion_overhead.py` on Python
3.14.4 and NumPy 2.5.4. Inputs and registry were constructed before timing;
each case used a 100-call warmup and seven repeats. Array cases converted 1,024
float64 energies from MeV to keV per call:

| Boundary operation | Median | Repeat range |
| --- | ---: | ---: |
| Numeric scalar pass-through | 0.254 µs/call | 0.253–0.257 µs/call |
| Pint scalar conversion | 15.352 µs/call | 15.326–15.371 µs/call |
| Numeric array pass-through (1,024 values) | 0.303 µs/call | 0.301–0.307 µs/call |
| Pint array conversion (1,024 values) | 18.146 µs/call | 18.082–18.295 µs/call |

The measured Pint array conversion rate was 56.4 million values/s. These
microbenchmarks isolate boundary normalization; they do not represent full
simulation throughput. Numeric inputs retain a pass-through fast path, and
conversion cost is paid once before numerical kernels.
