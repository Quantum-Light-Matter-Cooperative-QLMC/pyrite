# Physical unit inputs

PyRITE accepts unit-bearing values at selected public configuration boundaries.
The boundary converts them to the field's existing numeric representation;
case payloads, transport buffers, result arrays, and checkpoint schemas remain
numeric. This keeps the established internal conventions while catching a
dimension mismatch before simulation.

## Public interface and dependency

Install the optional extra and use the one project registry:

```bash
pip install 'pyrite-mc[units]'
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
The issue's suggested `pint>=0.26,<0.27` range resolved to Pint 0.26.2 under
Python 3.14.4; PyRITE's supported floor is Python 3.13. Pint is installed only
by the `units` extra (and the test dependency group).

| Candidate | Evaluation |
| --- | --- |
| Pint | Chosen. A single registry is straightforward to share from `pyrite.units`; scalar and array quantities convert with `.to`, dimensional mismatches raise, and explicit strings can be parsed by that same registry. Pint's serialization format is not used for PyRITE records: normalized numbers keep checkpoints and identities independent of registry state. |
| unyt | Viable for NumPy-oriented scalar and array quantities, conversions, dimensional errors, custom units, and registries. Its array-first model is useful for scientific arrays; PyRITE needs boundary conversion for mostly scalar immutable configuration records and Python/TOML string parity. The Context7 entry for unyt returned an unrelated project, so the comparison used the [official unyt documentation](https://unyt.readthedocs.io/en/stable/usage.html). |
| Astropy Units | Viable for scalar and array conversion, custom units, and explicit equivalencies. It provides more astronomy-specific equivalency machinery than this API needs and carries the broader Astropy package context. See the [official Astropy unit conversion documentation](https://docs.astropy.org/en/stable/units/conversion.html). |
| Custom algebra | Rejected. It would require PyRITE to own parsing, conversion factors, dimensional validation, arrays, registry behavior, and serialization instead of keeping that machinery in a maintained dependency. |

Pint follows SI in treating radians as dimensionless. PyRITE therefore checks
that angle inputs use a recognized angular unit before conversion to a `*_deg`
field. Solid-angle fields require steradian units. Plain numbers retain each
field's existing convention, and arbitrary dimensionless quantities are
rejected for angle and solid-angle fields. No implicit energy/frequency or
other physical equivalencies are enabled.

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
