# Migrate scripts to physical quantities

Scientific deterministic helpers and grating/detector operations have separate
opt-in adapters, described in the [scientific quantities guide](issue-816-scientific-quantities.md).
They do not alter the numeric solver or catalog contracts.

Use this guide to move an existing numeric script to explicit Pint quantities.
You do not have to migrate: every numeric spelling keeps its meaning, raises no
warning and has no scheduled removal. Exact fields and units are listed in the
[units reference](../reference/python-api/units.md); the design contract is in
[physical units](../repo-design/physical-units.md).

## Assign quantities with semantic names

Each unit-suffixed numeric field has a keyword-only semantic spelling that
requires a Pint quantity. Use the shared registry from `pyrite.units`:

```python
from pyrite import Scene
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.config import Physics
from pyrite.geometry import Slab
from pyrite.units import Q_, ureg

beam = Beam(
    kinetic_energy=60 * ureg.keV,  # was energy_keV=60
    repetition_rate=5 * ureg.kHz,  # was rep_rate_hz=5000
    bunch_charge=1 * ureg.pC,  # was bunch_charge_pc=1
    transverse=Spot(),
    longitudinal=PointBunch(),
)
target = Slab(
    "hopg",
    thickness=2 * ureg.um,  # was thickness_ang=20_000
    tilt=45 * ureg.degree,  # was tilt_deg=45
    width=5 * ureg.mm,
    height=5 * ureg.mm,
)
physics = Physics(inelastic_model="shell-soft-hard", inelastic_cutoff=Q_(50, "eV"))
scene = Scene(beam, target)
```

Any compatible unit works; the value converts once to the existing numeric
field. Supplying both spellings of one field raises `TypeError`, even when the
values agree. A semantic name refuses a bare number, the wrong dimension, and
a dimensionless value where an angle or solid angle is expected:

```python
Slab("hopg", thickness=2000, footprint=footprint)  # TypeError: needs a length quantity
Slab(
    "hopg", thickness=2 * ureg.um, tilt=Q_(0.1, ""), footprint=footprint
)  # ValueError: not an angle
target.with_thickness(2 * ureg.keV)  # ValueError: wrong dimension
physics.with_inelastic_cutoff(50)  # TypeError: needs an energy quantity
```

Dimensionless controls keep their numeric form: fractions such as
`energy_spread_frac`, direction vectors, Miller indices, mode orders, counts and
seeds are not quantities.

## Read quantities back and extract numbers

The semantic property returns a fresh quantity in the canonical unit. Convert
it for display; take `.magnitude` or keep using the suffixed field for numbers:

```python
beam.kinetic_energy  # <Quantity(60.0, 'kiloelectron_volt')>
beam.kinetic_energy.to("MeV")  # display conversion, storage unchanged
beam.energy_keV  # 60.0, numeric keV
target.thickness.to("nm")  # <Quantity(2000.0, 'nanometer')>, up to rounding
```

Read-back values are copies. Changing a returned array or calling `.ito(...)`
never alters the configuration object.

## Replace values

`dataclasses.replace` still takes the numeric fields. For a quantity, use the
owning `with_*` method:

```python
thicker = target.with_thickness(5 * ureg.um)
retuned = physics.with_inelastic_cutoff(Q_(0.1, "keV"))
tracked = retuned.with_secondary_threshold(Q_(1, "keV"))
untracked = tracked.with_secondary_threshold(None)  # optional values accept None
```

Portable quantum state recipes use one revalidating method:

```python
from pyrite.quantum import TransverseStateSpec

probe = TransverseStateSpec.recipe(
    "aperture_probe", kinetic_energy=60 * ureg.keV, semiangle=20 * ureg.mrad
)
narrower = probe.with_parameters(semiangle=10 * ureg.mrad)
narrower.parameter_quantities["semiangle"]  # <Quantity(0.01, 'radian')>
```

## Inspect advanced material and transverse-state quantities

CIF-derived catalog data stays numeric. Use the material schema's detached
read-back properties when you want cell parameters or densities with units;
material authoring already accepts semantic `density`, `b_factor` and
`mosaic_fwhm` quantities. The lattice mapping is mixed-unit: `system` remains
a string, lengths are in Å, and angles are in degrees. Number densities count
**atoms per Å³**, not electric charge.

```python
from pyrite.materials.catalog import load_material_catalog

catalog = load_material_catalog()
crystal = catalog.crystals["silicon"]
crystal.lattice_parameters["a"].to("nm")
crystal.cell_volume.to("nm**3")
crystal.number_densities  # ordered (element, 1/Å³) quantities
crystal.b_factor  # isotropic Debye-Waller B, Å²
```

Transverse-state preset functions accept Pint inputs and retain numeric
solver-facing states. The corresponding read-back exposes **derived geometry
and quadrature hints**, not an exact reconstruction of each original preset:

```python
from pyrite.montecarlo.channeling.states import gaussian, translate, aperture_probe

state = gaussian(waist_ang=2 * ureg.nm, dimension=1)
state.centre_position  # array quantity in Å
state.momentum_extent  # per-axis inverse Å; sampling hint, not mean momentum
shifted = translate(state, offset_ang=3 * ureg.nm)
shifted.displacement.to("nm")
probe = aperture_probe(60 * ureg.keV, semiangle_rad=20 * ureg.mrad)
probe.kinetic_energy.to("keV")
probe.semiangle.to("mrad")
```

`SampledState.sample_positions` returns copied position axes and its optional
`kinetic_energy` is metadata. `ApertureProbe.aberration_quantities` returns a
new mapping of length and angle quantities. Mutating read-back values never
changes numeric state data or identity. `amplitude` and `wavefunction` still
take and return numeric arrays; no Pint objects enter the solvers.

## Sweep and optimize semantic fields

`Sweep.axes` and optimization domains accept semantic paths whose owner
defines `with_<name>`. Each value must be a quantity:

```python
from pyrite import Sweep

sweep = Sweep(
    scene,
    axes={
        "beam.kinetic_energy": [30 * ureg.keV, 0.045 * ureg.MeV],
        "target.thickness": [200 * ureg.nm, 0.5 * ureg.um],
    },
)
```

Numeric paths such as `beam.energy_keV` keep numeric values and refuse
quantities. See
[quantity parameter paths](../reference/optimization-studies.md#quantity-parameter-paths)
for studies.

## Read results with units

Result tables and arrays remain numeric. Request a quantity view from the
object that owns the measure:

| Numeric source | Quantity view | Measure |
| --- | --- | --- |
| `result.cxr.incoherent` (`SpectralDensity`) | `.quantify()` | node-sampled density per energy per sr per incident electron |
| Directional spectrum | `.quantify()` | per energy per sr, on its angular grid |
| `PhotonContinuumTally`, `PhotonEventTally` | `.quantify()` | binned aperture density and integrated counts |
| `SpatialResult` | `.quantify_spectra(...)`, `.quantify_image(...)` | selected-pixel density; pixel-integrated image |
| `AcquisitionBatch` | `.quantify()` | detected counts per exposure |
| `ChannelingSpectrum` | `.quantify()` | node density per energy per sr per incident electron; line yields |

Change the energy unit with `to_energy`. It rescales a density inversely with
its energy axis, so integrated yields are unchanged:

```python
import numpy as np

spectrum = result.cxr.incoherent.quantify()
in_kev = spectrum.to_energy("keV")
# 1 photon/eV is 1000 photons/keV; the band integral is identical:
np.trapezoid(in_kev.density, in_kev.photon_energy)
```

Integrate each original component with its declared sampling and support:
`result.integrated_yield(mode="incoherent")` returns native component yields
and a sum of their integrals, including uncovered-window metadata. Node-only
trapezoidal calculations are inappropriate for bin means or mixed sampling.
`combined_cxr()` combines compatible CXR-grid components; bremsstrahlung retains
its own grid, and coherent/incoherent CXR remain alternatives.

Mean current and acceptance are explicit operations:

```python
from pyrite.units import Q_

intrinsic = result.cxr.incoherent.quantify()
directional_rate = intrinsic.rate(current=Q_(20, "nA"))
# A scalar source needs acknowledgment of uniform density over case acceptance:
accepted = result.accepted_spectra(mode="incoherent", assume_uniform=True)
accepted_rate = accepted["cxr"].rate(current=Q_(20, "nA")).to_energy("keV")
per_ampere = accepted["cxr"].rate_per_current()  # valid even for a zero-current beam
per_nA_density = per_ampere.density.to("1/eV/s/nA")
```

`rate()` requires a finite nonnegative scalar current quantity and multiplies
per-primary photon yield by $I/|e|$. `rate_per_current()` uses $1/|e|$ directly;
its canonical current denominator is ampere. The stored yield is held fixed:
for coherent CXR, changing the bunch population/shape requires new radiation
evaluation, rather than a current-only transformation. Either operation refuses an
already current-normalized view. `normalization_basis` distinguishes
`per_incident_electron`, `per_second`, and `per_current`; `normalization`
retains source/observation provenance and `measure` retains the angular measure.
Pint's dimensionless steradian cannot enforce those semantic distinctions.

Observation-average results use their matching summed pixel solid angle;
transmission is already included. They cannot recover intrinsic source spectra.
Spatial and aperture spectra already include acceptance, and measured spatial
spectra already include response. Their quantity views accept the same current
operations without applying acceptance, transmission or response again.
Acquisition counts remain owned by exposure and realization; they have no
current-normalization methods. Legacy `DirectionalSpectrum.rate(current_na=...)`
and aperture exports also accept `current=Q_(...)`, keeping numeric return
types; passing both current spellings raises an error.

Export each native component separately, with paired energy/density units and
semantic columns, including component and selected CXR mode for accepted
scalar views. No interpolation or summed spectral density is implied:

```python
import json

accepted_rate.to_dataframe().write_csv("cxr-rate.csv")
with open("cxr-rate.units.json", "w") as stream:
    json.dump(accepted_rate.dataframe_units(), stream, indent=2)
```

Numeric `result.to_dataframe()` and `result.dataframe_units()` remain supported.
`spectral_flux(units=...)` and `spectral_flux_units(units=...)` were deprecated
in 0.27.0 and removed in 0.29.0. Use `accepted_spectra()` or `quantify()`, then
explicit current transforms and native `to_dataframe()`/`dataframe_units()`;
use `integrated_yield()` for physical totals.

Each entry holds a Pint-parsable `unit` and a separate `basis`. The basis
records counts that Pint cannot express, such as photons per incident
electron: the electron in that phrase is a particle count, not Pint's charge
unit. Saved-array downloads from `pyrite app configuration` embed the same
`units` map in their NPZ `metadata_json` or CSV header line, covering each
component, its energy axis and the record's `scale` factor.

## Load saved results

Checkpoints, observation stores, remote work items and case records store
plain numbers in their documented suffixed units. Loading an old file needs no
migration and does not import Pint. Rebuild quantities after loading through
the same views, for example `loaded.cxr.incoherent.quantify()` or a
configuration's semantic properties. Quantities are never pickled into saved
payloads, and scientific identities and digests are unchanged by switching a
script to quantities.

## Compatibility status

| Area | Numeric spelling | Quantity spelling | Status |
| --- | --- | --- | --- |
| Beam energy, timing and normalization | `energy_keV`, `rms_fs`, `offsets_fs`, `rep_rate_hz`, `bunch_charge_pc` | `kinetic_energy`, `rms_duration`, `arrival_offsets`, `repetition_rate`, `bunch_charge` | Both supported |
| Target and body geometry | `thickness_ang`, `tilt_deg`, `*_mm`, `CrystalCut.*_deg` | `thickness`, `tilt`, `size`, `roll`/`miscut` | Both supported |
| Detectors, poses and filters | `observation_angle_deg`, `center_mm`, `pitch_mm`, `size_mm`, `thickness_mm` | `observation_angle`, `center`, `pitch`, `size`, `thickness` | Both supported |
| Acquisition | `exposure_s`, `measured_edges_eV`, `hit_threshold_eV` | `exposure`, `measured_edges`, `hit_threshold` | Both supported |
| Physics and energy grids | `*_cutoff_eV`, `secondary_threshold_eV`, `EnergyBins.cxr`/`brem` | `inelastic_cutoff`, `radiative_cutoff`, `secondary_threshold`; `cxr_energies`/`brem_energies` | Both supported |
| Quantum setup and recipes | `step_ang`, `zone_max_ang`, `waist_ang`, `kinetic_energy_eV`, … | `step_length`, `zone_max_length`, `waist`, `kinetic_energy`, … | Both supported |
| `Numerics.channeling_*` lengths | `channeling_dz_ang`, `channeling_zone_max_ang` | Set through `QuantumNumerics` / `QuantumRegion` | Numeric only |
| Experimental axial diagnostic, real-space solvers, response grids | `*_eV`, `*_ang`, `*_rad` | — | Numeric only; quantities raise `TypeError` |
| Material schema values | `lattice` (Å, deg), `V_cell`, `B_ang2`, composition | `lattice_parameters`, `cell_volume`, `b_factor`, `number_densities`, `mosaic_fwhm` | Numeric records; detached quantity read-back |
| Channeling transverse states | `centre`, `k_extent`, `k_resolution`, `axes`, `offset` | `centre_position`, `momentum_extent`, `momentum_resolution`, `sample_positions`, `displacement` | Numeric solver contracts; quantity read-back |
| Crystal constants and deterministic scientific helpers | `*_eV`, `*_ang`, `*_inv_mm` | — | Numeric contract; follow-up in #816 |
| CLI options, profile TOML | Option-name and key units; TOML also takes unit strings | — | Numeric CLI; unit-string CLI input is tracked separately |
| `cxr_metrics` peak keys | `peak_flux`, `peak_flux_per_na` | `peak_spectral_flux_density`, `peak_spectral_flux_density_per_na` | Removed in 0.26.0 (deprecated since 0.24.0) |

A future deprecation of a numeric spelling follows the
[release policy](../repo-design/releasing.md): it is announced in a shipping
release and supported for two minor versions before removal.


For slabs, specify full transverse dimensions directly with `Slab(width=5 *
ureg.mm, height=5 * ureg.mm)`. Use `width=None, height=None` for a
laterally infinite slab. `Slab(footprint=None)` and the quantity-valued
`Footprint.width` and `Footprint.height` spellings were removed in 0.29.0.
For a `Stack`, `Footprint(width_mm=5, height_mm=5)` remains supported.
