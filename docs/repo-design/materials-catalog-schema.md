# Materials catalog schema

The bundled `pyrite/data/catalog/` directory is the canonical schema-version-1 catalog for crystals, media, runnable materials, beams, detectors, physics and numerics presets, and campaign profiles. `load_material_catalog()` validates the entire document and returns deeply immutable typed records. Invalid or unknown fields fail closed with grouped, location-qualified errors.

CIF structures are parsed with Gemmi under its MPL-2.0 license option. Explicit
CIF symmetry operations take precedence over the named space group; affine
operations expand the asymmetric unit modulo integer translations, with
special-position images deduplicated at a fractional tolerance of `1e-12`.
Partial occupancy is rejected. Cell parameters retain their literal CIF values.
The migration from `crystals` (#340) changes only floating-point rounding in the
bundled catalog (within `2e-15` relative / `1e-12` absolute), but exact content
hashes can change: affected checkpoint records are recomputed and generated
SBETHE/ELSEPA tables must be regenerated for the new material identity.

## Catalog layout

A catalog is either a catalog directory with one TOML file per object, or a single TOML file holding every table. Both load, validate, and accept CLI edits identically; the bundled catalog is a directory.

```text
catalog/
  catalog.toml            # schema_version = 1 (root keys only)
  crystals/<name>.toml    # one file per [crystals.<name>]
  media/<name>.toml
  materials/<name>.toml
  beams/<name>.toml
  detectors/<name>.toml
  physics/<name>.toml     # named physics presets (ADR-0021)
  numerics/<name>.toml    # named numerics presets (ADR-0021)
  filters/<name>.toml     # named finite downstream plates or sequences (issue #665)
  targets/<name>.toml     # named scalar slab/stack targets (issue #665)
  gratings/<name>.toml    # named diffraction gratings and mounts (issue #1080)
  profiles/<name>.toml    # profile fields plus its [overrides.MATERIAL] tables
  cifs/                   # catalog-owned CIF structures
  curves/<name>.csv       # tabulated instrument curves, or curves/<name>.toml
  energy-grid-artifacts/  # immutable line-grid artifacts, not catalog source
```

An object file holds the object's body without its `[table.name]` header; sub-tables are relative (`[validation]`, `[[stack]]`, `[overrides.hopg]`, `[longitudinal]`). The file stem is the object name, so a name cannot be defined twice. Stems use letters, digits, `.`, `_`, and `-`, start with a letter or digit, and must not differ from another stem in the same table only by case. Unknown entries, non-`.toml` files, and table content placed in `catalog.toml` are rejected with the offending path; names beginning with `.` are ignored. Directory objects are ordered by name, while a single file keeps declaration order; order affects listings, never identity.

CLI edits assemble the directory into one document, apply the change, validate the whole result, and then rewrite only the object files whose text changed (creating or deleting files for new, renamed, or removed objects). The single-file examples below show `[table.NAME]` headers; in a directory the same body lives in `table/NAME.toml`.

## Root tables

```toml
schema_version = 1

[profiles.NAME]      # campaign defaults and membership
[beams.NAME]         # reusable beam distribution
[detectors.NAME]     # reusable detector geometry
[physics.NAME]       # reusable physics models and cutoffs
[numerics.NAME]      # reusable sampling, integration and grid policy
[filters.NAME]       # reusable downstream finite plate or plate sequence
[targets.NAME]       # reusable scalar slab or stack target
[gratings.NAME]      # diffraction grating and spectrometer mount record
[crystals.NAME]      # CIF-backed crystalline phase
[media.NAME]         # amorphous composition
[materials.NAME]     # runnable film or stack
```

`profiles`, `crystals`, `media`, and `materials` are required. `beams`, `detectors`, `physics`, `numerics`, `filters`, `targets`, and `gratings` are optional. `energy_grids` is accepted only as a legacy compatibility input; new grids are immutable artifacts referenced from profiles.

Bundled profiles contain neither artifact references nor legacy line rows, so every bundled material uses automatic case-local line grids. For an opt-in fixed grid, line rows resolve from the selected profile's artifact reference for that material, then its own `[energy_grids.MATERIAL]` legacy table. There is no cross-material fallback: profile-named tables, including `[energy_grids.standard]`, do not supply or seed another material's rows. Explicit fixed line grids remain supported. Without a fixed grid, all energies use the existing automatic `sinc-nyquist` resolution and `kinematic-ceiling` bandwidth policy from the case's own material and trajectories. Derivation is not a prerequisite.

Installing one material's immutable artifact repoints only that material in the selected profile. Other materials and profiles retaining the old digest keep their coordinates. Removing bundled fixed coordinates changes case/cache identity once for every affected profile; existing checkpoints are not reused or deleted. See [ADR-0013](../adr/0013-automatic-line-grids-by-default.md).

## Crystals and media

Photon grids belong to numerics, not crystals. Crystal `E_grid` fields are rejected as of 0.18.0. Use profile grids or `Sweep.energy_bins` instead.

A crystal names a packaged CIF and a validation record. Optional metadata includes phase/source identifiers, Debye--Waller `B_ang2`, beam or surface orientation, pinned reflection families, layer count, and mosaic FWHM. CIF paths must resolve below packaged `data/cifs`; arbitrary filesystem paths are rejected. Lattice, expanded basis, unit-cell volume, and number-density composition are derived from the CIF rather than duplicated in TOML.

```toml
[crystals.example]
cif = "cifs/example.cif"
validation_id = "example-structure"
hkl_families = [[0, 0, 2]]
beam_uvw = [0, 0, 1]
B_ang2 = 0.3
```

An amorphous medium supplies element number densities in atoms/Å³:

```toml
[media.sio2]
composition = { Si = 0.022, O = 0.044 }
```

## Materials and stacks

A material selects a crystal and optional label, substrate, or fixed stack. Profiles provide its scan grids. Stack layers are in beam-entrance order and name either a crystal or medium, with thickness in Å and optional orientation. The film remains the radiation-producing crystal; layer-aware transport and self-absorption use the complete resolved stack.

```toml
[materials.example]
crystal = "example"
label = "Example film"

[[materials.example.stack]]
material = "sio2"
thickness_ang = 1.0e4
```

## Profiles and scan descriptors

`[profiles.NAME]` can define material membership, a named beam, detector, emission policy, X-ray dispersion model, energy-grid references, shared scan descriptors, and `[profiles.NAME.overrides.MATERIAL]` tables. An omitted `E_grid_brem` resolves to the built-in 25 eV uniform continuum grid; case lowering raises its start to the material floor and replaces its stop with the case beam energy plus one step. Explicit grids, material overrides, and stored artifacts retain their precedence. Supported grids decode to immutable float64 arrays. A descriptor uses one of these forms:

```toml
energy_keV = { values = [30.0, 50.0] }
tilt_deg = { linspace = [-2.0, 2.0, 9] }
thickness_ang = { logspace = [2.0, 5.0, 7] }
tilt_azim_deg = { arange = [0.0, 360.0, 30.0] }
```

Electron-count grids, line grids, bremsstrahlung grids, thickness-in-layers, and per-energy line grids use the same validated descriptor model where applicable. Units belong in field names: electron energy is keV, photon energy is eV, thickness is Å, and angles are degrees.

Physical grid values and
profile beam, detector, filter, response, and acquisition fields also accept
explicit quantity strings. For example, `energy_keV = "60 keV"`,
`distance_mm = "10 cm"`, `polar_deg = "0.5 rad"`, and
`exposure_s = "500 ms"` resolve through the same Pint adapter as Python
configuration objects. Existing numeric TOML values retain their field units.
See the [physical-units boundary and supported fields](physical-units.md).

## Beams and detectors

A named beam contains display metadata plus transverse and longitudinal source parameters. Profiles attach it with `beam = "NAME"`; the name and label are removed during resolution so identity follows values. The full field and unit reference is [Beam phase space](../physics/beam-transport/beam-phase-space.md).

`[profiles.NAME.detectors.ID]` declares one detector with scalar acceptance or pixel geometry; `[profiles.NAME.detectors]` may instead map an ID to a named `[detectors.REFERENCE]` object. Named objects support both geometries. Pixel rows may contain detector-specific `scorer`, `response`, and `acquisition` tables; their scalar acceptance is derived from pose and grid. IDs identify result/checkpoint ownership and therefore enter new collection identities, while the referenced object's name and display label do not. Legacy `detector` and `physical_detector` rows remain readable and migrate to IDs `default` and `physical`; they cannot be combined with a `detectors` collection. A profile without any detector declaration resolves one implicit scalar `default`, deprecated in 0.9.0 and refused from 0.11.0. Partial scalar acceptance follows the same window. The bundled `reference` object explicitly declares the existing Timepix3 aperture with ideal response and raw spectra; selecting it through the legacy single-detector field preserves checkpoint identity. A scalar row with `kind = "point"` is an ideal point observer (`PointDetector`): optional `observation_angle_deg` (default 0, on axis) and no `polar_acceptance_deg` or `solid_angle_sr`, which it refuses; the bundled `point` object is the on-axis form that admitted axial fibre cases require. `pyrite detector set NAME --point` and `pyrite profile detector set PROFILE ID --point` write it. See [campaign configuration](../reference/campaign-configuration.md#detector-collections-filters-and-counting-observations) for the complete form and identity behavior.

## Physics and numerics presets

`[physics.NAME]` holds `emission` and any profile key that is a `Physics` field (elastic, inelastic, bremsstrahlung, radiative, pair, positron, atomic-electron, shell-ionization, channeling-model and transition-radiation selectors and their cutoffs), plus an optional display `description`. Physical energy cutoffs and the transition lattice origin also accept explicit compatible quantity strings. Omitted keys keep built-in defaults; sampling, integration and grid numerics are rejected. A profile attaches one with `physics = "NAME"`. Before validation the catalog folds the preset under the profile's own physics keys, which stay explicit overrides, and validates the merged values for every referencing profile. Consumers, run identities and checkpoints therefore see only effective values: renaming a preset, adding an unused one, or moving equal inline values into a preset does not change identity. The bundled `electron-standard` preset restates the built-in defaults.

`[numerics.NAME]` holds scalar Monte Carlo history counts (`cxr_histories`, `brem_histories`; one positive integer each — sweep grids stay in profiles), reflection/mosaic quadrature, integration and channeling-zone numerics, and optional `[adaptive_sampling]` and `[line_grid_policy]` tables; a profile attaches it with `numerics = "NAME"`. Channeling depth and zone lengths accept explicit compatible quantity strings. History counts and `adaptive_sampling` form one sampling policy: a profile that sets any of them takes none of them from its preset, so fixed counts and adaptive sampling never mix. The removed spellings `cxr_trials`/`brem_trials`, `cxr_electrons`/`brem_electrons`, `n_electrons`/`n_electrons_brem`, `precision` and the `*_electrons` adaptive keys are read errors naming their replacement (#1007). Physical models and cutoffs are rejected. The bundled `default` preset restates the built-in integration and mosaic numerics and leaves history counts, reflection truncation and adaptive sampling to the built-in defaults and default adaptive policy.

Profile runs record each referenced preset's name and definition digest (`presets` beside `parameter_sha256` in the dataset identity and case manifest) as an unhashed label. See [ADR-0021](../adr/0021-named-physics-numerics-presets.md).

## Validation and editing

Prefer `pyrite profile`, `pyrite beam`, `pyrite detector`, `pyrite material`, and `pyrite energy-grid` for supported mutations. Validate a complete alternate catalog without running simulation:

```bash
pyrite material validate path/to/catalog        # directory
pyrite material validate path/to/materials.toml  # single file
```

New crystal phases require a packaged CIF, explicit provenance, a validation record and ledger row, and transport support for every constituent element.

## Named targets

Scalar `[targets.NAME]` objects (or `targets/NAME.toml` in directory catalogs)
are reusable target definitions. Attach one through
`[profiles.PROFILE.targets]` with `MATERIAL = "NAME"`. This mapping does not
add profile members; the key must match the target's entrance recipe.
Target geometry replaces inherited profile geometry for that material, and
per-material geometry overrides alongside the reference are rejected.
Beam energies, photon grids and numerical/physical policies remain profile-owned.

```toml
[targets.film]
kind = "slab"
material = "silicon"
thickness_ang = 20000.0
tilt_deg = 30.0
tilt_azim_deg = 0.0

[profiles.example.targets]
silicon = "film"
```

Use a runnable entrance recipe without an existing substrate/stack.
`kind = "stack"` instead accepts `layers`, an ordered array of at least two
inline tables with `material` and positive scalar `thickness_ang`. The first
material is a runnable entrance recipe; remaining materials are crystal or
medium keys, with optional existing `beam_uvw` and finite `azimuth_deg` fields.
All layers use angstroms; tilt angles use degrees.

Both variants accept scalar tilts (defaults 30 and 0 degrees), paired positive
`crystal_width_mm` / `crystal_height_mm` (defaults 5 mm each), and optional
boolean `mosaic` / `allow_normal_incidence` (both false). Slabs optionally
accept positive `groove_spacing_ang` with the existing groove restrictions.
`footprint = "unbounded"` replaces both dimensions with a laterally unbounded
slab (`Slab(width=None, height=None)`) and excludes them. Slabs also accept
`cut = {surface_hkl = [h, k, l]}`, the nonzero slab-normal plane of
`CrystalCut(surface_hkl=...)`. Omitting either key keeps the existing finite
footprint and catalog orientation, so existing identities are unchanged.
Optional `description` is nonempty display metadata. Other cut fields,
array sweeps, arbitrary variants and unknown fields are rejected.

Names are reserved case-insensitively across bundled/user layers. Bundled
objects remain read-only. Python uses `pyrite.geometry.load_target(NAME)`
(or an explicit `catalog=`) to obtain an existing `Slab` or `Stack`. Resolved
physical values alone affect identities: equivalent inline and named target
geometry gives the same campaign/checkpoint identity; name-only changes do
not invalidate it. `pyrite target` manages the lifecycle.

## Named filters

`[filters.NAME]` is one complete finite plate, or `kind = "sequence"` with a
nonempty ordered `plates` array of complete plates (no nested references).
`[[profiles.PROFILE.filters]]` entries use `ref = "NAME"`. A plate reference may
override plate fields; a sequence reference accepts only `ref` and expands in
place, keeping each member's absolute pose. Only expanded plate values enter
observation identity. `pyrite.geometry.load_filter` returns one plate and
rejects a sequence; `load_filters` returns an ordered tuple for either form.

## Instrument curves and gratings

These records let a laboratory keep instrument data, including confidential
datasheet curves, in a private catalog layer (see
[private instrument catalogs](../reference/catalogs-and-workspaces.md#private-instrument-catalogs)).
They are validated declarative data; the ALEX-s CCD response and grating
physics that consume them are tracked in issue #1081.

### Tabulated curves

A curve is a dimensionless response (`quantity` is `efficiency`,
`quantum_efficiency`, `transmission` or `reflectance`) sampled on an energy or
wavelength abscissa. Objects reference it by name; `NAME` resolves to exactly
one of `curves/NAME.csv` or `curves/NAME.toml` under the catalog root that owns
the referencing object file (the user layer for user objects). Curve files and
the `curves/` directory may not be symlinks, and the resolved path must stay
inside `curves/`.

CSV form: `#` comment lines carrying TOML `key = value` provenance, one header
row `<energy|wavelength> [<unit>], value [<unit>]` (the value unit is optional
and defaults to dimensionless), then two numbers per row:

```text
# source = "Detector datasheet, document ID"
# revision = "B"
# digitization = "WebPlotDigitizer, 1 nm steps"
# confidential = true
# quantity = "quantum_efficiency"
wavelength [nm], value [percent]
10.0, 20.0
20.0, 50.0
```

TOML form: the same provenance keys at top level, plus `abscissa`
(`"energy"` or `"wavelength"`), `abscissa_unit`, optional `value_unit`, and
equal-length `x` and `y` arrays.

| Key | Required | Meaning |
|---|---|---|
| `source` | yes | Source document (title or identifier) |
| `revision` | no | Source document revision |
| `digitization` | yes | How the numbers were obtained |
| `confidential` | yes | `true` when the source must not be published |
| `quantity` | yes | One of the four dimensionless quantities above |
| `label` | no | Display text |

Units convert through the shared Pint registry: energies to eV, wavelengths to
energy by `E = hc/λ`, and values to a fraction (`percent` is accepted). The
abscissa must be finite and strictly increasing in file order, at least two
rows are required, and converted values must lie in `[0, 1]`. The loaded
`MaterialCatalog.curves` maps each referenced curve name to a record whose
`energy_eV` ascends, with values reordered to match.

### Gratings

```toml
[gratings.example-ff]
label = "Example flat-field grating"
groove_density_per_mm = 1200   # positive, 1/mm
order = -1                     # nonzero integer
order_convention = "sum"       # m λ = d (sin α + sin β); "difference": d (sin α − sin β)
coating = "Au"                 # optional display text
blank_mm = [30.0, 50.0]        # optional width and height
efficiency_curve = "example-ff-efficiency"   # optional curve name

[gratings.example-ff.mount]
r_mm = 350.0                   # entrance arm, required
alpha_deg = 85.0               # incidence angle from the grating normal, required
band_nm = [5.0, 20.0]          # band ends, required by r_prime_mm / beta_deg
r_prime_mm = [470.0, 460.0]    # exit arm at the two band ends
beta_deg = [-80.0, -75.0]      # diffraction angle at the two band ends
beta_H_deg = -78.0             # optional
r_H_prime_mm = 465.0           # optional
focal_plane_length_mm = 25.0   # optional
```

Angles are measured from the grating normal and must lie strictly between
−90° and 90°; lengths are positive. Physical fields accept compatible quantity
strings (`"2.5 cm"`). The record is display and inventory data only; nothing
here enters a simulation identity.

### ALEX-s detector response

A pixel detector `[response]` table may use `kind = "alexs"`:

```toml
[detectors.lab-ccd.response]
kind = "alexs"
sensor_format = [2048, 512]    # columns, rows (positive integers)
sensor_code = "EXAMPLE-BI"     # vendor sensor designation
readout_rate_hz = "1 MHz"      # positive frequency
gain_mode = "high"             # vendor gain mode name
qe_curve = "lab-ccd-qe"        # optional curve name
```

The first four keys are required. Observations that lower such a detector fail
with a "not yet supported" error until issue #1081 lands.
