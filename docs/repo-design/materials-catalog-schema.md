# Materials catalog schema

The bundled `pyrite/data/catalog/` directory is the canonical schema-version-1 catalog for crystals, media, runnable materials, beams, detectors, and campaign profiles. `load_material_catalog()` validates the entire document and returns deeply immutable typed records. Invalid or unknown fields fail closed with grouped, location-qualified errors.

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
  profiles/<name>.toml    # profile fields plus its [overrides.MATERIAL] tables
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
[crystals.NAME]      # CIF-backed crystalline phase
[media.NAME]         # amorphous composition
[materials.NAME]     # runnable film or stack
```

`profiles`, `crystals`, `media`, and `materials` are required. `beams` and `detectors` are optional. `energy_grids` is accepted only as a legacy compatibility input; new grids are immutable artifacts referenced from profiles.

Bundled profiles contain neither artifact references nor legacy line rows, so every bundled material uses automatic case-local line grids. For an opt-in fixed grid, line rows resolve from the selected profile's artifact reference for that material, then its own `[energy_grids.MATERIAL]` legacy table. There is no cross-material fallback: profile-named tables, including `[energy_grids.standard]`, do not supply or seed another material's rows. Explicit fixed line grids remain supported. Without a fixed grid, all energies use the existing automatic `sinc-nyquist` resolution and `kinematic-ceiling` bandwidth policy from the case's own material and trajectories. Derivation is not a prerequisite.

Installing one material's immutable artifact repoints only that material in the selected profile. Other materials and profiles retaining the old digest keep their coordinates. Removing bundled fixed coordinates changes case/cache identity once for every affected profile; existing checkpoints are not reused or deleted. See [ADR-0013](../adr/0013-automatic-line-grids-by-default.md).

## Crystals and media

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

`[profiles.NAME]` can define material membership, a named beam, detector, emission policy, X-ray dispersion model, energy-grid references, shared scan descriptors, and `[profiles.NAME.overrides.MATERIAL]` tables. Supported grids decode to immutable float64 arrays. A descriptor uses one of these forms:

```toml
energy_keV = { values = [30.0, 50.0] }
tilt_deg = { linspace = [-2.0, 2.0, 9] }
thickness_ang = { logspace = [2.0, 5.0, 7] }
tilt_azim_deg = { arange = [0.0, 360.0, 30.0] }
```

Electron-count grids, line grids, bremsstrahlung grids, thickness-in-layers, and per-energy line grids use the same validated descriptor model where applicable. Units belong in field names: electron energy is keV, photon energy is eV, thickness is Å, and angles are degrees.

## Beams and detectors

A named beam contains display metadata plus transverse and longitudinal source parameters. Profiles attach it with `beam = "NAME"`; the name and label are removed during resolution so identity follows values. The full field and unit reference is [Beam phase space](../physics/beam-transport/beam-phase-space.md).

`[profiles.NAME.detectors.ID]` declares one detector with scalar acceptance or pixel geometry; `[profiles.NAME.detectors]` may instead map an ID to a named `[detectors.REFERENCE]` object. Named objects support both geometries. Pixel rows may contain detector-specific `scorer`, `response`, and `acquisition` tables; their scalar acceptance is derived from pose and grid. IDs identify result/checkpoint ownership and therefore enter new collection identities, while the referenced object's name and display label do not. Legacy `detector` and `physical_detector` rows remain readable and migrate to IDs `default` and `physical`; they cannot be combined with a `detectors` collection. A profile without any detector declaration resolves one implicit scalar `default`. See [Sweep profiles](../guides/sweep-profiles.md) for the complete form and identity behavior.

## Validation and editing

Prefer `pyrite profile`, `pyrite beam`, `pyrite detector`, `pyrite material`, and `pyrite energy-grid` for supported mutations. Validate a complete alternate catalog without running simulation:

```bash
pyrite material validate path/to/catalog        # directory
pyrite material validate path/to/materials.toml  # single file
```

New crystal phases require a packaged CIF, explicit provenance, a validation record and ledger row, and transport support for every constituent element.
