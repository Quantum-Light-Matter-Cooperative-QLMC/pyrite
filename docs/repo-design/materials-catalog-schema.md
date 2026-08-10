# Materials catalog schema

The bundled `cxr_mc/data/materials.toml` is the canonical schema-version-1
catalog for crystals, media, runnable materials, beams, and campaign profiles.
`load_material_catalog()` validates the entire document and returns deeply
immutable typed records. Invalid or unknown fields fail closed with grouped,
location-qualified errors.

## Root tables

```toml
schema_version = 1

[profiles.NAME]      # campaign defaults and membership
[beams.NAME]         # reusable beam distribution
[crystals.NAME]      # CIF-backed crystalline phase
[media.NAME]         # amorphous composition
[materials.NAME]     # runnable film or stack
```

`profiles`, `crystals`, `media`, and `materials` are required. `beams` is
optional. `energy_grids` is accepted only as a legacy compatibility input;
new grids are immutable artifacts referenced from profiles.

## Crystals and media

A crystal names a packaged CIF and a validation record. Optional metadata
includes phase/source identifiers, Debye--Waller `B_ang2`, beam or surface
orientation, pinned reflection families, layer count, and mosaic FWHM. CIF
paths must resolve below packaged `data/cifs`; arbitrary filesystem paths are
rejected. Lattice, expanded basis, unit-cell volume, and number-density
composition are derived from the CIF rather than duplicated in TOML.

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

A material selects a crystal and optional label, substrate, or fixed stack.
Profiles provide its scan grids. Stack layers are in beam-entrance order and
name either a crystal or medium, with thickness in Å and optional orientation.
The film remains the radiation-producing crystal; layer-aware transport and
self-absorption use the complete resolved stack.

```toml
[materials.example]
crystal = "example"
label = "Example film"

[[materials.example.stack]]
material = "sio2"
thickness_ang = 1.0e4
```

## Profiles and scan descriptors

`[profiles.NAME]` can define material membership, a named beam, detector,
emission policy, energy-grid references, shared scan descriptors, and
`[profiles.NAME.overrides.MATERIAL]` tables. Supported grids decode to immutable
float64 arrays. A descriptor uses one of these forms:

```toml
energy_keV = { values = [30.0, 50.0] }
tilt_deg = { linspace = [-2.0, 2.0, 9] }
thickness_ang = { logspace = [2.0, 5.0, 7] }
tilt_azim_deg = { arange = [0.0, 360.0, 30.0] }
```

Electron-count grids, line grids, bremsstrahlung grids, thickness-in-layers,
and per-energy line grids use the same validated descriptor model where
applicable. Units belong in field names: electron energy is keV, photon energy
is eV, thickness is Å, and angles are degrees.

## Beams and detectors

A named beam contains display metadata plus transverse and longitudinal source
parameters. Profiles attach it with `beam = "NAME"`; the name and label are
removed during resolution so identity follows values. The full field and unit
reference is [Beam phase space](../physics/beam-transport/beam-phase-space.md).

Detector blocks define observation angle, polar acceptance, solid angle, and
an optional instrument-response specification. Missing detector values inherit
as described in [Configuration resolution](configuration-resolution.md).

## Validation and editing

Prefer `pyrite profile`, `pyrite beam`, `pyrite material`, and `pyrite
energy-grid` for supported mutations. Validate a complete alternate catalog
without running simulation:

```bash
pyrite material validate path/to/materials.toml
```

New crystal phases require a packaged CIF, explicit provenance, a validation
record and ledger row, and transport support for every constituent element.
