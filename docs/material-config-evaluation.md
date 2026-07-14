# Material configuration: declarative catalog evaluation

Per-material configuration now lives in the immutable, CIF-backed catalog at
`src/cxr_mc/data/materials.toml`. This supersedes the former split across a
structure table, per-material registries, source-level crystal defaults, and
scan-grid tables. The consolidation changes configuration
ownership, not the reflection-ranking physics.

## Current ownership

```text
data/cifs/<phase>.cif          lattice + symmetry-expanded fractional basis
             │
data/materials.toml           crystal metadata, profiles, media, materials, stacks
             │
materials.catalog.CATALOG     validated, ordered, deeply immutable typed records
             ├── config.py    material scan/sweep projections
             └── sweep.py     reflection resolution + Cartesian case construction
```

Production loading is offline-only. `crystals` 1.7 parses each bundled CIF and
expands symmetry; cxr-mc remains responsible for form factors, structure
factors, reflection selection, attenuation, transport, and radiation.

The package boundary exposes `MaterialCatalog`, `CrystalInfo`, `CrystalSpec`,
`MediumSpec`, `MaterialSpec`, `ScanSpec`, `LayerSpec`, and
`load_material_catalog`. `CATALOG.material_keys` preserves `[materials]`
declaration order. It is the complete runnable catalog; `mats_to_sim.toml` is
only the smaller ordered user selection consumed by `cxr scan --all` and remote
`--all` commands.

## Scan values and precedence

Each profile is complete and profiles do not inherit from one another. A
material selects one profile, then material-local scan fields replace the
corresponding profile fields. A thickness override also replaces the profile's
other thickness spelling, so the resolved scan has exactly one source:

- `thickness_ang`: physical thickness in Å;
- `thickness_layers`: positive integer layer count, converted with the
  referenced crystal's CIF `c` and required `layers_per_cell` as
  `layers * c / layers_per_cell`.

Every scalar is treated as a one-point grid. Descriptor semantics are exactly
NumPy's:

- `values = [...]` preserves the explicit sequence;
- `arange = { start, stop, step }` excludes `stop` exactly as `np.arange` does;
- `linspace = { start, stop, num, endpoint }` includes `stop` by default;
- `logspace = { start, stop, num, endpoint, base }` includes the exponent
  endpoint by default and uses base 10 unless specified.

For `linspace` and `logspace`, `endpoint = false` is explicit and honored.

## Orientation and reflection selection

`beam_uvw` is required on every crystal. Most crystals leave reflection choice
to `dominant_reflections`, using the existing `n_families` default and the
unchanged `|S(g)| exp(-W) / g^2` ranking policy.

Special cuts use `hkl_families` in the crystal row. Each entry is the positive
representative of a reciprocal family; the catalog deterministically expands
it to both `+hkl` and `-hkl`. A nonempty `hkl_reason` is mandatory whenever a
family is pinned, making the bypass of automatic ranking explicit. `cxr scan`
prints the resolved `beam_uvw` and full `hkl_list`; checkpoints persist those
resolved values in each case.

The CLI still permits `--beam-uvw H K L` and `--n-families N`. The latter has no
effect on a catalog-pinned reflection list, which is why the resolved list—not
the requested family count—is the provenance stored and displayed.

Changing the default family count, `g_max_invang`, or the ranking metric remains
a physics change. It requires fresh-context review and validation-ledger work;
the catalog consolidation did not change those values.

## Validation and failure policy

Run `uv run cxr check-config` for the bundled catalog or
`uv run cxr check-config path/to/materials.toml` for an explicit complete
catalog. The loader accumulates path-qualified schema and semantic errors.

Runnable compositions containing elements absent from cxr-mc's transport
constants are rejected. A supported element without a packaged Mott transport
CSV is nonfatal: loading logs a warning and electron transport uses the analytic
screened-Rutherford fallback.

Every crystal has a nonempty `validation_id`, and repository tests require each
ID to appear as an exact row in `docs/physics-validation-ledger.md`. Structure
provenance lives in the phase-specific CIF; run metadata such as B factor,
orientation, scan defaults, and pin reasons lives in `materials.toml`.

The locked `crystals` 1.7.0 dependency is GPLv3. A licensing review is required
before distributing cxr-mc source, wheels, binaries, or containers that include
it.
