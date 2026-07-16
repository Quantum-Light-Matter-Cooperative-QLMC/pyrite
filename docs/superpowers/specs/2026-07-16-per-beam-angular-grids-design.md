# Per-Beam Line Grids and Quantized Angle Sweeps

## Goal

Reduce the standard angular sweep, extend the supported beam-energy range, and
select a bounded coherent-line grid for each beam energy. All generated polar
and azimuthal case angles must be quantized to the nearest half degree so
equivalent geometries are not recomputed under incrementally different floating
point values.

## Standard Scan Definition

The standard profile will use these beam energies, in order:

```text
30, 50, 100, 150, 200, 250, 300 keV
```

The standard polar grid will be the ten endpoint-inclusive values from
`linspace(0, 89, 10)`. The standard azimuthal grid will be the ten
endpoint-inclusive values from `linspace(90, 180, 10)`. Material-specific angle
overrides on HOPG and h-BN will be removed so this reduced angular domain applies
to every runnable material.

Each beam energy will select its own uniform coherent-line grid:

| Beam energy | Line-grid start | Line-grid end |
| ---: | ---: | ---: |
| 30 keV | 10 eV | 2500 eV |
| 50 keV | 10 eV | 3000 eV |
| 100 keV | 50 eV | 3500 eV |
| 150 keV | 50 eV | 4000 eV |
| 200 keV | 50 eV | 4500 eV |
| 250 keV | 50 eV | 5000 eV |
| 300 keV | 50 eV | 5000 eV |

The grids will include both endpoints and retain approximately 3 eV uniform
spacing. Endpoint inclusion takes priority over making every spacing exactly
3 eV because the line kernel's optimized path requires a uniform grid, while
the requested upper bounds should be represented exactly.

Ten eV is the low-energy boundary for the 30 and 50 keV cases. The atomic form
factor and attenuation data are available there, while the coherent-spectrum
kernel deliberately excludes resonance energies at or below 10 eV. Extending
below 10 eV is outside this change because it would require changing and
independently validating that physics boundary.

## Catalog Model and Override Precedence

The catalog scan schema will gain a declarative per-beam line-grid mapping. The
standard profile owns the mapping, and catalog validation will require exactly
one matching line grid for every configured beam energy. Duplicate mapping
entries and missing or extra beam-energy entries will fail fast with a precise
catalog path in the error.

The existing fixed `E_grid_line` descriptor remains supported. A material-level
fixed descriptor takes precedence over the profile's per-beam mapping, retaining
the ability to give an exceptional material one wider or otherwise specialized
line grid in the future. The current material-specific line-grid overrides will
be removed so all existing runnable materials use the new standard mapping.

The immutable catalog projection will expose either a fixed line grid or the
per-beam mapping without returning writable arrays. `material_grid` and
`material_sweep` will preserve that representation rather than flattening it to
one shared array.

## Case Construction

`build_cases` will resolve the coherent-line grid after selecting the case's beam
energy. A fixed `Sweep.E_grid_line` continues to apply to every energy; a
per-beam mapping selects the entry whose numeric beam energy matches the case.
The chosen grid will continue through the existing exact energy-grid encoder so
checkpoint and worker behavior remain compatible.

Before the Cartesian product is formed, `build_cases` will quantize every polar
and azimuthal input to the nearest 0.5 degrees. Quantization applies at this
central boundary to catalog scans, notebook or CLI overrides, and direct library
use. It is not limited to this particular standard profile.

Quantization will use an explicit nearest-half rule that is symmetric for
positive and negative angles. It will preserve first-occurrence order and remove
duplicates introduced by rounding. Names, stored `tilt_deg` values, stored
`tilt_azim_deg` values, radians passed to the geometry kernel, and random seeds
will all derive from the quantized, deduplicated arrays. This prevents duplicate
physical cases and keeps checkpoints stable.

Other scalar-or-sequence sweep dimensions will not be rounded or deduplicated.
Bremsstrahlung-grid behavior remains unchanged.

## Compatibility and Errors

Direct `Sweep` callers using a single NumPy `E_grid_line` remain compatible.
Legacy encoded grid tuples and exact nonuniform arrays continue to round-trip
through the existing codec. A per-beam mapping that lacks the selected beam
energy will raise a clear `ValueError` before any simulation starts.

The catalog schema version remains unchanged because this is an additive scan
descriptor accepted by the current loader, not a reinterpretation of existing
catalog files.

## Verification

Focused tests will cover:

- the seven standard beam energies and their exact line-grid endpoints;
- approximately 3 eV uniform spacing for every per-beam grid;
- ten endpoint-inclusive standard polar and azimuthal inputs;
- half-degree quantization for positive and negative angles;
- order-preserving removal of duplicates created by quantization;
- quantized values in case names, stored fields, and geometry inputs;
- fixed material or direct-`Sweep` line-grid precedence;
- catalog rejection of duplicate, missing, and extra per-beam entries;
- exact/read-only grid behavior through catalog loading and case encoding; and
- unchanged behavior for callers that supply one fixed line grid.

Verification will use the smallest relevant catalog, sweep, configuration, and
case-building test modules first, followed by the canonical repository
verification command. The isolated branch begins with three unrelated catalog
expectation failures on committed `main` (material count, historical h-BN
thickness, and a serialized crystal-grid fingerprint); those failures will be
reported separately rather than attributed to this feature.
