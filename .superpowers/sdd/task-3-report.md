# Task 3 Report: Germanium-family crystals

## Status

Implemented the four requested layered germanium-family crystals with exact
source settings, reciprocal surface orientations, pinned allowed reflections,
catalog material rows, focused regressions, and unverified ledger entries.

## Structures and provenance

| Key | Model and source | Surface / reflection | B [A2] | Expanded basis |
| --- | --- | --- | ---: | --- |
| `gep` | COD 1562070 CC0, conventional C2/m | (10-1) / (20-2) | 0.35 | Ge12P12 |
| `ges` | COD 8104282 CC0, normalized standard Pnma | (100) / (200) | 1.07 | Ge4S4 |
| `gese` | COD 4003515 CC0, standard Pnma y=3/4 origin | (100) / (200) | 1.10 | Ge4Se4 |
| `gese2` | MP mp-540625 initial experimental structure, CC BY 4.0, transformed losslessly to P21/c:b1 | (001) / (002) | 0.6 | Ge16Se32 |

Each CIF records its accession, primary DOI, redistribution license, setting
choice or transform, and `Validation:` marker. The GeSe2 record includes the
matminer snapshot date and SHA-256, identifies B=0.6 A2 as a catalog
placeholder, and explicitly excludes tetragonal high-pressure COD 1521080.

The catalog uses `surface_hkl` rather than a direct-axis approximation, so the
two monoclinic surfaces are oriented by their reciprocal normals. Each
`hkl_reason` records why the physical cleavage plane is extinct and why the
first parallel allowed reflection is pinned.

## TDD evidence

RED command, before any catalog or CIF production changes:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test \
  tests/test_germanium_crystals.py
```

Observed: `4 failed`. Every parameter failed at `CATALOG.crystal(key)` with the
expected missing-crystal `KeyError` for `gep`, `ges`, `gese`, and `gese2`.

GREEN command after the minimal CIF and catalog implementation:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test \
  tests/test_germanium_crystals.py
```

Observed: `4 passed in 0.62s`.

The parameterized regression pins all six lattice parameters, volume, expanded
site count and stoichiometry, `surface_hkl`, pinned family, B value, and finite
nonzero `F_g`, `chi_g`, and `U_g` at 1500 eV.

## Verification

- Focused test: `4 passed in 0.64s`.
- `uv run cxr check-config`: valid bundled catalog with 31 materials, 31
  crystals, and 1 medium.
- `uv run python scripts/dev.py lint`: all checks passed.
- `uv run python scripts/dev.py typecheck`: 0 errors, warnings, or information.
- `git diff --check`: clean.

An additional read-only projection confirmed the exact loaded volumes
497.901914, 164.280808, 182.628745, and 1394.084071 A3 and finite nonzero
couplings for all four pinned reflections.

## Physics and source self-review

- Units: lattice constants and volumes are in A and A3; B values are in A2;
  the catalog and scattering functions use those existing conventions.
- Coordinates: GeP preserves the COD conventional C2/m axes; GeS coordinates
  are transformed consistently with the standard-Pnma axis normalization;
  GeSe preserves COD 4003515's standard-Pnma y=3/4 origin; GeSe2 applies the
  packet's explicit `a'=A, b'=-C, c'=B` and `(x',y',z')=(x,-z,y)` transform.
- Multiplicity/stoichiometry: parser symmetry expansion gives Ge12P12, Ge4S4,
  Ge4Se4, and Ge16Se32, matching Z=12, 4, 4, and 16 respectively.
- Extinctions: C centering removes GeP (10-1); Pnma removes odd h00 for GeS and
  GeSe; the P21/c c-glide removes odd 00l for beta-GeSe2. The tested doubled
  reflections are finite and nonzero.
- Scope/status: all four validation-ledger rows remain `unverified`; no
  experimental PXR/CBS agreement or human sign-off is claimed.

## Scope and concerns

Per the task boundary, no golden JSON, shared count assertion, or
`mats_to_sim.toml` file was changed. The expected future aggregate catalog
fixture/count reconciliation remains outside Task 3.

No source or parser blocker remains. GeSe2's one-number B=0.6 A2 is deliberately
not presented as experimental evidence and remains the only material-data
qualification in this task.

Independent review approved the task with no Critical or Important findings.
Its one Minor documentation note was addressed by recording the B placeholder
and (001)/(002) extinction contract directly in `gese2.cif` as well as in the
catalog and ledger.
