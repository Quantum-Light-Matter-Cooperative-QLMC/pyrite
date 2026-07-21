# Bespoke per-material line grids

**Date:** 2026-07-20
**Status:** Design approved; implementation pending
**Supersedes (partially):** the uniform-grid output of
[`2026-07-16-line-grid-max-energy-design.md`](2026-07-16-line-grid-max-energy-design.md)

## Problem

Today every `profile = "standard"` material shares a single
`E_grid_line_by_energy` table (7 rows, 30–300 keV) defined in
`[profiles.standard]` of `src/cxr_mc/data/materials.toml`, plus a single shared
`E_grid_brem`. `scripts/analyze_line_grid_bounds.py` derives that table by
scanning **all** requested materials together and, per beam energy, keeping the
single worst-case *driver* — the material/geometry whose 95% coherent-line
coverage energy is highest. The result is one grid per energy, applied
uniformly.

That worst-case-driven grid is wasteful for materials whose emission lines top
out well below the driver's ceiling: e.g. a material whose coherent lines end
near 3 keV still receives the 300-keV driver's 16.4 keV / 5451-point grid, most
of whose bins are empty. Empty bins cost spectrum-evaluation time and memory
for no physics.

## Goal

Give each of the four standard scan-set crystals — **hopg, diamond, wse2,
mose2** (the existing `DEFAULT_MATERIALS` of `line_grid_bounds_job.py`) — its
own per-energy line grid and its own bremsstrahlung grid, each sized to *that
material's* measured intensity coverage at the currently-specified angle combos.
All other materials continue to inherit the shared `[profiles.standard]` grids
as a conservative fallback.

### In scope
- Bespoke `E_grid_line_by_energy` (per-energy `stop`/`num`) for the 4 crystals.
- Bespoke `E_grid_brem` (single grid per material) for the 4 crystals.
- All 7 standard beam energies (30, 50, 100, 150, 200, 250, 300 keV).

### Out of scope
- Any change to the catalog TOML schema (already supports per-material scan
  overrides — see below).
- Grid `start` energy: kept from the profile's per-energy row. `start` is a
  low-energy floor, not a material-specific driver; only `stop`/`num` become
  bespoke.
- Materials outside the 4 (they keep the shared profile grids).
- Automatic editing of `materials.toml`: the script derives and *reports*; a
  human applies the values, unchanged from today.

## Key enabling fact

No schema change is required. `materials/catalog.py::_parse_materials`
(around line 857) already merges any scan key present in a `[materials.X]`
block on top of its referenced profile:

```python
values = dict(profiles.get(str(profile), {}))
...
values.update({name: row[name] for name in _SCAN_KEYS if name in row})
scan = _scan(values, f"{path}.scan", ...)
```

So a `[materials.hopg]` block that declares its own `E_grid_line_by_energy`
and `E_grid_brem` overrides the profile for that material only. The change is
therefore **data + analyze-script restructure**, not a parser change.

## Design

### Chosen approach: outer material loop (Approach A)

Wrap the existing per-energy derivation in a per-material loop rather than
rewriting the ranking/refine core. `derive_bounds` already produces
per-material results when handed a single material: `reference_scan`, the
geometry plan, the `start_eV`, and the `driver = max(refined, key=coverage)`
all derive from `materials[0]`. With one material in the list, the driver is
that material's own worst geometry — exactly the bespoke bound wanted.

Rejected — Approach B (single combined coarse pass, group/refine per material):
same total compute (coarse Ne=200 is negligible; refine count is `top_k × 4`
either way), but it rewrites the ranking/refine core and yields coarser
checkpoint granularity. Approach A localizes the change to loop structure and
file routing, leaving the numeric core untouched.

### Data flow

```
analyze_line_grid_bounds.py \
    --materials hopg,diamond,wse2,mose2 \
    --energies 30,50,100,150,200,250,300

for m in materials:                          # NEW outer loop
    rows_m = derive_bounds([m], energies)    # 7 line rows; driver = m's own worst geometry
    brem_stop_m = max over rows_m of row["brem_stop_eV"]   # brem is one grid/scan → collapse energies

combined JSON:
  { "hopg":   {"line_rows": [7 rows], "brem": {"stop_eV": ..., "raw_eV": ..., "step_eV": 25.0}},
    "diamond": {...}, "wse2": {...}, "mose2": {...} }
report: one per-material block

HUMAN → materials.toml: each [materials.m] gains
  E_grid_line_by_energy (7 rows, bespoke stop/num, profile start) + E_grid_brem (bespoke stop)
[profiles.standard] grids remain the fallback for the other ~46 materials.
```

### Component changes

**`scripts/analyze_line_grid_bounds.py`**

- *Numeric core unchanged.* `derive_bounds`, `_geometry_plan`, `_resume_phase`,
  `coverage_energy`, `margined_stop`, `spacing_num` keep their current bodies.
- *Per-material checkpoint files.* Each material gets its own flat checkpoint
  in today's exact format: `<json_out>.<m>.json` and its phase sidecar
  `<json_out>.<m>.json.phase.json`. This reuses the proven resume/phase
  machinery verbatim per material — the sliced SLURM job depends on that
  format. A nested single file was rejected because it would force a rewrite of
  the `existing_rows` / phase-state resume logic.
- *Outer orchestrator* `derive_all_materials(materials, energies, …)`:
  - iterates materials in order;
  - skips a material whose per-material file is already complete (all requested
    energies present);
  - resumes an in-progress material from its phase sidecar;
  - threads the `--max-minutes` soft slice budget across the whole
    material × energy set — checked between materials as well as between phases
    — and returns the exit-75 "work remains" contract when the budget is
    exhausted.
- *Merge step:* once every material is complete, write the combined
  `<json_out>` and print one report block per material.
- *Brem aggregation:* `derive_bounds` already computes `brem_stop_eV` per
  energy row. The bespoke per-material brem `stop` is the `max` over that
  material's 7 rows (each already margined). `start = 0`, `step = 25` kept.

**Wide brem diagnostic grid**

`_build_case` currently leaves `E_grid_brem` at the profile's 30-keV grid while
overriding `E_grid_line` with the wide `WIDE_GRID_EV`. To *bound* brem per
material the diagnostic brem grid must likewise clear the widest true 95% brem
coverage energy, or `coverage_energy` would truncate.

- Add `WIDE_BREM_STOP_EV` (default 40000.0), `WIDE_BREM_STEP_EV` (25.0), and
  `WIDE_BREM_EV = np.arange(0.0, WIDE_BREM_STOP_EV, WIDE_BREM_STEP_EV)`.
- Pass `E_grid_brem=WIDE_BREM_EV` into the diagnostic `material_sweep(...)`.
- Add a `--brem-grid-stop` CLI flag mirroring `--grid-stop`.
- The brem `coverage_energy` call stays `allow_shortfall=False`, so a material
  whose brem 95% exceeds the diagnostic ceiling fails loud with
  `CoverageGridTooNarrow` — the same no-silent-truncation guarantee the line
  channel already enforces (issue_notes.md #1). 40 keV carries headroom over
  the current 30-keV grid, which resolves brem 95% for these materials at
  ≤300 keV today (the existing job runs without tripping the guard).

**`scripts/line_grid_bounds_job.py`**

- `DEFAULT_ENERGIES` → `"30,50,100,150,200,250,300"` (all 7).
- `DEFAULT_MATERIALS` already `"hopg,diamond,wse2,mose2"`.
- Add `--brem-grid-stop` passthrough into the slice command.
- Per-material checkpoint files persist across slices, so sliced-SLURM resume
  is unchanged.

**`src/cxr_mc/data/materials.toml`**

Each of the 4 `[materials.X]` blocks gains, from the report:

```toml
E_grid_line_by_energy = [
  { energy_keV = 30.0,  grid = { linspace = { start = <profile start>, stop = <bespoke>, num = <bespoke>, endpoint = true } } },
  ... 7 rows ...
]
E_grid_brem = { arange = { start = 0.0, stop = <bespoke>, step = 25.0 } }
```

`[profiles.standard]` `E_grid_line_by_energy` and `E_grid_brem` are left as-is,
serving the remaining materials.

### Testing

- **Orchestrator:** `derive_all_materials` returns per-material rows; with a
  mocked `run_cases`, each material's driver is its own geometry (not a shared
  worst case).
- **Per-material resume:** a partial `<out>.<m>.json` plus its phase sidecar
  resumes that material correctly; a completed material is skipped.
- **Brem aggregation:** per-material brem `stop` equals the `max` of the 7
  per-energy `brem_stop_eV`.
- **Wide brem grid:** `WIDE_BREM_EV` is wired into `_build_case`;
  `CoverageGridTooNarrow` is raised when a material's brem 95% exceeds the
  diagnostic ceiling and `allow_shortfall` is off.
- **Catalog:** a material carrying both `E_grid_line_by_energy` and
  `E_grid_brem` overrides parses and its resolved `ScanSpec` grids differ from
  the profile's.
- **Golden fixture:** regenerate `tests/data/material_catalog_golden.json`; the
  4 materials' scan grids change.
- **Existing:** update `tests/test_analyze_line_grid_bounds.py` for the new
  combined output shape and per-material files.

## Risks / open items

- **Compute:** four independent coarse+refine passes. Refine (Ne=2000) is the
  cost driver at ~9 min/case on qlmc; total refines ≈ `top_k × 4 materials ×
  7 energies`. Runs sliced via `line_grid_bounds_job.py` on the lab box.
- **Brem ceiling assumption:** 40 keV diagnostic ceiling assumes brem 95%
  coverage stays below it for these materials at ≤300 keV. If a material trips
  `CoverageGridTooNarrow`, widen `--brem-grid-stop` — the failure is explicit,
  never silent.
