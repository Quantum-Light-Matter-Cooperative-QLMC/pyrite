# Plan: Bespoke per-material line grids

**Source spec:** `docs/superpowers/specs/2026-07-20-bespoke-line-grid-per-material-design.md`
**Execution:** subagent-driven-development, same session.
**Date:** 2026-07-20

## Session context (survives compaction — read first)

- **Worktree:** `/home/alexa/dev/cxr-mc/.claude/worktrees/bespoke-line-grids-impl`
  on branch `worktree-bespoke-line-grids-impl`. Run ALL commands from there.
- **Branch base / merge-base:** commit `074c932` (the azim=90 WIP commit).
  Branch started from it. Spec doc lives at commit `2a9aae6`.
- **Ledger:** `.superpowers/sdd/progress.md` (create on first task complete).
- **Venv is symlinked** to the main repo's `.venv` (avoids multi-GB CUDA
  re-download). The editable install still points at MAIN `src`, so tests MUST
  shadow it with PYTHONPATH. Canonical run recipe in this worktree:

  ```bash
  WT=/home/alexa/dev/cxr-mc/.claude/worktrees/bespoke-line-grids-impl
  rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache PYTHONPATH="$WT/src" \
    uv run --no-sync python -m pytest <files> -q
  ```
  Do NOT use `scripts/dev.py test` here — it re-syncs and re-downloads CUDA.
- **Baseline:** `tests/test_line_grid_bounds.py` + `tests/test_analyze_line_grid_bounds.py`
  = 22 tests passing at branch start.

## Scope boundary (IMPORTANT)

Code deliverable = **analyze script restructure + job script + their tests**,
verified with mocked `run_cases`. OUT of scope for this branch:

- **`src/cxr_mc/data/materials.toml` edits** — a human applies the bespoke
  values from the script's *report*, after running the derivation on the lab
  box (GPU). The script derives and reports; it never edits the TOML.
- **Golden fixture regeneration** (`tests/data/material_catalog_golden.json`) —
  downstream of the TOML edit, so not this branch.
- The catalog-override parse test (spec Testing bullet) uses a **synthetic**
  in-memory material carrying both grid keys, NOT the real TOML — the enabling
  fact (parser already merges scan overrides) is verified without a data edit.

## Global constraints (reviewer attention lens — exact values from spec)

- 4 crystals, fixed order: **hopg, diamond, wse2, mose2**.
- 7 beam energies: **30, 50, 100, 150, 200, 250, 300** keV.
- Line grid: only `stop`/`num` become bespoke; `start` kept from the profile's
  per-energy row. `num` preserves ~3 eV endpoint-inclusive spacing (existing
  `spacing_num`, `TARGET_SPACING_EV = 3.0`).
- Brem grid: ONE grid per material. `start = 0.0`, `step = 25.0`,
  `stop = max` over that material's 7 per-energy `brem_stop_eV` (each already
  +5%-margined, rounded to 100 eV via `margined_stop`).
- Wide brem diagnostic grid: `WIDE_BREM_STOP_EV = 40000.0`,
  `WIDE_BREM_STEP_EV = 25.0`, `WIDE_BREM_EV = np.arange(0.0, WIDE_BREM_STOP_EV, WIDE_BREM_STEP_EV)`.
- **No silent truncation.** Brem `coverage_energy` call stays
  `allow_shortfall=False` (its default) — a material whose brem 95% exceeds the
  diagnostic ceiling must raise `CoverageGridTooNarrow`, never clamp.
- Numeric core is UNCHANGED: `derive_bounds`, `_geometry_plan`, `_resume_phase`,
  `coverage_energy`, `margined_stop`, `spacing_num` keep their current bodies.
  The change is loop structure + file routing + the brem diagnostic grid.
- Per-material checkpoint files keep today's EXACT flat format:
  `<json_out>.<m>.json` + phase sidecar `<json_out>.<m>.json.phase.json`.
  (Reuse the proven resume/phase machinery verbatim per material.)
- Combined output JSON shape (written once every material complete):
  ```json
  { "hopg":   {"line_rows": [<7 rows>],
               "brem": {"stop_eV": <float>, "raw_eV": <float>, "step_eV": 25.0}},
    "diamond": {...}, "wse2": {...}, "mose2": {...} }
  ```
- Report: one per-material block.

## Key facts discovered (so implementers need not re-derive)

- `cxr_mc.config.material_sweep(material, *, theta_obs_deg=90.0, **overrides)`
  accepts `E_grid_brem` as an override (threaded to the sweep). `_build_case`
  currently passes `E_grid_line=WIDE_GRID_EV, E_grid_line_by_energy=None` and
  does NOT pass `E_grid_brem`, so the brem channel uses the profile's 30-keV
  grid. Add `E_grid_brem=WIDE_BREM_EV`.
- Brem coverage is measured from `result["brem_wide"]` on `result["E_grid_brem"]`
  in `_candidate_from_result` (already present). Widening the input grid widens
  these result arrays.
- `derive_bounds(materials, energies, ...)` already produces per-material output
  when handed a single-material list: `reference_scan = materials[0]`, and
  `driver = max(refined, key=coverage)` is that material's own worst geometry.
  So the orchestrator is an OUTER LOOP over single-material `derive_bounds`
  calls, not a core rewrite.
- `CoverageGridTooNarrow` + `coverage_energy(..., allow_shortfall=...)` live in
  `src/cxr_mc/line_grid_bounds.py`. Confirm the default is `allow_shortfall=False`
  during Task 1 (spec asserts it).
- Exit-75 "work remains" contract: `main()` returns 75 when `complete is False`.
  Orchestrator must thread `max_seconds` across the whole material×energy set
  and preserve this.

---

## Task 1 — Wide brem diagnostic grid

**File:** `scripts/analyze_line_grid_bounds.py` (+ `tests/test_analyze_line_grid_bounds.py`)
**Model:** cheap (mechanical, one file + test).

Changes:
1. Add module constants near `WIDE_GRID_*`:
   `WIDE_BREM_STOP_EV = 40000.0`, `WIDE_BREM_STEP_EV = 25.0`,
   `WIDE_BREM_EV = np.arange(0.0, WIDE_BREM_STOP_EV, WIDE_BREM_STEP_EV)`.
2. `_build_case`: pass `E_grid_brem=WIDE_BREM_EV` into `material_sweep(...)`.
3. Add `--brem-grid-stop` CLI flag (mirror `--grid-stop`; default
   `WIDE_BREM_STOP_EV`). In `main()`, rebuild the module `WIDE_BREM_EV` from
   `args.brem_grid_stop` and `WIDE_BREM_STEP_EV` (same `global` pattern used for
   `WIDE_GRID_EV`).
4. Verify the brem `coverage_energy` path keeps `allow_shortfall=False` (default).

Tests (TDD):
- `WIDE_BREM_EV` is passed into the sweep built by `_build_case` (assert the
  built case's brem grid equals `WIDE_BREM_EV`; or monkeypatch `material_sweep`
  and assert the `E_grid_brem` kwarg).
- With `allow_shortfall=False`, a synthetic brem intensity whose 95% mass sits
  ABOVE the diagnostic ceiling raises `CoverageGridTooNarrow`
  (drive `_candidate_from_result` or `coverage_energy` directly with a crafted
  `E_grid_brem`/`brem` pair).

**Verify:** run the two test files with the recipe above.

---

## Task 2 — Per-material orchestrator, checkpoints, brem aggregation, combined output

**File:** `scripts/analyze_line_grid_bounds.py` (+ tests)
**Model:** standard (multi-concern, integration with resume machinery).
**Depends on:** Task 1 (WIDE_BREM_EV present).

Add `derive_all_materials(materials, energies, ...)`:
- Iterate materials in the given order.
- Per material `m`, call the EXISTING `derive_bounds([m], energies, ...)` with
  its OWN checkpoint file `<json_out>.<m>.json` and phase sidecar
  `<json_out>.<m>.json.phase.json` (reuse `main()`'s existing resume-load and
  `_atomic_write_json`/`_save`/`_save_phase` wiring, but per material).
- Skip a material whose per-material file already has ALL requested energies.
- Resume an in-progress material from its phase sidecar.
- Thread the `--max-minutes` soft budget across the WHOLE material×energy set —
  checked between materials as well as between phases — and return the
  exit-75 "work remains" signal when exhausted (propagate a `complete` bool).
- Brem aggregation per material: `brem_stop = max` over that material's 7 rows'
  `brem_stop_eV`; emit `{"stop_eV": brem_stop, "raw_eV": <the row's brem_raw_eV
  that produced the max>, "step_eV": 25.0}`.
- Merge step (only when EVERY material complete): write combined `<json_out>`
  in the shape in Global Constraints, and print one report block per material
  (reuse `_print_report` per material with a per-material header line).

Rewire `main()` to call `derive_all_materials` instead of the single
`derive_bounds`. Preserve the exit-75 contract
(`test_main_returns_tempfail_when_budget_leaves_work` must still pass — update
it if the monkeypatch target moves from `derive_bounds` to
`derive_all_materials`, keeping the same behavioural assertion).

Tests (TDD):
- Orchestrator returns per-material rows; with mocked `run_cases`/`_scan_specs`/
  `_run_specs`, EACH material's driver is its own geometry (not a shared worst
  case). Assert distinct per-material output for ≥2 materials.
- Per-material resume: a partial `<out>.<m>.json` + its phase sidecar resumes
  that material; a completed per-material file is skipped (scanner not called).
- Brem aggregation: per-material brem `stop` equals the `max` of the 7
  per-energy `brem_stop_eV`.
- Combined output shape: keys are the materials, each with `line_rows` and
  `brem{stop_eV,raw_eV,step_eV}`.

**Verify:** run the analyze test file.

---

## Task 3 — Job script defaults + brem-grid-stop passthrough

**File:** `scripts/line_grid_bounds_job.py` (+ a small test if a job test exists;
otherwise assert via `--dry-run`/`_slice_payload`).
**Model:** cheap (mechanical).
**Independent of Tasks 1–2** (may run any time).

Changes:
1. `DEFAULT_ENERGIES = "30,50,100,150,200,250,300"` (all 7).
2. `DEFAULT_MATERIALS` already `"hopg,diamond,wse2,mose2"` — leave as-is.
3. Add `--brem-grid-stop` (default matching the analyze default, `40000.0`):
   thread it through `_slice_payload` (emit `--brem-grid-stop <g>` into the
   command), `_job_script`, `_metadata`, `start()`, and the `start` subparser.
4. Per-material checkpoint files persist across slices → sliced-SLURM resume
   unchanged (no code needed; note in report).

Tests (TDD):
- `_slice_payload(...)` output contains `--brem-grid-stop` with the passed
  value and `--energies 30,50,100,150,200,250,300` by default.
- `--brem-grid-stop` appears in `_metadata`.

**Verify:** run the job test (or a `python -c` assertion on `_slice_payload`).

---

## Task 4 — Catalog override parse test (synthetic)

**File:** the catalog test suite (`tests/test_material_catalog.py` or nearest).
**Model:** cheap.
**Independent.**

Add a test proving a material carrying BOTH `E_grid_line_by_energy` and
`E_grid_brem` overrides parses and its resolved `ScanSpec` grids differ from the
profile's — using a synthetic in-memory catalog/TOML fragment, NOT an edit to
the shipped `materials.toml`. This verifies the spec's "key enabling fact"
(parser merges per-material scan overrides) without touching shipped data.

**Verify:** run that test file with the recipe.

---

## Final

After all tasks: broad whole-branch review (most capable model), then
`superpowers:finishing-a-development-branch`. The bespoke TOML values + golden
regen happen in a SEPARATE follow-up after the lab-box derivation run — note
this in the branch-finish summary so it is not silently dropped.

## Pre-flight note for the executor

`tests/test_analyze_line_grid_bounds.py::test_main_returns_tempfail_when_budget_leaves_work`
monkeypatches `analyze.derive_bounds`. Task 2 moves `main()` onto
`derive_all_materials`; that test's patch target must move accordingly. This is
an expected, plan-sanctioned edit, not scope creep.
