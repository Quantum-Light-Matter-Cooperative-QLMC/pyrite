# `cxr line-grid` Command Group Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold `scripts/line_grid_bounds_job.py` and `scripts/analyze_line_grid_bounds.py` into a `cxr line-grid` subcommand group that derives per-material line-grid bounds (locally or on qlmc), applies them to `materials.toml` with zero manual editing, supports CLI-managed manual overrides + notes, geometry/thickness overrides, persistent defaults, and independent golden regeneration.

**Architecture:** New `src/cxr_mc/line_grid/` package. Job management (`status`/`attach`/`logs`/`stop`) delegates to `remote.py` for byte-identical output; the SLURM payload switches from a path-invoked script to `python -m cxr_mc.line_grid.derive`. `apply` regenerates only three owned regions of `materials.toml` (per-material `E_grid_line_by_energy`, `E_grid_brem`, and `[profiles.standard] energy_keV`) by surgical text replacement, no new dependency. Provenance/notes and persistent defaults live in tool-owned sidecar TOMLs, leaving the material catalog schema and its golden untouched except for real bound-value changes.

**Tech Stack:** Python 3, `argparse`, stdlib `tomllib` (read), hand-emitted TOML text (write), NumPy, existing `cxr_mc.remote` / `cxr_mc.montecarlo.runner` / `cxr_mc.materials` machinery, pytest.

## Global Constraints

- Run everything through the shared uv cache: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test [path -k name]`. Bare `uv run` fails (read-only default cache).
- `scripts/dev.py lint|typecheck|format` take NO path args; run repo-wide, or `uv run python -m ruff check <file>` / `uv run pyright <file>` for one file.
- New/edited physics needs a derivation docstring + `Validation: <id>` marker + ledger row. **This plan adds NO new physics** — derivation math is unchanged; only its packaging, CLI, and I/O move. No ledger changes.
- Notebook changes stay output-free; not relevant here.
- Only stdlib `tomllib` is available for TOML — it is read-only. All TOML writing is hand-emitted text (surgical) or simple emitters. Do NOT add `tomlkit`/`tomli_w`.
- `materials.toml` catalog decoder (`src/cxr_mc/materials/_catalog_decode.py`) must NOT be modified — provenance is a sidecar, not a catalog field.
- Commit after every task. Keep the full test suite green at each commit.

---

## File Structure

```
src/cxr_mc/line_grid/
  __init__.py    # cxr line-grid CLI: add_subparser + _cli_* handlers; delegates job mgmt to remote.py
  bounds.py      # moved from src/cxr_mc/line_grid_bounds.py — coverage_energy/margined_stop/spacing_num + CoverageGridTooNarrow
  derive.py      # moved from scripts/analyze_line_grid_bounds.py — derivation + `python -m` entry
  job.py         # moved from scripts/line_grid_bounds_job.py — submit + SLURM job-script builder
  defaults.py    # read/write src/cxr_mc/data/line_grid_defaults.toml
  provenance.py  # read/write src/cxr_mc/data/line_grid_provenance.toml
  apply.py       # materials.toml surgical write-back + set/set-brem/show
  golden.py      # regen-golden independent serializer

src/cxr_mc/data/
  line_grid_defaults.toml     # NEW, tool-owned
  line_grid_provenance.toml   # NEW, tool-owned (may be absent → treated empty)

tests/
  test_line_grid_bounds.py         # reimport cxr_mc.line_grid.bounds
  test_line_grid_derive.py         # renamed from test_analyze_line_grid_bounds.py
  test_line_grid_job.py            # renamed from test_line_grid_bounds_job.py
  test_line_grid_defaults.py       # NEW
  test_line_grid_provenance.py     # NEW
  test_line_grid_apply.py          # NEW
  test_line_grid_golden.py         # NEW
  test_line_grid_cli.py            # NEW (status/attach delegation, dispatch)
```

Deleted: `scripts/line_grid_bounds_job.py`, `scripts/analyze_line_grid_bounds.py`, `src/cxr_mc/line_grid_bounds.py`.

---

## Task 1: Create package, move `bounds.py`

Move the pure math helpers into the new package and repoint importers. Keeps the suite green before any behavioral change.

**Files:**
- Create: `src/cxr_mc/line_grid/__init__.py` (empty for now)
- Move: `src/cxr_mc/line_grid_bounds.py` → `src/cxr_mc/line_grid/bounds.py`
- Modify importers: `scripts/analyze_line_grid_bounds.py`, `tests/test_line_grid_bounds.py`, `src/cxr_mc/sweep.py` (docstring mention only)

**Interfaces:**
- Produces: `cxr_mc.line_grid.bounds.{coverage_energy, margined_stop, spacing_num, CoverageGridTooNarrow}` — same signatures as today's `line_grid_bounds`.

- [ ] **Step 1: Move the file with git**

```bash
mkdir -p src/cxr_mc/line_grid
git mv src/cxr_mc/line_grid_bounds.py src/cxr_mc/line_grid/bounds.py
: > src/cxr_mc/line_grid/__init__.py
```

- [ ] **Step 2: Repoint the importers**

In `tests/test_line_grid_bounds.py` change the import:

```python
from cxr_mc.line_grid.bounds import (
    CoverageGridTooNarrow,
    coverage_energy,
    margined_stop,
    spacing_num,
)
```

In `scripts/analyze_line_grid_bounds.py` change:

```python
from cxr_mc.line_grid.bounds import coverage_energy, margined_stop, spacing_num
```

Grep for any other reference and update the module path text only (e.g. the `src/cxr_mc/sweep.py` docstring mentioning the analyze diagnostic — leave prose, just fix any literal `cxr_mc.line_grid_bounds`):

```bash
TOKENSAVE_DISABLE_GREP_HOOK=1 rg -l "cxr_mc\.line_grid_bounds|line_grid_bounds import" src tests scripts
```

- [ ] **Step 3: Run the moved module's tests**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_bounds.py`
Expected: PASS (same tests, new import path).

- [ ] **Step 4: Run the analyze tests (still via script, unmoved)**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_analyze_line_grid_bounds.py`
Expected: PASS (script still imports the new path).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(line-grid): move line_grid_bounds into cxr_mc.line_grid package"
```

---

## Task 2: Move derivation into `line_grid/derive.py`

Relocate `scripts/analyze_line_grid_bounds.py` verbatim into the package with a `python -m` entry, delete the script, repoint its test. No behavior change yet.

**Files:**
- Move: `scripts/analyze_line_grid_bounds.py` → `src/cxr_mc/line_grid/derive.py`
- Rename test: `tests/test_analyze_line_grid_bounds.py` → `tests/test_line_grid_derive.py`
- Delete: (script removed by the move)

**Interfaces:**
- Produces: `cxr_mc.line_grid.derive.main(argv=None) -> int`, `derive_all_materials(...)`, `build_parser()`, module constants `DIAGNOSTIC_THICKNESS_ANG`, `WIDE_GRID_STOP_EV`, `WIDE_BREM_STOP_EV`, `WIDE_BREM_STEP_EV`, and the `python -m cxr_mc.line_grid.derive` CLI with today's exact flags (`--materials --energies --top-k --coarse-ne --refine-ne --max-workers --coarse-engine --grid-stop --grid-step --brem-grid-stop --json-out --max-minutes`).

- [ ] **Step 1: Move the file**

```bash
git mv scripts/analyze_line_grid_bounds.py src/cxr_mc/line_grid/derive.py
git mv tests/test_analyze_line_grid_bounds.py tests/test_line_grid_derive.py
```

- [ ] **Step 2: Fix the module's own bounds import + drop the shebang/path comment**

In `src/cxr_mc/line_grid/derive.py` the bounds import is already `from cxr_mc.line_grid.bounds import ...` (fixed in Task 1). Remove the now-wrong `# scripts/analyze_line_grid_bounds.py` path comment at the top and update the usage examples in the module docstring to:

```python
    python -m cxr_mc.line_grid.derive
    python -m cxr_mc.line_grid.derive --materials hopg,diamond --energies 30,50
    python -m cxr_mc.line_grid.derive --json-out /tmp/line_grid_bounds.json
```

Keep `if __name__ == "__main__": raise SystemExit(main())` — that is the `python -m` entry.

- [ ] **Step 3: Repoint the test loader**

`tests/test_line_grid_derive.py` currently loads the script by file path via a `_load_script("analyze_line_grid_bounds")` helper. Replace that helper and its call sites with a direct import:

```python
from cxr_mc.line_grid import derive as analyze
```

Delete the `_load_script` helper and every `analyze = _load_script(...)` line (the module object is now imported once at top). `CoverageGridTooNarrow` import becomes:

```python
from cxr_mc.line_grid.bounds import CoverageGridTooNarrow
```

- [ ] **Step 4: Verify the module runs as `-m`**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python -m cxr_mc.line_grid.derive --materials hopg --energies 30 --coarse-ne 1 --refine-ne 1 --max-minutes 0.001 --json-out /tmp/lg_smoke.json`
Expected: exits 0 or 75 (budget), prints a report or "work remains" — no import error.

- [ ] **Step 5: Run the derive tests**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_derive.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(line-grid): move derivation to cxr_mc.line_grid.derive with python -m entry"
```

---

## Task 3: Persistent defaults sidecar (`defaults.py`)

Tool-owned defaults file with partial-merge writes and today's-behavior fallbacks.

**Files:**
- Create: `src/cxr_mc/line_grid/defaults.py`
- Create: `src/cxr_mc/data/line_grid_defaults.toml`
- Test: `tests/test_line_grid_defaults.py`

**Interfaces:**
- Produces:
  - `load_defaults() -> dict` returning keys `tilts:list[float]`, `azimuths:list[float]`, `thickness_ang:list[float]`, `brem_step_ev:float`, `energies:list[float]`, `materials:list[str]`. Missing file → built-in `FALLBACK` dict.
  - `update_defaults(**changes) -> dict` — merge only supplied keys over the current file, atomic-write, return the new dict. `changes` values of `None` are ignored.
  - `FALLBACK: dict` — the built-in defaults equal to today's constants.
  - `DEFAULTS_PATH: Path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_line_grid_defaults.py
import cxr_mc.line_grid.defaults as d


def test_load_missing_file_returns_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "DEFAULTS_PATH", tmp_path / "absent.toml")
    got = d.load_defaults()
    assert got["thickness_ang"] == [1.0e7]
    assert got["brem_step_ev"] == 25.0
    assert got["tilts"] == []  # [] = fall back to profile angles
    assert got["energies"] == [30, 40, 50, 60, 100, 150, 200, 250, 300]


def test_update_is_partial_merge_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "DEFAULTS_PATH", tmp_path / "def.toml")
    d.update_defaults(thickness_ang=[5.0e6, 1.0e7], brem_step_ev=None)
    got = d.load_defaults()
    assert got["thickness_ang"] == [5.0e6, 1.0e7]
    assert got["brem_step_ev"] == 25.0          # None ignored → fallback kept
    assert got["materials"] == d.FALLBACK["materials"]  # untouched key preserved
```

- [ ] **Step 2: Run to verify it fails**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_defaults.py`
Expected: FAIL — `ModuleNotFoundError: cxr_mc.line_grid.defaults`.

- [ ] **Step 3: Implement `defaults.py`**

```python
# src/cxr_mc/line_grid/defaults.py
"""Tool-owned persistent defaults for `cxr line-grid` (NOT the material catalog).

Stored at src/cxr_mc/data/line_grid_defaults.toml so derive/submit read a single
source of standing defaults that `--set-default` / `defaults --set` update. Empty
`tilts`/`azimuths` mean "use each material's profile tilt_deg/tilt_azim_deg" —
today's behavior — so a missing file reproduces current output exactly.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path

DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "line_grid_defaults.toml"

FALLBACK: dict = {
    "tilts": [],
    "azimuths": [],
    "thickness_ang": [1.0e7],
    "brem_step_ev": 25.0,
    "energies": [30, 40, 50, 60, 100, 150, 200, 250, 300],
    "materials": ["hopg", "diamond", "wse2", "mose2"],
}


def load_defaults() -> dict:
    merged = dict(FALLBACK)
    if DEFAULTS_PATH.exists():
        with open(DEFAULTS_PATH, "rb") as f:
            merged.update(tomllib.load(f))
    return merged


def _emit(values: dict) -> str:
    def scalar(v):
        if isinstance(v, str):
            return f'"{v}"'
        return repr(v)

    lines = ["# managed by `cxr line-grid defaults --set`; edit via CLI, not by hand", ""]
    for key in ("tilts", "azimuths", "thickness_ang", "energies", "materials"):
        items = ", ".join(scalar(x) for x in values[key])
        lines.append(f"{key} = [{items}]")
    lines.append(f"brem_step_ev = {values['brem_step_ev']!r}")
    lines.append("")
    return "\n".join(lines)


def update_defaults(**changes) -> dict:
    values = load_defaults()
    for key, val in changes.items():
        if val is None:
            continue
        if key not in FALLBACK:
            raise KeyError(f"unknown default {key!r}")
        values[key] = val
    text = _emit(values)
    DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(DEFAULTS_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, DEFAULTS_PATH)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return values
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_defaults.py`
Expected: PASS.

- [ ] **Step 5: Seed the checked-in defaults file**

Create `src/cxr_mc/data/line_grid_defaults.toml` with the fallback content so the package ships a real file:

```toml
# managed by `cxr line-grid defaults --set`; edit via CLI, not by hand

tilts = []
azimuths = []
thickness_ang = [10000000.0]
energies = [30, 40, 50, 60, 100, 150, 200, 250, 300]
materials = ["hopg", "diamond", "wse2", "mose2"]
brem_step_ev = 25.0
```

- [ ] **Step 6: Commit**

```bash
git add src/cxr_mc/line_grid/defaults.py src/cxr_mc/data/line_grid_defaults.toml tests/test_line_grid_defaults.py
git commit -m "feat(line-grid): tool-owned persistent defaults sidecar"
```

---

## Task 4: Geometry/thickness overrides in `derive.py`

Thread `--tilts/--azimuths/--thickness` through the diagnostic geometry and read defaults from Task 3. Unset flags reproduce today's scan.

**Files:**
- Modify: `src/cxr_mc/line_grid/derive.py` (`_build_case`, `_geometry_plan`, `derive_bounds`, `derive_all_materials`, `build_parser`, `main`)
- Test: `tests/test_line_grid_derive.py` (add cases)

**Interfaces:**
- Consumes: `cxr_mc.line_grid.defaults.load_defaults`.
- Produces: `_geometry_plan(materials, reference_scan, tilts=None, azimuths=None, thicknesses=None)` returning coarse specs of `(material, tilt, azim, thickness)` 4-tuples; `_build_case(material, energy_keV, tilt_deg, tilt_azim_deg, thickness_ang, n_electrons)`. `main` gains `--tilts --azimuths --thickness --set-default --brem-step` flags.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_line_grid_derive.py
from cxr_mc.line_grid import derive as analyze


def test_geometry_plan_defaults_to_profile_angles_single_thickness():
    class Scan:
        tilt_deg = [5.0, 45.0]
        tilt_azim_deg = [100.0, 180.0]
    coarse, _ = analyze._geometry_plan(["hopg"], Scan())
    # 2 tilts x 2 azimuths x 1 default thickness, all 4-tuples ending in 1e7
    assert len(coarse) == 4
    assert all(len(spec) == 4 for spec in coarse)
    assert {spec[3] for spec in coarse} == {analyze.DIAGNOSTIC_THICKNESS_ANG}


def test_geometry_plan_overrides_expand_thickness_axis():
    class Scan:
        tilt_deg = [5.0]
        tilt_azim_deg = [180.0]
    coarse, _ = analyze._geometry_plan(
        ["hopg"], Scan(), tilts=[5.0], azimuths=[180.0], thicknesses=[1.0e6, 1.0e7]
    )
    assert len(coarse) == 2
    assert sorted(spec[3] for spec in coarse) == [1.0e6, 1.0e7]
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_derive.py -k geometry_plan`
Expected: FAIL — `_geometry_plan` takes 2 args / specs are 3-tuples.

- [ ] **Step 3: Update `_build_case` and `_geometry_plan`**

Add `thickness_ang` param to `_build_case` (replacing the pinned `DIAGNOSTIC_THICKNESS_ANG` in the `material_sweep(...)` call):

```python
def _build_case(material, energy_keV, tilt_deg, tilt_azim_deg, thickness_ang, n_electrons):
    sweep = material_sweep(
        material,
        thickness_ang=thickness_ang,
        energy_keV=energy_keV,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        E_grid_line=WIDE_GRID_EV,
        E_grid_line_by_energy=None,
        E_grid_brem=WIDE_BREM_EV,
    )
    return build_cases(sweep, n_electrons=n_electrons)[0]
```

Rewrite `_geometry_plan` to accept overrides and emit 4-tuples:

```python
def _geometry_plan(materials, reference_scan, tilts=None, azimuths=None, thicknesses=None):
    raw_tilts = tilts if tilts else reference_scan.tilt_deg
    raw_azims = azimuths if azimuths else reference_scan.tilt_azim_deg
    thicks = thicknesses if thicknesses else [DIAGNOSTIC_THICKNESS_ANG]
    tilt_vals = [float(v) for v in _quantized_angles(raw_tilts)]
    azim_vals = [float(v) for v in _quantized_angles(raw_azims)]
    coarse = [
        (material, tilt, azim, float(thick))
        for material in materials
        for tilt in tilt_vals
        for azim in azim_vals
        for thick in thicks
    ]
    return coarse, []
```

- [ ] **Step 4: Thread 4-tuples through `_run_specs`/`_candidate_from_result`/`_resume_phase`/`derive_bounds`**

Every place that unpacks `(material, tilt_deg, tilt_azim_deg)` becomes `(material, tilt_deg, tilt_azim_deg, thickness_ang)` and passes `thickness_ang` into `_build_case`. In `_run_specs`:

```python
def _run_specs(specs, energy_keV, n_electrons, max_workers, engine):
    cases = [
        _build_case(material, energy_keV, tilt_deg, tilt_azim_deg, thickness_ang, n_electrons)
        for material, tilt_deg, tilt_azim_deg, thickness_ang in specs
    ]
    results = run_cases(cases, max_workers=max_workers, engine=engine)
    return [
        _candidate_from_result(material, tilt_deg, tilt_azim_deg, result)
        for (material, tilt_deg, tilt_azim_deg, _thick), result in zip(specs, results, strict=True)
    ]
```

In `derive_bounds`, the refined-specs rebuild list becomes 4-tuples carrying each candidate's thickness. Add a `thickness` field to `Candidate` (default `DIAGNOSTIC_THICKNESS_ANG`) so the refine phase can reconstruct the geometry:

```python
@dataclass(frozen=True)
class Candidate:
    material: str
    tilt_deg: float
    tilt_azim_deg: float
    thickness_ang: float
    coverage_energy_eV: float
    total_intensity: float
    incoherent_coverage_energy_eV: float
    incoherent_total_intensity: float
```

Update `_candidate_from_result(material, tilt_deg, tilt_azim_deg, thickness_ang, result)` to accept and store thickness, and its callers. The refine rebuild:

```python
[(c.material, c.tilt_deg, c.tilt_azim_deg, c.thickness_ang) for c in candidates]
```

- [ ] **Step 5: Plumb overrides through `derive_bounds`/`derive_all_materials`**

Add `tilts=None, azimuths=None, thicknesses=None` params to `derive_bounds` and `derive_all_materials`, pass them to `_geometry_plan`.

- [ ] **Step 6: Add CLI flags + defaults read in `main`/`build_parser`**

In `build_parser` add:

```python
    parser.add_argument("--tilts", default=None, help="comma polar tilts (deg); default: profile/persistent")
    parser.add_argument("--azimuths", default=None, help="comma azimuths (deg); default: profile/persistent")
    parser.add_argument("--thickness", default=None, help="comma crystal thicknesses (Angstrom)")
    parser.add_argument("--brem-step", type=float, default=None, help="E_grid_brem step (eV) for downstream apply")
    parser.add_argument("--set-default", action="store_true", help="persist supplied geometry/energies/materials as new defaults")
```

In `main`, resolve from persistent defaults when a flag is unset, and honor `--set-default`:

```python
from cxr_mc.line_grid import defaults as lg_defaults

def _floats(s):
    return [float(x) for x in s.split(",")] if s else None

def main(argv=None):
    args = build_parser().parse_args(argv)
    persisted = lg_defaults.load_defaults()
    tilts = _floats(args.tilts) or (persisted["tilts"] or None)
    azimuths = _floats(args.azimuths) or (persisted["azimuths"] or None)
    thicknesses = _floats(args.thickness) or (persisted["thickness_ang"] or None)
    global WIDE_GRID_EV, WIDE_BREM_EV
    WIDE_GRID_EV = np.arange(WIDE_GRID_START_EV, args.grid_stop, args.grid_step)
    WIDE_BREM_EV = np.arange(0.0, args.brem_grid_stop, WIDE_BREM_STEP_EV)
    materials = args.materials.split(",") if args.materials else list(CATALOG.materials)
    if args.energies:
        energies = [float(e) for e in args.energies.split(",")]
    else:
        energies = [float(e) for e in CATALOG.material(materials[0]).scan.energy_keV]
    if args.set_default:
        lg_defaults.update_defaults(
            tilts=_floats(args.tilts), azimuths=_floats(args.azimuths),
            thickness_ang=_floats(args.thickness),
            brem_step_ev=args.brem_step,
            energies=energies if args.energies else None,
            materials=materials if args.materials else None,
        )
    combined, complete = derive_all_materials(
        materials, energies, args.top_k, args.coarse_ne, args.refine_ne, args.max_workers,
        json_out=args.json_out, coarse_engine=args.coarse_engine,
        max_seconds=None if args.max_minutes is None else args.max_minutes * 60.0,
        tilts=tilts, azimuths=azimuths, thicknesses=thicknesses,
    )
    # ... unchanged reporting/return ...
```

- [ ] **Step 7: Run the full derive test file + a regression smoke**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_derive.py`
Expected: PASS (existing tests still pass — unset flags reproduce the profile-angle, single-thickness scan).

- [ ] **Step 8: Commit**

```bash
git add src/cxr_mc/line_grid/derive.py tests/test_line_grid_derive.py
git commit -m "feat(line-grid): geometry/thickness overrides + persistent-default read in derive"
```

---

## Task 5: Move submit/job builder into `job.py`, switch payload to `-m`

Relocate `scripts/line_grid_bounds_job.py`, switch the SLURM payload from a path-invoked script to `python -m cxr_mc.line_grid.derive`, drop the script from `SYNC_PATHS`.

**Files:**
- Move: `scripts/line_grid_bounds_job.py` → `src/cxr_mc/line_grid/job.py`
- Modify: `src/cxr_mc/remote.py` (`SYNC_PATHS`: remove `"scripts/analyze_line_grid_bounds.py"`)
- Rename test: `tests/test_line_grid_bounds_job.py` → `tests/test_line_grid_job.py`

**Interfaces:**
- Produces: `cxr_mc.line_grid.job.{start, _slice_payload, _job_script, _metadata, DEFAULT_*}` — same signatures as today. `start(...) -> jobid`.

- [ ] **Step 1: Move the files**

```bash
git mv scripts/line_grid_bounds_job.py src/cxr_mc/line_grid/job.py
git mv tests/test_line_grid_bounds_job.py tests/test_line_grid_job.py
```

- [ ] **Step 2: Switch the payload command to the module**

In `src/cxr_mc/line_grid/job.py`, `_slice_payload`, replace the script path invocation line:

```python
        "run --no-sync python -m cxr_mc.line_grid.derive",
```

(was `"run --no-sync python scripts/analyze_line_grid_bounds.py"`). Update the two help prints in `start`:

```python
    print(f"status: cxr line-grid status {jobid}")
    print(f"attach: cxr line-grid attach {jobid}")
```

- [ ] **Step 3: Remove the deleted script from SYNC_PATHS**

In `src/cxr_mc/remote.py`, delete the `"scripts/analyze_line_grid_bounds.py",` line from `SYNC_PATHS` (the derivation now ships under `src`, already synced).

- [ ] **Step 4: Repoint the job test**

In `tests/test_line_grid_job.py` replace the by-path loader with a direct import:

```python
from cxr_mc.line_grid import job
```

Delete the `spec_from_file_location("line_grid_bounds_job", ...)` block. Update the payload assertion to expect the module command, and update/remove the `SYNC_PATHS` assertion:

```python
    assert "python -m cxr_mc.line_grid.derive" in payload
    assert "scripts/analyze_line_grid_bounds.py" not in remote.SYNC_PATHS
```

- [ ] **Step 5: Run the job tests**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_job.py tests/test_remote.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(line-grid): move submit into cxr_mc.line_grid.job; SLURM payload uses python -m derive"
```

---

## Task 6: Provenance sidecar (`provenance.py`)

Tool-owned source/note store keyed by material / channel / energy; drives sticky-manual and `show`.

**Files:**
- Create: `src/cxr_mc/line_grid/provenance.py`
- Test: `tests/test_line_grid_provenance.py`

**Interfaces:**
- Produces:
  - `load() -> dict` — nested `{material: {"brem": {...}, "line": {energy_str: {...}}}}`; missing file → `{}`.
  - `get_line(material, energy) -> dict | None`, `get_brem(material) -> dict | None`.
  - `set_line(material, energy, source, note=None)`, `set_brem(material, source, note=None)` — merge + atomic write.
  - `is_manual_line(material, energy) -> bool`, `is_manual_brem(material) -> bool`.
  - `PROVENANCE_PATH: Path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_line_grid_provenance.py
import cxr_mc.line_grid.provenance as p


def test_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "PROVENANCE_PATH", tmp_path / "absent.toml")
    assert p.load() == {}
    assert p.is_manual_line("hopg", 60.0) is False


def test_set_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "PROVENANCE_PATH", tmp_path / "prov.toml")
    p.set_line("hopg", 60.0, "manual", note="widened for detector X")
    p.set_brem("hopg", "derived job 458 (2026-07-22)")
    assert p.is_manual_line("hopg", 60.0) is True
    assert p.get_line("hopg", 60.0)["note"] == "widened for detector X"
    assert p.is_manual_brem("hopg") is False
    assert p.get_brem("hopg")["source"].startswith("derived job 458")
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_provenance.py`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `provenance.py`**

```python
# src/cxr_mc/line_grid/provenance.py
"""Tool-owned source/note store for cxr line-grid bounds (NOT the catalog).

materials.toml holds pure grid values; this sidecar records who set each grid and
why, keyed material -> channel -> energy. Used for sticky-manual protection in
`apply` and for `show`. Absent file == no provenance (everything "derived").
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path

PROVENANCE_PATH = Path(__file__).resolve().parent.parent / "data" / "line_grid_provenance.toml"


def load() -> dict:
    if not PROVENANCE_PATH.exists():
        return {}
    with open(PROVENANCE_PATH, "rb") as f:
        return tomllib.load(f)


def _energy_key(energy) -> str:
    return f"{float(energy):g}"


def get_line(material, energy):
    return load().get(material, {}).get("line", {}).get(_energy_key(energy))


def get_brem(material):
    return load().get(material, {}).get("brem")


def is_manual_line(material, energy) -> bool:
    rec = get_line(material, energy)
    return bool(rec and rec.get("source") == "manual")


def is_manual_brem(material) -> bool:
    rec = get_brem(material)
    return bool(rec and rec.get("source") == "manual")


def _record(source, note):
    rec = {"source": source}
    if note:
        rec["note"] = note
    return rec


def set_line(material, energy, source, note=None):
    data = load()
    mat = data.setdefault(material, {})
    mat.setdefault("line", {})[_energy_key(energy)] = _record(source, note)
    _write(data)


def set_brem(material, source, note=None):
    data = load()
    data.setdefault(material, {})["brem"] = _record(source, note)
    _write(data)


def _emit(data: dict) -> str:
    lines = ["# managed by cxr line-grid; do not hand-edit", ""]

    def block(header, rec):
        lines.append(f"[{header}]")
        lines.append(f'source = "{rec["source"]}"')
        if "note" in rec:
            lines.append(f'note = "{rec["note"]}"')
        lines.append("")

    for material in sorted(data):
        entry = data[material]
        if "brem" in entry:
            block(f"{material}.brem", entry["brem"])
        for energy in sorted(entry.get("line", {}), key=float):
            block(f"{material}.line.{energy}", entry["line"][energy])
    return "\n".join(lines)


def _write(data: dict):
    text = _emit(data)
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(PROVENANCE_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, PROVENANCE_PATH)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
```

- [ ] **Step 4: Run the tests to verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_provenance.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/line_grid/provenance.py tests/test_line_grid_provenance.py
git commit -m "feat(line-grid): tool-owned provenance sidecar for source/note + sticky-manual"
```

---

## Task 7: `apply.py` — materials.toml write-back

Surgical regeneration of the three owned regions from a combined derivation JSON, honoring sticky-manual via the provenance sidecar, inserting new energies into the profile array, validating by re-parsing.

**Files:**
- Create: `src/cxr_mc/line_grid/apply.py`
- Test: `tests/test_line_grid_apply.py`

**Interfaces:**
- Consumes: `cxr_mc.line_grid.provenance`, `cxr_mc.line_grid.bounds.spacing_num`, `cxr_mc.materials.CATALOG` (for validation reload only).
- Produces:
  - `emit_line_block(rows) -> str` — the `E_grid_line_by_energy = [ … ]` text (one inline table per line, `start=10.0` at ≤60 keV else `50.0`, `endpoint=true`).
  - `emit_brem_line(stop_eV, step_eV) -> str` — the `E_grid_brem = { arange = {...} }` text.
  - `apply_bounds(toml_text, combined, *, force=False, provenance_mod=provenance) -> tuple[str, list[str]]` — returns `(new_toml_text, skipped)` where `skipped` names material/energy entries left untouched because sticky-manual. Pure string transform (no disk I/O) for testability.
  - `apply_file(json_path, *, materials=None, force=False, dry_run=False) -> None` — load JSON + `materials.toml`, call `apply_bounds`, stamp provenance for written entries, write (unless dry-run), reload catalog to validate.

- [ ] **Step 1: Write the failing tests (pure transform first)**

```python
# tests/test_line_grid_apply.py
import tomllib

from cxr_mc.line_grid import apply

BASE_TOML = '''schema_version = 1

[profiles.standard]
energy_keV = { values = [30.0, 100.0] }

[materials.hopg]
label = "HOPG"
profile = "standard"
E_grid_line_by_energy = [
  { energy_keV = 30.0, grid = { linspace = { start = 10.0, stop = 2600.0, num = 864, endpoint = true } } },
  { energy_keV = 100.0, grid = { linspace = { start = 50.0, stop = 4600.0, num = 1518, endpoint = true } } },
]
E_grid_brem = { arange = { start = 0.0, stop = 136500.0, step = 25.0 } }
'''

COMBINED = {
    "hopg": {
        "line_rows": [
            {"energy_keV": 30.0, "start_eV": 10.0, "stop_eV": 2700.0, "num": 897},
            {"energy_keV": 100.0, "start_eV": 50.0, "stop_eV": 4600.0, "num": 1518},
        ],
        "brem": {"stop_eV": 140000.0, "step_eV": 25.0, "raw_eV": 133000.0},
    }
}


class _NoManual:
    def is_manual_line(self, *a):
        return False

    def is_manual_brem(self, *a):
        return False


def test_apply_rewrites_only_owned_blocks_and_reparses():
    new_text, skipped = apply.apply_bounds(BASE_TOML, COMBINED, provenance_mod=_NoManual())
    assert skipped == []
    assert "stop = 2700.0, num = 897" in new_text
    assert 'label = "HOPG"' in new_text            # untouched line preserved
    tomllib.loads(new_text)                        # still valid TOML


def test_apply_skips_manual_line_unless_forced():
    class ManualHopg30:
        def is_manual_line(self, material, energy):
            return material == "hopg" and float(energy) == 30.0

        def is_manual_brem(self, *a):
            return False

    new_text, skipped = apply.apply_bounds(BASE_TOML, COMBINED, provenance_mod=ManualHopg30())
    assert "hopg:30" in skipped
    assert "stop = 2600.0, num = 864" in new_text   # original 30 keV row kept
    forced, skipped2 = apply.apply_bounds(
        BASE_TOML, COMBINED, force=True, provenance_mod=ManualHopg30()
    )
    assert skipped2 == []
    assert "stop = 2700.0, num = 897" in forced
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_apply.py`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the emitters + pure transform**

```python
# src/cxr_mc/line_grid/apply.py
"""Surgical write-back of derived line-grid bounds into materials.toml.

Owns exactly three regions and regenerates only those, preserving everything else
byte-for-byte: per-material `E_grid_line_by_energy` block, per-material
`E_grid_brem` line, and `[profiles.standard] energy_keV` values. No TOML writer
dependency — blocks are located by text scan and replaced with hand-emitted text
matching the existing one-inline-table-per-line format. Provenance/sticky-manual
comes from cxr_mc.line_grid.provenance; the catalog is reloaded post-write to
validate.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from pathlib import Path

from cxr_mc.line_grid import provenance as _provenance
from cxr_mc.line_grid.bounds import spacing_num

_MATERIALS_TOML = Path(__file__).resolve().parent.parent / "data" / "materials.toml"


def _line_start_eV(energy_keV: float) -> float:
    # Matches the existing catalog convention: 10 eV floor at <=60 keV, 50 eV above.
    return 10.0 if float(energy_keV) <= 60.0 else 50.0


def emit_line_block(rows) -> str:
    lines = ["E_grid_line_by_energy = ["]
    for row in sorted(rows, key=lambda r: float(r["energy_keV"])):
        e = float(row["energy_keV"])
        start = float(row.get("start_eV", _line_start_eV(e)))
        stop = float(row["stop_eV"])
        num = int(row["num"])
        lines.append(
            f"  {{ energy_keV = {e:g}, grid = {{ linspace = "
            f"{{ start = {start:g}, stop = {stop:g}, num = {num:d}, endpoint = true }} }} }},"
        )
    lines.append("]")
    return "\n".join(lines)


def emit_brem_line(stop_eV: float, step_eV: float) -> str:
    return f"E_grid_brem = {{ arange = {{ start = 0.0, stop = {float(stop_eV):g}, step = {float(step_eV):g} }} }}"


def _section_span(text: str, header: str) -> tuple[int, int]:
    """[start, end) char span of a `[header]` section body (to next `[` or EOF)."""
    m = re.search(rf"(?m)^\[{re.escape(header)}\]\s*$", text)
    if not m:
        raise KeyError(f"section [{header}] not found")
    body_start = m.end()
    nxt = re.search(r"(?m)^\[", text[body_start:])
    body_end = body_start + nxt.start() if nxt else len(text)
    return body_start, body_end


def _replace_assignment(section_text: str, key: str, new_assignment: str) -> str:
    """Replace `key = ...` (single line, or multi-line `[ ... ]` array) in a section."""
    # Multi-line array form: key = [ ... ] spanning lines.
    array = re.search(rf"(?m)^{re.escape(key)}\s*=\s*\[.*?^\]", section_text, re.S)
    if array:
        return section_text[: array.start()] + new_assignment + section_text[array.end():]
    single = re.search(rf"(?m)^{re.escape(key)}\s*=.*$", section_text)
    if single:
        return section_text[: single.start()] + new_assignment + section_text[single.end():]
    raise KeyError(f"assignment {key} not found in section")


def _merge_line_rows(text, material, new_rows, force, provenance_mod):
    header = f"materials.{material}"
    body_start, body_end = _section_span(text, header)
    body = text[body_start:body_end]
    existing = _parse_line_rows(body)  # {energy: row-dict from current TOML}
    merged = dict(existing)
    skipped = []
    for row in new_rows:
        e = float(row["energy_keV"])
        if not force and provenance_mod.is_manual_line(material, e):
            skipped.append(f"{material}:{e:g}")
            continue
        merged[e] = {"energy_keV": e, "start_eV": row.get("start_eV", _line_start_eV(e)),
                     "stop_eV": row["stop_eV"], "num": row["num"]}
    block = emit_line_block(list(merged.values()))
    new_body = _replace_assignment(body, "E_grid_line_by_energy", block)
    return text[:body_start] + new_body + text[body_end:], skipped


def _parse_line_rows(section_body: str) -> dict:
    wrapped = "[materials._tmp]\n" + section_body
    parsed = tomllib.loads(wrapped)["materials"]["_tmp"].get("E_grid_line_by_energy", [])
    out = {}
    for item in parsed:
        e = float(item["energy_keV"])
        g = item["grid"]["linspace"]
        out[e] = {"energy_keV": e, "start_eV": g["start"], "stop_eV": g["stop"], "num": g["num"]}
    return out


def _merge_brem(text, material, brem, force, provenance_mod):
    if not force and provenance_mod.is_manual_brem(material):
        return text, [f"{material}:brem"]
    header = f"materials.{material}"
    body_start, body_end = _section_span(text, header)
    body = text[body_start:body_end]
    new_body = _replace_assignment(
        body, "E_grid_brem", emit_brem_line(brem["stop_eV"], brem["step_eV"])
    )
    return text[:body_start] + new_body + text[body_end:], []


def _insert_energies(text, energies) -> str:
    body_start, body_end = _section_span(text, "profiles.standard")
    body = text[body_start:body_end]
    m = re.search(r"(?m)^energy_keV\s*=\s*\{\s*values\s*=\s*\[([^\]]*)\]", body)
    if not m:
        return text
    current = [float(x) for x in m.group(1).replace(" ", "").split(",") if x]
    union = sorted(set(current) | {float(e) for e in energies})
    rendered = ", ".join(f"{e:g}" for e in union)
    new_body = body[: m.start()] + f"energy_keV = {{ values = [{rendered}] }}" + body[m.end():]
    return text[:body_start] + new_body + text[body_end:]


def apply_bounds(toml_text, combined, *, force=False, provenance_mod=_provenance):
    text = toml_text
    skipped = []
    all_energies = set()
    for material, entry in combined.items():
        rows = entry["line_rows"]
        all_energies.update(float(r["energy_keV"]) for r in rows)
        text, sk = _merge_line_rows(text, material, rows, force, provenance_mod)
        skipped.extend(sk)
        text, skb = _merge_brem(text, material, entry["brem"], force, provenance_mod)
        skipped.extend(skb)
    text = _insert_energies(text, all_energies)
    return text, skipped
```

Note: `_parse_line_rows` reuses `tomllib` so `emit`/`parse` round-trip through the same grammar the catalog uses. `spacing_num` import is available for the `set` command (Task 8).

- [ ] **Step 4: Run the pure-transform tests to verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_apply.py`
Expected: PASS.

- [ ] **Step 5: Add `apply_file` + its test**

```python
# add to tests/test_line_grid_apply.py
import json as _json


def test_apply_file_writes_stamps_provenance_and_validates(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    json_path = tmp_path / "combined.json"
    json_path.write_text(_json.dumps(COMBINED))
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    stamped = []
    monkeypatch.setattr(apply._provenance, "set_line",
                        lambda m, e, s, note=None: stamped.append((m, float(e), s)))
    monkeypatch.setattr(apply._provenance, "set_brem", lambda m, s, note=None: None)
    monkeypatch.setattr(apply._provenance, "is_manual_line", lambda *a: False)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *a: False)
    apply.apply_file(json_path, slurm_id="458", date="2026-07-22")
    assert "stop = 2700.0, num = 897" in toml_path.read_text()
    assert ("hopg", 30.0, "derived job 458 (2026-07-22)") in stamped
```

Implement `apply_file`:

```python
def apply_file(json_path, *, materials=None, force=False, dry_run=False,
               slurm_id="?", date="", regen_golden=False):
    combined = json.loads(Path(json_path).read_text())
    if materials:
        wanted = set(materials.split(",") if isinstance(materials, str) else materials)
        combined = {m: v for m, v in combined.items() if m in wanted}
    original = Path(_MATERIALS_TOML).read_text()
    new_text, skipped = apply_bounds(original, combined, force=force)
    tomllib.loads(new_text)  # validate structure before touching disk
    if dry_run:
        _print_diff(original, new_text)
        return
    _atomic_write(_MATERIALS_TOML, new_text)
    from cxr_mc.materials import load_catalog  # re-parse via the real loader to validate
    load_catalog(_MATERIALS_TOML)
    source = f"derived job {slurm_id} ({date})"
    for material, entry in combined.items():
        for row in entry["line_rows"]:
            e = float(row["energy_keV"])
            if force or not _provenance.is_manual_line(material, e):
                _provenance.set_line(material, e, source)
        if force or not _provenance.is_manual_brem(material):
            _provenance.set_brem(material, source)
    if skipped:
        print(f"[line-grid apply] kept manual overrides: {', '.join(skipped)} (use --force to replace)")
    else:
        print("[line-grid apply] golden is now stale; run `cxr line-grid regen-golden`"
              if not regen_golden else "[line-grid apply] regenerating golden")
```

Add `_atomic_write` (same pattern as defaults) and a minimal `_print_diff` using `difflib.unified_diff`. Confirm `cxr_mc.materials` exposes a `load_catalog(path)`; if the public loader has a different name, use the one the codebase already uses to build `CATALOG` (grep `def load` in `src/cxr_mc/materials/catalog.py`) — the goal is a real re-parse that raises on malformed output.

- [ ] **Step 6: Run the apply tests**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_apply.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cxr_mc/line_grid/apply.py tests/test_line_grid_apply.py
git commit -m "feat(line-grid): surgical materials.toml write-back with sticky-manual + energy insertion"
```

---

## Task 8: `set` / `set-brem` / `show`

Single-entry manual overrides via the same write-back path, stamped `manual` + note; `show` merges values and provenance.

**Files:**
- Modify: `src/cxr_mc/line_grid/apply.py` (add `set_line_grid`, `set_brem_grid`, `show`)
- Test: `tests/test_line_grid_apply.py` (add cases)

**Interfaces:**
- Consumes: `spacing_num` (already imported), `provenance.set_line/set_brem`.
- Produces:
  - `set_line_grid(material, energy, stop_eV, *, num=None, start_eV=None, note=None) -> None`
  - `set_brem_grid(material, stop_eV, *, step_eV=None, note=None) -> None`
  - `show(material=None) -> str` (returns printable report; caller prints).

- [ ] **Step 1: Write the failing test**

```python
def test_set_line_grid_stamps_manual_and_autocomputes_num(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    calls = []
    monkeypatch.setattr(apply._provenance, "set_line",
                        lambda m, e, s, note=None: calls.append((m, float(e), s, note)))
    monkeypatch.setattr(apply._provenance, "is_manual_line", lambda *a: False)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *a: False)
    apply.set_line_grid("hopg", 30.0, 3000.0, note="widen tail")
    text = toml_path.read_text()
    assert "stop = 3000.0" in text
    assert calls == [("hopg", 30.0, "manual", "widen tail")]
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_apply.py -k set_line_grid`
Expected: FAIL — `set_line_grid` missing.

- [ ] **Step 3: Implement `set_line_grid` / `set_brem_grid` / `show`**

```python
def set_line_grid(material, energy, stop_eV, *, num=None, start_eV=None, note=None):
    e = float(energy)
    start = float(start_eV) if start_eV is not None else _line_start_eV(e)
    n = int(num) if num is not None else spacing_num(start, float(stop_eV), 3.0)
    row = {"energy_keV": e, "start_eV": start, "stop_eV": float(stop_eV), "num": n}
    original = Path(_MATERIALS_TOML).read_text()
    new_text, _ = _merge_line_rows(original, material, [row], True, _provenance)
    tomllib.loads(new_text)
    _atomic_write(_MATERIALS_TOML, new_text)
    _provenance.set_line(material, e, "manual", note=note)


def set_brem_grid(material, stop_eV, *, step_eV=None, note=None):
    from cxr_mc.line_grid.defaults import load_defaults
    step = float(step_eV) if step_eV is not None else float(load_defaults()["brem_step_ev"])
    original = Path(_MATERIALS_TOML).read_text()
    new_text, _ = _merge_brem(original, material, {"stop_eV": float(stop_eV), "step_eV": step}, True, _provenance)
    tomllib.loads(new_text)
    _atomic_write(_MATERIALS_TOML, new_text)
    _provenance.set_brem(material, "manual", note=note)


def show(material=None) -> str:
    original = tomllib.loads(Path(_MATERIALS_TOML).read_text())
    mats = original["materials"]
    keys = [material] if material else list(mats)
    out = []
    for key in keys:
        block = mats.get(key, {})
        rows = block.get("E_grid_line_by_energy", [])
        out.append(f"=== {key} ===")
        for item in rows:
            e = float(item["energy_keV"])
            g = item["grid"]["linspace"]
            rec = _provenance.get_line(key, e)
            tag = f'manual: {rec.get("note", "")}' if rec and rec["source"] == "manual" else (
                rec["source"] if rec else "derived")
            out.append(f"  {e:>6g} keV  stop={g['stop']:>8g}  num={g['num']:>5}  [{tag}]")
        brem = block.get("E_grid_brem", {}).get("arange")
        if brem:
            rec = _provenance.get_brem(key)
            tag = rec["source"] if rec else "derived"
            out.append(f"  brem stop={brem['stop']:g} step={brem['step']:g}  [{tag}]")
    return "\n".join(out)
```

- [ ] **Step 4: Run the tests to verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_apply.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/line_grid/apply.py tests/test_line_grid_apply.py
git commit -m "feat(line-grid): CLI-managed manual line/brem overrides + show"
```

---

## Task 9: `regen-golden` (`golden.py`) + dev.py alias

Independent serializer that rebuilds `tests/data/material_catalog_golden.json` by parsing `materials.toml` directly (NOT via the runtime resolver).

**Files:**
- Create: `src/cxr_mc/line_grid/golden.py`
- Modify: `scripts/dev.py` (add `regen-golden` subcommand)
- Test: `tests/test_line_grid_golden.py`

**Interfaces:**
- Produces:
  - `build_golden() -> dict` — the serialized catalog structure, computed independently (raw `tomllib` + own grid materialization), matching the shape `tests/test_material_catalog.py` asserts against.
  - `regen(check=False) -> int` — write the golden, or (check) diff-only and return nonzero on drift.
  - `GOLDEN_PATH: Path`.

**Interfaces note:** The exact serialized shape is defined by `tests/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden` and the current `tests/data/material_catalog_golden.json`. `build_golden` must reproduce that shape field-for-field. Read both before implementing; the fields include `crystal_keys`, `configured_crystal_keys`, `material_keys`, per-crystal `lattice/V_cell/mosaic_fwhm_deg/composition/basis`, per-material `scan` grid fingerprints (`{"shape", "sha256"}` via the test's `_fingerprint`/`_array` helper), and `E_grid_line_by_energy` per energy.

- [ ] **Step 1: Study the golden shape**

Read `tests/data/material_catalog_golden.json` (via a `python3 - <<'EOF'` json heredoc — works without uv) and `tests/test_material_catalog.py` fingerprint helpers so `build_golden` emits the identical structure and hashing (`hashlib.sha256(array.tobytes()).hexdigest()`, `list(array.shape)`).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_line_grid_golden.py
import json
from pathlib import Path

from cxr_mc.line_grid import golden


def test_build_reproduces_checked_in_golden():
    checked_in = json.loads(golden.GOLDEN_PATH.read_text())
    rebuilt = golden.build_golden()
    assert rebuilt == checked_in


def test_check_flags_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(golden, "GOLDEN_PATH", tmp_path / "g.json")
    (tmp_path / "g.json").write_text('{"material_keys": ["stale"]}')
    assert golden.regen(check=True) != 0          # drift → nonzero
    assert (tmp_path / "g.json").read_text() == '{"material_keys": ["stale"]}'  # not written
```

- [ ] **Step 3: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_golden.py`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement `golden.py`**

Implement `build_golden` by parsing `src/cxr_mc/data/materials.toml` with `tomllib` and independently materializing each grid descriptor (`values`/`arange`/`linspace`/`logspace`) with NumPy — deliberately NOT importing `cxr_mc.materials.CATALOG`. Reuse the crystal/lattice math the golden needs from the CIF/crystal parsing that the test expects (read the test to see whether crystal fields come from the catalog or can be lifted from the golden's existing crystal block; if crystal fingerprints require the full CIF pipeline, import only the low-level CIF/lattice helpers, not the resolver under test). `regen`:

```python
def regen(check=False) -> int:
    import difflib
    rebuilt = json.dumps(build_golden(), indent=2, sort_keys=True) + "\n"
    if check:
        current = GOLDEN_PATH.read_text() if GOLDEN_PATH.exists() else ""
        if rebuilt != current:
            print("".join(difflib.unified_diff(
                current.splitlines(True), rebuilt.splitlines(True),
                "golden(current)", "golden(rebuilt)")))
            return 1
        return 0
    GOLDEN_PATH.write_text(rebuilt)
    print(f"[line-grid] wrote {GOLDEN_PATH}")
    return 0
```

**Independence guardrail test** — add:

```python
def test_build_golden_does_not_import_resolver():
    import ast, inspect
    src = inspect.getsource(golden)
    tree = ast.parse(src)
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "cxr_mc.materials" not in imported and "cxr_mc.materials.catalog" not in imported
```

If reproducing crystal fingerprints truly requires resolver internals, narrow this guardrail to forbid only the top-level `CATALOG`/resolver entry, and document the exact low-level helper imported — but prefer full independence.

- [ ] **Step 5: Run the golden tests + the real catalog golden test**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_golden.py tests/test_material_catalog.py`
Expected: PASS (rebuilt golden equals checked-in; catalog test still green).

- [ ] **Step 6: Add the `scripts/dev.py regen-golden` alias**

In `scripts/dev.py` add a `cmd_regen_golden` that calls `from cxr_mc.line_grid.golden import regen; raise SystemExit(regen(args.check))`, register `regen-golden` in the subparser table with a `--check` flag (mirror the existing `add_parser` pattern near line 340).

- [ ] **Step 7: Verify the alias**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py regen-golden --check`
Expected: exit 0, no diff (golden matches).

- [ ] **Step 8: Commit**

```bash
git add src/cxr_mc/line_grid/golden.py scripts/dev.py tests/test_line_grid_golden.py
git commit -m "feat(line-grid): independent regen-golden command + dev.py alias"
```

---

## Task 10: CLI wiring — `cxr line-grid` group

Register the group, delegate job verbs to `remote.py` for identical `-v/-vv` output, wire `derive/submit/apply/set/set-brem/defaults/show/regen-golden`.

**Files:**
- Modify: `src/cxr_mc/line_grid/__init__.py` (add `add_subparser` + `_cli_*`)
- Modify: `src/cxr_mc/cli.py` (import + register)
- Test: `tests/test_line_grid_cli.py`

**Interfaces:**
- Consumes: `remote.job_status(jobid, detail)`, `remote.attach(jobid)`, `remote._cli_logs(args)`, `remote._stop_jobid`, `remote._latest_jobid`; `job.start`, `derive.main`, `apply.*`, `defaults.*`, `golden.regen`.
- Produces: `cxr_mc.line_grid.add_subparser(sub)` registering the `line-grid` parser with `func` defaults.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_line_grid_cli.py
import argparse

from cxr_mc import line_grid


def _parse(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    line_grid.add_subparser(sub)
    return ap.parse_args(argv)


def test_status_delegates_to_remote_job_status_with_detail(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid.remote, "job_status",
                        lambda jobid=None, detail=0: seen.update(jobid=jobid, detail=detail))
    args = _parse(["line-grid", "status", "job7", "-vv"])
    args.func(args)
    assert seen == {"jobid": "job7", "detail": 2}


def test_apply_dispatches_with_pull_and_force(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid, "_pull_combined", lambda: "combined.json")
    monkeypatch.setattr(line_grid.apply, "apply_file",
                        lambda path, **kw: seen.update(path=path, **kw))
    args = _parse(["line-grid", "apply", "--pull", "--force"])
    args.func(args)
    assert seen["path"] == "combined.json" and seen["force"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_cli.py`
Expected: FAIL — `add_subparser` missing.

- [ ] **Step 3: Implement `__init__.py`**

```python
# src/cxr_mc/line_grid/__init__.py
"""`cxr line-grid` command group. Job verbs delegate to cxr_mc.remote for
byte-identical output with `cxr remote`; derive/apply/set/defaults/regen-golden
call the package modules."""

from __future__ import annotations

from datetime import date

from cxr_mc import remote
from cxr_mc.line_grid import apply, defaults, job
from cxr_mc.line_grid import golden as _golden


def _cli_derive(args):
    return job._noop if False else __import__("cxr_mc.line_grid.derive", fromlist=["main"]).main(
        _derive_argv(args)
    )


def _derive_argv(args):
    argv = []
    for flag in ("materials", "energies", "tilts", "azimuths", "thickness"):
        val = getattr(args, flag)
        if val:
            argv += [f"--{flag}", val]
    if args.set_default:
        argv.append("--set-default")
    return argv


def _cli_submit(args):
    job.start(
        materials=args.materials or job.DEFAULT_MATERIALS,
        energies=args.energies or job.DEFAULT_ENERGIES,
        slice_minutes=args.slice_minutes,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
    )
    return 0


def _cli_status(args):
    remote.job_status(args.jobid, detail=args.verbose)
    return 0


def _cli_attach(args):
    remote.attach(args.jobid)
    return 0


def _cli_logs(args):
    return remote._cli_logs(args)


def _cli_stop(args):
    jobid = args.jobid or remote._latest_jobid()
    if not jobid:
        raise SystemExit("no jobs to stop")
    remote._stop_jobid(jobid)
    return 0


def _pull_combined():
    """scp the combined derivation JSON back from the remote box; returns local path."""
    return remote.pull_file(job.DEFAULT_JSON_OUT)  # use whatever remote pull primitive exists


def _cli_apply(args):
    path = _pull_combined() if args.pull else args.json
    if not path:
        raise SystemExit("no JSON: pass a path or --pull")
    apply.apply_file(path, materials=args.materials, force=args.force,
                     dry_run=args.dry_run, date=str(date.today()),
                     regen_golden=args.regen_golden)
    if args.regen_golden and not args.dry_run:
        _golden.regen()
    return 0


def _cli_set(args):
    apply.set_line_grid(args.material, args.energy, args.stop,
                        num=args.num, start_eV=args.start, note=args.note)
    return 0


def _cli_set_brem(args):
    apply.set_brem_grid(args.material, args.stop, step_eV=args.step, note=args.note)
    return 0


def _cli_defaults(args):
    if args.set:
        defaults.update_defaults(
            tilts=_floats(args.tilts), azimuths=_floats(args.azimuths),
            thickness_ang=_floats(args.thickness), brem_step_ev=args.brem_step,
        )
    for key, val in defaults.load_defaults().items():
        print(f"{key} = {val}")
    return 0


def _cli_show(args):
    print(apply.show(args.material))
    return 0


def _cli_regen_golden(args):
    return _golden.regen(check=args.check)


def _floats(s):
    return [float(x) for x in s.split(",")] if s else None


def add_subparser(sub):
    ap = sub.add_parser("line-grid", help="derive/apply per-material line-grid bounds (local or on qlmc)")
    g = ap.add_subparsers(dest="lg_command", required=True)

    for name in ("derive", "submit"):
        p = g.add_parser(name)
        p.add_argument("--materials", default=None)
        p.add_argument("--energies", default=None)
        p.add_argument("--tilts", default=None)
        p.add_argument("--azimuths", default=None)
        p.add_argument("--thickness", default=None)
        p.add_argument("--set-default", action="store_true")
        if name == "submit":
            p.add_argument("--slice-minutes", type=float, default=job.DEFAULT_SLICE_MINUTES)
            p.add_argument("--no-sync", action="store_true")
            p.add_argument("--dry-run", action="store_true")
            p.set_defaults(func=_cli_submit)
        else:
            p.set_defaults(func=_cli_derive)

    st = g.add_parser("status")
    st.add_argument("jobid", nargs="?", default=None)
    st.add_argument("-v", "--verbose", action="count", default=0)
    st.set_defaults(func=_cli_status)

    at = g.add_parser("attach")
    at.add_argument("jobid", nargs="?", default=None)
    at.set_defaults(func=_cli_attach)

    lg = g.add_parser("logs")
    lg.add_argument("jobid", nargs="?", default=None)
    lg.add_argument("-f", "--follow", action="store_true")
    lg.set_defaults(func=_cli_logs)

    sp = g.add_parser("stop")
    sp.add_argument("jobid", nargs="?", default=None)
    sp.set_defaults(func=_cli_stop)

    ap_apply = g.add_parser("apply")
    ap_apply.add_argument("json", nargs="?", default=None)
    ap_apply.add_argument("--materials", default=None)
    ap_apply.add_argument("--pull", action="store_true")
    ap_apply.add_argument("--force", action="store_true")
    ap_apply.add_argument("--regen-golden", action="store_true")
    ap_apply.add_argument("--dry-run", action="store_true")
    ap_apply.set_defaults(func=_cli_apply)

    se = g.add_parser("set")
    se.add_argument("material")
    se.add_argument("--energy", type=float, required=True)
    se.add_argument("--stop", type=float, required=True)
    se.add_argument("--num", type=int, default=None)
    se.add_argument("--start", type=float, default=None)
    se.add_argument("--note", default=None)
    se.set_defaults(func=_cli_set)

    sb = g.add_parser("set-brem")
    sb.add_argument("material")
    sb.add_argument("--stop", type=float, required=True)
    sb.add_argument("--step", type=float, default=None)
    sb.add_argument("--note", default=None)
    sb.set_defaults(func=_cli_set_brem)

    df = g.add_parser("defaults")
    df.add_argument("--set", action="store_true")
    df.add_argument("--tilts", default=None)
    df.add_argument("--azimuths", default=None)
    df.add_argument("--thickness", default=None)
    df.add_argument("--brem-step", type=float, default=None)
    df.set_defaults(func=_cli_defaults)

    sh = g.add_parser("show")
    sh.add_argument("material", nargs="?", default=None)
    sh.set_defaults(func=_cli_show)

    rg = g.add_parser("regen-golden")
    rg.add_argument("--check", action="store_true")
    rg.set_defaults(func=_cli_regen_golden)

    return ap
```

Simplify `_cli_derive` to a plain import at top (`from cxr_mc.line_grid import derive`) and `return derive.main(_derive_argv(args))` — the `__import__` dance above is only to avoid a heavy import at module load; use a top-level import unless it introduces a cycle. Resolve `_pull_combined`/`remote.pull_file` and `job.start`'s keyword surface against the real `remote.py` primitives (grep `_cli_pull` / the scp helper it uses; `job.start` currently takes `materials`/`energies` kwargs — confirm names).

- [ ] **Step 4: Register in `cli.py`**

In `src/cxr_mc/cli.py`, add `line_grid` to the import and register it:

```python
        from . import analyze, archive, check, check_config, export, line_grid, remote, scan, slim

        scan.add_subparser(sub)
        export.add_subparser(sub)
        analyze.add_subparser(sub)
        slim.add_subparser(sub)
        archive.add_subparser(sub)
        remote.add_subparser(sub)
        line_grid.add_subparser(sub)
        check.add_subparser(sub)
        check_config.add_subparser(sub)
```

Add `cxr line-grid <subcommand> …` to the module docstring's command list.

- [ ] **Step 5: Run the CLI tests + smoke the real entry**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_line_grid_cli.py`
Expected: PASS.

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr line-grid show hopg`
Expected: prints hopg's rows with `[derived]`/`[manual: …]` tags, no error.

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr line-grid defaults`
Expected: prints the persistent defaults.

- [ ] **Step 6: Commit**

```bash
git add src/cxr_mc/line_grid/__init__.py src/cxr_mc/cli.py tests/test_line_grid_cli.py
git commit -m "feat(line-grid): wire cxr line-grid command group into the CLI"
```

---

## Task 11: Full verification + docs sweep

Repo-wide green, lint/type clean, stale references gone.

**Files:**
- Modify: `AGENTS.md` / `docs/repo_map.md` only if they reference the deleted script paths.

- [ ] **Step 1: Grep for stale script references**

```bash
TOKENSAVE_DISABLE_GREP_HOOK=1 rg -n "scripts/analyze_line_grid_bounds|scripts/line_grid_bounds_job|cxr_mc\.line_grid_bounds" src tests scripts docs README.md AGENTS.md
```

Expected: no hits. Fix any doc/prose references to point at `cxr line-grid` / `cxr_mc.line_grid.*`.

- [ ] **Step 2: Full test suite**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test`
Expected: PASS (all).

- [ ] **Step 3: Lint + typecheck**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py lint`
Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py typecheck`
Expected: clean.

- [ ] **Step 4: Full verify**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify`
Expected: PASS.

- [ ] **Step 5: Commit any doc fixes**

```bash
git add -A
git commit -m "docs(line-grid): repoint references to cxr line-grid command group"
```

---

## Self-Review Notes

- **Spec coverage**: command surface (Task 10), status/attach parity (Task 10 delegation to `remote.job_status(detail=verbose)`), local `derive` (Tasks 2/4), remote `submit` + `-m` payload switch (Task 5), `apply` write-back + energy insertion + sticky-manual (Task 7), geometry/thickness overrides + `--set-default` (Task 4), persistent defaults incl. `brem_step_ev` (Task 3), provenance sidecar (Task 6), `set`/`set-brem`/`show` (Task 8), `regen-golden` independence + `--check` + dev alias (Task 9), full move + script deletion (Tasks 2/5), tests repointed (Tasks 1/2/5). Covered.
- **Independence guardrail**: Task 9 asserts `golden.py` does not import the resolver — the one spec requirement most easily violated.
- **Open verification points flagged inline** (resolve during execution, not guesses): the exact `remote` pull primitive for `_pull_combined`, the public catalog re-parse function name in `apply_file`, `job.start`'s kwarg surface, and the precise golden serialized shape. Each step says what to grep/read.
