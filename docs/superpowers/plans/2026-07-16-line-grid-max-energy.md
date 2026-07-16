# Empirically-Bounded Line-Grid Upper Energies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the standard profile's undocumented line-grid `stop` values in `src/cxr_mc/data/materials.toml` with values derived from simulated 99%-cumulative-intensity coverage across every standard-profile catalog material and geometry.

**Architecture:** A pure, unit-tested math module (`src/cxr_mc/line_grid_bounds.py`) computes coverage energies and margined/rounded bounds from an `(E_grid, spec)` pair. A committed CLI script (`scripts/analyze_line_grid_bounds.py`) drives the existing `Sweep`/`build_cases`/`run_case` machinery through a coarse-then-refine empirical scan and reports/serializes the derived bounds. A one-off patch step then mechanically applies those bounds to the catalog TOML, its regression test, and the golden fingerprint file.

**Tech Stack:** Python, NumPy, pytest, the existing `cxr_mc.montecarlo` / `cxr_mc.sweep` / `cxr_mc.config` / `cxr_mc.materials` modules. No new dependencies.

## Global Constraints

- Coverage target: 99% cumulative coherent-line intensity (exact user directive — not 99.9%/99.99%).
- Safety margin: +15% above the measured 99% coverage energy, then round up to the nearest 100 eV.
- Spacing convention: `num = round((stop - start) / 3.0) + 1` (endpoint-inclusive, ~3 eV spacing), matching every existing `E_grid_line_by_energy` row.
- `E_grid_line_by_energy` stays profile-level/shared (not per-material) — only `stop`/`num` change; `start` is untouched.
- No changes to `sweep.py`, `catalog.py`, or `spectrum.py` — this is a data recalibration only.
- Physics basis: `E_res` is maximized near `tilt_polar ≈ 0` (`v·g ∝ cos(tilt_polar)`), so the coarse scan concentrates on the two smallest standard polar tilts, with larger-tilt spot checks.
- Sandbox note: every `uv run python ...` invocation in this worktree needs `dangerouslyDisableSandbox: true` (the sandbox blocks writes to `~/.cache/uv`).
- Reference spec: `docs/superpowers/specs/2026-07-16-line-grid-max-energy-design.md`.

---

### Task 1: Pure coverage/bound math helpers

**Files:**
- Create: `src/cxr_mc/line_grid_bounds.py`
- Test: `tests/test_line_grid_bounds.py`

**Interfaces:**
- Produces:
  - `coverage_energy(E_grid: np.ndarray, spec: np.ndarray, coverage: float = 0.99) -> float`
  - `margined_stop(raw_energy_eV: float, margin: float = 0.15, round_to: float = 100.0) -> float`
  - `spacing_num(start_eV: float, stop_eV: float, target_spacing_eV: float = 3.0) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_line_grid_bounds.py
import numpy as np
import pytest

from cxr_mc.line_grid_bounds import coverage_energy, margined_stop, spacing_num


def test_coverage_energy_flat_spectrum_half_coverage_is_midpoint():
    E_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    spec = np.ones_like(E_grid)
    assert coverage_energy(E_grid, spec, coverage=0.5) == pytest.approx(2.0)


def test_coverage_energy_flat_spectrum_near_full_coverage_is_last_point():
    E_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    spec = np.ones_like(E_grid)
    assert coverage_energy(E_grid, spec, coverage=0.99) == pytest.approx(4.0)


def test_coverage_energy_zero_spectrum_returns_grid_start():
    """A geometry with no coherent-line intensity places no requirement on
    grid width, and must not be mistaken for the widest requirement when
    candidates are ranked by this value -- so it returns the grid's floor,
    not its ceiling."""
    E_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    spec = np.zeros_like(E_grid)
    assert coverage_energy(E_grid, spec, coverage=0.99) == pytest.approx(0.0)


def test_coverage_energy_early_spike_is_captured_quickly():
    E_grid = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    spec = np.array([0.0, 100.0, 0.0, 0.0, 0.0])
    assert coverage_energy(E_grid, spec, coverage=0.99) == pytest.approx(2.0)


def test_margined_stop_exact_multiple_of_round_to():
    assert margined_stop(2000.0, margin=0.15, round_to=100.0) == pytest.approx(2300.0)


def test_margined_stop_rounds_up_past_round_to():
    assert margined_stop(2001.0, margin=0.15, round_to=100.0) == pytest.approx(2400.0)


@pytest.mark.parametrize(
    ("start_eV", "stop_eV", "expected_num"),
    [
        (10.0, 2500.0, 831),
        (10.0, 3000.0, 998),
        (50.0, 3500.0, 1151),
        (50.0, 4000.0, 1318),
        (50.0, 4500.0, 1484),
        (50.0, 5000.0, 1651),
    ],
)
def test_spacing_num_matches_existing_catalog_grids(start_eV, stop_eV, expected_num):
    assert spacing_num(start_eV, stop_eV) == expected_num
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python scripts/dev.py test tests/test_line_grid_bounds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cxr_mc.line_grid_bounds'`

- [ ] **Step 3: Write the implementation**

```python
# src/cxr_mc/line_grid_bounds.py
"""Pure helpers for scripts/analyze_line_grid_bounds.py: turn a simulated
coherent-line spectrum into a coverage energy, and a coverage energy into a
catalog-ready line-grid ``stop``/``num`` pair.

See docs/superpowers/specs/2026-07-16-line-grid-max-energy-design.md.
"""

import numpy as np


def coverage_energy(E_grid: np.ndarray, spec: np.ndarray, coverage: float = 0.99) -> float:
    """The smallest energy on ``E_grid`` at or below which the trapezoidal-
    integrated ``spec`` reaches ``coverage`` (0-1) of its total integral over
    ``E_grid``.

    Returns ``E_grid[0]`` when ``spec`` integrates to zero: a geometry with no
    coherent-line intensity places no requirement on the grid width, and must
    not be mistaken for the widest requirement when candidates are ranked by
    this value.
    """
    E_grid = np.asarray(E_grid, dtype=float)
    spec = np.asarray(spec, dtype=float)
    if E_grid.size < 2:
        raise ValueError("E_grid must have at least two points")
    increments = np.diff(E_grid) * (spec[:-1] + spec[1:]) / 2.0
    total = increments.sum()
    if total <= 0.0:
        return float(E_grid[0])
    cumulative = np.cumsum(increments) / total
    index = int(np.searchsorted(cumulative, coverage))
    index = min(index, cumulative.size - 1)
    return float(E_grid[index + 1])


def margined_stop(raw_energy_eV: float, margin: float = 0.15, round_to: float = 100.0) -> float:
    """Add a safety margin above ``raw_energy_eV`` and round up to the
    nearest ``round_to`` eV, so the catalog ``stop`` clears the measured
    coverage energy even for materials/geometries the empirical scan didn't
    sample exactly."""
    margined = raw_energy_eV * (1.0 + margin)
    return float(np.ceil(margined / round_to) * round_to)


def spacing_num(start_eV: float, stop_eV: float, target_spacing_eV: float = 3.0) -> int:
    """The endpoint-inclusive ``num`` for ``linspace(start_eV, stop_eV, num)``
    closest to ``target_spacing_eV`` uniform spacing -- the convention already
    used by every ``E_grid_line_by_energy`` row in materials.toml."""
    return int(round((stop_eV - start_eV) / target_spacing_eV)) + 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python scripts/dev.py test tests/test_line_grid_bounds.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/line_grid_bounds.py tests/test_line_grid_bounds.py
git commit -m "feat: add coverage-energy and grid-bound math helpers"
```

---

### Task 2: Empirical scan CLI script

**Files:**
- Create: `scripts/analyze_line_grid_bounds.py`

**Interfaces:**
- Consumes: `coverage_energy`, `margined_stop`, `spacing_num` from Task 1's `cxr_mc.line_grid_bounds`; `cxr_mc.config.material_sweep`; `cxr_mc.materials.CATALOG`; `cxr_mc.montecarlo.runner.run_case`; `cxr_mc.sweep.build_cases`.
- Produces: a `derive_bounds(materials, energies, top_k, coarse_ne, refine_ne) -> list[dict]` function (used directly by Task 3) and a `main()` CLI entry point.

This task has no pytest target (it's a dev tool that runs real Monte Carlo simulations); its acceptance is the smoke run in Step 2.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
# scripts/analyze_line_grid_bounds.py
"""Empirically derive per-beam-energy line-grid upper bounds (``stop``) for
the standard profile's ``E_grid_line_by_energy`` in
``src/cxr_mc/data/materials.toml``, from simulated coherent-line intensity
coverage rather than an undocumented cap.

See docs/superpowers/specs/2026-07-16-line-grid-max-energy-design.md for the
full method. Summary: for every requested material and standard beam energy,
run a small-Ne diagnostic spectrum on a wide ``E_grid_line`` at the two
smallest standard polar tilts (near tilt=0, where E_res is maximized) across
every standard azimuth, plus a couple of larger-tilt spot checks. Rank by the
energy at which 99% of coherent-line intensity is captured, refine the top
candidates at higher Ne, then report a +15%-margined ``stop`` and the ``num``
that preserves ~3 eV endpoint-inclusive spacing.

    uv run python scripts/analyze_line_grid_bounds.py
    uv run python scripts/analyze_line_grid_bounds.py --materials hopg,diamond --energies 30,50
    uv run python scripts/analyze_line_grid_bounds.py --json-out /tmp/line_grid_bounds.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from cxr_mc.config import material_sweep
from cxr_mc.line_grid_bounds import coverage_energy, margined_stop, spacing_num
from cxr_mc.materials import CATALOG
from cxr_mc.montecarlo.runner import run_case
from cxr_mc.sweep import build_cases

COVERAGE = 0.99
MARGIN = 0.15
ROUND_TO_EV = 100.0
TARGET_SPACING_EV = 3.0
WIDE_GRID_EV = np.arange(10.0, 10000.0, 5.0)
COARSE_NE = 500
REFINE_NE = 5000
TOP_K = 5
SPOT_CHECK_TILT_INDICES = (4, 9)  # the standard grid's ~40deg and 89deg points


@dataclass(frozen=True)
class Candidate:
    material: str
    tilt_deg: float
    tilt_azim_deg: float
    coverage_energy_eV: float
    total_intensity: float


def _run_geometry(material, energy_keV, tilt_deg, tilt_azim_deg, n_electrons):
    """One transported spectrum at a fixed material/energy/geometry, on the
    wide diagnostic grid. Pins the material's first configured thickness so
    only the tilt/azimuth axis varies across the scan."""
    thickness_ang = float(np.atleast_1d(CATALOG.material(material).scan.thickness_ang)[0])
    sweep = material_sweep(
        material,
        thickness_ang=thickness_ang,
        energy_keV=energy_keV,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        E_grid_line=WIDE_GRID_EV,
        E_grid_line_by_energy=None,
    )
    case = build_cases(sweep, n_electrons=n_electrons)[0]
    return run_case(case)


def _candidate(material, energy_keV, tilt_deg, tilt_azim_deg, n_electrons):
    result = _run_geometry(material, energy_keV, tilt_deg, tilt_azim_deg, n_electrons)
    E_grid, spec = result["E_grid"], result["spec"]
    return Candidate(
        material=material,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        coverage_energy_eV=coverage_energy(E_grid, spec, COVERAGE),
        total_intensity=float(np.trapezoid(spec, E_grid)),
    )


def _scan(materials, energy_keV, tilts, azimuths, n_electrons):
    return [
        _candidate(material, energy_keV, tilt, azim, n_electrons)
        for material in materials
        for tilt in tilts
        for azim in azimuths
    ]


def _warn_degenerate_zero_tilt(materials, near_zero_tilts, near_zero_candidates):
    """Flag any material whose exact tilt=0 geometry radiates zero coherent-
    line intensity while its next-smallest tilt does not -- a possible
    geometric degeneracy at tilt=0, distinct from ordinary MC noise. Does not
    change ranking: coverage_energy already returns the grid floor for
    zero-intensity geometries, so they can't win the top-k selection below."""
    zero_tilt, next_tilt = near_zero_tilts
    for material in materials:
        zero_total = sum(
            c.total_intensity for c in near_zero_candidates
            if c.material == material and c.tilt_deg == zero_tilt
        )
        next_total = sum(
            c.total_intensity for c in near_zero_candidates
            if c.material == material and c.tilt_deg == next_tilt
        )
        if zero_total <= 0.0 and next_total > 0.0:
            print(
                f"[warn] {material}: zero coherent-line intensity at "
                f"tilt={zero_tilt:g} deg but not at tilt={next_tilt:g} deg -- "
                "possible geometric degeneracy at exact zero tilt; tilt=0 "
                "remains a valid production sweep point regardless."
            )


def derive_bounds(materials, energies, top_k=TOP_K, coarse_ne=COARSE_NE, refine_ne=REFINE_NE):
    reference_scan = CATALOG.material(materials[0]).scan
    near_zero_tilts = [float(reference_scan.tilt_deg[i]) for i in (0, 1)]
    spot_check_tilts = [float(reference_scan.tilt_deg[i]) for i in SPOT_CHECK_TILT_INDICES]
    azimuths = [float(a) for a in reference_scan.tilt_azim_deg]

    rows = []
    for energy_keV in energies:
        near_zero = _scan(materials, energy_keV, near_zero_tilts, azimuths, coarse_ne)
        _warn_degenerate_zero_tilt(materials, near_zero_tilts, near_zero)
        spot_check = _scan(materials, energy_keV, spot_check_tilts, azimuths, coarse_ne)

        top = sorted(near_zero, key=lambda c: c.coverage_energy_eV, reverse=True)[:top_k]
        best_spot = max(spot_check, key=lambda c: c.coverage_energy_eV)
        flagged = best_spot.coverage_energy_eV > top[0].coverage_energy_eV
        candidates = top + ([best_spot] if flagged else [])

        refined = [
            _candidate(c.material, energy_keV, c.tilt_deg, c.tilt_azim_deg, refine_ne)
            for c in candidates
        ]
        driver = max(refined, key=lambda c: c.coverage_energy_eV)

        start_eV = float(reference_scan.E_grid_line_by_energy[energy_keV][0])
        stop_eV = margined_stop(driver.coverage_energy_eV, MARGIN, ROUND_TO_EV)
        num = spacing_num(start_eV, stop_eV, TARGET_SPACING_EV)
        rows.append(
            dict(
                energy_keV=energy_keV,
                raw_eV=driver.coverage_energy_eV,
                start_eV=start_eV,
                stop_eV=stop_eV,
                num=num,
                driver_material=driver.material,
                driver_tilt_deg=driver.tilt_deg,
                driver_azim_deg=driver.tilt_azim_deg,
                spot_check_flagged=flagged,
            )
        )
    return rows


def _print_report(rows):
    header = (
        f"{'energy':>8} {'raw_eV':>10} {'stop_eV':>9} {'num':>6} "
        f"{'driver':>14} {'tilt':>6} {'azim':>7} {'spot?':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['energy_keV']:>8g} {row['raw_eV']:>10.1f} {row['stop_eV']:>9.1f} "
            f"{row['num']:>6d} {row['driver_material']:>14} "
            f"{row['driver_tilt_deg']:>6.2f} {row['driver_azim_deg']:>7.2f} "
            f"{'yes' if row['spot_check_flagged'] else 'no':>6}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--materials",
        default=None,
        help="comma-separated catalog material keys (default: every standard-profile material)",
    )
    parser.add_argument(
        "--energies",
        default=None,
        help="comma-separated beam energies in keV (default: every standard beam energy)",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--coarse-ne", type=int, default=COARSE_NE)
    parser.add_argument("--refine-ne", type=int, default=REFINE_NE)
    parser.add_argument("--json-out", default=None, help="optional path to write rows as JSON")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    materials = args.materials.split(",") if args.materials else list(CATALOG.materials)
    if args.energies:
        energies = [float(e) for e in args.energies.split(",")]
    else:
        energies = [float(e) for e in CATALOG.material(materials[0]).scan.energy_keV]
    rows = derive_bounds(materials, energies, args.top_k, args.coarse_ne, args.refine_ne)
    _print_report(rows)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[analyze_line_grid_bounds] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test it end to end**

Run:
```bash
uv run python scripts/analyze_line_grid_bounds.py \
  --materials hopg --energies 30 --coarse-ne 20 --refine-ne 20 --top-k 1
```
Expected: prints the header row and exactly one data row for `energy=30`, no traceback, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_line_grid_bounds.py
git commit -m "feat: add empirical line-grid bound analysis script"
```

---

### Task 3: Derive the real bounds and apply them to the catalog

**Files:**
- Modify: `src/cxr_mc/data/materials.toml:8-16`
- Modify: `tests/test_material_catalog.py:203-225`
- Modify: `tests/data/material_catalog_golden.json` (every material's `scan.E_grid_line_by_energy`)

**Interfaces:**
- Consumes: `scripts/analyze_line_grid_bounds.py`'s `--json-out` rows (`energy_keV`, `start_eV`, `stop_eV`, `num` per row), Task 1's `cxr_mc.line_grid_bounds` (for recomputing golden fingerprints the same way the packaged catalog will), the `_array_fingerprint` helper already defined in `tests/test_material_catalog.py:550-555`.

This task runs the full empirical scan (slow: 39 materials x 7 energies x [2 near-zero tilts + 2 spot-check tilts] x 10 azimuths ~= 21,840 coarse-stage simulations, plus refine-stage reruns of the top candidates per energy). Run it in the background or expect it to take a long time; there is no way to shrink this scope without narrowing the coverage guarantee the design calls for.

- [ ] **Step 1: Run the full empirical scan**

Run (expect this to take a long time; consider `run_in_background`):
```bash
uv run python scripts/analyze_line_grid_bounds.py --json-out /tmp/line_grid_bounds.json
```
Expected: a 7-row report table (one row per standard beam energy) printed to stdout, and `/tmp/line_grid_bounds.json` written with the same 7 rows. Read the printed table; if any row has `spot?  yes`, note which material/tilt drove it -- that beam energy's bound came from a spot-checked larger tilt, not the near-zero-tilt prior, and is worth a second look before trusting it.

- [ ] **Step 2: Patch materials.toml, the regression test, and the golden fingerprints from the derived rows**

```python
# /tmp/apply_line_grid_bounds.py -- run once, not committed
import json
import re
from pathlib import Path

import numpy as np

rows = json.loads(Path("/tmp/line_grid_bounds.json").read_text())
rows.sort(key=lambda r: r["energy_keV"])

# ---- 1. src/cxr_mc/data/materials.toml -------------------------------------
TOML_PATH = Path("src/cxr_mc/data/materials.toml")
text = TOML_PATH.read_text()
OLD_LINES = {
    30.0: "  { energy_keV = 30.0, grid = { linspace = { start = 10.0, stop = 2500.0, num = 831, endpoint = true } } },",
    50.0: "  { energy_keV = 50.0, grid = { linspace = { start = 10.0, stop = 3000.0, num = 998, endpoint = true } } },",
    100.0: "  { energy_keV = 100.0, grid = { linspace = { start = 50.0, stop = 3500.0, num = 1151, endpoint = true } } },",
    150.0: "  { energy_keV = 150.0, grid = { linspace = { start = 50.0, stop = 4000.0, num = 1318, endpoint = true } } },",
    200.0: "  { energy_keV = 200.0, grid = { linspace = { start = 50.0, stop = 4500.0, num = 1484, endpoint = true } } },",
    250.0: "  { energy_keV = 250.0, grid = { linspace = { start = 50.0, stop = 5000.0, num = 1651, endpoint = true } } },",
    300.0: "  { energy_keV = 300.0, grid = { linspace = { start = 50.0, stop = 5000.0, num = 1651, endpoint = true } } },",
}
for row in rows:
    old = OLD_LINES[row["energy_keV"]]
    assert old in text, f"expected line for {row['energy_keV']} keV not found in materials.toml"
    new = (
        f"  {{ energy_keV = {row['energy_keV']:g}, grid = {{ linspace = "
        f"{{ start = {row['start_eV']:g}, stop = {row['stop_eV']:g}, "
        f"num = {row['num']}, endpoint = true }} }} }},"
    )
    text = text.replace(old, new)
TOML_PATH.write_text(text)
print("materials.toml updated")

# ---- 2. tests/test_material_catalog.py -------------------------------------
TEST_PATH = Path("tests/test_material_catalog.py")
test_text = TEST_PATH.read_text()

bounds_lines = "\n".join(
    f"        {row['energy_keV']:g}: ({row['start_eV']:g}, {row['stop_eV']:g})," for row in rows
)
new_dict = "expected_bounds = {\n" + bounds_lines + "\n    }"
test_text = re.sub(r"expected_bounds = \{[^}]*\}", new_dict, test_text, count=1)

# worst-case deviation of (stop-start)/(num-1) from the 3 eV target, over every
# row; the tolerance must clear this or the spacing assertion below is flaky.
max_dev = max(
    abs((row["stop_eV"] - row["start_eV"]) / (row["num"] - 1) - 3.0) for row in rows
)
tolerance = round(max_dev + 0.001, 3)
test_text = re.sub(
    r"spacing\[0\] == pytest\.approx\(3\.0, abs=[0-9.]+\)",
    f"spacing[0] == pytest.approx(3.0, abs={tolerance})",
    test_text,
    count=1,
)
TEST_PATH.write_text(test_text)
print(f"test_material_catalog.py updated (spacing tolerance abs={tolerance})")

# ---- 3. tests/data/material_catalog_golden.json ----------------------------
import hashlib


def _array_fingerprint(values):
    array = np.asarray(values, dtype="<f8")
    return {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


GOLDEN_PATH = Path("tests/data/material_catalog_golden.json")
golden = json.loads(GOLDEN_PATH.read_text())
new_fingerprints = {
    str(row["energy_keV"]): _array_fingerprint(
        np.linspace(row["start_eV"], row["stop_eV"], row["num"], endpoint=True)
    )
    for row in rows
}
for material in golden["materials"].values():
    material["scan"]["E_grid_line_by_energy"] = new_fingerprints
GOLDEN_PATH.write_text(json.dumps(golden, indent=1) + "\n")
print("material_catalog_golden.json updated")
```

Run: `uv run python /tmp/apply_line_grid_bounds.py`
Expected: three "updated" print lines, no `AssertionError`.

- [ ] **Step 3: Run the catalog tests to confirm the patch is self-consistent**

Run: `uv run python scripts/dev.py test tests/test_material_catalog.py -v`
Expected: PASS, including `test_standard_profile_uses_requested_angles_energies_and_line_grids` and `test_packaged_catalog_matches_independent_serialized_golden`.

- [ ] **Step 4: Commit**

```bash
git add src/cxr_mc/data/materials.toml tests/test_material_catalog.py tests/data/material_catalog_golden.json
git commit -m "fix(materials): derive standard-profile line-grid bounds from simulated coverage"
```

---

### Task 4: Full verification and branch TODO

**Files:**
- Modify: `TODO.md` (this worktree's branch copy)

- [ ] **Step 1: Run full verification**

Run: `uv run python scripts/dev.py verify`
Expected: PASS (lint, typecheck, full test suite). If the pre-existing three unrelated catalog-expectation failures noted in the per-beam-angular-grids plan reappear, confirm they're unrelated to this change (material count / historical h-BN thickness / a serialized crystal-grid fingerprint unrelated to line grids) before proceeding; otherwise every failure must trace back to this change.

- [ ] **Step 2: Overwrite the branch TODO.md**

```markdown
# TODO — line-grid-max-energy

This branch carries one backlog item: replace the standard profile's
line-grid upper energy bounds with values derived from simulated coherent-
line intensity coverage.

## Empirically-bounded line-grid upper energies (Done)

**Problem:** `E_grid_line_by_energy`'s `stop` values in
`src/cxr_mc/data/materials.toml` were set by an undocumented scaling with no
recorded derivation. A since-discarded uncommitted change further capped
`stop` at 3000 eV for every beam energy above 50 keV; empirical checking
showed this discarded 25-65% of diamond/silicon coherent-line intensity at
moderate-to-large tilt and higher beam energy (`mc_spectrum`'s segment-keep
filter truncates real radiated intensity outside `E_grid_line`, not just
discretization resolution).

**Implementation:** `scripts/analyze_line_grid_bounds.py` (committed,
reusable) runs a coarse-then-refine empirical scan over every standard-
profile catalog material and the standard tilt/azimuth sweep, concentrated
near `tilt_polar ~= 0` (where `E_res` is maximized) with larger-tilt spot
checks, to find the energy at which 99% of coherent-line intensity is
captured per standard beam energy. The measured bound gets a +15% safety
margin, rounded to the nearest 100 eV, with `num` recomputed to preserve
~3 eV endpoint-inclusive spacing. Results were applied to
`materials.toml`'s `E_grid_line_by_energy`, `tests/test_material_catalog.py`,
and `tests/data/material_catalog_golden.json`.

See `docs/superpowers/specs/2026-07-16-line-grid-max-energy-design.md` for
the full design and `docs/superpowers/plans/2026-07-16-line-grid-max-energy.md`
for the implementation plan.
```

Run: `git add TODO.md && git commit -m "docs: scope branch TODO to the line-grid-max-energy task"`

- [ ] **Step 3: Final check**

Run: `git log --oneline -8` and `git status`
Expected: a clean working tree and a commit history showing the helpers, the script, the derived catalog/test/golden patch, and the TODO update, on top of `e47a895`.
