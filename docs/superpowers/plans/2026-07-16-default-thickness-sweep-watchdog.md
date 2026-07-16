# Default Thickness Sweep + Penetration Watchdog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every material in the cross-material comparison the same default
thickness sweep (100nm/500nm/1um/4um/10um/20um/50um/100um), and add a
pre-run "penetration watchdog" that uses the existing trajectory Monte Carlo
to skip thicker slabs once a beam energy's transmitted electron population
has already died out at a thinner one.

**Architecture:** The thickness sweep is a one-line change to the shared
`[profiles.standard]` grid in `src/cxr_mc/data/materials.toml` (materials on
that profile inherit it automatically); two per-material thickness overrides
(hopg, hbn) are removed so they inherit it too, and `mos2` is split into a
free-standing bulk entry (inherits the sweep) plus a new
`mos2-on-sapphire` entry that keeps its old fixed 3-layer-on-sapphire
config. The watchdog is a new function, `gate_cases_by_penetration`, added to
`config.py` (the module already shared by both the CLI scan runner and the
marimo scan app) that runs a cheap trajectory-only Monte Carlo
(`montecarlo.simulate_trajectories`, the SAME prebuilt transport behind
`plots.plot_penetration_survival`) at normal incidence for the thinnest
untested thickness in each beam energy's sweep, walks upward, and once the
transmitted fraction drops below 5%, drops every larger thickness for that
energy from the case list before the expensive spectral Monte Carlo ever
sees them.

**Tech Stack:** Python 3, TOML material catalog (`tomllib`), NumPy,
`cxr_mc.montecarlo.simulate_trajectories` (CPU electron transport), pytest.

## Global Constraints

- New default thickness grid (`[profiles.standard].thickness_ang`), in
  Angstrom: `[1000.0, 5000.0, 10000.0, 40000.0, 100000.0, 200000.0, 500000.0,
  1000000.0]` (100nm, 500nm, 1um, 4um, 10um, 20um, 50um, 100um).
- Watchdog survival floor: `0.05` (5%) of the incident electron population
  still exiting the far face of the slab.
- Watchdog reference geometry: normal incidence (`tilt_deg` nearest 0.0) —
  the deepest-penetrating tilt in any sweep, so a normal-incidence death at a
  given thickness guarantees death at every larger tilt too (see Task 4 for
  the argument in full).
- Watchdog electron count: `Ne=500` per pre-run transmission check (matches
  the existing default `Ne` used by `plots.plot_penetration_survival` /
  `plots.altair_trajectories.penetration_survival_chart`, so this reuses an
  already-validated statistical operating point rather than inventing a new
  one).
- `src/cxr_mc/sweep.py` must stay free of any `montecarlo` import (its
  module docstring states it is kept cheap to import/test); the watchdog
  therefore lives in `config.py`, not `sweep.py`, and is called by the
  orchestrators (`scan.py`, `notebooks/scan_app.py`) AFTER `build_cases`,
  not inside it.
- No new physics equation is introduced. The watchdog only consumes an
  existing validated transport diagnostic (`simulate_trajectories`'s
  `n_transmitted` count, already covered by the `electron-transport`
  validation marker). Per `AGENTS.md`, a new `Validation:` marker / ledger
  row is required only for new/edited physics — this change needs neither.
- `mos2` changes meaning in `mats_to_sim.toml`'s active list: previously "MoS2
  on sapphire, 3 layers", it becomes "bulk free-standing MoS2" (consistent
  with its TMD siblings mose2/wse2/etc., all profile=`standard` with no
  substrate). The old on-sapphire config is preserved verbatim under the new
  key `mos2-on-sapphire` (not added to the active `mats_to_sim.toml` list,
  matching the existing `mos2-on-sio2-si` precedent of being runnable via
  `cxr scan mos2-on-sapphire` but not part of the default `--all` sweep).
- Use `uv run python scripts/dev.py test|lint|typecheck|verify` for all
  verification commands (per `AGENTS.md`); never call `pytest`/`ruff`/`pyright`
  directly.

---

### Task 1: Branch TODO.md

**Files:**
- Modify: `TODO.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks (documentation only).

- [ ] **Step 1: Replace `TODO.md`'s content with a branch-scoped summary**

Per `AGENTS.md`'s branch convention ("`TODO.md` on branches contains details
scoped to their specific task"), replace the whole file content with:

```markdown
# TODO / Backlog

## Default thickness sweeps + penetration watchdog

Give every `mats_to_sim.toml` material the same default thickness sweep
(100nm/500nm/1um/4um/10um/20um/50um/100um) via `[profiles.standard]`,
including hopg/hbn (previously pinned to a single ~1mm slab) and mos2
(split into a free-standing bulk entry plus a new `mos2-on-sapphire`
device entry that keeps the old 3-layer-on-sapphire config). Before each
`cxr scan` case runs, a penetration watchdog runs a cheap trajectory-only
Monte Carlo (`montecarlo.simulate_trajectories`, the prebuilt
penetration-depth transport) at normal incidence for each (material, beam
energy, thickness); once the transmitted electron fraction drops below
5%, that thickness is kept (so the dying case is represented) but every
thicker one is skipped for that beam energy, so effectively dead cases
are not wastefully re-simulated.
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs: scope branch TODO to the thickness-sweep watchdog task"
```

---

### Task 2: Materials catalog — default thickness sweep, hopg/hbn, mos2 split

**Files:**
- Modify: `src/cxr_mc/data/materials.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: a `[profiles.standard]` with an 8-value `thickness_ang` sweep
  inherited by every material that doesn't override it; `[materials.hopg]`
  and `[materials.hbn]` with no `thickness_ang` override (so they inherit
  the sweep too); `[materials.mos2]` with no `substrate`/`thickness_layers`
  (bulk, profile default); a new `[materials.mos2-on-sapphire]` with
  `crystal = "mos2"`, `thickness_layers = 3`, `substrate = "sapphire"` (the
  old `mos2` config, renamed).

- [ ] **Step 1: Expand the standard profile's thickness grid**

In `src/cxr_mc/data/materials.toml`, change:

```toml
[profiles.standard]
thickness_ang = 100000.0
```

to:

```toml
[profiles.standard]
thickness_ang = { values = [1000.0, 5000.0, 10000.0, 40000.0, 100000.0, 200000.0, 500000.0, 1000000.0] }
```

- [ ] **Step 2: Drop the hopg and hbn thickness overrides**

Change:

```toml
[materials.hopg]
label = "HOPG"
profile = "standard"
thickness_ang = 1000e4

[materials.hbn]
label = "h-BN"
profile = "standard"
thickness_ang = 1000e4
```

to:

```toml
[materials.hopg]
label = "HOPG"
profile = "standard"

[materials.hbn]
label = "h-BN"
profile = "standard"
```

- [ ] **Step 3: Split mos2 into a bulk entry and a new on-sapphire entry**

Change:

```toml
[materials.mos2]
label = "MoS2"
profile = "standard"
thickness_layers = 3
substrate = "sapphire"
```

to:

```toml
[materials.mos2]
label = "MoS2"
profile = "standard"

[materials.mos2-on-sapphire]
label = "MoS2 on sapphire"
profile = "standard"
crystal = "mos2"
thickness_layers = 3
substrate = "sapphire"
```

(Insert `[materials.mos2-on-sapphire]` immediately after `[materials.mos2]`,
before the existing `[materials.mos2-on-sio2-si]` block — this fixes
`CATALOG.material_keys`' order, which Task 3's golden fixture must match.)

- [ ] **Step 4: Sanity-check the catalog loads**

Run:

```bash
uv run python -c "from cxr_mc.materials import CATALOG; print(len(CATALOG.materials)); print(CATALOG.material('mos2').substrate, CATALOG.material('mos2-on-sapphire').substrate); print(CATALOG.material('hopg').scan.thickness_ang.tolist())"
```

Expected: `49`, then `None sapphire`, then the 8-value list
`[1000.0, 5000.0, 10000.0, 40000.0, 100000.0, 200000.0, 500000.0, 1000000.0]`.
(Tests still fail at this point — that's expected; Task 3 updates them.)

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/data/materials.toml
git commit -m "feat(materials): default thickness sweep, split mos2 on-sapphire stack"
```

---

### Task 3: Update catalog-dependent tests and the golden fixture

**Files:**
- Modify: `tests/test_material_catalog.py`
- Modify: `tests/data/material_catalog_golden.json`

**Interfaces:**
- Consumes: Task 2's catalog (49 materials, new thickness grids).
- Produces: a green `tests/test_material_catalog.py`, `tests/test_sweep.py`,
  and `tests/test_check_config.py` (these three are the full set of tests
  that read `CATALOG` derived state).

- [ ] **Step 1: Update the material-count and hbn-thickness assertions**

In `tests/test_material_catalog.py`, in
`test_packaged_catalog_exposes_frozen_ordered_public_api`, change:

```python
    assert len(CATALOG.crystals) == 48
    assert len(CATALOG.materials) == 48
```

to:

```python
    assert len(CATALOG.crystals) == 48
    assert len(CATALOG.materials) == 49
```

(`crystals` is untouched — `mos2-on-sapphire` reuses the `mos2` crystal via
`crystal = "mos2"`, it does not add a new `[crystals.*]` entry.)

Then change:

```python
    np.testing.assert_array_equal(
        CATALOG.material("hbn").scan.thickness_ang,
        [10000000.0],
    )
```

to:

```python
    np.testing.assert_array_equal(
        CATALOG.material("hbn").scan.thickness_ang,
        [1000.0, 5000.0, 10000.0, 40000.0, 100000.0, 200000.0, 500000.0, 1000000.0],
    )
```

- [ ] **Step 2: Run the golden test to see it fail with the current fixture**

Run:

```bash
uv run python scripts/dev.py test tests/test_material_catalog.py -k test_packaged_catalog_matches_independent_serialized_golden
```

Expected: FAIL (`material_keys` length/order mismatch and/or `thickness_ang`
sha256 mismatches for every `profile = "standard"` material, since almost
every material's `scan.thickness_ang` fingerprint changed shape from `[1]`
to `[8]`).

- [ ] **Step 3: Regenerate the golden fixture from the new catalog**

`tests/data/material_catalog_golden.json` is a hand-maintained JSON snapshot
of `CATALOG`'s public/derived state (see `tests/test_material_catalog.py`'s
`test_packaged_catalog_matches_independent_serialized_golden` for the exact
shape it's checked against). There is no committed generator script for it
(the fixture was authored once, directly, in commit `df14789`). Run this
one-off regeneration script — it rewrites ONLY the `material_keys`,
`materials`, and `special_grids.hbn_thickness_ang` entries; `crystals`,
`media`, `crystal_keys`, `configured_crystal_keys`, `resolved_stacks`, and
the `mote2_*` special grids are untouched because Task 2 didn't touch
crystals, media, or mote2:

```bash
uv run python <<'PY'
import hashlib
import json
from pathlib import Path

import numpy as np

from cxr_mc.materials import CATALOG


def fp(values):
    array = np.asarray(values, dtype="<f8")
    return {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


path = Path("tests/data/material_catalog_golden.json")
golden = json.loads(path.read_text())

materials = {}
for key in CATALOG.material_keys:
    spec = CATALOG.material(key)
    scan = spec.scan
    e_grid_line = None if scan.E_grid_line is None else fp(scan.E_grid_line)
    e_grid_line_by_energy = None
    if scan.E_grid_line_by_energy is not None:
        e_grid_line_by_energy = {
            str(energy): fp(grid) for energy, grid in scan.E_grid_line_by_energy.items()
        }
    materials[key] = {
        "label": spec.label,
        "profile": spec.profile,
        "crystal_key": spec.crystal_key,
        "substrate": spec.substrate,
        "stack": [
            {
                "material": layer.material,
                "thickness_ang": layer.thickness_ang,
                "beam_uvw": list(layer.beam_uvw) if layer.beam_uvw else None,
                "azimuth_deg": layer.azimuth_deg,
            }
            for layer in spec.stack
        ],
        "scan": {
            "thickness_ang": fp(scan.thickness_ang),
            "energy_keV": fp(scan.energy_keV),
            "tilt_deg": fp(scan.tilt_deg),
            "tilt_azim_deg": fp(scan.tilt_azim_deg),
            "E_grid_line": e_grid_line,
            "E_grid_line_by_energy": e_grid_line_by_energy,
            "E_grid_brem": fp(scan.E_grid_brem),
        },
    }

golden["material_keys"] = list(CATALOG.material_keys)
golden["materials"] = materials
golden["special_grids"]["hbn_thickness_ang"] = CATALOG.material("hbn").scan.thickness_ang.tolist()

path.write_text(json.dumps(golden, indent=2) + "\n")
print("regenerated", path)
PY
```

- [ ] **Step 4: Review the diff before trusting it**

Run:

```bash
git diff --stat tests/data/material_catalog_golden.json
git diff tests/data/material_catalog_golden.json | grep -E '^\+|^-' | grep -oE '"(label|profile|crystal_key|substrate)"' | sort | uniq -c
```

Expected: the diff touches `thickness_ang` fingerprints for essentially every
`profile = "standard"` material (that's the intended, sweep-wide change),
PLUS `mos2`'s `substrate` line (now `null`) and a brand-new `mos2-on-sapphire`
block, PLUS the top-level `material_keys` list gaining `"mos2-on-sapphire"`
right after `"mos2"`, PLUS `special_grids.hbn_thickness_ang`. No `crystals`,
`media`, `resolved_stacks`, or `mote2_*` lines should appear in the diff — if
they do, stop and investigate before continuing (that would mean something
in Task 2 touched more than intended, or a concurrent branch's changes to
this same file leaked in — check `git log -1 --format=%H` against the
fixture's own `provenance.source_commit` field for a stale/foreign base).

- [ ] **Step 5: Run the full catalog-dependent test set**

Run:

```bash
uv run python scripts/dev.py test tests/test_material_catalog.py tests/test_sweep.py tests/test_check_config.py
```

Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add tests/test_material_catalog.py tests/data/material_catalog_golden.json
git commit -m "test(materials): refresh golden fixture for the default thickness sweep"
```

---

### Task 4: `gate_cases_by_penetration` in `config.py`

**Files:**
- Modify: `src/cxr_mc/config.py`
- Test: `tests/test_config_penetration_watchdog.py` (new)

**Interfaces:**
- Consumes: a `cases` list as produced by `sweep.build_cases` — each `case`
  dict must have `E0_keV: float`, `thickness_ang: float`, `tilt_deg: float`,
  `composition: list[tuple[str, float]]`, and `abs_layers:
  list[tuple[float, float, list]] | None`.
- Produces: `gate_cases_by_penetration(cases, *, floor=0.05, Ne=500,
  seed=0) -> tuple[list[dict], list[dict]]` returning `(kept_cases,
  dropped_cases)`; module constants `PENETRATION_SURVIVAL_FLOOR = 0.05` and
  `PENETRATION_WATCHDOG_NE = 500`. Task 5 and Task 6 import
  `gate_cases_by_penetration` from `cxr_mc.config`.

- [ ] **Step 1: Write the failing logic tests (mocked transport)**

Create `tests/test_config_penetration_watchdog.py`:

```python
"""Tests for config.gate_cases_by_penetration -- the pre-run penetration
watchdog that drops thickness values a beam energy has already died in.
These tests mock cxr_mc.config.simulate_trajectories so they exercise ONLY
the gating logic (grouping, cutoff detection, early-exit, reference-case
selection), not real electron transport; test_gate_cases_by_penetration_
drops_with_real_transport at the bottom of this file covers real physics.
"""

from cxr_mc import config


def _case(E0_keV, thickness_ang, tilt_deg=0.0, composition="C", abs_layers=None):
    return {
        "E0_keV": E0_keV,
        "thickness_ang": thickness_ang,
        "tilt_deg": tilt_deg,
        "composition": composition,
        "abs_layers": abs_layers,
    }


def test_gate_cases_by_penetration_keeps_dying_case_drops_thicker_ones(monkeypatch):
    calls = []

    def fake_simulate_trajectories(E0_keV, Ne, thickness_ang, **kwargs):
        calls.append((E0_keV, thickness_ang))
        fraction = 1.0 if thickness_ang <= 200.0 else (0.5 if thickness_ang <= 400.0 else 0.01)
        return {"n_transmitted": int(round(fraction * Ne))}

    monkeypatch.setattr(config, "simulate_trajectories", fake_simulate_trajectories)

    cases = [
        _case(30.0, t, tilt_deg=tilt)
        for t in (100.0, 200.0, 400.0, 800.0)
        for tilt in (0.0, 30.0)
    ]
    kept, dropped = config.gate_cases_by_penetration(cases, floor=0.05, Ne=100, seed=0)

    assert sorted({c["thickness_ang"] for c in kept}) == [100.0, 200.0, 400.0]
    assert {c["thickness_ang"] for c in dropped} == {800.0}
    # 800.0 was never simulated -- it was inferred dead once 400.0 crossed the floor
    assert calls == [(30.0, 100.0), (30.0, 200.0), (30.0, 400.0)]


def test_gate_cases_by_penetration_is_independent_per_energy(monkeypatch):
    def fake_simulate_trajectories(E0_keV, Ne, thickness_ang, **kwargs):
        fraction = 0.01 if (E0_keV == 30.0 and thickness_ang >= 200.0) else 1.0
        return {"n_transmitted": int(round(fraction * Ne))}

    monkeypatch.setattr(config, "simulate_trajectories", fake_simulate_trajectories)

    cases = [_case(E0, t) for E0 in (30.0, 300.0) for t in (100.0, 200.0, 400.0)]
    kept, dropped = config.gate_cases_by_penetration(cases, floor=0.05, Ne=50, seed=0)

    assert {c["thickness_ang"] for c in kept if c["E0_keV"] == 30.0} == {100.0, 200.0}
    assert {c["thickness_ang"] for c in kept if c["E0_keV"] == 300.0} == {100.0, 200.0, 400.0}
    assert {c["thickness_ang"] for c in dropped} == {400.0}


def test_gate_cases_by_penetration_keeps_everything_when_nothing_dies(monkeypatch):
    monkeypatch.setattr(
        config,
        "simulate_trajectories",
        lambda E0_keV, Ne, thickness_ang, **kwargs: {"n_transmitted": Ne},
    )

    cases = [_case(30.0, t) for t in (100.0, 200.0, 400.0)]
    kept, dropped = config.gate_cases_by_penetration(cases, floor=0.05, Ne=10, seed=0)

    assert kept == cases
    assert dropped == []


def test_gate_cases_by_penetration_uses_normal_incidence_reference(monkeypatch):
    seen_compositions = []

    def fake_simulate_trajectories(E0_keV, Ne, thickness_ang, composition=None, **kwargs):
        seen_compositions.append(composition)
        return {"n_transmitted": Ne}

    monkeypatch.setattr(config, "simulate_trajectories", fake_simulate_trajectories)

    cases = [
        _case(30.0, 100.0, tilt_deg=45.0, composition="wrong"),
        _case(30.0, 100.0, tilt_deg=0.0, composition="right"),
    ]
    config.gate_cases_by_penetration(cases, floor=0.05, Ne=10, seed=0)

    assert seen_compositions == ["right"]


def test_gate_cases_by_penetration_drops_with_real_transport():
    # A real (unmocked) cxr_mc.montecarlo.simulate_trajectories call: a 15 keV
    # beam through a light element (carbon) should fully transmit at 10nm,
    # be fully absorbed well before 100um, and 200um (never checked) is
    # inferred dead too. If this fails, print `fraction` per thickness (see
    # the mocked tests above for the shape) and adjust thickness/energy --
    # do not weaken the assertion to pass regardless of physics.
    composition = [("C", 0.1136)]  # graphite-like number density, atoms/Ang^3
    cases = [
        _case(15.0, t, composition=composition)
        for t in (100.0, 1_000_000.0, 2_000_000.0)
    ]
    kept, dropped = config.gate_cases_by_penetration(cases, floor=0.05, Ne=80, seed=0)

    assert 100.0 in {c["thickness_ang"] for c in kept}
    assert {c["thickness_ang"] for c in dropped} == {2_000_000.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run python scripts/dev.py test tests/test_config_penetration_watchdog.py
```

Expected: FAIL with `AttributeError: module 'cxr_mc.config' has no attribute
'gate_cases_by_penetration'` (or `simulate_trajectories`, for the
`monkeypatch.setattr` calls).

- [ ] **Step 3: Implement `gate_cases_by_penetration`**

In `src/cxr_mc/config.py`, add the import (alongside the existing
`from .sweep import Sweep`):

```python
from .montecarlo import simulate_trajectories
```

Then append, after `trajectory_sweep`:

```python
# Below this fraction of the incident electron population is still exiting
# the far face, a thicker slab in the same (material, beam energy) sweep is
# statistically indistinguishable from "the beam is dead" -- see
# gate_cases_by_penetration.
PENETRATION_SURVIVAL_FLOOR = 0.05
PENETRATION_WATCHDOG_NE = 500  # electrons per pre-run transmission check


def gate_cases_by_penetration(
    cases,
    *,
    floor: float = PENETRATION_SURVIVAL_FLOOR,
    Ne: int = PENETRATION_WATCHDOG_NE,
    seed: int = 0,
):
    """Drop thickness values a beam energy has already died in, before the
    (expensive) spectral Monte Carlo ever runs them.

    For each beam energy present in ``cases``, walks that energy's distinct
    ``thickness_ang`` values ascending and runs one direct-CPU transmission
    check per thickness with :func:`cxr_mc.montecarlo.simulate_trajectories`
    -- the SAME prebuilt penetration-depth transport behind
    :func:`cxr_mc.plots.plot_penetration_survival` -- at normal incidence
    (``beam_dir`` left at its ``simulate_trajectories`` default, +z), using
    the (energy, thickness) group's normal-incidence case for
    composition/stack, and ``Ne`` electrons. The transmitted fraction
    (``n_transmitted / Ne``) is a direct Monte-Carlo estimate of the
    fraction of the original beam still exiting the far face of that
    thickness of crystal -- the average remaining electron population at
    the end of that slab.

    Normal incidence is the reference geometry because it MAXIMIZES
    transmission at fixed nominal thickness: any nonzero tilt lengthens the
    in-material path length needed to reach a given depth below the entry
    surface by ~1/cos(tilt), so the beam sees more scattering and stopping
    power per unit depth at any tilt > 0 than at normal incidence. If the
    beam is already dead at normal incidence for a given thickness, it is
    at least as dead at every larger tilt in the sweep -- so checking only
    tilt=0 is a safe, cheap proxy for the whole tilt grid (one trajectory
    MC per (energy, thickness) instead of one per (energy, thickness, tilt,
    azimuth)).

    The first thickness whose transmitted fraction drops below ``floor`` is
    KEPT (so the "beam is basically dead" case is still represented in the
    checkpoint) but every LARGER thickness for that same energy is dropped
    WITHOUT running the check -- transmission is monotonically
    non-increasing with thickness at fixed energy, so a thicker slab can
    only be equally or more dead.

    Returns ``(kept_cases, dropped_cases)``, both preserving the input's
    relative order. ``dropped_cases`` is empty when no beam energy's
    transmitted fraction crosses ``floor`` anywhere in the sweep's
    thickness grid.
    """
    reference_case: dict[tuple[float, float], dict] = {}
    thicknesses_by_energy: dict[float, set] = {}
    for case in cases:
        energy = case["E0_keV"]
        thickness = case["thickness_ang"]
        thicknesses_by_energy.setdefault(energy, set()).add(thickness)
        key = (energy, thickness)
        current = reference_case.get(key)
        if current is None or abs(case["tilt_deg"]) < abs(current["tilt_deg"]):
            reference_case[key] = case

    cutoff_by_energy: dict[float, float | None] = {}
    for energy, thicknesses in thicknesses_by_energy.items():
        cutoff = None
        for thickness in sorted(thicknesses):
            ref = reference_case[(energy, thickness)]
            abs_layers = ref.get("abs_layers")
            total_thickness = float(abs_layers[-1][1]) if abs_layers is not None else thickness
            segs = simulate_trajectories(
                energy,
                Ne,
                total_thickness,
                composition=ref["composition"],
                layers=abs_layers,
                seed=seed,
            )
            fraction = float(segs["n_transmitted"]) / Ne
            if fraction < floor:
                cutoff = thickness
                break
        cutoff_by_energy[energy] = cutoff

    kept, dropped = [], []
    for case in cases:
        cutoff = cutoff_by_energy.get(case["E0_keV"])
        if cutoff is not None and case["thickness_ang"] > cutoff:
            dropped.append(case)
        else:
            kept.append(case)
    return kept, dropped
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run python scripts/dev.py test tests/test_config_penetration_watchdog.py -v
```

Expected: PASS, all 5 tests. If
`test_gate_cases_by_penetration_drops_with_real_transport` fails because the
chosen thickness/energy didn't actually cross the floor, print `fraction`
for each of the three thicknesses (temporarily), pick a thickness where
"clearly alive" and "clearly dead" bracket the default sweep's range, update
the test's numbers, and rerun -- this is expected iteration, not a plan gap.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/config.py tests/test_config_penetration_watchdog.py
git commit -m "feat: add penetration watchdog to gate thickness sweeps by transmitted fraction"
```

---

### Task 5: Wire the watchdog into `cxr scan` (CLI)

**Files:**
- Modify: `src/cxr_mc/scan.py`
- Test: `tests/test_sweep.py` (scan.py's tests already live here — see
  `test_scan_checkpoints_under_registry_name` for the existing
  `monkeypatch.setattr(scan, ...)` convention this follows)

**Interfaces:**
- Consumes: `gate_cases_by_penetration` from Task 4
  (`cxr_mc.config.gate_cases_by_penetration`).
- Produces: `scan._run_material` calls the watchdog between `build_cases`
  and `run_sweep`, and prints a one-line summary when anything is dropped.

- [ ] **Step 1: Write the failing wiring test**

In `tests/test_sweep.py`, add (near `test_scan_checkpoints_under_registry_name`):

```python
def test_run_material_applies_penetration_watchdog(monkeypatch, tmp_path):
    import argparse

    from cxr_mc import scan

    captured = {}

    def fake_gate(cases, **kwargs):
        captured["input_len"] = len(cases)
        half = len(cases) // 2
        return cases[:half], cases[half:]

    def fake_run_sweep(cases, results, checkpoint_dir=None, checkpoint_path=None, **kw):
        captured["run_sweep_cases"] = len(cases)

    monkeypatch.setattr(scan, "gate_cases_by_penetration", fake_gate)
    monkeypatch.setattr(scan, "run_sweep", fake_run_sweep)

    args = argparse.Namespace(
        material="mose2",
        workers=0,
        quick=True,
        n_families=None,
        beam_uvw=None,
        checkpoint_dir=str(tmp_path),
    )
    scan.run(args)

    assert captured["input_len"] > 0
    assert captured["run_sweep_cases"] == captured["input_len"] // 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run python scripts/dev.py test tests/test_sweep.py -k test_run_material_applies_penetration_watchdog
```

Expected: FAIL with `AttributeError: <module 'cxr_mc.scan' ...> does not
have the attribute 'gate_cases_by_penetration'` (monkeypatch can't patch a
name the module never imported).

- [ ] **Step 3: Wire the watchdog into `_run_material`**

In `src/cxr_mc/scan.py`, change the import line:

```python
from .config import default_settings, material_sweep
```

to:

```python
from .config import (
    PENETRATION_SURVIVAL_FLOOR,
    default_settings,
    gate_cases_by_penetration,
    material_sweep,
)
```

Then, in `_run_material`, change:

```python
    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    print(
        f"{material}: {len(cases)} cases across "
```

to:

```python
    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    cases, dropped = gate_cases_by_penetration(cases)
    if dropped:
        dead_energies = sorted({c["E0_keV"] for c in dropped})
        print(
            f"{material}: penetration watchdog dropped {len(dropped)} case(s) "
            f"at {len(dead_energies)} beam energy(ies) "
            f"({', '.join(f'{e:g} keV' for e in dead_energies)}) -- "
            f"electron population already below {100 * PENETRATION_SURVIVAL_FLOOR:g}% "
            f"before those thicknesses"
        )
    print(
        f"{material}: {len(cases)} cases across "
```

(The existing `print` after this already reads `len(cases)` and
`len({c['name'] for c in cases})` from whatever `cases` is bound to, so it
automatically reports the POST-gating count with no further edit.)

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run python scripts/dev.py test tests/test_sweep.py -k test_run_material_applies_penetration_watchdog
```

Expected: PASS.

- [ ] **Step 5: Run the full scan-related test file**

Run:

```bash
uv run python scripts/dev.py test tests/test_sweep.py
```

Expected: PASS, no regressions in the other ~40 tests in this file.

- [ ] **Step 6: Commit**

```bash
git add src/cxr_mc/scan.py tests/test_sweep.py
git commit -m "feat(scan): gate cases by the penetration watchdog before running"
```

---

### Task 6: Wire the watchdog into the marimo scan app

**Files:**
- Modify: `notebooks/scan_app.py`
- Test: `tests/test_scan_app.py`

**Interfaces:**
- Consumes: `gate_cases_by_penetration` from Task 4.
- Produces: the same watchdog behavior in the interactive scan runner as
  Task 5 gave the CLI, verified with the same static-source-text regression
  style already used by `tests/test_scan_app.py` (marimo apps are not
  imported/executed in tests; see the existing
  `test_scan_app_discovers_ordered_material_labels_from_catalog` test for
  the convention).

- [ ] **Step 1: Write the failing static regression test**

In `tests/test_scan_app.py`, add:

```python
def test_scan_app_gates_cases_by_penetration_before_running() -> None:
    source = APP.read_text()

    assert "gate_cases_by_penetration" in source
    build_at = source.index("build_cases(sweep")
    gate_at = source.index("gate_cases_by_penetration(cases")
    run_at = source.index("run_sweep(")
    assert build_at < gate_at < run_at
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run python scripts/dev.py test tests/test_scan_app.py -k test_scan_app_gates_cases_by_penetration_before_running
```

Expected: FAIL (`gate_cases_by_penetration` not yet present in the source).

- [ ] **Step 3: Wire the watchdog into the cases-building cell**

In `notebooks/scan_app.py`, change the first cell's import and return:

```python
    from cxr_mc.config import COLLAPSE_AZIMUTH, default_settings, material_sweep
    from cxr_mc.materials import CATALOG
    from cxr_mc.plots import stream_chunk
    from cxr_mc.run import run_sweep
    from cxr_mc.sweep import build_cases, geometry_table

    return (
        COLLAPSE_AZIMUTH,
        CATALOG,
        build_cases,
        default_settings,
        geometry_table,
        material_sweep,
        mo,
        run_sweep,
        stream_chunk,
    )
```

to:

```python
    from cxr_mc.config import (
        COLLAPSE_AZIMUTH,
        default_settings,
        gate_cases_by_penetration,
        material_sweep,
    )
    from cxr_mc.materials import CATALOG
    from cxr_mc.plots import stream_chunk
    from cxr_mc.run import run_sweep
    from cxr_mc.sweep import build_cases, geometry_table

    return (
        COLLAPSE_AZIMUTH,
        CATALOG,
        build_cases,
        default_settings,
        gate_cases_by_penetration,
        geometry_table,
        material_sweep,
        mo,
        run_sweep,
        stream_chunk,
    )
```

Then change the cases-building cell:

```python
@app.cell
def _(build_cases, default_settings, geometry_table, material_sweep, material_ui):
    MATERIAL = material_ui.value

    settings = default_settings()
    sweep = material_sweep(MATERIAL)  # full parametric grid (data/materials.toml)

    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    print(f"{len(cases)} cases across {len({c['name'] for c in cases})} configs")
    return MATERIAL, cases, settings
```

to:

```python
@app.cell
def _(
    build_cases,
    default_settings,
    gate_cases_by_penetration,
    geometry_table,
    material_sweep,
    material_ui,
):
    MATERIAL = material_ui.value

    settings = default_settings()
    sweep = material_sweep(MATERIAL)  # full parametric grid (data/materials.toml)

    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    cases, dropped = gate_cases_by_penetration(cases)
    if dropped:
        dead_energies = sorted({c["E0_keV"] for c in dropped})
        print(
            f"penetration watchdog dropped {len(dropped)} case(s) at "
            f"{len(dead_energies)} beam energy(ies) "
            f"({', '.join(f'{e:g} keV' for e in dead_energies)})"
        )
    print(f"{len(cases)} cases across {len({c['name'] for c in cases})} configs")
    return MATERIAL, cases, settings
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run python scripts/dev.py test tests/test_scan_app.py
```

Expected: PASS, both tests in the file.

- [ ] **Step 5: Structural marimo check**

Run:

```bash
uv run marimo check notebooks/scan_app.py
```

Expected: no errors (this is the same structural gate the penetration-controls
design doc cites for `analysis_app.py`; it validates the notebook's cell
graph/imports without executing the Monte Carlo).

- [ ] **Step 6: Commit**

```bash
git add notebooks/scan_app.py tests/test_scan_app.py
git commit -m "feat(scan-app): gate cases by the penetration watchdog before running"
```

---

### Task 7: Full verification and wrap-up

**Files:**
- None (verification only), unless a fix-up is required.

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: a fully green branch ready for review/merge.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
uv run python scripts/dev.py test
```

Expected: PASS. If any failures show up outside the files this plan touched,
check whether they pre-existed on `origin/main` before Task 1 (a concurrent
branch, `docs/superpowers/plans/2026-07-16-per-beam-angular-grids.md`, is
also editing `materials.toml` and this same golden fixture on a different
worktree -- that is not a signal this plan did anything wrong, just a merge
to coordinate later) before treating it as a regression from this work.

- [ ] **Step 2: Lint and type-check**

Run:

```bash
uv run python scripts/dev.py lint
uv run python scripts/dev.py typecheck
```

Expected: PASS.

- [ ] **Step 3: Full verify**

Run:

```bash
uv run python scripts/dev.py verify
```

Expected: PASS.

- [ ] **Step 4: Smoke-test a real (small) scan with the watchdog active**

Run a real, short scan to see the watchdog's print output on genuine
transport, not mocks:

```bash
uv run python scan.py mose2 --quick --workers 0
```

Expected: the run completes; if any beam energy's population actually dies
within the 8-value default thickness grid for this material, a
`penetration watchdog dropped ...` line appears before the per-tilt
progress output. Either outcome (drops or none) is a valid pass -- this
step is checking the wiring runs cleanly end-to-end with real
`simulate_trajectories` calls, not asserting a specific drop.

- [ ] **Step 5: Confirm the branch TODO summary still matches what shipped**

Re-read `TODO.md` (from Task 1) and confirm it still accurately describes
the change -- update it if any detail drifted during implementation (e.g. a
different default `Ne`, a different survival floor).

- [ ] **Step 6: Final commit (only if Step 5 changed anything)**

```bash
git add TODO.md
git commit -m "docs: sync branch TODO with the shipped implementation"
```
