# M6 — Renderer-Neutral Frame Builders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the last MEDIUM duplication cluster on `refactor/dedup-followthrough` (`docs/dedup-inventory.md` M6): the matplotlib renderer (`plots/sweeps.py`) and the Altair renderer (`plots/altair_sweeps.py`) each independently re-derive the "reduce every swept case to its best record per cell/point" logic, instead of sharing one tidy-data builder.

**Architecture:** Introduce `src/cxr_mc/plots/_frames.py`, a renderer-neutral module below `sweeps.py` in the DAG (leaf: `_style` → `_common` → **`_frames`** → `sweeps` → `altair_sweeps`). It owns the case-reduction math as plain-Python/tidy-`pandas.DataFrame` builders with zero matplotlib/Altair imports. `sweeps.py` renders those frames with `pcolormesh`/`ax.plot`; `altair_sweeps.py` renders them with `alt.Chart`. Every numeric value either renderer displays must trace to exactly one frame-builder call.

**Tech Stack:** Python, NumPy, pandas, matplotlib, Altair — no new dependencies.

## Global Constraints

- This is a **verbatim-logic refactor**: the numbers each function currently produces must not change. No new physics, no changed default parameters.
- `tests/test_plots_exports.py` / `tests/test_montecarlo_exports.py` / `tests/test_results_exports.py` are frozen (`set(pkg.__all__) == FROZEN_EXPORTS`) — do not remove any name currently re-exported from `cxr_mc.plots`. `altair_sweeps` (and the new `_frames`) stay **internal**, not re-exported from `cxr_mc.plots` — matches the existing `altair_spectra`/`altair_detectors`/`altair_trajectories` convention.
- Verify every step with `uv run pytest`, `uv run ruff check .`, `uv run pyright` — repo baseline is 201 tests / 0 ruff / 0 pyright (per `TODO.md`) before this plan starts.
- On this Windows box, `uv run pytest` can access-violate on pandas/pyarrow string columns (documented pyarrow-DLL-order issue). If it does, fall back to `uv run python -c "import pyarrow; import pytest, sys; sys.exit(pytest.main(['-q']))"` (import pyarrow before anything touches `cxr_mc`/pandas) — CI (Linux) is the real gate.
- Follow the branch's established verification idiom (used for M2/M3/M5): a **scratch script** (in the OS scratch dir, not committed) that builds synthetic records, calls the old and new code paths, and diffs the numeric output — not a new permanent pytest test, since none of `plot_heatmaps`/`plot_metric_vs`/`plot_scan` currently have direct behavioral tests (only the export-freeze test touches them).
- Commit convention on this branch: stage the relevant files **by name** (never `git add -A`) — the pre-existing unstaged `D analysis.py` is out of scope and must stay untouched.

---

## File Structure

- **Create** `src/cxr_mc/plots/_frames.py` — renderer-neutral tidy-data builders and shared guards. No matplotlib/Altair imports (only numpy/pandas + `..results` + `._common`).
- **Modify** `src/cxr_mc/plots/sweeps.py` — `plot_heatmaps`'s quantities-path and `plot_metric_vs` delegate to `_frames.heatmap_frame`/`_frames.metric_vs_frame` instead of their own inline per-cell reduction loops; `plot_scan` delegates its heatmap-vs-lines mode pick and hue selection to `_frames.scan_mode`/`_frames.pick_hue`. `_value_heatmap`'s `value=` callable path is untouched (it isn't part of the M6 duplication — no Altair equivalent exists for it).
- **Modify** `src/cxr_mc/plots/altair_sweeps.py` — delete its local `_ndistinct`, `_effective_x`, `metric_vs_frame`, `heatmap_frame` definitions; import all four from `_frames`. `scan_charts` delegates its mode/hue-pick to the same `_frames.scan_mode`/`_frames.pick_hue` used by `plot_scan`.
- **Modify** `tests/test_altair_sweeps.py` — update the `heatmap_frame`/`metric_vs_frame` import to come from `cxr_mc.plots._frames` (this file is a regular test, not one of the three frozen export-freeze tests, so its imports are free to move with the code).
- **Modify** `docs/dedup-inventory.md`, `TODO.md` — mark M6 done, cite the commit hash once committed (avoid the M5/results-split staleness recurrence — update these in the SAME commit).

---

## Task 1: Extract `_frames.py` with the shared guards (`_ndistinct`, `_effective_x`, `scan_mode`, `pick_hue`)

**Files:**
- Create: `src/cxr_mc/plots/_frames.py`
- Modify: `src/cxr_mc/plots/altair_sweeps.py:57-71` (delete `_ndistinct`, `_effective_x`)

**Interfaces:**
- Produces: `_ndistinct(recs, field) -> int`, `_effective_x(recs, x, hue) -> str`, `scan_mode(recs, x, y, heatmap_min, force=None) -> str` (`"heatmap"` or `"lines"`), `pick_hue(recs, x, y, panel, hue=None) -> tuple[str, str]` (returns `(line_x, hue)`).
- Consumes: nothing new — pure functions over `recs` (list of record dicts, each with a `"case"` sub-dict) and `field`/`x`/`y`/`panel` string keys.

- [ ] **Step 1: Create `_frames.py` with its module docstring and the `_ndistinct` guard**

```python
"""_frames

Renderer-neutral tidy-data builders for the parametric-sweep figures. Both
:mod:`cxr_mc.plots.sweeps` (matplotlib) and :mod:`cxr_mc.plots.altair_sweeps`
(Altair) render FROM these builders -- the per-cell/per-point "reduce every
swept case to its best record" reduction lives here exactly once. No
matplotlib or Altair imports; renderers turn the returned ``pandas.DataFrame``
/ guard values into figures.
"""

import pandas as pd

from ..results import records_for_cases, selection_score
from ._common import _metrics_map


def _ndistinct(recs, field):
    return len({r["case"][field] for r in recs if field in r["case"]})
```

- [ ] **Step 2: Move `_effective_x` verbatim from `altair_sweeps.py`**

Cut `_FALLBACK_X` and `_effective_x` out of `src/cxr_mc/plots/altair_sweeps.py` (currently lines 50 and 62-71) and paste into `_frames.py`, unchanged:

```python
_FALLBACK_X = ("tilt_deg", "thickness_ang", "E0_keV", "tilt_azim_deg", "B_ang2")


def _effective_x(recs, x, hue):
    """If ``x`` sweeps <2 values, substitute the first fallback knob that
    actually sweeps (and isn't the hue), so a 1-D scan/scan-frame is a
    meaningful curve instead of a vertical stack. Shared by
    :func:`cxr_mc.plots.sweeps.plot_metric_vs` and :func:`metric_vs_frame`."""
    if _ndistinct(recs, x) >= 2:
        return x
    return next(
        (f for f in _FALLBACK_X if f != x and f != hue and _ndistinct(recs, f) >= 2),
        x,
    )
```

- [ ] **Step 3: Add `scan_mode`, generalizing the duplicated heatmap-vs-lines pick**

`plots/sweeps.py:570-576` (`plot_scan`) and `plots/altair_sweeps.py:304-310` (`scan_charts`) both compute this identically. Add to `_frames.py`:

```python
def scan_mode(recs, x, y, heatmap_min, force=None):
    """Heatmap-vs-lines auto-pick shared by ``plot_scan`` (matplotlib) and
    ``scan_charts`` (Altair): heatmap when BOTH ``x`` and ``y`` sweep at least
    ``heatmap_min`` values, else lines. ``force`` ("heatmap"|"lines")
    overrides the pick."""
    if force in ("heatmap", "lines"):
        return force
    nx, ny = _ndistinct(recs, x), _ndistinct(recs, y)
    return "heatmap" if nx >= heatmap_min and ny >= heatmap_min else "lines"
```

- [ ] **Step 4: Add `pick_hue`, generalizing the duplicated line-mode axis/hue pick**

`plots/sweeps.py:596-603` and `plots/altair_sweeps.py:332-340` both do this identically (denser axis -> x, sparser -> hue, falling back to `panel`, then to `other`). Add to `_frames.py`:

```python
def pick_hue(recs, x, y, panel, hue=None):
    """Line-mode axis/hue pick shared by ``plot_scan`` and ``scan_charts``:
    the denser of ``x``/``y`` becomes the line x-axis, the sparser becomes
    ``hue`` (unless ``hue`` is given explicitly), falling back to ``panel``
    if the sparser axis doesn't vary, and finally to the sparser axis itself
    if nothing else varies (a single line). Returns ``(line_x, hue)``."""
    nx, ny = _ndistinct(recs, x), _ndistinct(recs, y)
    line_x, other = (x, y) if nx >= ny else (y, x)
    if hue is not None:
        return line_x, hue
    if _ndistinct(recs, other) >= 2:
        return line_x, other
    if _ndistinct(recs, panel) >= 2:
        return line_x, panel
    return line_x, other
```

- [ ] **Step 5: Run ruff + pyright on the new file in isolation**

Run: `uv run ruff check src/cxr_mc/plots/_frames.py && uv run pyright src/cxr_mc/plots/_frames.py`
Expected: both clean (the file isn't imported anywhere yet, so this only checks syntax/typing of the new code itself).

- [ ] **Step 6: Commit**

```bash
git add src/cxr_mc/plots/_frames.py src/cxr_mc/plots/altair_sweeps.py
git commit -m "$(cat <<'EOF'
refactor(plots): M6 step 1 -- extract shared scan guards into _frames.py

_ndistinct/_effective_x move out of altair_sweeps.py verbatim;
scan_mode/pick_hue generalize the heatmap-vs-lines and line-axis/hue pick
that plot_scan and scan_charts each re-implemented. altair_sweeps.py now
imports these instead of defining them.
EOF
)"
```

Note: `altair_sweeps.py` will fail to import after this commit until Task 2 adds the `from ._frames import ...` line and updates callers — if you want each task to leave the tree importable, fold this commit into Task 2's instead of committing standalone. (Recommended: do Tasks 1+2 as one commit — see Task 2's own commit step, which supersedes this one.)

---

## Task 2: Move `metric_vs_frame` / `heatmap_frame` into `_frames.py`; `altair_sweeps.py` renders only

**Files:**
- Modify: `src/cxr_mc/plots/_frames.py` (add `metric_vs_frame`, `heatmap_frame`)
- Modify: `src/cxr_mc/plots/altair_sweeps.py` (delete the two frame builders; import from `_frames`; `metric_vs_chart`/`heatmap_chart`/`scan_charts` call the imported versions)
- Modify: `tests/test_altair_sweeps.py:14-20` (import `heatmap_frame`/`metric_vs_frame` from `cxr_mc.plots._frames`)

**Interfaces:**
- Consumes: `_ndistinct`, `_effective_x` from Task 1; `records_for_cases`, `selection_score` from `..results`; `_metrics_map` from `._common`.
- Produces: `metric_vs_frame(results, settings, *, x="thickness_ang", metric="line_flux", hue="E0_keV", select="quality_peak", cases=None, rel_prominence=0.03, line_metric="sharpness") -> pd.DataFrame` (columns `x, metric, hue`); `heatmap_frame(results, settings, *, quantity="peak_flux", x="tilt_azim_deg", y="tilt_deg", panel="E0_keV", select="quality_peak", cases=None, rel_prominence=0.03, line_metric="sharpness", min_flux_frac=0.02, min_line_quality=0.2) -> pd.DataFrame` (columns `x, y, panel, value`). Both need `_AXIS_SPECS`/`_axis_disp`/`_value_label`/`_FLUX_GATED` from `sweeps.py` — import them (creates a `_frames -> sweeps`? NO: see note below, these constants move to `_frames.py` too, since `sweeps.py` will need them back via import to avoid a circular `_frames <-> sweeps` dependency).

**IMPORTANT — avoiding a circular import:** `heatmap_frame`/`metric_vs_frame` need `_AXIS_SPECS`, `_axis_disp`, `_value_label`, `_FLUX_GATED` (currently defined in `sweeps.py`). Since `_frames.py` must sit BELOW `sweeps.py` in the DAG (per `docs/repo_map.md`'s "leaf -> driver" convention), these four names move from `sweeps.py` into `_frames.py`, and `sweeps.py` imports them back (`from ._frames import _AXIS_SPECS, _axis_disp, _value_label, _FLUX_GATED, ...`). `_HEATMAP_QUANTITIES`, `_EXTRA_QUANTITIES`, `_METRIC_LABELS`, `_resolve_quantity` stay in `sweeps.py` (nothing in `_frames.py` needs them — `heatmap_frame` takes an already-resolved bare `quantity` key, matching its current Altair-side contract).

- [ ] **Step 1: Move `_AXIS_SPECS`, `_axis_disp`, `_value_label`, `_FLUX_GATED` from `sweeps.py` into `_frames.py`**

Cut verbatim from `src/cxr_mc/plots/sweeps.py` (currently lines 22-28 `_AXIS_SPECS`, lines 68-73 `_FLUX_GATED`, lines 81-84 `_axis_disp`, lines 87-90 `_value_label`) into `_frames.py`, directly below the `_ndistinct`/`_effective_x`/`scan_mode`/`pick_hue` block from Task 1. Do not reword the docstrings/comments.

In `src/cxr_mc/plots/sweeps.py`, replace those four definitions with an import:

```python
from ._frames import _AXIS_SPECS, _FLUX_GATED, _axis_disp, _value_label
```

(`_axis_label` stays in `sweeps.py` — it's matplotlib-axis-label-only, not needed by the frame builders.)

- [ ] **Step 2: Move `metric_vs_frame` verbatim from `altair_sweeps.py` into `_frames.py`**

Cut `src/cxr_mc/plots/altair_sweeps.py:74-111` (`metric_vs_frame`) unchanged into `_frames.py`. It already only touches `records_for_cases`, `pd`, `_effective_x`, `_metrics_map`, `_AXIS_SPECS`, `selection_score`, `_value_label` — all now available in `_frames.py`.

- [ ] **Step 3: Move `heatmap_frame` verbatim from `altair_sweeps.py` into `_frames.py`**

Cut `src/cxr_mc/plots/altair_sweeps.py:166-215` (`heatmap_frame`) unchanged into `_frames.py`. It needs `_FLUX_GATED` (now in `_frames.py`) in place of the module-level import `altair_sweeps.py` currently does from `.sweeps` — no code change, just confirm the name resolves locally now.

- [ ] **Step 4: Rewrite `altair_sweeps.py`'s imports and delete the moved code**

Replace `src/cxr_mc/plots/altair_sweeps.py`'s import block (currently lines 21-38) with:

```python
import altair as alt

from ..results import selection_score  # noqa: F401  (used only via records_for_cases path below)
from ._frames import (
    _effective_x,
    _ndistinct,
    heatmap_frame,
    metric_vs_frame,
    pick_hue,
    scan_mode,
)
from .sweeps import (
    _HEATMAP_QUANTITIES,
    _METRIC_LABELS,
    _AXIS_SPECS,
    _axis_label,
    _resolve_quantity,
)
```

Delete the now-duplicate `_FALLBACK_X`, `_ndistinct`, `_effective_x`, `metric_vs_frame`, `heatmap_frame` bodies from `altair_sweeps.py` (moved in Steps 2-3 above; only the `import` line references them now). Keep `_VEGA_SCHEME`/`_scheme` (Altair-only, no matplotlib equivalent) in `altair_sweeps.py`.

Re-check: does `altair_sweeps.py` still need `from ..results import records_for_cases`? Yes — `scan_charts` calls `records_for_cases(results, cases)` directly (line 299) to build `recs` before calling `scan_mode`/`pick_hue`. Keep that import; drop the now-unused `selection_score` import instead if nothing in the trimmed file calls it directly (check with `uv run ruff check`, which will flag an unused import — F401 — if so, remove it rather than leaving the `noqa`).

- [ ] **Step 5: Rewrite `scan_charts` to call the shared `scan_mode`/`pick_hue`**

In `src/cxr_mc/plots/altair_sweeps.py`'s `scan_charts` (currently lines 273-357), replace the inline mode pick (lines 304-310) with:

```python
    mode = scan_mode(recs, x, y, heatmap_min, force)
```

and replace the inline line-mode axis/hue pick (lines 332-340) with:

```python
    line_x, hue = pick_hue(recs, x, y, panel, hue)
```

removing the now-dead local `nx, ny = _ndistinct(recs, x), _ndistinct(recs, y)` computation that fed the deleted inline logic (keep any other use of `nx`/`ny` if present elsewhere in the function — check before deleting: `nx`/`ny` in the current code are ONLY used for the mode pick, so they can be removed entirely once `scan_mode` replaces that block).

- [ ] **Step 6: Update `tests/test_altair_sweeps.py`'s import**

```python
from cxr_mc.plots._frames import heatmap_frame, metric_vs_frame
from cxr_mc.plots.altair_sweeps import (
    heatmap_chart,
    metric_vs_chart,
    scan_charts,
)
```

- [ ] **Step 7: Run the Altair sweep test file**

Run: `uv run pytest tests/test_altair_sweeps.py -v` (or, if the pyarrow/pandas segfault hits: `uv run python -c "import pyarrow, pytest, sys; sys.exit(pytest.main(['tests/test_altair_sweeps.py', '-v']))"`)
Expected: all 11 tests PASS, unchanged from baseline (values are bit-identical — the functions moved, not changed).

- [ ] **Step 8: Full verification**

Run: `uv run ruff check . && uv run pyright && uv run pytest -q`
Expected: 0 ruff errors, 0 pyright errors, 201/201 tests passing (same count as branch baseline — no tests added or removed yet).

- [ ] **Step 9: Commit (supersedes Task 1's standalone commit if not yet committed)**

```bash
git add src/cxr_mc/plots/_frames.py src/cxr_mc/plots/altair_sweeps.py src/cxr_mc/plots/sweeps.py tests/test_altair_sweeps.py
git commit -m "$(cat <<'EOF'
refactor(plots): M6 step 2 -- move metric_vs_frame/heatmap_frame into _frames.py

altair_sweeps.py no longer defines the tidy-data reduction itself; it
imports metric_vs_frame/heatmap_frame (plus the scan_mode/pick_hue guards)
from the new renderer-neutral _frames.py and only renders. _AXIS_SPECS,
_axis_disp, _value_label, _FLUX_GATED move from sweeps.py into _frames.py
(sweeps.py imports them back) to keep the _frames -> sweeps dependency
direction acyclic. Verbatim logic move -- values unchanged, verified via
tests/test_altair_sweeps.py (still 11/11) plus the full suite.
EOF
)"
```

---

## Task 3: `plots/sweeps.py` renders FROM the shared frames (the actual dedup)

This is the task that removes the duplication: today, `plot_heatmaps`'s quantities-path and `plot_metric_vs` each run their OWN per-cell/per-point "pick the best record via `selection_score`" loop directly over `recs`, redundant with the `heatmap_frame`/`metric_vs_frame` logic Task 2 just centralized. After this task, both matplotlib functions get their reduced data from those same two functions.

**Files:**
- Modify: `src/cxr_mc/plots/sweeps.py` (`plot_heatmaps` lines ~253-310, `plot_metric_vs` lines ~416-513, `plot_scan` lines ~558-620)

**Interfaces:**
- Consumes: `heatmap_frame`, `metric_vs_frame`, `scan_mode`, `pick_hue` from `._frames` (Tasks 1-2).
- Produces: `plot_heatmaps`, `plot_metric_vs`, `plot_scan` keep their EXACT existing signatures and return types (`list[Figure]` / `Figure | None` / `list[Figure]`) — this is an internal-only change, no caller anywhere (including `plots/detectors.py`'s `plot_eaglexo_charge_map`, which uses the separate `value=` path, untouched) sees a difference.

- [ ] **Step 1: Write the verification scratch script FIRST (before touching `sweeps.py`)**

This branch's convention for a pure-reduction refactor is a diff script run once against the OLD code (via `git stash`) and once against the NEW code, comparing the actual rendered data (not the Figure objects, which aren't diffable). Write this to the scratch dir (not the repo):

```python
# scratchpad/verify_m6_sweeps.py
"""Dump plot_heatmaps / plot_metric_vs / plot_scan's underlying numeric data
for a synthetic record set, so it can be diffed before/after the M6 _frames
refactor. Run once on old sweeps.py (git stash), once on new (git stash pop),
diff the two dumps -- must be byte-identical."""

import json
import numpy as np

from cxr_mc.plots.sweeps import plot_heatmaps, plot_metric_vs, plot_scan


def _record(name, E0, tilt, azim, amp, n=80):
    E = np.linspace(1000.0, 5000.0, n)
    spec = amp * np.exp(-(((E - 2500.0) / 40.0) ** 2))
    brem = np.linspace(0.5, 0.1, n)
    return {
        "E_grid": E, "spec": spec, "brem": brem, "scale": 1.0,
        "case": {
            "name": name, "crystal": "HOPG", "E0_keV": E0, "tilt_deg": tilt,
            "tilt_azim_deg": azim, "thickness_ang": 5.0e4,
        },
    }


def _store():
    store = {}
    for tilt in (-20.0, -10.0):
        for azim in (0.0, 30.0):
            name = f"HOPG t{tilt} a{azim}"
            store[name] = {
                E0: _record(name, E0, tilt, azim, amp=abs(tilt) + 0.1 * E0 + 0.01 * azim)
                for E0 in (30.0, 60.0)
            }
    return store


class _Settings:
    beam_current_na = 1.0


def _fig_lines(fig):
    return [
        {"x": list(map(float, ln.get_xdata())), "y": list(map(float, ln.get_ydata()))}
        for ax in fig.axes
        for ln in ax.get_lines()
    ]


def _fig_meshes(fig):
    return [
        np.nan_to_num(im.get_array().data, nan=-999.0).tolist()
        for ax in fig.axes
        for im in ax.get_images()
    ]


out = {}
figs = plot_heatmaps(_store(), _Settings(), x="tilt_azim_deg", y="tilt_deg", panel="E0_keV")
out["heatmaps"] = [_fig_meshes(f) for f in figs]

fig = plot_metric_vs(_store(), _Settings(), x="tilt_deg", metric="peak_flux", hue="E0_keV")
out["metric_vs"] = _fig_lines(fig)

figs = plot_scan(_store(), _Settings(), force="lines")
out["scan_lines"] = [_fig_lines(f) for f in figs]
figs = plot_scan(_store(), _Settings(), force="heatmap")
out["scan_heatmap"] = [_fig_meshes(f) for f in figs]

print(json.dumps(out, sort_keys=True))
```

- [ ] **Step 2: Capture the OLD baseline**

Run: `uv run python scratchpad/verify_m6_sweeps.py > scratchpad/m6_sweeps_before.json`
Expected: valid JSON written, no exceptions.

- [ ] **Step 3: Rewrite `plot_heatmaps`'s quantities-path to call `heatmap_frame`**

In `src/cxr_mc/plots/sweeps.py`, replace the body of the `quantities` loop (currently lines 259-309, everything between `quantities = quantities or _HEATMAP_QUANTITIES` and the final `return figs`) with a version that builds each panel's `Z` grid by pivoting `heatmap_frame`'s tidy output instead of re-deriving `best`/`Z` inline:

```python
    from ._frames import heatmap_frame  # local import: avoids a module-level cycle note above

    quantities = quantities or _HEATMAP_QUANTITIES
    panel_vals = sorted({r["case"][panel] for r in recs})

    figs = []
    for key, label, cmap in quantities:
        df = heatmap_frame(
            results, settings, quantity=key, x=x, y=y, panel=panel, select=select,
            cases=cases, rel_prominence=rel_prominence, line_metric=line_metric,
            min_flux_frac=min_flux_frac, min_line_quality=min_line_quality,
        )
        panels = []  # (panel_value, Z, x_edges, y_edges)
        for pv in panel_vals:
            pv_label = _value_label(panel, pv)
            sub = df[df["panel"] == pv_label]
            xs_disp = sorted(sub["x"].unique())
            ys_disp = sorted(sub["y"].unique())
            xi = {v: i for i, v in enumerate(xs_disp)}
            yi = {v: j for j, v in enumerate(ys_disp)}
            Z = np.full((len(ys_disp), len(xs_disp)), np.nan)
            for _, row in sub.iterrows():
                Z[yi[row["y"]], xi[row["x"]]] = row["value"]
            panels.append((pv, Z, _cell_edges(xs_disp), _cell_edges(ys_disp)))
        finite = [Z[np.isfinite(Z)] for _, Z, _, _ in panels]
        finite = np.concatenate(finite) if any(a.size for a in finite) else np.array([0.0, 1.0])
        vmin, vmax = float(finite.min()), float(finite.max())

        fig, axes = plt.subplots(
            1, len(panels),
            figsize=(min(3.6 * len(panels) + 1.2, 12.0), 4.2),
            squeeze=False, constrained_layout=True,
        )
        im = None
        for ax, (pv, Z, xe, ye) in zip(axes.ravel(), panels, strict=False):
            im = ax.pcolormesh(xe, ye, Z, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{_AXIS_SPECS.get(panel, (panel,))[0]} = {_value_label(panel, pv)}")
            ax.set_xlabel(_axis_label(x))
            ax.set_ylabel(_axis_label(y))
        assert im is not None
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
        fig.suptitle(f"{label}    (best per cell: {select})", fontsize=13)
        figs.append(fig)
    return figs
```

Note the subtlety already present in the ORIGINAL code that must be preserved: `heatmap_frame` DROPS gated-out cells as missing rows (so they never appear in `df`), matching the original's `continue` (leave-as-NaN) behavior exactly, since a dropped row also leaves that `(xi, yi)` cell at its `np.full(..., np.nan)` default. Also preserve: the original groups panel membership from `recs` directly (`xs = sorted({r["case"][x] for r in er})`) which can include x/y values that end up ALL-gated-out for a given panel and so vanish from `df` entirely for that panel -- in that edge case the original still drew an all-NaN column/row (from `xs`/`ys` derived from `recs`), while pivoting purely from `df` would silently drop that column/row. Guard this: derive `xs_disp`/`ys_disp` from the FULL per-panel record set (`_axis_disp(x, sorted({r["case"][x] for r in recs if r["case"][panel] == pv}))`), not from `df` alone, so an all-gated column still renders as blank. Use this corrected version:

```python
        for pv in panel_vals:
            pv_label = _value_label(panel, pv)
            er = [r for r in recs if r["case"][panel] == pv]
            xs_disp = sorted(set(_axis_disp(x, [r["case"][x] for r in er])))
            ys_disp = sorted(set(_axis_disp(y, [r["case"][y] for r in er])))
            xi = {v: i for i, v in enumerate(xs_disp)}
            yi = {v: j for j, v in enumerate(ys_disp)}
            Z = np.full((len(ys_disp), len(xs_disp)), np.nan)
            sub = df[df["panel"] == pv_label]
            for _, row in sub.iterrows():
                Z[yi[row["y"]], xi[row["x"]]] = row["value"]
            panels.append((pv, Z, _cell_edges(xs_disp), _cell_edges(ys_disp)))
```

(replace the naive `xs_disp = sorted(sub["x"].unique())` block from the first draft above with this one).

- [ ] **Step 4: Diff against the OLD baseline for the heatmap path ONLY**

Run: `uv run python scratchpad/verify_m6_sweeps.py > scratchpad/m6_sweeps_after_step3.json` then compare just the `"heatmaps"` key against `m6_sweeps_before.json` (e.g. `python -c "import json; a=json.load(open('scratchpad/m6_sweeps_before.json')); b=json.load(open('scratchpad/m6_sweeps_after_step3.json')); assert a['heatmaps']==b['heatmaps'], 'MISMATCH'; print('heatmaps OK')"`)
Expected: `heatmaps OK`. If it mismatches, the most likely cause is the all-gated-column edge case above — re-check `xs_disp`/`ys_disp` are derived from `recs`, not `df`.

- [ ] **Step 5: Rewrite `plot_metric_vs` to call `metric_vs_frame`**

Replace `src/cxr_mc/plots/sweeps.py`'s `plot_metric_vs` body (currently lines 439-512, everything from `recs = records_for_cases(...)` through `return fig`) with:

```python
    from ._frames import metric_vs_frame

    recs = records_for_cases(results, cases)
    if not recs:
        print("no results yet")
        return None

    eff_x = _effective_x(recs, x, hue)
    if eff_x != x:
        print(f"plot_metric_vs: x={x!r} has <2 swept values -> using x={eff_x!r} instead (it actually sweeps).")
    elif _ndistinct(recs, x) < 2:
        print(f"plot_metric_vs: x={x!r} has <2 swept values and nothing else sweeps -> single point(s).")

    df = metric_vs_frame(
        results, settings, x=x, metric=metric, hue=hue, select=select, cases=cases,
        rel_prominence=rel_prominence, line_metric=line_metric,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    hue_labels = sorted(df["hue"].unique())
    hue_vals = sorted({r["case"][hue] for r in recs})
    for j, (hv, hl) in enumerate(zip(hue_vals, [_value_label(hue, hv) for hv in hue_vals], strict=True)):
        sub = df[df["hue"] == hl].sort_values("x")
        col = energy_color(hv, hue_vals) if hue == "E0_keV" else COLORS[j % len(COLORS)]
        ax.plot(sub["x"], sub["metric"], "o-", color=col, lw=1.8, label=hl)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(_axis_label(eff_x))
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
    ax.set_title(
        f"{_METRIC_LABELS.get(metric, metric)} vs {_axis_label(eff_x)}  (best per point: {select})",
        fontsize=11,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(title=_AXIS_SPECS.get(hue, (hue,))[0], fontsize=9)
    fig.tight_layout()
    return fig
```

This needs `_effective_x`/`_ndistinct` importable in `sweeps.py` — add `_effective_x` and `_ndistinct` to the `from ._frames import ...` line at the top of `sweeps.py` (alongside `_AXIS_SPECS`, `_FLUX_GATED`, `_axis_disp`, `_value_label` from Task 2).

Preserve the EXACT original console messages: the original prints one of two different messages depending on whether an alternate x was found (`"...instead (it actually sweeps)."`) or not (`"...and nothing else sweeps -> single point(s)."`). `_effective_x` itself doesn't distinguish these cases (it just returns `x` unchanged if nothing sweeps) -- the `if eff_x != x / elif _ndistinct(recs, x) < 2` branching above reconstructs the original's two-message logic from `_effective_x`'s return value; verify both console paths still fire correctly by hand-testing with a single-valued-x synthetic store as part of Step 6.

- [ ] **Step 6: Diff against the OLD baseline for `metric_vs`**

Extend `scratchpad/verify_m6_sweeps.py`'s `_store()` call sites: also run `plot_metric_vs(_store(), _Settings(), x="thickness_ang", metric="peak_flux", hue="E0_keV")` (thickness is single-valued in `_store()`, exercising the fallback message path) and dump its `_fig_lines(fig)` under a new `"metric_vs_fallback"` key, in both the before/after captures. Then run:

Run: `uv run python scratchpad/verify_m6_sweeps.py > scratchpad/m6_sweeps_after_step5.json` and diff the `"metric_vs"` and `"metric_vs_fallback"` keys against baseline the same way as Step 4.
Expected: both keys match byte-for-byte between before/after.

- [ ] **Step 7: Rewrite `plot_scan` to call `scan_mode`/`pick_hue`**

Replace `src/cxr_mc/plots/sweeps.py`'s `plot_scan` body (currently lines 558-620) mode-pick block (lines 567-576, the local `_ndistinct` closure + `nx, ny`/`force` branch) with:

```python
    from ._frames import pick_hue, scan_mode

    mode = scan_mode(recs, x, y, heatmap_min, force)
```

and the line-mode axis/hue pick (lines 596-603) with:

```python
    line_x, hue = pick_hue(recs, x, y, panel, hue)
```

removing the now-unused local `_ndistinct` closure and the `nx, ny = _ndistinct(x), _ndistinct(y)` line once nothing else in the function references them (check: the heatmap branch's `print(f"plot_scan: heatmap mode ({x} x {y}, panel per {panel})")` doesn't use `nx`/`ny`, safe to remove).

- [ ] **Step 8: Diff against the OLD baseline for `scan_lines`/`scan_heatmap`**

Run: `uv run python scratchpad/verify_m6_sweeps.py > scratchpad/m6_sweeps_after_step7.json` and diff all four keys (`heatmaps`, `metric_vs`, `metric_vs_fallback`, `scan_lines`, `scan_heatmap`) against `m6_sweeps_before.json`.
Expected: all match byte-for-byte.

- [ ] **Step 9: Full verification**

Run: `uv run ruff check . && uv run pyright && uv run pytest -q`
Expected: 0 ruff, 0 pyright, 201/201 tests (unchanged count).

- [ ] **Step 10: Delete the scratch verification script (not committed)**

The `scratchpad/verify_m6_sweeps.py` file and its `*.json` dumps live in the OS scratch directory per this session's convention (or wherever you created them, as long as it's outside the repo / gitignored) — confirm `git status` shows nothing new from this task besides the intended `src/`/`tests/` edits before staging.

- [ ] **Step 11: Update `TODO.md` and `docs/dedup-inventory.md` in the SAME commit**

In `docs/dedup-inventory.md`, change the M6 entry (currently `*(TODO -- do last.)*`) to done, e.g.:

```
- **M6 — matplotlib `sweeps` vs `altair_sweeps` duplicate the reduction, not just rendering.**
  *(DONE -- this branch, uncommitted.)* Moved `heatmap_frame`/`metric_vs_frame`/
  `_effective_x`/`_ndistinct`/`scan_mode`/`pick_hue` into a new renderer-neutral
  `plots/_frames.py`. `sweeps.py`'s `plot_heatmaps`/`plot_metric_vs`/`plot_scan`
  now render from the same tidy-data frames `altair_sweeps.py`'s
  `heatmap_chart`/`metric_vs_chart`/`scan_charts` already did -- both renderers
  share one reduction. `_AXIS_SPECS`/`_axis_disp`/`_value_label`/`_FLUX_GATED`
  moved from `sweeps.py` into `_frames.py` (sweeps.py imports them back) to
  keep the `_frames -> sweeps` dependency acyclic. Verified via a scratch
  before/after diff script (matplotlib line/mesh data byte-identical) plus
  the full suite.
```

In `TODO.md`, remove the "Remaining work, in suggested order" section's M6 item (it was the only one) and fold a summary sentence into the "Also done (this branch)" paragraph, mirroring the style of the M2/M3/M5/results-split summaries already there. Also update `docs/repo_map.md`'s `plots/` section to add `_frames` to the submodule DAG line (`_style → _common → _frames → sweeps → {spectra, detectors, trajectories} → interactive`) and its own one-line description (leaf-most tidy-data builders, no matplotlib/Altair imports).

- [ ] **Step 12: Commit**

```bash
git add src/cxr_mc/plots/_frames.py src/cxr_mc/plots/sweeps.py src/cxr_mc/plots/altair_sweeps.py TODO.md docs/dedup-inventory.md docs/repo_map.md
git commit -m "$(cat <<'EOF'
refactor(plots): M6 -- sweeps.py renders from the shared _frames builders

plot_heatmaps's quantities-path, plot_metric_vs, and plot_scan now get
their per-cell/per-point best-record reduction from _frames.heatmap_frame /
_frames.metric_vs_frame / _frames.scan_mode / _frames.pick_hue -- the same
functions altair_sweeps.py's heatmap_chart/metric_vs_chart/scan_charts
already used. Matplotlib and Altair no longer independently re-derive the
reduction, only the rendering differs. Verified byte-identical against the
pre-refactor code via a before/after diff script over synthetic records
(mesh/line data unchanged) plus the full suite (201/201, 0 ruff, 0 pyright).
This closes the M6 dedup-inventory item -- the branch's numbered TODO
backlog is now empty (M4 and the M7 line_fwhm_eV/escape-helper sub-items
remain tracked only in docs/dedup-inventory.md, out of this branch's scope).
EOF
)"
```

---

## Self-Review Notes (for the executor)

- **Spec coverage:** `heatmap_frame`/`metric_vs_frame`/`_effective_x` (Task 2) and `scan_charts`'s duplicated mode/hue pick (Task 1's `scan_mode`/`pick_hue`, wired into both renderers in Tasks 2 and 3) are the four names dedup-inventory.md's M6 entry calls out by name — all four covered.
- **`_value_heatmap`/`value=` path:** deliberately untouched — it has no Altair counterpart (`plot_eaglexo_charge_map` is matplotlib-only per M3), so it isn't part of the M6 duplication and must not be refactored here.
- **Do not** touch `plots/detectors.py`'s `plot_eaglexo_charge_map` — it calls `plot_heatmaps(..., value=...)`, a path this plan leaves untouched; only the `quantities`-path changes.
