# Groove Verification Baseline Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore repository-wide verification without changing groove physics or absorbing unrelated checkout work.

**Architecture:** Repair stale compatibility and test contracts at their owning boundaries. `cxr_mc.remote` remains a thin facade over `_remote`; analysis-app tests describe current tabs while the one non-wrapping control row becomes responsive; results exports freeze intentional dataset helpers; numerical goldens preserve structure exactly while comparing floats at tight cross-Python tolerances.

**Tech Stack:** Python 3.14, pytest, NumPy, marimo, cxr-mc `scripts/dev.py`.

## Global Constraints

- Work only in `.worktrees/groove-verify-fixes` on `fix/groove-verification-baseline`.
- Do not change groove physics or production orientation calculations.
- Preserve exact keys, shapes, ordering, strings, integer values, and nonnumeric structure.
- Hide CUDA and lower CPU priority during verification.
- Use `PYTHONPATH=/home/alexa/dev/cxr-mc/.worktrees/groove-verify-fixes/src` with the repository `.venv` when the isolated worktree cannot hydrate dependencies offline.

---

### Task 1: Restore rebrem facade compatibility

**Files:**
- Modify: `src/cxr_mc/remote.py`
- Test: `tests/test_remote.py`

**Interfaces:**
- Consumes: `_remote.scripts._rebrem_flags`, `_rebrem_queue_script`, `_rebrem_chunked_queue_script`, `_rebrem_queue_metadata`; `_remote.lifecycle.start_rebrem_queue`.
- Produces: matching `cxr_mc.remote` aliases.

- [x] **Step 1: Verify existing facade tests fail**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/alexa/dev/cxr-mc/.worktrees/groove-verify-fixes/src /home/alexa/dev/cxr-mc/.venv/bin/python scripts/dev.py test tests/test_remote.py -k rebrem
```

Expected: five facade tests fail with `AttributeError` while direct owner-module behavior remains available.

- [x] **Step 2: Add thin aliases beside their owning sections**

Add to the scripts facade block:

```python
_rebrem_flags = scripts._rebrem_flags
_rebrem_queue_script = scripts._rebrem_queue_script
_rebrem_chunked_queue_script = scripts._rebrem_chunked_queue_script
_rebrem_queue_metadata = scripts._rebrem_queue_metadata
```

Add to the lifecycle facade block:

```python
start_rebrem_queue = lifecycle.start_rebrem_queue
```

- [x] **Step 3: Verify focused tests pass**

Run the Step 1 command. Expected: all selected tests pass.

### Task 2: Reconcile analysis-app contracts

**Files:**
- Modify: `notebooks/analysis_app.py`
- Modify: `tests/test_analysis_app.py`

**Interfaces:**
- Consumes: current six top-level tabs and direct `scans_tab()` placement under `Optimize`.
- Produces: responsive crystal controls and source-contract assertions matching current UI.

- [x] **Step 1: Preserve reproduced failures**

Baseline already fails `test_analysis_app_uses_five_top_level_tabs_and_action_names` because `Structure` is intentional and scan maps are a direct view, plus `test_control_rows_wrap_at_narrow_widths` because crystal controls omit `wrap=True`.

- [x] **Step 2: Update tab/action contract**

Rename the test to `test_analysis_app_uses_six_top_level_tabs_and_action_names`, include `"Structure"` among top-level tabs, and remove obsolete `"Inspect scan maps"` action expectation. Keep the four actual accordion action names.

- [x] **Step 3: Make crystal control row responsive**

Add `wrap=True` to the `mo.hstack` that contains `crystal_na_ui`, `crystal_nb_ui`, `crystal_nc_ui`, `crystal_bonds_ui`, and `crystal_layers_ui`.

- [x] **Step 4: Verify notebook and focused tests**

Run:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run ruff check --fix notebooks/analysis_app.py tests/test_analysis_app.py
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uvx marimo check notebooks/analysis_app.py
rtk env CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/alexa/dev/cxr-mc/.worktrees/groove-verify-fixes/src /home/alexa/dev/cxr-mc/.venv/bin/python scripts/dev.py test tests/test_analysis_app.py
```

Expected: formatting/check succeeds and analysis-app tests pass.

### Task 3: Freeze intentional dataset exports

**Files:**
- Modify: `tests/test_results_exports.py`

**Interfaces:**
- Consumes: `cxr_mc.results.__all__`.
- Produces: frozen compatibility contract containing `project_dataset`, `merge_dataset`, `LINE_RECORD_KEYS`, and `BREM_RECORD_KEYS`.

- [x] **Step 1: Preserve reproduced failure**

Baseline `test_all_matches_frozen_set` reports exactly those four extra implementation exports.

- [x] **Step 2: Add four symbols to `FROZEN_EXPORTS`**

Place dataset functions and key constants beside other selection exports:

```python
"project_dataset",
"merge_dataset",
"LINE_RECORD_KEYS",
"BREM_RECORD_KEYS",
```

- [x] **Step 3: Verify export tests**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/alexa/dev/cxr-mc/.worktrees/groove-verify-fixes/src /home/alexa/dev/cxr-mc/.venv/bin/python scripts/dev.py test tests/test_results_exports.py
```

Expected: all export-contract tests pass.

### Task 4: Replace bitwise float assumptions

**Files:**
- Modify: `tests/test_material_catalog.py`
- Modify: `tests/test_surface_orientation.py`

**Interfaces:**
- Consumes: serialized catalog lattice dictionaries and `_orientation_R` matrix.
- Produces: exact structural checks plus tight numerical comparisons.

- [x] **Step 1: Preserve cross-Python RED provenance**

The isolated qlmc baseline reports one-ULP lattice parser drift and orientation matrix drift. Local Python may pass bitwise, so the recorded qlmc failures are the RED evidence.

- [x] **Step 2: Split lattice structure from numerical values**

Check lattice keys and `"system"` exactly. Compare remaining values with `np.testing.assert_allclose(..., rtol=2e-15, atol=0.0)`.

- [x] **Step 3: Freeze orientation numerically**

Rename the direct/reciprocal-path and legacy orientation tests from bitwise
contracts to numerical contracts, then use:

```python
np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)
```

- [x] **Step 4: Verify focused numerical tests**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/alexa/dev/cxr-mc/.worktrees/groove-verify-fixes/src /home/alexa/dev/cxr-mc/.venv/bin/python scripts/dev.py test tests/test_material_catalog.py tests/test_surface_orientation.py
```

Expected: all selected tests pass; exact nonnumeric and shape contracts remain enforced.

### Task 5: Repository verification

**Files:**
- Verify all changed files.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: clean focused and repository-wide verification evidence.

- [x] **Step 1: Run focused regression cluster**

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/alexa/dev/cxr-mc/.worktrees/groove-verify-fixes/src /home/alexa/dev/cxr-mc/.venv/bin/python scripts/dev.py test tests/test_remote.py tests/test_analysis_app.py tests/test_results_exports.py tests/test_material_catalog.py tests/test_surface_orientation.py
```

- [x] **Step 2: Run groove-focused regression cluster**

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/alexa/dev/cxr-mc/.worktrees/groove-verify-fixes/src /home/alexa/dev/cxr-mc/.venv/bin/python scripts/dev.py test tests/test_groove.py tests/test_blaze.py tests/test_trajectories.py tests/test_plotly_trajectories.py
```

- [x] **Step 3: Run full lint and test verification**

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/alexa/dev/cxr-mc/.worktrees/groove-verify-fixes/src UV_CACHE_DIR=/tmp/cxr-mc-uv-cache nice -n 10 /home/alexa/dev/cxr-mc/.venv/bin/python scripts/dev.py lint
rtk env CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/alexa/dev/cxr-mc/.worktrees/groove-verify-fixes/src UV_CACHE_DIR=/tmp/cxr-mc-uv-cache nice -n 10 /home/alexa/dev/cxr-mc/.venv/bin/python scripts/dev.py test
```

Expected: lint passes; all tests pass except any explicitly identified sandbox-only multiprocessing restriction, which must be rechecked outside sandbox before completion claim.

- [x] **Step 4: Inspect surgical diff**

```bash
rtk git diff --check
rtk git diff --stat
rtk git status --short --branch
```

Expected: only plan, facade aliases, source-contract updates, responsive wrap, frozen exports, and numerical comparisons changed.
