# Penetration Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the analysis app select penetration energy, thickness, and polar tilt from the active material grid or manual bounded inputs.

**Architecture:** The Penetration tab derives catalog presets from `CATALOG[MATERIAL].scan`, resolves each source selector to one value, and passes those values to both `trajectory_sweep` calls.

**Tech Stack:** Python 3.14, Marimo, pytest AST/source regression checks.

## Global Constraints

- Keep this an app-only presentation change.
- Energy input is 1--300 keV; thickness input is 0.001--10 mm; tilt input is 0--89.9 degrees.
- Preserve the stack-aware `trajectory_sweep` path.

---

### Task 1: Test penetration source and bounds wiring

**Files:**

- Modify: `tests/test_analysis_app.py`
- Test: `tests/test_analysis_app.py`

**Interfaces:**

- Consumes: `notebooks/analysis_app.py` source.
- Produces: a regression guard for active-material grids, manual bounds, and resolved values.

- [ ] **Step 1: Write the failing test**

```python
def test_penetration_controls_offer_material_presets_and_bounded_manual_values() -> None:
    source = APP.read_text()
    for grid in ("scan.energy_keV", "scan.thickness_ang", "scan.tilt_deg"):
        assert grid in source
    for bound in ("start=1.0", "stop=300.0", "start=0.001", "stop=10.0", "stop=89.9"):
        assert bound in source
    assert "penetration_energy_keV" in source
    assert "penetration_thickness_ang" in source
    assert "penetration_tilt_deg" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python scripts/dev.py test tests/test_analysis_app.py -k penetration_controls`

Expected: FAIL because the app has no active-material penetration controls.

### Task 2: Resolve and consume penetration controls

**Files:**

- Modify: `notebooks/analysis_app.py:572-589,1288-1354`
- Test: `tests/test_analysis_app.py`

**Interfaces:**

- Consumes: `CATALOG[MATERIAL].scan.energy_keV`, `.thickness_ang`, and `.tilt_deg`.
- Produces: `penetration_energy_keV`, `penetration_thickness_ang`, and `penetration_tilt_deg` floats for `trajectory_sweep`.

- [ ] **Step 1: Implement the minimal controls**

Use source radios, material-grid dropdowns, and manual number inputs. Convert the manual thickness from mm to Angstrom. Pass `energies=(penetration_energy_keV,)`, `thickness_ang=penetration_thickness_ang`, and `tilts=(penetration_tilt_deg,)` into both trajectory sweeps.

- [ ] **Step 2: Run focused tests and structural validation**

Run: `uv run python scripts/dev.py test tests/test_analysis_app.py && uv run marimo check notebooks/analysis_app.py`

Expected: both commands PASS.
