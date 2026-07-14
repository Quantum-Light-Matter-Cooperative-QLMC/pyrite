# Cross-material comparison plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local line-to-bremsstrahlung ranking and two cross-material diagnostic plots to the analysis app.

**Architecture:** Keep the new ratio in `results.line_metrics`, expose it through the shared selection scoring API, and reuse `plot_material_comparison` for all three scatter plots. The notebook only composes the plots, so the metric and selection remain reusable by non-notebook consumers.

**Tech Stack:** Python 3.14, NumPy/SciPy, Matplotlib, marimo, pytest, uv.

## Global Constraints

- The local ratio is integrated coherent line flux divided by integrated incoherent brem flux over the existing plus-or-minus 1.5 FWHM window.
- A zero or negative local brem integral yields `NaN`; non-finite scores cannot win a selection.
- The raw-peak and local-ratio cross-material selections exclude dominant line energies below 100 eV; the existing quality-weighted plot remains unfiltered.
- The two new plots retain the existing x-axis, log y-axis, quality coloring, and labels.
- Keep reusable data and selection logic in `src/cxr_mc/`; limit `notebooks/analysis_app.py` to presentation.

---

### Task 1: Add the local peak-window line-to-brem metric

**Files:**
- Modify: `tests/test_results.py`
- Modify: `src/cxr_mc/results/metrics.py:104-178`
- Modify: `src/cxr_mc/results/scoring.py:18-47`

**Interfaces:**
- Consumes: `line_metrics(r, settings, rel_prominence=0.03, n_fwhm=3.0, metric="sharpness")`.
- Produces: `m["line_brem_ratio"]: float` and `selection_score(m, "line_brem_ratio"): float`.

- [ ] **Step 1: Write the failing metric and selection tests**

```python
def test_line_brem_ratio_uses_the_dominant_line_window():
    E = np.arange(50.0, 151.0, 1.0)
    spec = np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2)
    brem = 0.1 + 0.01 * E
    m = line_metrics(_record(E, spec, brem), default_settings())
    idx = int(np.argmax(spec))
    width = float(peak_widths(spec, [idx], rel_height=0.5)[0][0])
    half = max(round(3.0 * width / 2.0), 1)
    lo, hi = max(idx - half, 0), min(idx + half + 1, spec.size)
    expected = np.trapezoid(spec[lo:hi], E[lo:hi]) / np.trapezoid(brem[lo:hi], E[lo:hi])
    assert m["line_brem_ratio"] == pytest.approx(expected)


def test_line_brem_ratio_is_a_selection_mode():
    assert selection_score({"line_brem_ratio": 2.0}, "line_brem_ratio") == 2.0
    assert selection_score({"line_brem_ratio": np.nan}, "line_brem_ratio") == -np.inf
```

Import `peak_widths` from `scipy.signal` and `selection_score` from `cxr_mc.results` in the test module.

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `uv run python scripts/dev.py test tests/test_results.py -k line_brem_ratio`

Expected: FAIL because `line_brem_ratio` is absent and the selection mode is unknown.

- [ ] **Step 3: Implement the metric and selection mode**

In `line_metrics`, use the existing `lo` and `hi` window bounds and add:

```python
    brem_line_int = float(np.trapezoid(brem[lo:hi], E[lo:hi])) if hi > lo else 0.0
```

Then include this return value:

```python
        "line_brem_ratio": (line_int / brem_line_int) if brem_line_int > 0 else float("nan"),
```

Update the metric docstring to distinguish `line_brem_ratio` (one dominant line over local brem) from `coherent_brem_ratio` (all coherent flux over the full line grid). Add `"line_brem_ratio"` to `SELECTION_MODES`, document it in `selection_score`, and map it to `m["line_brem_ratio"]`.

- [ ] **Step 4: Run the focused metric tests and the full results test module**

Run: `uv run python scripts/dev.py test tests/test_results.py`

Expected: PASS.

- [ ] **Step 5: Commit the metric change**

```bash
git add tests/test_results.py src/cxr_mc/results/metrics.py src/cxr_mc/results/scoring.py
git commit -m "feat: rank lines by local brem ratio"
```

### Task 2: Make the material scatter selection energy-aware

**Files:**
- Create: `tests/test_material_comparison.py`
- Modify: `src/cxr_mc/plots/spectra.py:358-417`

**Interfaces:**
- Consumes: `plot_material_comparison(results_by_material, settings, select="quality_peak", min_line_eV=None)`.
- Produces: a Matplotlib figure whose selected annotations correspond to the highest finite `selection_score` at or above `min_line_eV`.

- [ ] **Step 1: Write the failing energy-floor selection test**

```python
def test_material_comparison_applies_minimum_line_energy():
    low = _record("Test", 30.0, 80.0, peak=100.0)
    high = _record("Test", 60.0, 200.0, peak=10.0)
    fig = plot_material_comparison(
        {"Test": {"scan": {30.0: low, 60.0: high}}},
        default_settings(),
        select="peak",
        min_line_eV=100.0,
    )
    assert fig.axes[0].texts[0].get_text() == "  Test (60 keV)"
```

Define `_record` with a 50--300 eV grid, a Gaussian `spec` centred at its `line_eV`, a constant positive `brem`, `scale=1.0`, and a complete `case` containing `name`, `E0_keV`, `tilt_deg`, `tilt_azim_deg`, and `thickness_ang`.

- [ ] **Step 2: Run the new plot test and verify it fails**

Run: `uv run python scripts/dev.py test tests/test_material_comparison.py -k minimum_line_energy`

Expected: FAIL with an unexpected `min_line_eV` keyword argument.

- [ ] **Step 3: Add filtered selection to the existing renderer**

Change the function signature to:

```python
def plot_material_comparison(
    results_by_material,
    settings,
    select="quality_peak",
    rel_prominence=0.03,
    line_metric="sharpness",
    min_line_eV=None,
):
```

After calculating `metrics`, filter records before `max`:

```python
        candidates = [
            r for r in recs
            if min_line_eV is None or metrics[id(r)]["line_eV"] >= min_line_eV
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: selection_score(metrics[id(r)], select))
```

Include `min_line_eV` in the title when supplied, for example `", line >= 100 eV"`, so readers can see why low-energy candidates are absent.

- [ ] **Step 4: Run plot tests and export contract**

Run: `uv run python scripts/dev.py test tests/test_material_comparison.py tests/test_plots_exports.py`

Expected: PASS.

- [ ] **Step 5: Commit the renderer change**

```bash
git add tests/test_material_comparison.py src/cxr_mc/plots/spectra.py
git commit -m "feat: filter cross-material plot selections"
```

### Task 3: Compose the requested comparisons in the analysis app

**Files:**
- Modify: `notebooks/analysis_app.py:1363-1384`

**Interfaces:**
- Consumes: `plot_material_comparison(..., select="peak", min_line_eV=100.0)` and `plot_material_comparison(..., select="line_brem_ratio", min_line_eV=100.0)`.
- Produces: Cross-material tab content containing the existing figure plus the two new diagnostic figures.

- [ ] **Step 1: Write the app-level expected composition as an executable smoke assertion**

Add the following source-level test to `tests/test_material_comparison.py`:

```python
def test_cross_material_tab_requests_new_comparisons():
    source = Path("notebooks/analysis_app.py").read_text()
    assert 'select="peak", min_line_eV=100.0' in source
    assert 'select="line_brem_ratio", min_line_eV=100.0' in source
```

Import `Path` from `pathlib`.

- [ ] **Step 2: Run the app composition test and verify it fails**

Run: `uv run python scripts/dev.py test tests/test_material_comparison.py -k cross_material_tab`

Expected: FAIL because neither requested call exists.

- [ ] **Step 3: Render the two new figures**

Replace the single-item figure list in `cross_material_tab` with:

```python
                [
                    _md,
                    plot_material_comparison(_by_material, settings, select="quality_peak"),
                    plot_material_comparison(
                        _by_material, settings, select="peak", min_line_eV=100.0
                    ),
                    plot_material_comparison(
                        _by_material,
                        settings,
                        select="line_brem_ratio",
                        min_line_eV=100.0,
                    ),
                ]
```

Expand `_md` to identify the three selection criteria and state that the two diagnostic selections apply a 100 eV floor. The helper titles identify the individual criterion and floor.

- [ ] **Step 4: Run the app test and Marimo structural validation**

Run: `uv run python scripts/dev.py test tests/test_material_comparison.py && uv run marimo check notebooks/analysis_app.py`

Expected: both commands exit 0.

- [ ] **Step 5: Run final targeted verification and commit**

Run: `uv run python scripts/dev.py test tests/test_results.py tests/test_material_comparison.py tests/test_plots_exports.py && uv run python scripts/dev.py lint`

Expected: both commands exit 0.

```bash
git add notebooks/analysis_app.py tests/test_material_comparison.py
git commit -m "feat: add cross-material diagnostic plots"
```

## Plan self-review

- Spec coverage: Task 1 implements the exact local ratio and non-finite handling; Task 2 applies the 100 eV selection floor while preserving the visual encoding; Task 3 adds both requested plots and validates the marimo app.
- Placeholder scan: no deferred implementation steps or unspecified error handling remain.
- Type consistency: `line_brem_ratio` is the same metric key in `line_metrics`, `selection_score`, the renderer, the notebook, and all test steps.
