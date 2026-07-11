# Zhai Supplementary Detected Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render complete Zhai supplementary spectra as detector-convolved flux in Fig. 1c units, with fixed visible bounds.

**Architecture:** Keep the cached Monte Carlo result intrinsic. Add a small figure-layer helper that applies the existing Fig. 1c detector broadening and absolute scaling immediately before plotting. The TMD and h-BN figure builders use that helper and own the display bounds; the marimo app copy describes the revised observable.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, pytest, ruff, marimo.

## Global Constraints

- Reuse Fig. 1c's `ZhaiAnchor` detector collection values: 0.066 sr, 16.6-degree acceptance, and 6.2415e9 electrons/s/nA.
- Preserve intrinsic `model_coherent_spectra` output and cache format.
- Do not add or alter transport or detector physics; use existing `convolve_detector`, `eds_fwhm_eV`, and `aperture_fwhm_eV`.
- Render all supplementary spectra as `Intensity (Phs/eV/s/nA)` with lower y-bound 0 and x-limits `[study.e_min_eV, study.e_max_eV]`.
- Keep the TMD figure a 2-by-2 panel, at an 8-inch width that fits the validation app output region.

---

### Task 1: Detector-scaled supplementary figure data and axes

**Files:**
- Modify: `checks/anchor_figures.py:718-767`
- Modify: `tests/test_anchor_figures.py:202-247`

**Interfaces:**
- Consumes: `SupplementaryCoherentStudy`, intrinsic spectra as `dict[float, np.ndarray]`, `ZhaiAnchor`, `beta_from_keV`, `eds_fwhm_eV`, `aperture_fwhm_eV`, and `convolve_detector`.
- Produces: `_supplementary_detected_spectrum(study: SupplementaryCoherentStudy, spectrum: np.ndarray) -> np.ndarray`, returning detector-convolved `Phs/eV/s/nA` values for plotting.
- Produces: supplementary figures with detected, normalized line data and explicit axis limits.

- [ ] **Step 1: Write the failing tests**

Add the following tests after `_synthetic_supplementary_spectra`:

```python
def test_supplementary_detected_spectrum_matches_fig1c_detector_scaling():
    study = af.supplementary_study("wse2")
    spectrum = _synthetic_supplementary_spectra(study)[-10.0]
    peak_eV = float(study.E_grid[np.argmax(spectrum)])
    anchor = af.ZhaiAnchor()
    fwhm_eV = float(
        np.hypot(
            af.eds_fwhm_eV(peak_eV),
            af.aperture_fwhm_eV(
                peak_eV,
                af.beta_from_keV(study.energy_keV),
                study.theta_obs_rad,
                anchor.dtheta_obs_rad,
            ),
        )
    )
    expected = af.convolve_detector(study.E_grid, spectrum, fwhm_eV)
    expected *= anchor.domega_sr * anchor.per_nA

    assert np.allclose(af._supplementary_detected_spectrum(study, spectrum), expected)


@pytest.mark.parametrize("crystal", ["wse2", "hbn"])
def test_supplementary_figures_use_detected_units_and_hard_bounds(crystal):
    study = af.supplementary_study(crystal)
    spectra = _synthetic_supplementary_spectra(study)
    fig = (
        af.figure_supplementary_hbn(study, study.thicknesses_nm[0], spectra)
        if crystal == "hbn"
        else af.figure_supplementary_tmd(study, study.thicknesses_nm[0], spectra)
    )

    expected = af._supplementary_detected_spectrum(study, spectra[study.polar_tilts_deg[0]])
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), expected)
    assert all(ax.get_ylabel() == "Intensity (Phs/eV/s/nA)" for ax in fig.axes)
    assert all(ax.get_ylim()[0] == 0.0 for ax in fig.axes)
    assert all(ax.get_xlim() == (study.e_min_eV, study.e_max_eV) for ax in fig.axes)
    if crystal == "wse2":
        assert tuple(fig.get_size_inches()) == (8.0, 7.0)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_anchor_figures.py -k supplementary -v`

Expected: FAIL because `_supplementary_detected_spectrum` does not exist and the figures still use intrinsic units and autoscaled limits.

- [ ] **Step 3: Implement the minimal figure-layer helper and use it**

Add this helper immediately before `figure_supplementary_tmd`:

```python
def _supplementary_detected_spectrum(
    study: SupplementaryCoherentStudy, spectrum: np.ndarray
) -> np.ndarray:
    """Return the Fig. 1c-detector response in Phs/eV/s/nA for one spectrum."""
    anchor = ZhaiAnchor()
    peak_eV = float(study.E_grid[np.argmax(spectrum)])
    fwhm_eV = float(
        np.hypot(
            eds_fwhm_eV(peak_eV),
            aperture_fwhm_eV(
                peak_eV,
                beta_from_keV(study.energy_keV),
                study.theta_obs_rad,
                anchor.dtheta_obs_rad,
            ),
        )
    )
    return convolve_detector(study.E_grid, spectrum, fwhm_eV) * (
        anchor.domega_sr * anchor.per_nA
    )
```

In both supplementary figure builders, plot `_supplementary_detected_spectrum(study, spectra[tilt_deg])`, set the y-label to `"Intensity (Phs/eV/s/nA)"`, call `ax.set_ylim(bottom=0.0)` and `ax.set_xlim(study.e_min_eV, study.e_max_eV)`, and replace every `intrinsic PXR+CBS only` title with `Zhai-detector-convolved PXR+CBS`. Set the TMD `figsize` to `(8, 7)`.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run pytest tests/test_anchor_figures.py -k supplementary -v`

Expected: PASS, including the detector-normalization and axis-bound tests.

- [ ] **Step 5: Render a synthetic TMD figure for visual inspection**

Run:

```powershell
@'
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "checks")
import anchor_figures as af
study = af.supplementary_study("wse2")
spectra = {tilt: np.exp(-0.5 * ((study.E_grid - (900 + 2 * tilt)) / 8) ** 2) for tilt in study.polar_tilts_deg}
af.figure_supplementary_tmd(study, 42.0, spectra).savefig("$env:TEMP/zhai-supplementary-tmd.png", dpi=150)
'@ | uv run python -
```

Expected: an 8-inch-wide figure with four complete panels, detected-unit labels, a zero y baseline, and no x-padding beyond 800–1200 eV.

- [ ] **Step 6: Commit the tested implementation**

```bash
git add checks/anchor_figures.py tests/test_anchor_figures.py
git commit -m "fix(checks): render detected Zhai supplementary spectra"
```

### Task 2: Describe the detected supplementary observable in the validation app

**Files:**
- Modify: `notebooks/validation_app.py:427-432,496-497`

**Interfaces:**
- Consumes: the existing supplementary figure builders, which now return detected coherent spectra.
- Produces: validation-app copy that accurately describes the rendered data without changing controls or cached data.

- [ ] **Step 1: Update the user-facing section description**

Replace the supplementary introduction with:

```markdown
These are separate from the Fig. 1c anchor: each run contains coherent
PXR+CBS emission, convolved with the same Zhai EDS-plus-aperture detector
response and normalized to **Phs/eV/s/nA**. WSe₂ and MoSe₂ each render four
polar-tilt panels at a selected reported thickness; h-BN overlays its four
requested tilts at 921 nm. Every study uses a 200 keV beam and the reported
energy window.
```

Change the spinner title to `Running detector-convolved supplementary spectra`.

- [ ] **Step 2: Verify the source copy**

Run: `rg -n "detector-convolved supplementary|same Zhai EDS|Phs/eV/s/nA|intrinsic coherent" notebooks/validation_app.py`

Expected: the new detected-spectrum wording is present and the obsolete `intrinsic coherent PXR+CBS emission` wording is absent.

- [ ] **Step 3: Run the focused regression tests and lint**

Run: `uv run pytest tests/test_anchor_figures.py -v; uv run ruff check checks/anchor_figures.py tests/test_anchor_figures.py notebooks/validation_app.py`

Expected: all anchor-figure tests pass and ruff reports no violations.

- [ ] **Step 4: Commit the app copy and final verification**

```bash
git add notebooks/validation_app.py
git commit -m "docs(validation): describe detected supplementary spectra"
```
