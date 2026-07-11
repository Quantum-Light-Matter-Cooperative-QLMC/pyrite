# Zhai Reported Tilt Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the supplementary validation cases reproduce the orientations and beam energies reported by Zhai et al., without silently assigning zero to unreported TEM azimuths.

**Architecture:** Introduce an immutable per-spectrum condition containing beam energy, polar tilt, and optional azimuth. A study owns four conditions: four energies at the reported 17 degree/130 degree h-BN orientation, or four polar tilts with unknown azimuth for each TMD. Modeling requires either a reported azimuth or an explicit exploratory override, so the published TMD metadata remains inspectable and the app cannot accidentally present zero as reported.

**Tech Stack:** Python dataclasses, NumPy, pytest, marimo, Markdown provenance.

## Global Constraints

- Preserve the published h-BN conditions: 921 nm, 17 degrees polar, 130 degrees azimuth, and 17.5/20/22.5/25 keV.
- Preserve the published MoSe2/WSe2 conditions: 200 keV and 10/15/17.5/20 degrees polar.
- Represent the unreported MoSe2/WSe2 azimuth as `None`; never substitute zero implicitly.
- Do not change production transport equations or physics-ledger claim states.

---

### Task 1: Freeze the reported condition metadata

**Files:**
- Modify: `tests/test_anchor_figures.py`
- Modify: `checks/anchor_figures.py`

**Interfaces:**
- Produces: `SupplementaryCondition(energy_keV: float, polar_tilt_deg: float, azimuth_deg: float | None)`.
- Produces: `SupplementaryCoherentStudy.conditions: tuple[SupplementaryCondition, ...]`.
- Produces: `SupplementaryCoherentStudy.condition_labels: tuple[str, ...]` for stable plotting/cache keys.

- [ ] **Step 1: Write failing metadata tests**

Replace the old shared-energy assertions with exact expected conditions:

```python
assert tuple((c.energy_keV, c.polar_tilt_deg, c.azimuth_deg) for c in hbn.conditions) == (
    (17.5, 17.0, 130.0),
    (20.0, 17.0, 130.0),
    (22.5, 17.0, 130.0),
    (25.0, 17.0, 130.0),
)
assert tuple(c.azimuth_deg for c in wse2.conditions) == (None, None, None, None)
assert tuple(c.polar_tilt_deg for c in wse2.conditions) == (10.0, 15.0, 17.5, 20.0)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_anchor_figures.py::test_supplementary_studies_match_requested_windows_and_thicknesses -v`

Expected: FAIL because `conditions` does not exist and h-BN still has a shared 200 keV energy.

- [ ] **Step 3: Implement the condition dataclass and registries**

Add:

```python
@dataclass(frozen=True)
class SupplementaryCondition:
    energy_keV: float
    polar_tilt_deg: float
    azimuth_deg: float | None
```

Replace the study's shared `polar_tilts_deg` and `energy_keV` fields with `conditions`, populating the exact tuples in the global constraints.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_anchor_figures.py::test_supplementary_studies_match_requested_windows_and_thicknesses -v`

Expected: PASS.

### Task 2: Apply known azimuths and reject unknown ones

**Files:**
- Modify: `tests/test_anchor_figures.py`
- Modify: `checks/anchor_figures.py`

**Interfaces:**
- Consumes: `SupplementaryCoherentStudy.conditions`.
- Produces: spectra keyed by `SupplementaryCondition`.
- `model_coherent_spectra(..., exploratory_azimuth_deg: float | None = None)` raises `ValueError("azimuth is unreported")` when a condition has `azimuth_deg is None` and no explicit override is supplied.

- [ ] **Step 1: Write failing geometry tests**

Monkeypatch `tilted_geometry`, transport, and spectrum helpers, then assert the h-BN calls include `np.deg2rad(130.0)`. Add a TMD test asserting:

```python
with pytest.raises(ValueError, match="azimuth is unreported"):
    af.model_coherent_spectra(af.supplementary_study("wse2"), 42.0, ne=1)
```

Add a second assertion that `exploratory_azimuth_deg=35.0` reaches geometry with 35 degrees while leaving `condition.azimuth_deg is None`.

- [ ] **Step 2: Verify RED**

Run the two new tests with `uv run pytest tests/test_anchor_figures.py -k "reported_azimuth or unreported_azimuth" -v`.

Expected: FAIL because geometry receives no azimuth and TMD modeling silently uses zero.

- [ ] **Step 3: Implement condition-driven modeling and cache keys**

Iterate over `study.conditions`, resolve `azimuth_deg = condition.azimuth_deg` or the explicit exploratory override, reject when both are `None`, and call:

```python
beam_dir, n_hat = tilted_geometry(
    study.theta_obs_rad,
    np.deg2rad(condition.polar_tilt_deg),
    np.deg2rad(azimuth_deg),
)
```

Use `condition.energy_keV` for transport and the immutable condition as the spectrum key. Thread the override through `cached_coherent_spectra`, include it in the cache key, and bump `_ZHAI_CACHE_SCHEMA` so old azimuth-zero caches cannot be reused.

- [ ] **Step 4: Update figure builders and synthetic fixtures**

Make TMD titles show polar tilt plus `azimuth unreported`; make h-BN line labels show beam energy and its title show `polar 17 degrees, azimuth 130 degrees`. Update overview selection to use the final condition rather than a numeric tilt key.

- [ ] **Step 5: Verify GREEN**

Run: `uv run pytest tests/test_anchor_figures.py -v`

Expected: all anchor-figure tests PASS.

### Task 3: Correct the app copy and provenance

**Files:**
- Modify: `notebooks/validation_app.py`
- Modify: `docs/validation/zhai-supplementary.md`

**Interfaces:**
- Consumes: corrected study conditions and figure labels from Tasks 1-2.
- Produces: user-facing text that distinguishes reported angles from unknown azimuths.

- [ ] **Step 1: Update validation-app copy**

State that MoSe2/WSe2 provide four reported polar tilts at 200 keV but no reported azimuth. Add a numeric `exploratory azimuth` control for those studies and pass its value explicitly to the cache/model call; do not render it for h-BN. State that 921 nm h-BN uses 17 degrees polar and 130 degrees azimuth across 17.5-25 keV.

- [ ] **Step 2: Replace the provenance claim**

Transcribe all Supplementary Table 4 graphite and h-BN `(theta_til, phi_til)` pairs. Replace “azimuthal angle (resolved)” with “TMD azimuthal angle (unreported)” and explicitly retract the claim that the SI establishes `phi = 0`.

- [ ] **Step 3: Verify documentation and notebook hygiene**

Run: `uv run nbstripout` and `rg -n "azimuth = 0 is correct|implicitly computed at \*\*azimuth = 0" docs/validation/zhai-supplementary.md notebooks/validation_app.py`

Expected: notebook cleanup exits 0 and the search returns no matches.

### Task 4: Full verification and integration

**Files:**
- Verify all modified files.

**Interfaces:**
- Produces: a verified feature commit suitable for fast-forwarding `main`.

- [ ] **Step 1: Run focused and static checks**

Run: `uv run pytest tests/test_anchor_figures.py -v`, `uv run ruff check .`, and `uv run pyright`.

Expected: all commands exit 0.

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest`.

Expected: all tests pass.

- [ ] **Step 3: Commit implementation**

Stage only the intended source, test, notebook, and provenance files; commit as `fix(validation): use reported Zhai tilt orientations`.

- [ ] **Step 4: Fast-forward main and verify**

From `C:/dev/cxr-mc`, run `git merge --ff-only codex/zhai-reported-tilts`, then rerun `uv run pytest tests/test_anchor_figures.py -v` on `main`.

Expected: fast-forward succeeds and focused tests pass on `main`.
