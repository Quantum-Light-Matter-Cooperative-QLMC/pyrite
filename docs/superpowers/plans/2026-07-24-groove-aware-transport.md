# Groove-Aware Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat the complete periodic sawtooth as a material/vacuum boundary for electrons, coherent photons, bremsstrahlung photons, and trajectory rendering.

**Architecture:** `montecarlo/groove.py` owns pure exact geometry. `simulate_trajectories` keeps its existing flat vectorized path unchanged and adds groove-only event handling plus separate vacuum diagnostics. Spectrum, runner, and plot layers consume those APIs without merging vacuum legs into radiating segments.

**Tech Stack:** Python, NumPy/CuPy-compatible spectrum kernels, Matplotlib, Altair, Plotly, pytest.

## Global Constraints

- `groove=None` remains bit-for-bit identical to flat-slab transport and spectra.
- Only material path contributes scattering, stopping, and radiation.
- Vacuum travel preserves energy and direction and advances clock by `L_vacuum / beta`.
- Re-entry does not increment exit counters.
- Unsupported layers or observation directions raise `ValueError`.
- Photon groove escape remains laterally periodic; electron launch and side exits retain finite-footprint behavior.
- Exact ray-plane events replace spatial ray marching in production.
- Every new physics function has source/assumptions/limiting-case docstring text and `Validation: blazed-groove-geometry`.
- Ledger status remains `unverified`; only a human may mark `signed-off`.
- Preserve unrelated dirty changes in `src/cxr_mc/run.py`, `tests/test_run.py`, `src/cxr_mc/remote.py`, and `src/cxr_mc/_remote/transport.py`.
- Run shell commands through `rtk`; run `uv` with `UV_CACHE_DIR=/tmp/cxr-mc-uv-cache`.

---

### Task 1: Exact Sawtooth Geometry

**Files:**
- Modify: `src/cxr_mc/montecarlo/groove.py`
- Modify: `src/cxr_mc/plots/trajectories.py`
- Test: `tests/test_groove.py`

**Interfaces:**
- Consumes: `GrooveSpec(spacing_ang, depth_ang, tilt_polar_rad)`.
- Produces: `surface_depth_ang(x, spec)`, `in_material(position, thickness_ang, spec, width_ang=None, height_ang=None)`, and `first_surface_event(position, direction, spec, transition=None)`.
- `first_surface_event` returns nearest strictly forward exact crossing distance or `np.inf`; `transition` accepts `"exit"` or `"entry"` and uses two-sided material predicates.

- [ ] **Step 1: Write failing analytic surface and material-predicate tests**

```python
def test_surface_depth_matches_both_analytic_facets():
    x = np.array([0.0, SPEC.depth_ang * np.tan(TP), SPEC.spacing_ang])
    assert_allclose(surface_depth_ang(x, SPEC), [0.0, SPEC.depth_ang, 0.0])


def test_material_predicate_includes_surface_and_excludes_groove_void():
    x = 0.25 * SPEC.spacing_ang
    z = surface_depth_ang(x, SPEC)
    assert in_material(np.array([x, 0.0, z]), 1e6, SPEC)
    assert not in_material(np.array([x, 0.0, z - 1e-5]), 1e6, SPEC)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py -k 'surface_depth or material_predicate'`

Expected: import failure because production helpers do not exist.

- [ ] **Step 3: Implement vectorized surface and scalar material predicate**

```python
def surface_depth_ang(x, spec):
    u = np.mod(x, spec.spacing_ang)
    x_valley = spec.depth_ang * np.tan(spec.tilt_polar_rad)
    return np.where(
        u <= x_valley,
        u / np.tan(spec.tilt_polar_rad),
        (spec.spacing_ang - u) * np.tan(spec.tilt_polar_rad),
    )


def in_material(position, thickness_ang, spec, width_ang=None, height_ang=None):
    p = np.asarray(position, dtype=float)
    inside = (p[..., 2] >= surface_depth_ang(p[..., 0], spec)) & (
        p[..., 2] <= thickness_ang
    )
    if width_ang is not None:
        inside &= np.abs(p[..., 0]) <= width_ang / 2
    if height_ang is not None:
        inside &= np.abs(p[..., 1]) <= height_ang / 2
    return inside
```

- [ ] **Step 4: Write failing exact-event tests**

```python
def test_surface_event_exit_then_later_entry_matches_reference_march():
    p = np.array([0.25 * SPEC.spacing_ang, 0.0, 0.75 * SPEC.depth_ang])
    d = np.array([1.0, 0.0, -0.2])
    d /= np.linalg.norm(d)
    s_exit = first_surface_event(p, d, SPEC, transition="exit")
    p_vac = p + s_exit * d
    s_entry = first_surface_event(p_vac, d, SPEC, transition="entry")
    assert_allclose(s_exit, _march_transition(p, d, "exit"), rtol=2e-4)
    assert_allclose(s_entry, _march_transition(p_vac, d, "entry"), rtol=2e-4)


def test_tangent_surface_event_is_skipped():
    p = np.array([0.0, 0.0, 0.0])
    d = np.array([np.sin(TP), 0.0, np.cos(TP)])
    assert np.isinf(first_surface_event(p, d, SPEC, transition="exit"))
```

- [ ] **Step 5: Run exact-event tests and verify RED**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py -k 'surface_event or tangent'`

Expected: import failure because `first_surface_event` does not exist.

- [ ] **Step 6: Implement exact periodic facet candidates**

Implement both plane families from

```text
working: dot((cos(tp), 0, -sin(tp)), r) = k * spacing * cos(tp)
relief:  dot((sin(tp), 0,  cos(tp)), r) = k * spacing * sin(tp)
```

For each family, examine adjacent floor/ceiling period indices around the transformed start coordinate, accept `s > eps`, require intersection `z` inside `[0, depth]`, classify with `in_material(q - eps*d)` and `in_material(q + eps*d)`, and choose nearest accepted event. Deterministic family order resolves apex/valley ties. Geometry epsilon is `max(32*np.finfo(float).eps*spacing, 1e-12*spacing)`.

- [ ] **Step 7: Remove duplicated plot surface formula and verify GREEN**

Make `plots.trajectories.groove_profile_z` delegate to `surface_depth_ang`.

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py`

Expected: all groove geometry tests pass.

- [ ] **Step 8: Commit**

```bash
rtk git add src/cxr_mc/montecarlo/groove.py src/cxr_mc/plots/trajectories.py tests/test_groove.py
rtk git commit -m "feat(transport): add exact groove events"
```

### Task 2: Electron Exit, Vacuum Flight, and Re-entry

**Files:**
- Modify: `src/cxr_mc/montecarlo/transport.py`
- Test: `tests/test_groove.py`

**Interfaces:**
- Consumes: Task 1 geometry helpers.
- Produces additive arrays `vacuum_start_ang`, `vacuum_end_ang`, `vacuum_E_keV`, `vacuum_t_ang`, `vacuum_elec_id`.

- [ ] **Step 1: Write failing diagnostics and invariant tests**

```python
def test_groove_transport_records_only_material_segments_and_vacuum_invariants():
    out = simulate_trajectories(**_SIM_KW, groove=SPEC)
    assert set(("vacuum_start_ang", "vacuum_end_ang", "vacuum_E_keV",
                "vacuum_t_ang", "vacuum_elec_id")) <= out.keys()
    assert np.all(in_material(out["r_mid"], out["thickness_ang"], SPEC))
    assert out["vacuum_start_ang"].shape == out["vacuum_end_ang"].shape
    assert out["vacuum_start_ang"].shape[1:] == (3,)


def test_groove_none_preserves_legacy_arrays_bit_for_bit():
    old = simulate_trajectories(**_SIM_KW)
    explicit = simulate_trajectories(**_SIM_KW, groove=None)
    for key in old:
        assert_equal(old[key], explicit[key])
    assert explicit["vacuum_start_ang"].shape == (0, 3)
```

Use deterministic seed and a shallow, physically meaningful sawtooth. Add a controlled geometry/RNG test proving a vacuum leg keeps energy/direction, advances its next material segment start time by `L_vacuum / beta`, and resumes material scattering/stopping.

- [ ] **Step 2: Run transport tests and verify RED**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py -k 'transport or vacuum or reentry'`

Expected: missing vacuum keys or material midpoint outside profile.

- [ ] **Step 3: Add groove-only boundary event path**

Keep the existing `groove is None` boundary block byte-for-byte. In the groove branch:

```python
s_surface = np.array([
    first_surface_event(pi, di, groove, transition="exit")
    for pi, di in zip(p, d, strict=False)
])
surface_first = s_surface < step
step = np.minimum(step, s_surface)
```

Record/apply stopping only to truncated material `step`. For each surface exit, call `first_surface_event(exit_point, direction, groove, transition="entry")`. If finite, append vacuum diagnostics, advance position to the entry-side epsilon nudge, advance clock by vacuum length/current beta, preserve energy/direction, and leave electron alive. If infinite, count permanent entrance-face exit and terminate. Bound groove event handling by `max_steps`; raise `RuntimeError` on repeated zero-length events.

- [ ] **Step 4: Return typed empty and populated vacuum arrays**

```python
"vacuum_start_ang": np.concatenate(vac_start) if vac_start else np.empty((0, 3)),
"vacuum_end_ang": np.concatenate(vac_end) if vac_end else np.empty((0, 3)),
"vacuum_E_keV": np.concatenate(vac_E) if vac_E else np.empty(0),
"vacuum_t_ang": np.concatenate(vac_t0) if vac_t0 else np.empty(0),
"vacuum_elec_id": np.concatenate(vac_id).astype(np.int64) if vac_id else np.empty(0, dtype=np.int64),
```

- [ ] **Step 5: Verify GREEN and flat regression**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py tests/test_montecarlo.py tests/test_finite_transverse_geometry.py`

Expected: all selected tests pass; explicit and implicit `groove=None` outputs remain bitwise equal.

- [ ] **Step 6: Commit**

```bash
rtk git add src/cxr_mc/montecarlo/transport.py tests/test_groove.py
rtk git commit -m "feat(transport): cross groove vacuum gaps"
```

### Task 3: Grooved Bremsstrahlung Escape and Runner Forwarding

**Files:**
- Modify: `src/cxr_mc/montecarlo/spectrum.py`
- Modify: `src/cxr_mc/montecarlo/runner.py`
- Test: `tests/test_groove.py`
- Test: `tests/test_spectrum_escape_helpers.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `escape_distance_ang`, runner transport payload `groove`.
- Produces: `mc_brem_spectrum(..., groove=None)` and `_brem_wide_from_segments(..., groove=None)`.

- [ ] **Step 1: Write failing Beer-Lambert and error tests**

```python
def test_brem_groove_gain_matches_beer_lambert_escape():
    flat = mc_brem_spectrum(segments, grid, **kw)
    grooved = mc_brem_spectrum(segments, grid, groove=SPEC, n_hat=_groove_nhat(SPEC), **kw)
    expected = _independent_brem_with_escape(
        segments, grid, escape_distance_ang(segments["r_mid"][:, 0],
                                            segments["r_mid"][:, 2], SPEC), **kw
    )
    assert_allclose(grooved, expected, rtol=2e-12)
    assert np.all(grooved >= flat)


def test_brem_groove_rejects_layers_and_wrong_direction():
    with pytest.raises(ValueError):
        mc_brem_spectrum(segments, grid, groove=SPEC, layers=layers, **kw)
    with pytest.raises(ValueError):
        mc_brem_spectrum(segments, grid, groove=SPEC, n_hat=[0, 0, -1], **kw)
```

- [ ] **Step 2: Write failing runner forwarding tests**

Patch `runner.mc_brem_spectrum`, capture keyword arguments, and assert the identical `GrooveSpec` reaches normal spectrum, brem-only rebuild, and regenerated wide-brem calls. Existing line-only forwarding remains covered.

- [ ] **Step 3: Run and verify RED**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py tests/test_spectrum_escape_helpers.py tests/test_run.py -k 'groove and (brem or forwarding)'`

Expected: unexpected `groove` keyword or missing forwarded argument.

- [ ] **Step 4: Implement spectrum API and restrictions**

Add `groove=None` to `mc_brem_spectrum`. When present, reject `layers`, require `n_hat` equal within `_THETA_TOL` to `(cos(tp), 0, -sin(tp))`, and set

```python
L_esc = xp.asarray(
    escape_distance_ang(seg_r[:, 0], z_mid, groove),
    dtype=REAL,
)
```

before finite-footprint selection, so periodic groove escape takes precedence. Share the same direction validator with coherent `mc_spectrum`.

- [ ] **Step 5: Forward groove through all runner paths**

Add `groove=None` to `_brem_wide_from_segments`; pass it into every `mc_brem_spectrum` call. Forward from `_brem_for_case` and `_spectrum_case`. Do not change `repair_brem_wide`; it already delegates to `_brem_for_case`.

- [ ] **Step 6: Verify GREEN and chunk invariance**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py tests/test_spectrum_escape_helpers.py tests/test_run.py tests/test_chunk_invariance.py`

Expected: all selected tests pass.

- [ ] **Step 7: Commit only task files/hunks**

```bash
rtk git add src/cxr_mc/montecarlo/spectrum.py src/cxr_mc/montecarlo/runner.py tests/test_groove.py tests/test_spectrum_escape_helpers.py
rtk git add -p tests/test_run.py
rtk git commit -m "feat(spectrum): use grooved brem escape"
```

### Task 4: Vacuum Legs in 2D and 3D Trajectory Views

**Files:**
- Modify: `src/cxr_mc/plots/trajectories.py`
- Modify: `src/cxr_mc/plots/altair_trajectories.py`
- Modify: `src/cxr_mc/plots/plotly_trajectories.py`
- Test: `tests/test_altair_trajectories.py`
- Test: `tests/test_plotly_trajectories.py`

**Interfaces:**
- Consumes: transport `vacuum_*` arrays.
- Produces display-unit `vacuum_start_xyz`, `vacuum_end_xyz`, `vacuum_E`, `vacuum_t_fs`, `vacuum_elec_id`; separate faint render layers/traces.

- [ ] **Step 1: Write failing shared-data and renderer tests**

```python
def test_vacuum_legs_stay_separate_from_radiating_track_data():
    data = _trajectory_data(case, Ne=4, seed=7)
    assert len(data["vacuum_start_xyz"]) == len(data["vacuum_end_xyz"])
    assert len(data["start_xyz"]) == len(data["E"])


def test_plotly_vacuum_trace_is_faint_and_reveal_gated():
    trace = vacuum_legs_trace(data, reveal_until_fs=1.0)
    assert trace.name == "vacuum legs"
    assert trace.opacity == 0.4
    assert np.nanmax(trace.customdata[:, 1]) <= 1.0
```

Add Altair assertions for separate low-opacity `mark_rule`; confirm vacuum rows never enter `track_segments_frame`.

- [ ] **Step 2: Run and verify RED**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_altair_trajectories.py tests/test_plotly_trajectories.py -k vacuum`

Expected: missing shared data keys/render helpers.

- [ ] **Step 3: Extend shared trajectory data**

Project vacuum endpoints through the same beam/detector basis and display-unit conversion as material endpoints. Keep arrays separate. Add empty typed arrays for flat trajectories.

- [ ] **Step 4: Render separate faint vacuum legs**

Matplotlib: draw separate low-alpha rules using terminal-exit styling.

Altair: add `vacuum_segments_frame(data)` and layer its low-opacity rules beneath energy-colored material tracks.

Plotly: add a `"vacuum legs"` `Scatter3d` trace using Turbo energy colors, width 3, opacity 0.4, and `vacuum_t_fs` reveal filtering. Update both static figure and animation trace indices so future vacuum legs remain hidden.

- [ ] **Step 5: Verify GREEN**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_altair_trajectories.py tests/test_plotly_trajectories.py`

Expected: all renderer and animation tests pass.

- [ ] **Step 6: Commit**

```bash
rtk git add src/cxr_mc/plots/trajectories.py src/cxr_mc/plots/altair_trajectories.py src/cxr_mc/plots/plotly_trajectories.py tests/test_altair_trajectories.py tests/test_plotly_trajectories.py
rtk git commit -m "feat(plots): render groove vacuum legs"
```

### Task 5: Physics Review and Final Verification

**Files:**
- Modify only if required: `docs/physics-validation-ledger.md`
- Create by independent verifier: `docs/validation/blazed-groove-geometry.md`

**Interfaces:**
- Consumes: completed implementation and tests.
- Produces: independent validation report; ledger remains non-human status.

- [ ] **Step 1: Run focused integration suite**

Run:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test \
  tests/test_groove.py tests/test_spectrum_escape_helpers.py tests/test_run.py \
  tests/test_altair_trajectories.py tests/test_plotly_trajectories.py \
  tests/test_chunk_invariance.py tests/test_montecarlo_exports.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run lint, typecheck, and full tests**

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py lint
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py typecheck
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test
```

Expected: all commands exit 0.

- [ ] **Step 3: Dispatch fresh-context physics verifier**

Verifier derives facet signs/bands before reading implementation, checks tangent/apex/valley handling, flat-profile limit, coherent no-re-entry proof, Beer-Lambert brem path, memoryless exponential resampling, vacuum clock units, and finite-footprint decoupling. Verifier may write only `docs/validation/blazed-groove-geometry.md`.

- [ ] **Step 4: Run final broad code review**

Review complete branch diff against `docs/superpowers/specs/2026-07-24-groove-aware-transport-design.md`. Fix every Critical/Important finding, rerun covering tests, then re-review.

- [ ] **Step 5: Final verification**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify`

Expected: exit 0 with no failures.
