# Finite Transverse Crystal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Model an optional finite rectangular crystal footprint so lateral electron escape and six-face photon self-absorption reduce yield per incident electron.

**Architecture:** Add a vectorized ray-to-rectangular-prism boundary primitive, then call it only from finite-footprint branches in transport and spectra. The public sweep/case API uses full mm dimensions; segment dictionaries store validated Å dimensions. The existing slab calculations remain unexecuted and bit-for-bit identical when both dimensions are omitted.

**Tech Stack:** Python 3.13+, NumPy, optional CuPy through the existing `xp` backend, pytest, uv, Ruff, Pyright.

## Global Constraints

- The sample-frame prism is `[-width/2,width/2] × [-height/2,height/2] × [0,thickness]`.
- `crystal_width_mm` and `crystal_height_mm` are full dimensions; use `1 mm = 1e7 Å`.
- Both dimensions are `None` for the legacy slab, or both are strictly positive; partial/zero/negative pairs raise `ValueError`.
- Retain the fixed far-field `n_hat`; do not implement detector distance, parallax, polygonal footprints, or rotated faces.
- A Gaussian entry point outside the finite prism increments `n_missed`, makes no segments, and remains in `Ne`.
- New physics docstrings include assumptions, limiting case, and `Validation: finite-transverse-crystal`; ledger the claim as `unverified`. An independent verifier authors the validation write-up.
- Run checks with `uv run python scripts/dev.py …`.

---

## File Structure

- `src/cxr_mc/montecarlo/geometry.py`: face constants, dimension validation, vectorized first-exit primitive.
- `src/cxr_mc/montecarlo/transport.py`: finite transport, incident misses, side diagnostics, stored Å dimensions.
- `src/cxr_mc/materials/attenuation.py`: capped per-layer ray path.
- `src/cxr_mc/montecarlo/spectrum.py`: finite escape distance in coherent and brem paths.
- `src/cxr_mc/montecarlo/__init__.py`, `tests/test_montecarlo_exports.py`: frozen re-export contract.
- `src/cxr_mc/sweep.py`, `src/cxr_mc/montecarlo/runner.py`: optional swept public dimensions and driver forwarding.
- `tests/test_finite_transverse_geometry.py` (new), `tests/test_montecarlo.py`, `tests/test_multilayer.py`, `tests/test_spectrum_escape_helpers.py`, `tests/test_chunk_invariance.py`, `tests/test_sweep.py`, `tests/test_run.py`: regression coverage.
- `docs/physics-validation-ledger.md`, `docs/validation/finite-beam-size.md`, and independently-authored `docs/validation/finite-transverse-crystal.md`: validation records.

### Task 1: Implement and test shared rectangular-prism exits

**Files:**
- Modify: `src/cxr_mc/montecarlo/geometry.py`
- Modify: `src/cxr_mc/montecarlo/__init__.py`
- Modify: `tests/test_montecarlo_exports.py`
- Create: `tests/test_finite_transverse_geometry.py`

**Interfaces:**
- Produces `validate_transverse_dimensions(width, height, *, unit) -> tuple[float | None, float | None]`.
- Produces `first_prism_exit(r, d, *, z_min_ang, z_max_ang, width_ang=None, height_ang=None, xp=np) -> tuple[Array, Array]`.
- Face values are `X_MIN=0, X_MAX=1, Y_MIN=2, Y_MAX=3, Z_MIN=4, Z_MAX=5`; ties select the lowest value.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_finite_transverse_geometry.py`:

```python
import numpy as np
import pytest

from cxr_mc.montecarlo.geometry import X_MAX, Z_MIN, Z_MAX, first_prism_exit, validate_transverse_dimensions


def test_first_prism_exit_selects_nearest_face_and_deterministic_corner():
    r = np.array([[4., 0., 5.], [0., 0., 5.], [0., 0., 5.]])
    d = np.array([[1., 0., 0.], [1., 1., 0.], [0., 0., -1.]])
    distance, face = first_prism_exit(r, d, z_min_ang=0., z_max_ang=10., width_ang=10., height_ang=10.)
    np.testing.assert_allclose(distance, [1., 5., 5.])
    assert face.tolist() == [X_MAX, X_MAX, Z_MIN]


def test_first_prism_exit_ignores_parallel_faces_and_falls_back_to_slab():
    distance, face = first_prism_exit(
        np.array([[0., 0., 2.], [0., 0., 7.]]),
        np.array([[0., 1., -.5], [0., 1., .5]]),
        z_min_ang=0., z_max_ang=10.,
    )
    np.testing.assert_allclose(distance, [4., 6.])
    assert face.tolist() == [Z_MIN, Z_MAX]


@pytest.mark.parametrize(("width", "height"), [(None, 1.), (1., None), (0., 1.), (-1., 1.)])
def test_validate_transverse_dimensions_rejects_invalid_pairs(width, height):
    with pytest.raises(ValueError):
        validate_transverse_dimensions(width, height, unit="Ang")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python scripts/dev.py test tests/test_finite_transverse_geometry.py -v`

Expected: FAIL with an import error for `first_prism_exit`.

- [ ] **Step 3: Implement the exact helper**

In `geometry.py`, add constants plus validation. Add `first_prism_exit` with the following candidate construction; use `xp.where(component != 0, numerator / component, xp.inf)` before filtering non-forward values so zero components never divide:

```python
def first_prism_exit(r, d, *, z_min_ang, z_max_ang, width_ang=None, height_ang=None, xp=np):
    width_ang, height_ang = validate_transverse_dimensions(width_ang, height_ang, unit="Ang")
    r, d = xp.asarray(r), xp.asarray(d)
    if width_ang is None:
        numerators = (z_min_ang - r[..., 2], z_max_ang - r[..., 2])
        components = (d[..., 2], d[..., 2])
        faces = xp.asarray([Z_MIN, Z_MAX])
    else:
        hx, hy = width_ang / 2.0, height_ang / 2.0
        numerators = (-hx-r[..., 0], hx-r[..., 0], -hy-r[..., 1], hy-r[..., 1],
                      z_min_ang-r[..., 2], z_max_ang-r[..., 2])
        components = (d[..., 0], d[..., 0], d[..., 1], d[..., 1], d[..., 2], d[..., 2])
        faces = xp.asarray([X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX])
    candidates = xp.stack(
        [xp.where(c != 0.0, n / c, xp.inf) for n, c in zip(numerators, components, strict=True)],
        axis=-1,
    )
    candidates = xp.where(candidates > 0.0, candidates, xp.inf)
    choice = xp.argmin(candidates, axis=-1)
    return xp.take_along_axis(candidates, choice[..., None], axis=-1)[..., 0], faces[choice]
```

Document that the origins are inside the prism and the `None` limit is a z-only slab. Re-export the helper and face constants in `montecarlo.__init__` and add every re-export to `FROZEN_EXPORTS`.

- [ ] **Step 4: Run the focused tests**

Run: `uv run python scripts/dev.py test tests/test_finite_transverse_geometry.py tests/test_montecarlo_exports.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/montecarlo/geometry.py src/cxr_mc/montecarlo/__init__.py tests/test_finite_transverse_geometry.py tests/test_montecarlo_exports.py
git commit -m "feat: add finite-prism ray exit geometry"
```

### Task 2: Enforce the prism in transport and preserve incident normalization

**Files:**
- Modify: `src/cxr_mc/montecarlo/transport.py`
- Modify: `tests/test_montecarlo.py`

**Interfaces:**
- Consumes `simulate_trajectories(..., crystal_width_mm=None, crystal_height_mm=None)`.
- Produces segment keys `crystal_width_ang`, `crystal_height_ang`, `n_side_exited`, and `n_missed`.

- [ ] **Step 1: Write failing transport tests**

Append to `tests/test_montecarlo.py`:

```python
def test_finite_footprint_truncates_lateral_transport_and_counts_side_exit():
    cp = crystal_params("hbn")
    segs = simulate_trajectories(
        30., 64, 1e8, composition=cp["composition"], seed=4, elastic_model="sr",
        beam_dir=[.8, 0., .6], max_steps=1, crystal_width_mm=1e-6, crystal_height_mm=1e-6,
    )
    assert segs["n_side_exited"] == 64
    assert segs["n_transmitted"] == segs["n_backscattered"] == 0
    assert np.all(np.abs(segs["r_mid"][:, 0]) < 5.)


def test_finite_footprint_counts_missed_gaussian_entries_without_changing_ne():
    cp = crystal_params("hbn")
    segs = simulate_trajectories(
        30., 2_000, 100., composition=cp["composition"], seed=7, elastic_model="sr",
        max_steps=1, beam_fwhm_mm=1., crystal_width_mm=1e-6, crystal_height_mm=1e-6,
    )
    assert segs["Ne"] == 2_000
    assert segs["n_missed"] > 1_900
    assert np.all(np.abs(segs["r_mid"][:, :2]) <= 5.)


def test_omitted_footprint_is_bitwise_legacy_transport():
    cp = crystal_params("hbn")
    kw = dict(composition=cp["composition"], seed=19, elastic_model="sr", max_steps=5)
    old = simulate_trajectories(30., 40, 100., **kw)
    new = simulate_trajectories(30., 40, 100., crystal_width_mm=None, crystal_height_mm=None, **kw)
    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "elec_id", "layer"):
        assert np.array_equal(old[key], new[key]), key
```

Also expand the existing beam-FWHM invariance test to compare the two new zero-valued counters, and parametrize invalid partial/zero/negative mm pairs.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python scripts/dev.py test tests/test_montecarlo.py -k "finite_footprint or beam_fwhm" -v`

Expected: FAIL with unexpected `crystal_width_mm`.

- [ ] **Step 3: Implement finite transport**

Add the two parameters and convert them once:

```python
width_mm, height_mm = validate_transverse_dimensions(
    crystal_width_mm, crystal_height_mm, unit="mm"
)
width_ang = None if width_mm is None else width_mm * 1.0e7
height_ang = None if height_mm is None else height_mm * 1.0e7
finite_footprint = width_ang is not None
```

After optional Gaussian offsets, initialize `alive` from `abs(pos[:, 0]) <= width_ang / 2` and `abs(pos[:, 1]) <= height_ang / 2` only in the finite branch; set `n_missed = int((~alive).sum())`. Keep today's all-true `alive` initialization in the legacy branch.

For each layer free flight, use `first_prism_exit(p, d, z_min_ang=z_top_L, z_max_ang=z_bot_L, width_ang=width_ang, height_ang=height_ang)` only when finite. Replace a sampled step only where `step > exit_distance`. `X_*`/`Y_*` terminate and increment `n_side_exited`; external `Z_MIN`/`Z_MAX` retain current backscatter/transmission behavior; internal z faces retain the existing nudge into the neighboring layer. Return typed empty arrays rather than `np.concatenate([])` when all entries miss. Store internal dimensions and calculate:

```python
"n_stopped": int(Ne - n_back - n_trans - n_side - n_missed),
"n_side_exited": n_side,
"n_missed": n_missed,
"crystal_width_ang": width_ang,
"crystal_height_ang": height_ang,
```

Extend the docstring with prism bounds, mm/Å conversion, spillover normalization, the all-`None` limiting case, and `Validation: finite-transverse-crystal`.

- [ ] **Step 4: Run the transport tests**

Run: `uv run python scripts/dev.py test tests/test_montecarlo.py -k "finite_footprint or beam_fwhm" -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/montecarlo/transport.py tests/test_montecarlo.py
git commit -m "feat: stop finite crystals at lateral faces"
```

### Task 3: Cap layered attenuation at the selected face

**Files:**
- Modify: `src/cxr_mc/materials/attenuation.py`
- Modify: `src/cxr_mc/montecarlo/__init__.py`
- Modify: `tests/test_multilayer.py`
- Modify: `tests/test_montecarlo_exports.py`

**Interfaces:**
- Produces `_layer_path_length(z_mid, n_z, exit_distance_ang, z_top, z_bot)`.
- Extends `_stack_tau(layers, z_mid, n_z, E, *, exit_distance_ang=None)`; omitted cap uses the old code path unchanged.

- [ ] **Step 1: Write failing layer tests**

Add to `tests/test_multilayer.py`:

```python
def test_layer_path_length_keeps_lateral_ray_in_current_layer():
    path = _layer_path_length(np.array([250., 750.]), 0., np.array([7., 7.]), 0., 500.)
    np.testing.assert_allclose(path, [7., 0.])


def test_stack_tau_with_side_exit_stops_before_substrate():
    film, sub = [("C", .176)], [("W", .0632)]
    layers = [(0., 500., film), (500., 1000., sub)]
    z, energy = np.array([250.]), np.array([1500.])
    tau = _stack_tau(layers, z, 0., energy, exit_distance_ang=np.array([10.]))
    np.testing.assert_allclose(tau, 10. * _mu_total_inv_ang(film, energy))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python scripts/dev.py test tests/test_multilayer.py -k "layer_path_length or side_exit or stack_tau" -v`

Expected: FAIL with missing helper/signature.

- [ ] **Step 3: Implement capped ray paths**

For nonzero `n_z`, intersect `[0, exit_distance_ang]` with the two ray parameters at `z_top` and `z_bot`; its nonnegative overlap is the path length in that layer. For `n_z == 0`, return the full exit distance only for the layer containing `z_mid`. Use the existing NumPy/CuPy backend dispatch.

When `exit_distance_ang is None`, leave `_stack_tau`'s current `_layer_dz / abs(n_z)` loop verbatim. When supplied, use:

```python
tau = 0.0
for z_top, z_bot, comp in layers:
    path = _layer_path_length(z_mid, n_z, exit_distance_ang, float(z_top), float(z_bot))
    tau = tau + _mu_total_inv_ang(comp, E) * path
return tau
```

Re-export `_layer_path_length` and add it to `FROZEN_EXPORTS`.

- [ ] **Step 4: Run the tests**

Run: `uv run python scripts/dev.py test tests/test_multilayer.py tests/test_montecarlo_exports.py -v`

Expected: PASS; existing uncapped multilayer formulas remain green.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/materials/attenuation.py src/cxr_mc/montecarlo/__init__.py tests/test_multilayer.py tests/test_montecarlo_exports.py
git commit -m "feat: cap layered attenuation at finite crystal faces"
```

### Task 4: Apply finite escape distances to coherent and bremsstrahlung spectra

**Files:**
- Modify: `src/cxr_mc/montecarlo/spectrum.py`
- Modify: `tests/test_spectrum_escape_helpers.py`
- Modify: `tests/test_chunk_invariance.py`

**Interfaces:**
- Produces `_segment_escape_distance(segments, n_hat, *, xp)`.
- Leaves `mc_spectrum` and `mc_brem_spectrum` public signatures unchanged; they consume stored segment dimensions.

- [ ] **Step 1: Write failing spectrum tests**

Extend `tests/test_spectrum_escape_helpers.py`:

```python
from cxr_mc.montecarlo.spectrum import _segment_escape_distance


def test_segment_escape_distance_prefers_near_side_face():
    segments = {
        "r_mid": np.array([[4., 0., 5.]]), "thickness_ang": 100.,
        "crystal_width_ang": 10., "crystal_height_ang": 10.,
    }
    np.testing.assert_allclose(
        _segment_escape_distance(segments, np.array([1., 0., .01]), xp=np), [1.]
    )
```

Add a one-segment valid fixture (`E_keV`, `v_hat`, `L_ang`, `t_ang`, `elec_id`, `layer`, `Ne`) and evaluate a lateral-facing `n_hat` for widths `10 Å` and `1000 Å`. Assert both `mc_spectrum` and `mc_brem_spectrum` are larger for `10 Å`: amplitudes are equal and Beer--Lambert attenuation is shorter. Add the same fixture to `tests/test_chunk_invariance.py` and assert `chunk=1` matches the one-shot result for both functions.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python scripts/dev.py test tests/test_spectrum_escape_helpers.py tests/test_chunk_invariance.py -v`

Expected: FAIL with missing `_segment_escape_distance`.

- [ ] **Step 3: Implement finite-spectrum branches**

Implement:

```python
def _segment_escape_distance(segments, n_hat, *, xp):
    r = xp.asarray(segments["r_mid"], dtype=REAL)
    width, height = segments.get("crystal_width_ang"), segments.get("crystal_height_ang")
    if width is None or height is None:
        return _escape_length(r[:, 2], segments["thickness_ang"], n_hat[2])
    distance, _ = first_prism_exit(
        r, n_hat, z_min_ang=0., z_max_ang=segments["thickness_ang"],
        width_ang=width, height_ang=height, xp=xp,
    )
    return distance
```

In `mc_spectrum`, call this helper after the `idx` selection. For a finite footprint use `tau = L_esc * mu` without layers and `_stack_tau(..., exit_distance_ang=L_esc)` with layers; preserve the existing slab branches exactly otherwise. In `mc_brem_spectrum`, use the helper for the finite no-layer path. For finite layers build `tau` inside each chunk from `_layer_path_length(z_mid[sl], n_hat[2], L_esc[sl], z_top, z_bot)[:, None] * mu_i[None, :]`; retain the existing `_layer_dz * inv_nz` loop for infinite footprints.

Add six-face attenuation, fixed-far-field, all-`None` limit, and `Validation: finite-transverse-crystal` to both physics docstrings.

- [ ] **Step 4: Run the tests**

Run: `uv run python scripts/dev.py test tests/test_spectrum_escape_helpers.py tests/test_chunk_invariance.py tests/test_multilayer.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/montecarlo/spectrum.py tests/test_spectrum_escape_helpers.py tests/test_chunk_invariance.py
git commit -m "feat: attenuate radiation to finite crystal faces"
```

### Task 5: Expose finite footprints in sweeps and runner paths

**Files:**
- Modify: `src/cxr_mc/sweep.py`
- Modify: `src/cxr_mc/montecarlo/runner.py`
- Modify: `tests/test_sweep.py`
- Modify: `tests/test_run.py`

**Interfaces:**
- Consumes `Sweep(crystal_width_mm: ScalarOrSeq | None = None, crystal_height_mm: ScalarOrSeq | None = None)`.
- Produces cases with those mm keys and finite names ending `footprint=<width:g>x<height:g>mm`.

- [ ] **Step 1: Write failing sweep/runner tests**

Add to `tests/test_sweep.py`:

```python
def test_build_cases_sweeps_rectangular_footprints_and_labels_them():
    cases = build_cases(Sweep(
        material="mose2", thickness_ang=100., energy_keV=30., tilt_deg=0.,
        crystal_width_mm=[.1, .2], crystal_height_mm=[.3, .4],
    ))
    assert {(c["crystal_width_mm"], c["crystal_height_mm"]) for c in cases} == {
        (.1, .3), (.1, .4), (.2, .3), (.2, .4),
    }
    assert all("footprint=" in c["name"] for c in cases)


@pytest.mark.parametrize(("width", "height"), [(.1, None), (None, .1), (0., .1), (.1, -.1)])
def test_build_cases_rejects_invalid_footprint(width, height):
    with pytest.raises(ValueError):
        build_cases(Sweep(material="mose2", crystal_width_mm=width, crystal_height_mm=height))
```

In `tests/test_run.py`, monkeypatch `runner.simulate_trajectories`, invoke `_transport_case` and `_brem_for_case` with both finite fields, and assert every captured call receives those exact values.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python scripts/dev.py test tests/test_sweep.py tests/test_run.py -k "footprint or transport_case or brem_for_case" -v`

Expected: FAIL because `Sweep` has no footprint fields and runner does not forward them.

- [ ] **Step 3: Implement public plumbing**

Add the two optional `ScalarOrSeq` fields. Before the `build_cases` product, turn `(None, None)` into one legacy pair; otherwise validate both finite arrays and cross the arrays with thickness, tilt, and azimuth. Put the two values in every case; finite names gain the stated suffix while infinite names remain exact. Add `width [mm]` and `height [mm]` to `geometry_table`.

Pass `crystal_width_mm=case.get("crystal_width_mm")` and `crystal_height_mm=case.get("crystal_height_mm")` in both calls in `_transport_case` and the call in `_brem_for_case`. Document these optional case keys in `run_case`. Do not modify checkpoint identity beyond the now-distinct case names.

- [ ] **Step 4: Run the tests**

Run: `uv run python scripts/dev.py test tests/test_sweep.py tests/test_run.py -k "footprint or transport_case or brem_for_case" -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/sweep.py src/cxr_mc/montecarlo/runner.py tests/test_sweep.py tests/test_run.py
git commit -m "feat: expose finite crystal footprints in sweeps"
```

### Task 6: Record the physics scope and run final verification

**Files:**
- Modify: `docs/physics-validation-ledger.md`
- Modify: `docs/validation/finite-beam-size.md`
- Create later, independently: `docs/validation/finite-transverse-crystal.md`

**Interfaces:**
- Produces ledger id `finite-transverse-crystal` at `unverified`; it cannot be advanced by the implementation author.

- [ ] **Step 1: Add the validation records**

Add an `unverified` ledger row anchored to `geometry.py::first_prism_exit`, `transport.py::simulate_trajectories`, and `spectrum.py::{mc_spectrum,mc_brem_spectrum}`, with source “rectangular-prism ray intersection + Beer--Lambert” and the exact new test anchors. Amend the finite-beam-size row and write-up: its “zero spectrum effect” claim applies only when the footprint is omitted; with a finite footprint sampled `x,y` controls missed incidence and lateral escape.

- [ ] **Step 2: Run focused feature coverage**

Run: `uv run python scripts/dev.py test tests/test_finite_transverse_geometry.py tests/test_montecarlo.py tests/test_spectrum_escape_helpers.py tests/test_chunk_invariance.py tests/test_multilayer.py tests/test_sweep.py tests/test_run.py tests/test_montecarlo_exports.py -v`

Expected: PASS.

- [ ] **Step 3: Request an independent re-derivation**

Give a fresh context only the new ledger row, public signatures, and design spec. Before it reads implementation bodies, it must derive: ray-box first exit; capped Beer--Lambert path; mm/Å conversion; all-`None` recovery; lateral `n_z=0` layer residence; z-layer crossing; tie convention; and per-incident normalization under spillover. It writes `docs/validation/finite-transverse-crystal.md` under the contract in `docs/validation/README.md`. The implementation author must not write the document or change the status.

- [ ] **Step 4: Run final repository verification**

Run: `uv run python scripts/dev.py verify`

Expected: PASS with skill checks, Ruff, Pyright, and all tests green.

- [ ] **Step 5: Commit implementation-side validation docs**

```bash
git add docs/physics-validation-ledger.md docs/validation/finite-beam-size.md
git commit -m "docs: ledger finite transverse crystal model"
```

Commit the independent document only after it exists:

```bash
git add docs/validation/finite-transverse-crystal.md
git commit -m "docs: validate finite transverse crystal geometry"
```

## Plan self-review

- Spec coverage: Tasks 1–2 cover the rectangular geometry, units, misses, lateral transport, diagnostics, and legacy limit. Tasks 3–4 cover six-face coherent/brem attenuation and layered stacks. Task 5 provides optional sweep/driver access. Task 6 supplies the required validation lifecycle.
- Placeholder scan: no unassigned implementation or generic test step remains. The independent write-up is intentionally assigned to a separate fresh context, as required by the repository’s physics-validation contract.
- Interface consistency: public case keys are `crystal_width_mm`/`crystal_height_mm`; segment keys are `crystal_width_ang`/`crystal_height_ang`; shared geometry uses `width_ang`/`height_ang`; capped attenuation uses `exit_distance_ang`.
