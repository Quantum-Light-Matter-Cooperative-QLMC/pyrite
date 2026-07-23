# Blazed Groove Escape-Path Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Autogenerate a blazed sawtooth groove profile on the crystal beam-entrance face — working (exit) facet automatically perpendicular to the observation direction, relief (entry) facet perpendicular to the e-beam — and fold its exact closed-form geometry into electron entry and photon self-absorption in the MC.

**Architecture:** A new pure-geometry module `montecarlo/groove.py` holds a frozen `GrooveSpec` plus O(1) closed-form functions for electron entry points and photon escape distances (no ray marching). `simulate_trajectories` gains an optional `groove=` that offsets electron start positions onto the relief facets; `mc_spectrum` gains an optional `groove=` that replaces the flat-face escape distance `z/(-n_z)` with the sawtooth closed form. `Sweep`/`build_cases`/`runner` plumb a single user knob: `groove_spacing_ang`.

**Tech Stack:** numpy (ufunc-only math so cupy arrays pass through), pytest, existing cxr-mc MC pipeline.

## Global Constraints

- Restricted geometry, validated hard at case-build time AND in `GrooveSpec` construction: `theta_obs_rad == pi/2` (exact within `1e-9`), `tilt_azim_deg == 180.0`, `0 < tilt_deg < 90`. Anything else raises `ValueError`.
- v1 exclusions (raise `ValueError` when combined with grooves): `layers`/`abs_layers`/substrate stacks, finite footprint (`crystal_width_mm`/`crystal_height_mm`), `mc_spectrum_solid_angle` with `n_side > 1`.
- `groove=None` must be **bit-for-bit** identical to today's outputs everywhere (regression-tested).
- All groove math uses numpy ufuncs only (`np.mod`, `np.floor`, `np.maximum`, arithmetic) so cupy arrays dispatch through `__array_ufunc__` — no `np.asarray`, no `.item()`, no python branching on array values inside hot paths.
- New physics functions need derivation docstrings (source eq, assumptions, limiting case), a `Validation: <id>` marker, and a ledger row (repo rule, `AGENTS.md`). Implementation author must NOT advance ledger status; fresh-context verifier + human sign-off only.
- Every command runs as `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test <path> -k <name>`.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Geometry (single source of truth for every task)

Sample frame: entrance face nominal plane `z = 0`, depth `+z` into the slab, back face `z = thickness`. At `theta_obs = 90°`, `tilt_azim = 180°`, `tilt_polar = tp` (from `tilted_geometry`, verified numerically 2026-07-23):

```
beam_dir b = ( sin tp, 0,  cos tp )     # electrons, into slab
n_hat    n = ( cos tp, 0, -sin tp )     # photons to detector, out the ENTRANCE face
b · n = 0   (theta_obs = 90 exactly)
```

Sawtooth grooves run along y (profile invariant in y), period Λ (user input `groove_spacing_ang`), tooth apexes at `x = kΛ, z = 0`, valley floor depth `h`:

- **Working facet A** (photon exit): plane family `n·r = kΛ cos tp`, physical band `z ∈ [0, h]`. Facet is ⊥ `n` (photons cross at normal incidence) and ∥ `b` (electrons never cross it).
- **Relief facet B** (electron entry): plane family `b·r = kΛ sin tp`, physical band `z ∈ [0, h]`. Facet is ⊥ `b` (electrons enter at normal incidence) and ∥ `n` (exit photons never cross it → zero shadowing by construction).
- Closing the unit cell fixes the depth: **`h = Λ sin(tp) cos(tp) = Λ sin(2·tp)/2`**.

**Photon escape distance** from emission point `(x, z)` along `n` (derivation: successive A-plane crossings are spaced `Λ cos tp` in path and exactly `h` in depth, so exactly one crossing lands in the physical band `[0, h)`; crossings deeper than `h` are interior points below the valley floor):

```
c  = Λ cos(tp)                    # A-plane spacing along n
s1 = c - mod(x cos(tp) - z sin(tp), c)     # first A-plane crossing, s1 ∈ (0, c]
z1 = z - s1 sin(tp)                        # depth at that crossing
m  = max(floor(z1 / h), 0)                 # whole periods still inside material
L_esc = s1 + m·c
```

Limiting cases: `h→0` (fixed z): `L_esc → z/sin tp` = flat-face path; point just inside its own facet A: `L_esc → 0`.

**Electron entry** for a lab ray whose flat-face `z=0` intersection is `(x0, y0, 0)` (all such points are vacuum — at `z=0` material exists only on the measure-zero apex lines — so continue along `b` to the first B-plane crossing; the same one-crossing-in-band argument applies with spacing `Λ sin tp` and depth step `h`):

```
s1 = mod(-x0 sin(tp), Λ sin(tp))          # s1 ∈ [0, Λ sin tp)
entry = (x0 + s1 sin(tp), y0, s1 cos(tp))  # on facet B, z_entry ∈ [0, h)
```

**Phase sampling:** groove effects depend on `x mod Λ`. With `beam_fwhm_mm=None` every electron enters at the origin (one phase) — unphysical for a real beam ≫ Λ wide. When `groove` is given and `beam_fwhm_mm is None`, draw `x0 ~ Uniform[0, Λ)` per electron from an independent child RNG stream (same spawn pattern as `beam_rng`).

v1 electron-transport approximation (document in docstrings + ledger): only the **start point** honors the grooves; in-flight boundary tests keep the flat faces (`z<0` / `z>thickness`). Error is O(h / electron range) in backscatter bookkeeping; the physics payload (photon self-absorption) is exact.

## File Structure

- Create: `src/cxr_mc/montecarlo/groove.py` — `GrooveSpec`, `blazed_groove_spec()`, `escape_distance_ang()`, `entry_points()` (pure geometry, no MC imports).
- Create: `tests/test_groove.py` — closed-form vs brute-force ray-march, limits, invariants.
- Modify: `src/cxr_mc/montecarlo/transport.py` (`simulate_trajectories`, start-position block near line 357) — `groove=` entry offsets + uniform phase fallback.
- Modify: `src/cxr_mc/montecarlo/spectrum.py` (`mc_spectrum` escape branch near line 360; `mc_spectrum_solid_angle` guard) — `groove=` escape distance.
- Modify: `src/cxr_mc/montecarlo/runner.py` (`_transport_case` near line 290, spectrum call near line 435) — build spec from case, thread through.
- Modify: `src/cxr_mc/sweep.py` (`Sweep` dataclass ~line 209, validation ~line 326, `build_cases` ~line 460) — `groove_spacing_ang` knob + validation.
- Modify: `src/cxr_mc/montecarlo/__init__.py` — export `GrooveSpec`, `blazed_groove_spec`.
- Modify: `docs/physics-validation-ledger.md`, `docs/repo_map.md`.

---

### Task 1: Groove geometry module

**Files:**
- Create: `src/cxr_mc/montecarlo/groove.py`
- Test: `tests/test_groove.py`

**Interfaces:**
- Produces: `GrooveSpec(spacing_ang: float, depth_ang: float, tilt_polar_rad: float)` frozen dataclass; `blazed_groove_spec(spacing_ang, theta_obs_rad, tilt_polar_rad, tilt_azim_rad) -> GrooveSpec` (validates restricted geometry); `escape_distance_ang(x, z, spec) -> array` (ufunc-only); `entry_points(x0, spec) -> (x_entry, z_entry)` (ufunc-only). Later tasks import these exact names.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_groove.py
import numpy as np
import pytest

from cxr_mc.montecarlo.groove import (
    GrooveSpec,
    blazed_groove_spec,
    entry_points,
    escape_distance_ang,
)

TP = np.deg2rad(45.0)
SPEC = blazed_groove_spec(
    spacing_ang=2.0e4, theta_obs_rad=np.pi / 2,
    tilt_polar_rad=TP, tilt_azim_rad=np.pi,
)


def _z_surf(x, spec):
    """Brute-force sawtooth profile height (depth of the surface) at x."""
    tp = spec.tilt_polar_rad
    lam, h = spec.spacing_ang, spec.depth_ang
    u = np.mod(x, lam)
    x_valley = h * np.tan(tp)
    return np.where(u <= x_valley, u / np.tan(tp), (lam - u) * np.tan(tp))


def _march_escape(x, z, spec, ds=0.05):
    """Reference ray march along n_hat until the point leaves the material."""
    tp = spec.tilt_polar_rad
    n = np.array([np.cos(tp), -np.sin(tp)])
    p = np.array([x, z], dtype=float)
    s = 0.0
    while p[1] > 0 and p[1] >= _z_surf(p[0], spec) - 1e-12:
        p += ds * n
        s += ds
    return s


def test_depth_closes_unit_cell():
    assert SPEC.depth_ang == pytest.approx(
        SPEC.spacing_ang * np.sin(TP) * np.cos(TP)
    )


def test_escape_matches_ray_march():
    rng = np.random.default_rng(7)
    lam, h = SPEC.spacing_ang, SPEC.depth_ang
    x = rng.uniform(0.0, 3 * lam, 200)
    z = rng.uniform(0.0, 4 * h, 200)
    inside = z >= _z_surf(x, SPEC) + 1e-6
    x, z = x[inside], z[inside]
    L = escape_distance_ang(x, z, SPEC)
    ref = np.array([_march_escape(xi, zi, SPEC) for xi, zi in zip(x, z)])
    assert np.all(np.abs(L - ref) < 0.2)  # ray-march step tolerance


def test_escape_flat_limit_small_depth():
    tiny = blazed_groove_spec(
        spacing_ang=1.0, theta_obs_rad=np.pi / 2,
        tilt_polar_rad=TP, tilt_azim_rad=np.pi,
    )
    z = np.array([5.0e4])
    L = escape_distance_ang(np.array([1234.5]), z, tiny)
    assert L[0] == pytest.approx(z[0] / np.sin(TP), rel=1e-3)


def test_escape_never_exceeds_flat_path_and_nonnegative():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 5 * SPEC.spacing_ang, 500)
    z = rng.uniform(SPEC.depth_ang, 10 * SPEC.depth_ang, 500)
    L = escape_distance_ang(x, z, SPEC)
    assert np.all(L >= 0.0)
    assert np.all(L <= z / np.sin(TP) + 1e-9)


def test_entry_points_land_on_relief_facet():
    rng = np.random.default_rng(11)
    x0 = rng.uniform(-3 * SPEC.spacing_ang, 3 * SPEC.spacing_ang, 300)
    xe, ze = entry_points(x0, SPEC)
    assert np.all((ze >= 0.0) & (ze < SPEC.depth_ang))
    # every entry point lies ON the surface profile
    assert np.allclose(ze, _z_surf(xe, SPEC), atol=1e-6)
    # displacement is along the beam direction
    assert np.allclose((xe - x0) / np.sin(TP) * np.cos(TP), ze, atol=1e-6)


def test_blazed_groove_spec_validates_geometry():
    with pytest.raises(ValueError):
        blazed_groove_spec(1e4, np.deg2rad(119.0), TP, np.pi)  # theta_obs != 90
    with pytest.raises(ValueError):
        blazed_groove_spec(1e4, np.pi / 2, TP, 0.0)  # azim != 180
    with pytest.raises(ValueError):
        blazed_groove_spec(1e4, np.pi / 2, 0.0, np.pi)  # tp == 0
    with pytest.raises(ValueError):
        blazed_groove_spec(-1.0, np.pi / 2, TP, np.pi)  # spacing <= 0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py`
Expected: FAIL / collection error `ModuleNotFoundError: No module named 'cxr_mc.montecarlo.groove'`

- [ ] **Step 3: Implement `groove.py`**

```python
# src/cxr_mc/montecarlo/groove.py
"""
montecarlo.groove

Blazed sawtooth groove profile on the crystal beam-entrance face, for
escape-path engineering of the coherent line yield. Restricted geometry:
theta_obs = 90 deg, tilt_azim = 180 deg, 0 < tilt_polar < 90 deg. In the
sample frame (entrance face z = 0, depth +z) this puts

    beam  b = ( sin tp, 0,  cos tp )
    n_hat n = ( cos tp, 0, -sin tp ),     b . n = 0.

Grooves run along y with period ``spacing_ang`` (apexes at x = k*spacing,
z = 0). The WORKING facet is the plane family ``n . r = k*spacing*cos(tp)``
(perpendicular to the observation direction, parallel to the beam); the
RELIEF facet is ``b . r = k*spacing*sin(tp)`` (perpendicular to the beam,
parallel to the exit rays -- zero shadowing by construction). Closing the
unit cell fixes the groove depth

    h = spacing * sin(tp) * cos(tp).

Source: elementary ray-plane intersection on a periodic sawtooth (no
literature equation); facet-choice rationale in
docs/superpowers/plans/2026-07-23-blazed-groove-geometry.md.
Assumptions: profile invariant along y; laterally infinite slab; photons
travel straight along n_hat (incoherent Beer-Lambert transport -- no wave
optics, consistent with mc_spectrum).
"""

from dataclasses import dataclass

import numpy as np

_THETA_TOL = 1e-9


@dataclass(frozen=True)
class GrooveSpec:
    """Blazed sawtooth groove profile (validated; build via blazed_groove_spec)."""

    spacing_ang: float
    depth_ang: float
    tilt_polar_rad: float


def blazed_groove_spec(spacing_ang, theta_obs_rad, tilt_polar_rad, tilt_azim_rad):
    """
    Construct the blazed GrooveSpec for the restricted geometry, deriving the
    depth h = spacing*sin(tp)*cos(tp) that closes the sawtooth unit cell
    (working facet perpendicular to n_hat, relief facet perpendicular to the
    beam -- see the module docstring).

    Limiting case: tilt_polar_rad -> 0 or 90 deg degenerates the sawtooth
    (h -> 0) AND breaks the entrance/exit face assignment, so both are
    rejected rather than silently producing a flat profile.

    Validation: blazed-groove-geometry
    """
    if not np.isclose(theta_obs_rad, np.pi / 2, atol=_THETA_TOL):
        raise ValueError(
            "blazed grooves require theta_obs = 90 deg exactly (beam "
            "perpendicular to observation; relief facet parallel to exit rays)"
        )
    if not np.isclose(tilt_azim_rad, np.pi, atol=_THETA_TOL):
        raise ValueError("blazed grooves require tilt_azim = 180 deg")
    tp = float(tilt_polar_rad)
    if not 0.0 < tp < np.pi / 2:
        raise ValueError("blazed grooves require 0 < tilt_polar < 90 deg")
    lam = float(spacing_ang)
    if lam <= 0.0:
        raise ValueError("groove spacing_ang must be positive")
    return GrooveSpec(
        spacing_ang=lam,
        depth_ang=lam * np.sin(tp) * np.cos(tp),
        tilt_polar_rad=tp,
    )


def escape_distance_ang(x, z, spec):
    """
    Straight-line path length [Ang] inside the grooved material from emission
    point (x, z) to the surface, along the observation direction
    n = (cos tp, 0, -sin tp).

    Derivation: exit rays are parallel to the relief facets, so a ray only
    ever crosses working-facet planes ``n . r = k*spacing*cos(tp)``.
    Successive plane crossings are spaced ``c = spacing*cos(tp)`` in path and
    exactly ``h`` in depth, so exactly one crossing depth lands in the
    physical facet band [0, h) -- crossings deeper than h are interior points
    below the valley floor. Hence, with d = n . (x, 0, z):

        s1 = c - mod(d, c);  z1 = z - s1*sin(tp)
        L  = s1 + max(floor(z1/h), 0) * c

    numpy ufuncs only, so cupy arrays dispatch through __array_ufunc__.
    Limiting cases: h -> 0 at fixed z gives L -> z/sin(tp), the flat
    entrance-face path of mc_spectrum's ``z_mid / (-n_hat[2])`` branch; a
    point just inside its own working facet gives L -> 0.

    Validation: blazed-groove-geometry
    """
    tp = spec.tilt_polar_rad
    st, ct = np.sin(tp), np.cos(tp)
    c = spec.spacing_ang * ct
    s1 = c - np.mod(x * ct - z * st, c)
    z1 = z - s1 * st
    m = np.maximum(np.floor(z1 / spec.depth_ang), 0.0)
    return s1 + m * c


def entry_points(x0, spec):
    """
    Electron entry point on the relief facet for a collimated ray whose
    flat-face z=0 intersection is (x0, y0, 0).

    Derivation: at z = 0 the material is only the measure-zero apex lines, so
    every ray continues along b = (sin tp, 0, cos tp) into the groove cut and
    enters through the first relief-facet plane ``b . r = k*spacing*sin(tp)``
    (rays are parallel to the working facets and never cross them). The same
    one-crossing-in-band argument as escape_distance_ang applies with plane
    spacing ``spacing*sin(tp)`` and depth step h:

        s1 = mod(-x0*sin(tp), spacing*sin(tp))
        entry = (x0 + s1*sin(tp), y0, s1*cos(tp)),  z_entry in [0, h)

    Limiting case: spacing -> 0 (h -> 0) gives z_entry -> 0, the flat face.

    Validation: blazed-groove-geometry

    Returns (x_entry, z_entry), shaped like x0 (y is untouched by the
    profile and handled by the caller).
    """
    tp = spec.tilt_polar_rad
    st, ct = np.sin(tp), np.cos(tp)
    s1 = np.mod(-x0 * st, spec.spacing_ang * st)
    return x0 + s1 * st, s1 * ct
```

- [ ] **Step 4: Run tests, verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py`
Expected: all 6 PASS

- [ ] **Step 5: Export from package**

In `src/cxr_mc/montecarlo/__init__.py`, add to the existing import/export block (match the file's current style):

```python
from .groove import GrooveSpec, blazed_groove_spec
```

and append `"GrooveSpec", "blazed_groove_spec"` to `__all__` if the module defines one.

- [ ] **Step 6: Lint + commit**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py lint`
Expected: clean

```bash
git add src/cxr_mc/montecarlo/groove.py src/cxr_mc/montecarlo/__init__.py tests/test_groove.py
git commit -m "feat(groove): blazed sawtooth entrance-face geometry, closed-form entry/escape

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Electron entry through the relief facets

**Files:**
- Modify: `src/cxr_mc/montecarlo/transport.py` (signature ~line 192, start-position block ~lines 356-378)
- Test: `tests/test_groove.py` (append)

**Interfaces:**
- Consumes: `entry_points(x0, spec)`, `GrooveSpec` from Task 1.
- Produces: `simulate_trajectories(..., groove: GrooveSpec | None = None)`. `groove=None` bit-for-bit unchanged. Later tasks pass `groove=` from the runner.

- [ ] **Step 1: Write failing tests (append to `tests/test_groove.py`)**

```python
from cxr_mc.montecarlo.transport import simulate_trajectories

_SIM_KW = dict(E0_keV=60.0, Ne=200, thickness_ang=2.0e5, element="C",
               n_atoms_per_ang3=0.1136, seed=42, elastic_model="sr")


def _tilt_kw():
    from cxr_mc.montecarlo.geometry import tilted_geometry
    beam, _ = tilted_geometry(np.pi / 2, TP, np.pi)
    return dict(beam_dir=beam, tilt_polar_rad=TP, tilt_azim_rad=np.pi)


def test_groove_none_is_bitwise_identical():
    a = simulate_trajectories(**_SIM_KW, **_tilt_kw())
    b = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=None)
    for k in ("E_keV", "L_ang", "t_ang", "elec_id"):
        np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_array_equal(a["r_mid"], b["r_mid"])


def test_groove_entries_start_on_relief_facet():
    segs = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=SPEC)
    # first segment of each electron starts at its entry point; entry depths
    # span [0, h) and multiple phases are sampled (uniform-phase fallback)
    first = np.searchsorted(segs["elec_id"], np.unique(segs["elec_id"]))
    r0 = segs["r_mid"][first]  # midpoints of first segments sit at z >= z_entry
    assert np.all(r0[:, 2] >= 0.0)
    # phases genuinely vary across electrons
    assert np.unique(np.round(r0[:, 0], 3)).size > 10


def test_groove_entry_depth_reproducible_with_seed():
    a = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=SPEC)
    b = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=SPEC)
    np.testing.assert_array_equal(a["r_mid"], b["r_mid"])
```

- [ ] **Step 2: Run, verify fail**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py -k groove_none or groove_entr`
Expected: FAIL `TypeError: simulate_trajectories() got an unexpected keyword argument 'groove'`

- [ ] **Step 3: Implement**

Add `groove=None` to the `simulate_trajectories` signature (after `tilt_azim_rad=0.0`), document it in the docstring (derivation summary from the plan header, the v1 flat-boundary approximation, `Validation: blazed-groove-geometry` marker), and replace the start-position block (currently `pos = np.zeros((Ne, 3))` ... `pos[:, :2] = project_beam_entry(...)`) with:

```python
    rng = np.random.default_rng(seed)
    pos = np.zeros((Ne, 3))
    if beam_fwhm_mm:
        # ... existing Gaussian-spot block unchanged ...
        pos[:, :2] = project_beam_entry(offsets, tilt_polar_rad, tilt_azim_rad)
    elif groove is not None:
        # Groove effects depend on x mod spacing; a point source samples one
        # phase only. Draw the lateral phase uniformly over one period from an
        # independent child stream (same spawn pattern as beam_rng, so the
        # main free-path/scattering draws are untouched).
        phase_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
        pos[:, 0] = phase_rng.uniform(0.0, groove.spacing_ang, size=Ne)
    if groove is not None:
        # Slide each ray along the beam to its relief-facet entry point.
        # v1 approximation: only the START point honors the grooves; in-flight
        # boundary tests keep the flat faces (error O(h / electron range) in
        # backscatter bookkeeping). Validation: blazed-groove-geometry
        x_e, z_e = entry_points(pos[:, 0], groove)
        pos[:, 0] = x_e
        pos[:, 2] = z_e
```

with `from .groove import entry_points` added to the imports at the top of `transport.py`.

Note the spawn index: `beam_rng` uses `spawn(2)[1]`; use `spawn(3)[2]` so the two streams stay independent even when both features are on.

- [ ] **Step 4: Run tests, verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py`
Expected: all PASS

- [ ] **Step 5: Run the existing transport regression suite**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_montecarlo.py`
Expected: PASS (groove=None default touches nothing)

- [ ] **Step 6: Commit**

```bash
git add src/cxr_mc/montecarlo/transport.py tests/test_groove.py
git commit -m "feat(transport): electron entry through blazed groove relief facets

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Photon escape through the working facets

**Files:**
- Modify: `src/cxr_mc/montecarlo/spectrum.py` (`mc_spectrum` signature ~line 97, escape block ~lines 350-368; `mc_spectrum_solid_angle` ~line 445)
- Test: `tests/test_groove.py` (append)

**Interfaces:**
- Consumes: `escape_distance_ang(x, z, spec)` from Task 1.
- Produces: `mc_spectrum(..., groove: GrooveSpec | None = None)`; `mc_spectrum_solid_angle` accepts and forwards `groove` only when `n_side == 1`, else raises `ValueError`.

- [ ] **Step 1: Write failing tests (append)**

```python
from cxr_mc.montecarlo.spectrum import mc_spectrum


def _hopg_spectrum(groove=None, thickness_ang=2.0e5):
    from cxr_mc.montecarlo.geometry import tilted_geometry
    beam, n_hat = tilted_geometry(np.pi / 2, TP, np.pi)
    segs = simulate_trajectories(
        E0_keV=60.0, Ne=400, thickness_ang=thickness_ang, element="C",
        n_atoms_per_ang3=0.1136, seed=42, elastic_model="sr",
        beam_dir=beam, tilt_polar_rad=TP, tilt_azim_rad=np.pi,
        groove=groove,
    )
    E_grid = np.linspace(500.0, 3000.0, 400)
    return mc_spectrum(
        segs, E_grid, crystal="hopg", hkl_list=[(0, 0, 2)],
        n_hat=n_hat, B_ang2=0.0, composition=[("C", 0.1136)],
        groove=groove,
    )


def test_spectrum_groove_none_bitwise():
    np.testing.assert_array_equal(_hopg_spectrum(), _hopg_spectrum(groove=None))


def test_spectrum_groove_boosts_line_yield():
    flat = _hopg_spectrum()
    grooved = _hopg_spectrum(groove=SPEC)
    # escape paths only ever shorten -> integrated line yield must not drop,
    # and for a slab many absorption lengths thick it must rise measurably
    assert grooved.sum() > flat.sum() * 1.05


def test_spectrum_groove_rejects_layers():
    from cxr_mc.montecarlo.geometry import tilted_geometry
    _, n_hat = tilted_geometry(np.pi / 2, TP, np.pi)
    segs = simulate_trajectories(
        E0_keV=60.0, Ne=10, thickness_ang=1.0e4, element="C",
        n_atoms_per_ang3=0.1136, seed=1, elastic_model="sr",
    )
    with pytest.raises(ValueError):
        mc_spectrum(
            segs, np.linspace(500.0, 3000.0, 50), crystal="hopg",
            hkl_list=[(0, 0, 2)], n_hat=n_hat, B_ang2=0.0,
            composition=[("C", 0.1136)], groove=SPEC,
            layers=[(0.0, 1.0e4, [("C", 0.1136)])],
        )
```

- [ ] **Step 2: Run, verify fail**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py -k spectrum`
Expected: FAIL `TypeError: mc_spectrum() got an unexpected keyword argument 'groove'`

- [ ] **Step 3: Implement**

Add `groove=None` to `mc_spectrum`'s signature. Immediately after argument normalization (before the segment loop), guard:

```python
    if groove is not None:
        if layers is not None:
            raise ValueError("groove escape is v1 single-slab only (no layers)")
        if (
            segments.get("crystal_width_ang") is not None
            and segments.get("crystal_height_ang") is not None
        ):
            raise ValueError("groove escape requires a laterally infinite slab")
        if n_hat[2] >= 0.0:
            raise ValueError(
                "groove escape requires exit through the entrance face "
                "(n_hat z-component < 0); check theta_obs/tilt geometry"
            )
```

In the escape block (infinite-footprint, `layers is None` branch), replace:

```python
                if n_hat[2] < 0:
                    L_esc = z_mid / (-n_hat[2])  # out the entrance face
                else:
                    L_esc = (thickness - z_mid) / n_hat[2]  # out the back face
```

with:

```python
                if n_hat[2] < 0:
                    if groove is not None:
                        # Blazed sawtooth entrance face: closed-form path to
                        # the working facet (grooves shorten, never lengthen,
                        # the flat-face path). Validation: blazed-groove-geometry
                        L_esc = escape_distance_ang(seg_r[idx, 0], z_mid, groove)
                    else:
                        L_esc = z_mid / (-n_hat[2])  # out the entrance face
                else:
                    L_esc = (thickness - z_mid) / n_hat[2]  # out the back face
```

(`seg_r[idx, 0]` is the same indexing already used for `z_mid = seg_r[idx, 2]`; `escape_distance_ang` is ufunc-only so the cupy path is untouched.) Import at top: `from .groove import escape_distance_ang`.

In `mc_spectrum_solid_angle`, accept `groove=None`, raise `ValueError` if `groove is not None and n_side > 1` ("detector tiles break the relief-facet parallelism; v1 supports n_side=1 only"), and forward `groove=groove` in the inner `mc_spectrum` call.

Extend the `mc_spectrum` docstring with the derivation summary + assumptions (straight-ray incoherent Beer-Lambert, restricted geometry, v1 exclusions) and the `Validation: blazed-groove-geometry` marker.

- [ ] **Step 4: Run tests, verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py`
Expected: all PASS

- [ ] **Step 5: Quantitative gain cross-check (analytic vs MC)**

Append and run one more test — deep uniform emitters see mean transmission boost `(sinγ·L/h)·(e^{h/(L sinγ)} − 1)`:

```python
def test_escape_gain_matches_analytic_mean():
    """Uniform-phase emitters at fixed deep z: mean exp(-L/L_abs) boost equals
    the closed-form sawtooth average (path savings uniform in [0, h)/sin tp)."""
    rng = np.random.default_rng(5)
    L_abs = 6.0e4  # [Ang] ~ HOPG (002) scale
    z = np.full(20000, 8.0e4)
    x = rng.uniform(0.0, 50 * SPEC.spacing_ang, z.size)
    L = escape_distance_ang(x, z, SPEC)
    t_grooved = np.exp(-L / L_abs).mean()
    t_flat = np.exp(-(z[0] / np.sin(TP)) / L_abs)
    h, sg = SPEC.depth_ang, np.sin(TP)
    analytic = (sg * L_abs / h) * np.expm1(h / (sg * L_abs))
    assert t_grooved / t_flat == pytest.approx(analytic, rel=1e-2)
```

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_groove.py -k analytic`
Expected: PASS

- [ ] **Step 6: Run spectrum regression suite + commit**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_montecarlo.py tests/test_mosaic.py`
Expected: PASS

```bash
git add src/cxr_mc/montecarlo/spectrum.py tests/test_groove.py
git commit -m "feat(spectrum): blazed-groove escape attenuation in mc_spectrum

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Case plumbing — `groove_spacing_ang` knob

**Files:**
- Modify: `src/cxr_mc/sweep.py` (`Sweep` dataclass ~line 209, validation near the `tilt_azim_deg == 90` check ~line 326, case-dict construction ~line 460-490)
- Modify: `src/cxr_mc/montecarlo/runner.py` (`_transport_case` ~line 290-320, spectrum call ~line 435; also the second `simulate_trajectories` (brem segs) and, if present, `_brem_for_case`'s escape usage — brem self-absorption stays flat-face in v1, note it)
- Test: `tests/test_sweep.py` (append; create the test function in whatever file already tests `build_cases` — check with `rg -n "build_cases" tests/`)

**Interfaces:**
- Consumes: `blazed_groove_spec` (Task 1); `simulate_trajectories(groove=)` (Task 2); `mc_spectrum(groove=)` (Task 3).
- Produces: `Sweep(groove_spacing_ang: float | None = None)`; case-dict key `"groove_spacing_ang"` (float, only present when set); runner builds the spec once per case via `blazed_groove_spec(case["groove_spacing_ang"], case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad)`.

- [ ] **Step 1: Write failing tests**

```python
def test_build_cases_carries_groove_spacing():
    sweep = Sweep(
        tilt_deg=45.0, tilt_azim_deg=180.0,
        groove_spacing_ang=2.0e4, thickness_ang=2.0e5, energy_keV=100.0,
    )
    cases = build_cases("hopg", sweep)  # match the file's existing call style
    assert all(c["groove_spacing_ang"] == 2.0e4 for c in cases)


def test_build_cases_groove_requires_azim_180():
    with pytest.raises(ValueError):
        build_cases("hopg", Sweep(
            tilt_deg=45.0, tilt_azim_deg=0.0, groove_spacing_ang=2.0e4,
        ))


def test_build_cases_groove_rejects_substrate():
    with pytest.raises(ValueError):
        build_cases("hopg", Sweep(
            tilt_deg=45.0, tilt_azim_deg=180.0, groove_spacing_ang=2.0e4,
            substrate="si",
        ))
```

Adapt the `build_cases(...)` invocation to the real signature in the existing tests (read the neighboring tests first — the material argument/spec form may differ).

- [ ] **Step 2: Run, verify fail**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_sweep.py -k groove`
Expected: FAIL `TypeError: Sweep.__init__() got an unexpected keyword argument 'groove_spacing_ang'`

- [ ] **Step 3: Implement sweep side**

In the `Sweep` dataclass add (near the other geometry scalars):

```python
    groove_spacing_ang: float | None = None
    # Blazed sawtooth grooves on the beam-entrance face (docs/superpowers/
    # plans/2026-07-23-blazed-groove-geometry.md). Scalar, not sweepable in
    # v1. Requires tilt_azim_deg == 180, 0 < tilt_deg < 90, theta_obs = 90,
    # no substrate/stack, no finite footprint.
```

In the validation section (same place as the `tilt_azim_deg == 90` ban), add:

```python
        if self.groove_spacing_ang is not None:
            if self.groove_spacing_ang <= 0:
                raise ValueError("groove_spacing_ang must be positive")
            azims = np.atleast_1d(np.asarray(self.tilt_azim_deg, dtype=float))
            if not np.allclose(azims, 180.0):
                raise ValueError("grooves require tilt_azim_deg == 180 for every case")
            tilts = np.atleast_1d(np.asarray(self.tilt_deg, dtype=float))
            if not np.all((tilts > 0.0) & (tilts < 90.0)):
                raise ValueError("grooves require 0 < tilt_deg < 90 for every case")
            if self.substrate is not None or self.stack:
                raise ValueError("grooves are v1 single-slab only (no substrate/stack)")
            if self.crystal_width_mm is not None or self.crystal_height_mm is not None:
                raise ValueError("grooves require a laterally infinite slab")
```

(match the file's actual validation idiom — it may live in `__post_init__` or in `build_cases`; put it wherever the azim-90 ban lives, and mirror the attribute names used there.) In the case-dict construction, alongside `tilt_azim_deg=float(azim)`, add:

```python
                    **(
                        {"groove_spacing_ang": float(sweep.groove_spacing_ang)}
                        if sweep.groove_spacing_ang is not None
                        else {}
                    ),
```

- [ ] **Step 4: Implement runner side**

In `runner.py` `_transport_case`, after `beam, n_hat = tilted_geometry(...)`:

```python
    groove = None
    if case.get("groove_spacing_ang") is not None:
        groove = blazed_groove_spec(
            case["groove_spacing_ang"], case["theta_obs_rad"],
            tilt_polar_rad, tilt_azim_rad,
        )
```

(`from .groove import blazed_groove_spec` at top). Pass `groove=groove` to BOTH `simulate_trajectories` calls (line segs and brem segs — the electrons enter through the same surface either way). Thread `groove` into the transport-products dict (`tp["groove"] = groove`) and pass `groove=tp.get("groove")` in both `mc_spectrum` call sites (the `radiators is None` branch; the per-layer branch is unreachable because grooves reject stacks — add `assert case.get("groove_spacing_ang") is None` there or just leave the guard in `mc_spectrum` to catch it). Bremsstrahlung self-absorption keeps the flat-face path in v1 — add a one-line comment where the brem spectrum is assembled: `# grooves: brem escape stays flat-face in v1 (line yield is the target; brem bias < groove depth / L_abs).`

- [ ] **Step 5: Run tests + full affected suite**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_sweep.py tests/test_groove.py tests/test_montecarlo.py`
Expected: PASS

- [ ] **Step 6: End-to-end smoke — one grooved case through the runner**

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python - <<'EOF'
import numpy as np
from cxr_mc.sweep import Sweep, build_cases
# adapt to the runner's real one-case entry point (rg -n "def run_case|def _run" src/cxr_mc/montecarlo/runner.py)
sweep = Sweep(tilt_deg=45.0, tilt_azim_deg=180.0, groove_spacing_ang=2.0e4,
              thickness_ang=2.0e5, energy_keV=100.0)
cases = build_cases("hopg", sweep)
print(len(cases), cases[0].get("groove_spacing_ang"))
EOF
```

Expected: prints case count and `20000.0`; then run the same case through the runner entry point used by `checks/` or an existing runner test, confirm a finite spectrum with no NaN and integrated yield ≥ the `groove_spacing_ang=None` twin.

- [ ] **Step 7: Commit**

```bash
git add src/cxr_mc/sweep.py src/cxr_mc/montecarlo/runner.py tests/test_sweep.py
git commit -m "feat(sweep,runner): groove_spacing_ang knob wires blazed grooves through cases

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Ledger, docs, verification handoff

**Files:**
- Modify: `docs/physics-validation-ledger.md` (geometry/transport section)
- Modify: `docs/repo_map.md` (montecarlo section pointer)

**Interfaces:**
- Consumes: everything above, merged.
- Produces: ledger row `blazed-groove-geometry` (status `unverified`); repo-map pointer.

- [ ] **Step 1: Add the ledger row**

In the geometry/transport table of `docs/physics-validation-ledger.md` (same table as `grazing-beam-projection`), append:

```markdown
| `blazed-groove-geometry` | blazed sawtooth entrance-face profile (period Λ, depth h = Λ sin(tp)cos(tp); working facet ⊥ n̂, relief facet ⊥ beam) with closed-form electron entry points and photon escape distances, restricted to θ_obs = 90°, tilt_azim = 180°, 0 < tilt_polar < 90° | `montecarlo/groove.py::{blazed_groove_spec, escape_distance_ang, entry_points}`; wired via `montecarlo/transport.py::simulate_trajectories` (`groove=`), `montecarlo/spectrum.py::mc_spectrum` (`groove=`), `montecarlo/runner.py`, `sweep.py::Sweep.groove_spacing_ang` | elementary periodic ray–plane intersection (no literature equation); design note `docs/superpowers/plans/2026-07-23-blazed-groove-geometry.md` | unverified | closed form vs brute-force ray march; h→0 flat-face limit; entry points land on profile; escape ≤ flat path; analytic sawtooth mean-gain identity | `tests/test_groove.py` | v1 approximations to re-verify independently: electron in-flight boundaries stay flat (entry point only honors grooves); brem self-absorption stays flat-face; single slab, laterally infinite, n_side=1 only. Implementation author must not advance this row. |
```

- [ ] **Step 2: Add repo-map pointer**

In `docs/repo_map.md`'s montecarlo section add one line:

```markdown
- `montecarlo/groove.py` — blazed sawtooth entrance-face grooves (escape-path engineering): closed-form entry/escape, `Sweep.groove_spacing_ang` knob.
```

- [ ] **Step 3: Full verify + commit**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify`
Expected: lint, typecheck, tests all green

```bash
git add docs/physics-validation-ledger.md docs/repo_map.md
git commit -m "docs: ledger row + repo-map pointer for blazed-groove-geometry

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Verification handoff**

Per `docs/validation/README.md`: dispatch the `physics-validator` agent (fresh context, must not be the implementing session) against the `blazed-groove-geometry` row. It authors `docs/validation/blazed-groove-geometry.md`; only the human marks `signed-off`.

---

## Deferred (explicitly out of v1 — do not build)

- General `theta_obs ≠ 90°` (relief facet no longer parallel to exit rays → shadowing model needed).
- Grooved electron in-flight boundaries (facet-aware backscatter/exit tests).
- Brem escape through grooves; substrate stacks; finite footprint; solid-angle tiling `n_side > 1`.
- Sweepable groove spacing arrays; materials.toml schema; CLI flag (add when a scan campaign needs them).
