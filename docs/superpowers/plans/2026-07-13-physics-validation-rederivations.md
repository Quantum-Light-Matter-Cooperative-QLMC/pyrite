# Physics Validation Rederivations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently rederive, compare, and anchor the `line-energy-dispersion`, `finite-time-lineshape`, and `absorption-length` physics claims.

**Architecture:** Three implementation-blind agents first produce independent derivation reports in disjoint scratch files. Only after those expressions are durable does an adjudicator compare them with production code, convert them into validation write-ups, add focused regression anchors and derivation markers, and update each ledger row according to the evidence.

**Tech Stack:** Python 3.13+, NumPy, pytest, Markdown validation ledger, cxr_mc scientific kernels.

## Global Constraints

- Only a human may mark a claim `signed-off`.
- A symbolic, dimensional, normalization, sign, or convention mismatch must set the claim to `discrepancy`; production physics must not be silently changed.
- Fresh-context derivation agents must not inspect the owning implementation, implementation-derived tests, another derivation report, or the current ledger row before recording their final implementation-neutral expression.
- Every touched physics function must document its source, assumptions, at least one limiting case, and its exact `Validation: <id>` marker.
- Preserve all unrelated worktree changes and avoid unrelated refactoring.
- Use CPU-focused tests and the canonical `uv run` commands.

---

### Task 1: Independent line-energy-dispersion derivation

**Files:**
- Create: `.superpowers/physics-validation/line-energy-dispersion-independent.md`

**Interfaces:**
- Consumes: phase matching for radiation from a charge moving at constant velocity through a periodic medium; reciprocal-vector convention `exp(i g dot r)`; intended output photon angular frequency from velocity vector `v`, reciprocal vector `g`, and unit observation direction `n_hat`.
- Produces: a final implementation-neutral expression for `omega(v, g, n_hat)` plus sign, units, coordinate, positive-tilt, and limiting-case checks.

- [ ] **Step 1: Create the implementation-blind derivation report**

Write the report with these exact sections:

```markdown
# Independent derivation: line-energy-dispersion

## Source and assumptions
## Phase-matching derivation
## Units and sign conventions
## Positive-tilt convention
## Limiting cases
## Final implementation-neutral expression
## Ambiguities
```

Derive from phase stationarity along `r(t) = v t`; keep `v` in units of `c` and `g`, `omega` in inverse length. Explicitly determine whether the numerator is `v dot g` or `-v dot g` under the stated Fourier convention, require a positive emitted frequency, check the nonrelativistic limit, and analyze equal-and-opposite polar tilts at zero azimuth without reading repository code.

- [ ] **Step 2: Self-review independence and completeness**

Confirm the report names no repository implementation symbol, contains no copied code, and includes the final equation, units, sign condition, coordinate convention, and at least one limiting case.

### Task 2: Independent finite-time-lineshape derivation

**Files:**
- Create: `.superpowers/physics-validation/finite-time-lineshape-independent.md`

**Interfaces:**
- Consumes: constant-amplitude integral `Q(P,T) = integral from -T/2 to T/2 of exp(i 2 P t) dt` and NumPy's normalized `sinc(x) = sin(pi x)/(pi x)` convention.
- Produces: `|Q|^2`, its zero-detuning value, full-width scaling, integrated normalization, and distributional `T -> infinity` limit.

- [ ] **Step 1: Create the implementation-blind derivation report**

Write the report with these exact sections:

```markdown
# Independent derivation: finite-time-lineshape

## Source and assumptions
## Fourier-integral derivation
## Sinc convention and normalization
## Units
## Limiting cases
## Final implementation-neutral expression
## Ambiguities
```

Evaluate the integral exactly, state the factor-of-two convention relating `P` to the physical angular-frequency detuning, prove the zero-detuning value, compute `integral |Q|^2 dP`, and state the correctly normalized delta-function limit. Express the final result both with `sin(x)/x` and NumPy's normalized `sinc` without reading repository code.

- [ ] **Step 2: Self-review independence and completeness**

Confirm the report names no repository implementation symbol, contains no copied code, and explicitly accounts for every factor of two and `pi` in the squared amplitude and delta limit.

### Task 3: Independent Henke-f2 absorption-length derivation

**Files:**
- Create: `.superpowers/physics-validation/absorption-length-independent.md`

**Interfaces:**
- Consumes: complex refractive index `n_refr = 1 - delta + i beta` with `beta = r_e lambda^2 n_atom f2 / (2 pi)` and an intensity Beer--Lambert law.
- Produces: attenuation coefficient and absorption length in Angstrom from element, photon energy in eV, atomic number density in inverse cubic Angstrom, and dimensionless `f2`.

- [ ] **Step 1: Create the implementation-blind derivation report**

Write the report with these exact sections:

```markdown
# Independent derivation: absorption-length

## Source and assumptions
## Field-to-intensity derivation
## Henke-f2 substitution
## Units and positivity
## Limiting cases
## Final implementation-neutral expression
## Ambiguities
```

Propagate a plane wave through the complex index, distinguish field-amplitude attenuation from intensity attenuation, derive `mu` and `L_abs = 1 / mu`, and reduce the result to a single expression in `r_e`, `lambda`, `n_atom`, and `f2`. Check the transparent-medium, zero-density, and high-energy limits without reading repository code.

- [ ] **Step 2: Self-review independence and completeness**

Confirm the report names no repository implementation symbol, contains no copied code, and resolves the sign convention for the imaginary refractive index without changing the positive physical attenuation coefficient.

### Task 4: Adjudicate derivations and write validation records

**Files:**
- Read: `.superpowers/physics-validation/line-energy-dispersion-independent.md`
- Read: `.superpowers/physics-validation/finite-time-lineshape-independent.md`
- Read: `.superpowers/physics-validation/absorption-length-independent.md`
- Create: `docs/validation/line-energy-dispersion.md`
- Create: `docs/validation/finite-time-lineshape.md`
- Create: `docs/validation/absorption-length.md`
- Read: `src/cxr_mc/montecarlo/geometry.py`
- Read: `src/cxr_mc/montecarlo/spectrum.py`
- Read: `src/cxr_mc/materials/crystal.py`
- Read: `checks/anchor_figures.py`

**Interfaces:**
- Consumes: the three durable independent reports and the owning implementations.
- Produces: three audit records containing the independent derivation, an explicit implementation diff, numerical spot checks where useful, and an adjudication of `match` or `discrepancy`.

- [ ] **Step 1: Copy each independent derivation into its durable validation record**

Use this structure for each document:

```markdown
# Validation: <id>

## Independent derivation
## Units, conventions, and limiting cases
## Implementation comparison
## Numerical spot check
## Adjudication
```

The independent section must preserve the agent's expression verbatim apart from Markdown cleanup. The comparison section must name the exact implementation expression and account for every sign, `2`, `pi`, `hbar c`, and unit conversion relevant to that claim.

- [ ] **Step 2: Run implementation-neutral numerical spot checks**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import numpy as np

# Finite-time normalization in the P convention:
for T in (10.0, 100.0):
    P = np.linspace(-20.0 / T, 20.0 / T, 400_001)
    q2 = T**2 * np.sinc(P * T / np.pi) ** 2
    print(T, np.trapezoid(q2, P), np.pi * T)

# Tilt invariance of the lab-frame Doppler denominator at azimuth zero:
theta = np.deg2rad(119.0)
for tilt in np.deg2rad([-17.0, 17.0]):
    beam = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])
    detector = np.array([np.sin(theta - tilt), 0.0, np.cos(theta - tilt)])
    print(np.dot(beam, detector), np.cos(theta))
PY
```

Expected: each finite-time integral agrees with `pi*T` within the finite integration-window error, and both beam/detector dot products agree with `cos(theta)` to floating-point precision.

- [ ] **Step 3: Record adjudication without changing production physics**

Write `match` only if the symbolic expression, dimensions, signs, normalization, coordinate conventions, and limiting cases agree. Otherwise write `discrepancy`, state the smallest exact difference, and stop production edits for that claim.

### Task 5: Add focused anchors and derivation markers

**Files:**
- Modify: `tests/test_anchor_figures.py`
- Modify: `tests/test_crystallography.py`
- Modify: `checks/anchor_figures.py`
- Modify: `src/cxr_mc/montecarlo/spectrum.py`
- Modify: `src/cxr_mc/materials/crystal.py`

**Interfaces:**
- Consumes: Task 4 adjudications.
- Produces: executable anchors for the sinc-to-delta normalization and Henke-f2 coefficient, plus complete `Validation:` back-references. No production numerical behavior changes.

- [ ] **Step 1: Add the finite-duration convergence anchor**

Add this test to `tests/test_anchor_figures.py`:

```python
def test_single_segment_lineshape_converges_to_closed_form(anchor):
    short_ratio = af.single_segment_anchor(anchor, 25.0, L_seg_ang=100.0)[2]
    long_ratio = af.single_segment_anchor(anchor, 25.0, L_seg_ang=3000.0)[2]

    assert abs(long_ratio - 1.0) < abs(short_ratio - 1.0)
    assert long_ratio == pytest.approx(1.0, abs=5e-4)
```

- [ ] **Step 2: Run the focused test and capture its existing-behavior result**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_anchor_figures.py::test_single_segment_lineshape_converges_to_closed_form -v
```

Expected: PASS with the current implementation. This is an anchor for already-existing physics, not a production behavior change.

- [ ] **Step 3: Add the analytic Henke-f2 coefficient anchor**

Import `cxr_mc.materials.crystal as crystal_module` in `tests/test_crystallography.py` and add:

```python
def test_absorption_length_matches_henke_f2_coefficient(monkeypatch):
    energy_eV = np.array([1000.0, 2500.0])
    number_density_per_ang3 = 0.05
    f2 = np.array([2.5, 1.25])

    monkeypatch.setattr(
        crystal_module,
        "henke_dispersion",
        lambda _element, _energy: (np.zeros_like(energy_eV), f2),
    )

    wavelength_ang = crystal_module.HC_EV_ANG / energy_eV
    expected = 1.0 / (
        2.0
        * crystal_module.R_E_ANG
        * wavelength_ang
        * number_density_per_ang3
        * f2
    )
    actual = crystal_module.absorption_length_ang(
        "Si", energy_eV, number_density_per_ang3
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14)
```

- [ ] **Step 4: Run the focused absorption test and capture its existing-behavior result**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_crystallography.py::test_absorption_length_matches_henke_f2_coefficient -v
```

Expected: PASS with the current implementation. This is an anchor for already-existing physics, not a production behavior change.

- [ ] **Step 5: Repair derivation docstrings and validation markers**

Update `checks/anchor_figures.py::line_energy_eV`, `src/cxr_mc/montecarlo/spectrum.py::mc_spectrum`, and `src/cxr_mc/materials/crystal.py::absorption_length_ang` so their docstrings state the cited equation or starting law, assumptions, one limiting case, and respectively include:

```text
Validation: line-energy-dispersion
Validation: finite-time-lineshape
Validation: absorption-length
```

For `mc_spectrum`, retain its broader coherent-spectrum documentation and add the finite-time marker adjacent to the sinc convention. Do not change executable expressions.

- [ ] **Step 6: Run the focused test files**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_anchor_figures.py tests/test_crystallography.py -v
```

Expected: PASS.

### Task 6: Update the validation ledger and verify the slice

**Files:**
- Modify: `docs/physics-validation-ledger.md`

**Interfaces:**
- Consumes: Task 4 adjudications and Task 5 test evidence.
- Produces: evidence-backed statuses, checks, anchor paths, notes, and progress counts for the three claims.

- [ ] **Step 1: Update each ledger row from evidence**

For each matching claim with a green focused anchor, set status to `anchored`, summarize the independent units/convention/limit checks, name the exact pytest anchor, and link its `docs/validation/<id>.md` write-up in notes. For any mismatch, set status to `discrepancy` and describe the difference instead. Do not change the signed-off count.

- [ ] **Step 2: Recalculate the progress line**

Count ledger statuses directly and update the summary. If all three claims match and anchor cleanly, the expected summary is:

```text
Progress: **0 / 31 signed-off** · 7 rederived · 4 anchored · 1 filtered · 1 blocked · 1 discrepancy.
```

- [ ] **Step 3: Run slice verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_anchor_figures.py tests/test_crystallography.py
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run pyright
```

Expected: all commands exit zero.

- [ ] **Step 4: Run full verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit run --all-files
```

Expected: all commands exit zero and the full pytest count is at least the 479-test baseline.
