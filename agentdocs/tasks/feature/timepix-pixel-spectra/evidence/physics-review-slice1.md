# Physics review — Slice 1 pixel angular reconstruction (issue #23)

Reviewer role: independent physics review under `.claude/skills/physics-review/SKILL.md`.
Scope: `PixelScorer.angular_shape` / `angular_tiles` nearest-tile reconstruction,
the oracle harness `checks/pixel_reconstruction_oracle.py`, and the high-statistics
evidence in `evidence/oracle-report.high-stat.json`.

**This is a reviewed recommendation, not a sign-off.** Nothing here marks any
ledger row `signed-off`; only a human does that. No implementation file, the
oracle harness, the task `README.md`, the GitHub issue, or any ledger row was
modified. Independent verification runs used read-only scratch scripts outside
the repository.

---

## Executive verdict

**Conditional fail on `(5,5)` — but the failure is a property of the tested
geometry's 1.5 mm pixel pitch, not of nearest-tile as a policy, and the task
doc's diagnosis of the mechanism is inverted.** I reproduced the effect
independently: at 60° polar the coherent-line integrated yield varies by
**+4.5 % per degree of polar angle** and its resonance peak moves by
**−7.3 eV per degree**, while the *azimuthal* dependence is **even in the
out-of-plane offset and first-order zero** (≤1.5 % over the full ±4.3°
half-span). All of the nearest-tile residual is polar; azimuthal binning is
nearly free. The task doc's read ("a 5-wide azimuthal bin under-resolves it
while a 5-wide polar bin doesn't") is exactly backwards, though its underlying
observation — that the near-zero pixels are the ones matching their tile in
polar — is correct. The correct acceptance criterion is therefore not a
`(N, M)` tuple but a **polar tile angular width**: `max|flux error| ≈
4.5 %/deg × (m−1)/2 × p`, with `m` pixels per polar tile at angular pitch `p`.
At the oracle's absurdly coarse `p = 1.074°/pixel`, a 1 % bar admits **no**
polar coarsening at all, so `(5,5)` fails (2.3–3.0 % measured, 4.8 % tile-edge
discontinuity); at the real Timepix3 pitch (`p = 0.039°`) the same bar admits
an 11× polar coarsening, so an anisotropic grid of roughly `(3, 22)` — 66
evaluations for a 65,536-pixel chip — is both cheaper than `(5,5)` in spirit
and comfortably inside tolerance. I also found that the line's intrinsic FWHM
is only **~9 eV**, which (a) makes "0.1 FWHM" ≈ 0.9 eV, ~44× stricter than the
alternative "40 eV" in the same provisional criterion, and *more* binding than
the flux criterion — so "line flux is the binding metric" is an artifact of the
harness measuring a window-diluted centroid; and (b) means naive per-energy-bin
interpolation would split the line into two peaks while the existing metric set
reports a *pass*. **Recommended next action: do not adopt `(5,5)`, do not jump
to interpolation. Re-express the reconstruction policy as an anisotropic,
angular-width-derived grid (fine polar / coarse azimuth), close three specific
harness metric gaps, and run a 3-geometry sweep — most importantly one with the
scattering-plane mirror symmetry broken — before Slice 2 freezes the
`PixelScorer` API.**

---

## 1. Why the error concentrates in line flux, and the polar/azimuthal asymmetry

### 1.1 What actually sets the direction dependence

The line spectrum is PXR + coherent bremsstrahlung (CBS), computed in
`src/pyrite/montecarlo/spectrum/lines.py::mc_spectrum` following Zhai SI
Eqs. 5–7 / Feranchuk Eqs. 13–14. Per segment and reflection the observation
direction `n̂` enters through exactly four places:

1. **Resonance denominator** (`_line_kin_core`, `_in_medium_kinematics`):
   `ω_res = (v·g) / (1 − Re n(ω) (v·n̂))`. Note `v·g` does **not** depend on
   `n̂`; the whole angular dependence of the line *energy* is the Doppler
   factor `D ≡ 1 − β Re n (v̂·n̂)`.
2. **PXR resonant denominator** (`_line_amp_sq_core`): `f_pxr ∝ 1/detuning`
   with `detuning = g² + 2 k·g = g² + 2 ω (n̂·g)`. `|A_PXR|² ∝ 1/detuning²`.
   This is the Bragg-condition sensitivity: it depends on `n̂` both directly
   through `n̂·g` and indirectly through `ω_res(n̂)`.
3. **Finite-time lineshape width** (`_sincsq_lineshape`): the half-width
   parameter carries `(1 − β v̂·n̂) = D`, so `∫ sinc² dE = π/a_width ∝ 1/D`.
   Combined with the explicit `ω_res ∝ 1/D` prefactor, the *integrated* line
   flux carries a kinematic `∝ 1/D²`.
4. **Escape absorption** `T_abs = exp(−μ(E) L_esc(n̂))`, with `L_esc` set by
   `n̂`'s sample-frame z-component.

So the integrated line flux factorizes as
`L(n̂) ∝ |A(n̂)|² · t_L · D(n̂)^-2 · T_abs(n̂)`,
while the line *position* carries only `ω_res ∝ 1/D`.

### 1.2 Why flux, not centroid

The centroid error is bounded by the fractional Doppler shift; the flux error
is not. Analytically, with `ψ` the beam–observation angle,

```
d ln E_res / dθ  =  β sinψ / (1 − β cosψ)
```

At 30 keV, `β = 0.3283`, `ψ = 60°`, `D = 0.8359`: `d ln E/dθ = 0.340 rad⁻¹ =
0.59 %/deg`. With `E_res ≈ 1257 eV` (my analytic value for HOPG (002),
`c = 6.711 Å` from the catalog, `|g| = 1.8725 Å⁻¹`) that predicts
**−7.5 eV/deg**.

I measured it directly (one transported population, 400 electrons, spectra
evaluated at 9 polar and 8 azimuthal probe directions through
`run_case_directions`, same isolation trick as the oracle):

| polar offset | line peak | line flux | continuum |
|---|---|---|---|
| −4.297° | 1288 eV | −18.23 % | +0.90 % |
| −1.074° | 1264 eV | −4.84 % | +0.24 % |
| 0 | **1258 eV** | 0 | 0 |
| +1.074° | 1249 eV | +5.00 % | −0.25 % |
| +4.297° | 1225 eV | +18.20 % | −1.07 % |

Peak shift **−7.3 eV/deg** against the predicted −7.5 eV/deg — the resonance
formula is confirmed to ~3 %. Line flux **+4.2 to +4.9 %/deg** (slightly
convex, call it 4.5 %/deg). Continuum **−0.25 %/deg**.

The ratio is the answer to the question. In relative terms the line *position*
moves at 0.58 %/deg and the line *yield* at 4.5 %/deg — a **7.7 : 1** ratio —
and the position enters an absolute tolerance against a low resonance energy
(1.26 keV), so 0.58 %/deg is only 7.3 eV/deg. Decomposing the 4.5 %/deg:

- kinematic `1/D²` Doppler term: `−2 d ln D/dθ` = **1.19 %/deg** (26 %);
- escape absorption + smooth dipole geometry: bounded above by the
  bremsstrahlung continuum's own gradient, **≤0.25 %/deg** (≤6 %) — and it has
  the *opposite* sign, so the true resonant contribution exceeds the residual;
- remainder, **≈3.3 %/deg (≥73 %)**: the amplitude, dominated by the PXR
  resonant denominator `|A_PXR|² ∝ (g² + 2ω n̂·g)⁻²`. My hand estimate of
  `d ln(detuning⁻²)/dθ` for this geometry is O(2–5) rad⁻¹, consistent in
  magnitude; I do not claim its sign independently because the sign depends on
  the `g`-harmonic orientation convention, which the code's own docstring
  (`tilted_geometry`) flags as unresolved.

**The continuum is the control that proves the point**: bremsstrahlung has no
resonant denominator, so its only angular dependence is the smooth
Bethe–Heitler dipole plus escape absorption. It comes in at 0.25 %/deg — 18×
weaker than the line. That is exactly why continuum clears its 1 % bar at
`(3,3)` while line flux does not.

### 1.3 The polar/azimuthal asymmetry: mirror symmetry, not "steeper in polar"

Refuting the task doc's read. The configuration has an exact **mirror symmetry
about the scattering plane**:

- `tilted_geometry` (and `directions_to_sample_frame`) fix the lab beam along
  `+z` and the detector at azimuth 0, i.e. in the `x–z` plane.
- The scene uses `tilt_deg=30, tilt_azim_deg=0`; the docstring is explicit that
  `ta = 0` "places the tilt in the scattering x–z plane, i.e. the normal has
  zero y-component". `g ∥ n` by default, so `g` is in the `x–z` plane too.
- Multiple scattering is statistically symmetric about the beam.

Therefore every scalar in `L(n̂)` — `v·n̂`, `n̂·g`, `L_esc`, and hence `D`,
`detuning`, `T_abs` — is an **even** function of the out-of-plane component
`n_y`. So `∂L/∂n_y = 0` at `n_y = 0`: azimuthal displacement is second order,
polar displacement is first order. Local `+x` increases polar and local `+y`
increases azimuth (`PlanarPose.from_observation`), so the detector's column
index is the first-order axis and the row index is the second-order axis.

Confirmed in my probe — azimuthal offsets give an **even, sign-independent**
response:

| azimuth offset | ±0.537° | ±1.074° | ±2.149° | ±4.297° |
|---|---|---|---|---|
| line flux error | +0.02 / −0.08 % | −0.26 / −0.06 % | +0.04 / +0.09 % | +1.53 / +0.89 % |
| continuum error | −0.001 % | −0.004 % | −0.016 % | −0.064 % |

Sign does not flip with offset sign (the signature of an even function), and
the magnitude is ~20–90× below the polar response at the same displacement.
The high-stat oracle shows the same thing once the rows are re-keyed by tile
offset rather than by pixel name:

| shape | pure-polar offset | flux err | pure-azimuth offset | flux err |
|---|---|---|---|---|
| (1,1) | ±4 px | +23.8 % / −16.4 % | ±4 px | −1.15 % / −1.20 % |
| (3,3) | ±1 px | +5.9 % / −4.3 % | ±1 px | −0.36 % / −0.60 % |
| (5,5) | ±0.5 px | +2.96 % / −2.02 % | ±0.5 px | −0.03 % / +0.07 % |

Every pixel that matches its tile in polar reports ≤1.2 % *regardless of its
azimuthal offset, up to the full ±4.3° detector half-span*. Every pixel with a
polar offset reports a flux error of −4.0 to −5.9 % per pixel of offset, with
the **sign tracking the sign of the polar offset** (`polar_only`, the one
representative pixel with a positive polar offset, is the one negative flux
error at `(5,5)`). Slope is nearly constant across `(1,1)`, `(3,3)` and
`(5,5)` — i.e. the response is close to linear in polar offset over the whole
±4.3° range.

**Verdict on item 1:** the mechanism is confirmed, refined, and its axis
assignment corrected. The residual is set by the *polar* Bragg-detuning and
Doppler sensitivity of the coherent line; azimuthal under-resolution is not
the cause and is nearly free. Two consequences the task doc should absorb:
(i) a **square** angular grid is structurally the wrong shape for this physics;
(ii) the azimuthal null is a **symmetry** result, so it survives to any polar
angle for an in-plane detector, and **breaks** the moment `tilt_azim_deg ≠ 0`
or the detector is placed off the scattering plane. That is the single most
important thing to test next (§6).

---

## 2. Correctness review of the oracle harness

Read: `checks/pixel_reconstruction_oracle.py`, `instrument/geometry.py`,
`instrument/attenuation.py`, `results/model.py`, `api.py::_simulate_planar`,
`montecarlo/runner/__init__.py::run_case_directions`, `detectors/spec.py`,
`detectors/timepix_response.py`.

### 2.1 What is correct

- **Production fidelity.** The harness reproduces `_simulate_planar` exactly:
  same `planar_detector_rays` → `angular_tiles` → `directions_to_sample_frame`
  → `run_case_directions` → `_attenuation_matrix` → `primary_transmission`
  chain, and indexes `tile_spec[tile_index[row, col]]` the same way
  `SpatialResult._materialize` indexes `intrinsic_by_tile[tile]`. No drift.
- **Apples-to-apples.** Verified: `run_case_directions` transports **once**
  (`_transport_case`) and then loops `_spectrum_case` per direction over the
  same segment arrays. I checked the spectrum phase for RNG consumption and
  found none — mosaic uses deterministic Gauss–Hermite quadrature
  (`_mosaic_quadrature`), so per-direction evaluation is deterministic given
  the transported population. Appending the 12 fine-pixel directions *after*
  the tile directions therefore cannot perturb the tile spectra, and the two
  branches share one noise realization as claimed.
- **Units and frames.** `primary_transmission` takes mm paths against inverse-mm
  coefficients (`linear_attenuation_inv_mm`) → dimensionless optical depth. ✓
  `directions_to_sample_frame` uses `directions @ R`, the row-vector transpose
  action, matching `tilted_geometry`'s `R.T @ n_hat`; handedness is preserved
  and both branches go through the same call in one array. ✓
  `solid_angle = A cos α / d²` with the obliquity sign-checked. ✓
- **Solid angle / transmission held identical.** True, and its omission from
  the intrinsic metrics is harmless: `ΔΩ` is a pixel-only multiplicative
  constant, so it cancels exactly from every relative error. The docstring's
  claim that filter transmission "cannot itself diverge between the two
  branches" is correct as written.
- **Timepix response is a linear, cached, seeded operator.** `Timepix3.score`
  → `get_response(...)` returns a `TimepixResponse` cached on
  `(grid, dE_mc, dE_out, n_mc, seed, thickness, bias)`; `.apply` is
  bin → `R @ n_in` → interp, i.e. linear and identical for tile and pixel. So
  there is **no** independent MC noise between the two branches and no
  amplitude-scale sensitivity. Good.

### 2.2 Findings that bias or blind the reported errors

**(a) The Timepix branch does not match production. [flag; conservative
direction]** `SpatialResult.spectra(measured=True)` scores
`intrinsic × ΔΩ × T(E)`; the harness scores raw `intrinsic`. Linearity makes
`ΔΩ` harmless, but the **energy-dependent** `T(E)` does not commute with `R`
(the response redistributes counts across energy: charge sharing, escape,
~450 eV ToT blur). So `timepix_binned_*` describes an *unfiltered* measured
spectrum, which production never produces for covered pixels — and the covered
pixels here are 9 of 12, at mean transmission 2e-4. Direction of bias: the
correctly-computed `filtered_line_flux_relative_error` is systematically
*smaller* than the unfiltered one for covered pixels (2.55→2.14, 2.37→1.57,
2.29→1.35 %), so filtering damps the reconstruction error and the reported
Timepix numbers are **conservative**. Not a blocker; fix before citing them as
production-representative.

**(b) The centroid metric is window-diluted and understates line displacement.
[flag; anti-conservative]** `_centroid_eV` integrates over the *entire* line
grid (100–4999 eV, 1634 samples). The reference centroid is 1110 eV while the
line peak is 1258 eV, so a large fraction of the integral is off-peak (sinc
tails, other reflections). Measured consequence: the centroid moves at
−4.6 eV/deg while the **peak** moves at −7.3 eV/deg — a factor 1.6 dilution.
The `(5,5)` "2.9 eV centroid error" corresponds to a real peak displacement of
~4 eV. Combined with finding (c) this materially changes the ruling.
**Recommend adding a windowed per-resonance peak-position and FWHM metric.**

**(c) No lineshape metric — and the intrinsic line is only ~9 eV FWHM.
[flag; blocking for any interpolation decision]** I measured the intrinsic
line: peak 1258 eV, **FWHM ≈ 9 eV**, while the peak shifts ~7.9 eV per 1.5 mm
pixel of polar offset. So a one-pixel polar offset displaces the line by
~0.9 FWHM. The current metric set (centroid, integrated flux, continuum)
cannot see lineshape distortion at all. That is *acceptable* for nearest-tile,
which preserves shape exactly and only places it at the wrong direction — but
it makes the harness **unfit to evaluate interpolation**, which is the
harness's own stated fallback. I demonstrated the false pass: per-energy-bin
linear interpolation of the spectrum across polar nodes 4 pixels apart reports
max |flux error| 0.89 % (a 6× improvement over nearest-tile's 5.5 %) while
producing a spectrum whose 9 eV line has been split into two peaks ~30 eV
apart; the centroid metric actually reports it as *worse* than nearest-tile
(5.95 eV vs 5.00 eV) for the wrong reason, and at 2-pixel node spacing reports
it as *better* (2.47 eV) while the double-peak artifact is still present. A
linear functional cannot detect a symmetric shape distortion. **Add an L1 or
peak-height/FWHM shape distance before any interpolation policy is evaluated.**

**(d) `array_split` tiles are unequal and three of the twelve representative
pixels are degenerate. [flag; the reported mean is biased low]**
`np.array_split(arange(9), 5)` yields `[0,1][2,3][4,5][6,7][8]`. The singleton
column group 8 makes `corner_tr`, `edge_right` and `corner_br` polar-exact by
construction — they report ~0 not because of physics but because their tile
contains one column. Those three are 25 % of the sample and drag the reported
`(5,5)` mean from the physically meaningful 2.3–3.0 % cluster down to 1.9 %.
**Report statistics over all 81 pixels, or exclude degenerate tiles, or state
the tile occupancy alongside each row.** The reported *max* (2.95 %) is sound.

**(e) Single realization, no error bars. [flag; small here]** One seed, no
repeats. The correlated design cancels most transport noise, but the realized
segment population is not exactly mirror-symmetric, so the azimuthal residuals
carry a stochastic floor. Empirically that floor is small — the polar-exact
pixels at `(5,5)` report 0.03–0.07 % — but it is inferred, not measured.
**Recommend 3–5 seeds and a reported spread.**

**(f) Point-sample pixels: a modelling limitation larger than the effect under
debate. [flag; important for Slice 2]** `planar_detector_rays` treats each
pixel as a *point sample carrying its full pitch area*. The "direct" reference
is therefore itself a point-sample approximation, not a pixel-solid-angle
integral. For flux this is benign (the response is nearly linear across a
pixel, so the error is second order). For the **line profile** it is not: at
1.5 mm pitch the resonance sweeps ~7.9 eV ≈ 0.9 FWHM *across a single pixel*,
so the physically correct pixel-integrated intrinsic line is roughly twice as
broad as the point-sample line the code reports. This is a property of the
whole pixel model, not of the reconstruction, and at this pitch it is *larger*
than the reconstruction error being argued about. It shrinks with pitch: at
the real Timepix3 55 µm pitch the intra-pixel shift is ~0.29 eV, negligible.
**Slice 2 should document the point-sample assumption explicitly and state the
pitch regime in which it holds.**

**(g) Minor, non-physics.** `pixel_names.index(name)` inside the loop is an
O(n) lookup where `enumerate` would do; `_relative_error` guards exact zero but
not near-zero denominators (not exercised here). Neither affects results.

**(h) Ledger status is correctly declared.** The harness carries no
`Validation:` marker and no `checks/README.md` row, and its docstring says so.
That is right for decision evidence. If any reconstruction-accuracy number
from this work is later asserted in a docstring, doc page, or test, it needs a
source equation, assumptions, a limiting case, a `Validation: <id>` marker, a
ledger row, and a fresh-context `physics-validation` pass — none of which
exists yet. The underlying `mc_spectrum` claims it depends on
(`finite-time-lineshape`, `xray-in-medium-resonance`, `line-absorption-tabulation`,
`self-absorption`) are separately ledgered and were **not** re-verified here.

---

## 3. Recommended line-flux tolerance bar

### 3.1 What the systematic actually is

This is not a noise-like error. It is a **smooth, monotone, spatially
structured ramp** across the detector in the polar direction at 4.5 %/deg,
which nearest-tile replaces with a **staircase**: constant within each tile,
discontinuous at tile boundaries, with step height equal to the full
tile-width gradient. At `(5,5)` on this geometry that is a 4.8 % sawtooth with
five discontinuities imprinted directly onto every energy-window image.

That matters for the bar, because a structured, discontinuous artifact is far
worse than a random error of equal RMS. Two reasons:

1. **It is confusable with signal.** The angular dependence of the PXR/CBS line
   yield *is* the physics a pixel-resolved spectral imager is built to measure.
   Any user fitting `L(θ)` across the chip is fitting the staircase, and any
   ROI comparison across a tile boundary reads a 4.8 % step that is not there.
2. **It does not average down.** Poisson error on a summed ROI falls as
   `1/√N`; this does not. For a full-chip sum at 10⁶ counts, statistics reach
   ~0.1 %, so a ≥1 % systematic dominates every realistic integration.

Comparable engineering systematics on this instrument: Timepix3 per-pixel
threshold dispersion after equalization ~1–2 % in effective threshold; beam
current stability and dead-time correction ~1 %; QE variation « 1 %. So ~1 %
is the natural "everything else" floor and there is no engineering value in
being much better, nor any excuse for being worse.

### 3.2 Recommended bars

| quantity | recommended bar | rationale |
|---|---|---|
| **Per-pixel integrated line flux** | **≤1 %** | at/below the dominant instrumental systematic (threshold dispersion), and below the statistics of any full-chip ROI sum |
| **Tile-boundary flux discontinuity** (new, binding) | **≤0.3 %** | a *discontinuous* artifact must sit below the smooth physical gradient it corrupts; 0.3 % is ~1/15 of a pixel-pitch-scale real variation and below single-pixel gain dispersion |
| **Line position, measured (Timepix) spectra** | **≤45 eV** (0.1 × instrumental FWHM) | the module's own response docstring puts ToT smearing at ≳450 eV FWHM; anything finer is invisible |
| **Line position, intrinsic/"true" spectra** | **≤0.1 × intrinsic FWHM ≈ 0.9 eV here** | the acceptance checks promise "discrete true spectra"; a true spectrum's line must not be displaced by a resolvable fraction of its own width |
| **Continuum** | **≤1 %** (keep as proposed) | met with ~30× margin; not binding |

**Direction-dependence.** The tolerance *magnitude* should be **uniform across
pixel directions** — a user has no reason to accept a worse spectrum at the
edge of the chip than at the centre, and a direction-dependent bar would
license exactly the structured artifact §3.1 rules out. But the **sampling
required to meet it is strongly direction-dependent**, because the gradient is.
Convert the bar into a sampling rule rather than into a per-direction
tolerance:

```
max |flux error|  ≈  |∂lnL/∂θ_pol| · (m − 1)/2 · p
```

with `m` = pixels per polar tile and `p` = pixel angular pitch [deg]. This
reproduces the evidence: `(5,5)` → `m=2`, `p=1.074°` → 4.5×0.537 = 2.4 %
(observed 2.3–3.0 %); `(3,3)` → `m=3` → 4.8 % (observed 4.3–5.9 %); `(1,1)` →
`m=9` → 19.3 % (observed 16.4–23.8 %, superlinear from mild convexity). The
continuum obeys the same rule with a coefficient 18× smaller.

Solving for a 1 % flux bar: **`(m − 1)·p ≤ 0.44°`**. For 0.1-FWHM line
position (0.9 eV at 7.3 eV/deg): **`(m − 1)·p ≤ 0.25°`** — i.e. the intrinsic
line-position criterion is ~1.8× *more* binding than the flux criterion.

**This overturns the task doc's "line flux is the binding metric."** That
conclusion follows only from comparing a window-diluted centroid (§2.2b)
against the lenient half of a provisional criterion whose two halves —
"40 eV" and "0.1 FWHM" — differ by a factor 44 for this geometry (0.1 FWHM =
0.9 eV, not 40 eV). **Recommend retiring the "40 eV or 0.1 FWHM" disjunction
and replacing it with "0.1 × the FWHM of the spectrum actually being
reported"**, which resolves to ~45 eV for Timepix-measured output and ~0.9 eV
for intrinsic output. Both consumers exist in the acceptance checks, so both
bars must hold for their respective products.

For azimuth, the second-order response needs no analogous rule at the 1 %
level: the measured azimuthal error stays ≤1.5 % even at the full ±4.3°
half-span, and ≤0.1 % within ±2.1°. **`N_az ≥ ceil(azimuthal span / 4.2°)`,
i.e. 2–3 tiles for any chip subtending ≲10°**, holds azimuthal error under
~0.1 %.

---

## 4. Ruling on `(5,5)`, with `(3,3)` and `(9,9)` for context

| shape | polar tile | max flux err | tile-edge step | intrinsic line shift | ruling |
|---|---|---|---|---|---|
| `(1,1)` | 9 px = 9.7° | 23.8 % | 40 % | ~63 eV = 7 FWHM | **fail**, catastrophically |
| `(3,3)` | 3 px = 3.2° | 5.9 % | ~9.6 % | ~7.9 eV = 0.9 FWHM | **fail** |
| `(5,5)` | 2 px = 2.1° | 2.95 % | ~4.8 % | ~4 eV = 0.45 FWHM | **fail** |
| `(9,9)` | 1 px | 0 (exact) | 0 | 0 | passes, but performs no reconstruction |

**`(5,5)` fails against my recommended bar**, on all three binding criteria:
2.95 % against a 1 % flux bar, ~4.8 % tile-edge discontinuity against a 0.3 %
bar, and 0.45 intrinsic FWHM of line displacement against a 0.1 FWHM bar. This
agrees with the task doc's tentative read on the *outcome* while disagreeing on
the reason and on which metric binds.

**But `(9,9)` is the wrong remedy, and the failure does not generalize to the
real hardware as stated.** The `(5,5)` failure is driven entirely by the
oracle's `p = 1.074°/pixel`, which is ~27× coarser than the Timepix3 hardware
this feature targets (55 µm at 80 mm = 0.0393°/pixel; the 256×256 chip subtends
the same ~10° as this 9×9 grid). Applying the §3.2 rule at the real pitch:

- 1 % flux bar → `m ≤ 12` → `N_pol ≥ 22` polar tiles for 256 columns;
- 0.1-FWHM intrinsic-line bar → `m ≤ 7` → `N_pol ≥ 37`;
- azimuth → `N_az = 3`.

So a real Timepix3 chip supports an **11× polar coarsening and an ~85× total
coarsening**: `angular_shape ≈ (3, 37)` = 111 radiation evaluations for a
65,536-pixel chip, comfortably inside every bar above and far under the
acceptance check's "no 262,144 independent radiation evaluations" bound.

The correct ruling is therefore: **nearest-tile is a sound policy; the square
`(N, N)` parameterization is not, and `(5,5)` is not a defensible default.**
Acceptance should be stated on the derived quantity `(m−1)·p` per axis, not on
the tuple.

---

## 5. Interpolation / adaptive refinement

**Priority 1 — do this now, before Slice 2 freezes the API. Not
interpolation: make the angular grid anisotropic and derive it from the
tolerance.** `angular_shape` is already a `(ay, ax)` tuple, so `(3, 37)` is
expressible *today* — but nothing in the code, docs, or defaults tells a user
that the two axes are physically inequivalent, and `PixelScorer`'s default
`(1, 1)` plus the task doc's `(5,5)` example both point at the square case.
Recommended:

- document that `ax` (columns / local `+x`) samples the **first-order** polar
  gradient and `ay` (rows / local `+y`) samples a **second-order** azimuthal
  one, for a detector on the scattering plane;
- provide a constructor or helper that picks `(ay, ax)` from a requested
  tolerance and the scene geometry, rather than making users guess a tuple;
- record the chosen shape and the resulting bound in provenance (the
  acceptance checks already require angular sampling to alter observation
  identity).

This removes the entire question with no new physics, no new interpolation
kernel, and no lineshape risk.

**Priority 2 — gated, low urgency: resonance-warped interpolation, if and only
if Priority 1 is measured to be too slow.** I tested naive per-energy-bin
linear interpolation across polar nodes and it is a trap, for a concrete
physical reason: the line is ~9 eV FWHM and moves ~7.9 eV per pixel, so linear
interpolation between nodes more than ~1 pixel apart produces a **two-peak**
spectrum rather than a shifted line. It fixes flux (0.68–0.89 % vs 5.5 % for
nearest-tile at the same node count) while corrupting the observable the
feature exists to deliver — and the current metric set reports a pass. If
interpolation is ever built, it must interpolate **amplitude and resonance
position separately**, warping the energy axis by the analytic
`ω_res(n̂) = ħc (v̄·g)/(1 − Re n (v̄·n̂))` the kernel already computes, and it
must be gated on a lineshape metric (§2.2c), not on flux and centroid.

**Adaptive refinement: not warranted as such.** The gradient is smooth,
monotone, near-linear over ±4.3°, and analytically predictable from
`β sinψ/(1 − β cosψ)` plus the detuning geometry. A fixed grid sized from the
known gradient is simpler, cheaper and more reproducible than adaptivity, and
does not perturb observation identity/provenance run-to-run.

**One adaptive idea that *is* worth costing (medium priority):** transport is
the expensive phase and per-direction spectrum evaluation is comparatively
cheap (`run_case_directions` re-runs only `_spectrum_case`). If that cost ratio
is favourable — it needs a measurement I did not make — the runner could
transport once, probe 3 polar directions, fit `∂lnL/∂θ_pol` for the actual
scene, and auto-size `ax` to meet the requested tolerance. That converts my
geometry-specific 4.5 %/deg into a per-run measured quantity and removes the
generalization risk in §6 entirely. Recommend costing this before committing to
a hard-coded default.

---

## 6. Is the single-geometry evidence enough to generalize?

**No.** The mechanism I identified predicts strong, structured geometry
dependence, and one of its predictions — the azimuthal null — is exactly the
one the architectural recommendation leans on. Specifically:

- The polar coefficient carries `β sinψ/(1 − β cosψ)` (→ 0 at near-normal
  observation, grows toward grazing) *and* the detuning geometry through
  `n̂·g`, which does **not** vanish with `ψ`. So 4.5 %/deg is a
  geometry-specific number and could move by a large factor.
- **The azimuthal null is a symmetry result, not a smallness result.** It holds
  for *any* polar angle as long as the beam, `g` and `n̂` are coplanar. It is
  destroyed the moment `tilt_azim_deg ≠ 0`, the detector is placed at nonzero
  azimuth, or an asymmetric reflection (`recip_miscut_rad`) is used. In that
  case the azimuthal gradient becomes first order and "coarse azimuth" is
  actively wrong.
- The line FWHM (9 eV here) sets whether any interpolation is viable and scales
  with segment length and beam energy.

Recommended additions before Slice 2 proceeds, in priority order:

1. **Broken mirror symmetry — highest information content.** Same hopg /
   30 keV / 60° polar / same grid, but with `tilt_azim_deg = 45` (or,
   equivalently, `PlanarPose.from_observation(azimuth_deg=45)`). This is the
   decisive test of the anisotropic-grid recommendation. If the azimuthal
   response becomes first-order and comparable to the polar one, then §5's
   Priority 1 must become geometry-conditional — the fine axis has to be
   derived from the projection of the (v, g) plane onto the detector, or
   coarsening must be refused outright. **This one geometry alone can
   invalidate the recommendation, so it is not optional.**
2. **Near-normal polar angle**, e.g. polar 20° with a matching tilt, same
   target and energy. Tests the predicted `sinψ/(1 − β cosψ)` scaling of the
   polar coefficient and bounds the best case. Also checks that the
   polar/azimuth distinction degenerates sensibly as `n̂` approaches the
   symmetry axis.
3. **Different target and beam energy**, e.g. a non-HOPG catalog crystal at
   60–100 keV. `β` rises, `D` shrinks, `E_res` and the intrinsic line width
   both move, and the detuning sensitivity changes. Tests whether 4.5 %/deg is
   a constant of the model or a strong function of the case — which decides
   whether a fixed default `angular_shape` is ever safe or whether the
   auto-sizing probe in §5 is mandatory.

Optional cheap fourth: halve the distance to 40 mm, doubling the angular span
at fixed physics. This is the cheapest possible stress test of the width rule
and should reproduce `ε ∝ (m−1)·p` exactly.

I also recommend re-running the existing geometry with **the real Timepix3
pitch** (256×256 at 55 µm, or a 64×64 sub-grid at 55 µm to keep the cost down)
rather than 9×9 at 1.5 mm. The current grid's 1.074°/pixel is so far from the
hardware that it makes `(5,5)` look like a policy failure when it is a pitch
artifact, and it puts the point-sample assumption (§2.2f) outside its regime of
validity.

---

## Summary of actions for the implementation owner

1. Correct the "Slice 1 progress" reading in the task `README.md`: the residual
   is **polar**, not azimuthal, and the mechanism is the coplanar mirror
   symmetry of the (beam, `g`, `n̂`) configuration.
2. Do not adopt `(5,5)`. Re-express the acceptance criterion as
   `(m − 1) · p ≤ 0.44°` (flux) and `≤ 0.25°` (intrinsic line position) per
   polar axis, with `N_az ≥ 2–3`.
3. Retire the "40 eV or 0.1 FWHM" disjunction; state the line-position bar
   against the FWHM of the spectrum actually being reported (≈45 eV measured,
   ≈0.9 eV intrinsic).
4. Harness: add a windowed peak-position/FWHM metric, add a lineshape distance
   metric, apply the filter before the Timepix response, report over all
   pixels or flag degenerate tiles, and run 3–5 seeds.
5. Run the three-geometry sweep in §6, starting with the broken-mirror case.
6. Do not build interpolation yet. If it is ever needed, warp the energy axis
   by the analytic `ω_res(n̂)`; do not interpolate per energy bin.

**A human must still review and sign off. Nothing in this document is a
ledger sign-off, and no `Validation:` marker or ledger row is created or
implied by it.**
