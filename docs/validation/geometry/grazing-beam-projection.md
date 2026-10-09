# Grazing-beam projection — independent re-derivation

- **id**: `grazing-beam-projection`
- **anchor**: `montecarlo/geometry.py::project_beam_entry`; wired through `montecarlo/transport.py::simulate_trajectories` (`tilt_polar_rad`/`tilt_azim_rad`) and `montecarlo/runner/__init__.py::{_transport_case,_brem_for_case}`
- **source**: elementary rectangular ray–plane intersection (no literature equation)
- **verifier**: fresh context (did not write the implementation)

Process note: the derivation below was written after reading only the ledger row and the function docstring/signature. The Read chunk used to fetch the docstring unavoidably exposed the first ~13 lines of the body; the derivation was carried out from the stated geometry (lab-frame collimated bundle, rotation `R` mapping sample `+z` to the lab surface normal, ray–plane intersection with sample-frame `z = 0`) and its closed form was completed before inspecting the rest of the implementation, `_rotation_between`, or the transport/runner wiring.

## 1. Problem statement (from docstring + ledger only)

Lab frame: beam axis `+z_lab`; electron `i` is a ray with origin `o_lab = (u_i, v_i, 0)` (Gaussian beam-spot draw in the plane perpendicular to the beam) and direction `+z_lab` (perfect collimation). Sample frame: entrance face is the plane `z = 0`; the tilt rotation `R` maps sample `+z` onto the lab surface normal

n_lab = (sinθ cosφ, sinθ sinφ, cosθ),   θ = tilt_polar, φ = tilt_azim.

A lab vector `a` maps to the sample frame as `a_s = Rᵀ a`. Required output: the sample-frame `(x, y)` of the ray–plane intersection with `z = 0`.

## 2. Independent derivation

### 2.1 Ray–plane intersection

Sample-frame ray: origin `o_s = Rᵀ (u, v, 0)ᵀ`, direction `d = Rᵀ ẑ_lab`. Intersection with `z = 0`:

s* = − o_s·ẑ / d·ẑ ,    p0 = o_s + s* d ,    p0·ẑ = 0 exactly.

Output `(x, y) = (p0_x, p0_y)`. This is the whole physics; everything else is the choice of `R`.

### 2.2 Choice of R: minimal (Rodrigues) rotation

Taking `R` as the minimal rotation carrying `ẑ` to `n_lab` (rotation axis `k = ẑ × n_lab / |ẑ × n_lab| = (−sinφ, cosφ, 0)`, angle `θ`), Rodrigues gives

Rᵀ = I − sinθ K + (1 − cosθ) K²,   K a = k × a,

Rᵀ = [ 1 − (1−cθ)cφ²,  −(1−cθ)sφcφ,   −sθ cφ ] [ −(1−cθ)sφcφ,    1 − (1−cθ)sφ²,  −sθ sφ ] [ sθ cφ,           sθ sφ,           cθ    ]

with `cθ = cosθ`, `sθ = sinθ`, `cφ = cosφ`, `sφ = sinφ`.

Then

d = Rᵀ ẑ = (−sθ cφ, −sθ sφ, cθ), o_s,z = sθ (u cφ + v sφ), s* = −(sθ/cθ)(u cφ + v sφ).

Substituting into `p0 = o_s + s* d` and simplifying with `sθ²/cθ − (1−cθ) = (1−cθ)/cθ`:

p0_x = u [1 + cφ² (1−cθ)/cθ] + v sφ cφ (1−cθ)/cθ p0_y = u sφ cφ (1−cθ)/cθ    + v [1 + sφ² (1−cθ)/cθ] p0_z = 0.

### 2.3 Closed form

The map is linear, `(x, y)ᵀ = M (u, v)ᵀ`, with

M(θ, φ) = I₂ + (1/cosθ − 1) â âᵀ ,      â = (cosφ, sinφ).

`M` is symmetric positive-definite with eigenpairs

M â  = (1/cosθ) â        (stretch 1/cosθ along the tilt azimuth), M â⊥ = â⊥                (perpendicular extent unchanged),

and `det M = 1/cosθ` (footprint area magnification `A → A/cosθ`, i.e. the standard grazing-incidence flux dilution `cosθ` per unit face area). Special cases:

- `φ = 0`:  `M = diag(1/cosθ, 1)` → `(x, y) = (u/cosθ, v)`.
- `φ = 90°`: `M = diag(1, 1/cosθ)` → `(x, y) = (u, v/cosθ)`.

A convention-independent invariant (used as a probe below): mapping the entry point back to the lab, `p_lab = R (x, y, 0)ᵀ`, must have transverse coordinates exactly `(u, v)` — a collimated bundle along `ẑ_lab` cannot change an intersection point's lab-transverse position.

## 3. Cheap filters (pre-inspection)

- **Units**: `M` is dimensionless; Ang in → Ang out. Angles in radians. Pass.
- **Limits**:
  - `θ = 0`: `M = I` exactly — identity; the docstring's early-return makes it bit-for-bit. Pass.
  - `θ → 90°`: `1/cosθ → ∞`; any finite footprint is overfilled and almost all of a fixed-width Gaussian lands off-face → misses. This is precisely the overlap loss that must compete with the `1/cosθ` path-length enhancement (the ledger's ~89° flux-peak rationale). Pass.
  - Out-of-plane invariance: the component of `(u, v)` along `â⊥` is unchanged for every `θ`. Pass.
- **Signs/conventions**:
  - `M` depends on `φ` only through `â âᵀ`, so `φ` and `φ + 180°` give the same stretch — physically required (tilting toward `+â` or `−â` elongates the same axis). Pass.
  - No spurious in-plane rotation: for the minimal rotation, `M` is symmetric (pure stretch), so the projected Gaussian's principal axes are `â`, `â⊥`. Pass.
  - `s*` sign: for `θ > 0, φ = 0, u > 0`, `o_s,z = u sinθ > 0` and the ray must travel backward (`s* = −u tanθ < 0`) to meet `z = 0`; for a parallel bundle the plane extends both ways along the bundle, so negative `s*` is legitimate. Consistent.

## 4. Comparison against the implementation

*(Written after reading the implementation body, `_rotation_between` (`materials/crystal.py:345`), and the transport/runner wiring.)*

- `project_beam_entry` builds `n_lab = (sθ cφ, sθ sφ, cθ)`, `Rt = _rotation_between(ẑ, n_lab).T`, then `o = u·Rt[:,0] + v·Rt[:,1]` (= `Rᵀ (u, v, 0)ᵀ`, since column `j` of `Rᵀ` is `Rᵀ ê_j`), `s* = −o_z / d_z` with `d = Rt[:,2] = Rᵀ ẑ`, `p0 = o + s* d`, returning `p0[:, :2]`. Term-for-term the ray–plane intersection of §2.1.
- `_rotation_between(u_hat, t_hat)` returns `I + s K + (1−c) K²` with unit axis `k = u×t/|u×t|`, `s = |u×t|`, `c = u·t` — exactly the Rodrigues minimal rotation assumed in §2.2. Its antiparallel branch (`s < 1e-12`, `c < 0`) is unreachable for physical tilts `θ < 90°`. Same helper and same `(ẑ, n_lab)` arguments as `tilted_geometry`, so the projected footprint and the in-sample `beam_dir = Rᵀ ẑ` share one rotation convention. Consistent.
- `tilt_polar_rad = 0` (falsy test) short-circuits to `offsets_uv.copy()` — the bit-for-bit identity limit, independent of `tilt_azim_rad`. Matches §3.
- Wiring (`transport.py::simulate_trajectories`, lines 358–385): lab-frame Gaussian offsets (child RNG stream, `finite-beam-size`) pass through `project_beam_entry(offsets, tilt_polar_rad, tilt_azim_rad)` into `pos[:, :2]`, and only then the finite-footprint test `alive = (|x| ≤ width/2) & (|y| ≤ height/2)` sets `n_missed = (~alive).sum()`. Missed electrons generate no segments.
- Normalization: `simulate_trajectories` stores the *requested* electron count `"Ne": Ne` in the segments dict (transport.py:560); `mc_spectrum` and `mc_brem_spectrum` divide by `segments["Ne"]` (spectrum.py:441–442, 643). Misses therefore stay in the per-incident-electron denominator, as claimed.
- Runner wiring: `_transport_case` converts `tilt_deg`/`tilt_azim_deg` and passes `tilt_polar_rad`/`tilt_azim_rad` into *both* `simulate_trajectories` calls (line grid and brem, runner.py:297–333); `_brem_for_case` does the same for the brem-repair path (runner.py:388–405). Consistent.

**Result: matches.** No divergent factor, sign, axis, or convention found.

## 5. Numeric probes (independent)

Probe script (fresh formulas `M_ref` and own Rodrigues `R`, not implementation helpers); offsets `~N(0, 1 µm)` in Ang, N = 200 unless noted. All values are actual run output.

- `(θ=37°, φ=113°)`: `project_beam_entry` vs `M (u, v)ᵀ` — max abs diff `7.3e-12` Ang on ~1e4 Ang offsets (double-precision roundoff). Match.
- Lab-frame invariant: `R (x, y, 0)ᵀ` transverse residual vs `(u, v)` max `7.3e-12` Ang. The projected points really lie on the original lab rays.
- `(θ=75°, φ=0)`: `x` diff vs `u/cosθ` max `1.5e-11` Ang; `y` column bitwise equal to `v`. In-plane stretch, out-of-plane invariance. Match.
- `(θ=75°, φ=90°)`: `y` diff vs `v/cosθ` max `1.5e-11`; `x` diff vs `u` max `5.5e-12`. Stretch axis follows azimuth. Match.
- `φ=23°` vs `φ=203°` at `θ=60°`: max diff `2.2e-11`. `φ → φ+180°` symmetry.
- `θ=0, φ=2.3 rad`: output bitwise equal to input (`array_equal`), and a copy (not the same array). Identity limit pass.
- `θ=89°`: measured stretch `57.29868849854982` vs `1/cos 89° = 57.29868849854990`. Grazing limit pass.
- Wiring probe (`simulate_trajectories`, Si slab, 30 keV, `Ne=4000`, 1 mm-FWHM beam, 1 mm × 1 mm footprint, `max_steps=1`, seed 7): at `θ=85°, φ=0` the analytic lab-frame acceptance `P(miss) = 1 − erf(w·cosθ/(σ√2))·erf(w/(σ√2))` (w = 0.5 mm, σ = FWHM/2.3548) predicts `0.9378`; measured `n_missed/Ne = 0.93775`. Untilted control: predicted `0.4209`, measured `0.42825` (binomial σ ≈ 0.008 at N=4000; within ~1σ). The miss fraction is quantitatively the `cosθ`-shrunken acceptance window. Pass.
- `θ=85°, φ=90°` on the square footprint: `0.93525` — statistically identical by symmetry. Pass.
- `Ne` bookkeeping: `segs["Ne"] == 4000` (requested) with `n_missed = 3751` at 85°; spectra divide by `segments["Ne"]`, so misses stay in the normalization. Pass.
- Repo anchors (corroboration only): `tests/montecarlo/test_finite_transverse_geometry.py -k project_beam_entry` (5 passed) and `tests/montecarlo/test_montecarlo.py -k grazing` (2 passed).

## 6. Verdict

- Filters: units pass; limits pass; signs/conventions pass.
- Re-derivation: **matches** — closed form `M = I + (1/cosθ − 1) â âᵀ`, ray–plane intersection, Rodrigues rotation, and `n_missed`/`Ne` wiring all agree symbolically and numerically.
- Suggested status: `unverified → rederived` (human applies; `signed-off` reserved for human adjudication).

## Face-arrival delay (2026-10-08, #370)

The same intersection gives each electron's vacuum path to the face,
$s^*=-o_z/b_z=\mathbf p_0\cdot\mathbf b$, since $\mathbf o\perp\mathbf b$.
Analytic-spot transport now adds $s^*/\beta$ to $t_0$, so the bunch's pulse front
is perpendicular to the beam rather than parallel to the tilted face; zero tilt
adds exact zeros, and electron blocks restore it after resampling bunch offsets.
GDF beams already included their drift; grooved entries are unchanged. The
projection Jacobian $J=\partial\mathbf p_{0,xy}/\partial(u,v)$ feeds the coherent
transverse form factor ([`transverse-bunch-form-factor`](../radiation-physics/transverse-bunch-form-factor.md)).
