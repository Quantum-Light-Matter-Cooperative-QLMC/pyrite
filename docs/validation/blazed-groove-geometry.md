# Validation: blazed-groove-geometry

- **Claim id**: `blazed-groove-geometry`
- **Anchors**: `src/cxr_mc/montecarlo/groove.py::{blazed_groove_spec, escape_distance_ang, entry_points}`
- **Source**: elementary periodic ray–plane intersection (no literature
  equation); design note `docs/superpowers/plans/2026-07-23-blazed-groove-geometry.md`
- **Verifier**: independent fresh context (did not write the implementation),
  2026-07-23. Derivation below was written **before** reading the
  implementation bodies (only the ledger row and docstring headers were read
  first, per the contract in `docs/validation/README.md`).

## 1. What is claimed

Restricted geometry θ_obs = 90°, tilt_azim = 180°, tilt_polar tp ∈ (0°, 90°).
Sample frame: entrance face z = 0, depth +z. Then

- beam direction b = (sin tp, 0, cos tp),
- observation direction n̂ = (cos tp, 0, −sin tp), with b·n̂ = 0;
- sawtooth grooves along y, period Λ (`spacing_ang`), apexes at x = kΛ, z = 0;
- WORKING facets on planes n̂·r = kΛ cos tp (⊥ n̂, ∥ b);
- RELIEF facets on planes b·r = kΛ sin tp (⊥ b, ∥ n̂);
- groove depth h = Λ sin tp cos tp;
- closed-form electron entry point (`entry_points`) and photon escape path
  length (`escape_distance_ang`) as quoted in the docstrings.

## 2. Cheap filters

**Dimensions.** Λ, h, c = Λ cos tp, s1, L, x, z all carry Å; tp, θ are
radians (dimensionless in trig). Every claimed formula is a sum of Å-valued
terms times dimensionless trig/floor/mod factors. Pass.

**Signs / conventions.** With +z into the material, b_z = cos tp > 0 (beam
enters) and n̂_z = −sin tp < 0 (photon exits through the entrance face).
b·n̂ = sin tp cos tp − cos tp sin tp = 0, consistent with θ_obs = 90°. The
working facet normal is n̂ (facet ⊥ observation direction, so exit rays cross
it head-on); the relief facet normal is b (facet ⊥ beam, so exit rays graze
it — zero shadowing). Pass.

**Limiting cases.**
- tp → 0 or tp → 90°: h = Λ sin tp cos tp → 0 and one of b, n̂ becomes
  tangent to the face; the claim is that these are *rejected*, which is the
  correct behaviour (a silent flat profile would hide the degenerate
  entrance/exit-face assignment). Pass by design.
- h → 0 at fixed z (Λ → 0): claimed L → z/sin tp = z/|n̂_z|, the flat-face
  Beer–Lambert path used by `mc_spectrum`. Verified analytically in §3.3.
  Pass.
- Λ → 0: claimed z_entry → 0 (flat face). s1 ∈ [0, Λ sin tp) → 0, so
  z_entry = s1 cos tp → 0. Pass.

## 3. Independent derivation

### 3.1 Unit cell and depth h

Take the working plane through apex k, n̂·r = kΛ cos tp, and the relief plane
through apex k+1, b·r = (k+1)Λ sin tp. Since (n̂, b) is an orthonormal pair in
the xz-plane, the intersection (the valley line) is

    r_v = n̂ (kΛ cos tp) + b ((k+1)Λ sin tp)
    x_v = kΛ cos²tp + (k+1)Λ sin²tp = kΛ + Λ sin²tp
    z_v = −kΛ sin tp cos tp + (k+1)Λ sin tp cos tp = Λ sin tp cos tp .

Hence the closure depth

    h = Λ sin tp cos tp ,

and x_v ∈ (kΛ, (k+1)Λ) strictly, so the profile is a genuine sawtooth for all
tp ∈ (0°, 90°). Explicit surface profile per period:

    z_s(x) = (x − kΛ) cot tp            for x ∈ [kΛ, kΛ + Λ sin²tp]   (working)
    z_s(x) = ((k+1)Λ − x) tan tp        for x ∈ [kΛ + Λ sin²tp, (k+1)Λ] (relief)

with z_s = h at the valley and 0 at the apexes; material occupies z ≥ z_s(x),
and everything at z ≥ h is material for every x.

### 3.2 Electron entry points

Ray: r(s) = (x0, y0, 0) + s b, s ≥ 0, where (x0, y0, 0) is the flat-face
intersection. Because b·n̂ = 0, n̂·r(s) = x0 cos tp is constant: the ray is
parallel to every working plane and can never enter through one. It crosses
relief planes b·r = mΛ sin tp where

    b·r(s) = x0 sin tp + s  =  mΛ sin tp   ⇒   s_m = mΛ sin tp − x0 sin tp ,

spaced Λ sin tp apart in s. The first non-negative crossing is

    s1 = mod(−x0 sin tp, Λ sin tp) ∈ [0, Λ sin tp) ,

at depth z_entry = s1 cos tp ∈ [0, Λ sin tp cos tp) = [0, h). A relief plane's
*physical* facet is exactly its band z ∈ [0, h] (from apex (mΛ, 0) down to the
valley (mΛ − Λ cos²tp, h), by §3.1), so the first crossing always lands on real
surface, and no earlier surface hit is possible (working planes are unreachable,
earlier relief crossings do not exist by minimality of s1). Therefore

    entry = ( x0 + s1 sin tp,  y0,  s1 cos tp ) ,   z_entry ∈ [0, h) .

This matches the docstring formula exactly. Degenerate x0 on an apex gives
s1 = 0, entry at the apex itself — consistent.

### 3.3 Photon escape distance

Ray: r(s) = (x, 0, z) + s n̂ (y suppressed; profile is y-invariant). Because
n̂·b = 0, b·r(s) is constant: the ray is parallel to every relief plane and can
only exit through working facets. It crosses working planes n̂·r = k c,
c ≡ Λ cos tp, where

    n̂·r(s) = d + s ,   d ≡ n̂·(x, 0, z) = x cos tp − z sin tp ,

so crossings sit at s_k = k c − d, spaced c apart in path length. Depth at a
crossing: z(s) = z − s sin tp, so successive crossing depths decrease by
exactly c sin tp = Λ cos tp sin tp = h per crossing. First crossing ahead of
the ray:

    s1 = c − mod(d, c) ∈ (0, c] ,     z1 = z − s1 sin tp .

A working plane's physical facet is its band z ∈ [0, h] (apex (kΛ, 0) to
valley (kΛ + Λ sin²tp, h), §3.1). Crossings at depth ≥ h are interior points
(all of z ≥ h is material); crossings at depth < 0 are above the surface.
Since consecutive crossing depths differ by exactly h, precisely one crossing
lands in [0, h): the j-th one after s1 with j = floor(z1/h) (j = 0 when
z1 ∈ [0, h) already). The ray stays inside material until then — the only way
out is a working facet, and all earlier working-plane crossings are at depth
≥ h, i.e. interior. After exiting, the ray is parallel to the relief facets
and every later working-plane crossing is at depth < 0, so it never re-enters
(zero shadowing). Total in-material path:

    L = s1 + floor(z1/h) · c ,        z1 ≥ 0 .

For a point *on* the material (z ≥ z_s(x)) one shows z1 ≥ 0 always — e.g. on a
relief facet d = m c − z/sin tp gives z1 = 0 exactly (exit at the apex), and
interior points give z1 > 0 — so `max(floor(z1/h), 0)` equals `floor(z1/h)`
in exact arithmetic; the `max(·, 0)` is a floating-point guard for boundary
points where z1 underflows to a tiny negative, and is harmless (it returns
L = s1, the O(c) correct answer for a surface point). This matches the
docstring formula exactly.

**Limits.**
- h → 0 (Λ → 0, tp fixed): s1 ≤ c → 0 and floor(z1/h)·c → (z/h)·c
  = z · (Λ cos tp)/(Λ sin tp cos tp) = z/sin tp. So L → z/sin tp, the flat
  entrance-face path z/|n̂_z|. ✓
- Point just inside its own working facet (d → k c⁻, z ∈ [0, h)):
  s1 → 0⁺, z1 → z ∈ [0, h), floor = 0, L → 0. ✓

**Mean-gain identity** (independent version of the ledger's "analytic
sawtooth mean-gain identity"): for emission depth z ≥ h and x uniform over a
period, u ≡ s1/c is uniform on (0, 1] and

    L = c·( u + floor(z/h − u) )  ⇒  ⟨L⟩ = c·(z/h − 1/2) = z/sin tp − c/2 ,

i.e. the grooves shorten the mean escape path by exactly c/2 = Λ cos tp / 2
relative to the flat face, independent of z (for z ≥ h). Any test asserting
this identity is asserting the same geometry derived here.

## 4. Implementation comparison (read after §3 was frozen)

Read `src/cxr_mc/montecarlo/groove.py` in full after writing §1–3.

- `blazed_groove_spec`: validates θ_obs = 90° and tilt_azim = 180° (atol
  1e-9 rad), rejects tp ∉ (0, π/2) and spacing ≤ 0 with `ValueError`; sets
  `depth_ang = spacing * sin(tp) * cos(tp)`. Matches §3.1 exactly, including
  the reject-don't-degrade behaviour at tp ∈ {0, 90°}.
- `entry_points`: `s1 = mod(−x0·st, spacing·st)`; returns
  `(x0 + s1·st, s1·ct)`. Symbol-for-symbol identical to §3.2.
- `escape_distance_ang`: `d = x·ct − z·st`; `c = spacing·ct`;
  `s1 = c − mod(d, c)`; `z1 = z − s1·st`;
  `L = s1 + maximum(floor(z1/depth_ang), 0)·c`. Symbol-for-symbol identical
  to §3.3, including the `max(·, 0)` boundary guard adjudicated above. Uses
  only numpy ufuncs (`np.mod`, `np.floor`, `np.maximum`), consistent with the
  cupy `__array_ufunc__` dispatch claim.
- Wiring/frame consistency: `runner.py` builds `beam, n_hat =
  tilted_geometry(theta_obs, tp, azim)` and passes the same angles to
  `blazed_groove_spec`; `transport.py::simulate_trajectories` applies
  `entry_points` to the flat-face x (with a uniform groove-phase draw for
  point sources) before transport; `spectrum.py::mc_spectrum` guards the v1
  scope (single slab, laterally infinite, n̂_z < 0) and substitutes
  `escape_distance_ang` only for the escape path. `tilted_geometry(π/2, tp, π)`
  numerically returns b = (sin tp, 0, cos tp), n̂ = (cos tp, 0, −sin tp) for
  tp ∈ {0.2, 0.7, 1.2} rad — the exact frame assumed in §1. One boundary-edge
  note: `entry_points` can yield s1 = 0 → mod boundary, and
  `escape_distance_ang` gives s1 = c (not 0) for a point exactly on a working
  plane; both are measure-zero and consistent with a closed-material
  convention.

### Numeric spot checks (independent brute-force ray march)

Script: verifier-written point-in-material test from §3.1 (`z ≥ z_s(x)`,
sawtooth built from cot/tan branches — no reuse of implementation helpers),
seed 20260723, run 2026-07-23:

- 4 000 random (x, z ∈ [0, 4h], tp ∈ (0.05, π/2−0.05), Λ ∈ [0.5, 50] Å)
  interior emission points: closed form vs stepped ray march along n̂ with
  60-step bisection refinement — worst |ΔL| = 5.7·10⁻¹⁴ Å, 0 mismatches. PASS
- 2 000 interior points with z ≥ h: L ≤ z/sin tp always (escape never exceeds
  the flat-face path; max shortening observed 48.1 Å). PASS
- 5 000 random entry points: entry lies on a relief plane (residual of
  b·r/(Λ sin tp) from integer ≤ 5.7·10⁻¹⁴), z_entry ∈ [0, h), entry point on
  the §3.1 surface (max |z_e − z_s(x_e)|/Λ = 1.9·10⁻¹³), and the open segment
  from (x0, 0) to the entry stays in vacuum. PASS
- h → 0 limit: Λ = 10⁻⁴ Å, z = 50 Å, tp = 37°: L·sin tp/z − 1 = −7.1·10⁻⁷.
  PASS
- Mean-gain identity: tp = 28°, Λ = 7.3 Å, z = 5h, 20 000-point period
  average: ⟨L⟩ vs z/sin tp − c/2 relative error 1.6·10⁻⁶ (finite-sample
  discretization of the uniform phase). PASS
- Relief-facet boundary point (z = h/2 on the facet): L = z/sin tp exactly
  (exit at the apex), finite, matching the z1 = 0 boundary analysis. PASS
- Degenerate-geometry rejections (θ_obs ≠ 90°, tp ∈ {0, 90°},
  tilt_azim ≠ 180°, spacing ≤ 0) all raise `ValueError`. PASS

## 5. Adjudication

Independent derivation, symbolic comparison, and numeric brute-force checks
all agree with the implementation. No divergent factor, sign, exponent, unit,
or convention found. The v1 scope caveats recorded in the ledger (electron
in-flight boundaries stay flat — entry point only honors grooves; brem
self-absorption stays flat-face; single laterally-infinite slab, n_side = 1)
are *scope restrictions*, not errors in the claimed formulas; they are
enforced by explicit guards in `spectrum.py::mc_spectrum` and disclosed in
the ledger row and docstrings.

**Verdict: `rederived`** (filters + independent re-derivation pass;
`tests/test_groove.py` exists as the anchor but was not re-run here — the
independent ray march above covers the same comparison). Sign-off remains a
human action.
