# Validation: blazed-groove-geometry

- **Claim id**: `blazed-groove-geometry`
- **Anchors**:
  `src/cxr_mc/montecarlo/groove.py::{blazed_groove_spec,surface_depth_ang,in_material,first_surface_event,escape_distance_ang,entry_points}`;
  `src/cxr_mc/montecarlo/transport.py::simulate_trajectories`;
  `src/cxr_mc/montecarlo/spectrum/lines.py::mc_spectrum`; `src/cxr_mc/montecarlo/spectrum/brem.py::mc_brem_spectrum`
- **Source**: elementary periodic ray–plane intersection (no literature
  equation)
- **Verifier**: independent fresh context (did not write the implementation),
  2026-07-24. Sections 1--3 were derived **before** reading implementation
  bodies (only the ledger row, module/function derivation docstrings, and
  signatures were read first, per `docs/validation/README.md`).

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
  length (`escape_distance_ang`) as quoted in the docstrings;
- exact material-to-vacuum and vacuum-to-material facet events for arbitrary
  electron directions;
- material-only stopping, scattering, radiation, and Beer--Lambert optical
  depth, with energy/direction invariant vacuum legs and clock advance
  `L_vacuum/beta`;
- finite-footprint electron transport decoupled from laterally periodic photon
  escape, with claimed edge error `O(Λ / crystal_width)`.

## 2. Cheap filters

**Dimensions.** Λ, h, c = Λ cos tp, s1, L, x, z all carry Å; tp, θ are
radians (dimensionless in trig). Every claimed formula is a sum of Å-valued
terms times dimensionless trig/floor/mod factors. The stored clock has
`c t` units Å, so `Δt_ang = L[Å]/beta` is dimensionally correct.
Beer--Lambert optical depth `tau = mu[Å^-1] L[Å]` is dimensionless. Pass.

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

Analytic signs, flat-profile limits, exact valley-band handling, and
finite-footprint edge scaling pass after the implementation corrections
re-checked in §§4.1 and 4.4.

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

### 3.4 Facet-event signs, bands, and ties

Define signed plane functions

    W_k(r) = n̂·r − kΛ cos tp                 (working)
    R_k(r) = b·r − kΛ sin tp                 (relief).

On a physical working facet, material is on `W_k <= 0`; on a physical relief
facet, material is on `R_k >= 0`. Therefore, for ray direction d,

    working exit:  n̂·d > 0       working entry:  n̂·d < 0
    relief exit:   b·d < 0        relief entry:   b·d > 0.

These signs are independent of period index. Plane intersections outside
`0 <= z <= h` are not surface events: below the valley (`z > h`) both sides
are material; above an apex (`z < 0`) both sides are vacuum.

If the relevant plane rate is zero, the ray is tangent to that facet family
and cannot cross it. It may still reach the other family or a shared
apex/valley. At an endpoint tie, the physical rule follows the closed-material
predicate: compare states at `q − epsilon*d` and `q + epsilon*d`; accept only
an actual state change. A material--material or vacuum--vacuum contact is not
an event. Strictly forward `s > epsilon` plus one post-re-entry nudge prevents
the same boundary from being returned indefinitely.

### 3.5 Electron transport through vacuum

Let material collision distance `X` be exponential with rate `lambda^-1`.
If a facet truncates a flight after material distance `a` and no collision has
occurred, then

    P(X − a > y | X > a) = exp(−y/lambda).

Thus the residual material free path has the original exponential law.
Vacuum contributes zero collision hazard, so discarding the old residual and
sampling a fresh exponential after re-entry is statistically exact under the
transport kernel's local-rate model. Vacuum also has no stopping or scattering:

    E_after = E_before,       d_after = d_before.

The code's time coordinate is `c*t` in Å. Since speed is `beta*c`,

    Delta(c*t) = c L_vacuum/(beta*c) = L_vacuum/beta  [Å].

Only material intervals may enter segment arrays. A re-entry is not a
permanent exit and must not increment back/side/transmission counters.

### 3.6 Coherent and bremsstrahlung Beer--Lambert paths

For supported photon direction n̂, `b·n̂ = 0`; relief-plane coordinate stays
constant. At working planes, `dW_k/ds = n̂·n̂ = 1 > 0`, always the outward
working-facet sign. After the first physical outward crossing, later working
crossings have lower z and the ray cannot acquire an inward sign; relief
crossings are impossible. Hence no material re-entry exists for this direction
and `escape_distance_ang` is the complete material path.

For attenuation coefficient `mu(E)` in Å^-1, both coherent and bremsstrahlung
terms must use

    T_abs(E) = exp[−mu(E) L_esc].

Grooving changes `L_esc` only. Coherent amplitudes/resonance kinematics and
bremsstrahlung cross sections remain unchanged. Bremsstrahlung emission from
one material segment retains its factor
`n_i L_segment (d sigma_i/dk)/(4 pi Ne)`; vacuum legs supply neither
`L_segment` nor optical depth.

### 3.7 Finite photon footprint: independent edge scale

Electron launch and arbitrary-direction transport must use the finite prism.
Treating photon grooves as periodic is a separate approximation. For an
emitter at `(x,z)` and supported `n̂ = (cos tp,0,−sin tp)`, horizontal travel
before groove escape is

    Delta x = L_esc cos tp.

The downstream `+x` side face wins whenever

    W/2 − x < Delta x.

At fixed z, the affected edge strip therefore has width `Delta x`, not
generally one groove period. From §3.3,

    L_esc = z/sin tp − delta,       0 <= delta < Λ cos tp,

so

    Delta x = z cot tp − delta cos tp
            = z cot tp + O(Λ).

For emitters uniform in x, the ignored-side fraction at fixed z is

    f_edge(z) = min(Delta x/W, 1)
              = min(z cot(tp)/W + O(Λ/W), 1).

Thus `O(Λ/W)` is only the phase-scale correction to the edge width. It is not
the total finite-side error unless all relevant emission depths are `O(h)`.

## 4. Implementation comparison (read after §3 was frozen)

### 4.1 Geometry

- `blazed_groove_spec`, `surface_depth_ang`, `in_material`, `entry_points`,
  and `escape_distance_ang` match §§3.1--3.3 symbolically, including facet
  branches, signs, bands, and the flat-profile limit.
- `first_surface_event` uses working sign multiplier `+1` and relief multiplier
  `−1`, exactly matching §3.4. Zero plane rate is skipped. Interior random-ray
  transitions covered by the focused reference-march tests pass.
- Exact valley handling now uses
  `band_tol = 32*eps_machine*max(spacing, depth)` and admits candidates only
  inside `[-band_tol, h + band_tol]` before applying the two-sided material
  predicate. Re-check used:

      Λ = 2 Å, tp = 0.61 rad
      h = 0.9390993563190676 Å
      x_valley = h tan(tp) = 0.6563542536839531 Å
      p = (x_valley, 0, h + 1 Å), d = (0,0,−1)

  Exact geometry exits through the valley at `s = 1 Å`; implementation now
  returns exactly `1.0 Å`. Here `band_tol = 1.4210854715202004e−14 Å`, while
  the ray--plane result's former overshoot is one ulp.
- Independent exclusion probe displaced the nonphysical working-plane
  intersection by `1e−8 Å`, about `7.0e5*band_tol`, beyond the valley. The
  implementation rejected that candidate and returned the physical relief
  event `1.0000000148848758 Å`, matching the independently calculated distance
  exactly. Thus the tolerance does not admit materially out-of-band
  intersections.
- Tangencies away from facet endpoints remain skipped; endpoint state changes
  remain governed by the two-sided predicate.

`escape_distance_ang` returns the next working plane rather than zero for an
emitter exactly on a working facet because `s1 = c − mod(d,c)` is in `(0,c]`.
Its documented limit is from just inside, and radiating segment midpoints are
interior, so this separate measure-zero convention does not change the verdict.

### 4.2 Electron transport

`simulate_trajectories` compares sampled collision/prism/layer distances with
the exact surface exit, truncates the material segment at the first event,
applies material stopping and `step/beta`, and suppresses elastic scattering
at a facet. It then searches for an entry along the unchanged direction.
On re-entry it:

- copies post-material energy into the vacuum diagnostic;
- leaves energy and direction unchanged;
- advances `clock` by `distance/beta_from_keV(E)`;
- nudges into material by a period-scaled epsilon;
- resamples the material collision distance on the next iteration;
- does not increment exit counters.

Permanent no-reentry rays increment entrance-face exit count. Vacuum arrays
are separate from `r_mid`, `L_ang`, `E_keV`, and `v_hat`; spectrum kernels see
only material segments. This matches §§3.5--3.6. The exponential
memorylessness argument is exact for the kernel's locally exponential
free-path model.

### 4.3 Coherent and bremsstrahlung radiation

Both spectrum functions validate the supported direction and single-slab
scope. `mc_spectrum` and `mc_brem_spectrum` call the same
`escape_distance_ang`; both form `exp(−mu*L_esc)`. Vacuum legs never enter
either segment sum. Coherent amplitude/resonance expressions and
bremsstrahlung cross sections are unchanged. Supported-direction no-reentry
proof in §3.6 therefore matches implementation.

### 4.4 Finite-footprint decoupling

Transport correctly honors finite launch misses, material side exits, and a
side face encountered before vacuum re-entry. Both photon kernels deliberately
ignore stored finite dimensions when `groove` is present, producing the same
periodic result with or without footprint metadata.

That decoupling remains intentionally implemented. The correction design,
ledger row, and coherent-spectrum docstring/comment now state the independently
derived magnitude:

    f_edge(z) = min(L_esc cos(tp)/W, 1)
              = min(z cot(tp)/W + O(Λ/W), 1).

They explicitly identify `O(Λ/W)` as only the periodic-phase correction and
retain the `min(...,1)` cap. Runtime still gives `groove` precedence over stored
finite dimensions in both photon kernels, so the documented approximation and
implemented laterally periodic model now agree. Example magnitude remains:
`Λ = 2 μm`, `W = 5 mm`, `tp = 45°`, `z = 20 μm` gives affected fraction
`4e−3` (0.4%), while `Λ/W = 4e−4` (0.04%) is only the smaller phase correction.

### 4.5 Checks run

- Current-tree focused CPU re-check:
  three endpoint/reference tests passed (`3 passed, 30 deselected`).
- Independent numeric probe confirms the exact one-ulp valley event and
  materially out-of-band rejection described in §4.1.
- Supplied remote evidence for the exact fix snapshot:
  `tests/montecarlo/test_groove.py + tests/plots/test_trajectories.py` — 34 passed; broader
  suite — 192 passed; typecheck and scoped Ruff clean.
- No heavy Monte Carlo, GPU job, or external comparison was run.

## 5. Adjudication

Analytic profile, ordinary facet signs, flat limit, supported-direction
no-reentry, material/vacuum state machine, clock units, energy/direction
invariance, material-only radiation, and coherent/bremsstrahlung
Beer--Lambert paths match.

The two former discrepancies are corrected: closed facet-band endpoints use a
geometry-scaled roundoff tolerance without accepting materially out-of-band
planes, and finite-side documentation carries the missing
`L_esc cos(tp) = z cot(tp) + O(Λ)` leading term while preserving intentional
periodic runtime behavior. No divergent factor, sign, unit, exponent, or
convention remains in the reviewed claim.

**Verdict: `rederived`.**

Suggested ledger edit (human applies): change status from `unverified` to
`rederived`; link this write-up and the endpoint/edge re-review evidence.
Do not mark `signed-off`.
