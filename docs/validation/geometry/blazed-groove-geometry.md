# blazed-groove-geometry

- **Claim id**: `blazed-groove-geometry`
- **Anchors**: `src/pyrite/montecarlo/groove.py::{blazed_groove_spec,surface_depth_ang,in_material,first_surface_event,escape_distance_ang,entry_points}`; `src/pyrite/montecarlo/transport/api.py::_simulate_trajectories`; `src/pyrite/montecarlo/transport/cores.py::make_cpu_transport_core`; `src/pyrite/montecarlo/transport/core_geometry.py::_groove_reentry_scalar`; `src/pyrite/montecarlo/transport/grooved_batches.py::GroovedVacuumCore`; `src/pyrite/montecarlo/spectrum/cxr/_spectrum.py::mc_cxr_spectrum`; `src/pyrite/montecarlo/spectrum/brem.py::mc_brem_spectrum`
- **Source**: elementary periodic ray–plane intersection (no literature equation)
- **Verifier**: independent fresh context (did not write the implementation), 2026-07-24. Sections 1--3 were derived **before** reading implementation bodies (only the ledger row, module/function derivation docstrings, and signatures were read first, per the [validation methodology](../methodology.md)).

## 1. What is claimed

Restricted geometry θ_obs = 90°, tilt_azim = 180°, tilt_polar tp ∈ (0°, 90°). Sample frame: entrance face z = 0, depth +z. Then

- beam direction b = (sin tp, 0, cos tp),
- observation direction n̂ = (cos tp, 0, −sin tp), with b·n̂ = 0;
- sawtooth grooves along y, period Λ (`spacing_ang`), apexes at x = kΛ, z = 0;
- WORKING facets on planes n̂·r = kΛ cos tp (⊥ n̂, ∥ b);
- RELIEF facets on planes b·r = kΛ sin tp (⊥ b, ∥ n̂);
- groove depth h = Λ sin tp cos tp;
- closed-form electron entry point (`entry_points`) and photon escape path length (`escape_distance_ang`) as quoted in the docstrings;
- exact material-to-vacuum and vacuum-to-material facet events for arbitrary electron directions;
- material-only stopping, scattering, radiation, and Beer--Lambert optical depth, with energy/direction invariant vacuum legs and clock advance `L_vacuum/beta`;
- finite-footprint electron transport decoupled from laterally periodic photon escape, with claimed edge error `O(Λ / crystal_width)`.

## 2. Cheap filters

**Dimensions.** Λ, h, c = Λ cos tp, s1, L, x, z all carry Å; tp, θ are radians (dimensionless in trig). Every claimed formula is a sum of Å-valued terms times dimensionless trig/floor/mod factors. The stored clock has `c t` units Å, so `Δt_ang = L[Å]/beta` is dimensionally correct. Beer--Lambert optical depth `tau = mu[Å^-1] L[Å]` is dimensionless. Pass.

**Signs / conventions.** With +z into the material, b_z = cos tp > 0 (beam enters) and n̂_z = −sin tp < 0 (photon exits through the entrance face). b·n̂ = sin tp cos tp − cos tp sin tp = 0, consistent with θ_obs = 90°. The working facet normal is n̂ (facet ⊥ observation direction, so exit rays cross it head-on); the relief facet normal is b (facet ⊥ beam, so exit rays graze it — zero shadowing). Pass.

**Limiting cases.**
- tp → 0 or tp → 90°: h = Λ sin tp cos tp → 0 and one of b, n̂ becomes tangent to the face; the claim is that these are *rejected*, which is the correct behaviour (a silent flat profile would hide the degenerate entrance/exit-face assignment). Pass by design.
- h → 0 at fixed z (Λ → 0): claimed L → z/sin tp = z/|n̂_z|, the flat-face Beer–Lambert path used by `mc_cxr_spectrum`. Verified analytically in §3.3. Pass.
- Λ → 0: claimed z_entry → 0 (flat face). s1 ∈ [0, Λ sin tp) → 0, so z_entry = s1 cos tp → 0. Pass.

Analytic signs, flat-profile limits, exact valley-band handling, and finite-footprint edge scaling pass after the implementation corrections re-checked in §§4.1 and 4.4.

## 3. Independent derivation

### 3.1 Unit cell and depth h

Take the working plane through apex k, n̂·r = kΛ cos tp, and the relief plane through apex k+1, b·r = (k+1)Λ sin tp. Since (n̂, b) is an orthonormal pair in the xz-plane, the intersection (the valley line) is

r_v = n̂ (kΛ cos tp) + b ((k+1)Λ sin tp) x_v = kΛ cos²tp + (k+1)Λ sin²tp = kΛ + Λ sin²tp z_v = −kΛ sin tp cos tp + (k+1)Λ sin tp cos tp = Λ sin tp cos tp .

Hence the closure depth

h = Λ sin tp cos tp ,

and x_v ∈ (kΛ, (k+1)Λ) strictly, so the profile is a genuine sawtooth for all tp ∈ (0°, 90°). Explicit surface profile per period:

z_s(x) = (x − kΛ) cot tp            for x ∈ [kΛ, kΛ + Λ sin²tp]   (working) z_s(x) = ((k+1)Λ − x) tan tp        for x ∈ [kΛ + Λ sin²tp, (k+1)Λ] (relief)

with z_s = h at the valley and 0 at the apexes; material occupies z ≥ z_s(x), and everything at z ≥ h is material for every x.

### 3.2 Electron entry points

Ray: r(s) = (x0, y0, 0) + s b, s ≥ 0, where (x0, y0, 0) is the flat-face intersection. Because b·n̂ = 0, n̂·r(s) = x0 cos tp is constant: the ray is parallel to every working plane and can never enter through one. It crosses relief planes b·r = mΛ sin tp where

b·r(s) = x0 sin tp + s  =  mΛ sin tp   ⇒   s_m = mΛ sin tp − x0 sin tp ,

spaced Λ sin tp apart in s. The first non-negative crossing is

s1 = mod(−x0 sin tp, Λ sin tp) ∈ [0, Λ sin tp) ,

at depth z_entry = s1 cos tp ∈ [0, Λ sin tp cos tp) = [0, h). A relief plane's *physical* facet is exactly its band z ∈ [0, h] (from apex (mΛ, 0) down to the valley (mΛ − Λ cos²tp, h), by §3.1), so the first crossing always lands on real surface, and no earlier surface hit is possible (working planes are unreachable, earlier relief crossings do not exist by minimality of s1). Therefore

entry = ( x0 + s1 sin tp,  y0,  s1 cos tp ) ,   z_entry ∈ [0, h) .

This matches the docstring formula exactly. Degenerate x0 on an apex gives s1 = 0, entry at the apex itself — consistent.

### 3.3 Photon escape distance

Ray: r(s) = (x, 0, z) + s n̂ (y suppressed; profile is y-invariant). Because n̂·b = 0, b·r(s) is constant: the ray is parallel to every relief plane and can only exit through working facets. It crosses working planes n̂·r = k c, c ≡ Λ cos tp, where

n̂·r(s) = d + s ,   d ≡ n̂·(x, 0, z) = x cos tp − z sin tp ,

so crossings sit at s_k = k c − d, spaced c apart in path length. Depth at a crossing: z(s) = z − s sin tp, so successive crossing depths decrease by exactly c sin tp = Λ cos tp sin tp = h per crossing. First crossing ahead of the ray:

s1 = c − mod(d, c) ∈ (0, c] ,     z1 = z − s1 sin tp .

A working plane's physical facet is its band z ∈ [0, h] (apex (kΛ, 0) to valley (kΛ + Λ sin²tp, h), §3.1). Crossings at depth ≥ h are interior points (all of z ≥ h is material); crossings at depth < 0 are above the surface. Since consecutive crossing depths differ by exactly h, precisely one crossing lands in [0, h): the j-th one after s1 with j = floor(z1/h) (j = 0 when z1 ∈ [0, h) already). The ray stays inside material until then — the only way out is a working facet, and all earlier working-plane crossings are at depth ≥ h, i.e. interior. After exiting, the ray is parallel to the relief facets and every later working-plane crossing is at depth < 0, so it never re-enters (zero shadowing). Total in-material path:

L = s1 + floor(z1/h) · c ,        z1 ≥ 0 .

For a point *on* the material (z ≥ z_s(x)) one shows z1 ≥ 0 always — e.g. on a relief facet d = m c − z/sin tp gives z1 = 0 exactly (exit at the apex), and interior points give z1 > 0 — so `max(floor(z1/h), 0)` equals `floor(z1/h)` in exact arithmetic; the `max(·, 0)` is a floating-point guard for boundary points where z1 underflows to a tiny negative, and is harmless (it returns L = s1, the O(c) correct answer for a surface point). This matches the docstring formula exactly.

**Limits.**
- h → 0 (Λ → 0, tp fixed): s1 ≤ c → 0 and floor(z1/h)·c → (z/h)·c = z · (Λ cos tp)/(Λ sin tp cos tp) = z/sin tp. So L → z/sin tp, the flat entrance-face path z/|n̂_z|. ✓
- Point just inside its own working facet (d → k c⁻, z ∈ [0, h)): s1 → 0⁺, z1 → z ∈ [0, h), floor = 0, L → 0. ✓

**Mean-gain identity** (independent version of the ledger's "analytic sawtooth mean-gain identity"): for emission depth z ≥ h and x uniform over a period, u ≡ s1/c is uniform on (0, 1] and

L = c·( u + floor(z/h − u) )  ⇒  ⟨L⟩ = c·(z/h − 1/2) = z/sin tp − c/2 ,

i.e. the grooves shorten the mean escape path by exactly c/2 = Λ cos tp / 2 relative to the flat face, independent of z (for z ≥ h). Any test asserting this identity is asserting the same geometry derived here.

### 3.4 Facet-event signs, bands, and ties

Define signed plane functions

W_k(r) = n̂·r − kΛ cos tp                 (working) R_k(r) = b·r − kΛ sin tp                 (relief).

On a physical working facet, material is on `W_k <= 0`; on a physical relief facet, material is on `R_k >= 0`. Therefore, for ray direction d,

working exit:  n̂·d > 0       working entry:  n̂·d < 0 relief exit:   b·d < 0        relief entry:   b·d > 0.

These signs are independent of period index. Plane intersections outside `0 <= z <= h` are not surface events: below the valley (`z > h`) both sides are material; above an apex (`z < 0`) both sides are vacuum.

If the relevant plane rate is zero, the ray is tangent to that facet family and cannot cross it. It may still reach the other family or a shared apex/valley. At an endpoint tie, the physical rule follows the closed-material predicate: compare states at `q − epsilon*d` and `q + epsilon*d`; accept only an actual state change. A material--material or vacuum--vacuum contact is not an event. Strictly forward `s > epsilon` plus one post-re-entry nudge prevents the same boundary from being returned indefinitely.

### 3.5 Electron transport through vacuum

Let material collision distance `X` be exponential with rate `lambda^-1`. If a facet truncates a flight after material distance `a` and no collision has occurred, then

P(X − a > y | X > a) = exp(−y/lambda).

Thus the residual material free path has the original exponential law. Vacuum contributes zero collision hazard, so discarding the old residual and sampling a fresh exponential after re-entry is statistically exact under the transport kernel's local-rate model. Vacuum also has no stopping or scattering:

E_after = E_before,       d_after = d_before.

The code's time coordinate is `c*t` in Å. Since speed is `beta*c`,

Delta(c*t) = c L_vacuum/(beta*c) = L_vacuum/beta  [Å].

Only material intervals may enter segment arrays. A re-entry is not a permanent exit and must not increment back/side/transmission counters.

### 3.6 Coherent and bremsstrahlung Beer--Lambert paths

For supported photon direction n̂, `b·n̂ = 0`; relief-plane coordinate stays constant. At working planes, `dW_k/ds = n̂·n̂ = 1 > 0`, always the outward working-facet sign. After the first physical outward crossing, later working crossings have lower z and the ray cannot acquire an inward sign; relief crossings are impossible. Hence no material re-entry exists for this direction and `escape_distance_ang` is the complete material path.

For attenuation coefficient `mu(E)` in Å^-1, both coherent and bremsstrahlung terms must use

T_abs(E) = exp[−mu(E) L_esc].

Grooving changes `L_esc` only. Coherent amplitudes/resonance kinematics and bremsstrahlung cross sections remain unchanged. Bremsstrahlung emission from one material segment retains its factor `n_i L_segment (d sigma_i/dk)/(4 pi Ne)`; vacuum legs supply neither `L_segment` nor optical depth.

### 3.7 Finite photon footprint: independent edge scale

Electron launch and arbitrary-direction transport must use the finite prism. Treating photon grooves as periodic is a separate approximation. For an emitter at `(x,z)` and supported `n̂ = (cos tp,0,−sin tp)`, horizontal travel before groove escape is

Delta x = L_esc cos tp.

The downstream `+x` side face wins whenever

W/2 − x < Delta x.

At fixed z, the affected edge strip therefore has width `Delta x`, not generally one groove period. From §3.3,

L_esc = z/sin tp − delta,       0 <= delta < Λ cos tp,

so

Delta x = z cot tp − delta cos tp = z cot tp + O(Λ).

For emitters uniform in x, the ignored-side fraction at fixed z is

f_edge(z) = min(Delta x/W, 1) = min(z cot(tp)/W + O(Λ/W), 1).

Thus `O(Λ/W)` is only the phase-scale correction to the edge width. It is not the total finite-side error unless all relevant emission depths are `O(h)`.

## 4. Implementation comparison (read after §3 was frozen)

### 4.1 Geometry

- `blazed_groove_spec`, `surface_depth_ang`, `in_material`, `entry_points`, and `escape_distance_ang` match §§3.1--3.3 symbolically, including facet branches, signs, bands, and the flat-profile limit.
- `first_surface_event` uses working sign multiplier `+1` and relief multiplier `−1`, exactly matching §3.4. Zero plane rate is skipped. Interior random-ray transitions covered by the focused reference-march tests pass.
- Exact valley handling now uses `band_tol = 32*eps_machine*max(spacing, depth)` and admits candidates only inside `[-band_tol, h + band_tol]` before applying the two-sided material predicate. Re-check used:

Λ = 2 Å, tp = 0.61 rad h = 0.9390993563190676 Å x_valley = h tan(tp) = 0.6563542536839531 Å p = (x_valley, 0, h + 1 Å), d = (0,0,−1)

Exact geometry exits through the valley at `s = 1 Å`; implementation now returns exactly `1.0 Å`. Here `band_tol = 1.4210854715202004e−14 Å`, while the ray--plane result's former overshoot is one ulp.
- Independent exclusion probe displaced the nonphysical working-plane intersection by `1e−8 Å`, about `7.0e5*band_tol`, beyond the valley. The implementation rejected that candidate and returned the physical relief event `1.0000000148848758 Å`, matching the independently calculated distance exactly. Thus the tolerance does not admit materially out-of-band intersections.
- Tangencies away from facet endpoints remain skipped; endpoint state changes remain governed by the two-sided predicate.

`escape_distance_ang` returns the next working plane rather than zero for an emitter exactly on a working facet because `s1 = c − mod(d,c)` is in `(0,c]`. Its documented limit is from just inside, and radiating segment midpoints are interior, so this separate measure-zero convention does not change the verdict.

### 4.2 Electron transport

`simulate_trajectories` compares sampled collision/prism/layer distances with the exact surface exit, truncates the material segment at the first event, applies material stopping and `step/beta`, and suppresses elastic scattering at a facet. It then searches for an entry along the unchanged direction. On re-entry it:

- copies post-material energy into the vacuum diagnostic;
- leaves energy and direction unchanged;
- advances `clock` by `distance/beta_from_keV(E)`;
- nudges into material by a period-scaled epsilon;
- resamples the material collision distance on the next iteration;
- does not increment exit counters.

Permanent no-reentry rays increment entrance-face exit count. Vacuum arrays are separate from `r_mid`, `L_ang`, `E_keV`, and `v_hat`; spectrum kernels see only material segments. This matches §§3.5--3.6. The exponential memorylessness argument is exact for the kernel's locally exponential free-path model.

### 4.3 Coherent and bremsstrahlung radiation

Both spectrum functions validate the supported direction and single-slab scope. `mc_cxr_spectrum` and `mc_brem_spectrum` call the same `escape_distance_ang`; both form `exp(−mu*L_esc)`. Vacuum legs never enter either segment sum. Coherent amplitude/resonance expressions and bremsstrahlung cross sections are unchanged. Supported-direction no-reentry proof in §3.6 therefore matches implementation.

### 4.4 Finite-footprint decoupling

Transport correctly honors finite launch misses, material side exits, and a side face encountered before vacuum re-entry. Both photon kernels deliberately ignore stored finite dimensions when `groove` is present, producing the same periodic result with or without footprint metadata.

That decoupling remains intentionally implemented. The correction design, ledger row, and coherent-spectrum docstring/comment now state the independently derived magnitude:

f_edge(z) = min(L_esc cos(tp)/W, 1) = min(z cot(tp)/W + O(Λ/W), 1).

They explicitly identify `O(Λ/W)` as only the periodic-phase correction and retain the `min(...,1)` cap. Runtime still gives `groove` precedence over stored finite dimensions in both photon kernels, so the documented approximation and implemented laterally periodic model now agree. Example magnitude remains: `Λ = 2 μm`, `W = 5 mm`, `tp = 45°`, `z = 20 μm` gives affected fraction `4e−3` (0.4%), while `Λ/W = 4e−4` (0.04%) is only the smaller phase correction.

### 4.5 Checks run

- Current-tree focused CPU re-check: three endpoint/reference tests passed (`3 passed, 30 deselected`).
- Independent numeric probe confirms the exact one-ulp valley event and materially out-of-band rejection described in §4.1.
- Supplied remote evidence for the exact fix snapshot: `tests/montecarlo/transport/test_groove.py + tests/plots/test_trajectories.py` — 34 passed; broader suite — 192 passed; typecheck and scoped Ruff clean.
- No heavy Monte Carlo, GPU job, or external comparison was run.

## 5. Adjudication

Analytic profile, ordinary facet signs, flat limit, supported-direction no-reentry, material/vacuum state machine, clock units, energy/direction invariance, material-only radiation, and coherent/bremsstrahlung Beer--Lambert paths match.

The two former discrepancies are corrected: closed facet-band endpoints use a geometry-scaled roundoff tolerance without accepting materially out-of-band planes, and finite-side documentation carries the missing `L_esc cos(tp) = z cot(tp) + O(Λ)` leading term while preserving intentional periodic runtime behavior. No divergent factor, sign, unit, exponent, or convention remains in the reviewed claim.

**Verdict: `rederived`.**

Suggested ledger edit (task owner applies): change status from `unverified` to `rederived`; link this write-up and the endpoint/edge re-review evidence. Do not mark `signed-off`.

## 6. Re-verification: exact per-electron grooved core (#972), 2026-11-10

**Scope:** `transport/cores.py::make_cpu_transport_core` with `grooved=True,
per_electron=True`, `core_geometry.py::_groove_reentry_scalar`, the signature
adapter, `grooved_batches.py::GroovedVacuumCore`, `beam_entry.py::initial_beam_positions`,
and `host_pass.py` routing. Paths are under `src/pyrite/montecarlo/`.
Sections 1–5 retain the historical review. This section supersedes the unfinished
#972 requirements with a separate fresh-context verification.

### 6.1 Independent derivation

This addendum begins from the ledger's elementary periodic ray–plane source
and the public geometry/core docstrings. It covers the exact CPU specialization
with `grooved=True, per_electron=True`; it does not re-certify photon scoring,
lockstep physics, or GPU parity. The verifier did not implement this change.

Put $s_t=\sin\theta$, $c_t=\cos\theta$, period $\Lambda>0$, and
$h=\Lambda s_t c_t$. Let $q=x\bmod\Lambda$. Intersecting the two facet families
with $z$ gives the independent surface expression

$$
f(x)=\begin{cases}
q c_t/s_t,&0\le q\le\Lambda s_t^2,\\
(\Lambda-q)s_t/c_t,&\Lambda s_t^2<q<\Lambda.
\end{cases}
$$

The two branches meet at $h$ and vanish at periodic apexes. Material is
$f(x)\le z\le T$ within any finite lateral footprint. For a unit direction
$\mathbf d$ and current position $\mathbf p$, the working and relief candidates
are respectively

$$
\ell_{w,k}=\frac{k\Lambda c_t-(c_t p_x-s_t p_z)}{c_t d_x-s_t d_z},
\qquad
\ell_{r,k}=\frac{k\Lambda s_t-(s_t p_x+c_t p_z)}{s_t d_x+c_t d_z}.
$$

Discard zero denominators, nonpositive distances, candidates outside the facet
depth band $[0,h]$, and candidates lacking the requested two-sided material
transition. The earliest remaining candidate is the physical surface event.
Prism back/side exits compete by the same positive ray distance. In vacuum,
any permanent prism exit must beat a later material entry; with no future
entry the history ends, whereas a later entry continues the same history.

These expressions have length units: unit-vector dot products and trigonometric
factors are dimensionless. Forward distances are positive for either direction
of lateral motion; integer plane indices range over both signs. A tangent has
no crossing unless another facet changes the material predicate. Boundary ties
need a declared numerical convention, rather than a sign reversal.

For a vacuum leg of length $\ell$, independent mechanics requires

$$
\mathbf p_{\rm next}=\mathbf p+\ell\mathbf d,\qquad
E_{\rm next}=E,\qquad \mathbf d_{\rm next}=\mathbf d,\qquad
(ct)_{\rm next}=ct+\ell/\beta(E),
\quad \beta(E)=\sqrt{1-(1+E/(mc^2))^{-2}}.
$$

The stored clock is a length (Angstrom, $c=1$), kinetic energy is keV,
and vacuum generates no stopping, scattering, or radiating material row.
A groove exit followed by entry is not a permanent electron exit. Its geometry
event count must persist for that history across every continuation; the
material-step count advances only for material rows. Independent bounded
counters prevent either physical-row truncation or an infinite geometric loop.

A continuation is a partition of a history, not a new history. All state that
can affect subsequent output must survive it: position, direction, energy,
clock, optical depth, RNG/event indices, material and groove-event counters.
If temporary output capacity is exhausted, replay must restore the complete
pre-attempt state and replace, rather than accumulate, provisional rows and
vacuum legs. Shrinking electron batches or increasing row capacity must leave
physical output and per-history exit counts unchanged.

For groove phase sampling, each electron needs a uniform lateral phase
$U_e\Lambda$ keyed by the root seed and global electron index $e$ in a dedicated
namespace. A block starting at $e_0$ must receive the identical sequence slice
$\{U_e:e_0\le e<e_0+N\}$; neither local row number nor output reservation can
enter the key. For a flat-face starting coordinate $x_0$, relief entry follows
$\ell_0=(-x_0s_t)\bmod(\Lambda s_t)$ and
$(x,z)=(x_0+\ell_0s_t,\ell_0c_t)$. As $\Lambda\to0$ at fixed nondegenerate
$\theta$, $h\to0$, entry depth and groove modulation vanish and the physical
geometry tends to a flat entrance face. Endpoint tilt angles remain excluded.


### 6.2 Comparison, correction and numeric evidence

The plane distances and working/relief signs agree with the compiled scalar
facet search. `_groove_reentry_scalar` compares the next entry against a finite
side exit; back-face exit cannot precede a later entry in the groove band
because the slab back face is below that band. A permanent no-entry history is
backscattered; a side that wins is classified as side exit. An entry creates a
vacuum diagnostic, rather than a radiating material row or a permanent exit.

`cores.py` initializes `surface_events` and `zero_surface_events` as
per-electron arrays (lines 359–360), resets each batch's vacuum count (375),
and charges no material step for a successful re-entry (1227). The separate
surface bound raises above `max_steps` (999–1001); the repeated zero-length
bound raises at two consecutive events (1004–1006). A shallow hand-placed
per-electron ray with direction proportional to $(1,0,0.02)$ independently
confirmed the surface-limit exception. Counters belong to each history, so a
preceding electron cannot spend a following electron's budget.

Here “material steps” means charged non-re-entry steps; a material fragment
ending in successful re-entry is uncharged. It is still retained as a material
row and contributes stopping and radiation. This refines the row/budget
wording in §6.1: a material-row count is not the termination counter.

At re-entry, `cores.py` records the post-material energy, preserves direction,
and adds precisely the vacuum length divided by its constant-energy beta
(lines 983–998). It suppresses elastic scattering at the facet. The next
material flight resamples its locally exponential free path, as in the
historical core (§4.2); it need not retain a previously sampled optical depth
across this boundary. There is no random draw during the vacuum leg itself.
Energy-limited material substeps retain their open-flight state. CPU grooved
continuation is refused by host routing; batch replays restart the complete
history, so the hypothetical continuation requirements in §6.1 do not assert
an implemented CPU continuation feature.

An independent elementary geometry point uses $\Lambda=20000$ Å,
$\theta=45^\circ$, $\mathbf p=(3000,0,8000)$ Å and
$\mathbf d=(1,0,0.1)/\sqrt{1.01}$. The working exit satisfies $x=z$:

$$
\ell_w=\frac{5000\sqrt{1.01}}{0.9}=5583.26423395605\ {\rm \mathring A},
\qquad x_w=z_w=8555.555555555555\ {\rm \mathring A}.
$$

The next relief plane is $x+z=20000$ Å, giving
$\ell_{\rm vac}=2639.3612742337705$ Å and entry
$(11181.818181818182,0,8818.181818181818)$ Å. The compiled per-electron core
returned these endpoints at displayed precision, one vacuum leg, and two
material rows terminating in transmission through the $T=50000$ Å back face,
with `max_steps=1`. The post-entry displacement adds only the declared
period-scaled nudge.

**Discrepancy found and corrected by the task owner.** The original host call
passed `max_segments = Ne * max_steps` into the batch driver's global retained
row guard (`batching.py`, near line 556). This confused the charged material
step budget with the larger material-row count. The direct core test supplied
extra row slots and therefore missed the host error. An ephemeral probe using
the numeric state above, dilute carbon $n=10^{-12}$ Å$^{-3}$, 60 keV,
`elastic_model='sr'`, seed 9, cutoff $10^{-6}$ keV and `max_steps=1` reached
`RuntimeError: segment buffer exhausted` in public host routing. Only the
entrance position/direction samplers were patched to inject the hand-placed
state; the geometry, transport core and batch driver were real.

The owner's correction in `host_pass.py::_run_per_electron` (near line 802)
uses `2 * Ne * max_steps` for grooved runs. Independently, at most
`max_steps` successful re-entries produce uncharged fragments, and at most
`max_steps` other rows, including the terminal row, are possible before the
charged-step limit. Therefore this bound admits every non-raising history.
The identical probe after correction returned two material rows, one vacuum
leg, and one transmitted electron. The new
`test_per_electron_host_keeps_uncharged_reentry_segments` also passed in the
verifier's own run. No implementation was edited by this verifier.

`GroovedVacuumCore` snapshots position, direction, energy and clock, grows
vacuum capacity using the required count, and restores the snapshot before
retry (lines 65–90). The core reconstructs its RNG, material/surface counters
and straggling accumulator for every full-history attempt. The adapter replaces
its entry for the same batch start (93), so a segment-capacity replay cannot
leave duplicate vacuum legs; final joining sorts batch starts (99). The generic
batch driver likewise restores its entry snapshot before segment replay and
adds exit counts only after acceptance. This matches the independently derived
replay contract, including batches that shrink after capacity growth.

`initial_beam_positions` uses draw zero of global electron $e$ in
`child_stream_root(seed, 3, 2)` when `counter_groove_phase=True` (line 97);
`host_pass.py` selects that mode for the per-electron route (488). Finite spot
and Twiss entrances retain their separately addressed distributions. The
signature adapter forwards groove geometry and the extra vacuum counter without
altering numerical values. Exact grooved CPU routing is selected explicitly;
`auto` retains lockstep and CUDA grooves remain refused. The small-period flat
limit has the same exit counts and convergent per-electron material lengths;
it is a numerical limit, not an assertion of bitwise equality at finite period.

### 6.3 Filters and verification limits

| Filter | Result and evidence |
| --- | --- |
| Dimensions and boundary units | **Pass.** Source geometry, entry sampler, kernel and retained positions/lengths use Å; kinetic energies use keV; clock uses Å with $c=1$; beta and directions are dimensionless. The mixed geometry tuple's trigonometric entries and finite-footprint flag are dimensionless, despite its shorthand `[Ang]` docstring annotation. No source table or unit-converting loader is involved. |
| Limits | **Pass.** Period vanishes at fixed nondegenerate tilt; entry relief depth vanishes and flat geometry is recovered. Endpoint tilts remain explicitly excluded. No-entry rays escape permanently; finite sides compete with later entry. |
| Signs and conventions | **Pass.** Positive ray distance, working outward normal, relief inward normal, depth-positive sample frame, permanent top/side classification, and vacuum clock direction agree. |
| Notation and normalization | **Pass.** Ledger, geometry docstrings, this derivation and kernel identify the same period/depth and material-only rows. No per-volume, solid-angle or population normalization factor enters this geometry claim. Vacuum diagnostics are a separate population of legs, not extra radiating histories. |
| Reference values and citations | **Pass / not applicable to tables.** Source is elementary geometry, with no paper figure, page or transcribed material datum to certify. The numeric facet point above is derived directly from the plane equations, independent of geometry helpers. Carbon data only supply negligible interaction rates in the bookkeeping probe. |
| Tolerances | **Pass within the stated floating-point model.** Geometry's depth-band allowance is $32\epsilon_{\rm machine}\max(\Lambda,h)$; the post-entry nudge is $\max(64\epsilon_{\rm machine}\Lambda,2\times10^{-12}\Lambda)$. These resolve boundary roundoff and have length units, rather than source uncertainty. Clock/direction anchors use relative $2\times10^{-13}$ roundoff tolerances; the flat-limit anchor allows aggregate $10^{-5}$ and per-electron $10^{-3}$ relative path differences at finite $\Lambda=0.01$ Å. These are finite-period convergence checks, not a source-error estimate. |
| Artifact validity and evidence scope | **Pass for current local evidence.** Review started at existing commit `833b4829f`; the owner committed the host row-bound correction and regression as `9b1977ff4`. The post-correction probe and regression result below certify that correction. The commands below and ephemeral probes were executed against that tree. No new external-code, CPU/GPU or FP32/FP64 comparison is claimed; historical remote counts in §4.5 are not used as evidence for #972. |

Verifier-executed focused checks, with `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run
pyrite-dev test`:

- `tests/montecarlo/transport/test_groove.py -k 'exact_per_electron or material_segments_and_vacuum_invariants or flat_face_limit or first_surface_event'`: **6 passed**, 42 deselected.
- Same module with `-k 'vacuum_leg_preserves or surface_event_exhaustion'`: **3 passed**, 45 deselected.
- Same module after the host correction, `-k 'host_keeps_uncharged'`: **1 passed**, 48 deselected.
- Ephemeral `uv run python` probes reproduced the analytic endpoints, the
  per-electron surface-event guard, and the host failure/correction above.

The first two focused runs reported the existing missing-BremsLib fallback
warning; they validate transport geometry and vacuum bookkeeping, not coupled
radiative physics. The task owner additionally reports 116 passed and 28 skipped
for the groove, per-electron and block-transport modules after the correction.
Block/phase/capacity evidence comes from the existing exact-array equality
anchors in `test_block_transport.py`; this verifier inspected their assertions.
The task owner owns the final docs build, rendered-math QA and documentation
checks; those checks must complete before publication of this write-up.

### 6.4 Adjudication

- **Claim**: `blazed-groove-geometry` — `src/pyrite/montecarlo/transport/cores.py::make_cpu_transport_core` (exact grooved per-electron CPU), with the geometry, adapter, phase and host owners above — elementary periodic ray–plane intersection.
- **Filters**: units **pass**; limits **pass**; signs/conventions **pass**; normalization **pass**; boundary units **pass**; reference transcription **not applicable**; tolerances **pass**; evidence validity **pass** within §6.3's local scope.
- **Re-derivation**: **matches after correction** — the divergent retained-row ceiling was $N_eN_{\max}$; the corrected groove ceiling $2N_eN_{\max}$ follows from separate charged-step and re-entry-event bounds. No geometric factor, sign or vacuum-clock discrepancy remains in the reviewed path.
- **Verdict**: **`rederived`** for the corrected exact per-electron path.
- **Write-up**: `docs/validation/geometry/blazed-groove-geometry.md`.
- **Suggested ledger change**: task owner may record this corrected #972 verification and set the claim to `rederived`, or `anchored` after its listed regressions pass the required project checks. Add the host uncharged-fragment regression to the anchor list. Never apply `signed-off` from this agent review.
