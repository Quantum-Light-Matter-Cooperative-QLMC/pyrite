# Physics review — Slice 1 follow-up geometries (issue #23)

Reviewer role: independent physics review under `.claude/skills/physics-review/SKILL.md`.
Scope: the two follow-up oracle passes recorded in the task `README.md`
("Broken-mirror-symmetry pass" and "Near-normal polar-angle pass",
2026-08-21, `remote-host` box) and their raw evidence
(`evidence/oracle-report.broken-symmetry.json`,
`evidence/oracle-report.near-normal.json`), reviewed against the actual
`mc_spectrum` implementation
(`src/pyrite/montecarlo/spectrum/lines.py`), the geometry mapping
(`src/pyrite/instrument/geometry.py`, `src/pyrite/instrument/model.py`,
`src/pyrite/montecarlo/geometry.py`), and the crystal refractive-index table
(`src/pyrite/materials/crystal.py`). Builds on, and in one place overturns,
`evidence/physics-review-slice1.md`.

**This is a reviewed recommendation, not a sign-off.** Nothing here marks any
ledger row `signed-off`; only a human does that. No implementation file, the
oracle harness, the task `README.md`, the GitHub issue, or any ledger row was
modified. Independent verification used read-only scratch scripts and direct
calls into the installed `pyrite` package, outside the repository's tracked
tree (`/tmp/.../scratchpad/probe_near_normal.py`,
`.../probe_hkl_split.py`, plus inline `uv run python3 -c` probes); nothing was
committed.

---

## Executive verdict

**The near-normal result is real, but its own stated explanation is wrong,
and the true mechanism is worse news for a fixed or auto-sized angular grid
than either the task doc or the first review anticipated.** I reproduced the
5–25x amplification directly against `mc_spectrum` and traced it to its
source: `polar_deg=20` matched with `Slab(tilt_deg=20)` does not test
"near-normal observation" in the sense either analysis intended. It places
the detector's central ray **exactly on the crystal's reciprocal vector `g`**
in the sample frame (verified: the sample-frame angle between the detector
centre and `g` is `|polar_deg − tilt_deg|`, which is `0°` here, against `30°`
at baseline) — a coordinate degeneracy, not a generic grazing/near-normal
angle. But working the actual angular derivatives through from the source
formulas at that point shows **both known mechanisms (the Doppler
denominator `D = 1 − β cosψ` and the PXR detuning `g² + 2ω(n̂·g)`) predict
*less* angular sensitivity there, not more** — the opposite of the measured
result, and also a direct refutation of the README's own `γ²`/`1/(1−β)`
hypothesis (this system is only mildly relativistic, `β = 0.328`, and `D`
never approaches zero at any tested `ψ`). Tracing the actual reconstructed
spectrum (not just its centroid/flux summary) at the near-normal centre pixel
shows why: the line the harness is measuring there is **not the same physical
feature as at baseline**. The `v0`-driven ~1648 eV resonance is present but
**20x weaker** than a second, dominant feature at ~150–300 eV, sourced by a
small (~2%), large-scattering-angle tail of segments whose in-medium
resonance lands inside HOPG's strongly dispersive near-edge band (`|1 − Re n|
~ 10⁻² to 10⁻¹` there, against `~10⁻⁵` to `10⁻³` everywhere the rest of the
model — including the resonance solver's own fixed 3-pass iteration and
both review's angular-derivative arguments — assumes). That tail is not
larger near-normal than at baseline by segment count (baseline actually has
*more* segments in that band, 5.1% vs 2.0%); what differs is that near-normal
gives it **disproportionate weight**, almost certainly because its detuning
sits close to its own separate Bragg-type zero there, while baseline's
same-sized tail does not. **This is a second, independent resonance crossing
close to the tested geometry, invisible to any smooth first/second-order
angular-gradient model** — not a steeper version of the same physics either
review analyzed. Recommendation: **do not treat "polar coefficient
`β sinψ/(1−βcosψ)`" as the governing quantity for angular-grid sizing at all.
A fixed default `angular_shape`, and the review's own `(3, 37)`-style
auto-sizing-from-a-slope-probe idea, are both unsafe as stated — not just
"needs tuning per geometry" but structurally unable to detect this failure
mode, because a 3-direction slope probe assumes one smooth resonance and this
regime has (at least) two.** The broken-symmetry pass, by contrast, checks
out as a clean instance of the mechanism the first review already predicted;
no artifact found there.

---

## 1. The near-normal geometry is a coordinate degeneracy, not "near-normal"

### 1.1 What `polar_deg` and `tilt_deg` actually control

`PlanarPose.from_observation` places the detector centre at spherical
`(distance, polar_deg, azimuth_deg=0)` in the **lab** frame, with local `+z`
pointing from source to detector (`src/pyrite/instrument/model.py:89-139`).
`tilted_geometry` (`src/pyrite/montecarlo/geometry.py:256-298`) then maps the
fixed lab beam (`+z_lab`) and this detector direction into the **sample**
frame — the frame `mc_spectrum` actually receives as `n_hat`/`beam_uvw` — via
one rotation `R` taking sample `+z` (the crystal `c`-axis / slab normal, and
by default the reciprocal vector `g`, since `g ∥ n` unless
`recip_miscut_rad` is set) onto the lab-frame tilted normal. Its own
docstring states the key invariant directly: *"For ta = 0 the detector's
sample-frame polar angle is simply theta_obs − tilt_polar; the lab-frame
quantity `1 − v0.n_hat` ... is tilt-invariant."*

I verified both halves of that claim from the rotation itself (not just the
docstring), and derived a second consequence the docstring doesn't spell out:
**the sample-frame angle between `n̂_center` and `g` is exactly
`|theta_obs − tilt_polar| = |polar_deg − tilt_deg|`, independent of
`distance_mm`, pitch, and (for `tilt_azim_deg = 0`) independent of anything
else.** Baseline: `|60 − 30| = 30°`. Near-normal pass: `|20 − 20| = 0°` —
*exactly* on-axis with `g`.

Numerically, from `tilted_geometry(polar_deg, tilt_deg, 0)`:

```
baseline (60°, 30°):      n_hat_center (sample frame) = (0.5, 0, 0.866) → angle to g = 30.0°
near-normal (20°, 20°):   n_hat_center (sample frame) = (0.0, 0, 1.0)   → angle to g =  0.0°  (== g direction)
```

This was **engineered by the test's own parameter choice** ("a matching
`Slab(tilt_deg=20.0)`", per the README) — not a generic property of
`polar_deg = 20°`. A `polar_deg = 20°` test with any *other* tilt (e.g.
`tilt_deg = 30°` kept, giving angle-to-`g` `= |20−30| = 10°`) would **not**
sit at this degeneracy. This alone means "near-normal is the worst case" is
not a safe generalization from this one run: the run tests one specific
coincidence (detector on-axis with `g`), conflated with "near-normal to the
beam" only because the test chose to match the two.

### 1.2 Both cited mechanisms predict the *opposite* of what was measured

**Doppler/kinematic term.** `ψ_phys` — the beam-to-observation angle that
sets `D = 1 − β cosψ` in `_in_medium_kinematics`
(`src/pyrite/montecarlo/spectrum/lines.py:519`, vacuum start
`denom = 1 − v_dot_n`) — is **exactly `polar_deg`**, and (like the `g`-angle
invariant above) is **tilt-invariant**: `v0·n̂ = z_lab · n̂_lab = cos(θ_obs)`
regardless of `tilt_deg`, because tilting the sample is a passive rotation
applied identically to both vectors. At `β = 0.3283` (30 keV,
`beta_from_keV`):

```
                       D = 1 − β cosψ     f(ψ) = β sinψ / D   [rad⁻¹]
baseline   (ψ=60°):    0.8359             0.340   (review's own figure)
near-normal(ψ=20°):    0.6916             0.162
```

`f(ψ)` is the review's own quoted coefficient (`d ln E_res/dθ`, and the
flux log-derivative is `−2 d ln D/dθ = 2f(ψ)`). It is **smaller** at 20° than
at 60°, not larger — the kinematic mechanism predicts *reduced* sensitivity
near-normal, exactly as the first review said, and the README's own
hypothesis that this is secretly a `γ²`-type blow-up doesn't survive contact
with the formula: `D` is bounded well away from zero at this `β`
(`D ∈ [0.67, 1.33]` over all `ψ`; a Cherenkov-type collapse needs `β → 1`,
i.e. `γ ≫ 1` — here `γ ≈ 1.06`). **The README's `1/(1−β) ~ γ²` explanation is
refuted, not just unverified.**

**Detuning/PXR term.** I re-derived `n̂·g = |g| cosθ_g` where `θ_g` is the
angle from §1.1. At the near-normal centre, `θ_g = 0` is by construction the
*minimum* of `θ_g` over the sphere, so `cosθ_g` is *stationary* there
(`cosθ_g ≈ 1 − θ_g²/2`, second-order in angular offset in **every** tangent
direction, not just azimuth) — a coordinate fact independent of any specific
physics. I confirmed this numerically by evaluating `omega_res`, `D`, and
`detuning = g² + 2·omega_res·(n̂·g)` from the actual rotated geometry across a
9-pixel column sweep at both scenes (`reciprocal_g_vector((0,0,2), lattice)`,
`|g| = 1.8725 Å⁻¹`):

```
baseline, per-pixel (1.074°) column step near centre:  ΔD/D ≈ 0.64%/px,  Δdetuning/detuning ≈ −0.64%/px
near-normal, same step near centre:                    ΔD/D ≈ 0.31%/px,  Δdetuning/detuning ≈ −0.16%/px
```

Both fractional sensitivities are **smaller** at near-normal, consistent with
§1.2's kinematic result and with the pole argument. **Every mechanism either
review analyzed says this geometry should be *quieter* than baseline, not
5–25x louder.** That is the actual discrepancy this follow-up needs to
resolve, and neither review's formula resolves it.

---

## 2. What is actually happening: a second resonance crosses in from the crystal's dispersive band

Since the line-centre kinematics don't explain the result, I inspected the
**actual reconstructed spectrum** at the near-normal centre pixel (not just
its centroid/flux summary), using `run_case_directions` directly — the same
technique both reviews used, isolated to one `hkl` family at a time
(`(0,0,2)` only) to remove the second reflection family
(`hkl_list = [(0,0,2),(0,0,-2),(0,0,4),(0,0,-4)]` is `api.build_case`'s
default for `hopg`) as a confound:

```
baseline (60°,30°), (002) only, column sweep:
  col=0..8 peak: 1288, 1282, 1273, 1264, 1258, 1249, 1240, 1234, 1225 eV   -- smooth, monotone, matches §1 of the first review exactly

near-normal (20°,20°), (002) only, column sweep:
  col=0..8 peak:  274,  277,  277,  280,  280,  172,  283,  169,  169 eV   -- NOT smooth, NOT monotone, pinned near 150-300 eV
```

The near-normal "peak" is not tracking a shifting resonance at all; it is
bouncing between grid points in a band nowhere near the ~1648 eV I compute by
hand-solving `omega_res = v0·g / (1 − v0·n̂)` for the *unscattered* beam
direction at this geometry (matches within 0.1 eV of the code's own 3-pass
fixed-point solve, which converges cleanly there — verified by hand-running
`_in_medium_kinematics`'s iteration). Checking the spectrum directly at the
centre pixel (4000-electron probe, same scene) confirms both features exist
simultaneously, at very different strengths:

```
window  100-400 eV:  peak 4.96e-11 at 277 eV;  integrated flux 4.50e-9
window 1400-1900 eV: peak 2.41e-12 at 1648 eV; integrated flux 2.60e-10   (the v0-predicted line -- present, but 20x weaker)
```

**The dominant contribution to this "line" is a different physical feature
than at baseline: a resonance sourced by large-scattering-angle segments
whose in-medium resonance energy falls in HOPG's strongly dispersive
near-edge/optical band.** Directly tabulating `refractive_index('hopg', E)`
confirms this band is real and matches: `Re n` departs sharply from 1 below
~300 eV (`0.870` at 50 eV, `0.966` at 100 eV, `0.986` at 150 eV) and has a
cusp exactly at the carbon K-edge, `284 eV` (`Re n`: `0.9986 → 1.0068 →
1.0005`, `Im n` jumping ~10x) — bracketing the observed 169–283 eV peak
locations almost exactly. This is the same band `_in_medium_kinematics`'s own
docstring names as the danger zone for its fixed 3-pass solve ("carbon:
6.24–285 eV, peaking at 4.766 at 6.40 eV... Re n(v.n_hat) can approach unity,
denom collapses toward a spurious Cherenkov-like zero"), and where its
assumed contraction rate (`Re n = 1 − δ`, `δ ~ 10⁻⁵` to `10⁻³`, "only true in
the X-ray regime") does not hold — here `δ` is `10⁻² ` to `10⁻¹`, one to four
orders of magnitude larger.

I checked whether near-normal simply has *more* segments whose vacuum
resonance falls in this band (which would make this "just more of the same
tail," not a new mechanism). It does not: pulling the actual transported
`segments['v_hat']` for both scenes and evaluating the vacuum resonance
`E_vac = ħc·(v·g)/(1 − v·n̂_center)` per segment,

```
                       frac(E_vac in 6-285 eV band)   frac(E_vac <= 0, rejected)
baseline   (60°,30°):  5.1%                            2.7%
near-normal(20°,20°):  2.0%                            3.4%
```

Baseline has the *larger* danger-band population by count, yet its
reconstructed line is clean. What differs is not the tail's size but its
**weight**: at near-normal this small tail dominates the integrated flux
(§2, 17x by count-window integral, 20x by peak height); at baseline the
equivalent-sized tail is negligible against the strong, well-separated ~1258
eV line (its own column sweep never leaves the 1225–1288 eV band). The most
plausible reading — not independently confirmed to the same rigor as the
rest of this section, flagged as such — is that the near-normal geometry's
`g`-alignment happens to put *this specific subpopulation's* PXR detuning
close to its own separate zero (a second, accidental near-Bragg condition for
the large-angle-scattered tail, distinct from the well-behaved `v0`-centred
line), producing the same `1/detuning²` resonant enhancement the first
review identified as the dominant mechanism for the *main* line — just for a
different, narrower, geometry-sensitive slice of the population. Because that
slice is small and its resonance condition is sensitive to exactly which
scattering angles qualify, its position/strength swings sharply from pixel to
pixel (`169↔283 eV`, `centroid 545↔2025 eV` between neighbouring
representative pixels in the committed evidence) — which is exactly the
non-monotonic, non-smooth pattern in `evidence/oracle-report.near-normal.json`
that neither review's gradient formula can produce.

**This reframes the mechanism.** It is not "the polar coefficient is larger
near-normal." It is: **this specific geometry (`g`-aligned detector centre)
brings a second, normally-negligible resonance close enough to its own Bragg
condition that it can dominate the reconstructed line, and that condition is
inherently local/non-smooth in angle** — the opposite of a well-behaved
gradient that any grid density handles given enough tiles.

---

## 3. Broken-mirror-symmetry pass: sanity-checked, no artifact found

Lower priority per the task, checked briefly. I reproduced the centre-pixel
spectrum at `tilt_azim_deg = 45°` (same `hopg`/30 keV/`polar_deg=60`/
`tilt_deg=30` otherwise): peak at **1258 eV**, matching baseline exactly (`ψ_phys`
is tilt-*and*-azimuth-invariant by the same lab-frame dot-product argument as
§1.2, so this is expected), and the low-energy/high-energy flux split
(100–400 eV: 5.3e-8; 1000–1500 eV: 2.34e-7 — high-energy line dominant by
~4.4x) looks like baseline's clean single-resonance regime, not the
near-normal pathology. The reported azimuthal-error growth
(`azimuth_only` becomes the `(5,5)` worst offender, +2.5% vs baseline's
≤1.5% even at full span) is consistent with the first review's own mechanism
(mirror-symmetry breaking makes azimuthal offset first-order) and shows none
of §2's non-monotonic/bimodal signature. **No artifact found; treat this
pass's conclusion (anisotropic-grid recommendation does not survive
`tilt_azim_deg ≠ 0`) as confirmed.**

---

## 4. Consequences for the anisotropic-grid recommendation and Slice 2

The first review's `(3, 37)`-style anisotropic grid was already invalidated
by the broken-symmetry pass for a *geometric* reason (azimuthal error becomes
first-order once coplanarity breaks). This pass invalidates it again, for a
*different and more serious* reason:

- **(a) Anisotropic-by-formula is now actively unsafe, not just
  geometry-conditional.** The broken-symmetry failure is at least still a
  smooth, single-resonance problem — picking the *right* aspect ratio (e.g.
  isotropic instead of `(3,37)`) would fix it. The near-normal failure is
  not: the worst pixel at `(5,5)` (`corner_tl`, −14.14%) is not simply "the
  most polar-offset pixel," the response is non-monotonic column-to-column,
  and the underlying physical driver (a second resonance's detuning crossing
  near zero somewhere in the sampled angular range) has no reason to align
  with either grid axis. No aspect ratio derived from a smooth-gradient
  model can be trusted to catch it; only exhaustive-enough sampling (or
  detecting the multi-resonance condition directly) can.

- **(b) A fixed default `angular_shape` is unsafe; the auto-sizing probe
  needs to be more than a slope probe.** The task doc's §5 "transport once,
  probe 3 polar directions, fit `∂lnL/∂θ_pol`, auto-size `ax`" idea assumes
  the quantity being fit is a single smooth line's log-slope. §2 shows a
  geometry can host **two competing resonances at once**, with the
  reconstruction error dominated by whichever one a given pixel happens to
  catch — a 3-direction linear fit would not detect this at all (it would
  return a small, confident-looking slope from whichever resonance dominates
  the 3 sampled directions, and silently miss the other). **This evidence
  makes some form of mandatory per-run safety check necessary before Slice 2
  ships a default** — but the check needs to be able to flag "more than one
  resonance feature is present in this window" (e.g. compare a coarse
  direct-sample spectrum's shape against the tile-reconstructed one across a
  few widely spaced directions, not just fit one slope), not just fit a
  gradient magnitude. Costing and designing that check is now a precondition
  for Slice 2, not an optional refinement.

- **(c) The oracle harness's existing metric gaps (first review, §2.2b/c)
  are more disqualifying here than at baseline.** The window-integrated
  centroid (`_centroid_eV` over the full 100–4999 eV grid) cannot distinguish
  "one line shifted" from "two lines of different relative weight," and in
  this geometry it is measuring the latter — the reported `36–355 eV` mean
  centroid errors in the README table are not interpretable as a position
  error at all. **Do not use this harness's centroid/flux metrics to
  characterize *any* near-`g`-axis geometry without first adding a
  lineshape/multi-peak diagnostic** (first review's §2.2c already asked for
  this to gate interpolation; this evidence shows nearest-tile
  characterization needs it too, in this regime).

- **(d) The known degenerate-tile artifact (first review, §2.2d) reproduces
  here and is more misleading than at baseline.** `corner_br` reports exactly
  `0.00%` flux/centroid error at `(5,5)` in
  `oracle-report.near-normal.json` — the same `np.array_split(arange(9),5)`
  singleton-tile artifact the first review flagged, now sitting directly
  next to a `−14.14%` neighbour (`corner_tl`) in a report that otherwise
  reads as "every metric got worse everywhere." A reader who does not know
  about the array_split artifact could easily misread `corner_br`'s `0%` as
  evidence the failure is confined to specific axes rather than pervasive.
  Fix per the original recommendation (report over all pixels or flag tile
  occupancy) before this evidence is used to size anything.

- **(e) Single-seed statistics are a bigger concern here than at baseline.**
  §2's mechanism is sourced by a ~2% tail of large-angle-scattered segments
  whose contribution can dominate the total — that is inherently
  higher-variance than a bulk-population-driven line, even at
  `N_ELECTRONS=4000`. The specific magnitudes in the near-normal table (and
  in my own 400-electron scratch probes, which show the same qualitative
  bouncing at lower fidelity) should not be treated as precise until a
  multi-seed spread is measured. The *existence* of the second resonance and
  its rough energy band are robust (reproduced across probe electron counts
  from 400 to 4000 and two independent scratch scripts); its exact per-pixel
  percentage is not.

---

## Summary of actions for the implementation owner

1. Do not adopt the README's `1/(1−β) ~ γ²` explanation for the near-normal
   result; it is refuted by the actual `D = 1 − β cosψ` formula at this
   `β`. Replace it with §2's mechanism (a second, `g`-alignment-triggered
   resonance from the large-scattering-angle tail, sourced in HOPG's
   dispersive near-edge band) as the working hypothesis, flagged for
   independent confirmation of the detuning-crossing claim specifically (the
   one piece of §2 not verified to the same rigor as the rest).
2. Treat "detector centre aligned with `g`" (`|polar_deg − tilt_deg| ≈ 0`, for
   `tilt_azim_deg = 0`) as a distinct, named danger case, separate from
   generic "near-normal to the beam." Do not extrapolate this run's magnitude
   to near-normal geometries generally; a `polar_deg = 20°` run with
   `tilt_deg` *not* matched (e.g. kept at 30°, giving angle-to-`g` `= 10°`)
   would test the smooth-gradient near-pole regime this run does not, and is
   still worth running to separate "close to the pole" from "exactly on it."
3. Both anisotropic-grid recommendations (this task's own `(3, 37)` idea and
   any successor) are unsafe as a shipped default. The §5 auto-sizing-probe
   idea needs a lineshape/multi-resonance check added to its design before
   it can be trusted, not just a slope fit — cost that explicitly as part of
   its design, not as a follow-up.
4. Harness: the first review's metric-gap list (§2.2 of
   `physics-review-slice1.md`) applies with higher urgency here — a
   lineshape/multi-peak diagnostic is now needed to characterize nearest-tile
   error, not only to gate interpolation. Fix the degenerate-tile reporting
   (§4d above) before this evidence is cited further. Add multi-seed spread
   before trusting exact percentages in the near-normal table.
5. Broken-symmetry pass: no issues found: treat its conclusion as confirmed.

**A human must still review and sign off. Nothing in this document is a
ledger sign-off, and no `Validation:` marker or ledger row is created or
implied by it.**
