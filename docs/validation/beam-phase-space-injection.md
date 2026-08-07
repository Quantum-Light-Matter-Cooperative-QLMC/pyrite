# Courant–Snyder transverse beam injection

**Validation id:** `beam-phase-space-injection`
**Code:** `transverse.py::resolve_transverse_distribution`; `::sample_transverse`;
`montecarlo/transport.py::simulate_trajectories`;
`montecarlo/geometry.py::beam_frame_basis`
**Source:** Courant & Snyder, *Ann. Phys.* **3**, 1–48 (1958); standard
normalized-emittance convention (e.g. Wiedemann, *Particle Accelerator
Physics*).
**Status:** rederived in fresh context on 2026-08-07.

## 1. What the code claims to compute

From the ledger row and docstrings, before touching the implementation body,
the intended quantity is:

- A stored policy carries **normalized** emittance `ε_n` [mm·mrad], Twiss
  `β_T` [m] and signed `α` per transverse plane, because geometric emittance
  is not invariant across the swept `energy_keV` axis.
- Per case energy, resolve the **geometric** emittance
  `ε_g = ε_n / (βγ)_rel`, with `(βγ)_rel = sqrt(γ_rel² − 1)`,
  `γ_rel = 1 + E/(m_e c²)`.
- Per-plane second moments follow Courant–Snyder:
  `σ_x = sqrt(ε_g β_T)`, `σ_x' = sqrt(ε_g γ_T)`, `γ_T = (1+α²)/β_T`,
  correlation `⟨x x'⟩ = −α ε_g`.
- Per-electron draw: `u1, u2 ~ N(0,1)` i.i.d.,
  `x = sqrt(ε_g β_T) u1`, `x' = sqrt(ε_g/β_T)(u2 − α u1)`.
- `beam_frame_basis(beam_dir)` supplies the rotation whose transverse
  columns the slopes `(x', y')` are measured against, and is claimed to be
  exactly the identity when `beam_dir = +z`.
- Units: input emittance is mm·mrad, input `β_T` is m; internal/diagnostic
  units are mm·rad (emittance) and mm/rad (`β_T`), matching
  `beam_metrics.PlaneMetrics`.

## 2. Independent derivation

### 2.1 Normalized → geometric emittance

Liouville's theorem makes the *normalized* emittance
`ε_n = β_rel γ_rel · ε_g` an adiabatic invariant under longitudinal
acceleration (no space charge, no nonlinear forces — exactly the assumptions
stated in the docstring). Hence

```
ε_g(E) = ε_n / (β_rel γ_rel)(E),  (β_rel γ_rel) = sqrt(γ_rel² − 1),  γ_rel = 1 + T/(m_e c²).
```

This is the textbook definition (Wiedemann §8.2 / Courant & Snyder framework
extended to acceleration). At fixed `ε_n`, `ε_g` is monotonically decreasing
in `(βγ)_rel`, i.e. in the inverse ratio of `βγ` between two energies —
matches the stated limiting case.

### 2.2 Courant–Snyder second moments and the per-plane draw

Define the Twiss/Courant-Snyder invariant `βγ_T − α² = 1` (standard
convention, `γ_T` here is the *Twiss* gamma, distinct from the relativistic
`γ_rel` above — the two symbols collide notationally in the literature but
not in the code, which spells them `beta_twiss`/`gamma_twiss` vs
`beta_gamma`). The single-particle invariant is

```
ε_g = γ_T x² + 2 α x x' + β_T x'².
```

The Gaussian-equivalent beam matrix (second moments matched to this
invariant, standard result) is

```
Σ = ε_g [[ β_T   −α  ]
         [ −α    γ_T ]]  ,  i.e.  ⟨x²⟩ = ε_g β_T,  ⟨x'²⟩ = ε_g γ_T,  ⟨x x'⟩ = −α ε_g.
```

Take `u1, u2` i.i.d. `N(0,1)` and set

```
x  = sqrt(ε_g β_T) u1
x' = sqrt(ε_g/β_T) (u2 − α u1).
```

Then, since `⟨u1²⟩ = ⟨u2²⟩ = 1` and `⟨u1 u2⟩ = 0`:

```
⟨x²⟩  = ε_g β_T ⟨u1²⟩                                   = ε_g β_T
⟨x'²⟩ = (ε_g/β_T) ⟨(u2−α u1)²⟩ = (ε_g/β_T)(1+α²)        = ε_g (1+α²)/β_T = ε_g γ_T
⟨x x'⟩ = ε_g ⟨u1(u2−α u1)⟩ = ε_g(⟨u1u2⟩ − α⟨u1²⟩)        = −α ε_g.
```

All three moments match the target `Σ` exactly (not just to leading order):
this is the standard Courant–Snyder / Cholesky-style sampler, confirmed
numerically below to machine-precision Monte Carlo agreement (N=2×10⁷,
arbitrary `ε_g=3.7×10⁻³` mm·rad, `β_T=1.8` mm/rad, `α=−0.6`):

```
target <x^2>   = 6.660000e-3   sample = 6.664188e-3
target <x'^2>  = 2.795556e-3   sample = 2.796073e-3
target <x x'>  = 2.220000e-3   sample = 2.221557e-3
```

### 2.3 Sign convention on `α`

With `Σ_{12} = −α ε_g`, positive `α` gives negative `⟨x x'⟩` (converging,
pre-waist beam — position and slope anti-correlated, shrinking spot), and
negative `α` gives positive `⟨x x'⟩` (diverging, post-waist — matches the
physical picture explicitly claimed in the `TransverseDistribution`
docstring: "negative describes a diverging beam past its waist"). At the
waist `α = 0` the correlation vanishes and `β_T = σ_x²/ε_g`, recovering a
bare spot-size specification. This is the Courant & Snyder (1958) sign
convention as used in, e.g., Wiedemann.

### 2.4 Units

Standard accelerator convention: normalized emittance is quoted in
mm·mrad, `β_T` in m (treating the "per-radian" implicit in `β_T`'s definition
as the base, undivided radian — only the emittance's angle factor is
expressed in the smaller mrad). Converting to a fully internally consistent
`(mm·rad, mm/rad, rad/mm)` system:

```
ε_g[mm·rad] = ε_n[mm·mrad] × 10⁻³ / (βγ)_rel        (mrad → rad, on the angle factor only)
β_T[mm/rad] = β_T[m] × 10³                          (m → mm; the radian denominator is untouched
                                                       because 1 rad is the unconverted reference angle)
```

Dimensional consistency check: `ε_g[mm·rad] × β_T[mm/rad] = mm²` ⇒
`x = sqrt(εβ)` is in mm. `ε_g[mm·rad] / β_T[mm/rad] = rad²` ⇒ `x' =
sqrt(ε/β)` is in rad. `⟨x x'⟩ = ε_g α` is `mm·rad × (dimensionless) = mm·rad`,
consistent with `x[mm] × x'[rad]`. All three are self-consistent and match
the units `beam_metrics.PlaneMetrics` expects for a round-trip.

### 2.5 `beam_frame_basis`: slopes referred to the beam axis

The claim is that `beam_frame_basis(d)` is the shortest-arc rotation `R`
taking `ẑ` to unit vector `d = (d_x, d_y, d_z)`, so that column 2 is `d`
itself and columns 0/1 are an orthonormal transverse pair the slopes are
measured against, with `R = I` exactly when `d = ẑ`.

Standard Rodrigues construction: rotation axis
`k = (ẑ × d)/|ẑ × d| = (−d_y, d_x, 0)/s`, `s = sqrt(d_x²+d_y²) = sinθ`,
angle `θ = atan2(s, d_z) = arccos(d_z)` (valid since `d_z = cosθ` for a unit
vector and `θ∈[0,π]`). Rodrigues' rotation formula,
`R = I + sinθ K + (1−cosθ)K²` with `K` the skew matrix of `k`, gives (by
direct expansion, using `k_x²+k_y²=1`, `k_z=0`):

```
R ẑ = (sinθ k_y, −sinθ k_x, cosθ) = (d_x, d_y, d_z) = d     exactly.
```

This confirms column 2 = `beam_dir` exactly for any `d`, and, since `R` is a
proper rotation (orthogonal, det = 1) by construction, columns 0/1 are an
orthonormal transverse basis for any `d`. As `θ → 0` (`d → ẑ`), `s → 0`
and the axis direction `k` becomes ill-defined (0/0), which is exactly why
the implementation needs — and has — the explicit `sin_theta < 1e-15 →
return I` branch rather than relying on floating-point continuity through
that removable singularity; away from that branch the formula is the
standard shortest-arc rotation used throughout rigid-body kinematics.

The per-electron direction assembly
`dir ∝ x' ê1 + y' ê2 + ê3` (`ê1,ê2,ê3` = columns of `R`, i.e. `(x',y',1)`
resolved in the beam frame, then renormalized) is the standard paraxial
slope-to-unit-vector map (`x'≈dx/dz`, `y'≈dy/dz`), valid for the same
small-angle regime the Courant–Snyder formalism itself assumes. On axis
(`beam_dir=+z`, `R=I`) this reduces to `dir ∝ (x', y', 1)` directly in the
lab frame, so the phase-space and legacy-spot descriptions agree axis for
axis, as claimed.

### 2.6 Limiting case: `ε_n → 0`

`ε_g = 0` makes `x = 0·β_T^{1/2}·u1 = 0` and `x' = 0` **identically**,
independent of the drawn `u1,u2` (exact multiplication by zero, not a
cancellation of finite terms), so the sampler is bit-for-bit the collimated
point source in position and direction — matches the claimed limiting case.

## 3. Diff against the implementation

Read after the derivation above (`transverse.py` lines 101–219;
`montecarlo/geometry.py::beam_frame_basis` lines 523–545;
`montecarlo/transport.py::simulate_trajectories` transverse-injection block,
~lines 1500–1570):

- `_resolve_plane`: `geometric = normalized_emittance_mm_mrad *
  _MM_MRAD_TO_MM_RAD / beta_gamma` with `_MM_MRAD_TO_MM_RAD = 1e-3` and
  `beta_twiss = beta_twiss_m * _M_TO_MM_PER_RAD` with `_M_TO_MM_PER_RAD =
  1e3` — matches §2.4 exactly, both factor and direction.
- `gamma_twiss = (1 + alpha_twiss**2) / beta_twiss` — matches §2.2's
  `γ_T = (1+α²)/β_T` exactly.
- `sigma_position_mm = sqrt(geometric*beta_twiss)`,
  `sigma_slope_rad = sqrt(geometric*gamma_twiss)` — matches
  `σ_x = sqrt(ε_gβ_T)`, `σ_x' = sqrt(ε_gγ_T)` exactly.
- `resolve_transverse_distribution`: `beta_gamma = sqrt(gamma_rel**2 - 1)`,
  `gamma_rel = 1 + energy_keV/_ELECTRON_REST_KEV` with
  `_ELECTRON_REST_KEV ≈ 510.999` keV — matches §2.1 exactly (standard
  relativistic identity and standard rest mass).
- `_sample_plane`: `position = sqrt(emittance*beta_twiss) * u1`,
  `slope = sqrt(emittance/beta_twiss) * (u2 - alpha_twiss*u1)` — matches
  §2.2's draw term for term, including the sign on `α` inside the slope
  expression (`u2 − α u1`, not `u2 + α u1`), which is what produces
  `⟨x x'⟩ = −α ε_g` and the sign convention checked in §2.3.
- `beam_frame_basis`: axis components passed to `_small_tilt_R` are
  `(-d[1]*theta/sin_theta, d[0]*theta/sin_theta)`, and `_small_tilt_R`
  builds the skew matrix of `(kx, ky, 0) = (dx_rad, dy_rad)/|(dx_rad,dy_rad)|`
  fed into `R = I + sinθ K + (1−cosθ)K²`. Substituting
  `dx_rad = −d_y θ/s`, `dy_rad = d_x θ/s` gives `ang = θ` and
  `(kx,ky) = (−d_y/s, d_x/s)`, exactly the axis derived in §2.5. The
  `sin_theta < 1e-15` branch returns `np.eye(3)` directly, matching the
  identity limit.
- `simulate_trajectories`: `dirs = x_prime[:,None]*basis[:,0] +
  y_prime[:,None]*basis[:,1] + beam_dir`, then row-normalized — matches
  §2.5's `(x',y',1)`-in-beam-frame construction exactly, including using
  `beam_dir` itself (not `basis[:,2]` recomputed) for the longitudinal term,
  which is the same vector by the §2.5 proof (`R ẑ = d`) so this is not a
  second, independent convention.
- Positions: `offsets = stack(x_mm*MM_TO_ANG, y_mm*MM_TO_ANG)` then
  `project_beam_entry` — an Å unit conversion and the (separately ledgered)
  `grazing-beam-projection` tilt handling, both outside this claim's scope.
- No divergence found between the independent derivation and the
  implementation at any of the checked terms (unit factors, `βγ` identity,
  Twiss algebra, correlation sign, rotation construction, direction
  assembly, or the `ε_n→0`/on-axis limiting cases).

## 4. Numeric spot-check

Independent NumPy Monte Carlo (N=2×10⁷, arbitrary `ε_g, β_T, α` — not values
drawn from the code or its tests) reproduces the three target second
moments to Monte-Carlo precision (§2.2), and `tests/test_transverse.py`
(23 parametrized cases, including
`test_geometric_emittance_scales_inversely_with_beta_gamma`,
`test_sampled_correlation_sign_follows_alpha`,
`test_sampled_moments_round_trip_through_beam_metrics`,
`test_zero_emittance_transport_converges_to_the_collimated_run`,
`test_beam_frame_basis_is_identity_on_axis_and_maps_z_onto_the_beam`) all
pass against the current worktree (`feature/beam-phase-space`).

## 5. Result

Independent derivation matches the implementation term for term: the
`mm·mrad→mm·rad` / `m→mm/rad` unit factors, the `ε_g=ε_n/(βγ)_rel` scaling
with `(βγ)_rel=sqrt(γ_rel²−1)`, the Courant–Snyder per-plane draw and its
exact second moments `⟨x²⟩=ε_gβ_T`, `⟨x'²⟩=ε_g(1+α²)/β_T`, `⟨xx'⟩=−αε_g`,
the signed-`α` convention, and the `beam_frame_basis` shortest-arc rotation
(exact identity on-axis, exact `Rẑ=beam_dir` off-axis). No unresolved
discrepancy found.

**Verdict:** `rederived`.
