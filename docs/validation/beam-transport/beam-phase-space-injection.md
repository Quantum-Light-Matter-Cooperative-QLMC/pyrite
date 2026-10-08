# Courant–Snyder transverse beam injection

**Validation id:** `beam-phase-space-injection` **Code:** `montecarlo/transverse.py::resolve_transverse_distribution`; `::sample_transverse`; `montecarlo/transport.py::simulate_trajectories`; `montecarlo/geometry.py::beam_frame_basis` **Source:** Courant & Snyder, *Ann. Phys.* **3**, 1–48 (1958); standard normalized-emittance convention (e.g. Wiedemann, *Particle Accelerator Physics*). **Status:** rederived in fresh context on 2026-08-07.

## 1. What the code claims to compute

From the ledger row and docstrings, before touching the implementation body, the intended quantity is:

- A stored policy carries **normalized** emittance `ε_n` [mm·mrad], Twiss `β_T` [m] and signed `α` per transverse plane, because geometric emittance is not invariant across the swept `energy_keV` axis.
- Per case energy, resolve the **geometric** emittance `ε_g = ε_n / (βγ)_rel`, with `(βγ)_rel = sqrt(γ_rel² − 1)`, `γ_rel = 1 + E/(m_e c²)`.
- Per-plane second moments follow Courant–Snyder: `σ_x = sqrt(ε_g β_T)`, `σ_x' = sqrt(ε_g γ_T)`, `γ_T = (1+α²)/β_T`, correlation `⟨x x'⟩ = −α ε_g`.
- Per-electron draw: `u1, u2 ~ N(0,1)` i.i.d., `x = sqrt(ε_g β_T) u1`, `x' = sqrt(ε_g/β_T)(u2 − α u1)`.
- `beam_frame_basis(beam_dir)` supplies the rotation whose transverse columns the slopes `(x', y')` are measured against, and is claimed to be exactly the identity when `beam_dir = +z`.
- Units: input emittance is mm·mrad, input `β_T` is m; internal/diagnostic units are mm·rad (emittance) and mm/rad (`β_T`), matching `beam_metrics.PlaneMetrics`.

## 2. Independent derivation

### 2.1 Normalized → geometric emittance

Liouville's theorem makes the *normalized* emittance `ε_n = β_rel γ_rel · ε_g` an adiabatic invariant under longitudinal acceleration (no space charge, no nonlinear forces — exactly the assumptions stated in the docstring). Hence

```
ε_g(E) = ε_n / (β_rel γ_rel)(E),  (β_rel γ_rel) = sqrt(γ_rel² − 1),  γ_rel = 1 + T/(m_e c²).
```

This is the textbook definition (Wiedemann §8.2 / Courant & Snyder framework extended to acceleration). At fixed `ε_n`, `ε_g` is monotonically decreasing in `(βγ)_rel`, i.e. in the inverse ratio of `βγ` between two energies — matches the stated limiting case.

### 2.2 Courant–Snyder second moments and the per-plane draw

Define the Twiss/Courant-Snyder invariant `βγ_T − α² = 1` (standard convention, `γ_T` here is the *Twiss* gamma, distinct from the relativistic `γ_rel` above — the two symbols collide notationally in the literature but not in the code, which spells them `beta_twiss`/`gamma_twiss` vs `beta_gamma`). The single-particle invariant is

```
ε_g = γ_T x² + 2 α x x' + β_T x'².
```

The Gaussian-equivalent beam matrix (second moments matched to this invariant, standard result) is

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

All three moments match the target `Σ` exactly (not just to leading order): this is the standard Courant–Snyder / Cholesky-style sampler, confirmed numerically below to machine-precision Monte Carlo agreement (N=2×10⁷, arbitrary `ε_g=3.7×10⁻³` mm·rad, `β_T=1.8` mm/rad, `α=−0.6`):

```
target <x^2>   = 6.660000e-3   sample = 6.664188e-3
target <x'^2>  = 2.795556e-3   sample = 2.796073e-3
target <x x'>  = 2.220000e-3   sample = 2.221557e-3
```

### 2.3 Sign convention on `α`

With `Σ_{12} = −α ε_g`, positive `α` gives negative `⟨x x'⟩` (converging, pre-waist beam — position and slope anti-correlated, shrinking spot), and negative `α` gives positive `⟨x x'⟩` (diverging, post-waist — matches the physical picture explicitly claimed in the `TransverseDistribution` docstring: "negative describes a diverging beam past its waist"). At the waist `α = 0` the correlation vanishes and `β_T = σ_x²/ε_g`, recovering a bare spot-size specification. This is the Courant & Snyder (1958) sign convention as used in, e.g., Wiedemann.

### 2.4 Units

Standard accelerator convention: normalized emittance is quoted in mm·mrad, `β_T` in m (treating the "per-radian" implicit in `β_T`'s definition as the base, undivided radian — only the emittance's angle factor is expressed in the smaller mrad). Converting to a fully internally consistent `(mm·rad, mm/rad, rad/mm)` system:

```
ε_g[mm·rad] = ε_n[mm·mrad] × 10⁻³ / (βγ)_rel        (mrad → rad, on the angle factor only)
β_T[mm/rad] = β_T[m] × 10³                          (m → mm; the radian denominator is untouched
                                                       because 1 rad is the unconverted reference angle)
```

Dimensional consistency check: `ε_g[mm·rad] × β_T[mm/rad] = mm²` ⇒ `x = sqrt(εβ)` is in mm. `ε_g[mm·rad] / β_T[mm/rad] = rad²` ⇒ `x' = sqrt(ε/β)` is in rad. `⟨x x'⟩ = ε_g α` is `mm·rad × (dimensionless) = mm·rad`, consistent with `x[mm] × x'[rad]`. All three are self-consistent and match the units `beam_metrics.PlaneMetrics` expects for a round-trip.

### 2.5 `beam_frame_basis`: slopes referred to the beam axis

The claim is that `beam_frame_basis(d)` is the shortest-arc rotation `R` taking `ẑ` to unit vector `d = (d_x, d_y, d_z)`, so that column 2 is `d` itself and columns 0/1 are an orthonormal transverse pair the slopes are measured against, with `R = I` exactly when `d = ẑ`.

Standard Rodrigues construction: rotation axis `k = (ẑ × d)/|ẑ × d| = (−d_y, d_x, 0)/s`, `s = sqrt(d_x²+d_y²) = sinθ`, angle `θ = atan2(s, d_z) = arccos(d_z)` (valid since `d_z = cosθ` for a unit vector and `θ∈[0,π]`). Rodrigues' rotation formula, `R = I + sinθ K + (1−cosθ)K²` with `K` the skew matrix of `k`, gives (by direct expansion, using `k_x²+k_y²=1`, `k_z=0`):

```
R ẑ = (sinθ k_y, −sinθ k_x, cosθ) = (d_x, d_y, d_z) = d     exactly.
```

This confirms column 2 = `beam_dir` exactly for any `d`, and, since `R` is a proper rotation (orthogonal, det = 1) by construction, columns 0/1 are an orthonormal transverse basis for any `d`. As `θ → 0` (`d → ẑ`), `s → 0` and the axis direction `k` becomes ill-defined (0/0), which is exactly why the implementation needs — and has — the explicit `sin_theta < 1e-15 → return I` branch rather than relying on floating-point continuity through that removable singularity; away from that branch the formula is the standard shortest-arc rotation used throughout rigid-body kinematics.

The per-electron direction assembly `dir ∝ x' ê1 + y' ê2 + ê3` (`ê1,ê2,ê3` = columns of `R`, i.e. `(x',y',1)` resolved in the beam frame, then renormalized) is the standard paraxial slope-to-unit-vector map (`x'≈dx/dz`, `y'≈dy/dz`), valid for the same small-angle regime the Courant–Snyder formalism itself assumes. On axis (`beam_dir=+z`, `R=I`) this reduces to `dir ∝ (x', y', 1)` directly in the lab frame, so the phase-space and legacy-spot descriptions agree axis for axis, as claimed.

### 2.6 Limiting case: `ε_n → 0`

`ε_g = 0` makes `x = 0·β_T^{1/2}·u1 = 0` and `x' = 0` **identically**, independent of the drawn `u1,u2` (exact multiplication by zero, not a cancellation of finite terms), so the sampler is bit-for-bit the collimated point source in position and direction — matches the claimed limiting case.

## 3. Diff against the implementation

Read after the derivation above (`montecarlo/transverse.py` lines 101–219; `montecarlo/geometry.py::beam_frame_basis` lines 523–545; `montecarlo/transport.py::simulate_trajectories` transverse-injection block, ~lines 1500–1570):

- `_resolve_plane`: `geometric = normalized_emittance_mm_mrad * _MM_MRAD_TO_MM_RAD / beta_gamma` with `_MM_MRAD_TO_MM_RAD = 1e-3` and `beta_twiss = beta_twiss_m * _M_TO_MM_PER_RAD` with `_M_TO_MM_PER_RAD = 1e3` — matches §2.4 exactly, both factor and direction.
- `gamma_twiss = (1 + alpha_twiss**2) / beta_twiss` — matches §2.2's `γ_T = (1+α²)/β_T` exactly.
- `sigma_position_mm = sqrt(geometric*beta_twiss)`, `sigma_slope_rad = sqrt(geometric*gamma_twiss)` — matches `σ_x = sqrt(ε_gβ_T)`, `σ_x' = sqrt(ε_gγ_T)` exactly.
- `resolve_transverse_distribution`: `beta_gamma = sqrt(gamma_rel**2 - 1)`, `gamma_rel = 1 + energy_keV/_ELECTRON_REST_KEV` with `_ELECTRON_REST_KEV ≈ 510.999` keV — matches §2.1 exactly (standard relativistic identity and standard rest mass).
- `_sample_plane`: `position = sqrt(emittance*beta_twiss) * u1`, `slope = sqrt(emittance/beta_twiss) * (u2 - alpha_twiss*u1)` — matches §2.2's draw term for term, including the sign on `α` inside the slope expression (`u2 − α u1`, not `u2 + α u1`), which is what produces `⟨x x'⟩ = −α ε_g` and the sign convention checked in §2.3.
- `beam_frame_basis`: axis components passed to `_small_tilt_R` are `(-d[1]*theta/sin_theta, d[0]*theta/sin_theta)`, and `_small_tilt_R` builds the skew matrix of `(kx, ky, 0) = (dx_rad, dy_rad)/|(dx_rad,dy_rad)|` fed into `R = I + sinθ K + (1−cosθ)K²`. Substituting `dx_rad = −d_y θ/s`, `dy_rad = d_x θ/s` gives `ang = θ` and `(kx,ky) = (−d_y/s, d_x/s)`, exactly the axis derived in §2.5. The `sin_theta < 1e-15` branch returns `np.eye(3)` directly, matching the identity limit.
- `simulate_trajectories`: `dirs = x_prime[:,None]*basis[:,0] + y_prime[:,None]*basis[:,1] + beam_dir`, then row-normalized — matches §2.5's `(x',y',1)`-in-beam-frame construction exactly, including using `beam_dir` itself (not `basis[:,2]` recomputed) for the longitudinal term, which is the same vector by the §2.5 proof (`R ẑ = d`) so this is not a second, independent convention.
- Positions: `offsets = stack(x_mm*MM_TO_ANG, y_mm*MM_TO_ANG)` then `project_beam_entry` — an Å unit conversion and the (separately ledgered) `grazing-beam-projection` tilt handling, both outside this claim's scope.
- No divergence found between the independent derivation and the implementation at any of the checked terms (unit factors, `βγ` identity, Twiss algebra, correlation sign, rotation construction, direction assembly, or the `ε_n→0`/on-axis limiting cases).

## 4. Numeric spot-check

Independent NumPy Monte Carlo (N=2×10⁷, arbitrary `ε_g, β_T, α` — not values drawn from the code or its tests) reproduces the three target second moments to Monte-Carlo precision (§2.2), and `tests/montecarlo/test_transverse.py` (23 parametrized cases, including `test_geometric_emittance_scales_inversely_with_beta_gamma`, `test_sampled_correlation_sign_follows_alpha`, `test_sampled_moments_round_trip_through_beam_metrics`, `test_zero_emittance_transport_converges_to_the_collimated_run`, `test_beam_frame_basis_is_identity_on_axis_and_maps_z_onto_the_beam`) all pass against the current worktree (`feature/beam-phase-space`).

## 5. Result

Independent derivation matches the implementation term for term: the `mm·mrad→mm·rad` / `m→mm/rad` unit factors, the `ε_g=ε_n/(βγ)_rel` scaling with `(βγ)_rel=sqrt(γ_rel²−1)`, the Courant–Snyder per-plane draw and its exact second moments `⟨x²⟩=ε_gβ_T`, `⟨x'²⟩=ε_g(1+α²)/β_T`, `⟨xx'⟩=−αε_g`, the signed-`α` convention, and the `beam_frame_basis` shortest-arc rotation (exact identity on-axis, exact `Rẑ=beam_dir` off-axis). No unresolved discrepancy found.

**Verdict:** `rederived`.

## 6. Counter-addressed draws (#361 re-verification, 2026-10-07)

Fresh-context check of the stream change on branch `issue-361-adaptive-electron-count` (commit `df221caf`). The Courant–Snyder sampler of §2.2 is unchanged; only the source of $u_1, u_2$ changed. The derivation below was written from `docs/computation/random-streams.md` and the ledger row before reading `montecarlo/transport/kinematics.py::counter_uniforms` / `::counter_normals`. It also serves the spot (`finite-beam-size`), energy-spread (`beam-energy-spread-injection`) and bunch (`longitudinal-bunch-sampling`) rows, which use the same construction.

### 6.1 Uniform law

Let $s$ be the 64-bit root of a child namespace (first state word of `SeedSequence(seed).spawn(n)[i]`), $\Phi_g = \texttt{0x9E3779B97F4A7C15}$, and $h$ the SplitMix64 finalizer, a bijection on $\mathbb Z_{2^{64}}$. Electron $e$ has key $k_e = h(s + \Phi_g(e+1))$, and draw $c$ is

$$
z_{e,c} = h\bigl(k_e + \Phi_g(c+1)\bigr),\qquad
m_{e,c} = \lfloor z_{e,c}/2^{12}\rfloor,\qquad
u_{e,c} = \bigl(m_{e,c} + \tfrac12\bigr)\,2^{-52}.
$$

Under the counter-mode hypothesis that $z$ is uniform on $\{0,\dots,2^{64}-1\}$, $m$ is uniform on $\{0,\dots,2^{52}-1\}$ (each value has exactly $2^{12}$ preimages). Hence $u$ is uniform on the midpoint grid $\{(2j+1)\,2^{-53}\}$:

$$
u \in \bigl[2^{-53},\,1-2^{-53}\bigr]\subset(0,1),\qquad
\mathbb E[u]=\tfrac12,\qquad
\operatorname{Var}(u)=\tfrac{1}{12}\bigl(1-2^{-104}\bigr).
$$

There is no mass at $0$ or $1$. Since $2j+1<2^{53}$, the value $u$ is an exact double. The map $j\mapsto 2^{52}-1-j$ sends $u\mapsto 1-u$, so the law is exactly symmetric about $\tfrac12$.

### 6.2 Normal law by inverse CDF

$Z=\Phi^{-1}(u)$ is exactly $\mathcal N(0,1)$ conditioned on $|Z|\le z_{\max}$, up to $2^{-52}$ grid discreteness, where

$$
z_{\max} = -\Phi^{-1}\bigl(2^{-53}\bigr) = 8.2095,\qquad
P_{\mathcal N}\bigl(|Z|>z_{\max}\bigr)=2^{-52}\approx 2.2\times10^{-16}.
$$

The symmetry of §6.1 makes $\mathbb E[Z]=0$ exactly. The truncation lowers the variance by $O(z_{\max}\,\varphi(z_{\max}))\sim10^{-14}$, which is negligible. Each normal uses one uniform, so draw $c$ of electron $e$ maps one-to-one onto $Z_{e,c}$.

### 6.3 Independence and namespaces

Distinct $(e,c)$ pairs give distinct finalizer inputs $k_e+\Phi_g(c+1)$ except with probability $O(N^2C/2^{64})$ for $N$ electrons and $C$ counters. Distinct children have independent 64-bit roots $s$. Transport keys use the run seed itself, $h(\text{seed}+\Phi_g(e+1))$, which is not a `SeedSequence` child. Any pair of streams can therefore coincide only through a 64-bit additive collision of keys, with expected count $\approx N^2(C_1+C_2)/2^{64}$. For $N=10^6$ and $C=10^3$ transport counters this is $\sim5\times10^{-5}$. Even then, the effect would be one shared uniform between unrelated electrons. The guarantee is probabilistic, not structural, and has the same class as the existing inter-electron overlap of the transport core.

### 6.4 Courant–Snyder with counter normals

$u_1=Z_{e,0}$ and $u_2=Z_{e,1}$ (x plane) and $Z_{e,2}$, $Z_{e,3}$ (y plane) are i.i.d. $\mathcal N(0,1)$ by §6.2–6.3, so §2.2 holds verbatim: $\langle x^2\rangle=\varepsilon\beta_T$, $\langle x'^2\rangle=\varepsilon\gamma_T$, $\langle xx'\rangle=-\alpha\varepsilon$, and $\langle xy\rangle=0$. Because the draw is a pure function of $(\text{seed},e,c)$, a block $[a,b)$ reproduces rows $[a,b)$ of any larger draw. The $\varepsilon_n\to0$ limit is unchanged, since offsets scale as $\sqrt{\varepsilon_n}$ for fixed draws.

### 6.5 Implementation diff

```text
x = keys + GOLDEN * (counters + 1); splitmix64 finalizer
u = ((x >> 12).astype(float64) + 0.5) * 2**-52
normals = ndtri(u)
sample_transverse: z = counter_normals(child_stream_root(seed, 5, 4), n, 4, start=start)
                   x <- z[:,0], z[:,1];  y <- z[:,2], z[:,3]
```

This matches §6.1–6.4 term for term, including the 52-bit shift, half-offset, and draw roles. `child_stream_root` is `SeedSequence(seed).spawn(n)[i].generate_state(1, uint64)[0]`, matching the documented spawn table. The docstring bound "$|z|<8.3$" is true; the exact bound is $8.2095$, as `random-streams.md` states with $\approx 8.2$.

### 6.6 Numeric evidence (independent, CPU)

- A pure-Python SplitMix64 reference written for this check, with roots taken directly from `SeedSequence`, reproduces `counter_uniforms` bit for bit for all four children (`spawn(2)[1]`, `(4)[3]`, `(5)[4]`, `(6)[5]`) at $e\in[1000,1050)$, $c<5$. The four roots are distinct from each other and from the seed.
- Uniforms, $N=2\times10^6$ per counter, $c=0..4$: minimum $8.2\times10^{-8}$ and maximum $1-4.8\times10^{-7}$, with no value at $0$ or $1$. Means are $0.49965$–$0.50020$, $12\operatorname{Var}$ is $0.99896$–$1.00041$, and KS $p$ is $0.055$–$0.999$. The maximum within-electron cross-counter $\lvert\rho\rvert$ is $1.2\times10^{-3}$, against $1/\sqrt N=7.1\times10^{-4}$ over 10 pairs. Across-electron lag correlations at lags 1, 2, 7 and 64 are all $\le 8.8\times10^{-4}$. The $32\times32$ two-dimensional $\chi^2$ test gives $p=0.26$ for $(c_0,c_1)$ and $p=0.19$ for $(e,e+1)$. The maximum $\lvert\rho\rvert$ across children for the same electron is $1.6\times10^{-3}$, over 96 pairs. The maximum $\lvert\rho\rvert$ against the main transport counter stream (`_stream_uniform_scalar`, $c<4$) is $5.5\times10^{-3}$ at $N=2\times10^5$ over 80 pairs, about $2.4\sigma$, which is consistent with the maximum of 80 draws. Exact SplitMix state collisions between child ($c<5$) and main ($c<64$) streams at $5\times10^4$ electrons: 0.
- Normals, $N=10^7$: mean $4.1\times10^{-4}$ ($1.3\sigma$), variance $0.99969$ ($0.7\sigma$), skew $1.2\times10^{-4}$, excess kurtosis $-3.6\times10^{-3}$, and KS $D=2.8\times10^{-4}$ ($p=0.41$). For $\lvert Z\rvert>2,3,4$ the ratios to the exact tail are $0.999$, $0.993\pm0.006$ and $0.954\pm0.040$. Computed $\Phi^{-1}(2^{-53})=-8.2095$, and $\Phi^{-1}(u)+\Phi^{-1}(1-u)=0$ exactly on grid points.
- Courant–Snyder at $3000$ keV, with independent targets from CODATA $m_ec^2$ ($\beta\gamma$ agrees to $1.2\times10^{-9}$). For the x plane, $(\varepsilon_n,\beta_T,\alpha)=(2.3,1.7,-0.8)$; for the y plane, $(0.9,0.4,1.3)$. At $N=5\times10^6$, the ratios $\langle x^2\rangle/\varepsilon\beta_T$, $\langle x'^2\rangle/\varepsilon\gamma_T$ and $\langle xx'\rangle/(-\alpha\varepsilon)$ are $1.00028$, $0.99960$ and $1.00006$ (x) and $1.00066$, $1.00104$ and $1.00107$ (y). The RMS-emittance ratios are $0.99987$ and $1.00047$. Inter-plane $\rho(x,y)=9.8\times10^{-5}$. For three electrons, per-electron rows equal $\sqrt{\varepsilon\beta_T}\,\Phi^{-1}(u_{e,0})$ and the matching slope formulas built from reference $u_{e,c}$ for $c=0..3$.
- Prefix stability: `sample_transverse(n=1000)` equals the first 1000 rows of `n=1700`, and `start=600, n=400` equals rows $[600,1000)$, bit for bit. `counter_normals` blocks $[0,1)$, $[1000,1777)$, $[2999,3000)$ and $[17,3000)$ equal the slice of $[0,3000)$ for every child, and fewer draws give the leading columns.
- $\varepsilon_n\to0$: the policy rejects $\varepsilon_n=0$, so the limit is approached through scaling. At $\varepsilon_n=10^{-12}$ the offsets equal $10^{-6}\times$ those at $\varepsilon_n=1$ to $2.2\times10^{-16}$ relative, for all four arrays. An unset policy draws nothing.
- Point beam against `main` (`9af246f4`): `runner._transport_case` + `_spectrum_case` for hopg, 30 keV, 30° tilt, 40+10 electrons, both spot widths `None`. All 48 segment fields and 6 spectrum arrays are bit-identical on both the `per-electron` and `lockstep` cores. The default `Sweep` carries a 1 mm FWHM spot, and with it only `initial_r_ang` and `r_mid` differ from `main`. That is the documented one-time finite-spot realization change, not the point-beam path.
- Anchors listed in the row pass, along with `tests/dev/test_validation_ledger.py`; `pyrite-dev validation-ledger --check` exits 0.

### 6.7 Result (#361)

The counter construction yields an open-interval uniform law, an exactly symmetric $\mathcal N(0,1)$ truncated only at $8.21\sigma$, and draws that are independent across counters, electrons, and children up to 64-bit collision probability. Courant–Snyder moments are reproduced, the zero-emittance and point-beam limits hold, and blocks equal slices. No discrepancy. Verdict for the stream change: `rederived`; the anchored status is supported.
