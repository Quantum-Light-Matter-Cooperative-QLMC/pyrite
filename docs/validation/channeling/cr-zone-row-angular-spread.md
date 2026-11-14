# `cr-zone-row-angular-spread` — independent verification

- **Claim:** [`cr-zone-row-angular-spread`](../ledger-channeling.md#cr-zone-row-angular-spread),
  a measured omission bound: planar zone rows are scored along the straight mean
  segment $\hat{\mathbf v}$. Averaging over the retained angle distribution would
  change each row yield by $\Delta Y=\sum_a\left[Y_a\langle\theta_a\rangle+\tfrac12Y_{aa}\langle\theta_a^2\rangle\right]$.
  For `channeling_demo` the ledger states $\le 2\%$ of the zone-row band yield at the
  forward direction and $\le 0.5\%$ at the Bragg, $\pm1/\gamma$ and polar probes.
- **Code (measurement only):** `checks/channeling_zone_angular_spread.py::{_retained_moments,_tilted,main}`,
  reusing `checks/channeling_detector_components.py::{_profile_case,_bragg_directions,_subset,_detector_score}`.
- **Records:** `docs/validation/check-records/planar-zone/zone-angular-spread-1148-probes-ne16.json`,
  `docs/validation/check-records/planar-zone/zone-angular-spread-1148-polar-ne16.json`.
- **Verifier:** fresh context, 2026-11-13. Verifier scratch scripts:
  `scratch/zas_verify_moments.py`, `scratch/zas_verify_rows.py`, `scratch/zas_verify_solve.py`.

## 1. Independent derivation

### 1.1 Yield averaged over the row direction

Let $Y(\hat{\mathbf v})$ be the band yield of one zone-row population at fixed
detector direction $\hat{\mathbf n}$, with the rows pointing along
$\hat{\mathbf v}$. The scorers sum independent rows, so $Y$ is linear in the row set,
and a uniform rotation of every zone row gives the population response. Write the
direction of a retained electron as $\hat{\mathbf v}_0$ rotated by small angles
$\theta_x$ (towards the plane normal $\hat{\mathbf n}_p$) and $\theta_y$ (towards
the in-plane transverse axis $\hat{\mathbf e}_y=\hat{\mathbf n}_p\times\hat{\mathbf b}$).
Taylor expansion gives

$$
\langle Y\rangle-Y_0
=\sum_a Y_a\langle\theta_a\rangle
+\tfrac12\sum_{a,b}Y_{ab}\langle\theta_a\theta_b\rangle
+O(\theta^3).
$$

Kernels of the planar master equation are even in $q_y$, and the beam has no
$y$ component relative to $\hat{\mathbf v}_0$. The retained ensemble is therefore
symmetric under $\theta_y\to-\theta_y$, so $\langle\theta_y\rangle=0$ and
$\langle\theta_x\theta_y\rangle=0$, and the cross term $Y_{xy}$ drops out
whatever its value. The deviation about $\hat{\mathbf v}_0$ has
$\langle\theta_x^2\rangle=\operatorname{var}\theta_x+\langle\theta_x\rangle^2$. The
second term is $1.4\times10^{-11}\,\mathrm{rad}^2$ against
$1.8\times10^{-6}\,\mathrm{rad}^2$ and is negligible, so using the variance (as the
check does) is equivalent here.

### 1.2 Two-point estimator

With $D(d)=\tfrac12[Y(+d)+Y(-d)]-Y_0$ and $Y(\pm d)=Y_0\pm Y'd+\tfrac12Y''d^2\pm\tfrac16Y'''d^3+\tfrac1{24}Y''''d^4$,

$$
D(d)=\tfrac12Y''d^2+\tfrac1{24}Y''''d^4,\qquad
\frac{Y(+d)-Y(-d)}{2d}=Y'+O(d^2).
$$

At $d=\sigma$, $D(\sigma)$ is the second-order term $\tfrac12Y''\sigma^2$. Write
$Y-Y_0=c_2\theta^2+c_4\theta^4$ and $r=c_4\sigma^2/c_2$. The ratio
$q=D(2\sigma)/[4D(\sigma)]=(1+4r)/(1+r)$ then measures departure from a quadratic
response. A Gaussian average gives $c_2\sigma^2(1+3r)$ against the two-point value
$c_2\sigma^2(1+r)$. For $q<1$ (so $r<0$) the two-point rule overstates the
magnitude of the shift, which makes it conservative. The odd term
$\langle\theta_x\rangle[Y(+\sigma)-Y(-\sigma)]/(2\sigma)$ is $Y'\langle\theta_x\rangle$
to $O(\sigma^2)$.

### 1.3 Normal-axis moments

The plane-wave diagonal of the retained density at depth $z$ is the transverse
momentum distribution. Order $m$ in Bloch block $\kappa$ carries
$k_x=\kappa+mg$, so $\theta_x=(\kappa+mg)/p$ with $p=\gamma\beta mc/\hbar$
[Å$^{-1}$]. Zone rows run while an electron is retained, so a row segment at depth
$z$ is weighted by the retained trace $T(z)$, and

$$
\langle f\rangle=\frac{\int_0^L T(z)\,\bar f(z)\,dz}{\int_0^L T(z)\,dz},
$$

where $\bar f(z)$ is the moment of the normalized retained density. At $z=0$ the
entrance state is the snapped plane wave, so $\bar\theta_x=\psi_{\rm eff}$ and
$\overline{\theta_x^2}=\psi_{\rm eff}^2+\sigma_{\rm div}^2$.

### 1.4 Out-of-plane moment

A jump adds $q_y$ to $\kappa_y$ while the no-jump evolution is independent of
$\kappa_y$. The Fourier transform $\kappa_y\to\eta$ therefore multiplies each jump
weight by $\cos(q_y\eta)$ (`cr-planar-exit-out-of-plane`), and

$$
\frac{\tilde P(\eta)}{\tilde P(0)}=\langle\cos(p\,\theta_y\eta)\rangle,\qquad\theta_y=\kappa_y/p.
$$

Cumulant expansion of a symmetric distribution with moments $\mu_2,\mu_4$ gives

$$
-\frac{2\ln[\tilde P(\eta)/\tilde P(0)]}{(p\eta)^2}
=\mu_2-\frac{(p\eta)^2}{12}\left(\mu_4-3\mu_2^2\right)+O(\eta^4).
$$

The convention ($\cos(q_y\eta)$, $\theta_y=\kappa_y/p$) and the factor $-2/(p\eta)^2$
are therefore correct. For a leptokurtic (heavy-tailed) distribution the estimator
lies **below** $\mu_2$ and rises monotonically as $\eta\to0$. If $\mu_4$ diverges
(a $\lvert\theta\rvert^{-5}$ or slower 1D tail), it approaches $\mu_2$ only as
$O(\lvert\eta\rvert)$. Neither arm is therefore an upper bound on $\mu_2$; it is a
lower estimate.

## 2. Comparison with the check

### 2.1 Zone solve reproduction

| Item | Production zone (`handoff._solve`) | Check | Result |
| --- | --- | --- | --- |
| Kernel | `tds_kernel + electronic_kernel`, `n_cells=bloch_cells`, `max_order=basis_size` | `out_of_plane_kernel(η=0)`, same options | cell kernels identical (max abs diff 0) |
| Retained subspace | `margin_eV` 300 | same | identical |
| Entry | `propagate(incidence=effective_rad, divergence)` | `entry_state(effective_rad, divergence)` | `effective_rad` $\approx -5\times10^{-17}$; no case divergence, so both are snapped to 0 |
| Retained trace at 250 / 3000 / 8750 / 20000 Å | 0.99873 / 0.98750 / 0.96697 / 0.92947 | 0.99874 / 0.98753 / 0.96703 / 0.92961 | agree to $\le2\times10^{-4}$ (production grid step 250.34 Å vs 250 Å) |
| Stopping | case value | none | `_solve` with $0$ and $0.04\,\mathrm{eV/Å}$ gives identical traces; $\Delta p/p\approx4\times10^{-5}$ |

The case derives the normal divergence from the sampled $\hat{\mathbf v}$
(`np.std(psi)`), while the check uses `case["divergence_rad"]`. For
`channeling_demo`, transport gives every zone row $\hat{\mathbf v}=\hat{\mathbf b}$
exactly, so both are zero. The angle grid `(kappa[:,None] + orders[None,:] g)/p`
matches the documented `density_exit_distribution` ordering of the
`propagate_batch` output `(depth, B, cells, N)`.

### 2.2 Depth resolution (finding)

The check samples the trace and moments at 8 depths (`linspace(1, 80, 8)` steps,
spacing $\approx2750$ Å) plus the entrance point. Resolving all 80 steps (verifier
scratch, same solver) gives:

| Moment | Check (8 depths) | Verifier (80 depths) | Ratio |
| --- | --- | --- | --- |
| $\langle\theta_x\rangle$ | $3.675\times10^{-6}$ | $3.687\times10^{-6}$ | 1.003 |
| $\langle\theta_x^2\rangle$ | $1.825\times10^{-6}$ | $1.907\times10^{-6}$ | **1.045** |
| $\langle\theta_y^2\rangle$, $\eta$ scale 0.1 | $2.838\times10^{-6}$ | $2.842\times10^{-6}$ | 1.002 |

The first interval $[0,3000]$ Å spans the first channeling oscillation
($\lambda_{\rm ch}/2\approx\pi d_p/(2\psi_c)\sim2000$ Å). Linear interpolation from
250 Å to 3000 Å misses the early rise of $\overline{\theta_x^2}$, so the normal rms
is 2.2% low (1.351 vs 1.381 mrad).

### 2.3 Out-of-plane arm convergence (finding)

$\eta$ scales are in units of $1/(p\psi_c)$, with $p\eta\sigma_y\approx$ the scale;
the 80-depth average is used:

| Scale | 0.4 | 0.2 | 0.1 | 0.05 | 0.025 |
| --- | --- | --- | --- | --- | --- |
| $\langle\theta_y^2\rangle$ [$10^{-6}\,\mathrm{rad}^2$] | 2.153 | 2.585 | 2.842 | 2.983 | 3.057 |

Successive differences halve (0.43, 0.26, 0.14, 0.07), so the approach is
$O(\eta)$, as for a heavy tail with divergent $\mu_4$. Linear extrapolation gives
$\mu_2\approx3.13\times10^{-6}\,\mathrm{rad}^2$, which is 10% above the 0.1 arm the
check uses. The ledger assumption that the smaller-$\eta$ arm "is the larger of the
two" is true but not conservative, because the estimator keeps rising. The extra
second moment comes from $\lvert\theta_y\rvert\gtrsim1/(p\eta)\approx15$–60 mrad,
i.e. $\gamma\theta\gtrsim0.5$. There the quadratic Taylor term overstates the
forward-peak loss, so its true weight in $\Delta Y$ is smaller than the factor
1.10 suggests.

### 2.4 Tilt construction

The zone rows of the transported demo case all have
$\hat{\mathbf v}=\hat{\mathbf b}=(-0.0370,-0.0370,0.9986)$ in the slab frame and
$\hat{\mathbf v}\cdot\hat{\mathbf n}_p=-4.6\times10^{-17}$, where
$\hat{\mathbf n}_p$ is `case_plane_incidence(case).plane_normal`, so the frames
agree. `_tilted` sets
$\hat{\mathbf v}'=\cos d\,\hat{\mathbf v}+\sin d\,\hat{\mathbf a}_\perp$ with
$\hat{\mathbf a}_\perp$ the unit projection of the axis orthogonal to
$\hat{\mathbf v}$, which is an exact rotation by $d$ in the
$(\hat{\mathbf v},\hat{\mathbf a})$ plane. The axes are $\hat{\mathbf n}_p$ (normal)
and $\hat{\mathbf n}_p\times\hat{\mathbf b}$ (in-plane). Only zone rows are turned,
and $L$ and $\mathbf r_{\rm mid}$ are held fixed.

The in-plane axis has a $z$ component of 0.052 (the beam is 3° from the face
normal), so the in-plane tilt moves the row endpoints by about $\pm0.9$ Å through
the slab faces. 14 of the 16 rows span the full slab. Clipping in the escape
integration gives the $\lvert d\rvert$-linear response seen in
`spec_characteristic` ($q=0.50$, relative shift $\le3.6\times10^{-6}$), so this
probe artifact is negligible.

### 2.5 Scorer dependence on $\hat{\mathbf v}$

- `spec` (native PXR/CBS): row `v_hat` enters the Doppler denominator
  (`cxr/_kernels.py::_vacuum_doppler`), the resonance $\omega_{\rm res}$, the
  formation pieces (`cxr/_formation.py`) and the reflection-row seeds
  (`reflection_rows.py`), so the dependence is genuine.
- `brem`: the case resolves `bremsstrahlung_model = "bremslib"`, whose DDCS is
  evaluated at $\arccos(\hat{\mathbf v}\cdot\hat{\mathbf n})$ (`brem.py`, lines
  1006–1007), so the dependence is genuine. Near forward the theta grid is fine
  (0.35 mrad steps). At the 30/37.5/45° polar probes, in-plane brem shows $q=0.50$
  (kink of the linear-in-$\theta$ interpolant at a node). The shifts are
  $\le4.6\times10^{-4}$, and the two-point value overstates a
  $\lvert\theta\rvert$ response by $1/\sqrt{2/\pi}=1.25$.
- `spec_characteristic`: isotropic; $\hat{\mathbf v}$ enters only the escape path
  (`segment_escape.py`, endpoints $\mathbf r_{\rm mid}\mp\tfrac12L\hat{\mathbf v}$).
  It is a working null test.
- The `case_channeling_spectrum` stub only replaces `out["spec_channeling"]`
  (`runner/emission.py`, line 368, imported at call time). `spec` is unaffected.

### 2.6 Records against the ledger row

| Ledger statement | Records | Verdict |
| --- | --- | --- |
| normal rms 1.35 mrad, mean 3.7 µrad | 1.351 mrad, 3.67 µrad | match |
| in-plane rms 1.69 mrad | 1.6846 mrad | rounds to **1.68** |
| arms 2.84e-6, 2.58e-6 | 2.838e-6, 2.581e-6 | match |
| forward `spec` −1.86%, `brem` −1.93% | −1.867%, −1.933% | match (−1.87% rounds) |
| Bragg/±1/γ `spec` −0.03…+0.37%, `brem` +0.09…+0.14% | −0.03…+0.37%, +0.09…+0.14% | match |
| polar `spec` −0.28…+0.04%, `brem` ≤0.05% | −0.28…+0.04%, 0.02–0.04% | match |
| characteristic ≤3.6e-6 | 3.58e-6 | match |
| $(\gamma\sigma)^2$ 2.1e-3 / 3.3e-3 | 2.14e-3 / 3.33e-3 ($\gamma=34.27$) | match |
| $q$ 0.84–1.24 for shifts > 1e-3 | 0.84–1.24 | match |
| outliers only for shifts ≤ 6e-4 | (0 0 −4)+out normal `spec` 6.89e-4 has $q=-0.70$; polar-37.5 normal `spec` 6.19e-4 has $q=1.31$ | **≤7e-4** |
| all-row/zone ratio 0.75–0.93 | 0.77–0.93 | lower end **0.77** |
| polar probes "JUNGFRAU" | point directions (`photons/sr/electron`), not the 2.05e-4 sr square integration | wording |

### 2.7 Corrected forward estimate

Both moment corrections scale the quadratic shifts as $\sigma^2$, which holds to
$q\approx0.96$–0.98 except for normal-axis `brem` ($q=0.84$).

| Forward | Ledger | Corrected $\sigma_x^2$ ($\times1.045$), $\sigma_y^2$ ($\times1.077$ / $\times1.103$) | Additionally applying the $q$-implied Gaussian factor $(1+3r)/(1+r)$ |
| --- | --- | --- | --- |
| `spec` | −1.87% | −1.99% / −2.02% | ≈ −1.98% |
| `brem` | −1.93% | −2.05% / −2.08% | ≈ −1.94% |

The forward omission is therefore $\approx2\%$ (1.9–2.1% across the stated
estimator choices), not strictly $\le2\%$. The probe and polar bounds ($\le0.5\%$)
have a factor of more than 1.3 margin even after a 10% moment correction (largest
probe magnitude 0.37% → 0.41%).

## 3. Cheap filters

- **Units:** angles are $(\kappa+mg)/p$ in Å$^{-1}$/Å$^{-1}$, giving rad;
  $\eta$ is in Å and $p\eta$ is dimensionless; shifts are ratios of band yields of
  the same unit. Pass.
- **Limits:** $\sigma\to0$ gives a zero shift. A quadratic response gives $q=1$
  (forward and Bragg-arm $q=0.94$–1.05). Isotropic characteristic emission is null
  to $3.6\times10^{-6}$. $\eta\to0$ recovers $\mu_2$ only in the limit (§1.4).
  Pass.
- **Signs/conventions:** cosine transform and $-2/(p\eta)^2$ are correct; the
  rotation is exact; the odd term has the correct form; the cross term vanishes by
  symmetry; the slab frame is consistent. Pass.

## 4. Verdict

The method is rederived, and the stated numbers match the records apart from three
small rounding or wording items. Two estimator biases (8-depth trapezoid,
unconverged out-of-plane arm) both bias the shift low by 4–10% in $\sigma^2$. They
move the forward value from 1.87–1.93% to about 2.0–2.1% before the measured
sub-quadratic response pulls `brem` back to about 1.94%. The forward statement
should read $\approx2\%$ ($\le2.1\%$) rather than $\le2\%$. The $\le0.5\%$ probe
bound holds with margin. No blocking discrepancy in the physics or the estimator
algebra was found.
