# `cr-retarded-photon-arrival` — far-field retarded arrival and time-resolved window spectra

- **Claim**: `cr-retarded-photon-arrival` — `montecarlo/channeling/temporal_radiation.py::{retarded_delay_fs,projected_arrival_fs,_time_weights,planar_time_resolved_spectrum,axial_time_resolved_spectrum}` — Liénard–Wiechert retarded time $t = t_{\rm ret} + \lvert\mathbf R - \mathbf r(t_{\rm ret})\rvert/c$ in the far field (Jackson 3rd ed., §14.1); Doppler denominator `cr-doppler-line-energy`; yields `cr-yield-assembly`; helicity `cr-axial-helicity-emission`.
- **Verifier**: fresh context; derivation written before the bodies were read.

## Independent derivation

### Arrival time

An electron enters at $(\mathbf r_{\perp i}, 0)$ at $t_i$ and moves at constant $\beta c\,\hat{\mathbf z}$. It is at depth $z$ at $t_e = t_i + z/(\beta c)$. A photon emitted there reaches the far-field detector at $R\hat{\mathbf n}$ at

$$
t = t_e + \frac{\lvert R\hat{\mathbf n} - \mathbf r\rvert}{c}
\simeq t_e + \frac{R}{c} - \frac{\hat{\mathbf n}\cdot\mathbf r}{c} + O\!\left(\frac{r^2}{Rc}\right).
$$

With $\mathbf r = (\mathbf r_{\perp i}, z)$, $\hat{\mathbf n} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$, and the common $R/c$ removed,

$$
\boxed{t_{\rm obs} = t_i - \frac{\hat{\mathbf n}_\perp\cdot\mathbf r_{\perp i}}{c} + \tau(z),\qquad
\tau(z) = \frac{z\,(1 - \beta\cos\theta)}{\beta c}.}
$$

The transverse term has a minus sign: an electron displaced toward the detector is heard earlier. $\theta$ is measured from the velocity, which is the beam $z$ axis.

Cancellation-free denominator: $1 - \beta\cos\theta = (1-\beta) + 2\beta\sin^2(\theta/2)$ with $1 - \beta = 1/[\gamma^2(1+\beta)]$.

Limits:

- $\tau(0) = 0$;
- $\theta = 0$ gives $\tau = z/[\gamma^2(1+\beta)\beta c]$;
- $\theta = \pi/2$ gives $z/(\beta c)$;
- $\theta = 0$ removes the transverse term.

### Window spreading

If emission is taken as uniform in $z$ on $[z_{j-1}, z_j]$, then $\tau$ is linear in $z$ and $t_{\rm obs}$ is uniform on $[\ell_{ij}, \ell_{ij} + \Delta_j]$. Here $\ell_{ij} = t_i - \hat{\mathbf n}_\perp\cdot\mathbf r_{\perp i}/c + \tau(z_{j-1})$ and $\Delta_j = \tau(z_j) - \tau(z_{j-1}) > 0$. The fraction in time bin $[a_m, a_{m+1}]$ is $F(a_{m+1}) - F(a_m)$ with $F(t) = \operatorname{clip}((t - \ell_{ij})/\Delta_j, 0, 1)$. Outside mass is $1 - \sum_m$. Summed over records of node $k$ with weights $w_i$, it conserves $P(k)$.

### Wave-train RMS

Window $j$ carries emission weight $p_j$, uniform on $[a_j, b_j] = [\tau(z_{j-1}), \tau(z_j)]$. A uniform segment has

$$
\langle t\rangle_j = \frac{a_j + b_j}{2},\qquad
\langle t^2\rangle_j = \frac{b_j^3 - a_j^3}{3(b_j - a_j)} = \frac{a_j^2 + a_j b_j + b_j^2}{3},
$$

so $\sigma^2 = \sum_j p_j\langle t^2\rangle_j/\sum p - (\sum_j p_j\langle t\rangle_j/\sum p)^2$. The span is $\tau(z_J) - \tau(z_0)$.

### Window spectra by cumulative differences

Let $S(L) = \int_0^L \mathrm dz\, \mathcal R(z)\, e^{-\tau_{\rm abs}(z)}\otimes F$. Here $\mathcal R$ is the line-rate vector from $\rho(z)$, $\tau_{\rm abs}(z)$ the attenuation from depth $z$ to the exit of the **full** slab, and $F$ the line-shape map from yields to spectrum. Then $S(z_j) - S(z_{j-1})$ is window $j$'s emission exactly, provided that:

- (i) $\rho(z)$ on $[0, z_{j-1}]$ is identical in both solves;
- (ii) the quadrature is additive at $z_{j-1}$, which holds for the trapezoid rule on a nested grid that contains $z_{j-1}$;
- (iii) $\tau_{\rm abs}(z)$ does not depend on $L$, which needs the full-slab layer;
- (iv) $F$ is linear in yields and independent of $L$ (no finite-length line width).

For the axial path, cumulative trapezoid sums of the helicity coherency $T_{ab}$ are additive by construction. The photon count is linear in the trace, so $\Delta N\,\Delta T_{++}/\operatorname{tr}\Delta T = \kappa\,\Delta T_{++}$ exactly.

## Source-to-code diff

| Item | Derivation | Code | Result |
|---|---|---|---|
| $\tau(z)$ | $z(1-\beta\cos\theta)/(\beta c)$ | `depth * doppler_denominator / (beta * C_ANG_PER_FS)` (temporal_radiation.py:69-72) | match |
| $1-\beta\cos\theta$ | $\gamma^{-2}/(1+\beta) + 2\beta\sin^2(\theta/2)$ | bloch.py:265-268 | match |
| $\beta$ | $\sqrt{1-\gamma^{-2}}$, $\gamma = 1 + T/m_ec^2$ | :50-54 | match |
| transverse | $-\hat{\mathbf n}_\perp\cdot\mathbf r_\perp/c$, $\hat{\mathbf n}_\perp = \sin\theta(\cos\phi, \sin\phi)$ | :83-85 | match (sign) |
| CDF spreading | $\operatorname{clip}((t - \ell)/\Delta, 0, 1)$, then diff | :142-148 | match |
| outside | $1 - \sum_m$ | :148 | match |
| RMS | $(a+b)/2$, $(a^2+ab+b^2)/3$ | :253-257 | match |
| span | $\tau(z_J) - \tau(z_0)$ | :279 | match |
| (i) nested $\rho$ | same propagation steps | `_depth_history`: `seg = L/ceil(L/(inner dz))`, sequential `propagate(seg, step)` (emission.py:622-645); edges must be multiples of `inner*dz` (:286-294) | match (see finding 1) |
| (ii) quadrature | trapezoid on nested $z$ | `np.trapezoid(values, z)` (emission.py:547) | match |
| (iii) full slab | $(0, L_{\rm full}\,n_z)$ | forced when `layers is None` (temporal_radiation.py:359-366); `_stack_tau` evaluated at the same $z$ | match |
| (iv) $F$ linear | modes $w_m \propto (a^\dagger V)(V^{-1}s)$, $s \propto$ yields; centres and widths from $K$ eigenvalues only | emission.py:146-153, :262-273; `energy_spread_eV = 0` forced (:375) | match |
| Stokes | difference of $I Q$, $I U$ numerators | `intensity * stokes` with `stokes = q_sum/spectrum` (:384-391, emission.py:944-945) | match |
| axial windows | difference of cumulative per-line photons and $T_{\pm\pm}$ | :478-488; `photons(d)` $\propto \operatorname{tr}$ `cumulative[d]` via `dipole_rate_per_ang`, which is linear in $\lvert x\rvert^2$ (axial_channeling.py:171-175) | match |
| axial delay | forward $\theta = 0$ | `theta=0.0` (:495) | match (see finding 3) |

No term diverges.

## Numerical checks

Scripts: the verifier's disposable `check_retarded.py`, `check_planar_windows.py` and `check_nesting_float.py` (not committed).

- **Exact geometry.** $t_{\rm obs}$ (code) against a 60-digit evaluation of $t_e + \lvert R\hat{\mathbf n} - \mathbf r\rvert/c - R/c$ with $R = 10^{16}$ Å. The grid was $T \in \{30\ {\rm keV}, 5, 18.5, 1000\ {\rm MeV}\}$, $\theta \in \{0, 10^{-3}, 0.3, \pi/2, 2.5\}$, $\phi \in \{0, 1.1\}$, $\mathbf r_\perp = (2, -1.3)\times10^4$ Å and $z = 5\times10^4$ Å. Max deviation is $5.1\times10^{-11}$ fs, consistent with the far-field remainder.
- **Limits.** $\tau(0) = 0$. The forward limit agrees to $0$ relative and $\theta = \pi/2$ to $4\times10^{-16}$.
- **Time weights against Monte Carlo** of uniform-depth emission: 40 weighted records, 3 nodes, 3 windows, $\theta = 0.2$, $\phi = 0.7$. Max $|\Delta| = 8\times10^{-5}$ against MC noise of about $2\times10^{-3}$. Outside mass matches and probability is conserved to $6\times10^{-17}$.
- **Piecewise-uniform RMS** formula: $1.59086$, against $1.59052$ sampled from $2\times10^6$ draws.
- **Real Si(110), 18.5 MeV** (`n_cells=2`, `max_order=5`, `dz=100`, `sample=500`), $L = 1500$ Å, windows at $0/500/1000/1500$ Å, $\theta = 2$ mrad:
  - 500 Å and 1000 Å cumulative histories equal the 1500 Å prefix **bitwise** in $z$ and line rates;
  - `EmissionLines.spectrum` linearity error $2\times10^{-16}$;
  - `time_integrated` equals one default-layer full solve to $7\times10^{-18}$;
  - full-slab against short-slab exit absorption for window 0 differs by $5\times10^{-3}$, so the override matters;
  - an independently assembled window $[500, 1000]$ (trapezoid of the full history's rates over that interval, `lines.spectrum`, forward dipole pattern; no absorption) matches the cumulative difference to $8.5\times10^{-16}$.
- Anchors are green: delay limits, CDF spreading, the stub chirp $E_c \propto T^{3/2}$ (slope $1.5E_ch/T$), real planar 17/18.5/20 MeV centroids against direct solves, and axial helicity per window.

## Verdict

- **Filters**: units pass (Å, fs, $c = 2997.92458$ Å/fs); limits pass ($\tau(0)$, forward, $\pi/2$, no transverse term forward, arrival spread conserving the integral); signs/conventions pass (minus $\hat{\mathbf n}_\perp\cdot\mathbf r_\perp$, $\theta$ from the beam $z$ axis, larger $t$ later).
- **Re-derivation**: `matches`.
- **Verdict**: `rederived`.

## Findings (non-blocking)

1. *Float edge in nesting guard.* `_check_prefix_depths` (temporal_radiation.py:286-294) accepts edges within `atol=1e-9` of a multiple of `inner*dz`. `_depth_history` uses `ceil(L/(inner*dz))` (emission.py:623). A ratio such as $3.0000000000000004$ passes the check but gives 4 segments, so the grid is not nested and the window differences pick up a trapezoid-grid mismatch. `check_nesting_float.py` found 300 accepted but non-nested cases, all with non-dyadic `dz` such as $0.1$ or $0.3$ Å. Integer-Å defaults are unaffected. Suggest mirroring the `ceil` logic in the guard or snapping edges to `k*inner*dz`.
2. *Anchor gap.* The ledger check "cumulative-depth windows telescope to the full spectrum" is pinned only with the stub `channeling_spectrum` (`test_depth_windows_are_cumulative_differences_through_the_full_slab` uses `stub_planar`). Real-solver nesting is verified above but not anchored. Suggest a small real Si(110) multi-window anchor.
3. *Axial timing scope.* The axial path bins 4π-integrated line yields but times them with the forward ($\theta = 0$) delay. Photons at $\theta \sim 1/\gamma$ have about twice the forward $\tau$. The time binning and wave-train RMS are exact for a forward detector only. The docstring says the line energies are forward; suggest saying the retardation is forward as well.
4. *Geometry convention.* $t_{\rm obs}$ assumes the entrance face is $z = 0$ and $\mathbf v \parallel \hat{\mathbf z}$. With a tilted `surface_normal` (planar), or a tilted analytic beam from `sample_incident_phase_space(..., tilt_polar_rad=...)`, the projection should use the full 3-D entry point, and $\theta$ in $\tau$ the angle to $\mathbf v$. Neither is guarded. The Assumptions cover "velocity along the slab axis"; suggest an explicit refusal or note.
5. *Wording.* The ledger Claim reads "Emission in each depth window **is** uniform in depth". It is a modelling assumption, already listed under Assumptions. Suggest "is taken as uniform".

## Resolution (implementer, 2026-11-07)

1. `_check_prefix_depths` mirrors `emission._depth_history` and also requires `ceil(L/(inner dz)) == rint(L/(inner dz))`; anchored with a `dz = 0.1` Å float edge.
2. `test_real_planar_windows_telescope_to_the_full_slab` pins the real Si(110) multi-window decomposition against one full-slab solve.
3. The axial docstring, design note and ledger say the retardation is forward-only.
4. A non-default planar `surface_normal` is refused; tilted analytic beams are documented as outside the timing model.
5. The ledger Claim says emission is *taken as* uniform within a window.
