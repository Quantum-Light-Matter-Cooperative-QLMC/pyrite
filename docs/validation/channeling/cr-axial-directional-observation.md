# `cr-axial-directional-observation`: directional scoring of unadmitted axial fibre cases

Fresh-context verification (2026-11-13) at commit `d0be8db36` on branch
`issue-1148-channeling-detector-components`. The verifier did not write the
implementation. It read the ledger row, the anchored
[`cr-axial-fiber-angular`](cr-axial-fiber-angular.md) derivation and the
docstrings, and derived the expectations below before reading the function
bodies. Scripts (not committed):
`scratch/verify_directional_observation.py` and
`scratch/verify_directional_absorption.py`.

**Verdict: `rederived`, with non-blocking findings.** No sign, factor,
phase convention, unit or estimator differs from the independent derivation.
The findings are an understated line bias of the linear quadrature, a wrong
order in the forward-only justification, and a crash for fewer than two line
electrons.

The claim adds no new emission physics. The polarization matrix, the
resonance-evaluated angular factor and the oblique attenuation belong to
`cr-axial-fiber-angular`, and the two-current estimator to
`cr-axial-fiber-regression`. This page checks the new parts: the depth
quadrature, the per-direction frame, the cone average, the batch error and the
admission gating.

## Independent derivation

### Exact weights of the linearly interpolated integrand

The anchored source is a double depth integral

$$
S_{\lambda\lambda'}(E,\mathbf n)=\frac{\alpha E}{4\pi^2(\hbar v)^2}
\int\!\!\int dz\,dz'\,e^{i\omega(z-z')}\,a(z)a(z')
\sum_{ab}m_{\lambda a}m_{\lambda'b}G_{ab}(z,z'),
\qquad \omega=\frac{E(1-\beta\cos\theta)}{\hbar v},
$$

with $a(z)=e^{-\mu(L-z)/(2\cos\theta)}$. The sign of the phase is the anchored
one: with lowering currents $G\propto e^{-i\Delta(z-z')/\hbar v}$ the
resonance is at $\omega=\Delta/\hbar v$.

Replace $f(z)=a(z)G(z,\cdot)$ by its piecewise-linear interpolant
$\sum_if_i\,\mathrm{hat}_i(z)$ on the recorded depths. The integral against
$e^{i\omega z}$ is then exactly $\sum_ic_i(\omega)f_i$ with
$c_i=\int\mathrm{hat}_i(z)e^{i\omega z}dz$. In two dimensions the tensor
product of hats is bilinear interpolation of $a\,a\,G$, so the double sum is
$\sum_{ij}c_ic_j^*\,a_ia_j\,G(z_i,z_j)$.

On an interval $[z_k,z_k+h]$ with $t=(z-z_k)/h$ and $u=\omega h$:

$$
\int_0^1(1-t)e^{iut}dt=\frac{1+iu-e^{iu}}{u^2}=\frac{i}{u}-\frac{e^{iu}-1}{u^2},
\qquad
\int_0^1t\,e^{iut}dt=\frac{(1-iu)e^{iu}-1}{u^2}=\frac{e^{iu}}{iu}+\frac{e^{iu}-1}{u^2}.
$$

The left node receives $h\,e^{i\omega z_k}$ times the first and the right node
times the second. Their Taylor series are

$$
\tfrac12+\tfrac{iu}{6}-\tfrac{u^2}{24}-\tfrac{iu^3}{120}+\dots,\qquad
\tfrac12+\tfrac{iu}{3}-\tfrac{u^2}{8}-\tfrac{iu^3}{30}+\dots
$$

The closed forms cancel catastrophically as $u\to0$: their rounding error is
about $\epsilon/u^2$.

- **$\omega=0$.** Both factors are $1/2$, so $c_i$ are the trapezoid weights $w_i$ exactly. For $\omega\ne0$, $c_i$ differs from $w_ie^{i\omega z_i}$ at first order in $u$.
- **Uniform grid.** Interior nodes get $c_i=h\,\mathrm{sinc}^2(\omega h/2)\,e^{i\omega z_i}$, the Fourier transform of a hat.

### What the quadrature does to a line

Take one line, $G=e^{-i\kappa(z-z')}$ with $\kappa=\Delta/\hbar v$, on a uniform
step $h$. Up to end corrections, the sum is $\mathrm{sinc}^2(\omega h/2)$ times
the trapezoid sum. That sum is periodic in $\omega$ with period $2\pi/h$, so it
carries replicas at $\omega=\kappa+2\pi m/h$.

- **Replicas.** At replica $m$ the factor is $\sin^2(\kappa h/2)/(\kappa h/2+\pi m)^2$. Relative to the line, the replica intensity is $[\sin(x/2)/(x/2+\pi m)]^4/\mathrm{sinc}^4(x/2)$ with $x=\kappa h$, i.e. about $(x/2\pi m)^4$. This is the removal of aliasing.
- **The line itself.** Its peak intensity is multiplied by $\mathrm{sinc}^4(\kappa h/2)\simeq1-(\kappa h)^2/6$. The trapezoid main peak is exact for a sampled pure exponential: its area is $2\pi L$, the continuum value. The linear quadrature therefore biases every line **low**, by about $(\Delta\,\Delta z/\hbar v)^2/6$, independent of $\theta$. By Parseval, the lost weight is the interpolant's modulus deficit between nodes, $1-\tfrac23\sin^2(\kappa h/2)$.

### Direction, attenuation, cone and units

- **Direction.** $(\theta,\phi)$ must be the angles of $\mathbf n$ in the profile frame whose $x,y$ are those of the currents $J_{x,y}$ and whose $z$ is the axis. If the frame's columns are the lab images of the crystal $x,y,z$, then the local direction is $F^{\mathsf T}\mathbf n$.
- **Attenuation.** A photon born at axial depth $z$ leaves a laterally infinite slab whose exit face is normal to the axis after a path $(L-z)/\cos\theta$. The amplitude is therefore $e^{-\mu(L-z)/(2\cos\theta)}$, i.e. the on-axis amplitude raised to $1/\cos\theta$. The path along the photon direction is needed (size below).
- **Cone.** A detector with half angle $\Delta\theta_{\rm obs}$ reads the solid-angle average $\Omega^{-1}\int_{\rm cone}S\,d\Omega$, with $d\Omega=\sin r\,dr\,d\chi$ about $\hat{\mathbf n}$. Each node is converted to $(\theta,\phi)$ about the axis separately.
- **Units.** $[c_i]=$ Å, $[\hbar v]=$ eV Å and $G$ is a mean per incident electron. So $\alpha E\,|c|^2/(\hbar v)^2$ is in eV$^{-1}$ per sr: photons/(eV sr incident electron), and a solid-angle average keeps that unit.
- **Forward-only cutoff.** At $\theta\ge\pi/2$ the resonance $E=\Delta/(1-\beta\cos\theta)\le\Delta\lesssim100$ eV lies below any keV grid. On the grid only off-resonance tails are dropped. The line-area density relative to the forward peak is $\tfrac12(\beta^2+1)(1-\beta)^2\approx\gamma^{-4}/4$ (not $\gamma^{-6}$). The backward hemisphere holds a fraction $O(\gamma^{-2})$ of the Lorentz-invariant count.
- **Batch error.** Batches $b$ of sizes $n_b$ (contiguous, $\sum n_b=N$) with iid primaries. Then $s^2=\sum_bn_b(\bar S_b-\bar S)^2/(B-1)$ is unbiased for the per-primary variance, and $s^2/N$ for the variance of the mean. $S$, and its cone average, is linear in $G$. So the per-batch spectra can be built from batch-mean correlations and cone-averaged before the spread is taken. $\bar S$ must equal $\sum n_b\bar S_b/N$.

## Comparison with the implementation

| Item | Expectation | Code | Result |
|---|---|---|---|
| Weights | closed forms and Taylor series above | `linear_depth_weights`: `left = i/u - (e-1)/u^2`, `right = e/(iu) + (e-1)/u^2`, base `h e^{i omega z_k}`; Taylor to $u^2$ for $\lvert u\rvert<10^{-3}$ | match |
| Phase sign | $e^{+i\omega z}$, as the anchored trapezoid | `exp(1j * omega * z)`; same `_phase_rate` | match |
| Oblique path | amplitude$^{1/\cos\theta}$ | on-axis `exp(-0.5 mu (z[-1] - z))` (with `depths[-1]` = slab thickness), raised to `1/cos(angle)` in `polarization_spectrum` | match |
| Frame | $F^{\mathsf T}\mathbf n$, columns = crystal $x,y,z$ | `record["frame"].T @ direction`, `zone.frame_lab` (axis = slab normal to $10^{-9}$, enforced by `axial_zone_frame`) | match |
| Cone | solid-angle average | `_direction_nodes`: Gauss–Legendre in $r$ with weight $\sin r$, uniform $\chi$, normalized | match |
| Backward | zero for $\theta\ge\pi/2$, weight kept in the average | `continue` (node contributes zero), weights not renormalized | match |
| SEM | $\sqrt{s^2/N}$ from the cone-averaged per-batch spectra | per-batch traces accumulated with the same weights; `sizes` from `bincount`; `mean` from the full correlation $=\sum n_b\bar G_b/N$ | match |
| Prefactor | $\alpha E/(4\pi^2(\hbar v)^2)$ | unchanged anchored line | match |

## Numerical checks

| Check | Result |
|---|---|
| `linear_depth_weights` vs mpmath hat integrals (40 digits), random non-uniform grid, $u_{\max}\in[-50,40]$, including both sides of the $10^{-3}$ switch | max relative error $7.8\times10^{-11}$ (at the switch: Taylor truncation $\sim u^3/30$ and closed-form cancellation $\sim\epsilon/u^2$ balance there) |
| Uniform interior weights vs $h\,\mathrm{sinc}^2(\omega h/2)e^{i\omega z_i}$ | $\le2.3\times10^{-15}$ |
| `polarization_spectrum(quadrature="linear")` vs own $\sum_{ij}c_ic_j^*a_i^{1/\cos\theta}a_j^{1/\cos\theta}G$ with my mpmath weights, random complex $G$, $\theta$ up to 1.2 rad | $6.1\times10^{-10}$ at every $\theta$, which is the CODATA $\alpha$ difference between my constant and `ALPHA_FS` |
| Cone nodes | weights sum to 1; weighted mean direction $=\hat{\mathbf n}$ to $10^{-16}$; $\langle\cos r\rangle=(1+\cos\Delta\theta)/2$ to $10^{-16}$ |
| Batch SEM, `array_split` of 13 into 8 (unequal sizes), $2\times10^5$ trials | $E[s^2]/\sigma^2=0.998$ |

One line on the default 15 Å step, 3.5 MeV ($\gamma=7.849$, $\hbar v=1957.2$ eV Å), $L=3000$ Å:

| $\Delta$ (eV) | $\kappa h$ | linear peak / exact | $\mathrm{sinc}^4(\kappa h/2)$ | area ratio ($\pm40/L$ window) | replica / peak | $(x/2\pi)^4$ form |
|---|---|---|---|---|---|---|
| 30 | 0.230 | 0.9912 | 0.9912 | 0.9944 | $1.6\times10^{-6}$ | $1.6\times10^{-6}$ |
| 40 | 0.307 | 0.9845 | 0.9845 | 0.9876 | $4.7\times10^{-6}$ | $4.7\times10^{-6}$ |
| 80 | 0.613 | 0.9391 | 0.9391 | 0.9421 | $6.3\times10^{-5}$ | $6.3\times10^{-5}$ |

On axis a line at $E$ has $\Delta\approx E/2\gamma^2$. A 10 keV line ($\Delta\approx81$ eV) is therefore low by about 6% at the peak against the trapezoid source, and a 2 keV line by about 0.4%. The anchor's band-integrated 1.0% (si-fibre/B) is consistent with this, because the low-energy lines dominate the band.

Oblique attenuation in 0.3 µm Si: depth-averaged intensity along $\mathbf n$ relative to along the axis.

| $\theta$ | 800 eV | 1850 eV (above K edge) | 3 keV | 10 keV |
|---|---|---|---|---|
| 0.1 rad | −0.05% | −0.06% | −0.02% | −0.001% |
| 30° | −1.5% | −1.7% | −0.5% | −0.02% |
| 1.2 rad | −15% | −17% | −5.7% | −0.2% |

The oblique path is therefore needed beyond a few tenths of a radian, and the code applies it. It scales with $\mu L$ for thicker unadmitted slabs.

The anchor `tests/montecarlo/channeling/test_axial_directional.py` passes with `--slow` under the 8 GiB cap: 15 passed in 296 s.

## Admission

| Requirement | Evidence | Result |
|---|---|---|
| Directional never matches an admitted entry | entries carry `observable="on-axis"`; `match_axial_case` compares `observable`; `axial_observable` returns `"directional"` for nonzero `theta_obs_rad`, `dtheta_obs_rad`, `domega_sr` or a `PlanarDetector` (lowering passes `directional=True`) | pass |
| Runs only under `accept-inaccurate` | `_resolve_axial` falls back to `unadmitted_axial_case` only on the opt-in; anchor `test_directional_observation_is_unadmitted_and_needs_the_opt_in` (off-axis, aperture, planar) | pass |
| Forged record | an admitted key with `observable="directional"` re-resolves through `match_axial_case(..., axial_observable(case))` and the record differs from the lookup, so it refuses | pass (by reading) |
| Admitted on-axis output unchanged | `batches=0` and no `directional` record unless `is_directional`; `batches` adds no draws (accumulation only); the anchor compares the persisted source and fate fractions of the same geometry bit for bit | pass |
| Refusal before transport | `refuse_axial_directions` runs before `_transport_case` in `run_case` and `run_case_directions`, and again in `directional_outputs`; anchor monkeypatches `_transport_case` to fail | pass |

## Findings

1. **Line bias of the linear quadrature (wording, non-blocking).** The row's "relative error $O((\Delta_{fi}\Delta z/\hbar v)^2)$" is right in order but hides the sign and size. Lines are low by $\approx(\Delta_{fi}\Delta z/\hbar v)^2/6$ at any $\theta$; the trapezoid main peak has no such bias. At the default 15 Å step this is about 6% for a line at 10 keV on axis (81 eV transition). The on-axis limiting case "within 2%" is band-integrated, not per bin. A $\theta=0$ directional (for example `PlanarDetector`) spectrum is therefore systematically below the admitted on-axis source at the top of the grid. A smaller `step_ang` would reduce it quadratically. Optional: use the trapezoid where its phase-step cap holds.
2. **Forward-only order (wording, non-blocking).** The module docstring and the row say the pattern beyond 90° is "$O(\gamma^{-6})$ of the forward peak". The line-area angular density at 90° is $\approx\gamma^{-4}/4$ of the forward value ($6.6\times10^{-5}$ at 3.5 MeV, against $\gamma^{-6}=4.3\times10^{-6}$). The backward hemisphere carries 0.6% of the count. The real justification is kinematic: backward resonances lie at $E\le\Delta\lesssim100$ eV, below any keV grid, so only off-resonance tails are dropped.
3. **Fewer than two line electrons (robustness, non-blocking).** `batches=min(DIRECTIONAL_BATCHES, count)` is 1 for one line electron. `fiber_on_axis_source` then raises `ValueError("batches must be 0 or between 2 ...")` during transport, so a one-history directional run fails late with an internal message. For `count = 0`, `batches=0` and `directional_record` would read `None` batch arrays. Neither produces a wrong number. For $N_e<2$, a pre-transport refusal or a NaN SEM would be clearer.
4. **SEM degrees of freedom (note).** With $B=\min(8,N_e)$ the SEM has $B-1\le7$ degrees of freedom, and its square root is slightly biased low. The row says "standard error", which is acceptable; the existing note on the weak $C_4$ check covers the 4-electron anchor.
