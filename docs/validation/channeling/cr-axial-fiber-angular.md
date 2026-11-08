# `cr-axial-fiber-angular`: off-axis polarization source of the fibred absorbing axial model

**Status:** see the [ledger row](../ledger-channeling.md#cr-axial-fiber-angular). Task-owner derivation (2026-11-08, #992 Plan item 3; revised the same day after the Si runs below). Fresh-context verification is recorded below it.

## Problem

`cr-axial-fiber-regression` estimates the source only for photons along the
axis $z$. There the photon carries no transverse momentum, and only the
transverse velocity couples to the field. For a direction
$\mathbf n=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$, three things
change:

- the photon's transverse wavevector $\mathbf q=(E\sin\theta/\hbar c)\hat n_\perp$ gives the electron a recoil;
- the longitudinal velocity $v_z\simeq\beta c$ couples through $e_z\ne0$;
- the Doppler phase changes.

The polarization basis is $\mathbf e_1=\hat\theta=(\cos\theta\cos\phi,\cos\theta\sin\phi,-\sin\theta)$ and $\mathbf e_2=\hat\phi=(-\sin\phi,\cos\phi,0)$.

## Velocity form at transverse dipole order

The first-order amplitude of a lowering transition $i\to f$ (Jackson §14;
Baier–Katkov–Strakhovenko, ch. 2) is

$$
\mathcal A_{\lambda,fi}=e_{\lambda\perp}\cdot\langle f|\tfrac{\mathbf v_\perp}{c}e^{-i\mathbf q\cdot\mathbf r}|i\rangle
+e_{\lambda z}\,\beta\,\langle f|e^{-i\mathbf q\cdot\mathbf r}|i\rangle .
$$

The transverse correction to $v_z$, of order $O(E_\perp/pv)$, is dropped, as in
`transition_operator` (`cr-dipole-rate`).

From $\mathbf v=(i/\hbar)[H,\mathbf r]$ in a fibre's retained eigenbasis,
$\mathbf r_{fi}=i\hbar c\,\mathbf J_{fi}/\Delta_{fi}$, with $\Delta_{fi}=E_i-E_f>0$
and $\mathbf J=\mathbf v/c$. In Bloch language this is the k·p result
$\langle u_{f,\kappa-q}|u_{i,\kappa}\rangle\simeq\hbar\mathbf q\cdot\mathbf v_{fi}/\Delta$.

At first order in $\mathbf q\cdot\mathbf r$, with $e_{1z}=-\sin\theta$ and
$e_{1\perp}=\cos\theta\,\hat n_\perp$:

$$
A_1^{\rm vel}=\hat n_\perp\cdot\mathbf J\left(\cos\theta-\beta\frac{E\sin^2\theta}{\Delta}\right),\qquad
A_2=\hat\phi\cdot\mathbf J .
$$

The $e_z$ term is first order in $q$. It is still of order one at $\theta\sim1/\gamma$, because $E/\Delta\sim2\gamma^2$.

The neglected terms are $O(qa)$ relative, where $a$ is the bound-state size.

- At time-reversal-invariant carriers ($\kappa=0$ or the zone edge), the $O(q)$ correction is in quadrature with $\mathbf J$. A single line's intensity error is therefore $O((qa)^2)$, even without inversion symmetry. Cross terms between lines can still enter at first order. Flat bound bands behave the same way at any $\kappa$.
- At $\kappa\ne0$ in dispersive (above-barrier) bands, a first-order correction of relative size $O(qa)$ times the κ-asymmetry remains.
- Lines forbidden on axis, such as $ns\to1s$, acquire $O(qa)$ amplitudes off axis. These quadrupole and magnetic terms are omitted.

## Resonance-evaluated Liénard–Wiechert factor

The $E$-dependent part of $A_1^{\rm vel}$ equals its value at the line resonance $E(1-\beta\cos\theta)=\Delta$, plus a term proportional to the detuning:

$$
\beta\sin^2\theta\,\frac{E}{\Delta}=\beta\sin^2\theta\,\frac{1}{1-\beta\cos\theta}
+\frac{\beta\sin^2\theta}{1-\beta\cos\theta}\,\frac{E(1-\beta\cos\theta)-\Delta}{\Delta}.
$$

In a closed system the detuning term multiplies
$(E(1-\beta\cos\theta)-\Delta)\,e^{i(E(1-\beta\cos\theta)-\Delta)z/\hbar v}$, which is
$-i\hbar v\,\partial_z$ of the phase. It is therefore a pure entrance/exit
boundary term.

- **Boundary terms.** The vacuum side, which is not modelled, cancels them. Crystal transition radiation is negligible at keV. Left in, they make $|A|^2\alpha E/\kappa^2$ grow linearly in $E$ off resonance.
- **In-line correction.** Under dissipation the detuning term leaves a pointwise amplitude correction of $[\beta\sin^2\theta/(1-\beta\cos\theta)]\,\delta/\Delta$, with $\delta=E(1-\beta\cos\theta)-\Delta$. The coefficient is at most 2. In the line area the first order cancels, so the correction is second order in the relative width (fresh-context Lorentzian check: 2.7e-3 at θγ = 1 and FWHM/E = 0.03).

The module therefore uses

$$
\boxed{A_1=\hat n_\perp\cdot\mathbf J\,\frac{\cos\theta-\beta}{1-\beta\cos\theta},\qquad A_2=\hat\phi\cdot\mathbf J ,}
$$

the angular factor of the Liénard–Wiechert field of `dipole_far_field`
(`cr-dipole-far-field-amplitude`). It is not the integrated-by-parts
acceleration form off resonance: that form would also multiply both components
by $\Delta/(E(1-\beta\cos\theta))$. This choice keeps $\theta=0$ identical to
the on-axis estimator.

Fresh-context check on 819567a16: a Gaussian line of relative width 0.025, integrated over E and the cone θ ≤ 1.5, agrees with the exact single-line value to −9.5e-5. The residual comes from the αE prefactor.

The rest-frame identity $(\cos\theta-\beta)/(1-\beta\cos\theta)=\cos\theta'$ makes the factor exactly the boosted dipole pattern. One line integrated over the sphere gives the Lorentz-invariant count $4\alpha\gamma^2\Delta|\mathbf J|^2/(3\hbar v)$ per unit depth.

**Consequence.** The factor is independent of the transition. For an ensemble with $C_4$ symmetry about the axis ($G_{xx}=G_{yy}$, $G_{yx}=-G_{xy}$), the intensity is azimuthally uniform at this order, and only $\cos2\phi$ harmonics can appear in general.

### Rejected: velocity form with a recoil current

The first implementation (commit 6a6c624d1) regressed $\mathbf K=\mathbf J/\Delta$ alongside $\mathbf J$ and used $A_1^{\rm vel}$ at the photon energy $E$.

On Si⟨100⟩ at 3.5 MeV, 0.3 µm (jobs 5345 and 5346, 2000 primaries each) it produced an off-resonance continuum.

- At $\theta\gamma=0.5$ the continuum rose from about $1\times10^{-7}$ to $5\times10^{-7}$ photons/(eV sr e) over 1–10 keV, above the lines.
- At $\theta\gamma\ge2.5$ it diverged toward the 10 keV grid edge, which is also where the 15 Å depth step aliases.
- Near-degenerate pairs after carrier-moving jumps made $K$ heavy-tailed. The result was a spurious $-3.5\%$ $\cos2\phi$ contrast at exact alignment, where $C_4$ forbids it.

## Two-time source

The same velocity/c lowering currents as on axis are regressed
(`cr-axial-fiber-regression`), giving $G_{ab}(z_i,z_j)$. With
$A_\lambda=\sum_am_{\lambda a}(\theta,\phi)J_a$:

$$
S_{\lambda\lambda'}(E,\mathbf n)=\frac{\alpha E}{4\pi^2(\hbar v)^2}\sum_{ij}v_iv_j^*\sum_{ab}m_{\lambda a}m_{\lambda'b}G_{ab}(z_i,z_j),
\qquad v_i=w_iA_i(E)^{1/\cos\theta}e^{iE(1-\beta\cos\theta)z_i/\hbar v}.
$$

$S_{\lambda\lambda'}=\langle A_\lambda A_{\lambda'}^*\rangle$ (quantum regression). It is Hermitian because $G_{ab}(z_i,z_j)=G_{ba}(z_j,z_i)^*$.

- **On axis:** $m$ is a rotation, and $\mathrm{tr}\,S$ is `signed_on_axis_spectrum`.
- **Stokes:** the parameters refer to $(\hat\theta,\hat\phi)$, with $V=2\,\mathrm{Im}\,S_{21}$ as in `axis_stokes`. The frame is right-handed, so $V>0$ is positive helicity.
- **Attenuation:** the exponent $1/\cos\theta$ takes the on-axis path $L-z$ to a face normal to the axis.
- **Depth sampling:** the trapezoid depth sum is periodic in the phase rate with period $2\pi/\Delta z$, so line replicas alias in. The largest phase step $E(1-\beta\cos\theta)\Delta z/\hbar v$ is capped (`max_phase_step`, default 1 rad).
- **Ensemble error:** $S$ is linear in $G$. The size-weighted batch spread $s^2=\sum_bn_b(\bar S_b-\bar S)^2/(B-1)$ is an unbiased estimate of the per-primary variance, and $s^2/N$ of the variance of the mean.

## Assumptions and scope

- Inherits every assumption of `cr-axial-fiber-regression`.
- Transverse dipole order, with the time-reversal condition above.
- Resonance-evaluated angular factor: boundary terms are dropped. The in-line error is pointwise $O(\delta/\Delta)$ with coefficient at most 2, and second order in width for the line area.
- `max_phase_step` covers the photon grid only. A retained transition with $\Delta\,\Delta z/\hbar v$ near $2\pi$ would still alias into the grid. For Si this is negligible: $\Delta\lesssim100$ eV at 15 Å gives 0.76 rad.
- No refraction and no longitudinal recoil.
- Forward hemisphere only, with the exit face normal to the axis.

## Limiting cases and checks

| Case | Expected | Anchor |
|---|---|---|
| $\theta=0$, with kick and lattice jumps | $\mathrm{tr}\,S$ = `fiber_on_axis_source`; identical events and correlation | `test_on_axis_trace_and_draws_match_the_on_axis_source_with_jumps` |
| No jumps | map equals an explicit depth double sum; the square toy is $C_4$ symmetric | `test_map_is_an_independent_double_sum_and_c4_symmetric` |
| $C_4$-symmetric $G$ | azimuthally uniform intensity | `test_c4_symmetric_correlation_gives_azimuthally_uniform_intensity` |
| Real, circular and complex dipoles | amplitude = Liénard–Wiechert projection (1e-10) | `test_resonant_amplitude_is_the_lienard_wiechert_field` |
| Single line at resonance | $S=\alpha EL^2AA^*/(2\pi\hbar v)^2$; $V=0$; boosted-dipole area ratio | `test_single_line_resonance_is_the_lienard_wiechert_line` |
| Cone integral; undersampled phase | $2\pi(1-\cos\theta_\max)$; refusal | `test_solid_angle_integral`, `test_refusals` |

## Literature cross-check (qualitative)

Korotchenko, Kunashenko & Dabagov, Eur. Phys. J. C 82, 196 (2022) (Zotero `NH9SX2CA`) predict two features for Si⟨100⟩ at 10 MeV and incidence $\vartheta_c/5$:

- a four-fold azimuthal structure of the angular distribution;
- purely linear net polarization.

They give no measured data. Their matrix element is
$\alpha_{if}\propto(\kappa_y,\kappa_x,\ldots)\langle xy\rangle_{if}$ (their Eq. 16). That is first order in the photon's transverse momentum times a quadrupole moment, i.e. the $O(qa)$ term omitted here. At the order implemented, a $C_4$-symmetric ensemble is azimuthally uniform, so their four-fold structure cannot be tested without the next order in $\mathbf q\cdot\mathbf r$ or exact recoil (a companion in fibre $\kappa-\mathbf q$). Their linear-polarization statement can be compared.

## Fresh-context verification (2026-11-08)

Independent verifier, fresh context. I derived the amplitude before reading the implementation. I first reviewed the velocity/recoil form at 6a6c624d1, then re-checked the resonance-evaluated form at 819567a16. Scripts (not committed): `scratch/verify_angular_line.py` (single line through the code) and `scratch/verify_angular_bloch_toy.py` (self-contained exact-recoil toy).

**Verdict: `rederived`, with minor findings.** None of the findings changes a sign, factor, convention or estimator.

### Independent derivation

For $\mathbf n=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$, the real basis $\hat\theta,\hat\phi$ and $\mathbf q=\mathbf k_\perp$, the photon number is

$$
\frac{d^2N}{dE\,d\Omega}=\frac{\alpha E}{4\pi^2(\hbar v)^2}\Bigl|\int dz\,\mathbf e_\lambda\cdot\langle f|\tfrac{\mathbf v}{c}e^{-i\mathbf q\cdot\mathbf r_\perp}|i\rangle\,e^{iE(1-\beta\cos\theta)z/\hbar v}\Bigr|^2 .
$$

This is the classical $e^2\omega^2/(4\pi^2c)\,|\int\mathbf n\times(\mathbf n\times\boldsymbol\beta)e^{i\omega(t-\mathbf n\cdot\mathbf r/c)}dt|^2$ divided by $\hbar\omega\cdot\hbar$, with $dt=dz/v$.

- **Velocity current.** $\mathbf v=(i/\hbar)[H,\mathbf r]$ gives $\mathbf J_{fi}=-i\Delta_{fi}\mathbf r_{fi}/\hbar c$, so $\mathbf r_{fi}=i\hbar c\mathbf J_{fi}/\Delta_{fi}$. In a fibre, the k·p form $\langle u_{f,\kappa-q}|u_{i,\kappa}\rangle\simeq\hbar\mathbf q\cdot\mathbf v_{fi}/\Delta$ gives the same first-order term, even though $\mathbf r$ is not closed within a fibre.
- **Velocity form.** First order in $\mathbf q\cdot\mathbf r$, with $e_{\theta z}=-\sin\theta$, gives $A_\theta^{\rm vel}=\hat n_\perp\cdot\mathbf J(\cos\theta-\beta E\sin^2\theta/\Delta)$ and $A_\phi=\hat\phi\cdot\mathbf J$.
- **Resonance.** At $E(1-\beta\cos\theta)=\Delta$, $A_\theta^{\rm vel}$ becomes $\hat n_\perp\cdot\mathbf J\,(\cos\theta-\beta)/(1-\beta\cos\theta)$. That is the projection of the Liénard–Wiechert field $\mathbf n\times((\mathbf n-\boldsymbol\beta)\times\mathbf a)$, up to the common factor $-(1-\beta\cos\theta)^{-2}$. It matches the boxed form and `angular_coefficients` term by term, including the cancellation-free $1-\beta\cos\theta=(1-\beta)+2\beta\sin^2(\theta/2)$.
- **Detuning term.** The difference from the velocity form is $-\beta\sin^2\theta\,\hat n_\perp\cdot\mathbf J\,\delta/[\Delta(1-\beta\cos\theta)]$, with $\delta=E(1-\beta\cos\theta)-\Delta$. For an undamped line, $\delta\,e^{i\delta z/\hbar v}=-i\hbar v\,\partial_z e^{i\delta z/\hbar v}$, so the term is a pure boundary term, as the derivation states.
- **Neglected terms.** The next terms are $\langle f|\mathbf v_\perp(\mathbf q\cdot\mathbf r)|i\rangle$ and $\beta\langle f|(\mathbf q\cdot\mathbf r)^2|i\rangle/2$. Both operators are even, so they vanish between opposite-parity states and give $O(qa)$ amplitudes to lines that are parity-forbidden on axis.

### Shared machinery, checked on 819567a16

| Item | Result |
|---|---|
| Doppler phase | $v_i\propto e^{+iE(1-\beta\cos\theta)z_i/\hbar v}$ with lowering currents ($\propto e^{-i\Delta z/\hbar v}$), so the resonance is at $E(1-\beta\cos\theta)=\Delta$. At $\theta=0$ this is identical to `signed_on_axis_spectrum`. Pass. |
| Ordering of $S$ | $\lvert M\rvert^2=\sum_{ij}e^{i\omega(z_i-z_j)}\langle J_b^\dagger(z_j)J_a(z_i)\rangle$, and quantum regression gives $\mathrm{Tr}[J_a\Lambda(\rho J_b^\dagger)]=G_{ab}(z_i,z_j)$ for $z_i\ge z_j$. `_Companions._record` stores `values[i,j,a,b]` $=\langle w_{jb}\vert J_a u_i\rangle$ and the diagonal $\mathrm{Tr}[J_a\rho J_b^\dagger]$. The einsum `plc,ecd,pmd->plme` gives $S_{\lambda\lambda'}=\langle A_\lambda A_{\lambda'}^*\rangle$. Pass. |
| Hermitian completion | `G[i,j,c,d] = conj(G[j,i,d,c])` for $i<j$. A synthetic G rebuilt from its lower triangle agrees to $4\times10^{-16}$. Pass. |
| Code $S$ vs my own sum | One line with generic complex $\mathbf J$, $\gamma=2.957$, $\Delta=10$ eV, 2401 depths, 5 θ up to $4/\gamma$, 7 φ, 61 E: maximum relative error $5.6\times10^{-11}$. My amplitude was built separately as $\mathbf e_\lambda\cdot(\mathbf J,\beta E_{\rm res}\sin\theta\,\hat n_\perp\cdot\mathbf J/\Delta)$. Pass. |
| Stokes $V$ sign | $V=2\,\mathrm{Im}\,S_{10}=2\,\mathrm{Im}(A_\phi A_\theta^*)$. On axis at $\phi=0$ this is `axis_stokes`'s $2\,\mathrm{Im}(d_yd_x^*)$, because $\mathbf J\propto-i\mathbf r$ is a common phase. $\hat\theta\times\hat\phi=\mathbf n$, so $V>0$ is positive helicity in every direction. $V/I$: code 0.0378483387758636, mine 0.0378483387758627. Pass. |
| Oblique attenuation | `amplitude**(1/cos theta)` with $e^{-\mu(L-z)/2}$ gives the intensity path $(L-z)/\cos\theta$. Against my direct sum the relative error is $1.9\times10^{-11}$. Pass, provided `exit_path_ang` is the on-axis distance to a face normal to the axis; this is documented but not enforced. |
| Batch SEM | For iid primaries, $\mathrm{Cov}(\bar S_b,\bar S)=\sigma^2/N$ gives $E\sum_bn_b(\bar S_b-\bar S)^2=(B-1)\sigma^2$, so $s^2/N$ is unbiased for the variance of the mean. Stokes values are linear in $G$, so averaging per batch is exact. `count` equals $\sum n_b$ because `_primaries` yields one event per incident. Pass. The square root is slightly biased and has only $B-1$ degrees of freedom; the doc correctly says "variance". |
| Companion estimator | It is back to two components with the same `fiber_lowering_currents` as on axis, and `_primaries` is unchanged apart from removed plumbing. Draws are untouched (no RNG in `carry`). The on-axis-identity anchor passes. Pass. |
| Phase-step cap | The trapezoid sum on a uniform grid is periodic in the phase rate with period $2\pi/\Delta z$. The cap uses the largest $E$, $\theta$ and $\Delta z$, and $1-\beta\cos\theta$ increases with $\theta$, so the bound is right. Pass; see minor finding 3. |

### Lorentz-invariant count and boosted pattern

- I integrated my own line area $dN/(dz\,d\Omega)=\alpha\Delta\lvert A_{\rm res}\rvert^2/(2\pi\hbar v(1-\beta\cos\theta)^2)$ over $4\pi$. The result, $4.489580820332\times10^{-10}$ Å$^{-1}$, equals $4\alpha\gamma^2\Delta\lvert\mathbf J\rvert^2/(3\hbar v)$ exactly (relative difference 0). That value is the rest-frame rate $\Gamma'=4\alpha\omega'^3\lvert r\rvert^2/3c^2$ with $\omega'=\gamma\Delta/\hbar$ and $r=\hbar cJ/\Delta$, divided by $\gamma v$.
- The code was given a Gaussian-envelope line (relative width 0.025). Integrating its output over $E$ and the cone $\theta\le1.5$ with `integrate_solid_angle` gives $4.26911\times10^{-10}$, against my $4.26952\times10^{-10}$ (relative $-9.5\times10^{-5}$). The residual is the $\alpha E$ prefactor across the line. That cone holds 95.1% of the full sphere.

### Exact-recoil Bloch toy

The toy is 1D: $H=\hbar^2(\kappa+ng)^2/2\gamma m+V_0\cos gx\,[+V_1\sin2gx]$, with $d=1.92$ Å, $V_0=40$ eV, 3.5 MeV ($\gamma=7.85$) and 81 plane waves. The exact amplitude uses the final state in fibre $\kappa-q$ and the symmetrized current $\tfrac12\{v,e^{-iqx}\}$, with $q$ and $E$ solved self-consistently at the exact resonance. Moduli are compared, since they are gauge invariant.

| Line | Relative error of each term at $q=10^{-3}\to10^{-1}$ Å$^{-1}$ | Slope | $\lvert A_\theta\rvert$ error at $\theta\gamma=0.5$–$5$ (units of $\lvert J\rvert$) |
|---|---|---|---|
| bound 1→0, $\kappa=0$ | $1.3\times10^{-8}\to1.3\times10^{-4}$ | 2.00 | $\le2.9\times10^{-5}$ |
| bound 1→0, $\kappa=0.15g$ | $7\times10^{-8}\to1.4\times10^{-4}$ | 1.65 | $\le3.1\times10^{-5}$ |
| bound 2→1, $\kappa=0.15g$ | $1.4\times10^{-6}\to1.4\times10^{-4}$ | 0.93 | $\le1.5\times10^{-5}$ |
| above-barrier 5→4, $\kappa=0.15g$ | $4\times10^{-4}\to4\times10^{-2}$ | 0.99 | $5.8$–$8.4\times10^{-3}$; exact-recoil line shift $\delta\Delta/\Delta=3$–$7\times10^{-3}$ |
| no inversion ($V_1=8$ eV), $\kappa=0$, bound and above-barrier | down to $1.3\times10^{-8}$ | 2.00 | $\le3.3\times10^{-5}$ |

The parity-forbidden 2→0 line at $\kappa=0$ ($\lvert J_{20}\rvert\sim10^{-19}$) acquires $\lvert A_{20}\rvert/\lvert J_{10}\rvert=1.6\times10^{-2}$ at $\theta\gamma=2.5$. This is the omitted $O(qa)$ amplitude, with $qa\approx0.03$.

### Resonance-evaluated factor vs velocity form within a line

For a single Lorentzian (dissipative) line, I compared the line area of the velocity form with that of the resonance-evaluated factor (θ and φ summed, window ±10 FWHM).

| $\theta\gamma$ | first-order coefficient $\beta\sin^2\theta/(1-\beta\cos\theta)$ | FWHM/E = 0.003 | 0.03 | 0.1 | 0.03, window ±400 FWHM |
|---|---|---|---|---|---|
| 0.5 | 0.40 | $-6.4\times10^{-6}$ | $-6.4\times10^{-4}$ | $-7.1\times10^{-3}$ | $+2.0\times10^{-2}$ |
| 1.0 | 0.99 | $+2.7\times10^{-5}$ | $+2.7\times10^{-3}$ | $+3.0\times10^{-2}$ | $+0.39$ |
| 2.5 | 1.68 | $+9.4\times10^{-5}$ | $+9.4\times10^{-3}$ | $+0.10$ | $+0.85$ |

The first-order term cancels in the area by symmetry, so the in-line area difference is quadratic in width. The growth with the window is the off-resonance velocity-form continuum that the boundary-term argument removes.

### Findings

Blocking: none.

Minor findings:

1. **The stated in-line error order is too small.** With $\delta=E(1-\beta\cos\theta)-\Delta$, as defined in the derivation, the pointwise amplitude error is $[\beta\sin^2\theta/(1-\beta\cos\theta)]\,\delta/\Delta$, not $O(\sin^2\theta\,\delta/\Delta)$. The coefficient reaches about 2 for $\theta\gtrsim1/\gamma$ (0.99 at $\theta\gamma=1$, 1.68 at 2.5). The ledger's "$O(\sin^2\theta\cdot$ detuning$/\Delta)$" is correct only if "detuning" means the photon-energy detuning $E-E_{\rm res}$. The derivation's "bounded by the relative line width" is right in magnitude. Suggested wording: "pointwise $O(\delta/\Delta)$ with coefficient $\beta\sin^2\theta/(1-\beta\cos\theta)\le2$; line area second order".
2. **The parity condition can be sharpened.** The toy confirms $O((qa)^2)$ at $\kappa=0$ and $O(qa)$ for dispersive bands at generic $\kappa$ (about 0.6–0.8% of $\lvert A\rvert$ at Si-like $q$, plus a comparable exact-recoil line shift). It also shows that at time-reversal-invariant carriers ($\kappa=0$ or the zone edge) a single line's intensity error stays $O((qa)^2)$ without inversion symmetry: the $O(q)$ correction is in quadrature with $J$. It can still enter at first order through cross terms between lines in $G$.
3. **`max_phase_step` does not cover transitions above the photon grid.** It bounds the photon-grid phase step, but a retained transition with $\Delta\,\Delta z/\hbar v$ near $2\pi$ still aliases into the grid. Not checked; irrelevant for Si ($\Delta\lesssim100$ eV and 15 Å give 0.76 rad).
4. **Wording of the Gaussian-line sentence.** The sentence "Fresh-context Gaussian-line check …" (6e-5 and 1.5%) comes from the old velocity-form code compared with my resonance-evaluated line area. On 819567a16 the same check gives $-9.5\times10^{-5}$, which is purely the $\alpha E$ prefactor.
5. **Cosmetic.** `_phase_rate` has an unused `_gamma` argument. `exit_path_ang` geometry is not enforced.

Anchors: `pyrite-dev test tests/montecarlo/quantum/test_channeling_axial_fiber_angular.py tests/montecarlo/quantum/test_channeling_axial_fiber_regression.py` gives 17 passed, 1 slow skipped.

**Suggested ledger change:** `unverified` → `rederived` (the task owner applies it, never `signed-off`). Reword the in-line error order (finding 1) and optionally add the time-reversal refinement (finding 2).
