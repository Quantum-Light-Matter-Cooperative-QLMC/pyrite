# `quantum-decoherence-budget` — cell-averaged transverse decoherence rate re-derivation

Fresh-context verification (issue #501). Inputs read before the derivation: the ledger rows `quantum-decoherence-budget` and `cr-lindblad-jump-operators` only. Quantity: the cell-averaged decay rate $\Gamma(\Delta)$ of a transverse coherence $\rho(x,x+\Delta)$ under the Lindblad dissipator of `cr-lindblad-jump-operators`, its TDS and electronic $G=0$ closed forms, the small-$\Delta$ diffusion limit, and the budget criterion $\ell=1/\Gamma(\Delta)$. Units: $\Gamma$ and $C$ in Å$^{-1}$, $\Delta$ in Å, $D$ in Å$^{-3}$.

## Independent derivation (written before reading the research note, the check, or `decoherence_rate`)

### (a) Local and cell-averaged rate; positivity; why only $G=0$ survives

The dissipator acts elementwise in the position basis:

$$
\frac{d\rho(x,x')}{dz}\supset\Bigl[C(x,x')-\tfrac12C(x,x)-\tfrac12C(x',x')\Bigr]\rho(x,x'),
\qquad C(x,x')=\sum_k L_k(x)L_k^*(x').
$$

With $H_\perp$ off every element evolves independently, so

$$
\rho(x,x';z)=\rho(x,x';0)\exp\Bigl\{-z\Bigl[\tfrac12C(x,x)+\tfrac12C(x',x')-C(x,x')\Bigr]\Bigr\}.
$$

The modulus decays at the *local* rate

$$
\Gamma_{\rm loc}(x,\Delta)=\tfrac12C(x,x)+\tfrac12C(x+\Delta,x+\Delta)-\operatorname{Re}C(x,x+\Delta),
$$

and $\operatorname{Im}C$ only rotates the phase. Since $\lvert a-b\rvert^2=\lvert a\rvert^2+\lvert b\rvert^2-2\operatorname{Re}(ab^*)$,

$$
\Gamma_{\rm loc}(x,\Delta)=\tfrac12\sum_k\bigl\lvert L_k(x)-L_k(x+\Delta)\bigr\rvert^2\ge0,
$$

and the cell average inherits the form and the sign, $\Gamma(\Delta)=\langle\Gamma_{\rm loc}\rangle_x=\tfrac12\sum_k\langle\lvert L_k(x)-L_k(x+\Delta)\rvert^2\rangle_x\ge0$, as the ledger states.

For a crystal of period $d$ the kernel is jointly periodic, $C(x+d,x'+d)=C(x,x')$, so it has the expansion $C(x,x')=\sum_G e^{iGx}c_G(x-x')$, $G=2\pi m/d$. Then $\langle C(x,x+\Delta)\rangle_x=c_0(-\Delta)$ and $\langle C(x,x)\rangle_x=\langle C(x+\Delta,x+\Delta)\rangle_x=c_0(0)$: every $G\ne0$ harmonic averages to zero over a cell, and

$$
\Gamma(\Delta)=c_0(0)-\operatorname{Re}c_0(-\Delta).
$$

The translation-averaged (period-$d$ invariant, Bloch block-diagonal) beam is what makes the uniform cell average the natural weight; it does not make $\Gamma_{\rm loc}$ independent of $x$ (see (e)).

### (b) TDS closed form

From the ledgered Einstein-model phase covariance, the $G=0$ term with $p\equiv q_x$, $q_y$ is

$$
c_0(x-x')=\frac{n_0}{(\hbar v)^2}\int\frac{dq_x\,dq_y}{(2\pi)^2}\,v(q)^2\,e^{iq_x(x-x')}\Bigl[1-e^{-a^2(q_x^2+q_y^2)}\Bigr],
$$

since $e^{-a^2G^2/2}\to1$ and $(p+G)^2+p^2+2q_y^2\to2q^2$ at $G=0$. I re-derived that kernel from the per-atom phase $\varphi(\mathbf r_\perp)=-(\hbar v)^{-1}\int\frac{d^2q}{(2\pi)^2}v(q)e^{i\mathbf q\cdot(\mathbf r_\perp-\mathbf R_\perp-\mathbf u_\perp)}$. The Gaussian displacement covariance is $\langle e^{-i(\mathbf q-\mathbf q')\cdot\mathbf u}\rangle-\langle e^{-i\mathbf q\cdot\mathbf u}\rangle\langle e^{i\mathbf q'\cdot\mathbf u}\rangle=e^{-a^2\lvert\mathbf q-\mathbf q'\rvert^2/2}-e^{-a^2(q^2+q'^2)/2}$, with $a^2=\langle u_x^2\rangle=B/8\pi^2$. The atom sum $\int dY\to2\pi\delta(q_y-q_y')$ and $\sum_X\to2\pi\sum_Gn_G\delta(q_x-q_x'-G)$ leaves $(2\pi)^2/(2\pi)^4=1/(2\pi)^2$. This reproduces the ledgered prefactor, the exponents and $e^{iGx}e^{ip(x-x')}$. Hence

$$
\Gamma_{\rm TDS}(\Delta)=\frac{n_0}{(\hbar v)^2}\int\frac{d^2q}{(2\pi)^2}\,v(q)^2\bigl(1-e^{-a^2q^2}\bigr)\bigl(1-\cos q_x\Delta\bigr).
$$

The 2D angular integral is $\int_0^{2\pi}d\phi\,[1-\cos(q\Delta\cos\phi)]=2\pi[1-J_0(q\Delta)]$, so

$$
\boxed{\Gamma_{\rm TDS}(\Delta)=\sum_s\frac{n_s}{(\hbar v)^2}\,\frac{1}{2\pi}\int_0^\infty q\,dq\,v_s(q)^2\bigl(1-e^{-a_s^2q^2}\bigr)\bigl(1-J_0(q\Delta)\bigr)}
$$

with $n_s$ the mean volume density of species $s$ (independent atoms add their covariances). All three factors check out: the $1/2\pi$ is $2\pi/(2\pi)^2$; the $(1-e^{-a^2q^2})$ is the $G=0$ Debye–Waller covariance; and $J_0$ comes from the azimuthal average of $\cos q_x\Delta$.

Units: $v=-4\pi e^2(Z-f_0)/q^2$ in eV Å$^3$, so $n v^2 q\,dq/(\hbar v)^2$ is Å$^{-3}\cdot$eV$^2$Å$^6\cdot$Å$^{-2}/($eV$^2$Å$^2)=$ Å$^{-1}$. The sign of $v$ drops out ($v^2$).

Limits: $\Gamma(0)=0$ ($J_0(0)=1$). As $\Delta\to\infty$, $J_0(q\Delta)\to0$ (Riemann–Lebesgue, integrable weight), so $\Gamma\to c_0(0)=\langle C(x,x)\rangle$, the mean TDS loss. With $q=4\pi s$, $a^2q^2=2Bs^2$, which is the Pennycook–Jesson $n\sigma_{\rm TDS}$ factor $(1-e^{-2Bs^2})$. Because $1-J_0$ peaks at $1.403$, $\Gamma(\Delta)$ is not monotone: it can overshoot the mean loss at intermediate $\Delta$, where $\operatorname{Re}C(x,x+\Delta)<0$. $C\propto(\hbar v)^{-2}$, so $\beta^2\Gamma_{\rm TDS}$ is energy independent.

### (c) Electronic LDA closed form

With $s_q(x)\equiv\sqrt{\rho_q(x)}$ (real, $\ge0$) and $w=w(\lvert\mathbf q_\perp\rvert)$,

$$
C_{\rm el}(x,x')=\int d^2q\,w(q)\,s_q(x)s_q(x')\,e^{iq_x(x-x')}.
$$

$\langle C_{\rm el}(x,x)\rangle=\int d^2q\,w\,\langle\rho_q\rangle=\int d^2q\,w\,\bar\rho$, because Gaussian smoothing preserves the cell mean. $\operatorname{Re}C_{\rm el}(x,x+\Delta)=\int d^2q\,w\,s_q(x)s_q(x+\Delta)\cos q_x\Delta$, because the $\sin$ part is odd in $q_x$ while $w$ and $s_q$ depend on $\lvert\mathbf q\rvert$ only. The same azimuthal integral gives

$$
\boxed{\Gamma_{\rm el}(\Delta)=2\pi\int_0^\infty q\,dq\,w(q)\Bigl[\bar\rho-J_0(q\Delta)\,\langle s_q(x)s_q(x+\Delta)\rangle_x\Bigr]}
$$

This matches the ledger. Here "$G=0$" means the cell average at fixed $\Delta$; the density modulation stays in the correlation $\langle s_qs_q\rangle$. Positivity: by Cauchy–Schwarz $\langle s_q(x)s_q(x+\Delta)\rangle\le\langle s_q^2\rangle=\bar\rho$, and $\lvert J_0\rvert\le1$, $w\ge0$, so the integrand is $\ge0$ pointwise in $q$. $\Gamma_{\rm el}(0)=0$. As $\Delta\to\infty$ the limit is $2\pi\bar\rho\int q\,dq\,w$, the mean electronic loss, reached only for $\Delta\gg1/q_{\min}$ (plasmon $q_{\min}\sim\hbar\omega_p/\hbar v\sim10^{-2}$ Å$^{-1}$, i.e. hundreds of Å). $C_{\rm el}$ is *not* exactly $\propto1/v^2$: the close Møller term and the $(\hbar cq_\perp)^2=2mc^2W(1-W/E)$ map carry $E$ and $\gamma$ dependence. So the ledger's exact $\beta^2$-invariance check is correctly restricted to TDS.

### (d) Small-$\Delta$ diffusion and $d\langle\theta^2\rangle/dz=D/k^2$

General statement: for smooth $L_k$, $\Gamma_{\rm loc}=\tfrac12\Delta^2\sum_k\lvert L_k'(x)\rvert^2+O(\Delta^3)$. Momentum diffusion for multiplicative jump operators follows from $\sum_k\langle L_k^\dagger p^2L_k-\tfrac12\{L_k^\dagger L_k,p^2\}\rangle$ with $[p,f]=-if'$ ($\hbar=1$, $p=q_x$). It gives $d\langle q_x^2\rangle/dz=\sum_k\langle\lvert L_k'\rvert^2\rangle_\rho$. For a cell-uniform state, therefore,

$$
\Gamma(\Delta)\simeq\tfrac12D\Delta^2,\qquad D\equiv\frac{d\langle q_x^2\rangle}{dz}=\sum_k\langle\lvert L_k'\rvert^2\rangle_x=\Gamma''(0).
$$

Explicitly:

- TDS: $1-J_0(q\Delta)\simeq q^2\Delta^2/4$, so $\Gamma\simeq\frac{\Delta^2}{4}\frac{n}{(\hbar v)^2}\frac1{2\pi}\int q^3v^2(1-e^{-a^2q^2})dq$. The direct count is $D=\frac{n}{(\hbar v)^2}\int\frac{d^2q}{(2\pi)^2}q_x^2v^2(1-e^{-a^2q^2})=\frac{n}{(\hbar v)^2}\frac{1}{4\pi}\int q^3v^2(1-e^{-a^2q^2})\,dq$ ($\int\cos^2\phi\,d\phi=\pi$), and the two agree as $\tfrac12D\Delta^2$.
- Electronic: $\langle s(x)s(x+\Delta)\rangle=\bar\rho-\tfrac12\Delta^2\langle s'^2\rangle+\dots$ gives $\Gamma''_{\rm el}(0)=2\pi\int q\,dq\,w[\bar\rho q^2/2+\langle s_q'^2\rangle]$. That equals $\int d^2q\,w\langle\lvert(s_qe^{iq_xx})'\rvert^2\rangle=\int d^2q\,w\langle s_q'^2+q_x^2s_q^2\rangle$. The density-gradient term $\langle s_q'^2\rangle$ is a position-localization contribution with no free-space analogue; it belongs to $D$ as defined here.

Log correction: at large $q$, $f_0\to0$, $v\to-4\pi e^2Z/q^2$ and $1-e^{-a^2q^2}\to1$, so $q^3v^2\propto1/q$. $D$ is then logarithmically divergent, with the Rutherford tail cut only by the kernel's $q_{\max}$ (ledgered at $10^4$ Å$^{-1}$). In $\Gamma(\Delta)$ the factor $1-J_0$ stops growing at $q\sim1/\Delta$, so $\Gamma\propto\Delta^2\ln(1/(\Delta q_0))$ for $1/q_{\max}\ll\Delta\ll a$. "$\Gamma\approx\tfrac12D\Delta^2$" is therefore literal only for $\Delta\lesssim1/q_{\max}$. For $\Delta\sim0.1$–$1$ Å it holds with an effective, $\Delta$-dependent $D$. Consequently $D$ itself, and any $\theta_{\rm rms}$ from it, depends on $q_{\max}$.

Angle: paraxially $\theta_x=q_x/k$, $k=\gamma m v/\hbar$, so $d\langle\theta_x^2\rangle/dz=D/k^2$ (Å$^{-3}$/Å$^{-2}$ = rad$^2$/Å). Highland's $\theta_0$ is the width of the Gaussian core (central 98%) and excludes the single-scattering tail. An rms from the full-tail $D$ is therefore larger than Highland and grows with $\ln q_{\max}$. A Highland comparison can only be order-of-magnitude unless $D$ is tail-truncated consistently.

### (e) Budget criterion

With $H_\perp$ off, (a) gives exactly $\lvert\rho(x,x+\Delta;z)\rvert=\lvert\rho(x,x+\Delta;0)\rvert e^{-\Gamma_{\rm loc}(x,\Delta)z}$. The statement $\lvert\rho\rvert\propto e^{-\Gamma(\Delta)z}$ with the *cell-averaged* $\Gamma$ is exact only where $\Gamma_{\rm loc}$ is $x$-independent. In general it holds as the initial slope of the cell-averaged coherence: for a uniform-modulus initial state, $\frac{d}{dz}\langle\lvert\rho\rvert\rangle\vert_{z=0}=-\Gamma\langle\lvert\rho\rvert\rangle$. At finite $z$, Jensen's inequality gives $\langle e^{-\Gamma_{\rm loc}z}\rangle_x\ge e^{-\Gamma z}$. So $\ell=1/\Gamma$ is the initial, cell-uniform-weighted coherence length, and the cell-averaged coherence that actually survives is *at least* $e^{-\Gamma z}$. The criterion is conservative for "decoherence kills the effect": coherent pairs where $\Gamma_{\rm loc}$ is small, e.g. TDS between channel-centre points where $W(d/2)/W(0)<10^{-2}$, live longer than $\ell$.

With $H_\perp$ on, position-space elements are mixed by the unitary flow and the elementwise solution fails. Then $\Gamma(\Delta)$ is neither an upper nor a lower bound on the decay of a given coherence. The relevant rate for an eigenstate coherence $\lvert n\rangle\langle m\rvert$ in the secular limit is $\tfrac12(\Gamma_n+\Gamma_m)-\sum_k\operatorname{Re}\langle n\vert L_k\vert n\rangle\langle m\vert L_k\vert m\rangle^*$. That is the state-weighted, not the cell-uniform, version of $\tfrac12\sum_k\lvert\cdot\rvert^2$. It reduces to $\Gamma(\Delta)$ only when $\lvert n\rangle,\lvert m\rangle$ sample the cell uniformly and are separated by $\Delta$. Electron bound states concentrate on the planes, where TDS is strongest, so their rate exceeds the cell average. With $H_\perp$ on, the criterion therefore gives an order-of-magnitude decoherence length, not a bound. The comparison $\Gamma L\ll1$ means decoherence is negligible over $L$ in the uniform-weight sense; $\Gamma L\gtrsim1$ does not by itself prove the effect is destroyed.

### Cheap filters

| check | result |
| --- | --- |
| units of $\Gamma$, $C$ | Å$^{-1}$ for both closed forms (above); $D$ in Å$^{-3}$; $D/k^2$ in rad$^2$/Å — pass |
| $\Gamma(0)=0$ | $J_0(0)=1$ (TDS); $\langle s_q^2\rangle=\bar\rho$ (electronic) — pass |
| $\Gamma(\Delta\to\infty)$ | mean TDS and electronic loss $\langle C(x,x)\rangle$ — pass |
| small $\Delta$ | $\tfrac12D\Delta^2$ with $\ln(1/\Delta)$ from the $1/q^4$ tail — pass, with the $q_{\max}$ caveat |
| $C\propto1/v^2$ | exact for TDS; approximate for electronic (Møller/$q_\perp$ map) — pass as ledgered (TDS only) |
| positivity | $\tfrac12\sum_k\lvert L_k(x)-L_k(x+\Delta)\rvert^2$; Cauchy–Schwarz for electronic — pass |
| monotonicity | not monotone ($1-J_0\le1.403$); overshoot of the mean loss is physical, not an error |

### Independent numeric point (own quadrature)

Si(110), 293.15 K, 3 MeV kinetic electron ($\beta=0.98928$, $k=1760.3$ Å$^{-1}$). The inputs are $n_{\rm Si}=8/5.431^3$ Å$^{-3}$ and $B=0.48166$ Å$^2$ from `thermal_b_factor("silicon", "Si", 293.15)`, so $a^2=0.0061003$ Å$^2$. The potential is $v(q)=-4\pi e^2(Z-f_0(q))/q^2$ with `cromer_mann_f0`, rescaled to $f_0(0)=Z$ and given an $s^{-4}$ tail beyond $s=6$ Å$^{-1}$, mirroring the ledgered model choices. The integral is cut at $q=10^4$ Å$^{-1}$ and uses adaptive `scipy.integrate.quad` panels 0.025 Å$^{-1}$ wide to 50 Å$^{-1}$, then 400 log panels. It uses no `decoherence_rate` helper.

| $\Delta$ [Å] | $\Gamma_{\rm TDS}$ [Å$^{-1}$] | $\beta^2\Gamma_{\rm TDS}$ [Å$^{-1}$] |
| --- | --- | --- |
| 1 | $6.5718\times10^{-5}$ | $6.4326\times10^{-5}$ |
| 1.92 | $6.5407\times10^{-5}$ | $6.4021\times10^{-5}$ |
| 5 | $6.5113\times10^{-5}$ | $6.3733\times10^{-5}$ |
| 100 | $6.5112\times10^{-5}$ | $6.3733\times10^{-5}$ |
| $\infty$ | $6.5112\times10^{-5}$ | $6.3733\times10^{-5}$ |

$\beta^2\Gamma_{\rm TDS}(\infty)=6.373\times10^{-5}$ Å$^{-1}$ reproduces the `cr-lindblad-jump-operators` mean TDS loss rate, 6.38e-5 Å$^{-1}$ at 17 MeV. TDS saturates by $\Delta\approx2$–5 Å: the screened nuclear kernel has range $\sim a$ and the screening length. At $\Delta=1$ Å it slightly overshoots the mean loss, as (b) predicts.

With the same tail cut, the TDS-only full-tail diffusion is $D_{\rm TDS}=\Gamma''(0)=0.04329$ Å$^{-3}$. That gives $\theta_{\rm rms}(1\,\mu{\rm m})=\sqrt{D_{\rm TDS}\cdot10^4\,\text{Å}}/k=11.8$ mrad at 3 MeV, above Highland's 7.31 mrad. My Highland value uses $X_0({\rm Si})=9.37$ cm, $\beta cp=3.4365$ MeV and the $1+0.038\ln(x/X_0\beta^2)$ correction, and reproduces the ledgered 7.3. This is the $\ln q_{\max}$ dependence from (d). An rms of 6.8 mrad, as ledgered, cannot come from the full-tail $\Gamma''(0)$ of this kernel, because the TDS part alone exceeds it. It must come from a truncated or finite-$\Delta$ $D$. See the comparison below.

## Comparison with the implementation and the research note

Read after the derivation above: `docs/research/physics/quantum-coherent-emission.md` §3, `montecarlo/channeling/operators.py::decoherence_rate` and its helpers, and `checks/quantum_decoherence_budget.py`.

### Symbolic

| term | derivation | `decoherence_rate` | agreement |
| --- | --- | --- | --- |
| TDS density | $n_s$ = mean volume density | `_site_densities(profile, 0)[s][0].real` = basis count / `V_cell` | match |
| TDS integrand | $\frac{n_s}{2\pi(\hbar v)^2}q\,v_s^2(1-e^{-a_s^2q^2})$ | `density * q * _potential**2 * -expm1(-a2 q**2) / (2 pi hbar_v**2)` | match |
| $a_s^2$ | $B_s/8\pi^2$ | `_displacement_sq` | match |
| $v_s$ | $-4\pi e^2(Z-f_0)/q^2$, $f_0(0)$ rescaled to $Z$, $s^{-4}$ tail | `_potential_table` | match |
| $1-J_0$ | $\Gamma=\text{mean}-\int(\dots)J_0$ | `tds_mean - w @ (tds_density * j0)` on `_oscillatory_panels` | match |
| electronic mean | $2\pi\bar\rho\int q\,dq\,w$ | `el_mean = rho_bar * (2 pi q w_q) @ channels.weight` | match |
| electronic overlap | $\langle s_q(x)s_q(x+\Delta)\rangle_x=\sum_h\lvert c_h(q)\rvert^2\cos(hg\Delta)$ (Parseval, real $s_q$) | `_RootDensity.autocorrelation` | match |
| electronic $J_0$ term | $2\pi\int q\,dq\,w\,J_0\langle ss\rangle$ | `2 pi q w channels.weight * j0 * overlap` | match |
| $\hbar v$ | $\hbar c\beta$ | `_kinematics` | match |

I find no divergent factor, sign, exponent or convention. The `cr-lindblad-jump-operators` row already verified the per-channel electronic weight `channels.weight`; I compare the electronic closed form here symbolically only.

### Numeric (Si(110), 293.15 K, 3 MeV, TDS)

Own Gauss–Legendre quadrature (16 nodes per panel; panels $\le\pi/4\Delta$ to 60 Å$^{-1}$, then 4000 log panels to $10^4$ Å$^{-1}$), with $n=8/5.431^3$:

| $\Delta$ [Å] | `decoherence_rate` | mine | relative |
| --- | --- | --- | --- |
| 0.03 | $5.730157\times10^{-6}$ | $5.729772\times10^{-6}$ | $+6.7\times10^{-5}$ |
| 0.1 | $2.790201\times10^{-5}$ | $2.790033\times10^{-5}$ | $+6.0\times10^{-5}$ |
| 0.3 | $5.909076\times10^{-5}$ | $5.908731\times10^{-5}$ | $+5.8\times10^{-5}$ |
| 1 | $6.572210\times10^{-5}$ | $6.571828\times10^{-5}$ | $+5.8\times10^{-5}$ |
| 1.92 | $6.541056\times10^{-5}$ | $6.540676\times10^{-5}$ | $+5.8\times10^{-5}$ |
| 5 | $6.511639\times10^{-5}$ | $6.511261\times10^{-5}$ | $+5.8\times10^{-5}$ |
| 1000 | $6.511614\times10^{-5}$ | $6.511236\times10^{-5}$ (at $\infty$) | $+5.8\times10^{-5}$ |

The constant $5.52\times10^{-5}$ of the offset is my lattice constant: 5.431 Å versus the catalog's `V_cell` = 160.1826 Å$^3$ ($a=5.43102$ Å). The residual is $\le1.2\times10^{-5}$, so the two quadratures agree.

### Check script

`checks/quantum_decoherence_budget.py` runs in about 2 s wall here (the docstring says under a second) and prints PASS. It confirms $\Gamma(0)=0$ ($\lvert\Gamma_{\rm el}(0)\rvert\sim10^{-19}$), TDS saturation from 100 to 1000 Å within $10^{-3}$, and $\beta^2\Gamma_{\rm TDS}$ spreads of $7.8\times10^{-15}$ (Si) and $2.4\times10^{-15}$ (diamond).

Every number in the research note's Si(110) and diamond(110) tables matches the check output to the quoted digits: $\Gamma_{\rm TDS}(1)$, $\Gamma_{\rm el}(1)$, $\ell(d/2)$, $\ell(1)$, $n_b$, $L_{\rm CR}$ and $\Gamma(d/2)L_{\rm CR}$. So do the $\Gamma_{\rm el}$ values at 100 and 1000 Å (3.88e-4, 4.20e-4) and the 90% saturation by 0.3 Å (0.908; $a=0.078$ Å). $D_{\rm eff}=2\Gamma(0.03)/0.03^2=0.0144$ Å$^{-3}$ gives 6.8 mrad, and my Highland value is 7.31 mrad.

### Findings (wording/notes; no formula change)

1. **$D$ definition versus the quoted $\theta_{\rm rms}$.** The ledger defines $D=d\langle q_x^2\rangle/dz$ (that is $\Gamma''(0)$) and quotes 6.8/0.77 mrad. The check computes $D_{\rm eff}=2\Gamma(\Delta)/\Delta^2$ at $\Delta=0.03$ Å, which effectively cuts the Rutherford tail at $q\sim1/\Delta$. The full-tail $\Gamma''(0)$ to $q_{\max}=10^4$ Å$^{-1}$ is 0.0433 Å$^{-3}$ for TDS alone, i.e. 11.8 mrad. The research note states the $\Delta=0.03$ Å choice and its log sensitivity; the ledger `Checks` field does not.
2. **$\Gamma L_f\le10^{-3}$ "in every cell".** Two problems:
   - With $L_f=1/g_\parallel$ and $g_\parallel\le\lvert g\rvert$, $L_f\ge1/\lvert g\rvert$. The note's "$L_f\le0.31$ Å (Si), $\le0.20$ Å (diamond)" is therefore the *minimum* $L_f$ ($g\parallel v$), not an upper bound; the check uses $1/\lvert g\rvert$. In a Bragg geometry $L_f=1/(g\sin\theta_B)$ is longer, but $\Gamma L_f$ stays $\ll1$ unless $g_\parallel$ is very small, so the conclusion holds while the inequality's direction is wrong.
   - Even with $L_f=1/g$, the bound fails at the largest tabulated $\Delta$ at 30 keV. Si: $\Gamma(1000)\cdot0.306=1.13\times10^{-3}$. Diamond: $\Gamma(100)\cdot0.201=1.31\times10^{-3}$, and $\Gamma(10)\cdot0.201=1.01\times10^{-3}$. It holds for $\Delta\le3$ Å.
3. **Energy scaling.** The research note's "every rate scales as $C\propto1/v^2$" is exact only for TDS. $\beta^2\Gamma_{\rm el}(1\,\text{Å})$ is $8.53\times10^{-5}$ at 30 keV against $8.65\times10^{-5}$ at 3 MeV (1.4%), because of the Møller and $q_\perp$-map energy dependence. The ledger's exact check is correctly TDS-only.
4. **Budget criterion, pointwise versus cell average.** $\lvert\rho\rvert\propto e^{-\Gamma(\Delta)z}$ is exact only with $\Gamma_{\rm loc}(x,\Delta)$. The cell-averaged $\Gamma$ is the initial rate, and by Jensen the surviving cell-averaged coherence is $\ge e^{-\Gamma z}$. This matters most for TDS, where $W(0)/W(d/2)>100$. The note states the $H_\perp$-on caveat (exact decay from the density solver) but not this one.
5. **§3.3 wording.** "Above about 0.3 Å survives only $\ell\approx0.07$ µm": at 0.3 Å the check gives $\ell=1/(5.36\times10^{-4}+2.27\times10^{-4})=0.13$ µm, and 0.07 µm is the 1 Å value. "0.06 µm at 1.7 Å" is not on the check's $\Delta$ grid (0.96 and 1.92 Å give 0.074 and 0.054 µm).
6. **Premature status.** Research note §3.1 already labels the claim "`rederived`" while the ledger row is `unverified`.
7. **Honest as stated.** These are labelled correctly:
   - the Highland comparison: consistency only, below Highland's $x/X_0$ range;
   - the Genz Eq. 19/Table VI comparison: order of magnitude, a different observable;
   - the diamond electronic term: graphite-oscillator surrogate, not channeling-grade;
   - the HOPG/hBN/MoS₂/WSe₂ cells: not computable. The check verifies the `ChannelingDataError` and fails if data appear.

## Verdict

`rederived`. The closed forms (a)–(d) re-derive term by term. `decoherence_rate` implements them, and the TDS branch agrees with an independent quadrature to $\le1.2\times10^{-5}$ after the lattice-constant offset. Findings 1–6 are wording and notes corrections, not formula discrepancies. No anchor test exists (the row's `Anchor` is empty), so `anchored` is not supported.

## Material-input update (#525)

The source-to-code derivation above records the original #501 calculation.
The current check uses the diamond 34 eV valence input instead of the former
graphite surrogate; its equations are unchanged. Refreshed numerical results
and source scope are in the [material survey](../../research/physics/channeling-materials.md)
and [diamond verification](diamond-valence-oscillator.md). The original
surrogate numbers in this historical verification are not current predictions.
