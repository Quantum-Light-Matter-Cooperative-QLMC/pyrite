# `cr-lindblad-jump-operators` — Lindblad scattering kernels for planar channeling

Verifier: fresh context. Ledger: `docs/validation/ledger-channeling.md`. Code: `src/pyrite/montecarlo/channeling/operators.py::{tds_kernel,electronic_kernel,electronic_channels,ElectronicChannels.distant,ElectronicChannels.close_loss,ElectronicChannels.close,ScatteringKernel.diagonal,ScatteringKernel.absorptive_potential_eV,ScatteringKernel.cell_kernel,decoherence_rate}`. Context: issue #504, including the maintainer-approved electronic LDA kernel.

## Claim

The incoherent part of the paraxial evolution is the dissipator

$$
\partial_z\rho(x,x')\supset\Bigl[C(x,x')-\tfrac12C(x,x)-\tfrac12C(x',x')\Bigr]\rho(x,x'),
\qquad W(x)=\tfrac{\hbar v}{2}C(x,x),
$$

with an Einstein-model TDS kernel

$$
C(x,x')=\frac{1}{(\hbar v)^2}\sum_G n_Ge^{iGx}\int\frac{dp\,dq_y}{(2\pi)^2}\,
\tilde v(p+G,q_y)\,\tilde v(p,q_y)\,e^{ip(x-x')}
\Bigl[e^{-a^2G^2/2}-e^{-a^2((p+G)^2+p^2+2q_y^2)/2}\Bigr],
$$

where $\tilde v(q)=-4\pi e^2(Z-f_0(q))/q^2$ and $a^2=B/8\pi^2$. The electronic kernel is $C_{\rm el}(x,x')=\int d^2q_\perp\,w(q)\sqrt{\rho_q(x)\rho_q(x')}\,e^{iq_x(x-x')}$, with $\rho_q$ the electron density Gaussian-smoothed at $\sigma=1/|q_\perp|$ and $w$ the per-electron PENELOPE-2024 GOS resolved in $q_\perp$. The diagnostic is $\Gamma(\Delta)=\langle\tfrac12C(x,x)+\tfrac12C(x+\Delta,x+\Delta)-{\rm Re}\,C(x,x+\Delta)\rangle_x$.

## Independent derivation (before reading the body)

### (a) TDS kernel from phase covariance

Paraxially, $i\hbar v\,\partial_z\psi=(H_\perp+V)\psi$. A slab $dz$ multiplies $\psi$ by $e^{i\phi}$ with $\phi(\mathbf r_\perp)=-(\hbar v)^{-1}\sum_jV_{{\rm proj},j}(\mathbf r_\perp-\mathbf R_{j\perp}-\mathbf u_{j\perp})$, a sum over the atoms in the slab. The transmitted density matrix is $e^{i\phi(\mathbf r)}\rho\,e^{-i\phi(\mathbf r')}$. Averaging over displacements to second order in $\phi$ gives

$$
\bigl\langle e^{i(\phi(\mathbf r)-\phi(\mathbf r'))}\bigr\rangle
\simeq e^{i(\langle\phi(\mathbf r)\rangle-\langle\phi(\mathbf r')\rangle)}
\exp\Bigl[-\tfrac12{\rm Var}\bigl(\phi(\mathbf r)-\phi(\mathbf r')\bigr)\Bigr].
$$

Here ${\rm Var}(\phi-\phi')={\rm Var}\,\phi+{\rm Var}\,\phi'-2\,{\rm Cov}(\phi,\phi')$. The mean phase is the continuum potential. The fluctuation is therefore the dissipator with $C={\rm Cov}[\phi(\mathbf r),\phi(\mathbf r')]$ per unit depth. The bracket has exactly the claimed form. $C$ is real, because $\phi$ is real.

The electron moves freely in $y$, and the beam is uniform in $y$. A jump $f(x)e^{iq_yy}$ therefore acts on the reduced $x$ state as $f(x)\rho f^*(x')$, so the reduced kernel is the covariance at equal $y$.

Write $V_{\rm proj}(\mathbf r)=\int\frac{d^2q}{(2\pi)^2}\tilde v(\mathbf q)e^{i\mathbf q\cdot\mathbf r}$, with $\tilde v$ the 3D transform at $q_z=0$. The sign follows from charge $-e$: the nucleus contributes $-4\pi Ze^2/q^2$ and the screening cloud $+4\pi e^2f_0/q^2$. Each axis of the displacement is independent and Gaussian with variance $a^2$, so $\langle e^{-i\mathbf k\cdot\mathbf u}\rangle=e^{-a^2k^2/2}$. For one atom,

$$
{\rm Cov}=\int\!\!\int\frac{d^2q\,d^2q'}{(2\pi)^4}\tilde v(q)\tilde v(q')
e^{i\mathbf q\cdot(\mathbf r-\mathbf R)-i\mathbf q'\cdot(\mathbf r'-\mathbf R)}
\Bigl[e^{-a^2|\mathbf q-\mathbf q'|^2/2}-e^{-a^2(q^2+q'^2)/2}\Bigr].
$$

The atom positions are summed per unit depth using the static site density $n(x_R)=\sum_Gn_Ge^{iGx_R}$. Integrating $y_R$ gives $2\pi\delta(q_y-q'_y)$, and integrating $x_R$ gives $2\pi n_G\,\delta(q_x-q'_x-G)$. Setting $q'_x=p$ and $q_x=p+G$ reproduces the claimed $C$ exactly, including the $(2\pi)^{-2}$, the $e^{iGx}e^{ip(x-x')}$ placement and the bracket.

Hermiticity holds: map $G\to-G$ and $p\to p+G$, then use $n_{-G}=n_G^*$. $C$ is also invariant under a common shift by one period.

**Phase convention.** The $n_G$ must synthesize the density in the same $x$ coordinate as the potential the states live in. `planar_profile` builds $U_n\propto\sum_je^{+i2\pi n\,\mathbf h\cdot\mathbf R_j}$ and synthesizes it with $e^{+ingx}$. That places sites at $x=-(\mathbf h\cdot\mathbf R_j)d$, mirrored relative to the "physical" Fourier convention. This is harmless if every quantity uses it. The required $n_G$ is therefore $V^{-1}\sum_je^{+i2\pi n\,\mathbf h\cdot\mathbf R_j}$, unsmeared.

### (b) $W=\hbar vC(x,x)/2$ and the textbook forms

With $i\hbar v\,\partial_z\psi=(H-iW)\psi$, the norm decays as $\partial_z|\psi|^2=-(2W/\hbar v)|\psi|^2$. The anticommutator term of the dissipator removes $C(x,x)$ per unit depth from the coherent channel. Hence $W=\hbar vC(x,x)/2$.

**$G=0$.** $\langle C\rangle=\bar n(\hbar v)^{-2}\int\frac{d^2q}{(2\pi)^2}\tilde v^2(1-e^{-a^2q^2})$. The first-Born amplitude is $f^e=-(\gamma m/2\pi\hbar^2)\tilde v$, and $\hbar k=\gamma mv$. Then $(\gamma m/2\pi\hbar^2)^2/k^2=1/(4\pi^2\hbar^2v^2)$, so

$$
\langle C\rangle=\bar n\int\frac{d^2q}{k^2}\,|f^e|^2\bigl(1-e^{-a^2q^2}\bigr)=\bar n\,\sigma_{\rm TDS}.
$$

With $s=q/4\pi$, $a^2q^2=2Bs^2$, $f^e=2\gamma(Z-f_x)/(a_0q^2)=\gamma(Z-f_x)/(8\pi^2a_0s^2)$ and $d^2q/k^2=8\pi\lambda^2s\,ds$, this becomes $\sigma_{\rm TDS}=8\pi\lambda^2\int s\,ds\,|f^e|^2(1-e^{-2Bs^2})$, the Pennycook–Jesson form used by Mendis (2024), Eqs. 8–9. The Mott–Bethe $f^e$ carries $\gamma$. That factor cancels against $\lambda^2\propto1/(\gamma v)^2$, leaving $C\propto1/v^2$ and $W\propto1/v$.

**$G\neq0$.** Shift $p=p_0-G/2$. Then $(p+G)^2+p^2+2q_y^2=2q^2+G^2/2$ with $\mathbf q=(p_0,q_y)$, and in $s$ units $a^2G^2/2=Bs_G^2$. The prefactor is $(\hbar v)^{-2}(2\pi\hbar^2/\gamma m)^2\cdot4\,d^2s=4\lambda^2d^2s$. So

$$
D_G=n_G\,4\lambda^2\int d^2s\,f^e\bigl(|\mathbf s+\tfrac12\mathbf s_G|\bigr)f^e\bigl(|\mathbf s-\tfrac12\mathbf s_G|\bigr)
\Bigl[e^{-Bs_G^2}-e^{-2Bs^2-Bs_G^2/2}\Bigr].
$$

This is the symmetric Hall–Hirsch / Bird–King absorptive form factor for an Einstein crystal.

### (c) $q_\perp$-resolved PENELOPE GOS

**Distant longitudinal.** The PENELOPE kernel per oscillator is $\propto dQ/[Q(1+Q/2mc^2)]$ on $Q_-<Q<Q'_k$, with prefactor $\mathcal P f_k/W$ and $\mathcal P=2\pi e^4/(mv^2)$. From $(\hbar cq)^2=Q(Q+2mc^2)$ one gets $dQ=\hbar^2c^2q\,dq/(Q+mc^2)$ and $Q(1+Q/2mc^2)=\hbar^2c^2q^2/2mc^2$. So $dQ/[Q(1+Q/2mc^2)]=2\,dq/[q(1+Q/mc^2)]$. With $q^2=q_\perp^2+q_-^2$, $q\,dq=q_\perp dq_\perp$ and $d^2q_\perp=2\pi q_\perp dq_\perp$, this gives

$$
\frac{d\sigma}{d^2q_\perp}=\frac{\mathcal Pf_k/W}{\pi q^2(1+Q/mc^2)},\qquad q\le q(Q'_k).
$$

For $q\gg q_-$ this is $2e^4f_k/(mv^2Wq^2)$, the Bethe dipole limit, since $|x_{n0}|^2=\hbar^2f/(2mW)$.

**Close (Møller).** For a free target at rest, exact two-body kinematics (in $mc^2=1$ units) give $p^2+p'^2-q^2=2(E-W)(E+2)$. That makes $\cos\theta'=(E-W)(E+2)/(pp')$, and so

$$
(\hbar cq_\perp)^2=2mc^2\,W\,(1-W/E).
$$

This is monotone for $W<E/2$. On that branch $dW/dq_\perp^2=(\hbar c)^2/[2mc^2(1-2W/E)]$ and $d\sigma/d^2q_\perp=(d\sigma/dW)(dW/dq_\perp^2)/\pi$. For $U_k\ll W\ll E$, $F^-\to1$ and $W\simeq(\hbar cq)^2/2mc^2$, which gives $d\sigma/d^2q=4e^4/(\hbar^2v^2q^4)$ per electron. That is Rutherford, and equals $(\hbar v)^{-2}(4\pi e^2/q^2)^2/(2\pi)^2$, the same normalization as the TDS kernel.

**Electronic LDA kernel.** The jumps are $L_{\mathbf q}=\sqrt{w\,d^2q}\;s_q(x)e^{iq_xx}$ with $s_q=\sqrt{\rho_q}$, and $\rho_q$ has harmonics $\rho_Ge^{-G^2/2q^2}$. Summing $L L^\dagger$ gives the claimed $C_{\rm el}$, which is PSD by construction. Its diagonal harmonics are $D_G=\rho_G\int d^2q\,w(q)\,e^{-G^2/2q^2}$. Units: $w$ [Å⁴] × $\rho$ [Å⁻³] × $d^2q$ [Å⁻²] = Å⁻¹.

### (d) Cell-averaged decoherence rate

The coherence obeys $\partial_z\ln\rho(x,x')\supset C(x,x')-\tfrac12C(x,x)-\tfrac12C(x',x')$, so its decay rate is the claimed $\Gamma$. Averaging over $x$ kills $e^{iGx}$ for $G\ne0$. The 2D angular average of $\cos q_x\Delta$ is $J_0(q\Delta)$. This gives:

$$
\Gamma_{\rm TDS}=\sum_s\frac{\bar n_s}{(\hbar v)^2}\frac{1}{2\pi}\int q\,dq\,\tilde v_s^2\bigl(1-e^{-a_s^2q^2}\bigr)\bigl(1-J_0(q\Delta)\bigr),
$$

$$
\Gamma_{\rm el}=2\pi\int q\,dq\,w(q)\Bigl[\bar\rho-J_0(q\Delta)\,\langle s_q(x)s_q(x+\Delta)\rangle_x\Bigr].
$$

Limits:

- $\Gamma(0)=0$.
- $\Gamma_{\rm TDS}(\Delta\to\infty)=\bar n\sigma_{\rm TDS}$.
- For small $\Delta$, $1-J_0\simeq q^2\Delta^2/4$. Against the $1/q^4$ Coulomb tails this gives $\Gamma\propto\Delta^2\ln(1/\Delta)$.

### Units and limits summary

| check | result |
| --- | --- |
| $C$ | $({\rm eV\,\AA^3})^2/({\rm eV\,\AA})^2\cdot{\rm \AA^{-3}}\cdot{\rm \AA^{-2}}={\rm \AA^{-1}}$ ✓ |
| $W=\hbar vC/2$ | eV ✓ |
| $a\to0$ | bracket $\to0$, so $C=0$ ✓ |
| velocity | $C\propto v^{-2}$, $W\propto v^{-1}$, no $\gamma$ ✓ |
| sign | $C(x,x)\ge0$ (variance), so $W\ge0$ absorbs ✓ |

## Diff against the implementation

- **`_potential_table`.** `-4π E2 (z - f0)/q²`, with $f_0$ rescaled by $Z/f_0(0)$, an $s^{-4}$ tail beyond $s=6$ Å⁻¹, and the value held constant below $q=0.02$ Å⁻¹. This is $\tilde v$ with the documented model choices.
- **`_site_densities`.** `exp(+i 2π h (hkl·R))/V` on true-period harmonics `h·step`: the `planar_profile` convention, unsmeared. With $e^{-Bs^2}$ applied and summed over elements, it equals `profile.nuclear_density` to $3\times10^{-15}$ for Si(110), (100), (111) and diamond (111).
- **`_tds_diagonal`.** Symmetric polar form: `v(|q+G/2|) v(|q−G/2|)`, bracket `exp(−2a²(G/2)²) − exp(−a²(q²+G²/4))`, angle $[0,\pi]$ doubled, all divided by $(2\pi\hbar v)^2$. This is $D_G$ exactly.
- **`tds_kernel`.**
  - Bin $l=r+\nu n$ covers $[(l-\tfrac12),(l+\tfrac12)]g/n$, so $p$ is the bra momentum $\kappa_r+\nu g+t$ and $p+G$ is the ket. The entry is `n_G ∫_bin dp ∫ dq_y/(2π)² v v [bracket]`, with `2×` for $q_y<0$, stored at `[r, ket, bra]` and divided by $(\hbar v)^2$. This matches $C=\sum K_{rij}e^{ik_ix}e^{-ik_jx'}$ with $k_i-k_j=G$ and coefficient $n_{h_i-h_j}$.
  - Every bin uses the same two Gauss–Legendre offsets $t$. $K[r,i,j]$ and $K[r,j,i]^*$ are therefore the same products, so Hermiticity is exact before the symmetrization.
  - For each $(t,q_y)$ node the block is the covariance matrix of $X_i=\tilde v(k_i)\sum_{\rm atoms}e^{-ik_i(x_R+u)}$. The blocks are therefore PSD and the cut $|h|\le H$ is a principal submatrix.
- **`ElectronicChannels.distant`.** `S_k/(π q² (1+Q/mc²))` with $q^2=q_\perp^2+q_-^2$ and $Q=\sqrt{(\hbar cq)^2+m^2c^4}-mc^2$, cut at $q\le q(Q'_k)$. Matches (c).
- **`close_loss`.** $\mathcal Pf_kF^-/W^2$ with $a=(E/(E+mc^2))^2=((\gamma-1)/\gamma)^2$ and $E'=E+U_k$, the PENELOPE-2024 binding-modified Møller factor.
- **`close`.** $W=(\hbar cq)^2/[mc^2(1+\sqrt{1-x})]$ with $x=2(\hbar cq)^2/(mc^2E)$ is the cancellation-free lower root of (c). The Jacobian is `ħc²/(2mc² root)` with $\sqrt{1-x}=1-2W/E$, divided by $\pi$. Matches.
- **`electronic_channels`.** Strengths, $q_-$ and $Q'_k$ reuse the `shell_gos_moments` kinematics. The near-threshold $W'_k$, $Q'_k$ and the triangular $p_{\rm dis}$ belong to that claim and are not re-derived here. This write-up checks only the $Q\to q_\perp$ map and that moments are preserved.
- **`electronic_kernel`, `_root_density`, `_electronic_diagonal`.** $\sqrt{\rho_q}$ harmonics on a log-$q$ table. Each bin block is $\sum_tA_{lt}\,c_tc_t^\dagger$ with $A\ge0$, which is PSD. $D_G=\rho_G\int w\,e^{-G^2/2q^2}$, with the close part integrated in $W$ over the full range, including $W>E/2$, as documented.
- **`ScatteringKernel.diagonal`, `absorptive_potential_eV`, `cell_kernel`.** These compute $D_0+2{\rm Re}\sum D_me^{imgx}$, $\tfrac12\hbar vC$, and the position-space synthesis followed by an FFT over residues, giving the cell offset. All match.
- **`decoherence_rate`.** Same integrands as (d). See the numerical finding below.

## Independent numerical checks

All checks are Si(110) at 293.15 K and 17 MeV ($\gamma=34.268$, $\hbar v=1972.43$ eV Å, $\lambda=7.0834\times10^{-4}$ Å, $B=0.48166$ Å², true period $d=1.92011$ Å). The scripts are in `scratch/verify_lindblad/` and use `xraydb.f0` with my own Mott–Bethe and quadrature. The documented model choices ($f_0\to Z$ rescale, $s^{-4}$ tail, $q\le10^4$ Å⁻¹) are reproduced, not the code helpers.

**(1) Hall–Hirsch / Pennycook–Jesson harmonics, Fourier route.** Adaptive `scipy` quadrature of the $s$-unit formulas in (b):

| $m$ | independent [Å⁻¹] | code `diagonal_coefficients` | ratio |
| --- | --- | --- | --- |
| 0 | $6.37910\times10^{-5}$ | $6.37911\times10^{-5}$ | 1.0000029 |
| 1 | $5.56936\times10^{-5}$ | $5.56937\times10^{-5}$ | 1.0000026 |
| 2 | $4.16242\times10^{-5}$ | $4.16243\times10^{-5}$ | 1.0000020 |
| 3 | $2.80595\times10^{-5}$ | $2.80595\times10^{-5}$ | 1.0000021 |

**(2) Position-space covariance route.** This route is fully independent of the Fourier bookkeeping. On a 4096² grid it computes $V_{\rm proj}$ from $\tilde v$, then ${\rm Var}_u[V_{\rm proj}(\mathbf r-\mathbf u)]=G_a*V^2-(G_a*V)^2$. It integrates over $y$ and sums planes of areal density $\bar n d$.

- Mean rate $6.3772\times10^{-5}$ Å⁻¹, against $6.3791\times10^{-5}$ (grid cutoff, $3\times10^{-4}$).
- $W(0)=0.36869$ eV against code $0.36887$ eV. $W(d/2)=0.00123$ eV against $0.00123$ eV. $W(0)/W(d/2)=299$.
- Harmonics $m=1..5$ agree to $\le4\times10^{-4}$.

The same grid gives ${\rm Cov}[\phi(x),\phi(x')]$ off the diagonal, which I compare with `cell_kernel(64)` at default $H=24$, $n_{\rm cells}=32$:

| $(x,x')$ [Å] | position space | `cell_kernel` |
| --- | --- | --- |
| $(0,0)$ | $3.739\times10^{-4}$ | $3.683\times10^{-4}$ (kept band; $H=48$: $3.727$) |
| $(0,0.15)$ | $6.791\times10^{-5}$ | $6.739\times10^{-5}$ ($H=48$: $6.785$) |
| $(0,0.48)$ | $7.048\times10^{-6}$ | $7.109\times10^{-6}$ |
| $(0,d)$ | $6.460\times10^{-8}$ | $6.335\times10^{-8}$ |
| $(0,2d)$ | $-3.94\times10^{-9}$ | $-3.82\times10^{-9}$ |
| $(0.3,0.6)$ | $9.256\times10^{-6}$ | $9.268\times10^{-6}$ |

The differences are the kept-band cut and the $g/n_{\rm cells}$ binning, and they shrink as $H$ grows.

**(3) Structure and limits.**

- `momentum_kernel` Hermitian to 0 (exact). The smallest eigenvalue over all residues divided by the largest is $+2\times10^{-14}$ (TDS) and $+1\times10^{-7}$ (electronic).
- Kept/full mean loss is 0.99204 (TDS) and 0.99987 (electronic).
- $C(2\,{\rm MeV})/C(50\,{\rm MeV})=1.0430964276394517$, equal to $\beta_{50}^2/\beta_2^2$ to $3\times10^{-16}$.
- **Si(111)** is non-centrosymmetric, which tests the phase convention. TDS $W$ maxima sit at $x=0$ and $0.7839$ Å, the same as the `nuclear` maxima. The `potential` minima sit at $0.016$ and $0.768$ Å, displaced toward each other by the overlap of the close plane pair. The convention is consistent.

**(4) Electronic channels**, Si catalog oscillators, 17 MeV:

- The $d^2q_\perp$ integral of my own distant integrand equals $\sum_kS_k\ln[Q'(Q_-+2mc^2)/(Q_-(Q'+2mc^2))]$ to $1\times10^{-8}$. The code's `distant` agrees to $4\times10^{-6}$ (my trapezoid).
- $\int d^2q\,$`close` equals $\int_{U}^{E/2}dW\,$`close_loss` to $3\times10^{-5}$.
- Exact two-body $q_\perp^2$ equals $2W(1-W/E)$ to $10^{-14}$ relative at $W/E=3\times10^{-6}$, $3\times10^{-3}$, 0.25 and 0.49.
- `close`$/[4e^4/(\hbar^2v^2q^4)]$ is $0.2857=4/14$ at $q=5$ Å⁻¹ (valence only, $W=95$ eV), $0.857=12/14$ at $q=20$ (below K, $W=1.5$ keV) and $0.99989$ at $q=100$. This is the Rutherford limit per electron.
- Diagonal harmonics $D_{0..5}$ from my own dense quadrature agree to $10^{-6}$ ($D_0=4.12202\times10^{-4}$ Å⁻¹).
- A direct position-space $C_{\rm el}(x,x')=\int2\pi q\,dq\,w\,s_q(x)s_q(x')J_0(q\Delta)$, with my own smoothing and square root, agrees with `cell_kernel` to $3\times10^{-4}$ on-site, $9\times10^{-4}$ at $\Delta=d$ and $1.2\times10^{-2}$ at $\Delta=5d$. The last is binning-limited, from the supercell phase across a bin.

**(5) Decoherence rate.**

- **Electronic.** Against my own dense quadrature with independent $\langle s_qs_q\rangle$, `decoherence_rate(...).electronic` agrees to $<10^{-4}$ for $\Delta\le5$ Å. The maximum error over $0.5$–$2000$ Å is $6.5\times10^{-3}$, at $\Delta=159$ Å.
- **TDS.** The reference uses 16-point Gauss–Legendre panels of width 0.02 Å⁻¹ up to $q=400$ Å⁻¹, then geometric panels to $10^4$:

| $\Delta$ [Å] | reference | code | rel |
| --- | --- | --- | --- |
| 0.2 | $4.89425\times10^{-5}$ | $4.89365\times10^{-5}$ | $-1.2\times10^{-4}$ |
| 1.0 | $6.43846\times10^{-5}$ | $6.43295\times10^{-5}$ | $-8.6\times10^{-4}$ |
| 1.920 | $6.40792\times10^{-5}$ | $6.43424\times10^{-5}$ | $+4.1\times10^{-3}$ |
| 3.0 | $6.38278\times10^{-5}$ | $6.50694\times10^{-5}$ | $+1.95\times10^{-2}$ |
| 5.0 | $6.37912\times10^{-5}$ | $6.27290\times10^{-5}$ | $-1.67\times10^{-2}$ |
| 11.17 | — | — | $-3.6\times10^{-2}$ (max) |
| 100 | $6.37910\times10^{-5}$ | $6.35763\times10^{-5}$ | $-3.4\times10^{-3}$ |

The error exceeds 1% for $2.5\lesssim\Delta\lesssim64$ Å. The reference agrees with the kept-band cell kernel: $\Gamma(d)$ estimated from `cell_kernel` plus the full-minus-kept diagonal is $6.408\times10^{-5}$, the reference value.

$\Gamma(0)=0$ exactly. $\Gamma/\Delta^2$ grows logarithmically down to the $q\le10^4$ Å⁻¹ cutoff. The repository test compares $\Gamma(10^5\,{\rm \AA})$ with the mean loss and passes at $10^{-3}$, but at that $\Delta$ the $J_0$ term is sampled far below Nyquist and only averages out by chance.

## Findings

1. **Numerical (blocking for `anchored`).** `src/pyrite/montecarlo/channeling/operators.py:805` integrates $\Gamma_{\rm TDS}$ on `_panels(_geometric_edges(0.05, 1e4), 8)`. These are ratio-2 geometric panels with 8 nodes. At $q\sim1/a\approx13$ Å⁻¹, where the TDS integrand peaks, a panel spans about 8 Å⁻¹. That is many periods $2\pi/\Delta$ of $J_0(q\Delta)$ once $\Delta\gtrsim1$ Å. Failing case: `decoherence_rate(planar_profile('silicon',(1,1,0)), 17e6, [3.0, 11.17]).tds` returns values $+1.95\%$ and $-3.6\%$ from the converged integral, where the converged integral is $6.38278\times10^{-5}$ and $\approx6.3791\times10^{-5}$ Å⁻¹. The formula is right. The quadrature needs panels narrower than about $\pi/\Delta$ over the support of the integrand, or an asymptotic treatment for large $q\Delta$. The electronic branch uses ratio-1.5 panels and has the same weakness at the $<0.7\%$ level. The test `test_decoherence_rate_limits` should probe an intermediate $\Delta$ (say 3–10 Å) against a converged reference.
2. **Minor.** The physical $C(x,x')$ is real. The kept band uses residues $\kappa_r\in[0,g)$ with $|h|\le H$, so it covers transfers $[-Hg,(H+1)g)$ asymmetrically. This leaves ${\rm Im}\,C$ up to $6\times10^{-4}$ of $\max|C|$ in `cell_kernel`. The result is still the correct projection $PCP$ for that basis and is not a defect.
3. **Minor, unledgered helper.** `ScatteringKernel.kept_diagonal` raises on a scalar `x_ang`, because its `einsum` needs a 1-D array. `diagonal` accepts scalars.
4. **Scope.** The PENELOPE oscillator model is inherited from `shell_gos_moments` and validated there: near-threshold $W'_k$ and $Q'_k$, $Q_-$ evaluated at $W'_k$ for the triangular band, and close range $[U_k,(E+U_k)/2]$. The electronic LDA localization is a maintainer model choice, not a derivation.

## Adjudication

Every expression in the claim re-derives from first principles: the dissipator, $W=\hbar vC/2$, the TDS phase-covariance kernel (bracket, $(\hbar v)^{-2}$, $(2\pi)^{-2}$, $n_G$ phase convention), the Pennycook–Jesson and Hall–Hirsch equivalences with relativistic $f^e$, the PENELOPE $Q\to q_\perp$ Jacobians, the binary-kinematics map, the Bloch-residue PSD argument and the $\Gamma(\Delta)$ integrands. The kernels agree with two independent numerical routes to $3\times10^{-6}$ (Fourier) and about $5\times10^{-4}$ (position space).

The one failed check is numerical. `decoherence_rate` evaluates the TDS $\Gamma(\Delta)$ with up to 3.6% quadrature error for $2.5\lesssim\Delta\lesssim64$ Å.

**Verdict: `discrepancy`, numerical and confined to `decoherence_rate`'s TDS quadrature (operators.py:805).** The kernel symbols (`tds_kernel`, `electronic_kernel`, `electronic_channels`, `ElectronicChannels.*`, `ScatteringKernel.*`) are `rederived`. After the quadrature is refined and an intermediate-$\Delta$ check is added, the whole claim qualifies for `rederived` without re-derivation.

## Resolution (lead, after the verdict)

Finding 1 is fixed. `decoherence_rate` now evaluates the $J_0$ term on panels at most half a period ($\pi/\Delta$) wide, up to $Q_1=2\pi\cdot400/\Delta$. It drops the oscillating tail beyond $Q_1$ behind a $\cos^2$ taper over $[0.75Q_1,Q_1]$, so no boundary term survives. The non-oscillating mean keeps the original grids. The electronic branch uses the same treatment. Against this write-up's converged reference, $\Gamma_{\rm TDS}$ agrees to $5.7\times10^{-6}$, $4.1\times10^{-6}$ and $2.8\times10^{-6}$ at $\Delta=1.92$, 3.0 and 5.0 Å. $\Gamma_{\rm el}(10^5\,{\rm \AA})$ now equals the electronic mean loss to $10^{-6}$. `test_tds_decoherence_rate_matches_converged_reference` pins those reference values, and `test_tds_decoherence_rate_matches_cell_kernel` pins agreement with the position-space cell kernel at $\Delta=d,2d$. Finding 3 is fixed: `kept_diagonal` accepts scalars. The fix was not re-verified in a fresh context.
