# `longitudinal-quantum-bunching`

## Claim and source

Research claim (#629): separable i.i.d. electrons on an energy comb
$E_n=E_0+n\hbar\Omega$ emit photons at $\omega=\ell\Omega$. The pair amplitude
is $\operatorname{Tr}(\rho\hat M)=\sum_n\rho_{n,n-\ell}A_n$, with a
state-independent bound $\max\lvert A\rvert\sum_n\sqrt{p_np_{n-\ell}}$. That
bound is attained, and for a Gaussian of rms $\sigma$ it equals
$e^{-(\hbar\omega)^2/8\sigma^2}$. The vacuum-drift bunching coefficient is
$B_\ell(z)=\sum_n\rho_{n+\ell,n}e^{i(k_{n+\ell}-k_n)z}$, with exact on-shell
$k_n$. For ideal PINEM with quadratic dispersion,
$\lvert B_\ell\rvert=\lvert J_\ell(4\lvert g\rvert\sin(2\pi\ell z/z_T))\rvert$,
where $z_T=4\pi(pc)^3\hbar c/[(mc^2)^2(\hbar\Omega)^2]$. This reduces to the
classical $\lvert J_\ell(\ell X)\rvert$, $X=8\pi\lvert g\rvert z/z_T$.

Sources cited by the ledger:

- the pair expansion of `quantum-pair-population-normalization`;
- positivity, $\lvert\rho_{nm}\rvert^2\le\rho_{nn}\rho_{mm}$;
- the Neumann–Graf addition theorem;
- the research note `docs/research/physics/longitudinal-coherence-recoil.md`, §3.2;
- Feist et al., Nature 521, 200 (2015) and Priebe et al., Nat. Photon. 11, 793
  (2017), both not inspected by the author or by this verifier.

Code under review is `src/pyrite/quantum/_longitudinal.py::{comb_wavenumbers_inv_ang,
talbot_length_ang, pinem_amplitudes, bunching_coefficient,
pinem_bunching_closed_form, klystron_bunching, pair_amplitude,
pair_amplitude_bound}`.

This derivation was written from the ledger row and the stated assumptions
before the implementation, the research note §3.2 and the tests were read.

## Setup and index convention

A comb state is $\rho=\sum_{nm}\rho_{nm}\lvert E_n\rangle\langle E_m\rvert$
with $p_n=\rho_{nn}$ and $\operatorname{Tr}\rho=1$. In a static medium the
one-photon operator for a mode at $\omega=\ell\Omega$ conserves energy:

$$
\hat M\lvert E_n\rangle=A_n\lvert E_n-\ell\hbar\Omega\rangle=A_n\lvert E_{n-\ell}\rangle .
$$

The ordinary-matrix trace gives

$$
\operatorname{Tr}(\rho\hat M)=\sum_k\langle E_k\rvert\rho\hat M\lvert E_k\rangle
=\sum_kA_k\langle E_k\rvert\rho\lvert E_{k-\ell}\rangle
=\sum_n\rho_{n,n-\ell}A_n .
$$

This matches the claim. For a pure state $c_n$ it is
$\sum_nc_nc_{n-\ell}^*A_n=\langle\psi\rvert\hat M\lvert\psi\rangle$. By the
i.i.d. limit of `quantum-pair-population-normalization`, this amplitude enters
only $N(N-1)\lvert\operatorname{Tr}(\rho\hat M)\rvert^2$. Only $\ell\ge1$
(emission) is physical.

### Bunching coefficient after drift

Each energy component is stationary in vacuum. At fixed $z$, component $n$
carries $e^{i(k_nz-E_nt/\hbar)}$ with $\hbar ck_n=\sqrt{E_n^2-(mc^2)^2}$. The
density is

$$
\lvert\psi(z,t)\rvert^2\;\to\;\sum_{nm}\rho_{nm}\,e^{i(k_n-k_m)z}\,e^{-i(n-m)\Omega t}.
$$

Write $n=m+\ell$. The coefficient of $e^{-i\ell\Omega t}$ is

$$
B_\ell(z)=\sum_m\rho_{m+\ell,m}\,e^{i(k_{m+\ell}-k_m)z}.
$$

This is the same diagonal $\rho_{n,n-\ell}$ as the pair amplitude. With
constant $A$ and $\rho(z)$ carrying the drift phases,
$\operatorname{Tr}(\rho(z)\hat M)=A\,B_\ell(z)$ exactly, with no conjugation
mismatch. Also $B_{-\ell}=B_\ell^*$. Drift changes only phases, so $p_n$ is
$z$-independent.

## State-independent bound

Positivity of $\rho$ gives $\lvert\rho_{n,n-\ell}\rvert\le\sqrt{p_np_{n-\ell}}$.
Then

$$
\lvert\operatorname{Tr}(\rho\hat M)\rvert\le\sum_n\lvert A_n\rvert\,\lvert\rho_{n,n-\ell}\rvert
\le\max_n\lvert A_n\rvert\sum_n\sqrt{p_np_{n-\ell}} .
$$

**Attainment.** Both inequalities are equalities for the pure state
$c_n=\sqrt{p_n}$ (any common linear phase $e^{in\theta}$ allowed). The
amplitudes must also have constant modulus $\max\lvert A\rvert$ on the
overlapping support, with phases that cancel those of $\rho_{n,n-\ell}$;
constant $A$ suffices. The ledger and the research note say "attained by
amplitudes $\sqrt{p_n}$" without the constant-$A$ condition. The code docstring
states it.

**Gaussian.** Let $p_n\propto e^{-(n\hbar\Omega)^2/2\sigma^2}$, so $\sigma$ is
the rms of the energy *probability*. In the continuum limit,

$$
\int dE\,\sqrt{p(E)\,p(E-\hbar\omega)}
=\frac{e^{-(\hbar\omega)^2/8\sigma^2}}{\sqrt{2\pi}\sigma}\int dE\,e^{-(E-\hbar\omega/2)^2/2\sigma^2}
=e^{-(\hbar\omega)^2/8\sigma^2}.
$$

On the discrete comb, write $s=\sigma/\hbar\Omega$. Poisson summation leaves a
relative correction of order $4e^{-2\pi^2s^2}$, and only for odd $\ell$, where
the half-integer shift enters. This is negligible for $s\gtrsim1$: the numeric
agreement is $10^{-15}$ at $s=3$. The equivalent statement is
$\lvert F_1\rvert=e^{-\omega^2\sigma_t^2/2}$, with $\sigma_t=\hbar/2\sigma$ the
rms duration of the transform-limited density. That is the classical Gaussian
bunch form factor.

## Talbot length

Expand $k(E_0+n\hbar\Omega)$ to second order. The derivatives of
$\hbar ck=\sqrt{E^2-(mc^2)^2}=pc$ are

$$
\frac{dk}{dE}=\frac{E}{\hbar c\,pc}=\frac1{\hbar v},\qquad
\frac{d^2k}{dE^2}=\frac{(pc)^2-E^2}{\hbar c\,(pc)^3}=-\frac{(mc^2)^2}{\hbar c\,(pc)^3}.
$$

The $n$-dependent quadratic phase is
$\tfrac12\lvert k''\rvert(\hbar\Omega)^2n^2z\equiv\tfrac12\alpha n^2$, with
$\alpha=\lvert k''\rvert(\hbar\Omega)^2z$. At $\alpha=2\pi$,
$\pi n^2\equiv\pi n\pmod{2\pi}$ is linear in $n$, giving a half-period-shifted
image. At $\alpha=4\pi$ every phase is $\equiv0$, giving an exact revival.
With the optical (Rayleigh) convention that $z_T$ is the exact-revival
distance,

$$
\boxed{z_T=\frac{4\pi}{\lvert k''\rvert(\hbar\Omega)^2}
=\frac{4\pi(pc)^3\,\hbar c}{(mc^2)^2(\hbar\Omega)^2}},
\qquad\tfrac12\alpha n^2=2\pi n^2z/z_T .
$$

The units are $\mathrm{eV^3\cdot eV\,\AA/(eV^2\,eV^2)}=\mathrm{\AA}$. The
non-relativistic limit is $z_T=4\pi m\gamma^3v^3/(\hbar\Omega^2)$ with
$\gamma\to1$. The factor $4\pi$ is a convention; it is consistent with the
closed form below. Whether it matches Priebe et al. (2017) is **source-dependent**.

## Ideal-PINEM closed form

**Amplitude convention.** A sinusoidal phase modulation
$e^{i\,2\lvert g\rvert\sin(\Omega t+\theta)}$, through the generating function
$e^{ix\sin\phi}=\sum_nJ_n(x)e^{in\phi}$, gives
$c_n=J_n(2\lvert g\rvert)e^{in\theta}$, which is normalized
($\sum_nJ_n^2=1$). The ledger takes $\theta=\arg g$. Other conventions
($\arg(-g)$, or $n\to-n$ for gain versus loss) change $B_\ell$ only by an
overall phase. They leave $\lvert B_\ell\rvert$ unchanged, because the Graf
modulus below depends only on $\lvert\sin\rvert$. The factor 2 in
$J_n(2\lvert g\rvert)$ is load-bearing: it fixes the energy-modulation
amplitude $\Delta E=2\lvert g\rvert\hbar\Omega$, the comb full width
$\approx4\lvert g\rvert$ quanta and $X$. It is internally consistent. Its
identification with the physical coupling of Feist et al. (2015) is
**source-dependent**.

**Drift.** With the quadratic phase,
$(k_{n+\ell}-k_n)z=\text{const}_\ell-\alpha\ell n$, where $k''<0$ and the
constant includes $k'\ell\hbar\Omega z$ and $\tfrac12k''(\hbar\Omega)^2\ell^2z$.
Then

$$
B_\ell=e^{i\ell\theta}e^{i\,\text{const}_\ell}\sum_nJ_{n+\ell}(x)J_n(x)e^{-in\ell\alpha},\qquad x=2\lvert g\rvert .
$$

Graf's addition theorem for integer order holds for all arguments:

$$
J_\nu(w)e^{i\nu\chi}=\sum_kJ_{\nu+k}(u)J_k(v)e^{ik\phi},
\qquad w=\sqrt{u^2+v^2-2uv\cos\phi}.
$$

With $u=v=x$ and $\phi=-\ell\alpha$, this gives
$w=2x\lvert\sin(\ell\alpha/2)\rvert$. Since $\ell\alpha/2=2\pi\ell z/z_T$,

$$
\boxed{\lvert B_\ell(z)\rvert=\bigl\lvert J_\ell\bigl(4\lvert g\rvert\sin(2\pi\ell z/z_T)\bigr)\bigr\rvert}.
$$

The checks are:

- $z=0$ gives $\delta_{\ell0}$, since a fresh PINEM state is unbunched.
- $\lvert B_\ell\rvert$ has period $z_T/(2\ell)$ in $z$.
- At $z=z_T/2$ every $\lvert B_{\ell\ne0}\rvert=0$, the shifted revival of the
  unbunched state.

## Classical limit and convergence

Hold $X=8\pi\lvert g\rvert z/z_T$ fixed and let $y=2\pi\ell z/z_T\to0$. Then
$4\lvert g\rvert y=\ell X$, and the argument becomes

$$
4\lvert g\rvert\sin y=\ell X\left(1-\frac{y^2}{6}+O(y^4)\right).
$$

So $\lvert B_\ell\rvert\to\lvert J_\ell(\ell X)\rvert$, with error
$\approx\lvert J_\ell'(\ell X)\rvert\,\ell X\,y^2/6=O\bigl((\ell z/z_T)^2\bigr)$.
At fixed $\ell$ this is the claimed $O((z/z_T)^2)$. The convergence is only at
fixed $X$; at fixed $\lvert g\rvert$ both sides tend to zero. The research
note states "at fixed $X$"; the ledger row omits it.

**Classical cross-check.** Take a velocity-modulated beam,
$E=E_0+\Delta E\sin\Omega t_0$. Its arrival time at $z$ is
$t=t_0+z\,dk/dE\cdot\hbar$, so
$\Delta t=z\hbar\lvert k''\rvert\Delta E\sin\Omega t_0$. Hence
$b_\ell=J_\ell(\ell\,\hbar\Omega\lvert k''\rvert\Delta Ez)$. With
$\Delta E=2\lvert g\rvert\hbar\Omega$:

$$
\hbar\Omega\lvert k''\rvert\Delta E\,z=2\lvert g\rvert\lvert k''\rvert(\hbar\Omega)^2z=8\pi\lvert g\rvert z/z_T=X .
$$

This matches the research note's $X=\hbar\Omega\lvert k''\rvert\Delta Ez$.

## Cutoff beyond the coherent comb width

Two independent arguments give the cutoff.

1. **Energy-overlap bound.** The PINEM populations $J_n(2\lvert g\rvert)^2$ are
   concentrated in $\lvert n\rvert\lesssim2\lvert g\rvert$, plus Airy tails of
   width $\sim\lvert g\rvert^{1/3}$. For $\ell>4\lvert g\rvert$,
   $\sum_n\sqrt{p_np_{n-\ell}}$ is small for every $z$, because drift does not
   change $p_n$.
2. **Closed form.** The maximum over drift is
   $\max_z\lvert B_\ell\rvert=\max_{0\le w\le4\lvert g\rvert}\lvert J_\ell(w)\rvert$.
   For $4\lvert g\rvert<\ell$ this is $\lvert J_\ell(4\lvert g\rvert)\rvert$,
   below the turning point. It is exponentially small only once
   $\ell-4\lvert g\rvert\gg\ell^{1/3}$, and it is an Airy transition near
   $\ell\approx4\lvert g\rvert$. Once $4\lvert g\rvert$ passes the first
   maximum of $J_\ell$ (at $\approx\ell+0.81\ell^{1/3}$), the maximum is
   *exactly* the classical maximum.

**Quantum parameter at the classical optimum.** Here
$y=\ell X/(4\lvert g\rvert)=X\,\ell\hbar\Omega/(2\Delta E)=X\,\hbar\omega/(2\Delta E)$.
The classical optimum is $X_{\rm opt}=1.84,1.53,1.28,1.18,1.08,1.03$ for
$\ell=1,2,5,10,32,179$, so $X_{\rm opt}\to1$ as $\ell\to\infty$. The note's
$y\approx\hbar\omega/(2\Delta E)$ therefore holds for $\ell\gg1$. The
classical asymptote $\max_X\lvert J_\ell(\ell X)\rvert\simeq
2^{1/3}\max{\rm Ai}\,\ell^{-1/3}=0.675\,\ell^{-1/3}$ matches the note's
$\approx0.68\,\ell^{-1/3}$.

The table gives $\max_z\lvert B_\ell\rvert$ from the closed form, maximized
over $w$ on a $2\times10^6$-point grid, against the classical value:

| $\lvert g\rvert$ | $\ell$ | $4\lvert g\rvert/\ell$ | quantum max | classical max |
| --- | --- | --- | --- | --- |
| 2 | 8 | 1.0 | 0.224 | 0.324 |
| 2 | 9 | 0.89 | 0.126 | 0.313 |
| 2 | 12 | 0.67 | $9.6\times10^{-3}$ | 0.286 |
| 10 | 44 | 0.91 | $2.7\times10^{-2}$ | 0.189 |
| 10 | 60 | 0.67 | $1.3\times10^{-7}$ | 0.171 |
| 3 | 30 | 0.40 | $<10^{-5}$ (test) | 0.214 |

The physics statement holds: coherence at $\hbar\omega$ needs a coherent comb
wider than $\hbar\omega$. "Vanishes once $\hbar\omega$ exceeds
$\approx4\lvert g\rvert\hbar\Omega$" is a soft threshold. A factor
$\approx1.5$ beyond it is needed for strong suppression at small $\ell$.

## Limits and filters

| check | result |
| --- | --- |
| units | $\operatorname{Tr}(\rho\hat M)$ carries $A$'s units; $B_\ell$, $J_\ell$, $z/z_T$, $X$ dimensionless; $z_T$ in Å |
| $N=1$, $\ell\to0$ | $B_0=1$ (trace); pair amplitude defined for $\ell\ge1$ only |
| mixture | diagonal $\rho$ gives $\operatorname{Tr}(\rho\hat M)=0$ and $B_{\ell\ne0}=0$ |
| $z=0$ PINEM | $B_\ell=\delta_{\ell0}$ |
| $\ell z/z_T\to0$ at fixed $X$ | classical klystron $\lvert J_\ell(\ell X)\rvert$ |
| non-relativistic $z_T$ | $4\pi mv^3/(\hbar\Omega^2)$ |
| phase convention of $g$ | $\lvert B_\ell\rvert$ invariant |
| Gaussian | $\lvert F_1\rvert^2=e^{-\omega^2\sigma_t^2}$, $\sigma_t=\hbar/2\sigma$ |

## Independent numerical check

A scratch script (not committed, with no `pyrite` imports) used mpmath at 50
digits for on-shell $k_n$ at 200 keV and $\hbar\Omega=1.55$ eV:

- $z_T=0.4776$ m. The $4\pi/(\lvert k''\rvert(\hbar\Omega)^2)$ from a
  50-digit finite difference agrees to $2\times10^{-17}$.
- The exact-$k$ sum against the closed form, for $\lvert g\rvert=3$ and
  $\ell=1,2,5$, differs by $\le2\times10^{-9}$ for
  $z/z_T\le0.13$ (one $\ell=5$ point $9\times10^{-9}$). The difference is
  $4\times10^{-4}$ at $z/z_T=0.25$ for $\ell=2$, where cubic dispersion
  becomes visible.
- For the classical limit at fixed $X=1.2$, $\ell=3$, the ratio
  error$/(z/z_T)^2\to24.0$, consistent with
  $\lvert J_3'(3.6)\rvert\cdot3.6\cdot(6\pi)^2/6$.
- For Gaussian overlaps at $s=3,8$ and $\ell\le10$, the discrete sum equals
  $e^{-\ell^2/8s^2}$ to $10^{-15}$.
- For the overlap bound against the drift maximum for PINEM ($\lvert g\rvert=2$),
  the bound always exceeds the drift maximum and is nearly saturated by it for
  $\ell>4\lvert g\rvert$.

## Comparison with the implementation

Read after the sections above were fixed.

| symbol | code | this derivation | agreement |
| --- | --- | --- | --- |
| `comb_wavenumbers_inv_ang` | $pc=\sqrt{T}\sqrt{T+2mc^2}$, $T=T_0+n\hbar\Omega$, divided by `HBARC_EV_ANG` | $\hbar ck_n=\sqrt{E_n^2-(mc^2)^2}$ | matches; it agrees with 50-digit $k_n$ to $5.7\times10^{-14}$ Å$^{-1}$, which is float64 ulp at $k\approx250.5$ Å$^{-1}$ |
| `talbot_length_ang` | $4\pi(pc)^3\hbar c/((mc^2)^2(\hbar\Omega)^2)$ | same | matches; the docstring's $k''$ and $2\pi n^2z/z_T$ also match |
| `pinem_amplitudes` | $J_n(2\lvert g\rvert)e^{in\arg g}$ | same | matches |
| `bunching_coefficient` | `np.diagonal(rho, offset=-l)` $=\rho_{n+\ell,n}$; $k_{n+\ell}-k_n$; correct for $\ell<0$ too | same | matches |
| `pinem_bunching_closed_form` | $\lvert J_\ell(4\lvert g\rvert\sin(2\pi\ell z/z_T))\rvert$ | same | matches |
| `klystron_bunching` | $\lvert J_\ell(\ell X)\rvert$ | same | matches |
| `pair_amplitude` | `diagonal(rho, -l) * A[l:]` $=\sum_n\rho_{n,n-\ell}A_n$; returns 0 for $\ell\le0$ | same | matches; numerically equal to $\sum_nc_nc_{n-\ell}^*$ and to $B_\ell(0)$ with $A=1$ |
| `pair_amplitude_bound` | $\sum_n\sqrt{p_np_{n-\ell}}$, without the $\max\lvert A\rvert$ factor (documented) | same | matches |

The anchors in
`tests/montecarlo/quantum/test_longitudinal_coherence_reference.py` pass
(34 tests, `rc=0`, at `50107731a` and again at `bf94479d5`, whose `_longitudinal.py` change is a slab-mismatch docstring only). The `checks/longitudinal_coherence_regimes.py`
script also exits 0.

### Findings in the evidence and wording

1. **The residual is roundoff, not cubic dispersion.** In
   `test_pinem_bunching_matches_neumann_graf_closed_form`, the comment says
   "Residual is the cubic dispersion term omitted by the closed form", the
   research note §3.2 says "differs … by the omitted cubic dispersion, below
   $2\times10^{-6}$", and the ledger Checks says "(2e-6, cubic dispersion
   residual)". At the tested $z/z_T\le0.05$ the cubic term contributes
   $\le6\times10^{-9}$. The observed code residual, $1$–$7\times10^{-7}$, is
   float64 cancellation in $k_{n+\ell}-k_n$: $\delta k\approx6\times10^{-14}$
   Å$^{-1}$ times $z\sim10^8$ Å. Code against the 50-digit exact sum gives
   $3.6\times10^{-7}$ and $6.7\times10^{-7}$; the 50-digit sum against the
   closed form gives $3\times10^{-10}$ and $6\times10^{-9}$. The tolerance is
   adequate, and the attribution should change.
2. **Two regime-table entries are grid artifacts.** In research note §5, the
   $\ell=10$ entries for $\lvert g\rvert=100,300$ are 0.302 and 0.294, and the
   $\ell=32$ entry for $\lvert g\rvert=300$ is 0.208. Each case has
   $4\lvert g\rvert$ beyond the first maximum of $J_\ell$, so the true drift
   maximum equals the classical 0.303 and 0.209. The 4001-point drift grid in
   `_bunching_cap` under-resolves the $J_\ell$ peak when $4\lvert g\rvert\gg\ell$.
   The $\lvert g\rvert=10$, $\ell=968$ "0" is float underflow; the true value
   is $\lvert J_{968}(40)\rvert\approx2.5\times10^{-1213}$. Verified values:
   $\lvert J_{179}(40)\rvert=7.34\times10^{-96}$ and
   $\lvert J_{968}(400)\rvert=1.75\times10^{-263}$.
3. **One bound is misstated.** Research note §3.2 says "For $\sigma_E\le1.5$ eV
   this is $\le2\times10^{-121}$ at 50 eV". In fact
   $e^{-2500/9}=2.30\times10^{-121}$, and the regime script prints `2.3e-121`.
   It should read $\le2.3\times10^{-121}$ or $\approx2\times10^{-121}$. This
   bullet is also Gaussian-specific: for a non-Gaussian spread the bound is the
   general overlap.
4. **The cutoff wording is loose.** In research note §3.2, the summary bullet
   and the ledger Checks, "vanishes once $\hbar\omega$ exceeds
   $\approx4\lvert g\rvert\hbar\Omega$" and "$J_\ell$ of an argument at most
   $4\lvert g\rvert<\ell$ is exponentially small" hold only for
   $\ell-4\lvert g\rvert\gg\ell^{1/3}$. Near threshold there is an Airy
   transition, for example 0.126 against a classical 0.313 at $\lvert g\rvert=2$,
   $\ell=9$. The anchor (test `ℓ=30`, $\lvert g\rvert=3$) sits deep in the
   suppressed regime. This is wording only.
5. **Attainment needs a condition.** In the ledger and research note §3.2,
   "attained by amplitudes $\sqrt{p_n}$" should add "with constant $A$", as the
   code docstring does.
6. **The classical limit needs "fixed $X$".** The ledger Claim should say "as
   $\ell z/z_T\to0$ at fixed $X$". The note and the code docstring already do.
7. **The quantum parameter is approximate.** "$2\pi\ell z/z_T\approx\hbar\omega/(2\Delta E)$
   at the classical optimum" is exact as $y=X_{\rm opt}\hbar\omega/(2\Delta E)$,
   with $X_{\rm opt}\to1$ only for $\ell\gg1$ ($X_{\rm opt}=1.84$ at $\ell=1$).
   It is acceptable as "$\approx$" for the soft-X-ray harmonics the note
   discusses.

## Source-dependent items

These items are not verifiable without Feist et al. (2015) and Priebe et al.
(2017):

- the identification of $g$ with the physical PINEM coupling, and the factor
  2 in $J_n(2\lvert g\rvert)$, which sets $\Delta E$, the comb width, $X$ and
  the $\lvert g\rvert\gtrsim\ell/4$ requirement;
- the sign or phase convention $e^{in\arg g}$, which is unobservable in
  $\lvert B_\ell\rvert$;
- whether the literature Talbot distance is $4\pi/(\lvert k''\rvert(\hbar\Omega)^2)$
  (exact revival) or half of it. The internal convention is self-consistent
  with the closed form.

The internal mathematics (index convention, bound, Gaussian, $z_T$ under the
stated convention, Graf closed form, classical limit) does not depend on those
sources.

## Adjudication

Every load-bearing identity re-derives:

- the pair amplitude and its index convention;
- its consistency with $B_\ell$;
- the positivity bound and its attainment;
- the Gaussian factor $e^{-(\hbar\omega)^2/8\sigma^2}$ for probability rms $\sigma$;
- the Talbot length with $4\pi$ under the exact-revival convention;
- the Neumann–Graf closed form;
- the fixed-$X$ classical limit with $O((z/z_T)^2)$ error.

The code matches symbolically and numerically. No factor, sign, exponent or
index differs. The findings are wording and evidence-attribution corrections
(items 1–7), not physics errors.

Verifier verdict: `rederived`. PINEM-convention fidelity to Feist and Priebe
remains source-dependent. Human sign-off is still required.

## Owner source check (2026-11-08, after the verdict)

The source-dependent items above were checked by the claim owner, not by this
verifier:

- **`g` normalization and phase.** Feist et al. (2015), Methods Eqs. (13)–(15):
  $S=\exp(g a^\dagger-g^*a)$ gives
  $\langle N\rvert S\lvert0\rangle=g^N\sum_n(-\lvert g\rvert^2)^n/[n!(n+N)!]=e^{iN\arg g}J_N(2\lvert g\rvert)$,
  with the cutoff $\lvert E-E_0\rvert=2\lvert g\rvert\hbar\Omega$. This matches
  `pinem_amplitudes`, including the factor 2 and the phase.
- **Talbot convention.** Neither Feist nor Priebe states $z_T$ in closed form.
  Feist propagates with the exact relativistic dispersion (Methods Eq. 16).
  Priebe et al. (2017) expect the shortest attosecond pulses at 2.75 mm for
  $g=3.95$ (120 keV, 800 nm). Exact-$k$ propagation of the ideal comb puts the
  peak density at 2.75 mm and the minimum FWHM at 2.90 mm; a factor-2 error
  in either $J_n(2\lvert g\rvert)$ or $z_T$ would move it to about 1.4 or
  5.5 mm. Anchor:
  `test_pinem_attosecond_focus_matches_priebe_2017`.
