# Minimal-phase CTR profile reconstruction verification

Validation: `optical-phase-retrieval`.

**Code:** `src/pyrite/detectors/optical_phase_retrieval.py::minimal_phase`,
`::reconstruct_minimal_phase_profile`, `::MissingBandModel`,
`::propagate_reconstruction_uncertainty`, `::FormFactorUncertainty`
(helpers `_extend`, `_chi2`, `_phi`, `_synthesize`, `_power_fractions`,
`_spectral_minima`).

**Sources:** B. Leissner et al., *Bunch length measurements using a Martin
Puplett interferometer at the TESLA Test Facility linac*, PAC 1999,
pp. 2172–2174: Eq. (2) (form factor), Eq. (6) (minimal phase, citing Lai and
Sievers, NIM A 397, 221 (1997)), Eq. (7) (inverse transform). Standard
Hilbert-transform (Titchmarsh) relations for functions analytic in a half
plane; Paley–Wiener log-integrability condition; Legendre chi function
$\chi_2(r)=\sum_{k\ge0}r^{2k+1}/(2k+1)^2$.

Reviewed at `aca5952a8` (branch `issues-615-617-optical-gates`), fresh
context, 2026-11-10.

## Part I — Independent derivation

Written from the source PDF, the ledger row and the module, class and function
docstrings and signatures only, before any implementation body was read.

### I.1 Conventions, analyticity and the sign of Eq. (6)

Leissner Eq. (2) is, with $z=ct$,

$$
f(\omega)=\int\rho(z)\,e^{i\omega z/c}\,dz=\int c\rho(ct)\,e^{i\omega t}\,dt .
$$

Write $\rho(t)$ for the time-domain density, so $f(\omega)=\int\rho(t)e^{i\omega t}dt$.
For a causal profile ($\rho=0$ for $t<0$) and $\omega=\omega_r+i\omega_i$,
$\lvert e^{i\omega t}\rvert=e^{-\omega_i t}$, so $f$ is analytic and bounded in the
**upper** half plane $\omega_i>0$. If $f$ also has no zeros there, $\ln f$
is analytic in the upper half plane, which is the minimal-phase case.

For $F$ analytic in the upper half plane and decaying suitably,
$F(\omega)=\frac{1}{i\pi}\,\mathrm{PV}\!\int_{-\infty}^{\infty}\frac{F(u)}{u-\omega}du$,
so $\operatorname{Im}F(\omega)=-\frac1\pi\mathrm{PV}\!\int\frac{\operatorname{Re}F(u)}{u-\omega}du$.
Real $\rho$ gives $f(-\omega)=f^*(\omega)$: $\ln\lvert f\rvert$ is even and
$\psi=\arg f$ is odd. Folding $(-\infty,0)$ onto $(0,\infty)$,

$$
\int_{-\infty}^{\infty}\frac{A(u)}{u-\omega}du
=\int_0^\infty A(u)\Bigl[\frac1{u-\omega}-\frac1{u+\omega}\Bigr]du
=\int_0^\infty\frac{2\omega\,A(u)}{u^2-\omega^2}du ,
$$

so with $F=\ln f$, $A=\ln\lvert f\rvert$:

$$
\psi(\omega)=-\frac{2\omega}{\pi}\,\mathrm{PV}\!\int_0^\infty\frac{\ln\lvert f(u)\rvert}{u^2-\omega^2}du
=-\frac{2\omega}{\pi}\int_0^\infty\frac{\ln\bigl(\lvert f(u)\rvert/\lvert f(\omega)\rvert\bigr)}{u^2-\omega^2}du ,
$$

the second form using $\mathrm{PV}\!\int_0^\infty du/(u^2-\omega^2)=\frac1{2\omega}\ln\lvert\frac{u-\omega}{u+\omega}\rvert\Big|_0^\infty=0$.
After the subtraction the integrand is regular at $u=\omega$, so the PV is
no longer needed. **This is Leissner Eq. (6), sign included, in the Eq. (2)
convention $e^{+i\omega t}$.**

*Sign check.* $h(t)=t\,e^{-t/\tau}/\tau^2$ ($t>0$) gives
$f=1/(1-i\omega\tau)^2$: double pole at $\omega=-i/\tau$ (lower half plane), no
zeros, so minimal phase, with $\lvert f\rvert=1/(1+x^2)$, $x=\omega\tau$ and
$\arg f=-2\arg(1-ix)=+2\arctan x$. Eq. (6) must therefore give
$\psi=2\arctan x$, equivalently
$\mathrm{PV}\!\int_0^\infty\ln(1+u^2)/(u^2-x^2)\,du=(\pi/x)\arctan x$. Checked
numerically below (§III.1) to $10^{-11}$.

### I.2 Inverse transform, Eq. (7) sign, normalization and units

$\rho(t)=\frac1{2\pi}\int_{-\infty}^\infty f(\omega)e^{-i\omega t}d\omega$; with the
Hermitian symmetry,

$$
\boxed{\rho(t)=\frac1\pi\int_0^\infty\lvert f(\omega)\rvert\cos\bigl(\psi(\omega)-\omega t\bigr)\,d\omega .}
$$

Leissner's printed Eq. (7), $\int_0^\infty\lvert f\rvert\cos(\psi+\omega z/c)\,d\omega$,
equals $\pi\rho(-t)$ with $t=z/c$: a time reversal, and without the $1/\pi$
(the printed figure is in arbitrary units). The ledger's statement that the
printed $+\omega z/c$ is $\rho(-t)$ in the Eq. (2) convention is confirmed
(numerically, §III.7, to $1.7\times10^{-5}$ of the peak).

*Normalization.* $\int\rho\,dt=f(0)=\lvert f(0)\rvert$ because $\psi(0)=0$
($\psi$ odd): $\int_{-\infty}^\infty\cos(\psi-\omega t)dt=2\pi\delta(\omega)\cos\psi(0)$
and half the delta lies on $[0,\infty)$, giving $\frac1\pi\cdot\pi\lvert f(0)\rvert$.

*Units.* With $E=\hbar\omega$ in eV, $t$ in fs and
$\hbar=6.582119569\times10^{-16}$ eV s $=0.6582119569$ eV fs (CODATA 2018,
exact digits beyond those not physical),

$$
\rho(t)\,[\mathrm{fs}^{-1}]=\frac1{\pi\hbar}\int_0^\infty\lvert f(E)\rvert\cos\bigl(\psi(E)-Et/\hbar\bigr)\,dE .
$$

Eq. (6) is invariant under $u\to\lambda u$, $\omega\to\lambda\omega$, so $\psi$
may be evaluated directly on the energy grid.

### I.3 Exact segment integral of a linear $g$

On $[x_a,x_b]$ let $g(x)=\alpha+\beta x$. Partial fractions,
$\frac1{x^2-\omega^2}=\frac1{2\omega}\bigl(\frac1{x-\omega}-\frac1{x+\omega}\bigr)$,
and $g(x)=g(\omega)+\beta(x-\omega)=g(-\omega)+\beta(x+\omega)$ give

$$
\frac{g(x)}{x^2-\omega^2}=\frac1{2\omega}\Bigl[\frac{g(\omega)}{x-\omega}-\frac{g(-\omega)}{x+\omega}\Bigr]
\;\Rightarrow\;
\int_{x_a}^{x_b}\frac{g\,dx}{x^2-\omega^2}=\frac1{2\omega}\Bigl[g(\omega)\ln\lvert x-\omega\rvert-g(-\omega)\ln(x+\omega)\Bigr]_{x_a}^{x_b},
$$

the $\beta$ terms cancelling, with $g(\pm\omega)$ the linear continuation of the
segment. For $g=\ln\lvert f\rvert-\ln\lvert f(\omega)\rvert$, $\ln\lvert f\rvert$
piecewise linear and $\omega$ a node: on a segment with an endpoint at
$\omega$, the segment's linear function equals $\ln\lvert f(\omega)\rvert$ at
$\omega$, so $g(\omega)=0$ exactly and the $\ln\lvert x-\omega\rvert\to-\infty$
endpoint carries a zero coefficient (the integrand $\to\beta/(2\omega)$ is
bounded). The $0\cdot\ln0$ must be taken as $0$; there is no log singularity
and no PV. On non-touching segments all logarithms are finite. The
$\psi$ contribution of a segment is $-(2\omega/\pi)$ times the bracket over
$2\omega$, i.e. $-\frac1\pi[\dots]$. At $\omega=0$, $\psi(0)=0$ by oddness.

### I.4 Power-law tail

For $x>X$ with $\lvert f\rvert\propto x^{-p}$: $g(x)=g(X)-p\ln(x/X)$,
$g(X)=\ln\lvert f(X)\rvert-\ln\lvert f(\omega)\rvert$, $r=\omega/X\le1$.

$$
g(X)\int_X^\infty\frac{dx}{x^2-\omega^2}=\frac{g(X)}{2\omega}\ln\frac{X+\omega}{X-\omega}=\frac{g(X)}{\omega}\operatorname{artanh}r .
$$

With $x=X/s$:
$\int_X^\infty\frac{\ln(x/X)}{x^2-\omega^2}dx=\frac1X\int_0^1\frac{-\ln s}{1-r^2s^2}ds=\frac1X\sum_k\frac{r^{2k}}{(2k+1)^2}=\frac{\chi_2(r)}{\omega}$,
using $\int_0^1-s^{2k}\ln s\,ds=(2k+1)^{-2}$. Hence

$$
T(\omega)=\frac1\omega\bigl[g(X)\operatorname{artanh}r-p\,\chi_2(r)\bigr],\qquad
\Delta\psi_{\rm tail}=-\frac{2\omega}{\pi}T=-\frac2\pi\bigl[g(X)\operatorname{artanh}r-p\,\chi_2(r)\bigr].
$$

$\chi_2(r)=\frac12[\mathrm{Li}_2(r)-\mathrm{Li}_2(-r)]$, $\chi_2(1)=\pi^2/8$. At
$r=1$ ($\omega=X$) $g(X)=0$ exactly and the integrand
$-p\ln(x/X)/(x^2-X^2)$ is integrable, so the $\operatorname{artanh}$ term is
$0$, not $0\cdot\infty$; the tail is $+(2/\pi)p\pi^2/8=p\pi/4$.

### I.5 Ambiguities

- **Time reversal:** $\rho(-t)\leftrightarrow f^*(\omega)$, same $\lvert f\rvert$.
- **Arrival time:** $\rho(t-t_0)\leftrightarrow e^{i\omega t_0}f$; the factor has
  unit modulus and no zeros, so Eq. (6) drops it (the reconstruction starts
  near $t=0$).
- **Zero flipping:** a zero $\omega_0$ of $f$ in the upper half plane can be
  replaced by $\omega_0^*$ via the Blaschke factor
  $(\omega-\omega_0^*)/(\omega-\omega_0)$, of unit modulus on the real axis.
  Eq. (6) returns the version with all zeros in the lower half plane.
  *Two-bunch example:* $\rho=[h(t)+a\,h(t-T)]/(1+a)$ gives
  $f=H(\omega)(1+ae^{i\omega T})/(1+a)$ with zeros at
  $e^{i\omega T}=-1/a$, i.e. $\omega_k=[(2k+1)\pi+i\ln a]/T$. For $a>1$,
  $\operatorname{Im}\omega_k=\ln a/T>0$: upper half plane, **non-minimal**.
  On the real axis $\lvert1+ae^{i\omega T}\rvert=\lvert a+e^{i\omega T}\rvert$, and
  $a+e^{i\omega T}$ has zeros at $\operatorname{Im}\omega=-\ln a/T<0$ and is
  bounded ($\to a$) as $\operatorname{Im}\omega\to\infty$, so
  $H(a+e^{i\omega T})/(1+a)$ is the minimal-phase partner, with profile
  $[a\,h(t)+h(t-T)]/(1+a)$: leading bunch larger, i.e. $a\to1/a$ after
  renormalizing. The modulation minima on the real axis sit at
  $\omega=(2k+1)\pi/T$, $E=(2k+1)\pi\hbar/T$.
- **Common scale:** $\ln\lvert cf\rvert-\ln\lvert cf(\omega)\rvert=\ln\lvert f\rvert-\ln\lvert f(\omega)\rvert$,
  so $\psi$ is exactly invariant under a common factor on the whole extended
  $\lvert f\rvert$ (including $g(X)$ in the tail; $p$ is unchanged). A charge
  error $N\to N(1+\delta)$ scales the derived $\lvert f\rvert^2$ by $(1+\delta)^{-2}$
  on every measured node. With the `"constant"` low continuation, the
  extended $\lvert f\rvert$ is scaled as a whole, so $\psi$ is invariant and
  $\rho$ only scales. With `"gaussian-to-unity"`, $\ln\lvert f\rvert=\ln\lvert f(E_{\rm lo})\rvert(E/E_{\rm lo})^2$
  is pinned to $\lvert f(0)\rvert=1$, so the scale is **not** common and
  enters $\psi$ (and the shape) only through this low continuation. Confirmed.

### I.6 Paley–Wiener and finiteness

A one-sided $\rho\in L^2$ requires
$\int_{-\infty}^\infty\lvert\ln\lvert f\rvert\rvert/(1+\omega^2)d\omega<\infty$.
A pure Gaussian $\lvert f\rvert=e^{-\sigma^2\omega^2/2}$ violates it: in Eq. (6)
$g(u)/(u^2-\omega^2)=-\sigma^2/2$ for all $u$, and the integral diverges
linearly in the cutoff $L$ ($\psi_L=\sigma^2\omega L/\pi$; §III.8). With a finite
band, a bounded low continuation and $\lvert f\rvert\propto E^{-p}$ above $X$,
the tail integrand behaves as $-p\ln x/x^2$, integrable, and the low part is
bounded on a finite interval, so $\psi$ is finite for every $p>0$; the closed
form of §I.4 is its value.

### I.7 Declared input uncertainty

$\lvert f\rvert^2=S/[N(N-1)S_1F_TR]\approx S/(N^2S_1F_TR)$. Charge:
$\lvert f\rvert^2\to\lvert f\rvert^2/q^2$, $q=1+\delta_N$, one $q$ for all nodes.
Transverse/response: $\lvert f\rvert^2\to\lvert f\rvert^2/(1+\epsilon)\approx\lvert f\rvert^2(1-\epsilon)$,
one $\epsilon$ (times a per-node relative RMS) per draw; to first order and for
a symmetric distribution, multiplying by $(1+\epsilon)$ is equivalent.

## Part II — Comparison with the implementation

### II.1 Term-by-term

| Item | Independent (Part I) | Code | Result |
| --- | --- | --- | --- |
| Eq. (6) sign/convention | $-\frac{2\omega}{\pi}\int g/(u^2-\omega^2)$, $e^{+i\omega t}$ | `values = -np.sum(seg, axis=1) / np.pi + tail` with `seg` = bracket of §I.3 | match |
| Segment $g(\omega)$, $g(-\omega)$ | linear continuation of segment | `plus = ga + beta * (w - a)`, `minus = ga + beta * (-w - a)`, `ga = lf[:-1] - level` | match |
| Touching segments | $g(\omega)=0$, $0\cdot\ln0\to0$ | `touches` mask (`a == w` or `b == w`) → `plus = 0`; `_safe_log(0) = 0` | match (exact float equality valid: `w` is drawn from the same node array) |
| $\omega=0$ | $\psi(0)=0$ | `np.where(w > 0, values, 0.0)` (masks `log(0)` artefacts at the first node) | match |
| Tail | $-\frac2\pi[g(X)\operatorname{artanh}r-p\chi_2(r)]$ | `tail = -(2/pi) * (log_term - p * _chi2(r))` | match |
| $r=1$ | artanh term $=0$ | `np.where(r < 1, g_top * arctanh(...), 0)` | match |
| $\chi_2$ | $\frac12[\mathrm{Li}_2(r)-\mathrm{Li}_2(-r)]$ | `0.5 * (spence(1 - r) - spence(1 + r))`; scipy `spence(z)` $=\mathrm{Li}_2(1-z)$ | match |
| Low continuation | gaussian: $\ln\lvert f_{\rm lo}\rvert(E/E_{\rm lo})^2$; constant | `_extend`: `log_f[0] * (e_low / lo) ** 2` from $E=0$; `np.full_like` | match; refuses $\lvert f(E_{\rm lo})\rvert>1$ |
| High nodes | $\ln\lvert f_{\rm hi}\rvert-p\ln(E/E_{\rm hi})$, same $p$ as tail | `l_high = log_f[-1] - p * np.log(e_high / hi)` | match |
| Inverse transform | $\frac1{\pi\hbar}\int\lvert f\rvert\cos(\psi-Et/\hbar)dE$ | `_synthesize`: `np.real(sum(...)) / (np.pi * _HBAR_EV_FS)`, phase `psi - E t / hbar` | match |
| Segment synthesis | $h\,e^{i\theta_a}[m_a\Phi_a(z)+m_b\Phi_b(z)]$, $z=i(\Delta\psi-ht/\hbar)$, $\Phi_a=\int_0^1(1-v)e^{zv}dv=(e^z-1-z)/z^2$, $\Phi_b=\int_0^1ve^{zv}dv=(e^z(z-1)+1)/z^2$ | `_phi` closed forms for $\lvert z\rvert\ge0.5$; series $\sum z^n/(n+2)!$ and $\sum(n+1)z^n/(n+2)!$, 10 terms, for $\lvert z\rvert<0.5$ | match; truncation $<0.5^{10}/12!\approx2\times10^{-12}$ |
| $\hbar$ | 0.6582119569 eV fs | `_HBAR_EV_FS = 0.6582119569509067` | match (CODATA 2018 $\hbar$ = 6.582119569…×10⁻¹⁶ eV s) |
| Power fractions | $\int_X^\infty m_X^2(x/X)^{-2p}=m_X^2X/(2p-1)$; $\int_X^\infty m_X(x/X)^{-p}=m_XX/(p-1)$ | `_power_fractions` | match (∞ when $2p\le1$ / $p\le1$) |
| Charge | $\lvert f\rvert^2/q^2$, one draw | `out / max(charge, 1e-6) ** 2`, one `standard_normal()` | match |
| Transverse/response | one draw per sample, fully correlated | `out * (1 + rel * rng.standard_normal())` (scalar draw broadcast) | match to first order (see F3) |
| Noise | independent per node | `sd * rng.standard_normal(f2.shape)` | match |
| Ambiguity flags | not unique; time reversal, linear phase, zero flip, missing band | `AMBIGUITIES`, `is_unique = False`, `time_reversed()` returns $(-t,\rho(-t))$ | match |

Chunking (`chunk` in `minimal_phase`, `_synthesize`) only slices the
evaluation nodes/times; result independent of chunk size (§III.2: chunk 7 vs
256 identical to 0.0).

### II.2 Findings (no code change made)

- **F1 (low; `src/pyrite/detectors/optical_phase_retrieval.py:238-245`, ledger
  `Checks`).** `_spectral_minima` compares each minimum with `depth` times the
  smaller of the *global* maxima on either side. Under a decaying envelope
  the first modulation minimum is not reported: in the ledger's own two-bunch
  case ($\tau=50$ fs, $T=300$ fs, $a=2$) the returned minima are
  0.020799, 0.034622, 0.048313, 0.062135 eV, while the first expected one,
  $\pi\hbar/T=0.006893$ eV, is missing ($\lvert f\rvert=0.262$ there vs
  $0.5\times0.477=0.238$). The test only requires $\ge3$ minima, so it passes.
  The ledger `Checks` text "spectral minima at $(2k+1)\pi\hbar/T$ reported" and
  the docstring "lower flanking maximum" overstate it. Fix: either describe it
  as a heuristic that can miss minima under a steep envelope, or compare with
  the adjacent local maxima (e.g. their geometric mean) instead of the global
  side maxima. This is a diagnostic flag, not the reconstructed profile.
- **F2 (info; `optical_phase_retrieval.py:122-137, 206-218`).** Eq. (6) is
  exact for piecewise-linear $\ln\lvert f\rvert$, but `_synthesize` interpolates
  $\lvert f\rvert$ (not $\ln\lvert f\rvert$) linearly. The mismatch is second order
  in node spacing (the gamma-2 profile agrees with the exact one to
  $3.8\times10^{-4}$ of the peak, the same as an independent trapezoid
  transform). The module docstring states both choices, so this is a
  documentation-consistent approximation; no fix needed.
- **F3 (info; `optical_phase_retrieval.py:402-405`).** Transverse/response draws
  multiply $\lvert f\rvert^2$ by $(1+\epsilon)$, whereas dividing by an
  uncertain $F_TR$ gives $1/(1+\epsilon)$. Equal to first order and symmetric
  for Gaussian $\epsilon$; at second order the mean is biased by
  $\sim\epsilon^2$. Acceptable for the declared few-percent use; could be noted
  in the docstring.
- **F4 (info).** The high continuation starts from the *instantaneous*
  $\lvert f(X)\rvert$. For an oscillating true spectrum the in-band phase error is
  $\approx-(2/\pi)\delta\operatorname{artanh}(\omega/X)$ with
  $\delta=\ln\lvert f(X)\rvert$ minus its local mean level (§III.6: 0.072 rad at
  $X=9.2$ in the worst phase, falling to $6.9\times10^{-3}$ after removing that
  term). This is an instance of the declared missing-band ambiguity, already
  recorded via `MissingBandModel`; no change requested.

## Part III — Independent numerical checks

Scripts `scratch/val_phase_retrieval.py`, `_2.py`, `_3.py` (own quadratures;
`scipy.integrate.quad` per segment, analytic phases). All runs capped at
1 GiB, rc 0.

1. **Gamma-2 Eq. (6), full analytic $\lvert f\rvert=1/(1+x^2)$, own quadrature vs
   $2\arctan x$:** differences $-1.7\times10^{-16}$, $-1.4\times10^{-15}$,
   $1.1\times10^{-15}$, $-5.6\times10^{-12}$, $-7.5\times10^{-12}$ at
   $x=0.1,0.5,1,3,10$. `minimal_phase` on nodes $[0,1]$ (401 linear) +
   $[1,200]$ (3000 log), $p=2$: max $\lvert\psi-2\arctan x\rvert=2.7\times10^{-6}$
   for $x\le50$ (piecewise-linear $\ln\lvert f\rvert$ discretization);
   $5.8\times10^{-6}$ at $x=100$.
2. **Arbitrary piecewise-linear $\ln\lvert f\rvert$** (41 random nodes on
   $[0,5]$, $\ln\lvert f\rvert=-0.3x^{1.5}+0.04x\sin3x$, $p=1.7$) vs own
   segment-wise quadrature plus quad tail: max difference
   $5.7\times10^{-12}$ rad over all nodes; $\psi(0)=0$; chunk 7 vs 256
   difference 0.0.
3. **Common scale:** $\max\lvert\psi(\ln\lvert f\rvert-0.37)-\psi\rvert=8.9\times10^{-16}$.
4. **Tail closed form, $X=1$, $p=1.5$, $g(X)=-0.8$:** quad vs closed
   $-2.340895220796$ vs $-2.340895220796$ at $r=0.3$;
   $-3.018537798149$ vs $-3.018537798148$ at $r=0.9$; $r=1$, $g(X)=0$:
   $-1.850550825204$ both ($=-p\pi^2/8$). `_chi2` equals the series at
   $r=0.3,0.9,0.999$ to $10^{-12}$; $\chi_2(1)=1.2337005501=\pi^2/8$.
5. **`_phi` vs complex quad** at nine $z$ including $10^{-8}i$, $0.49i$,
   $0.5i$, $0.51i$, $50i$, $1.5-2i$: max error $1.9\times10^{-11}$.
6. **Two-bunch** ($\tau=1$, $T=6$, $a=2$): zeros of $1+ae^{i\omega T}$ at
   $\operatorname{Im}\omega=0.11552453=\ln2/6$ (residual $<2.4\times10^{-15}$);
   $\max\bigl\lvert\lvert f_{\rm true}\rvert-\lvert f_{\rm min}\rvert\bigr\rvert=3.3\times10^{-16}$.
   `minimal_phase` from $\lvert f_{\rm true}\rvert$ vs $\arg f_{\rm min}$
   ($\omega<5$, uniform $d\omega=0.005$), with $X$ where the tail level
   matches the mean ($\lvert2+e^{iXT}\rvert=2$): $1.8\times10^{-3}$,
   $6.2\times10^{-4}$, $2.7\times10^{-4}$ at $X=8.68,17.06,33.81$ (converges
   as $\sim1/X$); vs $\arg f_{\rm true}$: $3.13$ rad (wrong half plane, as
   expected). Profile level (test case $\tau=50$ fs, $T=300$ fs):
   max $\lvert\rho_{\rm rec}-\rho_{\rm min}\rvert/{\rm peak}=0.0149$,
   $\lvert\rho_{\rm rec}-\rho_{\rm true}\rvert/{\rm peak}=0.496$, window area
   0.99994.
7. **Profile, gamma-2, $\tau=100$ fs** (test grid): max
   $\lvert\rho_{\rm code}-\rho_{\rm exact}\rvert/{\rm peak}=3.83\times10^{-4}$;
   own trapezoid inverse transform $3.75\times10^{-4}$; window area
   0.999889; RMS 141.382 fs vs $\sqrt2\tau=141.421$ fs; truncated magnitude
   fraction $7.96\times10^{-4}$; extrapolated power fraction 0.0127; code
   peak at $t=98.2$ fs ($\rho(t)$, peak $\tau$). Printed Eq. (7) with
   $+\omega t$ equals $\rho(-t)$ to $1.7\times10^{-5}$ of the peak.
8. **Paley–Wiener:** pure Gaussian, $\omega=1$: partial $\psi_L=3.1831$,
   $31.831$, $318.310$ for $L=10,100,1000$ ($=L/\pi$, linear divergence).
   Same Gaussian on band $[0.1,3]$ with the default continuation, $p=3$:
   $\psi\in[0,5.364]$, finite.
9. **`FormFactorUncertainty.sample`** (20000 draws): charge-only ratio
   sample$/\lvert f\rvert^2$ identical across nodes to $4.4\times10^{-16}$;
   implied $q$ mean 0.99930, std 0.04979 (declared 0.05). Per-node response
   RMS 0…0.1: correlation between nodes 10 and 29 $=1.0000$, std at node 29
   0.1006, node 0 std 0.

**Repository tests:**
`systemd-run … uv run pyrite-dev test tests/detectors/test_optical_phase_retrieval.py -p no:xdist`
→ 13 passed, rc 0.

## Verdict

- **Claim**: `optical-phase-retrieval` —
  `src/pyrite/detectors/optical_phase_retrieval.py::{minimal_phase, reconstruct_minimal_phase_profile, MissingBandModel, propagate_reconstruction_uncertainty, FormFactorUncertainty}` —
  Leissner et al., PAC 1999, Eqs. (2), (6), (7)
- **Filters**: units pass (eV, fs, $\hbar$ in eV fs, $\rho$ in fs$^{-1}$,
  area $\lvert f(0)\rvert$); limits pass ($\psi(0)=0$, $r=1$ tail, gamma-2
  closed form, common-scale invariance, Paley–Wiener finiteness);
  signs/conventions pass (Eq. (6) sign in the $e^{+i\omega t}$ convention,
  upper-half-plane analyticity, printed Eq. (7) $=\rho(-t)$)
- **Re-derivation**: `matches` — segment integral, touching-segment masking,
  power-law tail with $\chi_2$, Filon synthesis and normalization agree
  symbolically and to $\le6\times10^{-12}$ rad against independent
  quadrature; two-bunch $a>1$ returned as its $a\to1/a$ minimal partner.
- **Verdict**: `rederived`
- **Write-up**: `docs/validation/detectors/optical-phase-retrieval.md`
- **Suggested ledger change**: `Status: rederived`; append to `Checks`
  "Fresh-context derivation 2026-11-10 at `aca5952a8`
  (this write-up): Eq. (6) sign, segment and
  power-law-tail closed forms, inverse transform and charge/systematic
  sampling match; independent quadrature to 6e-12 rad". Reword the minima
  clause to "deep in-band minima (heuristic; the first minimum under a
  steep envelope can be missed)" or fix F1. Never `signed-off`.

## Owner follow-up

Applied after the verdict, at the next commit on top of `aca5952a8`:

- **F1 fixed.** `_spectral_minima` now compares each local minimum against
  the geometric mean of its adjacent peaks, reached by climbing
  monotonically left and right. It is still documented as a heuristic. In the
  $\tau=50$ fs, $T=300$ fs, $a=2$ test the first minimum $\pi\hbar/T$ is now
  reported. The test asserts that the first minimum is present and that no
  order $k$ is skipped.
- **F3 applied.** Transverse and response draws now divide $\lvert f\rvert^2$
  by $1+\epsilon\,\delta$, because the uncertain factor divides the measured
  spectrum. This removes the $O(\epsilon^2)$ bias.
- F2 and F4 are informational and need no change.

## 2026-11-14 — Published anchor: Leissner thesis Sec. 7.1 (parabolic continuation, Fig. 7.5)

Fresh context, uncommitted working tree on `issues-616-617-units-uncertainty`
(HEAD `88994e94a`). Scope: the new `MissingBandModel(low="parabolic-intensity")`
branch of `_extend`, `MinimalPhaseReconstruction.lobe_rms_duration_fs`
(`_lobe_rms`), metadata `input_intensity_divided_by`, the digitized
`tests/data/optical/leissner-1998-fig7-5.json`, and the proposed anchor
`tests/detectors/test_optical_phase_retrieval_leissner.py`. The 2026-11-10
verdict on the Eq. (6) map, synthesis and uncertainty sampling is not
revisited.

**Source:** B. Leissner, *Strahluntersuchungen am TESLA Linearbeschleuniger*,
Diplomarbeit RWTH Aachen, PITHA 99/4 (1998) (Zotero `MDUGLCM5`), read as page
images: Sec. 6.3 (thesis pp. 58–65, PDF pp. 60–67), Sec. 6.4.2 (thesis p. 67),
Sec. 7.1 (thesis pp. 69–74, PDF pp. 71–76). The derivation and the
independent reconstruction and digitization below were completed before the
implementation diff, the test and the JSON were opened.

### A.1 Source conventions

Eq. (6.9) is $\tilde\rho(\omega)=\frac1{2\pi}\int\rho(t)e^{i\omega t}dt$, the
same $e^{+i\omega t}$ convention as PAC Eq. (2). Eq. (6.20) is
$\psi_m(\omega)=-\frac{2\omega}{\pi}\int_0^\infty\frac{\ln[\xi(x)/\xi(\omega)]}{x^2-\omega^2}dx$
with $\xi=\lvert\tilde\rho\rvert$, identical to the ledger's Eq. (6). The
printed inverse, Eq. (6.22), is
$\rho(t-d)=\int_0^\infty\tilde\rho\cos[\psi_m+\omega d+\omega t]\,d\omega$. In
the Eq. (6.9) convention the causal inverse is
$\rho(t)=\frac1\pi\int_0^\infty\xi\cos(\psi_m-\omega t)\,d\omega$, so the
printed $+\omega t$ is the time reversal $\rho(-t)$. This is the same sign
issue the 2026-11-10 review found in PAC Eq. (7). §A.4 shows that Fig. 7.5
plots the causal $\rho(t)$, not the printed form.

**Table 7.1** fixes the spectral convention. Assume
$I=\lvert f\rvert^2=e^{-\omega^2\sigma_t^2}$. Then $I$ is a Gaussian in $f$
with $\sigma_f=1/(2\sqrt2\,\pi\sigma_t)=112.5\ {\rm GHz}/\sigma_t[\rm ps]$.
This gives 37.5, 45.4, 112.5, 150, 225 and 450 GHz for
$\sigma_t=3,2.48,1,0.75,0.5,0.25$ ps; the table lists 38, 45, 113, 150, 225
and 450. The amplitude-width reading $1/(2\pi\sigma_t)$ (159 GHz at 1 ps) is
excluded. The table's 60 and 75 GHz entries for 1.85 and 1.47 ps are 1–2 %
from 60.8 and 76.6 GHz, so they are rounded or were computed for 1.875 and
1.5 ps (informational).

### A.2 Parabolic join, Eqs. (6.10)–(6.13)

Expanding Eq. (6.9) for a normalized $\rho$ gives
$2\pi\tilde\rho=1+i\langle t\rangle\omega-\tfrac12\langle t^2\rangle\omega^2+O(\omega^3)$.
Hence

$$
(2\pi)^2\lvert\tilde\rho\rvert^2=1-(\langle t^2\rangle-\langle t\rangle^2)\,\omega^2+O(\omega^4)
=1-\sigma_t^2\omega^2+O(\omega^4).
$$

Eq. (6.12) is therefore correct in $\omega$. Two source typos do not affect
it: Eq. (6.10) omits the $\tfrac12$ on $\langle t^2\rangle\omega^2$, and the
prefactor of Eq. (6.12) should be $(2\pi)^{-2}$, not $(2\pi)^{-1}$. Both
disappear on normalization. Matching
$I(\omega)=A^2-B\omega^2$ to the measured $I$ in value and slope at $\omega_0$
(Eq. 6.13), with $I_0$ and $I_0'$ the measured intensity and slope there,
gives

$$
B=-\frac{I'_0}{2\omega_0},\qquad A^2=I_0+B\omega_0^2=I_0-\tfrac12\omega_0 I'_0 .
$$

Normalizing to $\lvert f(0)\rvert=1$ divides the whole band by $A^2$. Below
$\omega_0$ this gives

$$
\lvert f\rvert^2=1-\frac{B}{A^2}\,\omega^2 ,\qquad \frac{B}{A^2}\leftrightarrow\sigma_t^2\ \text{(Eq. 6.12)} .
$$

The construction needs $I'_0<0$, the falling flank. Then $B>0$ and
$1-B\omega^2/A^2\ge I_0/A^2>0$ on $[0,\omega_0]$, so the logarithm is finite.
The thesis places $\omega_0$ at "the point on the flank with the largest
slope", because that gives the longest bunch. The join is in energy rather
than $\omega$ only through the common factor $\hbar$, which cancels in $B/A^2$
after the change of variable.

**Limiting case.** Suppose an unfiltered Gaussian has
$I=e^{-\omega^2\sigma^2}$ and the join is at its steepest point,
$\omega_0=1/(\sqrt2\sigma)$, which is 37.5 GHz for 3 ps. Then
$I_0=e^{-1/2}$ and $I'_0=-2\sigma^2\omega_0e^{-1/2}$, so
$B=\sigma^2e^{-1/2}$, $A^2=\tfrac32e^{-1/2}$ and

$$
\frac{B}{A^2}=\tfrac23\,\sigma^2,\qquad \sqrt{B/A^2}=0.816\,\sigma
$$

(numerically confirmed: 6.000 ps$^2$ for $\sigma=3$ ps). As $\omega_0\to0$,
$B/A^2\to\sigma^2$, which is Eq. (6.12). At a finite join the parabola
recovers $\sigma_t$ only to $O(\omega_0^2\sigma^2)$, because Eq. (6.12) is
only an $O(\omega^2)$ statement.

### A.3 Filter and cutoff, Eqs. (6.28)–(6.29) and (7.1)

With $\epsilon(\omega,b)=1-e^{-b^2\omega^2}$, the condition
$\epsilon(2\pi f_{Ab})=e^{-1/2}$ gives
$b^2\omega^2=-\ln(1-e^{-1/2})$, so
$f_{Ab}=\sqrt{-\ln(1-e^{-1/2})}/(2\pi b)=0.153711/b[\rm ps]$ THz, which is
Eq. (6.29). The 154, 77 and 51 GHz cutoffs are $b=0.998$, 1.996 and
3.014 ps, i.e. $b=1,2,3$ ps rounded. The filtered spectrum
$I_{\rm Det}=\epsilon\,e^{-\omega^2\sigma^2}$ peaks at
$\omega_*^2=\ln(1+b^2/\sigma^2)/b^2$, with peak value
$\frac{b^2}{b^2+\sigma^2}\big(\frac{\sigma^2}{b^2+\sigma^2}\big)^{\sigma^2/b^2}$.

### A.4 Independent checks (no `pyrite` import)

Scripts are under `scratch/v617/` in the worktree (not committed).

1. **Own digitization** (`digitize.py`, `extract.py`): `pdftocairo -svg` of
   PDF pages 73 and 74. Curves were taken from the raw stroked polylines and
   identified by colour and dash pattern; no curve commands are present.
   Axes were calibrated by a linear fit to every tick-mark path. In Fig. 7.5
   the fit residual is $\le4\times10^{-3}$ ps in $t$ and $2\times10^{-5}$ in
   ordinate. The repository JSON agrees with this digitization to
   $\le1.4\times10^{-3}$ of peak at equal $t$, with a vertex offset of at most
   7 fs. The JSON's rounded calibration (324.0 units per 5 ps against the
   tick-fit 323.87) accounts for the offset, which is negligible.
2. **Fig. 7.4 forward model.** The digitized peaks are 0.2191, 0.7598 and
   1.4123 (154, 77, 51 GHz), at 50.7, 46.9 and 43.0 GHz. The figure has a
   sample spacing of about 3.9 GHz. The model peaks are at 51.7, 48.3 and
   44.1 GHz. The model peak ratios relative to 154 GHz are 3.474 and 6.518
   with $b$ from the labels, and 3.472 and 6.453 with $b=1,2,3$ ps exactly.
   The digitized ratios are 3.468 and 6.446, so the thesis used
   $b=1,2,3$ ps. The arbitrary scale is a common 5.61–5.68.
3. **Own reconstruction** (`kk.py`). The test uses
   $I=\epsilon e^{-\omega^2(3\,\rm ps)^2}$ on a 400 001-point grid with these
   choices:
   - join at the steepest point of the falling flank in $I$;
   - band top at $10^{-5}$ of the peak;
   - amplitude tail $\propto\omega^{-3}$.

   Eq. (6.20) is evaluated by direct quadrature of the regular subtracted
   integrand (no PV needed: the numerator vanishes at $x=\omega$), plus a
   numerically integrated power-law tail to $10^4\times$ the grid end. The
   causal cosine inverse is applied to a free time shift, and the comparison
   window is $[-5,25]$ ps.

   | $f_{Ab}$ | join | band top | $A^2$ | lobe RMS (own) | lobe RMS (Fig. 7.5) | rms / max $\lvert\Delta\rvert$ (rms-optimal shift) |
   | --- | --- | --- | --- | --- | --- | --- |
   | 154 GHz | 78.0 GHz | 202 GHz | 0.05577 | 2.104 ps | 2.035 ps | 0.0173 / 0.058 |
   | 77 GHz | 73.0 GHz | 195 GHz | 0.19271 | 2.225 ps | 2.167 ps | 0.0130 / 0.043 |
   | 51 GHz | 66.9 GHz | 190 GHz | 0.35638 | 2.358 ps | 2.314 ps | 0.0092 / 0.030 |

4. **Implementation against the independent reconstruction.** Running the
   test's `_reconstruct` gives the same join to 0.01 GHz and $A^2$
   (`input_intensity_divided_by`/7) to $1\times10^{-4}$ relative. The in-band
   phase agrees to $\le3.7\times10^{-3}$ rad, which is the coarseness of my
   quadrature. The profile agrees at the same $t$ to $\le1.6\times10^{-4}$ of
   peak with no shift. The lobe RMS agrees to $2\times10^{-4}$ ps (impl
   2.1043, 2.2248 and 2.3586 ps). The test reports a max of 0.049, 0.037 and
   0.027 with its minimax shift on $[-4,24]$ ps; my rms-optimal shift gives
   0.058, 0.043 and 0.030.
5. **Time reversal.** $\rho(-t)$, the printed Eq. (6.22), with a free shift
   in $\pm40$ ps gives rms 0.045, 0.045 and 0.040 and max 0.112, 0.111 and
   0.093. The causal $\rho(t)$ gives 0.017, 0.013 and 0.009 and max 0.058,
   0.043 and 0.030. Fig. 7.5 is therefore the causal minimal-phase profile:
   it rises, then undershoots. The anchor pins orientation, not only shape.
6. **Sensitivity to unstated choices** (rms-optimal shift; lobe RMS / rms /
   max):

   | variation | 154 GHz | 77 GHz | 51 GHz |
   | --- | --- | --- | --- |
   | baseline | 2.104 / 0.017 / 0.058 | 2.225 / 0.013 / 0.043 | 2.358 / 0.009 / 0.030 |
   | join at steepest $\sqrt I$ | 2.054 / 0.021 / 0.049 | 2.171 / 0.017 / 0.040 | 2.300 / 0.014 / 0.034 |
   | join $\omega_0\times0.9$ / $\times1.1$ / $\times0.8$ | 2.108 / 2.075 / 2.081 | 2.229 / 2.193 / 2.200 | 2.364 / 2.323 / 2.333 |
   | band top $10^{-3}$ | 2.072 / 0.012 / 0.038 | 2.194 / 0.009 / 0.025 | 2.331 / 0.007 / 0.016 |
   | band top $10^{-4}$ | 2.089 / 0.015 / 0.051 | 2.210 / 0.011 / 0.036 | 2.346 / 0.007 / 0.023 |
   | band top $10^{-7}$ | 2.127 / 0.021 / 0.067 | 2.246 / 0.016 / 0.052 | — |
   | $p=2$ / $4$ / $6$ (max) | 0.056 / 0.059 / 0.063 | 0.042 / 0.045 / 0.049 | — |
   | $b=1$, $3$ ps exactly | 2.104 / 0.017 / 0.058 | — | 2.357 / 0.009 / 0.030 |

   The join at the steepest point of $I$ is within 0.3 % of the
   longest-lobe join in all three cases, consistent with the thesis
   rationale. The residual is controlled mainly by the unstated band top,
   which sets where the $\omega^{-p}$ tail starts and thus the onset
   sharpness. A smaller exponent works in the same direction. The test's
   $10^{-5}$ choice is *not* the best-fitting one: $10^{-3}$ halves the
   residual. The tolerances were therefore not obtained by tuning that
   choice. The residual is concentrated at the onset, where the thesis's
   discrete sum (6.21) and its unstated sampling and $\omega_{\max}$ also
   act, so attributing it to those unstated items is supported.
7. **Text RMS values** (`text_rms.py`, `sigz.py`, `fig73.py`). The Sec. 7.1
   text quotes 2.12, 2.48 and 2.75 ps for the three reconstructions, 2.82 ps
   for an unfiltered 3 ps Gaussian, and 1.85 and 1.47 ps for 2 and 1.5 ps
   Gaussians behind the 77 GHz filter. The thesis convention is to stop at
   the first negative value and ignore later positive tails.
   - On the digitized Fig. 7.5 the positive-lobe RMS is 2.035, 2.167 and
     2.314 ps.
   - Other conventions do not reproduce the triple either. The signed RMS on
     the window is 2.17, 2.13 and 2.15. The RMS of $\max(\rho,0)$ is 2.84,
     2.89 and 2.93. FWHM/2.355 gives 2.33, 2.51 and 2.63. A Gaussian fit to
     the lobe gives 2.25, 2.39 and 2.54, and to the whole curve 2.21, 2.35
     and 2.50.
   - Reading $\sigma_z=1$ mm, i.e. $\sigma_t=3.336$ ps, as the bunch length
     gives 2.33, 2.44 and 2.58, with the unfiltered case at 3.18 ps. This is
     rejected; Fig. 7.3's own reference Gaussian is
     $\sigma=0.899$ mm $=3.000$ ps.
   - The independent pipeline gives 2.858 ps for the unfiltered case (text
     2.82, 1.3 %). For 2 and 1.5 ps behind 77 GHz it gives 1.571 and
     1.236 ps (text 1.85 and 1.47, about 18 % higher).
   - **Fig. 7.3 does reproduce the 77 GHz text value.** Fig. 7.3 is the
     thesis's other plot of the 77 GHz reconstruction, against the original
     Gaussian. Its digitized curve has positive-lobe RMS **2.482 ps**, and
     its undershoot ($-0.135$) is hidden below the axis floor at 0.
   - Fig. 7.3 is not the Fig. 7.5 77 GHz curve. Its best fit to the
     independent 77 GHz reconstruction has rms 0.074 (max 0.14) over any
     join from 36 to 146 GHz. Neither the thesis's doubled-slope
     "Extr. 2" (2.403 ps, rms 0.047) nor a join at the peak reproduces it.

   The text values are therefore consistent with Fig. 7.3. They belong to an
   unstated reconstruction setting other than the one plotted in Fig. 7.5,
   and Fig. 7.3 and Fig. 7.5 show different 77 GHz reconstructions. The
   claimed source inconsistency holds for Fig. 7.5 but is mis-stated as
   "inconsistent with its figure"; see G2.

**Repository tests:**
`systemd-run … uv run pyrite-dev test tests/detectors/test_optical_phase_retrieval_leissner.py tests/detectors/test_optical_phase_retrieval.py`
→ 19 passed, rc 0.

### A.5 Implementation comparison

| Item | Independent | Code | Result |
| --- | --- | --- | --- |
| Slope at $E_{\rm lo}$ | $I'_0$ | `np.gradient(intensity, e[:3], edge_order=2)[0]` (second-order one-sided, nonuniform-safe) | match; $A^2$ to $10^{-4}$ |
| $B$, $A^2$ | $-I'_0/(2E_0)$, $I_0+BE_0^2$ | `curvature = -slope / (2 * lo)`, `a2 = intensity[0] + curvature * lo * lo` | match |
| Normalization | band $/A^2$, $\lvert f(0)\rvert=1$ | `log_f = log_f - 0.5 * np.log(a2)` | match |
| Low band | $\tfrac12\ln(1-BE^2/A^2)$ | `0.5 * np.log1p(-curvature * e_low**2 / a2)`; argument $>-1$ guaranteed | match |
| Falling flank | $I'_0<0$ required | `if not slope < 0: raise ValueError` (also catches NaN) | match |
| Lobe RMS | contiguous positive lobe at the peak, stop at first negative | `_lobe_rms`: first `rho < 0` index each side, `_rms` on the slice | match (thesis Sec. 7.1, p. 72) |
| `input_intensity_divided_by` | $A^2$ | `f2[0] / exp(2 * log_f[measured][0])` $=I_0/(I_0/A^2)$; 1 for other models | match |

### A.6 Findings

- **G1 (low; `tests/detectors/test_optical_phase_retrieval_leissner.py:93-96`).**
  The tolerances are a regression margin above the observed residual, not a
  source uncertainty: 0.06 against an observed max of 0.049, 0.025 against an
  observed rms of 0.020, and 4 % against observed lobe-RMS offsets of 3.4 %,
  2.7 % and 1.9 %. The digitization uncertainty ($\le1.4\times10^{-3}$) is
  negligible. The real spread comes from the unstated high-frequency
  continuation (§A.4.6: max 0.038–0.067, lobe RMS 2.05–2.13 ps at 154 GHz),
  and the tolerances lie inside it. The fixed choices are stated and were not
  tuned, so the residual and tolerances are justified as "agreement within
  the spread of the source's unstated choices". The ledger and test should
  say so explicitly. The 154 GHz max (0.049 at 0.06) and lobe RMS (3.4 % at
  4 %) leave little margin: the test is a tight regression guard and would
  catch a sign or factor error (time reversal fails by 2×). Optional
  hardening: pin $A^2$ for the 77 GHz case to the independent closed form
  0.19271 (relative $10^{-3}$) instead of the weak bound
  `> 7.0 * 0.03` at line 97.
- **G2 (low; wording; test module docstring lines 11-13,
  `tests/data/optical/leissner-1998-fig7-5.json` `text_values_note`,
  ledger Notes).** "Text RMS values (inconsistent with its figure)" is
  incomplete. The text values are not the lobe RMS of Fig. 7.5, but the
  77 GHz value 2.48 ps *is* the positive-lobe RMS of Fig. 7.3 (2.482 ps).
  Fig. 7.3 is a different 77 GHz reconstruction from the Fig. 7.5 curve, so
  the thesis is internally inconsistent between its two 77 GHz figures. The
  text values follow the Fig. 7.3 setting, which is not recoverable from the
  stated method. The same family's 1.85 and 1.47 ps are also about 18 % above
  the stated method. Suggested wording: "Sec. 7.1 text values (2.12/2.48/2.75,
  1.85/1.47 ps) and Fig. 7.3 (lobe RMS 2.48 ps) come from a different,
  unstated reconstruction than Fig. 7.5 and are not pinned."
  `test_text_rms_values_are_not_the_figure_rms` remains true as written.
- **G3 (info; `tests/detectors/test_optical_phase_retrieval_leissner.py:39`).**
  The cutoff labels are rounded. Fig. 7.4 shows the thesis used $b=1,2,3$ ps
  (51.24, not 51 GHz). The effect on the anchor is below 0.1 % (§A.4.6). No
  change needed; the JSON could record it.
- **G4 (info; `src/pyrite/detectors/optical_phase_retrieval.py:78`).** The
  docstring paraphrase "the join giving the longest bunch" is Leissner's
  *rationale*. His operational rule is "the point on the flank with the
  largest slope". The two differ by $\le0.3$ % in lobe RMS here, and the
  choice is the caller's. Consider "Leissner: the steepest point of the
  falling flank, which gives about the longest bunch".
- **G5 (info; source errata).** These do not affect the code: the missing
  $\tfrac12$ in Eq. (6.10); the $(2\pi)^{-1}$ prefactor in Eq. (6.12), which
  should be $(2\pi)^{-2}$; and the $+\omega t$ in Eq. (6.22), which is
  $\rho(-t)$ while Fig. 7.5 shows $\rho(t)$. The physics-page row for
  `"parabolic-intensity"` could add the limiting case:
  $B/A^2\to\sigma_t^2$ only as $E_{\rm lo}\to0$; here
  $\sqrt{B/A^2}=1.53$, 1.63 and 1.75 ps for a 3 ps bunch.

### A.7 Verdict (2026-11-14)

- **Claim**: `optical-phase-retrieval` —
  `src/pyrite/detectors/optical_phase_retrieval.py::{MissingBandModel, reconstruct_minimal_phase_profile}`
  (new `"parabolic-intensity"` branch of `_extend`; `_lobe_rms`) —
  Leissner, Diplomarbeit PITHA 99/4 (1998), Eqs. (6.12)–(6.13), (6.20),
  (6.22), (6.28)–(6.29), (7.1), Fig. 7.5
- **Filters**: units pass (eV/fs; $b$ in ps; $A^2$ in input units); limits
  pass (falling-flank requirement, $\lvert f(0)\rvert=1$, $B/A^2=\sigma_t^2$ as
  $\omega_0\to0$); signs/conventions pass (Table 7.1 gives
  $I=e^{-\omega^2\sigma_t^2}$; Fig. 7.5 is causal $\rho(t)$, and the
  time-reversed form is excluded by 2–4× in residual)
- **Re-derivation**: `matches` — join, normalization, filter/cutoff and
  lobe-RMS convention agree symbolically; independent quadrature agrees with
  the code to $1.6\times10^{-4}$ of peak and $2\times10^{-4}$ ps in lobe RMS;
  the repository digitization agrees with an independent one to
  $1.4\times10^{-3}$
- **Verdict**: `rederived` (new branch); published anchor **supported** for
  `anchored`, with the scope below
- **Write-up**: `docs/validation/detectors/optical-phase-retrieval.md`
- **Suggested ledger change**: `Status: anchored`; `Anchor:`
  `tests/detectors/test_optical_phase_retrieval_leissner.py` (Leissner 1998
  Fig. 7.5, vector-path digitization). Note the scope in `Notes` (below);
  correct the text-value wording per G2. Never `signed-off`.

**Pinned by the anchor:** the Eq. (6) minimal-phase map, the causal
orientation of the inverse transform, the `"parabolic-intensity"`
continuation (value-and-slope join, $A^2$ normalization) and the positive-lobe
RMS convention. These are checked against a published reconstruction of three
filtered 3 ps Gaussians to rms $\le0.020$ and max $\le0.049$ of peak, and
lobe RMS within 3.4 %.

**Not pinned:**
- the high-frequency continuation (band top, exponent) and the thesis's
  discrete sum (6.21), which set the onset and the residual;
- the `"gaussian-to-unity"` and `"constant"` continuations;
- `propagate_reconstruction_uncertainty` and `FormFactorUncertainty`, which
  have internal tests only;
- the Sec. 7.1 text values and Fig. 7.3;
- any measured-data reconstruction (PAC Fig. 6 and thesis Figs. 6.11–6.14,
  where the data are unpublished).
