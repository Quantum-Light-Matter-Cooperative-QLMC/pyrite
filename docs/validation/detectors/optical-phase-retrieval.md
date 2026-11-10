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
