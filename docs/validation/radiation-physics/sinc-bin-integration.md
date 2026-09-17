# Validation: sinc-bin-integration

Implementation-context derivation and measurements for the bin-mean line
quadrature (#116), followed by the independent fresh-context re-derivation the
methodology requires (see
[Independent re-derivation](#independent-re-derivation)). Human sign-off is
pending.

## Claim

For one incoherent PXR/CBS row with width parameter $a > 0$ and resonance
$E_{\mathrm{res}}$, the mean of
$S(E) = \operatorname{sinc}^2\!\left(a (E - E_{\mathrm{res}})/\pi\right)$ over
a bin $[\epsilon_i, \epsilon_{i+1}]$ is

```{math}
:label: eq-sbi-bin-mean

\bar S_i = \frac{\pi}{a\,(\epsilon_{i+1} - \epsilon_i)}
\left[F(x_{i+1}) - F(x_i)\right],
\qquad
x_i = \frac{a(\epsilon_i - E_{\mathrm{res}})}{\pi},
\qquad
F(x) = \frac{\operatorname{Si}(2\pi x)}{\pi} - x \operatorname{sinc}^2 x,
```

with NumPy's $\operatorname{sinc} u = \sin(\pi u)/(\pi u)$ and bins from the
midpoint convention of `characteristic.py::_energy_bin_edges_and_widths`.

## Derivation

Substitute $x = a(E - E_{\mathrm{res}})/\pi$, so $\mathrm{d}E = (\pi/a)\,\mathrm{d}x$
and $\int_{\epsilon_i}^{\epsilon_{i+1}} S\,\mathrm{d}E = (\pi/a)\int_{x_i}^{x_{i+1}}
\operatorname{sinc}^2 x\,\mathrm{d}x$. For $F$ as defined,

```{math}
:label: eq-sbi-derivative

F'(x)
= \frac{1}{\pi}\cdot\frac{\sin(2\pi x)}{x}
- \left[\frac{2\pi\sin(\pi x)\cos(\pi x)}{\pi^2 x} - \frac{\sin^2(\pi x)}{\pi^2 x^2}\right]
= \frac{\sin^2(\pi x)}{\pi^2 x^2}
= \operatorname{sinc}^2 x ,
```

using $\operatorname{Si}'(t) = \sin t / t$ and $x\operatorname{sinc}^2 x =
\sin^2(\pi x)/(\pi^2 x)$. $F(0) = 0$ and $F$ is odd. As $x \to \infty$,
$\operatorname{Si}(2\pi x) \to \pi/2$ and $x\operatorname{sinc}^2 x \to 0$, so
$F(\pm\infty) = \pm 1/2$ and the whole-line energy integral is $\pi/a$.

The alternative $\operatorname{Si}(2\pi x)/(2\pi) - \sin^2(\pi x)/(\pi^2 x)$
has derivative $\sin^2(\pi x)/(\pi^2x^2) - \sin(2\pi x)/(2\pi x)$ and tends to
$1/4$; at $x = 3$ it differs from quadrature by more than 0.2
(`test_the_handoff_formula_is_wrong_by_a_factor_two_in_si`).

### Evaluation form

With $s(x) = +1$ for $x \ge 0$ and $-1$ otherwise, define the tail complement
$R(x) = F(x) - s(x)/2$. For $u > 0$,
$R(u) = [\operatorname{Si}(2\pi u) - \pi/2]/\pi - u\operatorname{sinc}^2 u$ and
$R(-u) = -R(u)$; at the convention boundary, $R(0) = -1/2$. Then

```{math}
:label: eq-sbi-tail-form

F(x_{i+1}) - F(x_i) = R(x_{i+1}) - R(x_i) + \tfrac12\left[s(x_{i+1}) - s(x_i)\right],
```

and the bracket is $1$ exactly when $x_i < 0 \le x_{i+1}$ and $0$ otherwise.
With $\operatorname{Si}(t) = \pi/2 - f(t)\cos t - g(t)\sin t$,
$R(u) \sim -[\cos(2\pi u) + 2\sin^2(\pi u)]/(2\pi^2 u) = -1/(2\pi^2 u)$, so
$R$ carries the tail with relative, not absolute, accuracy. The window
truncation is $(\pi/a)\{[1/2 - F(x_N)] + [F(x_0) + 1/2]\}$, written the same way.

## Filters

| Filter | Result |
|---|---|
| Units | $x$ dimensionless ($a$ in eV⁻¹); $\pi/a$ in eV; $\bar S$ dimensionless, replacing the dimensionless node sample, so the spectrum keeps photons eV⁻¹ sr⁻¹ electron⁻¹ |
| Antiderivative | $F$ against adaptive quadrature at nine points in $[-4.1, 40]$: $\le 5\times10^{-14}$ |
| Mass identity | Summed bin masses plus reported truncation equal $\pi/a$ to $10^{-14}$ relative on 3 eV and 0.375 eV grids and on a #101 windowed grid; the 0.375 eV masses nest into the 3 eV ones |
| Width → 0 | 5×10⁻⁵ eV bins reproduce the node sample to $3\times10^{-10}$ |
| $a \to \infty$ | resonance-bin fraction 0.908, 0.99901, 0.999999 at $a = 10, 10^3, 10^6$ eV⁻¹ |
| Tail accuracy | $R(x)\cdot(-2\pi^2 x) - 1 \le 1/x$ at $x = 10^4, 10^7, 10^{10}$ |
| Sign | a bin mass is non-negative (up to rounding) because the integrand is; bins wholly on one side of the resonance never receive the whole-line term |

Anchor: `tests/montecarlo/test_sinc_bin_integration.py`,
`tests/montecarlo/test_sinc_bin_integration_cuda.py`,
`tests/energy-grid/test_line_grid_quadrature.py`.

## Si evaluators

The host path uses `scipy.special.sici` for $t < 48$ and the auxiliary-function
asymptotic series ($f t = \sum (-1)^k (2k)!/t^{2k}$,
$g t^2 = \sum (-1)^k (2k+1)!/t^{2k}$, 13 terms; the first omitted term at
$t = 48$ is below $10^{-17}$) at and above, returning $\operatorname{Si} - \pi/2$
without subtracting from $\pi/2$. The CUDA/ROCm C preamble
(`_bin_quadrature.py::SINCSQ_BIN_PREAMBLE`) uses the power series for
$t \le 4$, the modified-Lentz continued fraction of $E_1(it)$ for $4 < t < 48$,
and the same asymptotic series. Against 40-digit `mpmath` on 1,500 points in
$[0, 10^{12}]$ (compiled with `gcc -ffp-contract=off` on host): absolute error
$\le 10^{-15}$ for $t \le 4$, and error times $t$ $\le
2.1\times10^{-15}$ for $t > 4$ (i.e. relative to the $1/t$ envelope). The
$t \le 4$ bound is stated with margin over the $6.7\times10^{-16}$ measured
here, because it is libm- and compiler-dependent: an independent host
`gcc -O2 -ffp-contract=off` build measures $7.8\times10^{-16}$.

## Routes and measurements

The deterministic CPU mass, limit, policy, per-hkl, and batched-route checks are
anchored in the tests listed above. CUDA/CuPy kernel checks are present but need
a CUDA runner. The fixed-transport 300 keV yield and backend/precision comparison
is prepared in `checks/sinc_bin_integration.py`; that heavy measurement remains
to be run on the remote GPU box.


(independent-re-derivation)=

## Independent re-derivation

Fresh context, 2026-09-17. Inputs before any derivation: the ledger row, the
`_bin_quadrature.py` module docstring (source equation, intended quantity,
signature, units, assumptions, limiting cases), and the profile convention of
`finite-time-lineshape`. The derivation below was completed and checked against
independent references (`scipy.integrate.quad`, 40-digit `mpmath`) before the
implementation body was read; the source-to-code comparison follows it.

### Antiderivative

With the NumPy convention $\operatorname{sinc} u = \sin(\pi u)/(\pi u)$, the
incoherent finite-time profile is
$S(E) = \operatorname{sinc}^2\!\left[a(E - E_{\mathrm{res}})/\pi\right]
= \left[\sin\!\left(a\Delta E\right)/\left(a\Delta E\right)\right]^2$ with
$\Delta E = E - E_{\mathrm{res}}$, so $a$ carries eV$^{-1}$ and the profile is
dimensionless with unit peak. Claim: $F(x) = \operatorname{Si}(2\pi x)/\pi -
x\operatorname{sinc}^2 x$ satisfies $F' = \operatorname{sinc}^2$. Differentiate
the two pieces separately, using $\operatorname{Si}'(t) = \sin t/t$ and
$x \operatorname{sinc}^2 x = \sin^2(\pi x)/(\pi^2 x)$:

$$
\frac{\mathrm{d}}{\mathrm{d}x}\frac{\operatorname{Si}(2\pi x)}{\pi}
= \frac{1}{\pi}\cdot\frac{\sin(2\pi x)}{2\pi x}\cdot 2\pi
= \frac{\sin(2\pi x)}{\pi x},
$$

$$
\frac{\mathrm{d}}{\mathrm{d}x}\frac{\sin^2(\pi x)}{\pi^2 x}
= \frac{2\pi\sin(\pi x)\cos(\pi x)}{\pi^2 x} - \frac{\sin^2(\pi x)}{\pi^2 x^2}
= \frac{\sin(2\pi x)}{\pi x} - \frac{\sin^2(\pi x)}{\pi^2 x^2}.
$$

Subtracting, the $\sin(2\pi x)/(\pi x)$ terms cancel exactly and

$$
\boxed{\;F'(x) = \frac{\sin^2(\pi x)}{\pi^2 x^2} = \operatorname{sinc}^2 x\;}
$$

independently reproducing {eq}`eq-sbi-derivative`. $F(0) = 0$ because
$\operatorname{Si}(0) = 0$ and $x\operatorname{sinc}^2 x \to 0$; both terms are
odd, so $F$ is odd. As $x\to+\infty$, $\operatorname{Si}(2\pi x)\to\pi/2$ and
$x\operatorname{sinc}^2 x = \sin^2(\pi x)/(\pi^2 x)\to 0$, hence
$F(\pm\infty) = \pm 1/2$ and $\int_{-\infty}^{\infty}\operatorname{sinc}^2 = 1$.

### Bin mean and Jacobian

The substitution $x = a(E - E_{\mathrm{res}})/\pi$ has
$\mathrm{d}E = (\pi/a)\,\mathrm{d}x$, so the energy mass of bin
$[\epsilon_i,\epsilon_{i+1}]$ is $(\pi/a)\left[F(x_{i+1}) - F(x_i)\right]$ and
dividing by the bin width $\epsilon_{i+1}-\epsilon_i$ gives exactly
{eq}`eq-sbi-bin-mean`. Setting $x_i \to -\infty$, $x_{i+1}\to+\infty$ gives the
whole-line mass $\pi/a$ in eV, the normalization the spacing-independence claim
rests on. The bin mean is dimensionless and replaces a dimensionless node
sample, so the spectrum stays a density in photons eV$^{-1}$ sr$^{-1}$
electron$^{-1}$.

### Tail form and the sign bracket

Define $R(x) = F(x) - s(x)/2$ with $s(x) = +1$ for $x \ge 0$ and $-1$
otherwise. Since $s$ is constant on each open half-line,

$$
F(x_{i+1}) - F(x_i) = R(x_{i+1}) - R(x_i)
+ \tfrac12\left[s(x_{i+1}) - s(x_i)\right],
$$

identically, which is {eq}`eq-sbi-tail-form`. The bracket takes only the values
$0$ and $1$ under $x_i < x_{i+1}$: it is $1$ exactly when $s(x_i) = -1$ and
$s(x_{i+1}) = +1$, i.e. when $x_i < 0 \le x_{i+1}$. The boundary case is
consistent rather than arbitrary: with $s(0) = +1$ the point $x = 0$ belongs to
the upper branch, $R(0) = F(0) - 1/2 = -1/2$, and a bin whose *lower* edge sits
exactly on the resonance ($x_i = 0$) takes no whole-line term because
$R(x_i) = -1/2$ already carries it; a bin whose *upper* edge sits exactly on the
resonance ($x_{i+1} = 0$) does take the term. Both reproduce $F(x_{i+1})-F(x_i)$.
Any consistent choice of $s(0)$ gives the same bin masses provided $R$ and the
bracket use the same one — the convention is bookkeeping, not physics, but it
must not be split between the two.

For the far field write $\operatorname{Si}(t) = \pi/2 - f(t)\cos t - g(t)\sin t$
with $f \sim t^{-1}(1 - 2!/t^2 + 4!/t^4 - \cdots)$ and
$g \sim t^{-2}(1 - 3!/t^2 + 5!/t^4 - \cdots)$. At $t = 2\pi u$, $u>0$,

$$
R(u) = \frac{\operatorname{Si}(2\pi u) - \pi/2}{\pi} - \frac{\sin^2(\pi u)}{\pi^2 u}
= -\frac{\cos(2\pi u)}{2\pi^2 u} - \frac{1 - \cos(2\pi u)}{2\pi^2 u}
+ O(u^{-2})
= -\frac{1}{2\pi^2 u} + O(u^{-2}),
$$

using $f(2\pi u)/\pi = 1/(2\pi^2 u) + O(u^{-3})$ and
$\sin^2(\pi u) = [1-\cos(2\pi u)]/2$. The two oscillatory $\cos(2\pi u)$ terms
cancel, so the claimed $R(u)\sim-1/(2\pi^2 u)$ asymptote — and the statement
that $R$ carries the tail with relative accuracy — is confirmed. Numerically
$R(u)\big/\left[-1/(2\pi^2u)\right]$ is $0.99949$, $0.999995$, $1.0000000$ at
$u = 10, 10^2, 10^4$.

Keeping the $O(u^{-2})$ terms exactly, with $d = f t - 1$, $s_1 = \sin(\pi u)$,
$c_1 = \cos(\pi u)$, $\cos t = 1 - 2s_1^2$, $\sin t = 2 s_1 c_1$ and
$y = t^{-2}$:

$$
R(u) = -\frac{1 + d}{2\pi^2 u} + \frac{d\,s_1^2}{\pi^2 u}
- \frac{2\,(g t^2)\,y\,s_1 c_1}{\pi},
$$

an exact rearrangement in which the $O(1)$ part of $f$ never multiplies $s_1^2$,
so no $\sin^2$ cancellation survives.

### The #101 handoff formula

$G(u) = \operatorname{Si}(2\pi u)/(2\pi) - \sin^2(\pi u)/(\pi^2 u)$ differs from
$F$ only in the coefficient of the sine integral, $1/(2\pi)$ instead of $1/\pi$
— a factor of two short. Consequences, all confirmed numerically: its derivative
is $\operatorname{sinc}^2 u - \sin(2\pi u)/(2\pi u)$, not $\operatorname{sinc}^2 u$;
$G(\infty) = 1/4$, so the whole-line mass it implies is $\pi/(2a)$, half the
true value; and $G(1/2) = 0.09210$ against the true
$\int_0^{1/2}\operatorname{sinc}^2 = 0.38685$. The write-up's and the issue's
assessment of that handoff is correct.

### Source-to-code comparison

| Derived quantity | `_bin_quadrature.py` | Agreement |
|---|---|---|
| $S(E) = \operatorname{sinc}^2[a\Delta E/\pi]$ | `_kernels.py::_sincsq_lineshape` with `a_width = dnm * t_L / (2 HBARC_EV_ANG)` (eV$^{-1}$) | same profile the bin mean integrates |
| $F(x)$ | `sincsq_antiderivative` = `sincsq_tail(x) + where(x>=0, 0.5, -0.5)` | identical |
| $R(x)$, $s(0)=+1$ | `sincsq_tail`: `si_minus_half_pi(t)/pi - u*sinc(u)**2`, `where(x>=0, r, -r)` | identical, including $R(0) = -1/2$ |
| bracket $=1$ iff $x_i<0\le x_{i+1}$ | host `(x[:, :-1] < 0.0) & (x[:, 1:] >= 0.0)`; device `if (x_lo < 0.0 && x_hi >= 0.0)` | identical on both routes |
| $\bar S_i = (\pi/a)\,\Delta F/\text{width}$ | `_host_bin_mean`: `mass * (pi / a) * inv_width`; `pyrite_sincsq_bin_mean`: `mass * (PYRITE_PI / a) * inv_width` | identical |
| exact far-tail rearrangement above | `sincsq_tail` far branch and the preamble's two-trig form, with `d` summed from its first correction term | term-by-term identical |
| $f t = \sum_k (-1)^k (2k)!/t^{2k}$, $g t^2 = \sum_k (-1)^k (2k+1)!/t^{2k}$ | `_asymptotic_series` / preamble loops, 13 terms | identical; first omitted term $26!/48^{26} = 7.8\times10^{-18}$ |
| bin edges | `bin_axis` → `characteristic._energy_bin_edges_and_widths` (midpoints, reflected outer half-widths, low edge clamped at 0 eV), FP64 | matches the claim's stated bin convention |

The branch boundary is the same number, $t = 48$, in the doc, the module
constant `_SI_ASYMPTOTIC_T`, the host `si_minus_half_pi`, the host
`sincsq_tail`, and both preamble functions; $R$ is continuous across it to
$2.2\times10^{-16}$ relative at $u = 48/2\pi$ from either side.

### Numerical evidence

All values below are from fresh scripts written for this verification, against
`scipy.integrate.quad` and 40-digit `mpmath` (not implementation helpers).

| Check | Result |
|---|---|
| $F$ vs adaptive quadrature, 9 points in $[-13.7, 51]$ | $\le 1.7\times10^{-16}$ absolute |
| $F'$ vs $\operatorname{sinc}^2$ by 40-digit numerical differentiation | $\le 2\times10^{-41}$ |
| $F(\pm 10^8) = \pm0.4999999995$ | consistent with $\pm1/2 - 1/(2\pi^2 x)$ |
| `sincsq_tail` vs `mpmath`, 61 points over $\lvert x\rvert\in[10^{-12},10^{12}]$, both signs | $\le 5.3\times10^{-15}$ relative |
| bin mean vs quadrature in eV coordinates, 12 random bins, $a = 3.7$ eV$^{-1}$ | $\le 7.9\times10^{-14}$ absolute |
| far-tail bins ($E_{\mathrm{res}}$ 9.4 keV outside the window) vs `mpmath` | $\le 1.4\times10^{-12}$ relative per bin, $1.0\times10^{-14}$ on the sum |
| captured + truncated vs $\pi/a$, $a\in\{0.01, 0.35, 3.2, 41\}$ eV$^{-1}$ | exact to the last bit in all four; captured vs summed bin masses $\le 1.4\times10^{-14}$ |
| positivity, 4000 random bins across the resonance and 2001 far-tail bins | minima $2.7\times10^{-14}$ and $1.9\times10^{-15}$, none negative |
| width $\to 0$ (coordinates shifted so $E-E_{\mathrm{res}}$ is exact) | second-order: $6\times10^{-5}$, $6\times10^{-9}$, $1.3\times10^{-11}$ at $w = 10^{-1}, 10^{-3}, 10^{-5}$ |
| width $\to 0$ at 8 keV | floors near $10^{-8}$; the limit is FP64 cancellation in $E - E_{\mathrm{res}}$ on the edges, not the quadrature |
| $a\to\infty$, 1 eV grid, $E_{\mathrm{res}}$ interior | resonance-bin mass fraction $0.934$, $0.99383$, $0.999938$, $0.9999994$ at $a = 10, 10^2, 10^4, 10^6$ eV$^{-1}$; total $\to\pi/a$ |
| CUDA/ROCm preamble compiled on host (`gcc -O2 -ffp-contract=off`) vs `mpmath`, 1005 points | $\operatorname{Si}-\pi/2$: $\le 7.8\times10^{-16}$ absolute for $t\le4$; $t\lvert\text{err}\rvert \le 1.6\times10^{-15}$ for $4<t<48$ and $\le 4.2\times10^{-16}$ for $t\ge48$ |
| same preamble, $R(x)$ on 408 points over 15 decades, both signs | $\le 1.3\times10^{-15}$ relative; device-vs-host bin means agree to $\le 2.2\times10^{-15}$ |

The float32 hazard the issue names is real and is avoided. On a 3 eV grid with
$a = 3$ eV$^{-1}$, forming the bin mass as a difference of float32 $F$ values
gives up to $1.1\times10^{-1}$ relative error on bins away from the resonance
(and $1.8\times10^{-1}$ at $x = 10^6$, where $F - 1/2$ in float32 is pure
cancellation noise), while the shipped $R$-difference route reproduces `mpmath`
to $8.7\times10^{-14}$. The implementation removes the hazard structurally: the
edges and the whole bin-mean evaluation are FP64 on every route, the fused
reduction accumulates in FP64, and the cast to `REAL` happens only on the
finished bin mean, where float32 storage costs the expected $6\times10^{-8}$
relative.

### Verdict

`rederived`. No divergence found in factor, sign, exponent, unit, or convention
between the independent derivation and the implementation, on either the host or
the device route. Two non-physics observations:

* The $t \le 4$ accuracy bound originally stated for the preamble
  ($6.7\times10^{-16}$ here, $7\times10^{-16}$ in the module docstring) was
  tight rather than conservative: a host `gcc -O2 -ffp-contract=off` build
  measured $7.8\times10^{-16}$ on 302 points. The bound is libm- and
  compiler-dependent, so both statements were relaxed to $\le 10^{-15}$, which
  holds with margin. No physics consequence.
* Recovery of the node sample as the bin width shrinks is limited near 8 keV by
  FP64 cancellation in the edge coordinate $E - E_{\mathrm{res}}$, not by the
  quadrature; the limiting case holds exactly once that cancellation is removed
  from the test coordinates.

Human sign-off remains open, as does the fixed-transport 300 keV yield and
backend/precision measurement prepared in `checks/sinc_bin_integration.py`.
