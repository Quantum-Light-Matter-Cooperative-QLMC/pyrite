# Validation: sinc-bin-integration

Implementation-context derivation and measurements for the bin-mean line
quadrature (#116). This is **not** the independent fresh-context re-derivation
the methodology requires; that and human sign-off are pending.

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
$\le 6.7\times10^{-16}$ for $t \le 4$, and error times $t$ $\le
2.1\times10^{-15}$ for $t > 4$ (i.e. relative to the $1/t$ envelope).

## Routes and measurements

The deterministic CPU mass, limit, policy, per-hkl, and batched-route checks are
anchored in the tests listed above. CUDA/CuPy kernel checks are present but need
a CUDA runner. The fixed-transport 300 keV yield and backend/precision comparison
is prepared in `checks/sinc_bin_integration.py`; that heavy measurement remains
to be run on the remote GPU box.

## Open

* Fresh-context independent re-derivation of {eq}`eq-sbi-bin-mean` and
  {eq}`eq-sbi-tail-form` against the code.
* Human sign-off.
