# sinc-bin-midpoint-series

Ledger row: [`sinc-bin-midpoint-series`](../ledger-core-coherent-physics.md#sinc-bin-midpoint-series). Status: unverified. Validation: `sinc-bin-midpoint-series`.

Implementation-context derivation of a cheaper evaluator for the exact [`sinc-bin-integration`](sinc-bin-integration.md) bin mean, for bins much narrower than a lobe of the line (#1003). It changes how the same mean is computed, not the quadrature.

## Claim

With $x = a(E-E_{\mathrm{res}})/\pi$ and $f(x) = \operatorname{sinc}^2 x$, a bin $[x_0, x_1]$ with centre $x_c = (x_0+x_1)/2$ and width $\Delta = x_1 - x_0$ has mean

```{math}
:label: eq-sbms-series

\bar f = f(x_c) + \frac{\Delta^2}{24} f''(x_c) + \frac{\Delta^4}{1920} f^{(4)}(x_c) + R_6,
```

and inside the window $\Delta \le 0.05$, $|x_c| \ge 2$ (`_SERIES_MAX_DX`, `_SERIES_MIN_X`) the remainder satisfies $|R_6| \le 2\times10^{-8}\, e(x_c)$, with $e(x) = 1/(2\pi^2x^2)$ the $\sin^2$-averaged envelope. That is below float32 rounding of the local envelope. Bins outside the window keep the two-tail evaluation `R(x_1) - R(x_0)` unchanged.

## Derivation

Taylor-expand $f$ about $x_c$ and average over $[x_c - \Delta/2, x_c + \Delta/2]$: odd terms vanish, and the $2n$-th term is $f^{(2n)}(x_c)\,(\Delta/2)^{2n}/(2n+1)!$, giving the coefficients $1/24$, $1/1920$ and $1/322560$ for $n = 1, 2, 3$. The remainder after the $n=2$ term is bounded by $\sup|f^{(6)}|\,\Delta^6/322560$ over the bin.

Write $f = A\,H/(2\pi^2)$ with $A = 1-\cos 2\pi x$ and $H = x^{-2}$. Then $|A^{(k)}| \le 2(2\pi)^k$ for $k \ge 0$ and $|H^{(j)}| = (j+1)!\,|x|^{-(j+2)}$. By Leibniz,

```{math}
\frac{|f^{(6)}|}{e(x)} \le 2\sum_{j=0}^{6}\binom{6}{j}(2\pi)^{6-j}(j+1)!\,|x|^{-j}.
```

At $|x| \ge 1.975$ (the smallest $|x|$ in a window bin) the sum is below $3.6\times10^{5}$, so $|R_6|/e \le 3.6\times10^5 \cdot (0.05)^6/322560 \approx 1.7\times10^{-8}$; $e$ itself varies by under 6% across the bin. Every derivative comes from one $\sin\pi x_c$, $\cos\pi x_c$ pair: $A = 2\sin^2\pi x$, $\sin 2\pi x = 2\sin\pi x\cos\pi x$, $\cos 2\pi x = (\cos\pi x - \sin\pi x)(\cos\pi x + \sin\pi x)$, which avoids the $1-\cos$ cancellation near zeros of $f$.

## Assumptions

- The `sinc-bin-integration` profile and bin convention; $x_0$, $x_1$ are formed in FP64 exactly as for the two-tail route.
- FP64 evaluation (`sincospi` on device, `sin`/`cos` of $\pi x$ on host).

## Limiting cases

- $\Delta \to 0$: $\bar f \to f(x_c)$, node sampling at the bin centre.
- Far from the resonance, the series and the exact mean both tend to the envelope.
- On a zero of $f$ ($x_c$ integer), $\bar f = \Delta^2 f''/24 + \dots$, which the series keeps.

## Numeric checks

Host mirror against `mpmath` adaptive quadrature (40 digits), 400 random window bins including $\Delta = 0.05$ and bins centred on zeros: maximum $|\bar f_{\rm series} - \bar f|/e = 3.0\times10^{-9}$. The existing FP64 two-tail evaluation on the same bins reaches $1.3\times10^{-7}$, because $R(x_1) - R(x_0)$ cancels when $\Delta \ll 1$; the series is the more accurate of the two inside the window. Regressions: `test_midpoint_series_matches_the_bin_integral_inside_its_window` (host, `scipy` quadrature), `test_device_midpoint_series_matches_the_host_mirror` and `test_series_dispatch_takes_only_window_bins` (CUDA).

## Status

Compiled in only when `PYRITE_BIN_MEAN_SERIES=1`; production kernels build with it off until #1003's measurement adopts it.
