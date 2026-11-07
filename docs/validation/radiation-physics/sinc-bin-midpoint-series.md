# sinc-bin-midpoint-series

Ledger row: [`sinc-bin-midpoint-series`](../ledger-core-coherent-physics.md#sinc-bin-midpoint-series). Status: rederived. Validation: `sinc-bin-midpoint-series`.

Implementation-context derivation of a cheaper evaluator for the exact [`sinc-bin-integration`](sinc-bin-integration.md) bin mean, for bins much narrower than a lobe of the line (#1003). It changes how the same mean is computed, not the quadrature.

## Claim

With $x = a(E-E_{\mathrm{res}})/\pi$ and $f(x) = \operatorname{sinc}^2 x$, a bin $[x_0, x_1]$ with centre $x_c = (x_0+x_1)/2$ and width $\Delta = x_1 - x_0$ has mean

```{math}
:label: eq-sbms-series

\bar f = f(x_c) + \frac{\Delta^2}{24} f''(x_c) + \frac{\Delta^4}{1920} f^{(4)}(x_c) + \frac{\Delta^6}{322560} f^{(6)}(x_c) + R_8,
```

and inside the window $\Delta \le 0.1$, $|x_c| \ge 2$ (`_SERIES_MAX_DX`, `_SERIES_MIN_X`) the remainder satisfies $|R_8| \le 2.5\times10^{-9}\, e(x_c)$, with $e(x) = 1/(2\pi^2x^2)$ the $\sin^2$-averaged envelope. That is below float32 rounding of the local envelope. Bins outside the window keep the two-tail evaluation `R(x_1) - R(x_0)` unchanged.

## Derivation

Taylor-expand $f$ about $x_c$ and average over $[x_c - \Delta/2, x_c + \Delta/2]$: odd terms vanish, and the $2n$-th term is $f^{(2n)}(x_c)\,(\Delta/2)^{2n}/(2n+1)!$, giving the coefficients $1/24$, $1/1920$, $1/322560$ and $1/92897280$ for $n = 1, \dots, 4$. The remainder after the $n=3$ term is bounded by $\sup|f^{(8)}|\,\Delta^8/92897280$ over the bin.

Write $f = A\,H/(2\pi^2)$ with $A = 1-\cos 2\pi x$ and $H = x^{-2}$. Then $|A| \le 2$, $|A^{(k)}| \le (2\pi)^k$ for $k \ge 1$, and $|H^{(j)}| = (j+1)!\,|x|^{-(j+2)}$. By Leibniz,

```{math}
\frac{|f^{(8)}(x)|}{e(x)} \le \sum_{j=0}^{7}\binom{8}{j}(2\pi)^{8-j}(j+1)!\,|x|^{-j} + 2\cdot 9!\,|x|^{-8}.
```

At $|x| = 1.95$, the smallest $|x|$ in a window bin, the sum is $1.148\times10^{7}$, so $|R_8| \le 1.24\times10^{-9}\,e(x)$ at $\Delta = 0.1$. Converting to the bin centre multiplies by at most $(2/1.95)^2 = 1.052$, so $|R_8| \le 1.3\times10^{-9}\,e(x_c)$, inside the claimed $2.5\times10^{-9}$. (Stopping after $f^{(4)}$ would need $\Delta \le 0.05$ for $1.7\times10^{-8}$.) Every derivative comes from one $\sin\pi x_c$, $\cos\pi x_c$ pair: $A = 2\sin^2\pi x$, $\sin 2\pi x = 2\sin\pi x\cos\pi x$, $\cos 2\pi x = (\cos\pi x - \sin\pi x)(\cos\pi x + \sin\pi x)$, which avoids the $1-\cos$ cancellation near zeros of $f$.

## Assumptions

- The `sinc-bin-integration` profile and bin convention. The device forms $x_c = (a/\pi)(\tfrac12(\epsilon_i+\epsilon_{i+1}) - E_{\rm res})$ and $\Delta = (a/\pi)(\epsilon_{i+1}-\epsilon_i)$ directly in FP64, so the window test and series need no division by $\pi$; out-of-window bins form $x_0$, $x_1$ exactly as before.
- FP64 evaluation (`sincospi` on device, `sin`/`cos` of $\pi x$ on host, whose phase error grows like $|x|\epsilon$). Device constant divisions are written as reciprocal products.

## Limiting cases

- $\Delta \to 0$: $\bar f \to f(x_c)$, node sampling at the bin centre.
- Far from the resonance, the series and the exact mean both tend to the envelope.
- On a zero of $f$ ($x_c$ integer), $\bar f = \Delta^2 f''/24 + \dots$, which the series keeps.

## Numeric checks

Host mirror against `mpmath` adaptive quadrature (40 digits), 400 random window bins with $\Delta$ from $10^{-7}$ to $0.1$, including $\Delta = 0.1$ and bins centred on zeros: maximum $|\bar f_{\rm series} - \bar f|/e = 2.6\times10^{-10}$. The FP64 two-tail evaluation reaches $2.6\times10^{-7}$ on the same bins, because $R(x_1) - R(x_0)$ cancels when $\Delta \ll 1$. The series has the smaller worst case over the window, but not on every bin set: its truncation error near $\Delta = 0.1$ is larger than the two-tail error on bins of comparable width, and both are far below float32 rounding. Regressions: `test_midpoint_series_matches_the_bin_integral_inside_its_window` (host, `scipy` quadrature), `test_device_midpoint_series_matches_the_host_mirror` and `test_series_dispatch_takes_only_window_bins` (CUDA).

## Independent verification

Fresh-context verification (2026-11-06, commit `1631c4cb7`) returned `rederived`.

- Coefficients rederived from the bin average and with `sympy`. The next term, $1/92897280$, predicts the measured residual at $\Delta = 0.1$: $-6.199\times10^{-11}$ against $-6.201\times10^{-11}$ at $x_c = 2$.
- Device C and host Python checked term by term; the assembled derivatives agree with `mpmath` to $10^{-59}$ relative, including negative $x_c$.
- Reference: Si closed form at 60 digits, cross-checked against `mpmath.quad`. Worst series error $2.61\times10^{-10}$ of $e$ for $|x_c| \le 64$, at zeros and peaks of $f$. On real fixture bins, $2.53\times10^{-10}$.
- Host mirror above $|x_c| \approx 10^6$: FP64 phase rounding dominates ($9.7\times10^{-9}$ at $10^7$). No production effect, since the mirror is test-only and the far cutoff is 64.
- Dispatch: out-of-window bins run the same statements as without the series. The series ignores `inv_width`, so it needs `inv_width = 1/(hi - lo)`, which every production caller provides.

The verifier found two errors on this page, corrected above: the proof used $|A^{(k)}| \le 2(2\pi)^k$ for $k \ge 1$, which is loose by 2 and did not reach the claimed bound relative to $e(x_c)$; and it claimed the series beats two-tail on every window bin set.

Production comparison reviewed, not rerun (SLURM 4036, CUDA, FP64 flush sums): `auto` with the series against `auto` without it, on HOPG 300 keV and h-BN 5 MeV, worst per-bin relative deviation $2.1\times10^{-10}$ (floor $10^{-6}$ of peak) and integral $4.2\times10^{-14}$, against a float32 half-ulp of $5.96\times10^{-8}$. Limits: deviation is compared on flush sums, not end-to-end spectra; the matrix route (`sincsq_bin_lineshape`) is not in the replay, though it calls the same function.

## Status

On by default since `CXR_BIN_REDUCTION_MODEL = "near-count-auto-v3-series"`. Device kernels only: the host evaluator stays two-tail. `PYRITE_BIN_MEAN_SERIES=0` compiles the series out for comparison.
