# `mixed-elastic-soft-moments`

## Scope and independence

Validation: `mixed-elastic-soft-moments`. Fresh-context verification on 2026-11-05, same state as [`mixed-elastic-split`](mixed-elastic-split.md). Source: PENELOPE-2024 Eqs. 4.15–4.16 and 4.38–4.39, and the quantile law of `elsepa-elastic-sampling`. Derived before reading the implementation.

Verdict: **`rederived`**.

## Independent derivation

With $\cos\theta=1-2\mu$,

$$
1-P_1=2\mu,\qquad
1-P_2=1-\tfrac12\left[3(1-2\mu)^2-1\right]=6\mu(1-\mu).
$$

Inserting these in Eq. 4.16a with $p(\mu)$ normalized to 1 gives Eq. 4.38, including its factors 2 and 6:

$$
\frac{1}{\lambda^{(s)}_{\rm el,1}}=2\Sigma T_1,\qquad
\frac{1}{\lambda^{(s)}_{\rm el,2}}=6\Sigma(T_1-T_2),\qquad
T_n=\int_0^{\mu_c}\mu^n p(\mu)\,d\mu=\int_0^{\xi_c}\mu(\xi)^n\,d\xi .
$$

The last equality is the substitution $d\xi=p\,d\mu$. For the sampler's law at log-energy fraction $f$, with $g=1-f$ and a frozen $\xi_c$,

$$
T_1=gQ_k+fQ_{k+1},\qquad
T_2=g^2Q^{(2)}_k+2fg\,Q^{(\times)}+f^2Q^{(2)}_{k+1},\qquad
Q^{(\times)}=\int_0^{\xi_c}\mu_k\mu_{k+1}\,d\xi .
$$

The cross term is required: $\int\mu_k\mu_{k+1}\,d\xi\neq\sqrt{Q^{(2)}_kQ^{(2)}_{k+1}}$ in general, and interpolating node $T_2$ values linearly would not reproduce this law.

## Source-to-code comparison

| Item | Derivation | Code | Result |
|---|---|---|---|
| Soft rates | $2\Sigma T_1$, $6\Sigma(T_1-T_2)$ | `_mixed_soft_rates_scalar` returns `2*rate*t1`, `6*rate*(t1-t2)` | matches, [1/cm] |
| Interpolation | $g,f$ linear; $g^2,2fg,f^2$ | same function | matches |
| Column order | $Q_k,Q_{k+1},Q^{(2)}_k,Q^{(\times)},Q^{(2)}_{k+1}$ | `_panel_moments` return and table unpack | consistent |
| Compound | rates add | `_layer_soft_rates` | matches |
| Atomic-electron scaling | carried by $\Sigma$ | `rate_scales` in `pack_elsepa_tables` | matches, angular density unchanged |
| $\xi_c=0$ | zero moments | early return | matches |

### Quadrature

`_panel_moments` uses 6-point Gauss–Legendre on the union of both rows' CDF knots. An independent exact polynomial integration of $\int_0^{\mu_c}\mu^np\,d\mu$ on each piecewise-linear node density, across all 42 elements and $C_1\in\{0.01,0.05,0.2\}$, gives maximum relative errors of $1.1\times10^{-9}$ on $Q$ and $1.7\times10^{-9}$ on $Q^{(2)}$. This is far below the $\sigma_l/\sigma$ table tolerance of `elsepa-elastic-sampling`. The cross term shares the same abscissae and was checked by the existing dense-quantile test.

### Positivity

On the released tables $T_2/T_1\le0.0103$ (H, $C_1=0.2$), so $T_2\le\tfrac23T_1$ holds with a wide margin and $\lambda^{-1}_{s,2}\ge\lambda^{-1}_{s,1}$ everywhere. See `mixed-elastic-hinge` for why this matters.

## Filters

Units pass: $T_n$ is dimensionless and the rate is [1/cm]. Limits pass: $\xi_c\to0$ gives zero rates, and $\mu\to0$ gives $\lambda^{-1}_{s,2}/\lambda^{-1}_{s,1}\to3$, the small-angle Eq. 4.25. Signs pass.

## Limits of this check

Reference data are the released ELSEPA tables as resolved locally. $Q^{(\times)}$ has no independent polynomial check, because $\mu_k\mu_{k+1}$ in $\xi$ is not a polynomial; it rests on the existing 2e6-point quantile test at $f=0$, $0.3$ and $1$, and the Lewis $P_2$ reference in `mixed-elastic-split` is consistent with it for the no-crossing case.
