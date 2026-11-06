# `mixed-elastic-split`

```{note}
This page records verification of the row-start hazard at 861642bbf. The core has since moved to thinning; see the [re-verification at e25dede8b](mixed-elastic-reverification.md).
```

## Scope and independence

Validation: `mixed-elastic-split`. Fresh-context verification on 2026-11-05, branch `issue-894-mixed-elastic-transport`, HEAD `19f59240e` plus the uncommitted move of `validate_mixed_parameters` to `_mixed_elastic_policy.py`, whose logic is unchanged. Source: PENELOPE-2024, NEA/MBDAV/R(2024)1, Eqs. 4.9–4.10, 4.33–4.37 and 4.95–4.98, read from the local PDF. The derivation below was recorded before the implementation or `mixed-elastic-transport.md` was read.

Verdict: **`discrepancy`**, scoped to the core's hazard at a node crossing. The table-level equations all match.

## Independent derivation

With $\mu=(1-\cos\theta)/2$, $1-P_1(\cos\theta)=2\mu$, so $\lambda_{\rm el,1}^{-1}=2\langle\mu\rangle/\lambda_{\rm el}$. Eq. 4.10 then gives the dimensionless hard fraction

$$
f_h\equiv\frac{\lambda_{\rm el}}{\lambda^{(h)}_{\rm el}}
=\frac{\lambda_{\rm el}}{\max(\lambda_{\rm el},C_1\lambda_{\rm el,1})}
=\min\!\left(1,\frac{2\langle\mu\rangle}{C_1}\right),
$$

and Eqs. 4.35–4.37 give $\xi_c=1-f_h$. Restricting a uniform deviate to $(\xi_c,1)$ gives $\xi=1-f_h(1-u)$ for $u\in[0,1)$.

The sampler of `elsepa-elastic-sampling` is $\mu(\xi,E)=(1-f)\mu_k(\xi)+f\mu_{k+1}(\xi)$. For any $\xi_c$ that does not depend on $f$, the sets $(0,\xi_c)$ and $(\xi_c,1)$ partition the unit interval, so every moment closes:

$$
\int_0^{\xi_c}\mu^n\,d\xi+\int_{\xi_c}^1\mu^n\,d\xi=\langle\mu^n\rangle(E).
$$

### Compound bound

Let $\Sigma_{1,i}=2\Sigma_i\langle\mu\rangle_i$ and $\Sigma_{h,i}=f_{h,i}\Sigma_i=\min(\Sigma_i,\Sigma_{1,i}/C_1)$. An active element has $\Sigma_{s,1,i}\le\Sigma_{1,i}=C_1\Sigma_{h,i}$, and an element with $f_{h,i}=1$ has no soft part, so at the lower node

$$
\frac{\sum_i\Sigma_{s,1,i}}{\sum_i\Sigma_{h,i}}\le C_1 .
$$

Inside a panel the bound gains the factor $\langle\mu\rangle(E)/\langle\mu\rangle_k$. Also $\sum_i\min(a_i,b_i)\le\min(\sum_i a_i,\sum_i b_i)$, so the summed hard rate never exceeds the material-level rate of Eq. 4.10, with equality when every element is active.

Markov's inequality bounds the cutoff at the node: $\mu_k(\xi_c)\le\langle\mu\rangle_k/f_h=C_1/2$.

## Source-to-code comparison

| Item | Source | Code | Result |
|---|---|---|---|
| Hard fraction | Eq. 4.10 | `mixed_elastic_tables`: `min(1, 2 <mu> / c1)` | matches, dimensionless |
| $\langle\mu\rangle$ | exact on piecewise-linear $p$ | `mean_angular_deflection` | matches, polynomial integral |
| Hard rate | $\Sigma f_h$ | `_mixed_hard_rate_scalar`, [1/cm] | matches, same bracket as the sampler |
| Hard quantile | Eq. 4.37 | `_mixed_hard_quantile_scalar` | matches |
| Parameter range | §4.4, $C_1,C_2\le0.2$ | `validate_mixed_parameters` | matches |
| $C_1=0$ | detailed | $f_h=1$, no soft rates, no draws | matches, see below |

$C_1=0$ is bitwise because `_stream_uniform_scalar` returns multiples of $2^{-53}$, for which $1-(1-u)=u$ exactly, and multiplying a rate by exactly 1 changes nothing.

### Real tables

All 42 transport elements, $C_1\in\{0.01,0.05,0.2\}$:

| Quantity | $C_1=0.01$ | $C_1=0.05$ | $C_1=0.2$ |
|---|---|---|---|
| Max soft deflection per hard step, in units of $C_1$ | 0.639 (H) | 0.724 (H) | 0.796 (Se) |
| Max $\langle\mu\rangle_{k+1}/\langle\mu\rangle_k$ on an active panel | 0.855 | 0.866 | 1.265 (Au, 0.3 keV) |
| Max $\mu_c$ | $5.5\times10^{-4}$ | $3.5\times10^{-3}$ | 0.019 |

The compound bound holds across these tables. The statement that the lower node has the larger $\langle\mu\rangle$ is false for heavy elements below about 1 keV at $C_1=0.2$. The bound survives because the ratio stays small.

## Discrepancy

The partition is exact at one energy, but a row evaluates its hazard and its quantile at two different energies:

- `cores.py:428` sets the hazard from the row-start energy `E_j`.
- `cores.py:942` consumes the optical depth with that hazard over the whole row.
- `cores.py:1170` selects the quantile from the row-end energy `E_keV[e]`.

First divergent term: on a row that crosses node $k+1$ downward, the hazard over the part of the row below the node is $\Sigma f_{h,k+1}$ where the claim requires $\Sigma f_{h,k}$. Since $f_{h,k}/f_{h,k+1}\approx1.15$–$1.3$ per panel, this is a discontinuous error, unlike the smooth row-start lag of detailed transport.

Rows are the hinge-split segments of an exponential step, so with $r$ the stale hard rate the mean residual row length at a crossing is $2/(3r)$. The expected number of hard events lost per crossing is

$$
\Delta N\simeq\frac23\left(\frac{f_{h,k}}{f_{h,k+1}}-1\right),
$$

independent of $C_1$ and $E$. Each lost event carries about $C_1$ of first moment. Through the frozen rate, the cumulative relative first-moment deficit from this term alone scales as $C_1S\lambda_{\rm el,1}/E$.

### Measurements

Each run used the real per-electron core and released tables, with a monkeypatched start at mid-slab so the medium is infinite and no electron exits. Stopping was continuous, without straggling. The noise-free reference is the Lewis result with energy dependence, $\langle P_\ell\rangle(s)=\exp[-\int_0^s ds'/\lambda_{\rm el,\ell}(E(s'))]$. Detailed transport reproduces it: Si at 30 keV gives $+0.05\pm0.17$ % on $1-\langle\cos\chi\rangle$, W at 30 keV gives $-0.26\pm0.21$ %.

| Case, path $0.3\lambda_{\rm el,1}$, $C_1=C_2=0.05$ | Crosses a node | $1-\langle\cos\chi\rangle$ vs Lewis | $1-\langle P_2\rangle$ vs Lewis |
|---|---|---|---|
| W, $E_0=28$ keV | no | $-0.30\pm0.24$ % | $-0.16\pm0.16$ % |
| W, $E_0=25.2$ keV | yes, 25 keV | $-1.91\pm0.24$ % | $-1.61\pm0.16$ % |
| W, $E_0=30$ keV (on a node) | yes, at the first row | $-1.38\pm0.24$ % | $-1.16\pm0.16$ % |
| Si, $E_0=30$ keV | yes | $-1.34\pm0.17$ % | $-0.97\pm0.11$ % |
| Si, $E_0=30$ keV, $C_1=C_2=0.2$ | yes | $-3.29\pm0.17$ % | $-3.41\pm0.11$ % |

The first three rows isolate the cause: the bias is absent without a crossing. A beam energy on a table node, such as 30, 100, 200 or 300 keV, always crosses at its first row.

Hard-event counts against the exact $\int\Sigma f_h\,ds$ agree with $\Delta N$ (Si, 30 to 24 keV, $C_1=C_2=0.05$, two crossings, about 18.7 events):

| Run | Hard events / exact integral |
|---|---|
| Core, defaults | $0.9809\pm0.0007$ |
| Core, $C_2=0.005$ | $0.9947\pm0.0007$ |
| Core, `max_dE_frac` = 0.002 | $0.9956\pm0.0007$ |
| Core, $C_1=C_2=0.2$ | $0.9376\pm0.0015$ |
| Independent toy stepper, row-start hazard, defaults | $0.979\pm0.001$ |
| Independent toy stepper, Eq. 4.96–4.98 thinning, defaults | $1.002\pm0.001$ |
| Independent toy stepper, thinning, $C_1=C_2=0.2$ | $0.999\pm0.003$ |

The bias shrinks with row length, which rules out an error in the rates or moments. Thinning removes it in the toy, which counts events only and does not model angles.

`mixed-elastic-transport.md` states that the jump "changes speed, not physics". These numbers contradict that.

## Deviations judged

**Per-element cutoff instead of material-level.** Acceptable. Each element's partition is exact, the soft deflection per hard step is bounded as derived above, and the summed hard rate equals Eq. 4.10 whenever every element is active.

**Frozen panel cutoff instead of a continuous $\xi_c(E)$.** The partition argument is correct and valuable. Its price is the hazard discontinuity above, which the core must resolve.

## Limits of this check

- The compound bound was derived and not measured on a compound in the Lewis test.
- The per-event hard first moment at a crossing row (quantile mismatch) was not resolved separately from the count deficit: the measured event-mean ratio is about $0.996$ with comparable noise.
- Fresh-context verification did not read PENELOPE's Fortran.
