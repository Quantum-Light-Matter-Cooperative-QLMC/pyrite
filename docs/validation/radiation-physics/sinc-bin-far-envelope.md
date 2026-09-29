# sinc-bin-far-envelope

Ledger row: [`sinc-bin-far-envelope`](../ledger-core-coherent-physics.md#sinc-bin-far-envelope). Status: rederived. Validation: `sinc-bin-far-envelope`.

Implementation-context derivation of the far-field approximation that the production `bin-mean` routes apply beyond $K$ first-zero widths of each resonance (#192). The exact bin mean is the [`sinc-bin-integration`](sinc-bin-integration.md) claim.

## Claim

With $x = a(E-E_{\mathrm{res}})/\pi = (E-E_{\mathrm{res}})/w$, where $w=\pi/a$ is the first-zero width, a bin $[\epsilon_i,\epsilon_{i+1}]$ lies in the far field when both its edges are at least $K$ widths from the resonance on the same side ($x_i \ge K$ or $x_{i+1} \le -K$). There the production routes write the exact bin mean of the $\sin^2$-averaged envelope $\tfrac12(\pi x)^{-2}$:

```{math}
:label: eq-sbfe-envelope-mean

\bar S_i^{\mathrm{far}} = \frac{1}{2\pi^2\,x_i\,x_{i+1}} .
```

All other bins keep the exact mean {eq}`eq-sbi-bin-mean`. Per line, the integrated-yield change is at most

```{math}
:label: eq-sbfe-yield-bound

\left|\frac{\Delta Y}{Y}\right| \le \frac{3}{2\pi^3 K^2},
```

which is $1.2\times10^{-5}$ at the production $K = 64$ (`BIN_MEAN_EXACT_WIDTHS`). Any single bin's mean changes by at most $1/(\pi^2K^2)$ of the line's peak density ($2.5\times10^{-5}$). Both bounds are conservative: the verification finds the sharp yield constant is $1/(2\pi^3K^2)$, and the pointwise bound carries a factor 2 of margin.

## Derivation

Write $\operatorname{sinc}^2x = \sin^2(\pi x)/(\pi x)^2 = \tfrac12(\pi x)^{-2} - \cos(2\pi x)/(2\pi^2x^2)$. Over a bin, the envelope term integrates in closed form. With $\mathrm{d}E = w\,\mathrm{d}x$ and $w(x_{i+1}-x_i) = \epsilon_{i+1}-\epsilon_i$:

```{math}
\int_{\epsilon_i}^{\epsilon_{i+1}} \frac{\mathrm{d}E}{2\pi^2 x^2}
= \frac{w}{2\pi^2}\left(\frac{1}{x_i}-\frac{1}{x_{i+1}}\right)
= \frac{\epsilon_{i+1}-\epsilon_i}{2\pi^2 x_i x_{i+1}},
```

and dividing by the bin width gives {eq}`eq-sbfe-envelope-mean`. The product form has no cancellation between $1/x_i$ and $1/x_{i+1}$, so float32 evaluation is safe. Far bins on one side are contiguous and share edges, so their envelope masses telescope to the envelope integral over the union $[X, Y]$, with $|X| \ge K$. The yield change on that side is therefore only the omitted oscillatory term. Integrating by parts, with $\int\cos(2\pi x)\,\mathrm{d}x = \sin(2\pi x)/(2\pi)$:

```{math}
\left|\int_X^Y \frac{\cos 2\pi x}{x^2}\,\mathrm{d}x\right|
\le \left[\frac{|\sin 2\pi x|}{2\pi x^2}\right]_{X,Y} + \int_X^Y \frac{|\sin 2\pi x|}{\pi x^3}\,\mathrm{d}x
\le \frac{1}{\pi X^2} + \frac{1}{2\pi X^2} = \frac{3}{2\pi X^2}.
```

The energy mass changes by at most $\frac{w}{2\pi^2}\cdot\frac{3}{2\pi K^2}$ per side. Relative to the line mass $w$, and summed over both sides, this gives {eq}`eq-sbfe-yield-bound`. Pointwise, $|\cos(2\pi x)|/(2\pi^2x^2) \le 1/(2\pi^2K^2)$ on the far set; the bin mean of that term is bounded the same way, and the $1/(\pi^2K^2)$ statement carries a factor of 2 margin.

## Assumptions

- The same incoherent finite-time $\operatorname{sinc}^2$ profile and bin convention as `sinc-bin-integration`.
- The line's weight is energy-independent within the line, so each line's error is the profile error times its weight. Summed over lines, the relative bound still holds for the total yield.
- float32 arithmetic in the far branch. Each FP64 edge and the FP64 resonance are split into a float32 head and a float32 remainder, and the offset is formed as $(\epsilon_{\rm head}-E_{\rm head}) + (\epsilon_{\rm rem}-E_{\rm rem})$. $x$ therefore stays near float32 relative accuracy; the error grows only once $E_{\rm res}/|\epsilon-E_{\rm res}| \gtrsim 10^6$ (about $400\times2^{-24}$ at $10^{10}$, re-check), well inside the bound. The fused reduction receives $E_{\rm res}$ already in float32 (remainder 0), and there the near pairs use the same float32 value. Rounding $E_{\rm res}$ to float32 before forming the offset shifts $x$ by up to $\mathrm{ulp}(E_{\rm res})/(2w)$ widths. With near bins on the exact FP64 resonance, the fresh-context verification found this breaks the yield bound once $E_{\rm res}/w$ approaches $2^{23}$: $5.4\times10^{-5}$ at 1.6 MeV, $w=0.01$ eV. The split removes that condition. `test_far_envelope_bound_holds_at_mev_resonances_on_narrow_lines` and `test_far_envelope_never_claims_the_core_of_a_sub_mev_width_line` pin both remainders.

## Limiting cases

- $K \to \infty$ recovers the exact bin mean (`exact_widths=None` on both routes).
- A bin much wider than $w$ far from the resonance: the oscillatory term averages out, and the envelope mean equals the exact mean up to $O(w/\Delta)$ of the envelope.
- A bin that contains, or lies within $K$ widths of, the resonance is exact by construction, so the line core and its near tails are untouched.

## Cost

Only bins within $\pm K$ widths of a resonance evaluate the FP64 sine integral, which is about $4K$ bins per line at spacing $h \ge w/2$. Every other (line, bin) pair is classified and evaluated in float32. Each FP64 edge and the resonance are split into a float32 head and remainder, so $\epsilon - E_{\mathrm{res}}$ stays near float32 relative accuracy, and far pairs accumulate in a compensated float32 sum. On the GPU, the fused reduction (one block per bin) compacts near pairs into a shared queue in lane order and evaluates them on full warps. Otherwise, on a consumer card, one near lane forces its whole warp through the FP64 path. Summation order stays fixed, so the reduction is deterministic.

| route (RTX 5080, SLURM 246 and later; 50,000 lines × 240,000 bins) | wall s | pairs/s |
| --- | --- | --- |
| exact bin mean | 10.05 | $1.2\times10^{9}$ |
| hybrid, first kernel (per-pair FP64 classification) | 2.81 | $4.3\times10^{9}$ |
| hybrid, float32 far path + near-pair compaction | 0.62 | $1.9\times10^{10}$ |
| node sampling (reference) | 0.11 | $1.1\times10^{11}$ |

The same run changes yield by $-8.6\times10^{-7}$ relative to the exact route. On host (NumPy), a 200-line, 120,000-bin case evaluates in 0.43 s instead of 3.53 s.

## Numeric checks (host, scratch)

Randomised lines, 300 per axis, $w$ log-uniform in 0.05–30 eV, on uniform 0.5 eV, uniform 3 eV and a piecewise 3/0.25/1.5 eV axis:

| $K$ | worst per-line $\lvert\Delta Y/Y\rvert$ | bound {eq}`eq-sbfe-yield-bound` | worst pointwise / peak | $1/(\pi^2K^2)$ |
| --- | --- | --- | --- | --- |
| 16 | $6.1\times10^{-5}$ | $1.9\times10^{-4}$ | $2.0\times10^{-4}$ | $4.0\times10^{-4}$ |
| 64 | $3.9\times10^{-6}$ | $1.2\times10^{-5}$ | $1.2\times10^{-5}$ | $2.5\times10^{-5}$ |

Regressions: `tests/montecarlo/test_sinc_bin_integration.py` (bound, near bins bit-identical, closed form beyond, production default) and `tests/montecarlo/test_sinc_bin_integration_cuda.py` (device and host hybrid agreement, reduction mass within the bound).

## Budget

The $1.2\times10^{-5}$ yield term is charged to the feature-window row of `tbl-line-budget-allocation`, alongside the measured-bandwidth shares ($3\times10^{-4}$ of its $5\times10^{-4}$).

Fresh-context validation and human sign-off pending.

## Independent verification

Fresh-context verifier, 2026-09-28, worktree HEAD `1a8434e9`. Derived from the ledger row and the `sinc-bin-integration` profile alone, before the implementation bodies or the derivation above were read. The verifier changed only this section. Scratch scripts are outside the repository.

- **Claim**: `sinc-bin-far-envelope` — `montecarlo/spectrum/lines/_bin_quadrature.py::_host_bin_mean`, `::_far_envelope_float32`, `SINCSQ_BIN_PREAMBLE::pyrite_sincsq_far_mean`, `::pyrite_sincsq_bin_mean_hybrid`, `_BIN_REDUCE_SOURCE`, `BIN_MEAN_EXACT_WIDTHS` — $\operatorname{sinc}^2x=\tfrac12(\pi x)^{-2}-\cos(2\pi x)/(2\pi^2x^2)$ plus integration by parts
- **Filters**: units pass; limits pass; signs/conventions pass (see the table)
- **Re-derivation**: `matches` for the envelope mean, telescoping, and both bounds. The implementation `differs` in one convention: the resonance energy is rounded to float32 *before* the offset $\epsilon-E_{\mathrm{res}}$ is formed. This contradicts the stated assumption (Assumptions, third bullet) and breaks {eq}`eq-sbfe-yield-bound` when $\mathrm{ulp}_{32}(E_{\mathrm{res}})/(2w)\gtrsim 1$ (finding 1)
- **Verdict**: `discrepancy` (narrow: the mathematics is rederived; the stated yield bound is unconditional, but the code meets it only under an unstated resolving-power condition)
- **Write-up**: `docs/validation/radiation-physics/sinc-bin-far-envelope.md`
- **Suggested ledger change**: `unverified` → `discrepancy` until either (a) the claim and page state the condition $\mathrm{ulp}_{32}(E_{\mathrm{res}})\ll 2w$ (equivalently $E_{\mathrm{res}}/w\ll 2^{23}$), backed by a regression at large $E_{\mathrm{res}}/w$, or (b) $E_{\mathrm{res}}$ receives the same head/remainder split as the edges. Either way, then `rederived`. Never `signed-off` from this context.

### Independent derivation

Let $x=(E-E_{\mathrm{res}})/w$. The line peak is $1$ and the line mass is $w\int\operatorname{sinc}^2x\,\mathrm{d}x=w$. On a bin $[a,b]$ with $ab>0$, the bin mean of $\operatorname{env}(x)=1/(2\pi^2x^2)$ in $E$ equals its mean in $x$, because the Jacobian cancels:

$$
\frac{1}{b-a}\int_a^b\frac{\mathrm{d}x}{2\pi^2x^2}=\frac{1}{2\pi^2(b-a)}\left(\frac1a-\frac1b\right)=\frac{1}{2\pi^2ab}.
$$

The product form contains no difference, so its float32 relative error is a few ulp. The left side ($a,b<0$) gives a positive result with the same formula. On one side, the far bins with shared edges cover $[X,Y]$ with $\lvert X\rvert\ge K$ ($Y$ is the window edge or $\pm\infty$). The envelope masses telescope to $\frac{w}{2\pi^2}(1/X-1/Y)$, so the yield change on that side is exactly $w\,I$, where $I=\int_X^Y\cos(2\pi x)/(2\pi^2x^2)\,\mathrm{d}x$. One integration by parts, bounding $\lvert\sin\rvert\le1$ in both the boundary term and the remainder, gives

$$
\lvert 2\pi^2 I\rvert\le\frac{1}{2\pi X^2}+\frac{1}{2\pi Y^2}+\frac1\pi\int_X^Y\frac{\mathrm{d}x}{x^3}
=\frac{1}{\pi X^2}.
$$

This is sharper than the page's $3/(2\pi X^2)$, because the page drops the $-1/(2\pi Y^2)$ from the remainder integral. A second integration by parts gives the asymptote $I\simeq-\sin(2\pi X)/(4\pi^3X^2)$. The two sides can have the same sign, so the attainable total is $1/(2\pi^3K^2)$. The results are:

| quantity | independent (sharp) | this verifier's rigorous bound | claimed | claim status |
| --- | --- | --- | --- | --- |
| per-line $\lvert\Delta Y/Y\rvert$ | $\to 1/(2\pi^3K^2)=3.94\times10^{-6}$ | $1/(\pi^3K^2)$ | $3/(2\pi^3K^2)=1.18\times10^{-5}$ | valid, loose by 3 |
| any bin mean / peak | $\to 1/(2\pi^2K^2)=1.24\times10^{-5}$ | $1/(2\pi^2K^2)$ | $1/(\pi^2K^2)=2.47\times10^{-5}$ | valid, loose by 2 (stated) |

The pointwise bound follows because a bin mean of $\cos(2\pi x)/(2\pi^2x^2)$ cannot exceed the function's supremum $1/(2\pi^2K^2)$ on $\lvert x\rvert\ge K$.

### Filters

| filter | result |
| --- | --- |
| units | $x$ and the bin mean are dimensionless relative to the peak; the mass is $w\times$ dimensionless; $\Delta Y/Y$ is dimensionless. Pass |
| $K\to\infty$ | no bin is far; `exact_widths=None` passes `np.inf` to both routes. Pass |
| bins wide relative to $w$ | the envelope mean is still the exact envelope integral; only the oscillatory term is dropped, and it is bounded by the same IBP. Pass |
| bin containing the resonance | needs $x_{\mathrm{lo}}<0<x_{\mathrm{hi}}$, so neither $x_{\mathrm{lo}}\ge K$ nor $x_{\mathrm{hi}}\le-K$ holds; a wide bin straddling $\pm K$ stays exact. With the float32 $E_{\mathrm{res}}$ shift $\delta$ (finding 1), misclassification needs $\delta>K$ widths, i.e. $E_{\mathrm{res}}/w\gtrsim10^9$. Pass for any realistic input |
| sign | the left-side product $x_{\mathrm{lo}}x_{\mathrm{hi}}>0$; the envelope is positive on both sides. Pass |
| $\lvert x\rvert\approx K$ | float32 rounding in `scale` and in the product moves the classification threshold by $\sim10^{-6}K$; the bound changes by the same relative amount. Pass |

### Reproduced numbers

Host route, 300 random lines per axis, $w$ log-uniform in 0.05–30 eV, $E_{\mathrm{res}}\in[3,17]$ keV:

| $K$ | step eV | worst $\lvert\Delta Y/Y\rvert$ | $1/(2\pi^3K^2)$ | worst pointwise | $1/(2\pi^2K^2)$ |
| --- | --- | --- | --- | --- | --- |
| 16 | 0.5 | $6.09\times10^{-5}$ | $6.30\times10^{-5}$ | $1.97\times10^{-4}$ | $1.98\times10^{-4}$ |
| 16 | 3.0 | $6.09\times10^{-5}$ | $6.30\times10^{-5}$ | $1.83\times10^{-4}$ | $1.98\times10^{-4}$ |
| 64 | 0.5 | $3.89\times10^{-6}$ | $3.94\times10^{-6}$ | $1.23\times10^{-5}$ | $1.24\times10^{-5}$ |
| 64 | 3.0 | $3.81\times10^{-6}$ | $3.94\times10^{-6}$ | $1.19\times10^{-5}$ | $1.24\times10^{-5}$ |

These agree with the page's table and sit at the sharp constants. `mpmath` `quadosc` gives $\max_{X\in[64,65]}\lvert I(X)\rvert=1.953\times10^{-6}$, compared with the asymptote $1/(4\pi^3K^2)=1.968\times10^{-6}$. The closed form $1/(2\pi^2ab)$ matches 30-digit quadrature to the last digit.

Stress test (host FP64 route, one-sided far set): $E_{\mathrm{res}}=1.6\times10^{6}+0.05$ eV, $w=0.01$ eV, spacing $w/2$, window from $-400$ to $+30$ widths. `float32(E_res)` is $0.05$ eV $=5$ widths low, and $\Delta Y/Y=5.37\times10^{-5}$. That is $4.5\times$ the claimed $1.18\times10^{-5}$; the predicted shift term $\delta/(2\pi^2K^2)$ is $6.2\times10^{-5}$.

Host-compiled C: `pyrite_sincsq_far_mean`, extracted from `SINCSQ_BIN_PREAMBLE` and built with `gcc -O2` under both `-ffp-contract=off` and `=fast`, was run on 200,000 random $(a, E_{\mathrm{res}}, \epsilon_{\mathrm{lo}}, \epsilon_{\mathrm{hi}})$ with $E_{\mathrm{res}}\in[50, 2\times10^6]$ eV and $w\in[10^{-3},30]$ eV. It gave 0 classification and 0 value mismatches against `_far_envelope_float32` (101,664 far pairs). Float32 $\tilde x$ was never non-monotone across 400 sorted random edge sets, and no side's far set was ever non-contiguous.

A Python emulation of the reduction queue (3,000 random tile patterns, 256 threads) showed a maximum pending count of 511 against a capacity of 512. Every near line was evaluated exactly once, and the queue was empty at exit.

Mutation tests (scratch copy of the test file; `_far_envelope_float32` monkeypatched; far-envelope and production-default tests):

| mutation | result |
| --- | --- |
| $2\pi^2\to\pi^2$ | 5 of 6 fail |
| classify on $\lvert x_{\mathrm{lo}}\rvert$ or $\lvert x_{\mathrm{hi}}\rvert\ge K$ | 3 fail |
| classify on $x_{\mathrm{hi}}\ge K$ / $x_{\mathrm{lo}}\le-K$ | 3 fail |
| $x_{\mathrm{lo}}^2$ in place of $x_{\mathrm{lo}}x_{\mathrm{hi}}$ | 5 fail |
| midpoint $\bar x^2$ in place of $x_{\mathrm{lo}}x_{\mathrm{hi}}$ | 3 fail |
| threshold $K/2$ | 5 fail |
| drop the edge float32 remainder | **all pass** (survives) |

### Findings

1. **Float32 resonance breaks the yield bound at high resolving power.** `_bin_quadrature.py:226` (`(float)e_r`) and `:323` (`E_res.astype(f32)`) round $E_{\mathrm{res}}$ before subtraction; only the edges get the head/remainder split (`:210`). Every far-bin coordinate is therefore shifted by $\delta=[\mathrm{fl}_{32}(E_{\mathrm{res}})-E_{\mathrm{res}}]/w$ widths, with $\lvert\delta\rvert\le \mathrm{ulp}_{32}(E_{\mathrm{res}})/(2w)$. On the FP64 `sincsq_bin_lineshape` route (the CPU backend, or `PYRITE_FP64=1`), the near bins use the unshifted FP64 resonance (`:342`), so the far and near sets no longer meet at the same true edge. This adds $\approx\delta(1/X^2-1/Y^2)/(2\pi^2)$ per side, which cancels between the sides only when they are symmetric. The claimed bound fails once $\lvert\delta\rvert\gtrsim3/\pi$ (see the stress test above). Production lines ($E_{\mathrm{res}}\lesssim10$ keV, $w\gtrsim10^{-2}$ eV, $E_{\mathrm{res}}/w\lesssim10^6$) have $\lvert\delta\rvert\lesssim0.03$, so they are safe by about $30\times$. The page's third assumption (line 51) says the offset is formed in FP64, which is not what the code does. The fused reduction casts `E_r` to float32 for near pairs too (`:644`), so it is self-consistent: the whole line shifts by $\delta$ (a `sinc-bin-integration` position effect), and there is no yield error.
2. **The yield constant is valid but loose by 3.** Page lines 40–42: keeping the $-1/(2\pi Y^2)$ from the remainder integral gives $1/(\pi X^2)$, and the attainable total is $1/(2\pi^3K^2)$. The measured $3.9\times10^{-6}$ confirms this. This is not an error; the ledger's "factor 3/2" check is satisfied as an upper bound.
3. **The pointwise bound is valid with the stated factor-2 margin** (page line 45). The measured value, $1.23\times10^{-5}$, is at the sharp $1/(2\pi^2K^2)$.
4. **The far set on each side is contiguous under the float32 classification** (`_bin_quadrature.py:209-214`). Near $\lvert x\rvert=K$, the head minus float32 resonance is Sterbenz-exact, so $\tilde x$ is monotone. Adjacent bins compute a shared edge from the same double, so telescoping holds up to per-bin float32 rounding ($\sim10^{-7}$ of a far mass $\le 1/(2\pi^2K)$).
5. **Host and device agree.** `_far_envelope_float32` (`:312-331`) reproduces `pyrite_sincsq_far_mean` (`:205-218`) operation for operation, bit-exact under host C with and without FMA contraction. The constants `0.318309886f` and `19.7392088f` are the float32 roundings of $1/\pi$ and $2\pi^2$. `_host_bin_mean` evaluates tails only on edges touching a near bin, then overwrites the far bins (`:360`). This matches `pyrite_sincsq_bin_mean_hybrid` (`:220-232`). The CUDA tests themselves were not run (no local GPU).
6. **The reduction queue is sound** (`:458`, `:491`, `:511`). The queue holds at most $2T-1$ entries (the capacity is $2T$; `shared_mem` at `:654` matches three $8T$ arrays plus $33\times4$ bytes). The flush condition reads a block-uniform `pending` after the third barrier. Every `warp_offset`, `queue`, or `queue_count` write is separated from the prior reads by a `__syncthreads`. `__ballot_sync(0xffffffff)` is reached by all 32 lanes, because the loop bound is block-uniform and `_REDUCE_THREADS = 256` is a multiple of 32. The order and the thread assignment are fixed, so the reduction is deterministic. The Kahan update (`:482-485`) and the final `far_sum - far_carry` are correct.
7. **The regressions discriminate the envelope formula and classification but not the precision handling.** `tests/montecarlo/test_sinc_bin_integration.py:267-270` draws $E_{\mathrm{res}}\le17$ keV, so dropping the edge remainder survives, and finding 1 is untested. A case with $E_{\mathrm{res}}/w\gtrsim10^7$ and a one-sided far set would pin both.
8. **The budget share is consistent.** `docs/validation/beam-transport/line-spectrum-error-budget.md:106`: $3\times10^{-4}+1.2\times10^{-5}\le5\times10^{-4}$. The share is conservative by $3\times$ (finding 2), conditional on finding 1's resolving-power condition.

## Independent verification (re-check)

Second fresh-context verifier, 2026-09-28, worktree HEAD `c5bb3f1e`. Derived from the ledger row and the `sinc-bin-integration` profile before reading the implementation, the author's text, or the first verifier's section. The verifier changed only this section. Scratch scripts, a host-compiled copy of `SINCSQ_BIN_PREAMBLE`, and mutated copies of the test functions are outside the repository.

- **Claim**: `sinc-bin-far-envelope` — `montecarlo/spectrum/lines/_bin_quadrature.py::_host_bin_mean`, `::_far_envelope_float32`, `SINCSQ_BIN_PREAMBLE::pyrite_sincsq_far_mean`, `::pyrite_sincsq_bin_mean_hybrid`, `_BIN_REDUCE_SOURCE`, `BIN_MEAN_EXACT_WIDTHS` — $\operatorname{sinc}^2x=\tfrac12(\pi x)^{-2}-\cos(2\pi x)/(2\pi^2x^2)$ plus integration by parts
- **Filters**: units pass; limits pass; signs/conventions pass (see the table)
- **Re-derivation**: `matches` — envelope mean, product form, telescoping, and both bounds, which are valid upper bounds. The offset now uses the float32 head/remainder split of both the edge and the resonance. Every route uses one resonance value for classification and for near bins
- **Verdict**: `rederived`
- **Write-up**: `docs/validation/radiation-physics/sinc-bin-far-envelope.md`
- **Suggested ledger change**: `unverified` → `rederived`. In Notes, replace "re-verification pending" with "re-verified 2026-09-28 (fresh context): the split restores the bound through $E_{\rm res}/w=10^{13}$". Never `signed-off` from this context.

### Independent derivation

Take $x=(E-E_{\mathrm{res}})/w$ with peak $1$. The line mass is $w$. On a bin $[a,b]$ with $ab>0$, the Jacobian cancels in the mean, so

$$
\bar S^{\mathrm{far}}=\frac{1}{b-a}\int_a^b\frac{\mathrm{d}x}{2\pi^2x^2}=\frac{1}{2\pi^2ab}.
$$

The product form has no subtraction, so its float32 error is a few ulp relative to itself. Contiguous far bins on one side share edges and telescope to $\frac{1}{2\pi^2}(1/X-1/Y)$. The exact means also telescope, so the yield change on that side is exactly $w\int_X^Y\cos(2\pi x)/(2\pi^2x^2)\,\mathrm{d}x$. One integration by parts gives

$$
\left\lvert\int_X^Y\frac{\cos2\pi x}{x^2}\,\mathrm{d}x\right\rvert
\le\frac{1}{2\pi X^2}+\left[\frac{1}{2\pi Y^2}+\frac{1}{2\pi}\left(\frac{1}{X^2}-\frac{1}{Y^2}\right)\right]
=\frac{1}{\pi X^2}.
$$

The rigorous total over both sides is therefore $1/(\pi^3K^2)$. Because $I(X)\simeq-\sin(2\pi X)/(4\pi^3X^2)$ asymptotically, the attainable value is $1/(4\pi^3K^2)$ per side and $1/(2\pi^3K^2)$ for two sides of equal sign. The claimed $3/(2\pi^3K^2)$ is a valid upper bound. The page's step drops the $-1/(2\pi Y^2)$ from the remainder integral, which makes it loose. For any single bin, the mean of the omitted term cannot exceed its supremum $1/(2\pi^2K^2)$ on $\lvert x\rvert\ge K$. The claimed $1/(\pi^2K^2)$ therefore holds with a factor-2 margin.

**Float32 offset.** Write $\epsilon=\epsilon_h+\epsilon_c$ and $E_{\rm res}=E_h+E_c$. Here $\epsilon_h=\mathrm{fl}_{32}(\epsilon)$, and $\epsilon_c=\mathrm{fl}_{32}(\epsilon-\epsilon_h)$, where the inner difference is exact in FP64. $E_h$ and $E_c$ are defined the same way. The code evaluates $\mathrm{fl}\big(\mathrm{fl}(\epsilon_h-E_h)+\mathrm{fl}(\epsilon_c-E_c)\big)$:

- If $\epsilon_h$ and $E_h$ are within a factor 2, Sterbenz makes $\epsilon_h-E_h$ exact. Otherwise $\lvert\epsilon-E_{\rm res}\rvert\gtrsim E_{\rm res}/2$, and one rounding at the result's own scale is relative.
- $\lvert\epsilon_c\rvert,\lvert E_c\rvert\le\tfrac12\mathrm{ulp}_{32}(E_{\rm res})$. Their float32 rounding and their difference each err by about $2^{-24}\,\mathrm{ulp}_{32}(E_{\rm res})\approx2^{-48}E_{\rm res}$.
- The final add rounds once relative to the result.

With $d=\epsilon-E_{\rm res}$, the relative error is therefore about $2^{-24}\left(1+c\,2^{-24}E_{\rm res}/\lvert d\rvert\right)$ with $c=O(1)$. The absolute error in $x$ is at most about $2^{-48}E_{\rm res}/w$ widths. Before the fix, the resonance remainder was missing, and this error was $2^{-24}E_{\rm res}/w$ widths. The float32 `scale` and the float32 $a$ add a few more ulp relative. Near the threshold, $\lvert x\rvert\approx K$, the threshold moves by less than $10^{-4}K$ for $E_{\rm res}/w\le10^{12}$. Near bins are evaluated from the true FP64 edges, so a bin at the boundary that is misclassified changes the result only within the stated bounds, at the slightly smaller $K$.

### Filters

| filter | result |
| --- | --- |
| units | $x$ and the bin mean are dimensionless relative to the peak. The mass is $w\times$ a dimensionless number. Pass |
| $K\to\infty$ | `exact_widths=None` passes `np.inf` (`_bin_quadrature.py:593`, `:656`), so no bin is far. Pass |
| bin containing the resonance | $x_{\rm lo}<0<x_{\rm hi}$ fails both far tests. With the split, $x$ is accurate to about $2^{-48}E_{\rm res}/w$ widths, so misclassifying the core would need $E_{\rm res}/w\sim10^{16}$. Pass |
| sign | $x_{\rm lo}x_{\rm hi}>0$ on either side, and the envelope is positive. Pass |
| host/device operation order | `_far_envelope_float32` (`:329`) and `pyrite_sincsq_far_mean` (`:211-212`) both compute `scale*((head-e_head)+(carry-e_carry))`. The C sequence has no multiply–add pair, so FMA contraction cannot change it. Pass |

### Route consistency (one resonance value per route)

| route | classification resonance | near-bin resonance | consistent |
| --- | --- | --- | --- |
| host `sincsq_bin_lineshape` → `_host_bin_mean` | FP64 $E_{\rm res}$, split (`:325-329`) | FP64 $E_{\rm res}$ (`:345`) | yes |
| CuPy matrix kernel, `REAL=float64` | `(double)e_r[row]`, split (`:225-228`) | same double (`:232`) | yes |
| CuPy matrix kernel, `REAL=float32` | `E_res_j` cast to float32 at `:600`, so $E_c=0$ | same float32 value | yes |
| fused reduction | float32 `e_r[line]`, remainder `0.0f` (`:485-486`) | `(double)e_r[q]` (`:520`) | yes |

On the two float32-input routes, the whole line, near and far, moves by $[\mathrm{fl}_{32}(E_{\rm res})-E_{\rm res}]/w$ widths. That is a position error of the `sinc-bin-integration` input precision, not a far-envelope yield error (finding 5).

### Reproduced numbers

Test suite: `tests/montecarlo/test_sinc_bin_integration.py -k "far_envelope or production"` gives 9 passed.

The first verifier's stress case, on the host route with $K=64$: $E_{\rm res}=1.6\times10^6+0.05$ eV, $w=0.01$ eV, spacing $w/2$, and a window from $-400$ to $+30$ widths. The bound is $1.18\times10^{-5}$.

| implementation | $\lvert\Delta Y/Y\rvert$ | worst pointwise / peak |
| --- | --- | --- |
| HEAD `c5bb3f1e` (`_host_bin_mean`) | $1.90\times10^{-6}$ | $7.75\times10^{-6}$ |
| HEAD via `sincsq_bin_lineshape` (CPU backend) | $1.90\times10^{-6}$ | — |
| scratch copy without the resonance remainder (pre-fix) | $5.37\times10^{-5}$ | $8.23\times10^{-6}$ |

The pre-fix value reproduces the first verifier's $5.37\times10^{-5}$.

Sweep: 60 lines per decade of $E_{\rm res}/w$, with $w$ log-uniform in $10^{-4}$–$30$ eV and spacing uniform in $0.2$–$3\,w$. Window placements were one-sided on the left, one-sided on the right, or symmetric, each with a random phase. $K=64$:

| $E_{\rm res}/w$ | worst $\lvert\Delta Y/Y\rvert$ | fraction of $3/(2\pi^3K^2)$ | worst pointwise / peak | fraction of $1/(\pi^2K^2)$ |
| --- | --- | --- | --- | --- |
| $10^{3}$ | $3.39\times10^{-6}$ | 0.29 | $1.09\times10^{-5}$ | 0.44 |
| $10^{5}$ | $3.67\times10^{-6}$ | 0.31 | $1.02\times10^{-5}$ | 0.41 |
| $10^{7}$ | $3.48\times10^{-6}$ | 0.29 | $1.10\times10^{-5}$ | 0.44 |
| $10^{8}$ | $3.64\times10^{-6}$ | 0.31 | $1.01\times10^{-5}$ | 0.41 |
| $10^{9}$ | $3.52\times10^{-6}$ | 0.30 | $1.09\times10^{-5}$ | 0.44 |
| $10^{10}$ | $3.65\times10^{-6}$ | 0.31 | $1.11\times10^{-5}$ | 0.45 |
| $10^{12}$ | $3.78\times10^{-6}$ | 0.32 | $1.11\times10^{-5}$ | 0.45 |
| $10^{13}$ | $3.62\times10^{-6}$ | 0.31 | $1.07\times10^{-5}$ | 0.43 |

With the window fixed and one-sided, removing the resonance remainder gives worst $\lvert\Delta Y/Y\rvert$ of $5.3\times10^{-6}$, $4.6\times10^{-5}$, and $3.1\times10^{-4}$ at $10^6$, $10^7$, and $10^8$. From $10^9$ upward it is about $1$, because the core is classified as far. With the fix, the same cases stay at $2.0\times10^{-6}$ throughout.

Offset accuracy, with exact rational reference and 40,000 random $(\epsilon, E_{\rm res}, w)$ pairs, $\lvert x\rvert\ge8$: the worst relative error of $(\epsilon_h-E_h)+(\epsilon_c-E_c)$ is at most $2\times2^{-24}$ for $E_{\rm res}/\lvert d\rvert\le10^6$. It rises to $6.8$, $58$, $222$, and $406\times2^{-24}$ at $10^7$, $10^8$, $10^9$, and $10^{10}$. This is the predicted $2^{-24}(1+c\,2^{-24}E_{\rm res}/\lvert d\rvert)$. Without the resonance remainder, the worst is $8.7\times10^9\times2^{-24}$.

Host-C bit identity: the preamble was compiled with `gcc -O2 -march=native` under both `-ffp-contract=off` and `=fast`. Over 20,000 random pairs (17,875 far, $E_{\rm res}\in[30,10^7]$ eV, $w\in[10^{-4},30]$ eV, $K\in\{16,64\}$), there were 0 classification and 0 value mismatches of `pyrite_sincsq_far_mean` against `_far_envelope_float32`. There were also 0 mismatches of `pyrite_sincsq_bin_mean_hybrid` against `_host_bin_mean` on far pairs.

The matrix kernel and the fused reduction were emulated per pair in host C on the stress axis. The yield error against the exact mean at the same resonance value was $1.90\times10^{-6}$ for the float64 matrix, $1.90\times10^{-6}$ for the float32 matrix, and $1.90\times10^{-6}$ for the fused reduction. For comparison against the FP64-resonance exact line, the float32 input moves the line by $-5.0$ widths and the windowed mass by $2.4\times10^{-4}$ (finding 5).

Float32 $x$ was monotone on 400 sorted random edge sets ($E_{\rm res}$ up to $3\times10^7$ eV, $w$ down to $10^{-4}$ eV). No side's far set was non-contiguous.

Mutation tests: scratch copies of `_far_envelope_float32` were monkeypatched under the test functions imported from a scratch copy of the test file.

| mutation | `mev[16]` | `mev[64]` | `core` | `near_exact` | `prod_default` |
| --- | --- | --- | --- | --- | --- |
| identity copy | pass | pass | pass | pass | pass |
| drop resonance remainder | **fail** | **fail** | **fail** | pass | pass |
| drop edge remainder | pass | pass | **fail** | pass | pass |
| drop both | **fail** | **fail** | pass | pass | pass |

Here `mev` is `test_far_envelope_bound_holds_at_mev_resonances_on_narrow_lines` and `core` is `test_far_envelope_never_claims_the_core_of_a_sub_mev_width_line`. Each single-remainder mutation is caught. Only `core` catches the dropped edge remainder.

### Findings

1. **Previous finding 1 is resolved.** `_bin_quadrature.py:211-212` (C) and `:325-329` (host) form $(\epsilon_h-E_h)+(\epsilon_c-E_c)$ in the same order. `pyrite_sincsq_bin_mean_hybrid` (`:225-228`) passes the FP64 resonance's remainder. The fused kernel (`:485-486`) passes `0.0f` and uses the same float32 `e_r` for near pairs (`:520`). The stress case falls from $5.37\times10^{-5}$ to $1.90\times10^{-6}$, and the bound holds through $E_{\rm res}/w=10^{13}$.
2. **"float32 relative accuracy at any absolute energy" is slightly overstated** (page line 51 and the preamble comment at `:199-202`). The two-term offset is accurate to about $2^{-24}(1+2^{-24}E_{\rm res}/\lvert d\rvert)$ relative, which is about $2^{-48}E_{\rm res}/w$ widths absolute. It is float32-relative only while $\lvert x\rvert\gtrsim2^{-24}E_{\rm res}/w$. This is harmless for the claim, because the classification threshold moves by less than $10^{-4}K$ for $E_{\rm res}/w\le10^{12}$. It is advisory wording only.
3. **The new regressions discriminate.** Both remainder mutations fail at least one test (table above). The dropped edge remainder is caught only by `core` (`tests/montecarlo/test_sinc_bin_integration.py:334-344`). The dropped resonance remainder is caught by all three new cases (`:315-344`). No test drives the C split at MeV: the CUDA tests (`tests/montecarlo/test_sinc_bin_integration_cuda.py:32`) draw $E_{\rm res}\in[900,1100]$ eV. A mutation in `pyrite_sincsq_bin_mean_hybrid` alone would therefore rely on the host-C bit-identity argument above. A MeV case in `test_hybrid_matrix_kernel_matches_the_host_hybrid` would pin it on the GPU. This is advisory.
4. **Host and C remain bit-identical** after the signature change (0 of 20,000 mismatches, with and without FP contraction). The order of `_far_envelope_float32` (`:329`) matches `pyrite_sincsq_far_mean` (`:206-219`).
5. **Out of scope for this claim: float32 resonance inputs move the line.** The float32 matrix route (`:600`) and the fused route (`:649`) round $E_{\rm res}$ before any use. At 1.6 MeV and $w=0.01$ eV this moves the entire line by up to $\mathrm{ulp}_{32}(E_{\rm res})/(2w)=6.25$ widths ($0.0625$ eV). The move is self-consistent, so the far-envelope yield bound holds. It is a position/precision property of the `sinc-bin-integration` float32 route. The `sinc-bin-integration` page should state it if MeV lines narrower than $\sim0.1$ eV are in scope.
6. **The bound looseness and pointwise margin are now stated** (page line 25): "sharp yield constant $1/(2\pi^3K^2)$", with a factor-2 pointwise margin. This agrees with this derivation and with the measured fractions (0.29–0.32 of the yield bound, which is about $0.9\times$ the sharp constant, and 0.41–0.45 of the pointwise bound). The page's integration-by-parts step (lines 40–42) still shows $3/(2\pi X^2)$ rather than the tighter $1/(\pi X^2)$. That is valid, but it is not the constant the text calls conservative. This is advisory.
7. **Cost prose (page line 61) mentions only the edge split.** It says "Each FP64 edge is split…". The Assumptions bullet (line 51) is the corrected statement. This is advisory.
8. **Ledger bookkeeping.** Row `sinc-bin-far-envelope` (`docs/validation/ledger-core-coherent-physics.md:171`) still reads `unverified`, although the first verification suggested `discrepancy`. With this re-check the suggested status is `rederived`. A human applies it, and only a human may later mark `signed-off`.
