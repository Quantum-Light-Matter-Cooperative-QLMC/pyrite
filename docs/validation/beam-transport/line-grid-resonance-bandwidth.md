# `line-grid-resonance-bandwidth`

Ledger row: [`line-grid-resonance-bandwidth`](../ledger-core-coherent-physics.md#line-grid-resonance-bandwidth). Status: unverified. Validation: `line-grid-resonance-bandwidth`.

## Source equation and selector

The incoherent line kernel evaluates each segment and reflection at its in-medium resonance $E_i$. Its density is $W_i\operatorname{sinc}^2[a_i(E-E_i)/\pi]$, with $w_i=\pi/a_i$ the first-zero width. The production coefficient is $W_i=\alpha\omega_i t_{L,i}^2 |A_i|^2 T_{\mathrm{abs},i}m_i/(4\pi^2\hbar c)$, including the mosaic quadrature weight $m_i$. This is the finite-time lineshape and absorption model used by both incoherent kernel routes. The whole-line mass is $M_i=W_iw_i$.

For an upper edge $S$, let $D_i=S-E_i$. Since $\sin^2 x\leq1$, the fraction of a line above the edge is bounded by

$$
q_i(S)\leq\begin{cases}
1,&D_i\leq0,\\
\min\{1,w_i/(\pi^2D_i)\},&D_i>0.
\end{cases}
$$

The selector finds the smallest $S$ with $\sum_i M_iq_i(S)/\sum_iM_i\leq\epsilon/2$, rounds it up to 100 eV, and caps it at the closed-form kinematic ceiling. Here $\epsilon=10^{-4}$ is the assigned upper-edge share. Characteristic lines supply a separate Lorentzian edge; the larger edge wins. A later spectrum audit applies the same bound with $\epsilon$ to the lines inside the spectrum pass's keep window, $E_i<S+0.2(S-\text{start})$, and refuses a case that exceeds it. Lines resonating beyond that window never reach the audit; the selector already counts them as wholly lost with the same production weights, so the end-to-end loss stays within $\epsilon/2$ from the selector alone. When $S$ is already the kinematic ceiling, no line resonates above the edge and the automatic bandwidth drops the same tails, so the audit records `capped_at_ceiling` and raises `LineGridTruncationWarning` instead.

The two-node collection pass uses the same in-medium roots, line filters, amplitude, and escape transmission as the spectrum pass. It selects the edge after electron transport and before allocating the final line axis. The closed-form ceiling remains the hard maximum.

## Assumptions and limit

The coefficient is treated as constant across one sinc line because the kernel evaluates it at $E_i$. The bound covers the upper tail of the incoherent PXR/CBS line model. It does not certify the lower edge, coherent interference, or nonuniform quadrature. For one line below an uncapped edge, the unrounded solution is $S=E_i+w_i/(\pi^2\epsilon/2)$.

The same audit sums $M_i$ per emitting electron and records the relative standard error of the mean per-electron line mass with the largest single-electron share. Above `LINE_YIELD_RELATIVE_SE_LIMIT` (0.1) the case still runs, but it is flagged `statistics_limited` and raises `LineYieldStatisticsWarning`. This is a statistical diagnostic, not part of the bound: at 5 MeV a rare electron scattered into the detector's $1/\gamma$ cone can carry most of a case's line mass (#201), and its wide, heavy lines also drive $S$. The error estimate only describes the sampled electrons, so a run that never sampled such an electron passes with a biased yield.

Focused anchors: `tests/energy-grid/test_line_grid_bandwidth.py` covers the tail inequality, production collection, selector plumbing, ceiling cap, and a two-electron h-BN transport case. Production-size peak memory, remote yield, shape, and detected-count comparisons remain outstanding. Human sign-off pending.

## Independent verification (fresh context, 2026-09-27)

Verifier: fresh-context agent that did not write the implementation. It derived the bounds below from the ledger row and this page's source statements before reading the implementation bodies. Suggested status: `discrepancy`, limited to the audit's coverage (finding A). The bound and selector mathematics match. The human adjudicates.

### Re-derivation of the $\operatorname{sinc}^2$ tail

With $\operatorname{sinc}u=\sin(\pi u)/(\pi u)$, the line $W\operatorname{sinc}^2[a(E-E_i)/\pi]=W\sin^2(ax)/(ax)^2$, with $x=E-E_i$. Its first zero is at $ax=\pi$, so $w=\pi/a$. Since $\int_{-\infty}^{\infty}\sin^2u/u^2\,du=\pi$, the whole mass is $M=W\pi/a=Ww$. For $D>0$,

$$
\int_D^\infty\frac{\sin^2(ax)}{(ax)^2}\,dx\leq\int_D^\infty\frac{dx}{a^2x^2}=\frac{1}{a^2D},
\qquad
q(D)\leq\frac{1/(a^2D)}{\pi/a}=\frac{1}{\pi aD}=\frac{w}{\pi^2D}.
$$

The exact one-sided fraction, with $X=aD$, is

$$
q(D)=\frac{1}{\pi}\left[\frac{\sin^2X}{X}+\frac{\pi}{2}-\operatorname{Si}(2X)\right].
$$

The bound is strict for every finite $D>0$, because $\sin^2<1$ almost everywhere. The ratio of bound to exact tail has a minimum of $1.483$ near $X\approx0.96$. It oscillates around 2 and tends to 2 as $X\to\infty$, because $\sin^2$ averages $\tfrac12$. At the operating point $\epsilon/2=5\times10^{-5}$, a single line has $X=1/(\pi\epsilon/2)\approx6.4\times10^3$ and a ratio of $2.000$. The cap $\min\{1,\cdot\}$ is valid. The tightest cap for $D>0$ would be $\tfrac12$, so capping at 1 is only more conservative. Counting $D\leq0$ wholly is also conservative.

### Re-derivation of the Lorentzian bound

For a Lorentzian of full width at half maximum $\Gamma$, the mass fraction above $D>0$ is

$$
\frac12-\frac1\pi\arctan\frac{2D}{\Gamma}=\frac1\pi\arctan\frac{\Gamma}{2D}\leq\frac{\Gamma}{2\pi D},
$$

because $\arctan y\leq y$ for $y\geq0$. The ratio tends to 1 as $D/\Gamma\to\infty$. The bound applies to each line, so it holds for any mixture of line yields. Solving it for $\epsilon$ gives the edge $E_c+\Gamma/(2\pi\epsilon)$.

### Limiting case

For one line, $w/(\pi^2D)=\epsilon/2$ gives $S=E_i+w/(\pi^2\epsilon/2)=E_i+2w/(\pi^2\epsilon)$ before rounding. This matches the page. The implementation docstring states the same result in terms of its argument `truncation_limit`, which is $\epsilon/2$.

### Comparison with the implementation

- `sincsq_upper_tail_bound` computes $\min\{1,w/(\pi^2D)\}$ and returns 1 for $D\leq0$. `_accumulate_edge_truncation` computes $1/(\pi a D)$ from the kernel's `a_width`, which is $a$. These are the same expression because $w=\pi/a$. The collect branch stores `pi / a_width`, the first-zero width, and $M=W\pi/a$ is used on both sides.
- The kernel width is $a=(\text{denom})\,t_L/(2\hbar c)$ (`_batched.py`, `_per_hkl.py`). Both use $t_L$ in Å, so $w=2\pi\hbar c/(\text{denom}\,t_L)$. That is the formula `resonance_populations` uses. The coefficient is $\alpha\,\omega\,t_L^2\,T_{\rm abs}\lvert A\rvert^2m/(4\pi^2\hbar c)$, with $\omega$ in $\text{Å}^{-1}$, $t_L$ in Å and $\hbar c$ in eV Å, so $W$ has units of $\text{eV}^{-1}$ and $M=Ww$ is dimensionless. The coefficient is evaluated at $E_i$, so the truncated mass is exactly $W$ times the tail integral.
- In `case_line_stop_eV`, the selector target is $\epsilon/\text{proxy\_safety}=10^{-4}/2$. The larger of the resonance edge, the characteristic edge and $\text{start}+100$ is rounded up to 100 eV and then capped at the ceiling.
- Bisection: $q_i(S)$ does not increase with $S$ for any line, so the sum does not either. `low` is the smallest resonance, where every line counts wholly and the loss is 1, which exceeds the limit. `high` is the ceiling, and the search runs only when the loss at the ceiling is within the limit. The invariant therefore holds. Sixty halvings reach the float64 resolution of the axis. A population that misses the limit at the ceiling returns the ceiling and is flagged `capped`.
- Audit: the upper fraction above `truncation_limit` $=\epsilon$ raises `LineGridToleranceError` unless the grid's last node is at or above the ceiling. At the ceiling, the audit records `capped_at_ceiling` and raises `LineGridTruncationWarning`. The lower fraction is only recorded. `kinematic-ceiling` and `resonance-population` share the same payload `stop_eV`, which is the kinematic ceiling (`campaign/sweep.py::_automatic_line_grid_policy`). The claim that the automatic bandwidth drops the same upper tails therefore holds. In-medium $n<1$ only increases the denominator, so no line resonates above the vacuum ceiling. Coherent emission is refused before selection (`_refuse_coherent_resolution`), so the audit sees only the incoherent route.
- Statistics: $\mathrm{RSE}=s\sqrt n/\sum_e M_e=s/(\bar M\sqrt n)$, the relative standard error of the mean per-electron mass. Electrons with zero mass count because the `bincount` uses `minlength=n_electrons`. The page describes this accurately as a diagnostic that only warns.

### Numeric checks

These scripts ran on the CPU backend with `uv run --no-sync python`.

- Tail: at $X=aD\in\{0.05,0.1,0.5,1,\pi,10,10^2,10^4\}$, the closed form matches `scipy.integrate.quad` to $10^{-6}$ relative. The bound-to-exact ratios are 13.2, 6.80, 1.84, 1.48, 2.09, 1.92, 2.009 and 2.000. The minimum over $X\in[10^{-3},10^5]$ is 1.4835. The whole mass, 8.49080, equals $\pi/a=8.49079$.
- Lorentzian: at $D/\Gamma=0.0435$, $0.435$, $4.35$ and $435$, the bound-to-exact ratios are 7.75, 1.34, 1.004 and 1.0000.
- Single line with $E_i=12000$ eV, $w=7.3$ eV and limit $5\times10^{-5}$, rounding disabled: `resonance_population_stop_eV` returns 26792.8928118, against the closed form 26792.8928118. Rounded, it returns 26800 with proxy fraction $4.9976\times10^{-5}$. With a 20 keV ceiling it returns 20000, `capped=True`.
- Random 20 000-line mixture: stop 271900. The loss at `stop` is $4.99867\times10^{-5}$, within the limit, and at `stop - 100` it is $5.00057\times10^{-5}$, over it. The rounded stop is therefore the smallest acceptable 100 eV multiple. The loss is non-increasing over 4000 trial edges. On the same lines, `_accumulate_edge_truncation` gives an upper fraction equal to the selector's to 16 digits and the same total mass.
- `pyrite-dev test tests/energy-grid/test_line_grid_bandwidth.py` passed 22 of 22.

### Finding A (discrepancy): the audit cannot see lines the final axis's keep window drops

Both incoherent routes remove a line before `_accumulate_edge_truncation` unless $E_{\rm lo}-p<E_i<E_{\rm hi}+p$, with $p=0.2(E_{\rm hi}-E_{\rm lo})$ (`_batched.py:321-324`, `:455`; `_per_hkl.py:485-492`). In the collection pass, $E_{\rm hi}$ is the ceiling $C$, so every physical line is collected. In the final pass, $E_{\rm hi}=S$. A line with $S+0.2(S-\text{start})\leq E_i\leq C$ is counted wholly by the selector ($q=1$). It is lost from the spectrum, but it is absent from both `line_mass` and `mass_above` in the audit. The audit's upper fraction is therefore a bound for kept lines only, not for the population the selector bounded. This page's statement that the later audit "applies the same bound with $\epsilon$ and refuses a case that exceeds it" does not hold for those lines.

Example: 5000 lines near 10 keV, plus one line at 300 keV holding $4\times10^{-5}$ of the mass. The selector gives $S=30400$ with proxy fraction $4.9988\times10^{-5}$. The final-axis keep boundary is 36470 eV, so the far line is dropped. The audit reports $9.99\times10^{-6}$ with the final-axis keep filter, against $4.9988\times10^{-5}$ with all lines. The true end-to-end loss is still within $\epsilon/2$, because the selector counted the far line and both passes use the same production weights. The gap is in the audit's role as an independent safety net: it cannot detect a collection pass and final pass that disagree about lines far above $S$. The two passes do use different tabulation windows for $\mu$, $n$ and $\chi_g$ (`_setup.py:398-401`). That difference is expected to be negligible against the factor-2 margin, but the audit is the check that should catch it.

Fix options, not applied: (a) in the final pass, count lines with $E_i\geq E_{\rm hi}+p$ wholly in `mass_above` and `line_mass` before the keep mask drops them; or (b) reword this page and the ledger row to say that the audit re-evaluates the bound over lines inside the final keep window, and that the selector alone accounts for lines resonating above $1.2S-0.2\,\text{start}$.

### Minor notes (no status impact)

- A characteristic edge above the ceiling is capped silently, and characteristic lines are not in the audit. This matches the automatic policy, so it does not contradict the capped-at-ceiling justification.
- The "2x the exact far tail" note in the ledger is asymptotic. Near $aD\approx1$ the ratio falls to 1.48, and it never falls below 1.
