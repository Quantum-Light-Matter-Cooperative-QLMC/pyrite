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

The selector finds the smallest $S$ with $\sum_i M_iq_i(S)/\sum_iM_i\leq\epsilon/2$, rounds it up to 100 eV, and caps it at the closed-form kinematic ceiling. Here $\epsilon=10^{-4}$ is the assigned upper-edge share. Characteristic lines supply a separate Lorentzian edge; the larger edge wins. A later spectrum audit applies the same bound with $\epsilon$ and refuses a case that exceeds it. When $S$ is already the kinematic ceiling, no line resonates above the edge and the automatic bandwidth drops the same tails, so the audit records `capped_at_ceiling` and raises `LineGridTruncationWarning` instead.

The two-node collection pass uses the same in-medium roots, line filters, amplitude, and escape transmission as the spectrum pass. It selects the edge after electron transport and before allocating the final line axis. The closed-form ceiling remains the hard maximum.

## Assumptions and limit

The coefficient is treated as constant across one sinc line because the kernel evaluates it at $E_i$. The bound covers the upper tail of the incoherent PXR/CBS line model. It does not certify the lower edge, coherent interference, or nonuniform quadrature. For one line below an uncapped edge, the unrounded solution is $S=E_i+w_i/(\pi^2\epsilon/2)$.

The same audit sums $M_i$ per emitting electron and records the relative standard error of the mean per-electron line mass with the largest single-electron share. Above `LINE_YIELD_RELATIVE_SE_LIMIT` (0.1) the case still runs, but it is flagged `statistics_limited` and raises `LineYieldStatisticsWarning`. This is a statistical diagnostic, not part of the bound: at 5 MeV a rare electron scattered into the detector's $1/\gamma$ cone can carry most of a case's line mass (#201), and its wide, heavy lines also drive $S$. The error estimate only describes the sampled electrons, so a run that never sampled such an electron passes with a biased yield.

Focused anchors: `tests/energy-grid/test_line_grid_bandwidth.py` covers the tail inequality, production collection, selector plumbing, ceiling cap, and a two-electron h-BN transport case. Production-size peak memory, remote yield, shape, and detected-count comparisons remain outstanding. Human sign-off pending.
