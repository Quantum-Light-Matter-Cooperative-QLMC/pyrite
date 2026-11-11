# `line-grid-resonance-bandwidth`

Ledger row: [`line-grid-resonance-bandwidth`](../ledger-core-coherent-physics.md#line-grid-resonance-bandwidth). Status: discrepancy. Validation: `line-grid-resonance-bandwidth`.

## Source equation and selector

The incoherent line kernel evaluates each segment and reflection at its in-medium resonance $E_i$. Its density is $W_i\operatorname{sinc}^2[a_i(E-E_i)/\pi]$, with $w_i=\pi/a_i$ the first-zero width. The production coefficient is $W_i=\alpha\omega_i t_{L,i}^2 |A_i|^2 T_{\mathrm{abs},i}m_i/(4\pi^2\hbar c)$, including the mosaic quadrature weight $m_i$. This is the finite-time lineshape and absorption model used by both incoherent kernel routes. The whole-line mass is $M_i=W_iw_i$.

For an upper edge $S$, let $D_i=S-E_i$. Since $\sin^2 x\leq1$, the fraction of a line above the edge is bounded by

$$
q_i(S)\leq\begin{cases}
1,&D_i\leq0,\\
\min\{1,w_i/(\pi^2D_i)\},&D_i>0.
\end{cases}
$$

The selector finds the smallest $S$ with $\sum_i M_iq_i(S)/\sum_iM_i\leq\epsilon/2$, rounds it up to 100 eV, and caps it at the closed-form kinematic ceiling. Here $\epsilon=10^{-4}$ is the assigned upper-edge share. Characteristic lines supply a separate Lorentzian edge; the larger edge wins. A later spectrum audit applies the same bound with $\epsilon$ to the lines inside the spectrum pass's keep window, $E_i<S+0.2(S-\text{start})$, and refuses a case that exceeds it. Lines resonating beyond that window never reach the audit; the selector already counts them as wholly lost with the same production weights, so the end-to-end loss stays within $\epsilon/2$ from the selector alone. When $S$ is already the kinematic ceiling, no line resonates above the edge and the automatic bandwidth drops the same tails, so the audit records `capped_at_ceiling` and raises `CxrGridTruncationWarning` instead.

The two-node collection pass uses the same in-medium roots, line filters, amplitude, and escape transmission as the spectrum pass. It selects the edge after electron transport and before allocating the final line axis. The closed-form ceiling remains the hard maximum.

## Assumptions and limit

The coefficient is treated as constant across one sinc line because the kernel evaluates it at $E_i$. The bound covers the upper tail of the incoherent PXR/CBS line model. It does not certify the lower edge, coherent interference, or nonuniform quadrature. For one line below an uncapped edge, the unrounded solution is $S=E_i+w_i/(\pi^2\epsilon/2)$.

The same audit sums $M_i$ per emitting electron and records the relative standard error of the mean per-electron line mass with the largest single-electron share. Above `CXR_YIELD_RELATIVE_SE_LIMIT` (0.1) the case still runs, but it is flagged `statistics_limited` and raises `CxrYieldStatisticsWarning`. This is a statistical diagnostic, not part of the bound: at 5 MeV a rare electron scattered into the detector's $1/\gamma$ cone can carry most of a case's line mass (#201), and its wide, heavy lines also drive $S$. The error estimate only describes the sampled electrons, so a run that never sampled such an electron passes with a biased yield.

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
- In `case_cxr_stop_eV`, the selector target is $\epsilon/\text{proxy\_safety}=10^{-4}/2$. The larger of the resonance edge, the characteristic edge and $\text{start}+100$ is rounded up to 100 eV and then capped at the ceiling.
- Bisection: $q_i(S)$ does not increase with $S$ for any line, so the sum does not either. `low` is the smallest resonance, where every line counts wholly and the loss is 1, which exceeds the limit. `high` is the ceiling, and the search runs only when the loss at the ceiling is within the limit. The invariant therefore holds. Sixty halvings reach the float64 resolution of the axis. A population that misses the limit at the ceiling returns the ceiling and is flagged `capped`.
- Audit: the upper fraction above `truncation_limit` $=\epsilon$ raises `CxrGridToleranceError` unless the grid's last node is at or above the ceiling. At the ceiling, the audit records `capped_at_ceiling` and raises `CxrGridTruncationWarning`. The lower fraction is only recorded. `kinematic-ceiling` and `resonance-population` share the same payload `stop_eV`, which is the kinematic ceiling (`campaign/sweep.py::_automatic_cxr_grid_policy`). The claim that the automatic bandwidth drops the same upper tails therefore holds. In-medium $n<1$ only increases the denominator, so no line resonates above the vacuum ceiling. Coherent emission is refused before selection (`_refuse_coherent_resolution`), so the audit sees only the incoherent route.
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

## Independent verification (2026-11-11, issue #1101)

### Source-first derivation and independence record

This fresh verifier read the target/source ledger records and AST-extracted
signatures/docstrings first. The derivation below was recorded separately in
`/tmp/resonance-independent-derivation.md` and appended here before reading
implementation bodies or the pre-existing write-up. The mathematical source
is the elementary finite-time Fourier integral underlying
`finite-time-lineshape`; the Lorentzian is the normalized profile identified
by `characteristic-radiation`. This verifies the tail policy conditional on
those source profiles, rather than revalidating their radiation amplitudes.

For a constant coefficient $W_i\geq0$, let

$$
f_i(E)=W_i\left[\frac{\sin(a_i(E-E_i))}{a_i(E-E_i)}\right]^2,
\qquad w_i=\frac{\pi}{a_i}>0.
$$

Indeed the rectangular-time amplitude is
$\int_{-T/2}^{T/2}e^{iqt}\,dt=T\sin(qT/2)/(qT/2)$.
Changing variable $u=a_i(E-E_i)$ and using Parseval gives
$\int_{-\infty}^{\infty}f_i(E)\,dE=W_i\pi/a_i=W_iw_i$.
The whole-real-line mass is the normalization; it is not the mass conditioned
on positive energies or on the retained interval. For $D>0$,

$$
\int_D^\infty\frac{\sin^2(a_i x)}{a_i^2x^2}\,dx
\leq\frac{1}{a_i^2D},\qquad
B_i(D)=\min\left(1,\frac{w_i}{\pi^2D}\right).
$$

For $D\leq0$, use $B_i=1$. For an upper stop $S$, the conservative
production-weight fraction is therefore

$$
F_+(S)=\frac{\sum_i W_iw_i B_i(S-E_i)}{\sum_i W_iw_i}.
$$

The lower-edge expression replaces $S-E_i$ by $E_i-A$ for start $A$.
A resonance discarded by a production keep window must be charged its entire
$W_iw_i$ by the selector; an audit restricted to retained lines alone cannot
certify losses of discarded lines. Both incoherent implementations must
freeze amplitudes, resonance prefactors and escape at $E_i$ for this
factorization to hold. A varying coefficient under the energy integral would
invalidate the simple fraction. Reflection, mosaic, layer and electron
weights must be the same nonnegative production weights in selection and
scoring, including all production finite/keep/cutoff masks.

For normalized Lorentzian FWHM $\Gamma>0$,

$$
L(E)=\frac{1}{\pi}\frac{\Gamma/2}{(E-E_0)^2+(\Gamma/2)^2},\qquad
\int_{E_0+D}^{\infty}L(E)\,dE
=\frac12-\frac1\pi\arctan\frac{2D}{\Gamma}
=\frac1\pi\arctan\frac{\Gamma}{2D}
\leq\frac{\Gamma}{2\pi D}.
$$

Consequently $S\geq\max_j(E_j+\Gamma_j/(2\pi\epsilon))$ bounds every
emitted characteristic line's upper tail by $\epsilon$, hence any
nonnegative mixture. The selected line set must equal the production line
set at the same relaxation cutoff. Zero width limits to a delta line.

For sinc lines the requested selector budget is $\epsilon/s$ with safety
factor $s=2$. For one line, before cap and rounding,

$$
S=E_0+\frac{s w}{\pi^2\epsilon}.
$$

The conservative tail is asymptotically twice the exact tail because
$\sin^2$ averages to $1/2$. All fractions are dimensionless: $a_i$ has units
$\mathrm{eV}^{-1}$, $w_i,D,\Gamma_i$ have units eV. Fractions are nonnegative,
monotone decreasing with the stop, tend to zero for infinite stop, and are
invariant under a common scaling of $W_i$. Empty/zero-mass populations cannot
set a physical stop. $\Gamma\to0$ leaves the highest characteristic centre.

Let $C$ denote the policy-supplied ceiling and $r=100\ \mathrm{eV}$ its
rounding step. Outward rounding uses $r\lceil S/r\rceil$, then the final
stop is capped at $C$ and must exceed start by at least $r$ when feasible.
The cap may destroy the requested tail bound: finite-time profiles extend
above every finite resonance ceiling. Such saturation must be identified
and warned about, rather than represented as an unconditional truncation
certificate. Above-budget uncapped stops should be refused by the audit;
the lower edge is reported but has no upper-edge budget allocation.
The independent `line-grid-kinematic-bandwidth` discrepancy, including
arbitrary energy-spread beams, remains outside this conditional ceiling
contract.

### Implementation comparison and filter results

The elementary mathematical subclaim matches. `sincsq_upper_tail_bound`
uses $w/(\pi^2D)$; `_accumulate_edge_truncation` uses
$1/(\pi aD)$, identically, and sums $W\pi/a$. No factor of two or missing
energy Jacobian occurs. Both `_per_hkl.py`'s independent-segment branch and
`_batched.py::_batched_incoherent_block` freeze their coefficients at the
in-medium resonance: couplings, prefactor and escape use that energy, while
only the sinc profile varies over output energy. The width is
$a=(\mathrm{denom})t_L/(2\hbar c)$ with $t_L$ in Å, so $a$ is in
$\mathrm{eV}^{-1}$, $W$ in $\mathrm{eV}^{-1}\mathrm{sr}^{-1}$ and $Ww$ in
$\mathrm{sr}^{-1}$ per sampled electron after the common normalization.
The production angular and electron normalizations cancel in the fraction.

For those two branches, collection and deposition see the same survivors:
line-electron cutoff, in-medium root, $E_i>10\ \mathrm{eV}$, padded energy
keep window, then finite positive total weight. Batched mosaic rows carry
`WM`; scalar orientations carry `wm`. The same post-mask arrays are audited
and reduced, including the batched queued CUDA path. The production collector
`_direction_populations` calls `_cxr_for_segments`, so layers, layer-specific
reflection sets/orientations, absorption geometry, electron limits and
production routing are read through the actual scorer. It does not use the
older $t_L^2$ proxy returned by `case_resonance_populations`.

`resonance_population_stop_eV` forms $M_i=W_iw_i$, drops nonfinite or
nonpositive mass, bisects the monotone bound, rounds outwards and caps it.
`case_cxr_stop_eV` uses $\epsilon/2$ for the sinc selector and $\epsilon$ for
the per-characteristic-line bound; its larger edge, with a feasible
start-plus-one-step floor, is rounded outwards and capped. Multiple detector
directions take the maximum selected stop. Policy resolution uses
$\epsilon=10^{-4}$, safety factor 2, and refuses feature windows with this
bandwidth because they spend the same budget allocation.

`characteristic_stop_eV` and `characteristic_line_seeds` select the same
positive-width, above-relaxation-cutoff, nonzero-yield transitions from the
same cross-section loader and per-composition maximum recommended cutoff.
At that default cutoff, production `_prepare_lines` recomputes the same
cascade response and applies the same energy/nonzero-response tests; a line
with zero captured bin mass may additionally be omitted from scoring, which
cannot increase the tail requirement. Characteristic bin profiles retain
unconditioned Lorentzian mass and extend to their outer bin edges; using the
last bin centre as the requested stop is conservative. The final cap can
invalidate the per-characteristic-line target, and these lines are not part
of the sinc audit. This is a conditional, capped policy, not a universal
certificate of characteristic truncation.

`check_cxr_truncation` gates only the upper edge. It refuses above-budget
uncapped sinc audits, warns at the supplied cap and reports the lower edge
without gating it. The cap comparison is correct as a policy comparison;
the historical assertions above that no resonance can exceed it, and that
in-medium refraction always increases its denominator, are not certified by
this verification. They must be read subject to the separate
`line-grid-kinematic-bandwidth` discrepancy. This review does not assert a
physical cap for arbitrary energy-spread beams.

The historical Finding A is accurately narrowed in the current ledger:
the final audit covers retained lines; the selector charges lines beyond
the final keep window wholly. Under identical production coefficients, the
selector bound covers those losses. Reading the old section after recording
the independent derivation did not reveal a new factor error in that rule.
Different tabulation windows between collection and final scoring remain a
numerical qualification, not an exact identity of the two computed weights.

| Filter | Result |
| --- | --- |
| Units and normalization | Pass for the sinc and Lorentzian tail formulas; whole-line, unconditioned mass |
| One-line, empty population, infinite stop and zero-width limits | Pass under the stated positive-width model and cap exception |
| Signs and inequality direction | Pass; conservative positive upper bounds |
| Reference transcription | No new tabulated values; existing material/cascade data are consumed, not independently recertified |
| Tolerances | $10^{-4}$ is a declared policy budget, not a fitted source uncertainty; quadrature tolerances below are independent numerical checks |
| Existing remote/parity evidence | No remote accuracy or GPU parity claimed; old remote command is a recipe, not a measured artifact |
| Every admitted incoherent population is collected and audited | **Fail for flight-grouped numerical substeps**, detailed below |

### Current discrepancy: flight-grouped emission bypasses collection and audit

The first divergent mathematical term is the inter-substep cross term. A
physical flight assembled from numerical substeps emits

$$
\left|\sum_j c_j F_j(E)e^{i\phi_j(E)}\right|^2
=\sum_j|c_j F_j(E)|^2
+2\operatorname{Re}\sum_{j<k}c_jc_k^*F_j(E)F_k(E)^*
 e^{i(\phi_j(E)-\phi_k(E))}.
$$

The independent-positive-sinc mixture has no such cross term and therefore
cannot be applied to this additional incoherent-between-flights route
without another bound or a refusal. This is distinct from either of the two
independent-segment implementations, whose tail factors agree exactly.

The control-flow mismatch is precise. `_prepare_spectrum` detects repeated
physical-flight ids and sets `st.grouped`. `_needs_per_hkl_route` routes those
rows to `_per_hkl.py`. Its `if gid_all is not None` branch sums substep fields
per physical flight, deposits their squared modulus, then returns **before**
the ordinary sinc branch calls `_accumulate_edge_truncation`.
`_mc_spectrum` refuses grouped rows only when
`(request.truncation_audit or {}).get("mass_only")` is true. The measured
bandwidth collector instead supplies `{"collect": []}`; it is admitted and
silently collects nothing. A regular edge audit is admitted too and has no
line-mass entries. `check_cxr_truncation` maps the resulting zero total to
zero upper/lower fractions. Thus a nonzero radiating population can be
reported as empty and its truncation is not checked. This directly fails the
ledger's every-deposited-population coverage requirement; it does not depend
on the separate kinematic-ceiling problem.

A bounded CPU reproduction used two 25 keV electrons in HOPG, thickness
4000 Å, seed 7, midpoint transport, then `subdivide_flights` with
`max_dE_frac=2e-4`, `max_substeps=256`. On its 401 rows,
`mc_cxr_spectrum` for (002), $B=0.8\ \mathrm{Å}^2$, take-off 119 degrees,
and 880–1068 eV in 2 eV steps returned a positive sum of spectral sample
values, $2.0140452115721133\times10^{-7}$, while `audit["collect"]`
remained empty. The production `_direction_populations` call on 50–100000 eV likewise
returned zero populations. A regular edge audit remained just its start/stop
fields; `check_cxr_truncation` returned upper and lower fractions of zero,
with `capped_at_ceiling=False`. Scratch reproduction:
`/tmp/resonance-grouped-probe.py`; this is local diagnostic evidence, not a
committed regression anchor. Transport warned that BremsLib tables were
unavailable and used uncoupled radiative scoring; no bremsstrahlung result
is used in the finding.

Suggested owner fix: refuse measured bandwidth for flight-grouped substeps
before collection, or supply a flight-field tail certificate and collect/audit
that quantity. Merely treating the substeps as independent sinc lines would
omit the displayed cross term. No implementation change was made here.

### Independent numerical checks and adjudication

Scratch script `/tmp/resonance-verify.py` used elementary formulas and
independent SciPy quadrature, rather than the production bin-integral helper:

- Seven sinc reference distances $aD=0.01,0.05,0.2,1,\pi,10,100$ passed.
  Finite quadrature plus the analytic positive remainder
  $1/(\pi U)$ enclosed the sine-integral expression at every point; maximum
  enclosure width was $1.59155\times10^{-5}$. Largest exact-tail/bound ratio
  among those points was 0.673457, below one.
- Five Lorentzian distances $D/\Gamma=0.01,0.1,1,10,1000$ matched independent
  infinite-range quadrature to absolute $10^{-10}$ and obeyed the bound.
- A single line at 12000 eV with first-zero width 7.3 eV and target
  $5\times10^{-5}$ returned 26800 eV, matching the independently rounded
  closed form.
- A seeded 2000-line mixture returned 27300 eV with bound
  $4.924380435439394\times10^{-5}$; the previous 100 eV step had
  $5.037276685750428\times10^{-5}$. The independently summed fraction and
  `_accumulate_edge_truncation` agreed to the displayed precision.
- `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
  tests/energy-grid/test_line_grid_bandwidth.py`: **26 passed**, 26 warnings,
  45.14 s. The warnings were the existing kinematic/thick-target screens.
  These passing anchors do not cover the flight-grouped failure above.

**Verdict: discrepancy**, limited to admitted flight-grouped population and
audit coverage. The sinc/Lorentzian mathematics, factors, dimensions,
production weighting in the two independent-segment routes, policy safety
allocation, and rounding/cap mechanics are independently rederived.
Suggested ledger change: set `line-grid-resonance-bandwidth` to `discrepancy`
and identify the grouped `collect`/ordinary-audit bypass, retaining this
mathematical subclaim and the separate ceiling qualification. Only the task
owner edits the ledger; human sign-off remains separate. Documentation tests,
HTML build and positive math-render inspection are run centrally by the task
owner after integrating this write-up.

### Post-verification remediation (implementation context)

Applied after this verdict by the #1101 task owner; not yet re-verified in
fresh context. `_mc_spectrum` now refuses **every** truncation audit on
flight-grouped rows (`collect`, edge, `mass_only`, adaptive per-electron
masses), not only `mass_only`. Measured bandwidth, line roulette and adaptive
CXR sizing with `max_dE_frac > 0` therefore fail loudly instead of reporting
zero mass. The automatic bandwidth policy, which takes no audit, is
unaffected. No flight-field tail certificate was derived. Anchor:
`tests/montecarlo/spectrum/test_spectrum_phases.py::test_truncation_audit_refuses_flight_grouped_substeps`.

## Independent re-verification (fresh context, 2026-11-11, post-remediation)

This is a fresh-context verifier at commit `e45ebefbd`. Before reading any
implementation body or the sections above, it read the ledger row's Claim,
Code, Source, Checks and Anchor, and the signatures and docstrings of
`sincsq_upper_tail_bound`, `resonance_population_stop_eV`,
`characteristic_stop_eV`, `case_cxr_stop_eV`, `_accumulate_edge_truncation`,
`check_cxr_truncation` and `_mc_spectrum`. It wrote the derivation below
first. Only the remediation commit's diff and this page's discrepancy section
were read afterwards. The dependence on `line-grid-kinematic-bandwidth` is
disclosed here but not adjudicated.

### Source-first derivation

**Line tail.** A finite-time line of energy-independent weight $W$ is
$W\,\operatorname{sinc}^2$ with $f(x)=\sin^2(ax)/(ax)^2$ and
$x=E-E_{\rm res}$. Its whole mass is $W\int_{-\infty}^{\infty}f\,dx=W\pi/a$.
The first zero lies at $w=\pi/a$, so the mass is $Ww$. Because
$\sin^2\le 1$, for $D>0$

$$
\int_D^\infty \frac{\sin^2(ax)}{(ax)^2}\,dx\le\int_D^\infty\frac{dx}{a^2x^2}
=\frac{1}{a^2D},
\qquad
\phi(D)\le\min\!\left(1,\frac{1/(a^2D)}{\pi/a}\right)
=\min\!\left(1,\frac{w}{\pi^2D}\right),
$$

with $\phi=1$ for $D\le 0$. The exact tail
$\left[\sin^2U/U+\pi/2-\operatorname{Si}(2U)\right]/\pi$, where $U=aD$,
tends to $1/(2\pi U)$. The bound is therefore asymptotically twice the true
tail and never below it.

**Lorentzian tail.** For a line
$L(x)=(\Gamma/2\pi)/(x^2+\Gamma^2/4)$, the tail is
$\int_D^\infty L\,dx=\tfrac12-\tfrac1\pi\arctan(2D/\Gamma)
=\tfrac1\pi\arctan(\Gamma/2D)\le\Gamma/(2\pi D)$, because $\arctan y\le y$
for $y\ge0$. Each line therefore keeps all but $\varepsilon$ of its mass
when $S\ge E_c+\Gamma/(2\pi\varepsilon)$. A per-line bound also holds for
any mixture of lines.

**Selector.** For lines $i$ with mass $m_i=W_iw_i$, the bounded lost
fraction is
$\Lambda(S)=\sum_i m_i\phi_i(S-E_i)/\sum_i m_i$, which does not increase
with $S$. The smallest admissible stop has $\Lambda(S)\le\varepsilon/2$.
Rounding it up only lowers $\Lambda$. Capping it at the ceiling
$S_{\max}$ takes $\min(S,S_{\max})$. The characteristic edge is
$\max_c\,[E_c+\Gamma_c/(2\pi\varepsilon)]$, and the larger edge wins. For one
line the stop is $E_{\rm res}+w/(\pi^2\varepsilon/2)$ before rounding.

**Coverage requirement.** The collector and every audit must see exactly
the terms the spectrum deposits, with the same $W_i$, $E_i$ and $w_i$. A
physical flight split into numerical substeps $k$ is reduced as one field:

$$
I(E)=\Bigl|\sum_k A_k(E)\Bigr|^2=\sum_k|A_k|^2
+2\operatorname{Re}\sum_{k<l}A_kA_l^{*}.
$$

Neither its total mass nor its tail is the independent mixture
$\sum_k|A_k|^2$. A valid certificate would need a bound on the summed field
itself. For example, $|A_k(E)|\le C_k/|E-E_k|$ gives
$I\le(\sum_kC_k)^2/(E-\max_kE_k)^2$, normalized by the flight's own
Parseval mass. Without such a certificate, every route that reduces grouped
substeps must refuse every audit. Otherwise a radiating population reads as
zero mass, and `check_cxr_truncation` maps zero mass to a zero fraction.

### Implementation comparison

| Item | Derivation | Implementation | Result |
|---|---|---|---|
| sinc tail | $\min(1,w/(\pi^2D))$, $1$ for $D\le0$ | `sincsq_upper_tail_bound`: the same; `_accumulate_edge_truncation`: $\min(1,1/(\pi a_wD))$, the same because $w=\pi/a_w$ | match |
| line mass | $W\pi/a$ | `weight * pi / a_width` (edge audit); `weight * width` (selector, with `width = pi / a_width` collected) | match |
| Lorentzian | $\Gamma/(2\pi D)$ per line | `centre + fwhm / (2 pi limit)`, maximized over emitted lines above the relaxation cutoff | match |
| selector | smallest $S$ with $\Lambda\le\varepsilon/2$, round up, cap | bisection on $\Lambda$ from the lowest resonance to the ceiling, `ceil` to 100 eV, `min(ceiling, ...)`, and $\varepsilon/$`proxy_safety` with `BANDWIDTH_PROXY_SAFETY` = 2 | match |
| combined edge | max of both edges, at least start + 100 eV, capped | `case_cxr_stop_eV` | match |
| production weights, batched | the audited triple equals the deposited triple | `_batched_incoherent_block` passes the same `E_r_f`, `aw_f` and `w_f` to the audit and to `accumulate_sincsq`/`_CxrBatch`; `_queue_unit_lines` does the same per CUDA unit | match |
| production weights, per-hkl | the same | section 7: the same `E_r[good]`, `a_width[good]` and `weight[good]` reach the audit and `_reduce_lineshapes` | match |
| grouped flights | refuse or certify | `_mc_spectrum`: `st.grouped and request.truncation_audit is not None` raises `ValueError` before any accumulation; a device backend refuses grouped rows earlier in `_refuse_grouped_reduction` | refuse, now complete |

#### Audit callers

Every caller that builds a truncation audit reaches `_mc_spectrum` through
`_cxr_for_segments` and `mc_cxr_spectrum`. `_mc_spectrum` has no other
caller in `src/`.

- `runner/cxr_grid.py::_direction_populations` (`collect`).
- `runner/case_driver.py::_spectrum_case_impl` and `_line_phase` (edge audit,
  then `check_cxr_truncation`).
- `runner/streamed_scoring.py` (edge audit through `EmissionBatch`).
- `runner/line_roulette.py::LineScreen._screen` (`mass_only`).
- `runner/adaptive.py::case_measure` (per-electron masses).
- `energy_grid/bandwidth_check.py` production and attribute (edge and
  `collect`).

The refusal is per call. A call that mixes grouped and single-row flights
also has `st.grouped` true and is refused. The coherent evaluation receives
no audit (`emission_adapters._cxr`). Two-beam transition radiation bypasses
`mc_cxr_spectrum` but refuses `line_grid_policy` in `validate_dtr_case`, so
it cannot produce an empty audit under this policy either.

One residual silent-zero path is benign. `check_cxr_truncation` still
returns fraction 0 when `line_mass` is 0, and `case_measure` still returns
zeros when `electron_mass` is absent. Each is now reached only when no line
is deposited either: no row resonates inside the keep window, or no row has
finite positive weight. In both cases the spectrum is also zero.

### Numerical evidence

The verifier wrote its own scratch probe; it does not reuse the previous
verifier's script. The run used the CPU backend under
`systemd-run MemoryMax=8G`. The setup was two 25 keV electrons in HOPG,
4000 Å, seed 11, lockstep midpoint transport, then `subdivide_flights`. The
spectrum used (002), $B=0.8\ \mathrm{Å}^2$, a take-off angle of 119° and an
axis from 880 to 1068 eV in 2 eV steps.

| Population | Unaudited spectrum sum | `collect` | edge | `mass_only` | adaptive (`n_electrons`) | runner `_direction_populations` |
|---|---|---|---|---|---|---|
| `max_dE_frac=0` (26 rows = 26 flights) | $1.668584\times10^{-7}$ | admitted, 1 chunk | admitted, line mass $6.753977\times10^{-7}$ | admitted, the same mass | admitted, the same mass | 26 lines |
| `max_dE_frac=2e-4` (391 rows, 26 flights) | $1.667479\times10^{-7}$ | **refused** | **refused** | **refused** | **refused** | **refused** |
| mixed (electron 0 substepped, 197 rows; electron 1 single rows, 15 rows) | $1.667567\times10^{-7}$ | **refused** | **refused** | n/a | n/a | n/a |

The probe also compared production weights against the deposited
spectrum. On the ungrouped population over 600–1399.95 eV at 0.05 eV, the
trapezoid integral of the per-electron spectrum was
$3.590788\times10^{-7}$. The audit line mass was $7.200206\times10^{-7}$
over 2 electrons. The resulting $\int S\,dE\cdot N_e/m=0.997412$. The
missing $2.59\times10^{-3}$ equals half the audited bound
$(2.215+2.964)\times10^{-3}/2$, as the factor-two asymptote predicts. On
that narrow axis `check_cxr_truncation` correctly refused, with a
$2.2\times10^{-3}$ upper fraction against the $10^{-4}$ share.

The elementary bounds were checked independently with SciPy's `sici`. Over
$aD\in\{0.01,\dots,10^4\}$, the largest exact-to-bound ratio was
$0.6735$, at $aD=1$. It tends to $0.5000$. The Lorentzian bound held at
$D/\Gamma\in\{0.01,\dots,10^3\}$ and was tight to $10^{-10}$ at $10^3$. A
single 12000 eV line with $w=7.3$ eV and limit $5\times10^{-5}$ gave a stop
of 26800 eV. The closed form gives 26792.89 eV, which rounds up to the same
value.

`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
tests/montecarlo/spectrum/test_spectrum_phases.py
tests/energy-grid/test_line_grid_bandwidth.py`: **62 passed**, 26 existing
thick-target/kinematic warnings.

### Filter table

| Filter | Result |
|---|---|
| Units: tail fractions dimensionless ($w$, $D$, $\Gamma$ in eV); mass $W\cdot$eV in both audit and selector | pass |
| Limits: one line gives $E_{\rm res}+w/(\pi^2\varepsilon/2)$; $\Gamma\to0$ puts the stop at the highest centre; $D\le0$ counts the whole line | pass |
| Bound direction: sinc bound $\ge$ exact (ratio $\le0.674$); $\arctan y\le y$ | pass |
| Notation: $\varepsilon/2$ in the ledger matches `truncation_limit / proxy_safety` in the docstring; $1/(\pi a_wD)=w/(\pi^2D)$ | pass |
| Production-weight coverage, independent rows (batched, per-hkl, CUDA unit queue) | pass |
| Grouped substeps: every audit kind refused, host and device, including mixed calls | pass (refusal, no certificate) |
| Ceiling cap preserves `line-grid-kinematic-bandwidth` (`stop` $\le$ ceiling; warn-not-refuse only at the ceiling) | pass mechanically. Correctness of the ceiling value itself is the separate `line-grid-kinematic-bandwidth` discrepancy, which affects energy-spread beams, and is not adjudicated here |

### Verdict

**`rederived`**, within the stated scope. The tail bounds, the selector and
production-weight coverage of independent-row populations match the
independent derivation. The flight-grouped discrepancy is resolved by
refusal: no truncation audit can now report zero mass for a grouped,
radiating population. No flight-field tail certificate exists, so measured
bandwidth, line roulette and adaptive CXR sizing are unavailable for
substepped transport. That is a scope limit and should be stated in the
ledger Claim. The result is conditional on the ceiling input from
`line-grid-kinematic-bandwidth`, which has its own open discrepancy.
Not `signed-off`.
