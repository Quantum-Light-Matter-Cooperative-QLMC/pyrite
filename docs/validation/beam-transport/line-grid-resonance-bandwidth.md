# `line-grid-resonance-bandwidth`

Ledger row: [`line-grid-resonance-bandwidth`](../ledger-core-coherent-physics.md#line-grid-resonance-bandwidth). Status: rederived. Validation: `line-grid-resonance-bandwidth`.

## Source equation and selector

The incoherent line kernel evaluates each segment and reflection at its in-medium resonance $E_i$. Its density is $W_i\operatorname{sinc}^2[a_i(E-E_i)/\pi]$, with $w_i=\pi/a_i$ the first-zero width. The production coefficient is $W_i=\alpha\omega_i t_{L,i}^2 |A_i|^2 T_{\mathrm{abs},i}m_i/(4\pi^2\hbar c)$, including the mosaic quadrature weight $m_i$. This is the finite-time lineshape and absorption model used by both incoherent kernel routes. The whole-line mass is $M_i=W_iw_i$.

For an upper edge $S$, let $D_i=S-E_i$. Since $\sin^2 x\leq1$, the fraction of a line above the edge is bounded by

$$
q_i(S)\leq\begin{cases}
1,&D_i\leq0,\\
\min\{1,w_i/(\pi^2D_i)\},&D_i>0.
\end{cases}
$$

The selector finds the smallest $S$ with $\sum_i M_iq_i(S)/\sum_iM_i\leq\epsilon/2$, rounds it up to 100 eV, and caps it at the closed-form kinematic ceiling. Here $\epsilon=10^{-4}$ is the assigned upper-edge share. Characteristic lines supply a separate Lorentzian edge; the larger edge wins. A later spectrum audit applies the same bound with $\epsilon$ to the lines inside the ceiling's keep window, $E_i<C+0.2(C-\text{start})$ for ceiling $C$, charges lines at or above $S$ wholly, and refuses a case that exceeds it. The stop is then raised, still under $C$, over every diffracted-bremsstrahlung and Kossel Bragg window the ceiling axis admits; both corrections are exactly zero outside that finite window, so they lose nothing above the axis (#1243). When $S$ is already the kinematic ceiling, no line resonates above the edge and the automatic bandwidth drops the same tails, so the audit records `capped_at_ceiling` and raises `CxrGridTruncationWarning` instead.

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

## Independent verification (fresh-context 2026-11-16, default flip, #1243)

This is a fresh-context verifier at branch head `97fae5d33` (`issue-1243-respop-default`).
It started at `29e1ed4f1` and re-read `git diff main...HEAD` after the
post-transport fallback commits landed. The scope is the default selection in
`_cxr_grid_policy.py::default_bandwidth_policy`, `::_resolve_bandwidth` and
`::kinematic_fallback_payload`, plus
`runner/cxr_grid.py::measured_bandwidth_fallback`. The tail bounds and the
selector were rederived in the previous sections and are not repeated. CPU
probes ran under `systemd-run MemoryMax=8G` with 2 to 20 electrons. The probe
scripts are scratch files and are not committed.

### Coverage requirement (derived before reading the routing code)

Let $\mathcal{L}$ be every term the case deposits on the line axis
$[E_{\rm start},S]$. The terms have mass $M_j$ and upper-tail fraction
$q_j(S)$. The default may hand $S$ to a measured stop only if the end-to-end
loss satisfies

$$
\frac{\sum_{j\in\mathcal{L}}M_j\,q_j(S)}{\sum_{j\in\mathcal{L}}M_j}\le\varepsilon .
$$

The selector bounds this sum over its collected population
$\mathcal{P}\subseteq\mathcal{L}$. The spectrum audit bounds it over the
lines inside its keep window,
$\mathcal{A}=\{j: E_j<S+0.2(S-E_{\rm start})\}$. The guarantee therefore
holds only when every $j\in\mathcal{L}$ is in $\mathcal{P}$ or in
$\mathcal{A}$. A term in neither is lost with no bound. If $\mathcal{A}$ is
empty, `check_cxr_truncation` maps $\sum M_j=0$ to a zero fraction and passes
silently. A route is covered when it either keeps `kinematic-ceiling`, or
satisfies this condition, or refuses loudly. The new fallback only converts a
loud refusal into a rescore, so it cannot repair a silent pass.

### Route table

| Route reachable from a default build | $\mathcal{P}$ / $\mathcal{A}$ coverage | Result |
|---|---|---|
| Incoherent single slab, batched route | same rows in collection and audit | covered |
| Per-reflection incoherent route (temporal profile, history mosaic `mosaic_route="mc-sampled"`, layered or grooved with the batch flag off) | step 7 of `_per_hkl.py` calls `_accumulate_edge_truncation` before `add_boxes`/reduction. A forced per-hkl route reproduces the batched audit exactly (h-BN 100 keV, 4 e: line mass $3.429965\times10^{-9}$ on both routes, $\int S\,dE\,N_e/M=0.99977$ on both) | covered; dropping `temporal_profile` from the fallback set is justified |
| Coherent / `"both"` | `default_bandwidth_policy` keeps `kinematic-ceiling` | covered |
| `max_dE_frac>0` | kept `kinematic-ceiling`; grouped rows refuse every audit | covered |
| Feature windows | `_resolve_bandwidth` keeps `kinematic-ceiling` when `windowed` | covered |
| Two-beam DTR | `validate_dtr_case` refuses any `line_grid_policy`; `_cxr_for_segments` would pass no audit to `case_dtr_spectrum` | unreachable (loud) |
| Axial fibre | `_validate_axial_channeling` refuses `line_grid_policy` | unreachable (loud) |
| Mixed transport | `resolve_cxr_grid` swaps a defaulted payload to kinematic; `refuse_mixed_cxr` refuses every line call | no line mass exists |
| Layered stacks | collection and audit go through `_cxr_for_segments_once` over `layer_radiators`, and the ceiling is taken over every radiator | covered |
| Groove | incoherent rows are batched with audit; DBR is disabled for grooves | covered |
| GDF beam | incoherent, flat face only; collection uses the realized trajectories | covered |
| Line roulette | `mass_only` screen sums are forwarded into the caller audit | covered (see latent note) |
| Adaptive per-electron | the `case_measure` band is the policy ceiling; `realized_case` precedes the fallback | covered |
| Streamed scoring | audit fed per block; refusal raises (no fallback) | loud |
| **Planar channeling** (`Channeling("planar")`) | see Finding 1 | **uncovered, silent** |
| **Automatic reflection selection** | selected lines beyond the keep window lie in neither $\mathcal{P}$ nor $\mathcal{A}$ (Finding 2) | **partially uncovered** |
| Diffracted brem (default for perfect single slabs), planar-stack TR | scored on the line axis, never in $\mathcal{P}$ or $\mathcal{A}$ (Finding 3) | uncovered, unaudited |

### Finding 1 (discrepancy): planar channeling truncates its source spectra silently

`_resolve_planar` neither refuses nor replaces `line_grid_policy`, and
`default_bandwidth_policy` does not look at channeling. A default planar case
therefore gets `resonance-population`. On zone rows,
`planar_zone_line_parts` removes the plane's own reciprocal row and the
source-owned reflections from native scoring. The channeling source
`spec_channeling` (CR plus the zone PXR/DCR of `case_zone_pxr_spectrum`) owns
their emission. That source is scored on the line axis
(`channeling_emission.case_channeling_spectrum`,
`np.asarray(transport["E_grid"])`), but it is neither collected nor audited.

The probe used the anchor geometry of
`tests/montecarlo/channeling/test_planar_zone_lines.py`: Si, plane
$(1\bar10)$, 17 MeV, 2 µm, a 3° tilt at 46° azimuth, the default bins and
4 electrons. Building the case gives `resonance-population` with ceiling
966200 eV. The collected population is empty (`n_lines = 0`), so the stop
falls to the Si K$\beta_3$ characteristic edge: the axis is $[50, 2700]$ eV.
The native spectrum is zero, and the audit records
`upper_fraction_bound = 0.0`, which passes. Rescoring the same CR source on
$[50, 2\times10^5]$ eV puts its peak at 10352 eV. **0.9917** of its integral
lies above the 2700 eV stop. All 15 Laue DBR reflections in the wide band are
dropped as well, because `bragg_reflections` keeps a reflection only within
1500 eV of the grid. The result is unchanged at `97fae5d33`: nothing refuses,
so the new fallback never fires. On `main` the same case keeps the
kinematic-ceiling axis.

The first divergent factor is the audit denominator. `check_cxr_truncation`
reads $\sum_{j\in\mathcal{A}}M_j=0$ for a case whose line axis carries the
nonzero `spec_channeling`, and maps it to a zero truncation fraction. The
fallback set in `default_bandwidth_policy` omits `channeling`.

### Finding 2: selected reflections beyond the audit keep window are unbounded

Under automatic selection (the default for catalog crystals with no pinned
`hkl_list`, e.g. silicon, diamond, MoS₂), $\mathcal{P}$ is collected over the
built `n_families=4` set. The pilots feed $\mathcal{A}$, but only for lines
with $E_j<S+0.2(S-E_{\rm start})$. A selected higher-|g| line beyond that
window is in neither set. Its mass is neither scored nor audited. The
shell-yield selection is also measured on the truncated axis, so such
shells add no in-band yield there.

MoS₂ at 100 keV with 2 electrons shows the size of the effect. The default
measured axis $[50, 31400]$ eV was refused (audit bound
$1.88\times10^{-4}$), which now triggers the fallback. On the
kinematic-ceiling axis ($[50,10^5]$ eV, 342618 selected reflections), the
scored yield fraction above 31400 eV is $9.29\times10^{-5}$. Of that,
$4.17\times10^{-5}$ lies above the 37670 eV keep edge, which no audit could
see. That unaudited share is 0.42 of $\varepsilon$. A case whose in-window
share stays under $\varepsilon$ while its beyond-window share does not would
pass silently. The fallback covers only refusals.

Two fixes would close this. The selector could collect $\mathcal{P}$ over the
selected set (or bound it by the selection cap). Alternatively, the audit
could be fed lines resonating beyond the keep window as wholly lost, as the
selector already does for its own population.

### Finding 3: other line-axis components outside the audit

Default diffracted bremsstrahlung (`case_diffracted_brem`, added to `spec`)
and planar-stack transition radiation (`spec_transition`, scored on
`E_grid`) are neither collected nor audited. A measured stop drops any DBR
reflection more than 1500 eV above the axis, and any TR above $S$. In the
non-channeling Si 17 MeV probe, DBR was $4\times10^{-9}$ against a native
$2.9\times10^{-2}$ (negligible), and the axis reached the ceiling. In the
channeling probe, all of the DBR was lost (Finding 1). TR was not
quantified.

### Fallback, identity and budget checks

- **Fallback payload.** `kinematic_fallback_payload(default)` equals the
  explicit per-call `kinematic-ceiling` payload for h-BN (1000 and 100 keV),
  MoS₂, silicon, HOPG (25, 333, 1000 keV), 2H-TaS₂ and diamond (5 MeV). It
  differs for a bare sweep without stored energy rows
  (`cxr_by_energy=None`, catalog orientation, no selection). There the
  default carries `grid_model = "automatic-v1"` and the explicit per-call
  payload does not, because `_cxr_grid_for_energy` sets it only when
  `not override`. Both HOPG and h-BN at 25 keV show this. The coordinates
  are the same, since `grid_model` is read nowhere else. The policy cache key
  differs, however, and the anchor only exercises the stored-row h-BN case.
  This is a minor issue: the docstring claim "the payload an explicit
  `kinematic-ceiling` would carry" fails on this path.
- **Phase state on fallback.**
  - The first attempt raises inside `_line_phase`, so the `"lines"` phase is
    never cached.
  - The audit, the temporal accumulators and `CoherentGridAudit` are rebuilt
    per `_spectrum_case_impl` call. `realized_case` runs once, before the
    `try`.
  - A budget refusal of the kinematic axis raises from the `except` block and
    is not re-caught.
  - The `tp` key `groove` is always present (`_transport_case`, artifact
    replay).
  - Latent: under `_spectrum_case_retry` (GPU), `phase_cache.setdefault
    ("line_screen", ...)` returns the first attempt's `LineScreen`. Its
    `_grid` is fixed to the refused two-node band, so the retry screen keeps
    the old keep window, and primaries whose lines all lie above it read
    zero mass and are not kept. This path is unreachable today: line roulette
    disables automatic selection, and with $\mathcal{A}\subseteq\mathcal{P}$
    the audit sees at most the selector's $\varepsilon/2$. A CPU probe (MoS₂,
    100 keV, roulette, forced refusal) gave identical cached and uncached
    results but carried no line mass.
- **Identity.** SHA-256 of the case `line_grid_policy` and of the whole case
  payload were compared between `main` (`38e07316c`) and HEAD. The cases
  were: explicit `kinematic-ceiling` (h-BN, Si 1 MeV), default + coherent,
  default + `max_dE_frac`, default + windows, and explicit
  `resonance-population`. All are bit-for-bit identical. The default h-BN
  1 MeV case and its temporal variant move (`kinematic-ceiling` to
  `resonance-population`). `line_bandwidth_model` is added exactly when
  `emission == "incoherent"`. That matches `coherent_emission = "coherent" in
  emission` in `default_bandwidth_policy`, and it over-keys (safely)
  incoherent cases that keep kinematic (`max_dE_frac`, windows, explicit,
  stored rows).
- **Budget.** Under the measured default, `_cxr_grid_for_energy` clips only
  the declared coarse grid (`min(num, max_points)`) and does not refuse.
  `resolved_coordinates` and `windowed_coordinates` raise
  `CxrGridToleranceError` when the target-spacing count exceeds
  `max_points` on the measured stop or on the fallback ceiling. No silent
  clipping was found.

### Anchor discrimination (`test_selected_reflections_feed_the_measured_bandwidth_audit`)

The test passes (with `--slow`, 9 s). Its ratio
$\int S\,dE\cdot N_e/M_{\mathcal{A}}\in[1-\text{edges},\,1+10^{-3}]$ does
discriminate wiring:

- Unaudited pilots leave no `line_mass` (the base set is replaced, not
  audited; the docstring's "only the base reflections" is inaccurate), so the
  test fails.
- Double-audited pilots push the ratio below $1-\text{edges}$, so the test
  fails.
- A missed in-axis shell is detected only above $10^{-3}$ of the mass, which
  is 10 times $\varepsilon$.

The test is structurally blind to Finding 2: a line outside the keep window
contributes to neither the numerator nor the denominator.

`pyrite-dev test tests/energy-grid/test_line_grid_bandwidth.py
tests/energy-grid/test_line_grid_windows.py
tests/energy-grid/test_line_grid_policy.py`: 97 passed, 1 skipped.

### Filter table

| Filter | Result |
|---|---|
| Units: fractions dimensionless, $M$ in the same photons/sr units in audit and spectrum | pass |
| Limits: an empty $\mathcal{A}$ must not certify a nonzero line-axis spectrum | **fail** (planar channeling) |
| Conventions: default ⇔ no `sources["bandwidth"]`; fallback only for defaulted, non-local, non-windowed | pass |
| Fallback payload equals explicit `kinematic-ceiling` | pass on stored-row/selection/reoriented builds; minor fail (`grid_model`) on bare sweeps |
| Identity of explicit and fallback-set cases vs `main` | pass (bit-for-bit) |
| Budget refusal after transport, no clipping | pass |

### Verdict

**`discrepancy`.** The default flip reaches planar channeling, where the line
axis carries `spec_channeling` (CR and zone PXR/DCR) and DBR. These are
outside both the collected population and the audit, so a 17 MeV Si case
loses 99.2 % of its CR above a 2700 eV stop with a passing zero-mass audit.
The first divergent factor is the audit's zero line-mass denominator, caused
by `default_bandwidth_policy` omitting `channeling`.

Also uncovered:

- Selected reflections beyond the audit keep window under automatic
  selection (unbounded; $0.42\,\varepsilon$ measured on MoS₂ 100 keV).
- Default DBR and planar-stack TR on the line axis (unaudited).

The temporal removal, coherent/`max_dE_frac`/windows fallbacks, identity
preservation, budget refusal and fallback mechanics verify.

Suggested ledger change: keep the claim at `discrepancy` until planar
channeling falls back to `kinematic-ceiling` (or its source spectra enter
$\mathcal{P}$ and $\mathcal{A}$), and until beyond-keep selected lines are
bounded. The minor `grid_model` and stale-`LineScreen` items go in the Notes.
Not `signed-off`.

## Independent re-verification (fresh context, 2026-11-16, #1243 remediation)

Verifier scope: remediation at `ac37b4d97577bb3ab5b4148419a53114398a127d`,
read-only implementation review plus this append-only write-up. The verifier
read the ledger, source statements and signatures/docstrings before inspecting
implementation bodies. The earlier independent tail derivations above are
accepted; this pass derives the routing and population-coverage requirement.

### Independent coverage derivation before implementation inspection

The cited source remains `finite-time-lineshape` and its elementary integral,
plus the Lorentzian source `characteristic-radiation`; no new physical law is
introduced. For an incoherent positive mixture with first-zero width $w_i$,
$M_i=W_iw_i$ is the whole-line mass, and the upper lost mass obeys

$$
L(S)\leq\sum_i M_i\min\left(1,\frac{w_i}{\pi^2(S-E_i)}\right),
$$

where the summand's fraction is defined as one for $E_i\geq S$.
Mass and integrated spectral yield have the same units (photons per steradian
per incident electron after the shared electron normalization); $w_i$ and
$S-E_i$ are in eV. Every mass and bound is nonnegative. Moving $S$ upward
cannot increase this bound. An empty native population can certify zero
native radiation only, never an additional nonzero source on the same axis.

Let $C$ be the existing ceiling, $A$ the lower edge, and
$K_C=C+0.2(C-A)$ its upper keep edge. To bound every native line scored by
the ceiling reference, the measured reduction must audit the same physical
population up to $K_C$, including centres beyond its own former keep edge
$S+0.2(S-A)$. Centres above $S$ must contribute their whole mass to the
upper bound. Merely collecting the original built reflection set does not
cover additional automatically selected reflections. For any extra component
$X$ (channeling, TR or DBR), the required total fraction is

$$
f_{\rm total}(S)=\frac{L_{\rm native}(S)+L_X(S)}{M_{\rm native}+M_X}.
$$

Auditing native radiation alone instead produces
$f_{\rm native}=L_{\rm native}/M_{\rm native}$. Their exact difference is

$$
f_{\rm total}-f_{\rm native}
=\frac{L_X-f_{\rm native}M_X}{M_{\rm native}+M_X}.
$$

Thus an unaudited component needs its own bound, or a kinematic fallback or
refusal, before the full line-axis tolerance can be certified. Smallness at
one probe point does not bound that term for all supported cases. The
one-line limiting case remains $S=E_i+2w_i/(\pi^2\varepsilon)$ before
rounding/capping. These are the criteria used below.

### Comparison against remediation

`_unmeasured_line_axis` tests non-`None` channeling/TR and mixed transport.
`resolve_cxr_grid` replaces a defaulted measured payload with
`kinematic_fallback_payload` before cache lookup, collection or allocation;
an explicit measured payload raises `ValueError`. This closes the earlier
planar-channeling zero-native-mass failure on that entry point, including an
empty channeling dictionary. The guard has no DBR clause.

`cxr_truncation_audit` now passes the ceiling to both native reduction
routes. `_line_keep_window` sets the upper keep edge to
$\max\{S+0.2(S-A),C+0.2(C-A)\}$ when $C>S$.
Both `_batched_tables` and `_accumulate_reflection` use this helper before
their resonance keep masks. `_accumulate_edge_truncation` charges
$E_i\geq S$ with $M_i$ in both `mass_above` and `line_mass`, exactly as
derived. The earlier selected-line gap above the measured upper keep window
is therefore closed for these routes. The lower keep edge remains that of
the measured axis; lower-edge certification is explicitly outside this
claim. The ceiling value itself remains a separate ledger claim.

| Route | Independent result |
|---|---|
| Joined `run_case` | `case_spectrum_inputs` calls the guarded resolver; channeling/TR/mixed default fallback and explicit refusal match the derivation |
| Public `score_spectra` of kept trajectories | `_scoring_transport` calls the same `case_spectrum_inputs`; same protection |
| Batched and per-reflection native reductions | Both use the widened ceiling upper keep window and the same audited mass; selected pilot calls carry the audit |
| Live streaming | Channeling/TR are already refused by `streaming_refusal(..., live=True)`; native reductions carry the ceiling audit. Its resolver calls `_resolve_policy_grid` directly, so this review does not infer the joined guard applies to every streaming case |
| Full artifact replay | `spectrum_from_artifact` reuses recorded spectrum inputs and calls `_spectrum_case` directly; it does **not** call the new guard. Fresh artifacts captured through the guarded joined path retain the ceiling grid. A previously captured measured channeling/TR artifact is not repaired or refused by this routing change alone |
| Streamed artifact replay | Also reuses recorded inputs; channeling and per-electron TR can be replayed (`live=False`). The new guard is bypassed, so the same historical-axis limitation applies |
| DBR on a perfect single slab | Still unaudited on both ordinary and replay scoring; see discrepancy below |

Artifact identity checks compare the stored case and optional physics
provenance, but `_checked_case_and_inputs` deliberately omits the expected
`E_grid` comparison for any case carrying `line_grid_policy`. Thus source
inspection supplies no blanket refusal guarantee for old short-axis
artifacts. This is a route-coverage concern, not a measured claim that such
an artifact exists in the user's store. Existing channeling replay anchors
use explicit line coordinates; they cannot prove the new automatic-policy
fallback on replay. No replay transport/spectrum probe was run in this pass.

### Remaining exact discrepancy: unaudited DBR term

`case_driver._line_phase` calls `check_cxr_truncation` before DBR runs.
Later `_spectrum_case_impl` adds `case_diffracted_brem(...)` directly to
`spec`; the DBR signature has no truncation-audit parameter.
`runner/diffracted_brem.py::_in_claim` admits perfect single slabs with
reflections and directions, independently of measured bandwidth.
`bragg_reflections` additionally keeps a DBR reflection only when its Bragg
centre is within 1500 eV of the supplied grid. Hence a centre satisfying
$S+1500\ \mathrm{eV}<E_B\leq C+1500\ \mathrm{eV}$ can be present on the
ceiling reference and wholly absent from measured scoring and its audit.
Widening the native resonance keep window cannot recover this term.

In the independent expression above set $X=\mathrm{DBR}$. The first exact
divergence is omission of $L_{\rm DBR}$ from the loss numerator and
$M_{\rm DBR}$ from the normalization denominator. The implemented native
fraction and the full line-axis fraction differ by

$$
\boxed{\frac{L_{\rm DBR}-f_{\rm native}M_{\rm DBR}}
{M_{\rm native}+M_{\rm DBR}}}.
$$

No inequality bounding this quantity by the remaining tolerance share is
implemented or supplied. The earlier Si probe's small DBR yield is evidence
at one point, not a uniform bound. This pass neither independently repeats
that value nor treats it as an anchor. The same structural question applies
to the separately added diffracted-characteristic correction; its source
model and signed direct-beam depletion require their own coverage argument
before claiming a bound on every line-axis component.

This does **not** invalidate the independently rederived native sinc bound.
It prevents upgrading the complete default line-axis guarantee to
`rederived` while unaudited DBR is admitted. Remedies: bound the extra
component, preserve/refuse the measured axis where it contributes, or
explicitly restrict the ledger claim and user-visible certificate to native
PXR/CBS plus the separately bounded direct characteristic model. Such a
restriction would not certify total line-axis truncation.

### Checks and filters

Tested source commit: `ac37b4d97577bb3ab5b4148419a53114398a127d`.
Only this write-up is modified. Commands used the project runner:

```text
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/energy-grid/test_line_grid_bandwidth.py tests/dev/test_docs.py
41 passed, 1 skipped (the selected-reflection slow anchor), 26 existing physics warnings.

UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test --slow tests/energy-grid/test_line_grid_bandwidth.py::test_selected_reflections_feed_the_measured_bandwidth_audit
1 passed; one overflow warning in the tail division for nonpositive distances, subsequently selected as whole mass by the distance branch.
```

The slow anchor now runs with the widened window and passes, but the direct
keep-window anchor supplies only an algebraic helper check. No new full
MoS2 transport probe, DBR tolerance bound, GPU job or external comparison was
performed. No source table values were transcribed; reference-data filters
are therefore not applicable. The prescribed docs build and positive HTML
math inspection are delegated to the task owner under the explicit
instruction not to build docs in this verifier. Only static docs tests were
run here; rendering is not yet certified by this pass.

| Filter | Result |
|---|---|
| Units and normalization of native mass and edge fractions | pass |
| Native one-line, nonnegative-mass and monotone-upper-edge limits | pass |
| Channeling/TR/mixed joined default fallback and explicit refusal | pass |
| Upper keep-window coverage of selected native lines in both reductions | pass |
| Total line-axis limit with nonzero unaudited DBR | **fail**: missing numerator and denominator terms given above |
| Ledger/page/code notation agreement | concern: the Claim and opening source summary still describe the old measured-axis keep edge; owner should update them to the ceiling keep edge |
| Replay of a previously captured short measured source axis | concern: stored-axis replay bypasses the new resolver guard |
| Existing source values/evidence artifacts | no new values or artifact-based parity claim; prior probe numbers not re-certified |
| Tolerances | native $\varepsilon/2$ and audit $\varepsilon$ unchanged; no justified DBR tolerance supplied |

### Verdict after remediation

**`discrepancy` for the full default line-axis claim.** The channeling/TR/mixed
joined routing and native selected-line upper-window fixes match the
independent derivation. Native coverage can be described as `rederived`
within that explicit scope. DBR remains outside the collected population
and audit, so the full-axis guarantee does not follow. Replay protection is
conditional on the recorded axis already being safe.

Suggested ledger change, for the owner: retain `Status: discrepancy`; record
that the two previous native/joined-route findings are resolved at
`ac37b4d97`, replace the stale keep-window description with the ceiling upper
keep edge, and retain DBR coverage plus historical recorded-axis replay as
open limitations. Never `signed-off`.

### Prospective compact-support fallback criterion (derived before inspecting window bodies)

For an extra numerically deposited profile with support
$[E_{-,r},E_{+,r}]$, retaining the full support inside $[A,S]$ makes its
additional upper loss exactly zero, even if the correction has either sign.
The sufficient guard is $S\geq\max_r E_{+,r}$ over all reflections visible
on the ceiling reference, with the same deposition convention. The physical
Laue field is not compact; this statement concerns the existing production
window truncation, not an independent bound on physical tails. A nominal
$E_B+1500\ \mathrm{eV}$ guard is sufficient only if that quantity really
bounds the window's absolute upper endpoint. A refractive shift of the window
centre must be included. Shared window/deposition functions would permit the
same guard to cover DBR and the signed Kossel correction, avoiding any new
positive-mixture argument for the latter.

### Observation bypass and production-support assessment

`directions.directional_outputs` calls `resolve_observation_cxr_grid` and
then overwrites each directional transport's `E_grid` with its result.
That resolver calls `_resolve_policy_grid` directly, without
`_unmeasured_line_axis`. The scalar fallback only replaces its local
`policy_payload`; the original case remains measured. Therefore a safe
scalar fallback does not protect multi-direction scoring: channeling/TR can
again receive an unaudited measured axis. `refuse_axial_directions` excludes
axial channeling only; it is not the planar/TR coverage guard. This is an
additional routing discrepancy at the reviewed commit.

The shared window inspection confirms that a plain
$E_B+1500\ \mathrm{eV}\leq S$ test is insufficient. With
$k=E_B/(\hbar c)$, $n_s=\hat{\mathbf n}\cdot\hat{\mathbf s}$ and
$Q_s=(k\hat{\mathbf n}+\mathbf g)\cdot\hat{\mathbf s}$, the numerical
window centre is $E_B+\delta E$, where

$$
\delta E=\operatorname{Re}\chi_0
\left(\frac{k}{2n_s}-\frac{k^2}{2Q_s}\right)
\frac{\hbar c Q_s}{|\hat{\mathbf n}\cdot\mathbf g|}.
$$

This shift can be positive. Also `np.arange(-span, span + step, step)`
can put the last node above `span` by up to one step. Here
$\mathrm{span}\leq1500\ \mathrm{eV}$ and the default step is at most
$1.5\ \mathrm{eV}$. Kossel uses the same window with an additional
`max_step`; its endpoint can differ from DBR's. A conservative common bound
is $E_B+\delta E+\mathrm{span}+\mathrm{step}_{\rm default}$, or the
looser $E_B+\max(0,\delta E)+1501.5\ \mathrm{eV}$. A floating-point
implementation should round its conservative endpoint upward.

Both corrections call `deposit`: node interpolation is exactly zero outside
the fine window, and bin-mean deposition integrates a cumulative profile
constant beyond the endpoints. Kossel's 100-FWHM line-reach test selects
source lines but does not enlarge the deposited window. Containing that
whole window below $S$ therefore prevents additional upper loss for both
production corrections. For bin means, a final-centre test is conservative
because the final bin edge is above $S$; any more permissive test must use
the actual bin edge. This certifies the existing numerically truncated
profiles only, not the physical infinite tails.

Recommended insertion seams for the owner:

- Share channeling/TR/mixed policy fallback/refusal between scalar and
  observation resolution, before cache acceptance. Observation must apply
  the condition across all `n_hats`.
- For DBR/Kossel, inspect production-eligible reflections on the ceiling
  reference, not the already narrowed grid. A shared post-resolution guard
  in `_resolve_policy_grid` must run on cache hits as well as fresh grids.
  Missing support containment should force the default to kinematic or
  refuse an explicit measured choice. The simplest still more conservative
  alternative is fallback/refusal whenever any ceiling-visible eligible
  Bragg reflection exists.
- Full and streamed artifact replay share
  `artifacts._checked_case_and_inputs`. Validate the recorded grid there,
  before handing it to `_spectrum_case` or `BlockSpectrumScorer`. An unsafe
  recorded source axis should be refused clearly, or re-resolved using
  transport evidence where available. A guard only in `resolve_cxr_grid`
  cannot protect replay. Public `score_spectra` already re-resolves through
  `case_spectrum_inputs`.

Even perfect support containment does not alone justify an
$\varepsilon$-of-total-*signed*-yield interpretation: a negative Kossel
correction can reduce the total denominator while native omitted mass stays
fixed. A native PXR/CBS certificate should say so explicitly; a total-yield
certificate needs that denominator treated or a conservative fallback for
these cases. No source changes are proposed as verifier edits.

## Remediation 2 (implementation context, 2026-11-16, #1243)

Not a verification. Responds to each open item of the re-verification above.

| Finding | Change | Anchor |
|---|---|---|
| Observation resolution bypasses the channeling/TR/mixed fallback | `runner/cxr_grid.py::_auditable_policy` is applied by `resolve_cxr_grid`, `resolve_observation_cxr_grid`, `cached_policy_cxr_grid` and `streamed_line_grid.resolve_streamed_line_grid`, before cache lookup | `test_observation_resolution_shares_the_unmeasured_axis_fallback` |
| DBR (and Kossel) outside population and audit | Compact-support criterion above: `_cover_bragg_support` raises $S$ to $\min(C,\max_r E_{+,r})$, with $E_{+,r}=E_B+\delta E+\mathrm{span}+\mathrm{step}_{\rm default}$ rounded up by $10^{-12}$ relative, over every reflection `bragg_reflections` admits on $[A,C]$ toward each direction. A Kossel `max_step` only lowers step and endpoint. Uses the case's built `hkl_list`, which is what DBR/Kossel score | `test_bragg_support_bounds_every_dbr_and_kossel_window_node`, `test_measured_stop_rises_to_the_bragg_support_under_the_ceiling` |
| Cache hits from pre-coverage resolutions | `MEASURED_COVERAGE_REVISION` enters measured cache keys only | `test_measured_grid_cache_key_carries_the_coverage_revision` |
| Stored-axis replay bypasses the guards | `artifacts._checked_case_and_inputs` calls `check_recorded_line_axis`: the recorded axis must reach the ceiling under a kinematic or fallback payload, and the Bragg support under a measured one; otherwise `TrajectoryArtifactError` | `test_replay_refuses_a_recorded_axis_short_of_todays_requirement` |
| Signed Kossel denominator | Not treated: the certificate is stated for native PXR/CBS plus characteristic lines; DBR/Kossel are covered by window containment only (Claim) | — |

Where the stop is raised up to $C$ by a Bragg window at $C$, the result is the kinematic axis. With the default side-on observation, no Laue reflection qualifies and stops are unchanged; for a forward direction (Si 1 MeV, $\hat n=(\sin0.6,0,\cos0.6)$) the support reaches 412 keV. Status stays `discrepancy` pending fresh-context re-verification.


## Independent re-verification 3 (fresh context, 2026-11-16, #1243 remediation 2)

Verifier scope: commit `5b6239611`, read-only implementation review, CPU
probes, and this append-only section. The criteria below were written from
the ledger Claim, the docstrings and the open items of the previous pass
before the bodies of `_cover_bragg_support`, `bragg_support_stop_eV`,
`bragg_window_upper_eV`, `_auditable_policy` and `check_recorded_line_axis`
were read. The native tail bounds rederived earlier are accepted unchanged.

### Coverage criteria (derived before reading the new bodies)

Let $A$ be the lower edge, $C$ the ceiling and $S$ the measured stop. Reflection
$r$ deposits on fine nodes $E_{r,j}=c_r+o_j$, where $c_r=E_{B,r}+\delta E_r$ and
$o_j=-s_r+j\,h_r$ for $j=0,\dots,n_r-1$. Here $\delta E_r$ is the refraction
shift and $s_r$ the half-span. The step is $h_r$, with $h_r^{\rm K}$ for Kossel.
Both `np.interp(..., right=0)` and the cumulative bin-mean integral vanish
above the last node. Therefore the correction adds zero upper loss on an axis
whose last node is at least

$$
E_{+,r}\geq c_r+o_{n_r-1}.
$$

The `arange` endpoint satisfies $o_{n-1}<s+h$; floating rounding of the
length can reach $o_{n-1}=s+h$ only to rounding error. The required
conditions are therefore:

1. Use $E_{+,r}=c_r+s_r+\max(h_r,h_r^{\rm K})$, rounded up, with the signed
   shift $\delta E_r$ included.
2. Include every reflection admitted at runtime on any axis $[A,S']$ with
   $S'\leq C$. `bragg_reflections` admits $E_B\in[A-1500,\,S'+1500]$, so
   the $[A,C]$ set is a superset. Use the same `hkl_list`, orientation,
   $\chi_0$, $|\chi_g|$, thickness and $\hat{\mathbf s}$.
3. Take the maximum over every direction that will score DBR on that axis.
4. If $E_{+,r}>C$, the kinematic reference loses the same tail. The stop
   $\min(\max_r E_{+,r},C)$ then adds zero upper loss relative to that
   reference.
5. Every route that returns a measured axis, including cache hits, must
   satisfy item 1. A cache key must therefore contain every input on which
   $\max_r E_{+,r}$ depends. For `bragg_support_stop_eV` these are the
   `_in_claim` switches (`hkl_list`, `abs_layers`, `layer_radiators`,
   `groove_spacing_ang`, `mosaic_fwhm_rad`, `mosaic_mc_fwhm_rad`), the
   orientation keys, `crystal`, `thickness_ang`, `B_ang2` and the direction.
6. Replay must refuse a stored axis that is short of the item-4 value under a
   measured payload, or short of $C$ under a kinematic/fallback payload. It
   must pass fixed, diagnostic and at-ceiling axes.

### Comparison

| Item | Implementation | Result |
|---|---|---|
| Endpoint bound | `bragg_window_upper_eV` returns $c+s+h_{\rm default}$ plus a relative $10^{-12}$. With $h^{\rm K}=\max(\min(a,b,m),f)\leq\max(\min(a,b),f)=h_{\rm default}$, Kossel's `max_step` cannot raise the endpoint. The shift is included through `_bragg_window_parts`, which is shared with `bragg_window` | pass |
| Reflection set | `bragg_support_stop_eV` calls `bragg_reflections(case, n, [A, C])` with default `margin_eV=1500`, the same `_bragg_energy`, thickness and `s_hat=(0,0,1)` as `case_diffracted_brem`, `_kossel_reflections` and `DiffractedDepthAccumulator`. DBR/Kossel score the built `hkl_list`; `selected_line_spectrum` discards the pruned case (`case_driver.py` passes the outer `case` to both corrections) | pass |
| Kossel line reach | The 100-FWHM reach filters source lines and does not widen the window. The window thickness is `edges[-1]` = slab thickness | pass |
| Directions | `_measured_cxr_grid` passes its full `n_hats` (scalar `(n_hat,)`, observation all directions); the maximum is taken over all | pass |
| Stop raise | `covered = min(support, ceiling)`. `resolved_coordinates` builds `linspace(start, stop, num)` with an exact last node; the local-resolution plan uses the same `stop`. A cache hit rebuilds `linspace(start_eV, stop_eV, num)` from the record | pass |
| Ceiling case | support above $C$ gives the kinematic end node | pass (zero extra loss relative to the reference) |
| `_auditable_policy` routes | scalar `resolve_cxr_grid`, `resolve_observation_cxr_grid`, `cached_policy_cxr_grid` and `resolve_streamed_line_grid` apply it before key construction. `rg` finds no other caller of `_resolve_policy_grid`/`_measured_cxr_grid`/`resolved_coordinates`; `energy_grid/bandwidth_check.py` calls the guarded `resolve_cxr_grid` | pass |
| `MEASURED_COVERAGE_REVISION` | enters only measured keys (scalar and observation share `_policy_cache_key`); pre-revision entries hash differently and miss; kinematic keys unchanged | pass for pre-coverage entries |
| Cache-key completeness (item 5) | `_policy_cache_key` keys `_RESOLUTION_INPUT_KEYS + _WINDOW_INPUT_KEYS + _WEIGHT_INPUT_KEYS`; `mosaic_fwhm_rad` is in none of them, but `_in_claim` reads it | **fail**, see below |
| Replay guard | `_checked_case_and_inputs` calls `check_recorded_line_axis` whenever the case has a policy or diagnostic grid and a recorded `E_grid`. It is shared by `spectrum_from_artifact` and `stream_spectrum_from_artifact`. Diagnostic grids return early; fixed grids keep the exact `E_grid` equality check. A kinematic or fallback payload requires $C$, and a measured payload requires $\min(\text{support},C)$ with the recorded `n_hat`. An explicit measured payload on channeling/TR/mixed raises through `_auditable_policy` | pass |
| Signed Kossel denominator | The Claim now restricts the $\varepsilon$ certificate to native PXR/CBS and characteristic lines. DBR/Kossel are covered by window containment only | consistent; no denominator claim to check |

### Exact discrepancy: Bragg support is not in the measured cache key

The measured stop now depends on
`montecarlo/runner/diffracted_brem.py:24` (`case.get("mosaic_fwhm_rad")`
inside `_in_claim`). A nonzero analytic mosaic gives `support = None`, so no
raise occurs. `runner/cxr_grid.py:663` builds the key from
`_RESOLUTION_INPUT_KEYS` (line 92), `_WINDOW_INPUT_KEYS` (line 139) and
`_WEIGHT_INPUT_KEYS` (line 148). None contains `mosaic_fwhm_rad`.
Before this commit that omission was harmless, because native resolution
reads `mosaic_mc_fwhm_rad` only. After it, a perfect crystal and the same case
with `mosaic_fwhm_rad > 0` share one measured key. If the mosaic case
resolves first, `_resolve_policy_grid` (`cxr_grid.py:751`) returns its
native stop to the perfect case on a cache hit. That stop never passes
through `_cover_bragg_support`. The first exact divergence is in item 1,
because the served stop satisfies

$$
S_{\rm hit}=S_{\rm native}<\min\!\left(\max_r E_{+,r},\,C\right).
$$

DBR/Kossel then lose their window mass above $S_{\rm hit}$, which neither the
native audit nor any other guard sees. The reverse order is harmless: the
mosaic case then receives a longer axis.

The CPU reproduction used Si at 100 keV, $\theta_{\rm obs}=0.9$, no tilt,
$N_e=2$, and an isolated cache directory:

| Run | Stop [eV] | Cache | DBR $\int S\,dE$ |
|---|---|---|---|
| Perfect crystal, cold cache | 96860.20 (= Bragg support; native 84800) | miss | 5.840507e-12 |
| `mosaic_fwhm_rad=1e-3` first, then the perfect crystal | 84800.0 | hit | 5.824139e-12 |

The cold run's DBR mass above 84800 eV on its own axis is 9.39e-15. The
audit passed on both runs (`resonance-population` kept), and the total line
yield differs by 2.97e-6 relative. The magnitude is small at this point, but
the window-containment guarantee is violated exactly. An artifact captured
from the cache-hit run is refused by its own replay through both
`spectrum_from_artifact` and `stream_spectrum_from_artifact` ("stops at
84800 eV, below the 96860.2 eV"). The cold-run artifact replays on both
routes, so the replay guard behaves as derived. The live run, however, has
already scored the unsafe axis.

Possible remedies for the owner: add `mosaic_fwhm_rad` to the measured key
inputs, or key on the `_in_claim` result or on the Bragg support itself. In
either case bump `MEASURED_COVERAGE_REVISION`, because poisoned entries
already sit under the perfect-crystal key. Adding a `None`-valued field
leaves that key unchanged. Alternatively, re-apply `_cover_bragg_support` to
a cache hit. No other `_in_claim` or `bragg_reflections` input is missing:
`hkl_list`, `crystal`, `thickness_ang`, `abs_layers`, `groove_spacing_ang`
and the observation angles are resolution keys. `layer_radiators`, the
orientation keys, `mosaic_mc_fwhm_rad` and `B_ang2` are measured-only keys.
The catalog susceptibility tables were already outside the key before this
change.

### Checks run

```text
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/energy-grid/test_line_grid_bandwidth.py tests/energy-grid/test_line_grid_windows.py::test_windows_keep_the_default_bandwidth_kinematic
40 passed, 1 skipped (selected-reflection slow anchor, not re-run; unchanged by this commit's native path)

UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/runner/test_trajectory_capture.py tests/montecarlo/channeling/test_channeling_replay.py tests/campaign/test_score_spectra.py tests/montecarlo/spectrum/test_transition_radiation_production.py
104 passed, 18 skipped (slow)
```

The probes were CPU-only, run under `systemd-run` with an 8 GiB cap. Their
scripts stayed in the session scratch directory.

- **Window endpoint:** 13452 reflections were tested: Si at 100 and
  1000 keV; thicknesses of $10^3$ Å (the case value), $10^5$ Å and
  $3\times10^6$ Å; 105 forward directions; and `max_step` values of
  $\infty$, 10, 1, 0.05 and $10^{-3}$ eV. The maximum of
  (last node − `bragg_window_upper_eV`) was $-2.0\times10^{-9}$ eV, and in
  every direction the window top was at most `bragg_support_stop_eV`. 7481
  windows had a positive refraction shift, so including the shift is
  load-bearing.
- **Raise below the ceiling:** with Si at 1 MeV, $\theta=0.6$ and no tilt,
  the cold run measured native 170800 eV, raised the stop to the support
  412011.63 eV ($<C=966200$) and passed the audit.
- **Cache collision:** the two cases give equal scalar and observation keys.
  The end-to-end results are tabulated above.
- **Replay:** the cold artifact replays on both routes and the cache-hit
  artifact is refused on both.

The docs build and the HTML math check were not run, per the task
instruction; the task owner must run them. No source values were
transcribed. No GPU or external comparison was run.

### Filter table

| Filter | Result |
|---|---|
| Units (eV endpoints, mass per sr per electron) | pass |
| Window endpoint bound, including signed shift, `arange` overshoot and Kossel `max_step` | pass (derived and probed) |
| Node and bin-mean `deposit` beyond the last node | pass: exactly zero / constant cumulative |
| Reflection set (built `hkl_list`; the $[A,C]$ superset; margin 1500 eV) | pass |
| Multiple directions | pass |
| `min(support, C)` adds zero extra upper loss vs the kinematic reference | pass on fresh resolution |
| `_auditable_policy` before cache lookup on every resolution route | pass; no bypassing route found |
| `MEASURED_COVERAGE_REVISION` vs pre-coverage entries | pass |
| Cache key complete for the new stop dependence | **fail**: `mosaic_fwhm_rad` omitted (`cxr_grid.py:92-148`, `:663`; read at `diffracted_brem.py:24`) |
| Replay guard refuses unsafe / admits safe axes (full and streamed) | pass |
| Ledger/page/code notation | pass: the Claim now states the ceiling keep window, Bragg-window raise and DBR/Kossel scope. Minor: the raised stop is not rounded to 100 eV, and the Claim does not say otherwise |
| Signed Kossel denominator | outside the restricted certificate, as stated in the Claim |

### Verdict

**`discrepancy`.** Window containment, the route guard, the coverage
revision and the replay guard rederive. The only remaining exact divergence
is the speed-cache key: it omits `mosaic_fwhm_rad`, which the Bragg-support
raise now reads. A cache hit can therefore serve a perfect-crystal case the
unraised native stop and drop DBR/Kossel window mass above it. This was
reproduced as 9.4e-15 of 5.84e-12 DBR, with the audit passing.

Suggested ledger change for the owner: keep `Status: discrepancy` and record
that remediation 2 closes the observation bypass, DBR/Kossel window coverage
on fresh resolution and stored-axis replay. List the open item as
"measured cache key omits `mosaic_fwhm_rad` (Bragg-support dependence)".
After the key fix, revision bump and an anchor test (perfect vs
analytic-mosaic keys differ, or a hit re-checks the support), the remaining
criteria above support `rederived`. Never `signed-off`.

### Confirmation (re-verification 3 follow-up)

Tested commit `2f9f2980e`. The fix adds `mosaic_fwhm_rad` to
`_WEIGHT_INPUT_KEYS`, which only measured keys read, and sets
`MEASURED_COVERAGE_REVISION = 2`. Because the revision changed, an entry that
a mosaic case wrote under the perfect-crystal key before the fix now misses.

- **Key inputs:** every field read by `_in_claim`, by `bragg_reflections` and
  by `bragg_support_stop_eV` is now in the measured key. These are
  `hkl_list`, `abs_layers`, `layer_radiators`, `groove_spacing_ang`,
  `mosaic_fwhm_rad`, `mosaic_mc_fwhm_rad`, `crystal`, `beam_uvw`,
  `azimuth_rad`, `recip_miscut_rad`, `surface_hkl`, `B_ang2`,
  `thickness_ang`, and the observation angles `theta_obs_rad`, `tilt_deg`
  and `tilt_azim_deg`. This was checked programmatically against
  `_RESOLUTION_INPUT_KEYS`, `_WINDOW_INPUT_KEYS` and `_WEIGHT_INPUT_KEYS`.
  Nothing is missing. The catalog susceptibility tables were already outside
  the key before this change.
- **Collision probe (p2):** a perfect case and an analytic-mosaic case now
  give different scalar and observation keys.
- **Cache-hit probe (p6; Si 100 keV, $\theta_{\rm obs}=0.9$):**
  - Mosaic first, then the perfect crystal: the perfect case misses the
    cache and gets stop 96860.20 eV, equal to its Bragg support.
  - Its DBR integral, 5.840507e-12, is identical to the cold run.
  - DBR mass lost above the stop is 0, and the line yields agree exactly.
- **Replay probe (p7):** the artifact from the formerly poisoned ordering now
  replays on both the full and streamed routes.
- **Anchors:** `tests/energy-grid/test_line_grid_bandwidth.py` gives 40
  passed and 1 skipped (slow). This includes the new
  `::test_measured_cache_key_separates_mosaic_from_perfect_crystals`.

A nonzero `mosaic_fwhm_rad` of `0.0` is now keyed separately from `None`,
although both are inside the DBR claim and give the same stop. This costs
cache efficiency only and does not affect correctness.

**Verdict: `rederived`.** It covers the following, within the Claim's stated
scope:

- the native PXR/CBS and characteristic certificate;
- DBR/Kossel coverage by window containment (signed-Kossel denominator
  excluded);
- the channeling/TR/mixed fallback on every resolution route;
- the coverage-revisioned cache;
- the replay guard.

It rests on the separate `line-grid-emission-ceiling` and
`line-grid-kinematic-bandwidth` claims. Suggested ledger status for the
owner: `rederived`. Never `signed-off`. The docs build and the HTML math check
remain for the task owner.

## Independent verification (fresh context, 2026-11-17, #1243): P4 stop extension (`line-grid-resonance-bandwidth`, #1243, commit 10ee52f87)

Verdict for the P4 clause: **discrepancy**. The bound, ladder and rung choice rederive. Three things diverge: the resolution of the rescored axis, and two allocation-retry (phase-cache) routes.

### Independent expression (derived before reading the bodies)

Line $i$ has mass $m_i = w_i \pi / a_i$ and resonance $E_i$. The `finite-time-lineshape` tail bound gives the share of that mass at or above a stop $X$:

$$
b_i(X) = m_i \begin{cases} 1, & E_i \ge X \\ \min\!\left(1, \dfrac{1}{\pi a_i (X - E_i)}\right), & E_i < X \end{cases}
$$

For a fixed set of lines, $U(X) = \sum_i b_i(X)$ cannot increase as $X$ increases, because every term is non-increasing and the step at $X = E_i$ goes down from $1$ to at most $1$. The ladder is $X_k = S + (C - S) 2^{-k/3}$ for $k = 0, \dots, 30$, with $X_0 = C$. Sorted in ascending order, the set of rungs that pass is a contiguous run ending at $C$. The first rung that passes, $U(X)/M \le \varepsilon$ with $M = \sum_i m_i$, is therefore the smallest stop that passes. On an axis $[s, X]$ the yield lost above $X$ from scored lines is at most $U(X)$. Lines outside the scoring keep window lie above $X$ and are counted whole. The certificate holds only when the rescored line set equals the audited set and the axis keeps its resolution.

### Checks

1. **Rung sums.** `_accumulate_edge_truncation` computes the same expression at each `extension_stops_eV`. The code agrees with an independent numpy evaluation to $2.2\times10^{-16}$ relative. Against the closed-form exact tail ($\pi/2 - [-\sin^2 u/u + \mathrm{Si}(2u)]$), the bound/exact ratio is at least $1.0057$ over 400 random lines and 31 rungs. The batched, per-hkl, `cxr_batch` queued (`_queue_unit_lines`), selection-pilot and screen routes all call the one accumulator, so every route that adds `mass_above` also adds `mass_above_extension`. The line set is the same as for `mass_above` (the ceiling keep window). Each accumulation builds a new array, so `snapshot_audit`/`restore_audit` restore it exactly (checked: both a fresh-key removal and a restore to the prior value). The roulette `_AUDIT_SUMS` merges the vector. The screen's scoring passes run without an audit, so nothing is counted twice.
2. **Loss above $S_\mathrm{ext}$.** The rescored grid is `linspace(start, S_ext, num)`, so it ends exactly at $S_\mathrm{ext}$. Lines between the $S_\mathrm{ext}$ keep window and the ceiling window lie above $S_\mathrm{ext}$ and are counted whole in the rung sum. The denominator `line_mass` covers the ceiling window. The ladder check is correct. **Divergence A:** "same resolution" does not hold under `sinc-shape`.
3. $S_\mathrm{ext} > S = $ `grid[-1]`, which is already at least `min(bragg_support, C)`, so the windows stay contained. Channeling, TR and mixed cases resolve kinematic before measurement, and their audit sits at the ceiling (warn only, no ladder). Coherent, windows and `max_dE_frac` cases default to kinematic. An explicit measured bandwidth gets no ladder and the refusal is re-raised. Replay re-audits the recorded measured axis and repeats the same extension. Streaming builds the ladder but refuses instead of rescoring, which is safe.
4. The ladder is strictly increasing with first rung $S + 2^{-10}(C - S)$. `extension_stop` selects the first rung that passes, or $C$ when none does. The independent check matched at three limits.

### Divergences

- **A, resolution.** `_cxr_grid_policy.py::kinematic_fallback_payload` excludes only `LOCAL_RESOLUTION_POLICY`. A defaulted bandwidth with `sinc-shape` resolution falls back to a kinematic + `sinc-shape` payload, a combination that `check_selector_combination` refuses. The non-measured branch of `_resolve_policy_grid` then builds it at the sinc-Nyquist step. Probe (hBN 1 MeV, per-call `sinc-shape`/`bin-mean`, forced refusal): measured axis $[50, 41600]$ eV at 1.110 eV, rescored axis $[50, 80300]$ eV at 3.0 eV.
- **B, stale line screen (GPU retry wrapper).** `_spectrum_case_retry` sets `_spectrum_phase_cache`, and `line_screen_for` keeps the screen from the refused pass. On the rescore the screen still scores on `self._grid = (start, S)` with no `ceiling_eV`, because the rescore has no audit. Its keep window therefore ends at $S + 0.2(S - s)$ (50150 eV) instead of $S_\mathrm{ext} + 0.2(S_\mathrm{ext} - s)$ (96470 eV). A primary whose lines all fall in that gap gets mass 0, so `kept=False`: it is neither scored nor charged. The hBN probe confirms the stale grid; that probe had no lines in the gap, so the size of the loss was not measured.
- **C, cached lines phase after a fallback.** If a later phase hits OOM after the fallback rescore, the retry reruns the original measured `tp`, and `_finished_or_run` returns the cached rescored lines. This raises `ValueError: operands could not be broadcast together with shapes (26751,) (13851,)`. It fails loudly.

A, B and C were already present in the P3 ceiling fallback.

### Residual (not demonstrated)

On the rescore, reflection selection reruns on the new axis, and its yield weights depend on the axis. A larger selected set would not be audited. MoS₂ 5° 1 MeV, natural refusal at 1.98e-3 above 31400 eV, extended to 49802.7 eV with bound 9.15e-5 (ceiling 966200 eV): both passes selected the same 342618 reflections at cutoff 64 Å⁻¹.

Focused tests: `tests/energy-grid/test_line_grid_bandwidth.py`, 44 passed, 1 slow test skipped.

### Remediation (implementation context, 2026-11-17, #1243)

(A) `kinematic_fallback_payload` also excludes `sinc-shape`, so a defaulted shape axis refuses rather than rescoring at another step. (B, C) `measured_bandwidth_fallback` gives the rescored case its own nested phase cache (`_spectrum_phase_cache["bandwidth_fallback"]`), so the refused axis's line-roulette screen and the rescored lines phase never cross axes on an allocation retry. Anchors: `tests/energy-grid/test_line_grid_bandwidth.py::test_sinc_shape_default_has_no_kinematic_fallback` (renamed under #1214, below), `::test_fallback_rescore_survives_an_allocation_retry` (without the fix it reproduces `operands could not be broadcast together with shapes (39651,) (13851,)`), `::test_fallback_case_gets_its_own_phase_cache`. Awaiting fresh-context re-verification.

## Independent re-verification (fresh context, 2026-11-17, #1243, commit 688394d03)

**(A) resolution: closed.** `kinematic_fallback_payload` returns `None` for `resonance-local` and `sinc-shape`, so only `sinc-nyquist` reaches the fallback. Its grid `resolved_coordinates(payload, target_step)` takes `target_step` from `sinc_feature_spacing(segments, ...)`, which does not depend on the stop, so the rescored step equals the measured one. A wider axis over the point budget raises `CxrGridToleranceError`; it is never coarsened. hBN 1 MeV, forced refusal: 3.0 eV on both passes, with node and with bin-mean quadrature. A defaulted `sinc-shape` + bin-mean case is refused. On channeling/TR/mixed transport a defaulted `sinc-shape` raises `ValueError` after transport, and its message suggested `kinematic-ceiling`, which `sinc-shape` refuses.

**(B, C) phase cache: closed.** On the measured pass `check_cxr_truncation` raises inside the `lines` phase, so `lines` and every later phase are never cached; only the roulette `line_screen` is, and only the measured pass reuses it. The rescore gets `phase_cache.setdefault("bandwidth_fallback", {})`, the same dict on every attempt. `_spectrum_case_retry` probe, hBN 1 MeV, 6 electrons, limit u/2 (interior rung 41800 → 80400 eV, bound 2.37e-5 ≤ 2.5e-5), with and without roulette. Injected 1–3 OOMs: late line, fallback characteristic ×3, early measured line then fallback characteristic, and mixed. Every run gave an identical grid, identical `spec`/`spec_characteristic` (difference 0.0), an identical refusal record and extension, and an identical roulette record.

**Earlier items:** unchanged code; 69 focused tests pass (bandwidth + GPU OOM retry).

**Divergence: reflection selection reruns on the rescored axis.** `selected_line_spectrum` chooses shells by in-band yield on the production axis, and the rescore reran it on [start, S_ext]. hBN 100 keV, 4 electrons, `line_yield_tolerance=1.9e-4`, forced refusal:

| pass | stop (eV) | reflections | cutoff | selector tail estimate |
|---|---:|---:|---:|---:|
| refused (audited) | 24300 | 2044 | 16 Å⁻¹ | 1.07e-6 |
| rescored, interior rung | 43225 | 478 | 10 Å⁻¹ | 1.79e-4 |
| rescored, ceiling | 100000 | 478 | 10 Å⁻¹ | — |

So the rescored line set was a strict subset of the audited one. Its share bound is $\varepsilon\,M_\mathrm{aud}/M_\mathrm{resc}$, here ×1.112. Measured on the same transport, the 478-reflection set had share 2.902e-5 against the forced 2.94e-5: it passes by chance, not by construction. A superset would be unbounded. With `tol=1e-2` both passes chose 478. Missing term: an audit of the rescored line set.

Outside this claim: the selection-scored yield fell about 10% on the wider axis (4.529e-9 → 4.075e-9, 4 electrons; owner `reflection-yield-tolerance-selector`). Adaptive `batch_means` ran on the refused axis after a fallback (diagnostic only). The extension stop could in principle move at ULP level between chunkings; no probe showed it.

Verdict: `discrepancy` (reflection-selection rescore only); A, B, C and all previously passing items rederive.

### Remediation 2 (implementation context, 2026-11-17, #1243)

The rescore now keeps the measured payload, so its own spectrum-phase audit covers exactly the lines it scores, under its own selection. Only the axis comes from the kinematic-shaped payload at $S_\mathrm{ext}$, with the same start and resolution. A refused rescore extends again from its own stop. Stops strictly increase and the ceiling only warns, so `spectrum_with_bandwidth_fallback` terminates, and a stop that does not rise re-raises the refusal. `bandwidth_fallback.refusals` records each refusal. `_spectrum_case` now passes the scored case and transport to adaptive `batch_means`. The channeling/TR/mixed refusal now names `sinc-nyquist`. Anchors: `tests/energy-grid/test_line_grid_bandwidth.py::test_a_refused_rescore_extends_again`, `::test_fallback_loop_refuses_a_stop_that_does_not_rise`, and the ceiling-fallback anchor now requires the rescore's `capped_at_ceiling` audit record. Awaiting fresh-context re-verification.

## Independent re-verification (fresh context, 2026-11-17, #1243, commit 30261ca4d)

Verdict for the P4 clause: **discrepancy**, limited to the observation-direction route. The item left open by the previous re-verification (the rescore's line set was not the one audited) is closed.

### Expression (derived before reading the new bodies)

Pass $n$ scores line set $L_n$ on $[s, X_n]$. Its certificate needs $U_n(X_n)/M_n \le \varepsilon$, where $U_n(X)=\sum_{i\in L_n} b_i(X)$ and $M_n=\sum_{i\in L_n} m_i$, both over the ceiling keep window, and $b_i(X)=m_i$ for $E_i\ge X$, else $m_i\min\{1, 1/(\pi a_i (X-E_i))\}$. A refusal at $X_n<C$ moves to the first passing rung of $X_n + (C-X_n)2^{-k/3}$, or to $C$, where the audit only warns. Then $X_{n+1} \ge X_n + 2^{-10}(C-X_n)$.

### Checks

1. **Own audit.** The rescored case keeps its measured payload, so `cxr_truncation_audit` builds an audit with `ceiling_eV` $=C$ and a ladder from the rescored `grid[-1]`. Scoring never reads the payload. Selection pilots, roulette screens and the batched, per-hkl and queued routes feed that audit, and the selected spectrum is the sum of the audited pilots. hBN 100 keV, 4 electrons, `line_yield_tolerance=1.9e-4`, one forced limit $2.941\times10^{-5}$ for every pass:

| pass | stop (eV) | reflections | audit share | result |
|---|---:|---:|---:|---|
| measured | 24300 | 2044 | $5.883\times10^{-5}$ | refused, extension 43225 |
| rescore | 43225 | 478 | $2.902\times10^{-5}$ | pass, not capped |

   On the rescore, $\int S\,dE\,N_e/M = 0.99987 \in [1-\text{edges}, 1+10^{-3}]$. With limit $1.18\times10^{-5}$, the rescore takes the capped path at 100000 eV and is audited there (share $1.17\times10^{-5}$).
2. **Termination.** `np.linspace` ends exactly on the chosen rung, so only a rung equal to $C$ reaches `capped_at_ceiling`. A stop that does not rise (only possible when $C-X_n$ is within about $10^3$ ULP) re-raises. `_line_keep_window` on an interior-rung axis keeps the ceiling window. Termination holds through float granularity: no cap bounds the number of refusals, each of which shrinks $C-X_n$ by at least $1-2^{-10}$.
3. **Side effects.** Point budget, replay `check_recorded_line_axis`, adaptive `batch_means` (now on the scored axis), Kossel/DBR containment, and the exclusion of coherent, windowed, `max_dE_frac`, channeling, TR and mixed cases are unchanged or correct. `line_grid_resolved` mixes kinematic-shaped axis fields (`bandwidth_policy="kinematic-ceiling"`, `schema=1`, stop $<C$) with a measured `truncation_audit`. It is not identity-bearing, but its policy label is misleading.
4. **Phase cache.** Nesting depth equals the refusal count. hBN 1 MeV, refusals $41800\to80400\to119000$ eV: five OOM plans gave spectra bit-identical to the no-OOM run, with the same refusal list.

### Exact discrepancy: observation directions

`directions.py::directional_outputs` runs `_spectrum_case` once per direction on the joint axis, and each direction's refusal rescores on that direction's own scalar axis. hBN 1 MeV, two directions:

- A refusal in only one direction raises `ValueError: all input arrays must have the same shape`.
- When both refuse, `E_grid` and `spec_by_direction` span $[50, 119000]$ eV with 39651 nodes, but `line_grid_resolved` still reports the joint measured axis ($[50, 41600]$ eV, 13851 nodes) with no `bandwidth_fallback` and no `truncation_audit`.

Missing: one joint fallback per case that rescores every direction on one extended observation axis and records it, or a refusal. Predates P4 (P3 ceiling fallback). Focused tests: 77 passed, 1 slow skipped.

### Remediation 3 (implementation context, 2026-11-17, #1243)

`directions.py::_joint_directional_spectra` scores every direction with the per-direction fallback disabled (`_joint_bandwidth_fallback`). On any direction's refusal it rebuilds one axis for all directions with `resolve_observation_cxr_grid` on the kinematic-shaped payload at the extended stop. It then rescores and re-audits every direction, repeating until none refuses. `line_grid_resolved` carries the scored axis's record, `bandwidth_fallback`, and per-direction `truncation_audits`. `runner/bandwidth_fallback.py::fallback_axis_payload` chooses the stop on both routes. After `MAX_EXTENSIONS = 2` refusals it goes straight to the ceiling, which bounds the passes independent of float granularity. Rescored records name the audited policy in `bandwidth_policy` and the axis-building one in `axis_policy`. Anchors: `tests/energy-grid/test_line_grid_bandwidth.py::test_observation_directions_extend_one_joint_axis` (without the fix it raises the stacking `ValueError`) and `::test_extensions_are_capped_before_the_ceiling`. Awaiting fresh-context re-verification.

## Independent re-verification (fresh context, 2026-11-17, #1243, commit 20ca9b11c)

Verdict for the P4 clause: **rederived**. The observation-direction discrepancy is closed; the scalar route is unchanged.

### Expression (derived before reading the new bodies)

For directions $D$ the returned axis $A=[s,X]$ has step $h=\min_{d\in D} h_d$ and $X \ge \max_d X^{\mathrm{Bragg}}_d$. Each direction's spectrum is certified when its own audit on $A$ gives $U_d(X)/M_d \le \varepsilon$ over the line set $L_d(A)$ it scores, or $X = C$, where the audit warns. With $n$ refusals the stop is a ladder rung for $n \le 2$ and $C$ for $n = 3$, so there are at most four passes.

### Checks

1. **Joint axis.** hBN 1 MeV, directions $n_0$ and $\pm 0.05$ rad, forced refusal at half the share. A refusal in direction 0, 1 or 2 rebuilt one axis from 41800 eV (13918 nodes) to 80400 eV (26785 nodes). All three directions were re-audited with audit `stop_eV` equal to `E_grid[-1]`, and the record's target step was 3.0 eV on every pass. The extension lies above the joint measured stop, so above the Bragg support over all directions. Outputs are rebuilt on each pass. The recovery wrappers re-raise the refusal and pass the flag through. The temporal grid is rebuilt (n = 6941533 → 13359332). The flag never reaches the scalar call, the case or the result keys.
2. **Termination.** With refusal at 0.9 × share everywhere, the directions route ran 41800 → 46625 → 51148 → $C$ = 119000 eV, where the third refusal's own rung (56491 eV) was overridden by the cap; the audits there are `capped_at_ceiling`. The scalar route ran 41600 → 46437.5 → 50972.7 → 119000 eV. The rise guard re-raises a refusal at the ceiling.
3. **Record.** `line_grid_resolved` carries `bandwidth_fallback`, per-direction `truncation_audits`, `bandwidth_policy = resonance-population` and `axis_policy = kinematic-ceiling`. Checkpoints, the result store and provenance copy it; the replay check reads only `E_grid`; `derive` reads only the scalar `line_grid_diagnostic`, which directional results never carried before or after.
4. **Tests.** The five focused modules: 120 passed, 4 slow tests skipped.

Minor: the "rescoring" warning precedes a re-raise at the ceiling; an extension can exceed the temporal sample cap (`ValueError`); a joint extension uses only the refusing direction's ladder, which is conservative and costs passes, not correctness.

The task owner applied `rederived` to the ledger row from this verdict.

### Remediation (A) replaced (implementation context, 2026-11-17, #1214)

#1214 makes `sinc-shape` the default resolution for bin-mean quadrature on the measured bandwidth and gives the kinematic default `sinc-oversampled` (`W/8`). Refusing every `sinc-shape` fallback would leave defaulted bin-mean cases without the refusal rescore. `kinematic_fallback_payload` therefore excludes only `resonance-local` again; a `sinc-shape` payload, defaulted or chosen, falls back to `kinematic-ceiling` + `sinc-oversampled` with the shape-only keys dropped. The rescore never takes the `sinc-nyquist` step that (A) caught (hBN 1 MeV: 1.11 → 3.0 eV). The bin-mean bound at `W/8` is `(π²/9)/64 = 1.7e-2`, against 1e-2 by construction for `sinc-shape`, so a refused bin-mean axis rescores under a looser bin-mean bound; the rescore's own truncation audit is unchanged. Channeling, transition-radiation and mixed-transport cases with a defaulted `sinc-shape` now resolve to the same fallback instead of raising. Anchor: `::test_sinc_shape_falls_back_to_the_oversampled_step_not_the_nyquist_one` (replaces `::test_sinc_shape_default_has_no_kinematic_fallback`). Awaiting fresh-context verification.

### One fallback rule (implementation context, 2026-11-17, #1265)

The #1214 rewrite above is retired. Resolution no longer depends on the bandwidth: `sinc-oversampled` takes `W/8` under `node` and the [`line-grid-sinc-shape-spacing`](line-grid-sinc-shape-spacing.md) step under `bin-mean`, on every axis. `kinematic_fallback_payload` therefore applies one rule: a `resonance-population` bandwidth with no `sources["bandwidth"]` becomes `kinematic-ceiling` with resolution, quadrature and budget unchanged. `resonance-local` with an unset bandwidth records `"implied by resonance-local"` there and refuses instead of falling back. The open bin-mean 1.7e-2 clause is closed because the fallback axis keeps the 1e-2 step. Anchors: `tests/energy-grid/test_line_grid_selector_table.py` (every combination against the resolver, fallback and `_auditable_policy`), `::test_defaulted_bin_mean_axis_falls_back_with_its_resolution_unchanged`, `::test_resonance_local_implies_its_bandwidth_and_never_falls_back`. Awaiting fresh-context verification.
