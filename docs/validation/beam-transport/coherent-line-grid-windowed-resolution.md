# `coherent-line-grid-windowed-resolution`

Ledger row: [`coherent-line-grid-windowed-resolution`](../ledger-core-coherent-physics.md#coherent-line-grid-windowed-resolution). Instrument: `montecarlo/spectrum/coherent_windows.py::coherent_window_seeds` (envelope, step, decoherence switch), `montecarlo/spectrum/coherent_windows.py::decoherence_bound`, `montecarlo/runner/line_grid.py::_windowed_line_grid` and `::_check_coherent_budget` (policy), `montecarlo/spectrum/diagnostics.py::coherent_retardation` (reducer geometry), `montecarlo/spectrum/coherent_windows.py::CoherentRowCollector` with `mc_spectrum(coefficient_capture=...)` (the reducer's own per-piece coefficients, read-only). Issue #350. No change to the spectrum: this row is grid policy derived from the coherent reducer's own phase, which [`coherent-line-grid-fringe-spacing`](coherent-line-grid-fringe-spacing.md) analyses.

## Claim

Current status: **`anchored`** (fresh-context verdict `rederived`, pinned by the windowed-vs-fine regression) for the frozen-carrier,
float64, `sinc_cutoff = None` scope ([2026-10-08 verification](#independent-verification-of-frozen-carrier-scope-2026-10-08)).
Material dispersion has no error certificate (item 5). This section states the current policy; dated
sections below are the historical record.

Automatic line-grid resolution serves `coherent_emission` when the policy carries feature windows. Per `(reflection, orientation)` row and observation direction:

1. **Envelope.** The row's coherent window is its resonance band $[\min_j E_j, \max_j E_j]$ over its radiating pieces, widened on each side to the smallest edge where the bound on the row's power beyond that edge is at most $\eta_\mathrm{leak} = 10^{-4}$ of the row's per-electron power. The bound is built from the reducer's own per-piece coefficients and holds for clustered jumps (revised after the 2026-10-05 and 2026-10-06 verifications). It covers the float64 reducer with `sinc_cutoff = None`: a `sinc_cutoff` is refused on this path, and a float32 reducer resolves with a `LineShapePrecisionWarning`. A halo that would leave the axis stops at the axis edge; the bound there is the tail beyond the axis, bandwidth truncation rather than window leakage, and is reported.
2. **Step.** Inside the window the Nyquist step is $h = \pi\hbar c / D$, with $D$ the span of the time support of *every* radiating piece of the row: the per-electron span $D_e$ for the $\sum_e |S_e|^2$ floor, the all-electron span for the $|\sum_e S_e|^2$ term. Nodes are placed at $h/8$ (`COHERENT_NYQUIST_OVERSAMPLING = 8`, adopted from the 2026-10-08 ladder; see [Default refinement](#default-refinement-2026-10-08)).
3. **Decoherence switch.** A 100 eV window bin takes the per-electron step when $F_\mathrm{max} N_e \le \eta_F = 10^{-4}$ over the bin, where with a physical bunch charge $N_e$ is the tail population bound $1 + s_\mathrm{pair}(H-1)$ for $H$ emitting samples and physical pair scale $s_\mathrm{pair}$ ([`coherent-physical-bunch-population`](../radiation-physics/coherent-physical-bunch-population.md)), and the all-electron step otherwise. $F$ is bounded only on a finite footprint, by the analytic $F_z = e^{-(\omega\sigma_z)^2}$ the reducer applies; with no offsets ($F \equiv 1$) or an infinite slab (empirical characteristic function, no closed-form bound) every bin takes the all-electron step.
4. **Budget.** The plan is refused, with its coordinate count, above the policy point budget `max_points` (`DEFAULT_MAX_POINTS = 600000`, `PYRITE_ENERGY_GRID_MAX_POINTS`). It is never coarsened. A policy without windows still refuses coherent emission (#117).
5. **Dispersion scope (re-scoped 2026-10-08).** Items 1–2 are claimed for the frozen-carrier row field. The material-dispersive phase is not covered by an a-priori bound: the rigorous dispersive excluded-power bound is reported through `LineShapePrecisionWarning` when the worst row exceeds $2\eta_\mathrm{leak}$ (up to $7.4\times10^{4}$ times the frozen reference on the reduced short-bunch case before #370; $64.9$ after the analytic transverse envelope, with only the row carrying two L1-fallback intervals still above $2\eta_\mathrm{leak}$) and is not a measured error. Dispersive sampling is checked instead by same-trajectory convergence against uniform full-axis references; in those cases the row windows cover the whole axis, so the references test dispersive sampling, not power excluded from the windows ([charge-weighted ladders](../radiation-physics/coherent-physical-bunch-population.md#charge-weighted-remote-checks-2026-10-08): yield, centroid and global FWHM errors at most $2.002\times10^{-4}$ at 30 and 60 keV), and by the opt-in runtime full-axis yield and centroid audits. Window-excluded dispersive power and a general dispersive certificate are out of scope for this row.

## Derivation

### Time-domain field of one row

The coherent reducer builds, per row, $A(E) = \sum_j c_j t_{L,j}\,\mathrm{sinc}(a_j(E - E_j)/\pi)\, e^{i\phi_j(E)}$ with $\phi_j = d_j\omega - \mathbf g\cdot\mathbf r_j$ (the in-medium $L_\mathrm{esc}\,\delta\omega$ slope is $\sim10^{-5}$ of it and is neglected, as in `coherent-line-grid-fringe-spacing`). Each term is the Fourier transform, in the retardation time $\tau = t - \hat n\cdot\mathbf r$, of a constant-amplitude carrier on the piece's support,

$$
e_j(\tau) = a_j\, e^{-i\omega_j \tau + i\psi_j}, \qquad |\tau - d_j| \le \tfrac12 \Delta d_j,
\qquad a_j = \frac{c_j}{1 - \beta\,\hat v_j\cdot\hat n},\quad \Delta d_j = (1 - \beta\,\hat v_j\cdot\hat n)\,t_{L,j},
$$

so that $|A|^2$ carries the autocorrelation of $e(\tau) = \sum_j e_j(\tau)$. The denominators are the in-medium ones the kernels use (`line_seeds.reflection_rows`). Within one electron the supports are disjoint and contiguous, and the phase $\omega\tau - \mathbf g\cdot\mathbf r(t)$ is continuous along the trajectory, so consecutive pieces join without a phase jump. By Parseval the per-electron power is $P_e = 2\pi\sum_{j\in e} a_j^2\,\Delta d_j$.

### Step: band limit and contributing set

$|A(E)|^2$ is the Fourier transform of the autocorrelation of $e(\tau)$, whose support is $[-D, D]$ with $D = \max_j(d_j + \Delta d_j/2) - \min_j(d_j - \Delta d_j/2)$. Uniform sampling at $h \le \pi\hbar c/D$ therefore reconstructs $|A|^2$ (Nyquist), and by Poisson summation the trapezoid sum of $|A|^2$ and of $E^k|A|^2$ is exact for $h \le 2\pi\hbar c/D$, the time-domain aliases at lags $2\pi\hbar c/h$ falling outside the autocorrelation support. Yield and centroid are thus exact at half the density the Nyquist step gives; the FWHM is read by linear interpolation at the half maximum, whose error falls as $h^2$. The current factor-eight oversampling is numerical policy informed by the corrected remote ladders below. This argument describes the fixed-carrier field; the nonlinear production sampling charge remains unresolved.

The span is taken over every radiating piece of the row, not only those resonating inside a bin. A cross term between the in-window field and the tail of a piece resonating elsewhere is bounded only by Cauchy–Schwarz, $2\|A_\mathrm{in}\|\,\|A_\mathrm{tail}\|$, i.e. by the *square root* of the tail's power share; holding it to $10^{-4}$ would need tail shares of $10^{-8}$, reach far beyond any bin. Distinct rows do not interfere (`cross-reflection-coherence`), so the span is per row.

For the $\sum_e |S_e|^2$ floor only one electron's pieces interfere, and $D_e = \max_e D$ over electrons bounds it. The all-electron span uses the reducer's own geometry, `diagnostics.coherent_retardation`: the offset-free transverse position on an infinite slab with active decoherence, the sampled position on a finite footprint.

### Decoherence switch

The reducer evaluates $(1-F)\sum_e|S_e|^2 + F|\sum_e S_e|^2$ (`coherent-inter-electron-decoherence`). By Cauchy–Schwarz $|\sum_e S_e|^2 \le N_e \sum_e |S_e|^2$ pointwise, so the inter-electron term is at most $F N_e$ times the floor at every energy. Where $F_\mathrm{max} N_e \le \eta_F$ over a bin, leaving that term's fringes unresolved can misplace at most that share of the floor there, and the per-electron step suffices. $F_z$ decreases with $\omega$, so its maximum over a bin is at the bin's lower edge. The empirical characteristic function of an infinite slab fluctuates around $1/N_e$ with no closed-form bound, so no bound below one is claimed and $F N_e$ is never small: those cases take the all-electron step, which in the offset-free geometry is close to the per-electron one.

### Envelope: the coherent tail is the jump spectrum

For undamped pieces, integrating each piece's transform gives, at a distance from every carrier,

$$
A(\omega) = \sum_j a_j\, \frac{e^{i(\omega-\omega_j)\tau_j^+} - e^{i(\omega-\omega_j)\tau_j^-}}{i(\omega - \omega_j)}\, e^{i\psi_j}.
$$

At a joint between consecutive pieces of one electron the two end terms share $\tau$ and phase, so they combine into one jump term $\big[a_{k}/(\omega-\omega_{k}) - a_{j}/(\omega-\omega_{j})\big]e^{i\omega\tau}$; a track end, or a gap where a piece is filtered out, contributes its own $a_j/(\omega - \omega_j)$. Beyond an upper edge $X$, with $u_i = X - E_i > 0$, each jump's tail integrates in closed form,

$$
\int_X^\infty \left|\frac{a_k}{E - E_k} - \frac{a_j}{E - E_j}\right|^2 dE
= \frac{a_k^2}{u_k} + \frac{a_j^2}{u_j} - 2 a_j a_k\,\frac{\ln(u_j/u_k)}{u_j - u_k},
$$

with limit $(a_k - a_j)^2/u$ for one carrier (the lower edge is the mirror image). Every amplitude here is the reducer's own: the coherent reduction of the case itself runs once on a two-node axis spanning the bandwidth, with a read-only `coefficient_capture` hook that records, per row, each kept piece's complex coefficient $c_{j,p} = \sqrt{\alpha\omega_j/(4\pi^2\hbar c)}\,t_{L,j}(A_\mathrm{PXR}+A_\mathrm{CBS})_p$ per polarization $p$, its vacuum centre and width ($\Delta d_j = 2\hbar c\,a_{\mathrm{vac},j}$), its formation constants, electron and reducer-geometry $d_j$. The time-domain amplitude is $a_{j,p} = c_{j,p}/\Delta d_j$, damped along the piece by $e^{-\tau_\mathrm{abs}/2}$; a jump's left value is the left piece's amplitude at its end, its right value the right piece's at its start, so absorption, continuous across a joint, adds no zeroth-order field discontinuity; different attenuation slopes still require complex denominators, as corrected below. Complex amplitudes enter the two-carrier integral as $|a_j|^2/u_j + |a_k|^2/u_k - 2\,\mathrm{Re}(a_j a_k^*)\ln(u_j/u_k)/(u_j-u_k)$, summed over polarizations, which add incoherently. The per-electron power is $2\pi\sum_{j,p}|a_{j,p}|^2\Delta d_j\langle e^{-\tau_\mathrm{abs}}\rangle_j$ by Parseval.

The real-denominator expressions above are exact only for undamped pieces.
For exponentially attenuated pieces the current implementation uses the
complex denominators and positive majorants derived in "Owner correction:
attenuation slopes and stable tail norms" below. Continuous endpoint fields
can still produce a joint term when their attenuation slopes differ.

**Clusters.** Let $T_J$ be a positive upper bound on the tail integral of jump term $J$ alone. Jumps of one electron closer than $\theta = M\hbar c/u_\mathrm{min}$ ($M$ = `COHERENT_JUMP_CLUSTER_SEPARATION` = 64, $u_\mathrm{min}$ the edge's distance to the nearest carrier) form a cluster, bounded by the triangle inequality in $L^2$ of the tail, $T_C \le (\sum_{J\in C}\sqrt{T_J})^2$, with no assumption on their phases.

Across clusters, write each jump term as two pieces whose moduli decrease over the tail,

$$
\mathcal J = \frac{a_1}{u_1} - \frac{a_2}{u_2} = \frac{a_1 - a_2}{u_1} + a_2\,\frac{u_2 - u_1}{u_1 u_2},
\qquad
B_J = \left[\sum_p\left(\frac{\lvert a_{1,p} - a_{2,p}\rvert}{u_1} + \frac{\lvert a_{2,p}\rvert\,\lvert u_2 - u_1\rvert}{u_1 u_2}\right)^2\right]^{1/2},
$$

with $u_i$ the carrier distances at the edge and $B_J$ the summed modulus there (`_RowJumps.edge_moduli`). The cross term of jumps $J, K$ at $\Delta\tau$ apart, $\int_X^\infty \mathcal J\mathcal K^* e^{iE\Delta\tau/\hbar c}\,dE$, is per polarization a sum of products $g(E)$ of those pieces with coefficients bounded in modulus by $B$'s; each $g$ is positive and decreasing to zero, so integrating by parts, $\lvert\int_X^\infty g\,e^{iE\Delta\tau/\hbar c}dE\rvert \le 2\hbar c\,g(X)/\lvert\Delta\tau\rvert$, and summing (Cauchy–Schwarz over polarizations) gives $\lvert\text{cross}_{JK}\rvert \le 2\hbar c\,B_J B_K/\lvert\Delta\tau\rvert$. This does not need $\lvert\mathcal J\rvert^2$ itself to be monotone, which fails for a joint whose two carriers cancel at an energy inside the tail; and for a near-cancelling joint $B_J$ stays of the order of $\lvert\mathcal J(X)\rvert$, where an envelope adding the two carriers in modulus would not. Jumps in clusters $k$ ranks apart are at least $k\theta$ apart, so the pair is at most $2u_\mathrm{min}B_JB_K/(kM)$; with $B_C = \sum_{J\in C}B_J$, $2B_CB_{C'} \le B_C^2 + B_{C'}^2$ and $\sum_{k=1}^{n_C-1}2/k \le 2(1+\ln n_C)$ per side, the cross terms sum to at most $4(1+\ln n_C)\,u_\mathrm{min}/M\cdot\sum_C B_C^2$. Jumps of different electrons do not interfere in the per-electron power. The bound is

$$
\ell(X) = \frac{\hbar c\left[\sum_C \big(\sum_{J\in C}\sqrt{T_J}\big)^2 + \Big[\frac{4(1+\ln n_C)\,u_\mathrm{min}}{M}\sum_C B_C^2\Big]_{n_C>1}\right]}{2\pi\sum_{j,p}|a_{j,p}|^2\Delta d_j\langle e^{-\tau_\mathrm{abs}}\rangle_j}.
$$

For a single-carrier end $u_\mathrm{min}B_J^2 \le T_J$, so on free ends the allowance is at most the earlier $4(1+\ln n_C)/M$ of the diagonal.

The inter-electron term adds at most $F(N_e - 1)$ times the floor's tail by the Cauchy–Schwarz step of the decoherence switch, with $F$ at the edge for the upper tail and at the axis start for the lower one. The window edge is the smallest $X$ with $[1 + F(N_e - 1)]\,\ell(X) \le \eta_\mathrm{leak}$, found by bisection.

This is where coherent and incoherent windows differ. The incoherent route leaks every piece's own two edges; the coherent per-electron field leaks only the amplitude *changes* along the track and its ends. The pre-verification rule took the $t_L^2$ proxy $a_j = 1/(1-\beta\hat v_j\cdot\hat n)$ and a fixed $\kappa = 2$ on the diagonal jump sum; the verifier showed that the coupling $c_j$ changes at first order across a deflection (through $\hat v\cdot\mathbf g$ in $A_\mathrm{PXR}$ and $A_\mathrm{CBS}$) and that clustered joints break the diagonal estimate (below). Both are what the reducer coefficients and the cluster bound now carry.

### Assumptions

- Each piece has a frozen carrier and coefficient, and an affine optical depth. Signed attenuation slopes are included exactly in the endpoint denominators. The reducer's energy-dependent in-medium phase is still omitted: no rigorous dispersion error charge has been established, so this is not yet a proved tolerance guarantee for the full production spectrum.
- Leakage is measured against diagonal per-electron Parseval power, not against the blended spectrum. In general $I \ge (1-F)\sum_e|S_e|^2$, and destructive inter-electron interference can make $I$ smaller than that diagonal power. A relative-to-spectrum tolerance therefore needs a separate lower bound.
- The cluster separation $M = 64$ and the bin width are numerical policy. The cross-term coefficient $4(1+\ln n_C)/M$ stays below one for $n_C \le 10^6$ clusters.
- The reducer's retardation phases are float64. In float32 (`_backend.py` default on CUDA) $d/\hbar c$ is rounded at $\lvert d\rvert \sim 10^7$ Å to about 1 Å, up to 0.7 rad per piece at keV energies; joints lose phase continuity and the tail is no longer the jump spectrum. `_coherent_rows` warns (`LineShapePrecisionWarning`, "float64 phases only") and resolution proceeds; `PYRITE_FP64=1` is the supported setting.
- No `sinc_cutoff`: the cutoff truncates each line's support after the captured coefficients and adds tails the jump spectrum does not carry; `_coherent_rows` refuses it.
- Leakage shares are policy shares: $2\eta_\mathrm{leak} + \eta_F = 3\times10^{-4}$ is charged to the feature-window row of `tbl-line-budget-allocation` ($5\times10^{-4}$). On the anchor the production reducer's tails are $\le 5.3\times10^{-5}$ per side to the axis stop and the independent model's $\le 8.0\times10^{-5}$ untruncated under the revised bound, inside that charge.

### Limiting cases

- **Single segment.** One undamped piece has two free ends of amplitude $a$, $\Delta d$ apart. Within $u < M\hbar c/\Delta d$ of its edge they form one cluster, $(2\sqrt{a^2/u})^2 = 4a^2/u$, and the bound returns exactly `sincsq_upper_tail_bound`, $w/(\pi^2 u)$ with $w = 2\pi\hbar c/\Delta d$ — the incoherent `sinc²` tail bound. Farther out the two ends decorrelate and the bound falls to $(1+4(1+\ln 2)/M)/2$ of it, still above the exact tail. Its span is $\Delta d$, so the Nyquist step is $\pi\hbar c/\Delta d = w/2$: two nodes per first-zero width of the incoherent `sinc²` (the incoherent policy's integration-exact step $w$ carries the same factor two between integration and reconstruction). Pinned by `test_single_segment_leak_is_the_incoherent_sinc_tail_bound`.
- **Collinear split.** A straight flight cut into pieces has equal amplitudes and carriers at its joint, which cancel; window and spans equal the unsplit flight's (`test_a_collinear_split_leaks_exactly_what_the_whole_flight_leaks`).
- **Clustered joints.** The verification's amplitude ramps (20 pieces within $\hbar c/u$, twice) and scattered tracks with moving carriers stay inside the bound against an independent quadrature of the time-domain field (`test_clustered_joints_stay_inside_the_bound`, `test_a_scattered_track_with_moving_carriers_stays_inside_the_bound`); exact/bound is 0.17–0.30 for the ramps and 0.06–0.10 for the scattered tracks. The 2026-10-06 counterexample to the earlier lemma (101 at 1000 eV joined to 100 at 1003 eV, a zero of $\lvert\mathcal J\rvert^2$ at 1303 eV beyond an 1100 eV edge) repeated at cluster separations: exact/bound 0.45 (`test_near_cancelling_joints_far_apart_stay_inside_the_bound`).
- **$F \to 0$.** Every bin takes the per-electron step (`test_vanishing_decoherence_takes_the_per_electron_step_everywhere`); $F \equiv 1$ takes the all-electron step everywhere (`test_unbounded_decoherence_takes_the_all_electron_step`).

## Measurements

### Anchor: windowed-auto against a fine explicit grid, identical trajectories

`tests/energy-grid/test_coherent_windowed_line_grid.py::test_windowed_auto_matches_a_fine_explicit_grid_on_identical_trajectories`: hopg, 30 keV, tilt 5°, azimuth 45°, 1 µm, $N_e = 3$, seed 0, `bunch_length_fs = 100` on the 5 mm footprint ($F_z = 0$, per-electron step throughout; two forward rows radiate). The window plan refined on its coherent seeds, and uniform axes over the whole bandwidth at fractions of the finest Nyquist step, all on one transport, against uniform ÷32.

Post-fix at `c2690a35` (reducer coefficients, cluster bound; windows $[10, 2001]$ eV for $(0\,0\,2)$ and $[605, 2718]$ eV for $(0\,0\,4)$). The 2026-10-06 cross-term allowance moves them to $[10, 1930]$ and $[596, 2718]$ eV at 30 649 coordinates; the test's yield, centroid and FWHM gates pass there, and the table is not re-measured:

| axis (Nyquist step ÷) | coordinates | yield rel. | centroid shift [eV] | FWHM rel. |
|---|---|---|---|---|
| windowed ÷2 (shipped) | 30 649 | $-5.7\times10^{-8}$ | $-3.6\times10^{-6}$ | $2.6\times10^{-3}$ |
| windowed ÷4 | 60 795 | $-3.6\times10^{-8}$ | $-2.8\times10^{-5}$ | $6.2\times10^{-4}$ |
| windowed ÷8 | 121 088 | $-3.4\times10^{-8}$ | $-2.9\times10^{-5}$ | $1.4\times10^{-4}$ |
| windowed ÷16 | 241 679 | $-4.1\times10^{-8}$ | $-2.1\times10^{-5}$ | $1.5\times10^{-5}$ |
| uniform ÷8 | 164 451 | $4.6\times10^{-9}$ | $-5.2\times10^{-6}$ | $1.3\times10^{-4}$ |
| uniform ÷16 | 328 902 | $4.5\times10^{-9}$ | $-4.8\times10^{-6}$ | $2.2\times10^{-5}$ |
| uniform ÷32 | 657 803 | reference | reference | reference |

This earlier ladder found yield and centroid converged inside the windows; the wider post-fix windows removed the $1.9\times10^{-6}$ yield offset the pre-fix backbone carried. FWHM converged as $h^2$ — $2.6\times10^{-3}$, $6.2\times10^{-4}$, $1.4\times10^{-4}$ at two, four and eight nodes per Nyquist step — because the dominant 0.97 eV line's half-maximum crossing is read by linear interpolation. The former two-node default met the harness's $10^{-2}$ shape share but missed issue #350's stricter $10^{-3}$ acceptance. This first refinement adopted four nodes; the corrected remote ladders below motivate the current eight-node default. The regression gates yield, centroid and FWHM at $10^{-3}$. The earlier pre-fix plan measured $6.8\times10^{-4}$ at two nodes; that was node placement, not convergence.

Pre-fix (proxy envelope, windows $[254, 1610]$ and $[589, 2760]$ eV), against uniform ÷16: windowed ÷2 at 28 484 coordinates gave yield $1.8\times10^{-6}$, centroid $-1.2\times10^{-3}$ eV, FWHM $6.8\times10^{-4}$.

**Production-reducer tails** (`test_the_production_reducer_leaks_less_than_the_window_bound`; one reflection, full axis at half the window step, power beyond the edge over the axis power; tails beyond the axis stop are not sampled):

| row | window edge [eV] | bound at edge | reducer tail |
|---|---|---|---|
| $(0\,0\,2)$ upper, post-fix | 2001.3 | $1.00\times10^{-4}$ | $4.7\times10^{-5}$ |
| $(0\,0\,4)$ upper, post-fix | 2717.7 | $0.98\times10^{-4}$ | $4.5\times10^{-5}$ |
| $(0\,0\,4)$ lower, post-fix | 605.2 | $1.00\times10^{-4}$ | $1.5\times10^{-5}$ |
| $(0\,0\,2)$ upper, pre-fix | 1609.9 | $1.00\times10^{-4}$ | $9.9\times10^{-5}$ |
| $(0\,0\,4)$ upper, pre-fix | 2760.5 | $1.00\times10^{-4}$ | $3.6\times10^{-5}$ |
| $(0\,0\,2)$ lower, pre-fix | 254.0 | — | $4.4\times10^{-5}$ |

Revised allowance (2026-10-06), CPU float64, with the re-verification's independent time-domain model (untruncated: to $X + 60u$):

| row | window edge [eV] | bound at edge | reducer tail to axis stop | model tail, untruncated |
|---|---|---|---|---|
| $(0\,0\,2)$ upper | 1930.2 | $1.00\times10^{-4}$ | $5.3\times10^{-5}$ | $8.0\times10^{-5}$ |
| $(0\,0\,4)$ upper | 2717.7 | $0.92\times10^{-4}$ | $4.5\times10^{-5}$ | $5.8\times10^{-5}$ |
| $(0\,0\,4)$ lower | 596.4 | $0.88\times10^{-4}$ | $1.5\times10^{-5}$ | $1.4\times10^{-5}$ |

This evaluation runs the reducer over the whole axis, as production does. Truncated at the axis stop the bound sits 1.9–2.0× above the reducer's upper tails; untruncated, 1.25× for $(0\,0\,2)$ and 1.6× for $(0\,0\,4)$, and 6× on the lower edge. The 2026-10-05 verification's $1.08\times10^{-4}$ and $1.36\times10^{-4}$ are not reproduced on CPU by either full-axis or sub-range calls; the shell exported `PYRITE_MC_BACKEND=cuda`, and the same anchor resolved on the float32 CUDA reducer puts the untruncated model tail at $1.8\times10^{-4}$ beyond the 1930 eV edge, so they were most plausibly float32 evaluations — the scope now carried by the precision assumption.

### Reference case: hopg 30 keV, tilt 5°, 1 mm, $N_e = 40$

`build_ladder_case("hopg", 30, 5, 0, n_electrons=40, seed=7)`, `bunch_length_fs = 100` — the `coherent-line-grid-fringe-spacing` reference transport. Windows per row (per-electron Nyquist 0.0122 eV for $(0\,0\,\pm2)$ and $(0\,0\,\pm4)$-family rows of the forward reflections, 0.018 eV for the backward ones): bands reach the axis start at 10 eV and the upper halo is 1.5–62 eV, with 97% of pieces joined to a neighbour. At the Nyquist step the plan holds 216 335 coordinates; at the shipped oversampling it holds about twice that, inside the budget.

### Remote ladders on identical trajectories (`pyrite remote`, GPU, `PYRITE_FP64=1`)

Hopg, tilt 5°, azimuth 0°, 1 mm, $N_e = 40$, seed 7, `bunch_length_fs = 100`, one transport per ladder; windowed-auto against a uniform reference at half the finest window step, plus windowed at half the shipped step (`auto_half`). Global observables are over the whole axis (dominated by a narrow low-energy feature near 17 eV); "band" observables over the line band from 200 eV.

| case | axis | coordinates | wall [s] | yield rel. | centroid rel. | FWHM rel. | band yield rel. | band FWHM rel. |
|---|---|---|---|---|---|---|---|---|
| 30 keV (`c2690a35`) | windowed | 546 045 | 43 | $1.0\times10^{-7}$ | $1.4\times10^{-8}$ | $1.5\times10^{-5}$ | $3.3\times10^{-5}$ | $5.6\times10^{-7}$ |
| 30 keV | reference | 1 092 082 | 140 | — | — | — | — | — |
| 60 keV (`f8821d07`) | windowed | 2 426 296 | 107 | $3.9\times10^{-7}$ | $5.7\times10^{-7}$ | $8.6\times10^{-3}$ | $5.6\times10^{-6}$ | $4.1\times10^{-4}$ |
| 60 keV | windowed, half step | 4 852 588 | 223 | $3.9\times10^{-8}$ | $2.9\times10^{-7}$ | $1.3\times10^{-6}$ | $5.9\times10^{-6}$ | $2.4\times10^{-5}$ |
| 60 keV | reference | 4 852 584 | 230 | — | — | — | — | — |

At 60 keV the global FWHM is that of a 0.014 eV feature, read at two nodes per Nyquist step: inside the $10^{-2}$ shape share, outside $10^{-3}$, and converged at four nodes, the $h^2$ behaviour of the anchor. Where an upper bound exceeds $10^{-4}$ the window reaches the 6000 eV axis stop (bandwidth truncation, reported). The 60 keV ladder first ran out of GPU memory in the per-electron grouped reduction: one electron's dense `(pieces, E_grid)` block at $1.5\times10^6$ coordinates is 11 GB, and `_flight_blocks` never splits a group; energy slicing (`lines/_kernels.py::_energy_slices`, exact because the reduction runs along rows) fixed it.

### Corrected remote ladders (2026-10-07)

The corrected capture/dispersion implementation was rerun on the same workload:
HOPG, tilt 5°, azimuth 0°, 1 mm, 40 sampled incident electrons, seed 7,
100 fs bunch, CUDA float64. Each ladder uses one `CaseLadder` transport for
every grid. These are numerical convergence comparisons, not directed
production certificates or an independent envelope validation.

SLURM jobs 1116 (`20261007-203238-3c85cda9`, 30 keV) and 1117
(`20261007-203242-5ed8e13d`, 60 keV) both completed. Their metadata records
revision `35fc0c3c0148d667389923dfda4d3cf86ff814f5` with `code_dirty: True`
and the same code digest
`7ec14e92e1121793e137f1e50aae31cba1c3f7e123cf4ca78f783d1e66c08bd2`.
The digest identifies the exported working tree; the revision alone does not.
Results are `coh350_ladder30_1007b.json` and
`coh350_ladder60_1007b.json` in the remote workspace. Local retrieved copies
were checked byte-for-byte against the existing ladder records.

Oversampling below means nodes per frozen-carrier Nyquist step. The automatic
policy used four for these runs; finer window grids divide each coherent seed's spacing by
two, three or four. The explicit uniform references use the finest default
row step divided by eight (30 keV) or six (60 keV). Wall times cover spectrum
evaluation only, with one evaluation per grid; transport took 2.24 s and
2.29 s respectively. No repeated-run timing uncertainty was measured.

| case | grid | coordinates | wall [s] | yield rel. | centroid rel. | FWHM rel. |
|---|---|---|---|---|---|---|
| 30 keV | automatic, oversampling 4 | 1 092 085 | 19.15 | 3.29e-8 | 9.28e-10 | 5.97e-6 |
| 30 keV | oversampling 8 | 2 184 166 | 38.49 | 7.83e-9 | 2.23e-9 | 1.86e-6 |
| 30 keV | oversampling 12 | 3 276 249 | 59.98 | 3.13e-9 | 2.46e-9 | 6.27e-7 |
| 30 keV | oversampling 16 | 4 368 330 | 80.90 | 1.52e-9 | 2.54e-9 | 1.64e-7 |
| 30 keV | uniform reference | 8 736 652 | 164.26 | — | — | — |
| 60 keV | automatic, oversampling 4 | 4 852 588 | 222.08 | 1.22e-7 | 3.72e-7 | **1.18e-3** |
| 60 keV | oversampling 8 | 9 705 171 | 463.94 | 4.02e-8 | 1.62e-8 | 2.00e-4 |
| 60 keV | oversampling 12 | 14 557 754 | 869.44 | 1.24e-8 | 6.94e-9 | 2.41e-4 |
| 60 keV | oversampling 16 | 19 410 336 | 1242.40 | 6.54e-9 | 3.43e-9 | 1.57e-5 |
| 60 keV | uniform reference | 29 115 499 | 2356.43 | — | — | — |

The stronger reference changes the earlier conclusion for the 60 keV
default grid: its global FWHM error exceeds the 1e-3 acceptance threshold.
Yield and centroid pass, and the finer window grids pass all three measured
global observables. FWHM error is not monotone across these node placements.
Above 200 eV, the default-grid yield/FWHM relative errors are
9.62e-6/3.34e-7 at 30 keV and 5.30e-6/8.90e-5 at 60 keV; restricting that
band does not satisfy the failed global FWHM criterion. The policy now uses
oversampling eight (2026-10-08), the first measured rung that passes all
three global gates in both cases. These measurements do not establish a
general shape-error bound or license full acceptance of the thick case.

### Default refinement (2026-10-08)

`COHERENT_NYQUIST_OVERSAMPLING` is now eight. Both the all-electron and
grouped coherent window steps are halved; the window envelope and
decoherence switch are unchanged. Coherent cache revision 6 and the
oversampling cache input prevent reuse of four-node axes. Point budgets
still refuse an oversized plan with its coordinate count, without coarsening.

An independent analytic straight-track test catches the interpolation
failure without transport or reference tables. Its retardation duration is
`2*pi*hbar*c`, giving the normalized density `sinc(E-E0)**2` with the
half-maximum root obtained independently of the sampled axis. On a symmetric
980--1020 eV window about 1000 eV, the former default gives FWHM
0.8870855282052617 eV against the exact 0.8858929413789048 eV, a relative
error 1.346197e-3. Eight nodes pass the issue's 1e-3 gate. This regression
covers that fixed-carrier example, not every line or node placement.

The recorded eight-node remote rungs cost 2,184,166 coordinates / 38.49 s
at 30 keV and 9,705,171 coordinates / 463.94 s at 60 keV. These are the
2026-10-07 refinement measurements, not new production runs. The latter
requires an explicit budget above the old smoke's 8,000,000-point allowance;
the default 600,000-point limit remains. Physical bunch weighting, production
sampling/envelope certification and the full 81-case profile remain open;
the ledger status stays `discrepancy`.

### `hopg_short`-class

Reduced `hopg_short` (hopg, 60 keV, 10 µm, tilt 45°, azimuth 135°, beam `1_atto` with `transverse_fwhm_mm = 0.1`, 5 mm footprint, $N_e = 200$): the 1 as bunch gives $F_z N_e > 10^{-4}$ below ~2.5–2.9 keV, so most of every window takes the all-electron Nyquist step (1.16–1.36 × 10⁻³ eV shipped, against 2.8–3.2 × 10⁻³ eV per-electron). Under the default 600 000 budget the plan is refused with its count (4 600 923 on a CPU transport of $4.1\times10^6$ segments). With `PYRITE_ENERGY_GRID_MAX_POINTS=5000000` on the GPU host (`PYRITE_FP64=1`, `f8821d07`) it completes: 39 187 segments, 2 607 240 coordinates, transport plus resolution 3.8 s, spectrum 637 s, finite incoherent and coherent spectra (coherent yield $2.03\times10^{-6}$). The full 81-case profile is not run.

## Open

- **Re-verification.** The 2026-10-06 re-verification (below) returned a narrowed `discrepancy`; its three items are addressed in "Resolution (task owner, 2026-10-06)". A fresh context re-checks them before the row moves.
- **Inter-electron term without a closed-form bound.** For a sub-femtosecond bunch the Cauchy–Schwarz switch is conservative by roughly $N_e$: random transverse positions make $|\sum_e S_e|^2 \approx \sum_e|S_e|^2$ on average, so the unresolved term is $\sim F$ of the floor, not $F N_e$. Using that statistical estimate, or the empirical $F$ of infinite slabs, would bring `hopg_short`-class cases under the budget but is a modelling decision outside #350.
- **Budget.** `hopg_short`-class plans need a `max_points` near $3\times10^6$–$5\times10^6$; the shipped default stays 600 000, so they run on a GPU host with `PYRITE_ENERGY_GRID_MAX_POINTS` raised.

## Fresh-context verification (2026-10-05)

Independent verifier, working from the ledger row, the docstrings and the parent rows (`coherent-line-grid-fringe-spacing`, `coherent-inter-electron-decoherence`) before reading the implementation bodies. Verdict: **`discrepancy`** on the envelope rule (item 1); the step, decoherence switch, limiting cases and reducer geometry (items 2–5) re-derive and match.

### Independent derivation

**Time-domain row field.** A straight piece of duration $t_L$ with velocity $\beta\hat{\mathbf v}$ contributes $\int e^{i(\omega t - \omega\hat{\mathbf n}\cdot\mathbf r(t) - \mathbf g\cdot\mathbf r(t))}\,dt$. With $\tau = t - \hat{\mathbf n}\cdot\mathbf r$, $d\tau = (1-\beta\hat{\mathbf v}\cdot\hat{\mathbf n})\,dt$ and $\mathbf g\cdot\mathbf r$ affine in $\tau$ with slope $\omega_j = \beta\mathbf g\cdot\hat{\mathbf v}/(1-\beta\hat{\mathbf v}\cdot\hat{\mathbf n})$, the piece is the transform $\int f(\tau)e^{i\omega\tau}d\tau$ of

$$
f_j(\tau) = a_j\,e^{-i\psi(\tau)},\qquad a_j = \frac{c_j}{1-\beta\hat{\mathbf v}_j\cdot\hat{\mathbf n}},\qquad \lvert\tau - d_j\rvert\le\tfrac12\Delta d_j,\qquad \Delta d_j = (1-\beta\hat{\mathbf v}_j\cdot\hat{\mathbf n})\,t_{L,j},
$$

with $\psi(\tau) = \mathbf g\cdot\mathbf r(\tau)$ continuous along one track. This reproduces the write-up's representation. Parseval gives $\int\lvert A\rvert^2 d\omega = 2\pi\sum_j a_j^2\Delta d_j$ per electron; in energy, $\int\lvert A\rvert^2 dE = \hbar c\int\lvert A\rvert^2 d\omega$.

**Jump spectrum.** Integrating every piece exactly (not asymptotically),

$$
A(\omega) = \frac{1}{i}\sum_{J} e^{i\omega\tau_J - i\psi(\tau_J)}\,\mathcal J_J(\omega),\qquad
\mathcal J_J = \frac{a_{\rm left}}{\omega-\omega_{\rm left}} - \frac{a_{\rm right}}{\omega-\omega_{\rm right}},
$$

with $a_{\rm left}=0$ at a track start and $a_{\rm right}=0$ at a track end. Phase continuity of $\psi$ is what lets the two end terms of a joint merge into one $\mathcal J_J$. Beyond an edge $X$, with $u_i = X - E_i > 0$,

$$
\int_X^\infty\left\lvert\frac{a_2}{E-E_2}-\frac{a_1}{E-E_1}\right\rvert^2 dE = \frac{a_1^2}{u_1}+\frac{a_2^2}{u_2}-2a_1a_2\,\frac{\ln(u_1/u_2)}{u_1-u_2},
$$

since $\int_0^\infty ds/[(u_1+s)(u_2+s)] = \ln(u_1/u_2)/(u_1-u_2)$; the one-carrier limit is $(a_2-a_1)^2/u$. This matches `_joint_tail` (`coherent_windows.py:105-115`) and the free-end weight `free * a**2 / u` (`:132-142`). The fraction is

$$
\ell(X) = \kappa\,\frac{\hbar c\,\sum_J\int_X^\infty\lvert\mathcal J_J\rvert^2 dE}{2\pi\sum_j a_j^2\Delta d_j},
$$

which matches `coherent_edge_leak` (`:160`) in factors and units: $\hbar c$ [eV Å] times $a^2/\mathrm{eV}$ over $a^2$ Å.

**Single segment.** Two free ends give $\lvert A\rvert^2 = 4a^2\sin^2(\Delta d\,x/2)/x^2 \le 4a^2/x^2 = \kappa\times$ the diagonal sum with $\kappa = 2$. Then $\ell = 2\hbar c/(\pi u\,\Delta d) = w/(\pi^2 u)$ with $w = 2\pi\hbar c/\Delta d$, which is `sincsq_upper_tail_bound`. **Matches.**

**Does $\kappa = 2$ bound $\lvert\sum_J\cdot\rvert^2$?** The pointwise inequality the request names, $\lvert\sum_J A_J\rvert^2 \le (\sum_J\lvert A_J\rvert)^2 \le M\sum_J\lvert A_J\rvert^2$ for $M$ jumps, is **not** what the code applies. The code applies the diagonal sum times $\kappa = 2$. That is exact for one piece (two jumps). For more jumps it holds only after integration, and only when each pair's cross term $\int e^{i\omega(\tau_m-\tau_n)}\mathcal J_m\mathcal J_n^*$ is suppressed, which needs $u\,\lvert\tau_m-\tau_n\rvert \gg \hbar c$. When $k\ge3$ jumps fall within $\hbar c/u$ of one another in $\tau$, their end terms add coherently, and the integrated tail can reach $k/2$ times the $\kappa = 2$ estimate. The code does not check that condition.

### Numeric checks (independent of implementation helpers except `_RowJumps`/`coherent_edge_leak`, which are the objects under test)

Here "exact" means the numerically integrated $\int\lvert A\rvert^2$ of the time-domain proxy, or of the production reducer, beyond the window edge.

| case | exact / $\kappa$-bound |
|---|---|
| synthetic single segment, $u\Delta d = 25, 100$ | 0.50, 0.50 |
| synthetic 20-piece random-walk and monotone tracks (end-dominated) | 0.49–0.52 |
| synthetic amplitude ramp, 20 pieces inside $\hbar c/u$ (×2 ramps) | **7.5, 10.8** |
| synthetic $\beta = 0.9$ random walk through near-$\hat{\mathbf n}$ directions, 40 tracks | median 0.52, **max 4.6** |
| anchor transport (hopg 30 keV, 1 µm, $N_e=3$), ideal proxy (contiguous $\tau$, continuous phase), rows (0 0 2), (0 0 4) | 0.50, 0.50 |
| same, proxy amplitude also carrying $\sqrt{\omega_j}$ | 0.44, 0.52 |

**Production reducer on the anchor transport.** `CaseLadder(..., coherent=True).lines` was run with a single reflection, and $F_z = 0$, so the output is the per-electron floor. It was integrated at a quarter of the window step. The upper-tail power fraction from the window edge to the axis stop is:

| row | window upper edge [eV] | code bound at edge | reducer tail to axis stop (lower bound on full tail) |
|---|---|---|---|
| (0 0 2) | 1609.9 | $1.00\times10^{-4}$ | $\mathbf{1.08\times10^{-4}}$ |
| (0 0 4) | 2760.5 | $1.00\times10^{-4}$ | $\mathbf{1.36\times10^{-4}}$ |

The (0 0 4) lower tail is $1.6\times10^{-5}$, inside its bound. The ideal proxy carries about 0.5 of the bound on the same pieces, so the reducer's tail is 2.2–2.7× the proxy's. Both values are truncated at the axis stop, so the true leak is larger still.

### Diff against the implementation

1. **Envelope — first divergent term: the per-piece coupling $c_j$.** The rule takes $a_j = 1/(1-\beta\hat{\mathbf v}_j\cdot\hat{\mathbf n})$ with $c_j$ fixed along a track (`coherent_windows.py:274`, assumption at write-up §Assumptions line 67). The reducer's coherent amplitude is $c_j = \sqrt{\alpha\,\omega_j/(4\pi^2\hbar c)}\,(A_{\rm PXR}+A_{\rm CBS})_{\rm pol}$ (`lines/_per_hkl.py:330`, `:601`, `:604`). Here $A_{\rm PXR}\propto\chi_{\mathbf g}/\text{detuning}$ and $A_{\rm CBS}\propto 1/(\gamma\,\hat{\mathbf v}\cdot\mathbf g)$. Both change at **first** order in a deflection through $\hat{\mathbf v}\cdot\mathbf g$, which the anchor tracks show halving across one scatter: $E_{\rm res}$ 1210 → 598 eV while $a_j$ moves only 1.178 → 1.199. The write-up's statement that a polarization change "enters the jump at second order in the deflection" does not hold for these factors. The consequence is that the envelope is a bound on the $t_L^2$ proxy, not on the reducer's leaked power: it is exceeded on the ledger's own anchor case.
2. **Envelope — $\kappa = 2$ is an estimate, not a bound,** for clustered joints ($u\,\Delta\tau \lesssim \hbar c$). The write-up states the suppression condition (line 61). The code neither enforces it nor falls back to the triangle-inequality form there.
3. **Step** $h = \pi\hbar c/D$ over the union support of every kept radiating piece (`:276-283`). This matches the band-limit argument: $\lvert A\rvert^2$ is the transform of an autocorrelation supported on $[-D,D]$. It also matches the Poisson statement (trapezoid exact for $h < 2\pi\hbar c/D$). The factor-two oversampling is labelled as measured numerical policy in `_line_grid_policy.py` (`COHERENT_NYQUIST_OVERSAMPLING` comment) and in the write-up. **Matches.** The span uses the in-medium $\Delta d_j$ where the reducer's formation factor uses the vacuum width with an escape-phase slope. The difference is $O(\delta\,\lvert\nabla L\rvert)$ and immaterial for $D$.
4. **Decoherence switch.** $(1-F)\sum_e\lvert S_e\rvert^2 + F\lvert\sum_e S_e\rvert^2 \le [1+F(N_e-1)]\sum_e\lvert S_e\rvert^2$ by Cauchy–Schwarz, so $F\lvert\sum_e S_e\rvert^2 \le F N_e\sum_e\lvert S_e\rvert^2$ pointwise. $F_z$ is evaluated at each bin's lower edge (`:327`), which is the maximum because $F_z$ decreases in $E$. The upper-tail factor is taken at the edge and the lower-tail factor at the axis start (`:286-298`). Both are conservative. $N_e$ counts every line electron with pieces, which is at least the number of nonzero $S_e$ in a row, so it is conservative too. Infinite slab and no-offset cases return no bound, so they take the all-electron step. **Matches; conservative.** The "share of the floor" meaning of $10^{-4}$ is an order-of-magnitude aliasing charge, not a strict quadrature-error bound (an undersampled term's quadrature error can reach about twice its size).
5. **`coherent_retardation`.** $d = t_{\rm mid} - \hat{\mathbf n}\cdot\mathbf r_{\rm mid} + \hat{\mathbf n}_\perp\cdot\mathbf r_{0\perp,e}$ on an infinite slab with active decoherence (`diagnostics.py:140-151`). This equals the reducer's $t_{\rm mid} - \hat{\mathbf n}\cdot(\mathbf r - (x_0,y_0,0))$ (`lines/_setup.py:575-578`), and the sign is right. A finite footprint keeps the sampled positions and has no $t_0$ (`:573`). $t_L = L/\beta$ on the piece length matches `_setup.py:471`. **Matches.**
6. **Limits.** A single segment gives the incoherent `sinc²` bound and the step $w/2$. $F\to0$ gives the per-electron step. A collinear split gives zero joint jump. **Pass** (focused test, 10 passed).
7. **Budget.** $2\times10^{-4} + 10^{-4} = 3\times10^{-4}$ of the $5\times10^{-4}$ feature-window row is arithmetically right. Measured on the anchor, the per-side leak is at least $1.36\times10^{-4}$, so the charge is understated. A charge of at least $3.7\times10^{-4}$ would still sit inside $5\times10^{-4}$. The effect on observables is small: the anchor's yield offset is $1.9\times10^{-6}$, because leaked power is backbone-sampled rather than lost.

### Suggested resolution (task owner)

Either put the reducer's per-piece coupling into the jump amplitudes ($a_j \to \lvert c_j\rvert/(1-\beta\hat{\mathbf v}_j\cdot\hat{\mathbf n})$ per polarization), or state and measure the proxy-to-reducer factor as policy and widen $\kappa$ to cover it. Either way, add a guard (or a triangle-inequality fallback) for joints closer than $\hbar c/u$. Then re-verify against the production-reducer tail on the anchor.

## Fresh-context re-verification (2026-10-06)

Independent verifier, at `c2690a35`. It re-derived the cluster bound and the cross-term allowance from the jump representation before reading `coherent_windows.py`. The reducer-tail numbers were then reproduced on the CPU float64 backend (`PYRITE_MC_BACKEND=cpu`), on the anchor transport of `test_the_production_reducer_leaks_less_than_the_window_bound`. Verdict: **`discrepancy`, narrowed.** The envelope dominates the float64 reducer's tail on the anchor, and the step, the decoherence switch and the limits still hold. Three gaps remain: the stated pairwise lemma is wrong by up to a factor of 2, the claim is not scoped to the float64 reducer, and it is not scoped to `sinc_cutoff = None`.

### Independent checks

**Time-domain model from the captured coefficients.** The verifier built its own time-domain field from the `CoherentRowCollector` rows, without using `_RowJumps` or `coherent_edge_leak`. Each piece is an exponentially damped carrier between its captured end transmissions, and the phase is chained to be continuous across joints. The model was transformed in closed form and integrated numerically beyond the edge. Truncated at the 3700 eV axis stop, it reproduces the production reducer's tail to within 0.7%. The reducer itself was evaluated over the full axis at a quarter and at a half of the window step, with identical results.

| row, side | edge [eV] | bound | reducer tail to axis edge | model tail to axis edge | model tail, untruncated |
|---|---|---|---|---|---|
| $(0\,0\,2)$ upper | 2001.27 | $1.000\times10^{-4}$ | $4.727\times10^{-5}$ | $4.745\times10^{-5}$ | $7.6\times10^{-5}$ |
| $(0\,0\,4)$ upper | 2717.71 | $0.983\times10^{-4}$ | $4.513\times10^{-5}$ | $4.544\times10^{-5}$ | $6.1\times10^{-5}$ |
| $(0\,0\,4)$ lower | 605.16 | $1.000\times10^{-4}$ | $1.546\times10^{-5}$ | $1.541\times10^{-5}$ | (to 0 eV) $1.54\times10^{-5}$ |

The untruncated figure is the integral to $X + 240u$, plus the $1/R$ remainder fitted between the $60u$ and $240u$ reaches. The model's Parseval power equals `CoherentRowField.power` to rounding. So on the float64 reducer the bound dominates the whole tail beyond the edge, both polarizations summed, for both radiating rows. The margin is $1.3\times$ for $(0\,0\,2)$ upper, $1.6\times$ for $(0\,0\,4)$ upper and $6.5\times$ for the lower edge. The write-up's "2.1–2.2×", and the ledger's "each about 2.1x", hold only for the upper tails truncated at the axis stop.

**Capture hook (item c).** For both rows, the spectrum with `coefficient_capture` (which forces the per-hkl route) is bit-identical to three other evaluations: the default route without capture, the per-hkl route without capture, and a repeat with capture. The rows captured on the two-node `[start, stop]` axis are bit-identical to those captured on a 0.05 eV production-range axis: amplitudes, centres and energies all agree. The keep pad of `0.2 (E[-1] - E[0])` depends only on the axis end nodes, so it matches as long as the production grid ends at `start` and `stop`. The hook is called after `good` is formed and only reads its arguments (`lines/_per_hkl.py:342-345`).

**Normalization of the 2026-10-05 figures (item d).** At the pre-fix edges, the full-axis float64 tails are $9.89\times10^{-5}$ ($(0\,0\,2)$, 1609.9 eV) and $3.56\times10^{-5}$ ($(0\,0\,4)$, 2760.5 eV). Sub-range reducer calls on $[X, 3700]$ eV give $4.65\times10^{-5}$ and $3.60\times10^{-5}$, normalized either by the full-axis total or by the sum of the two sub-ranges. The sub-range keep mask drops pieces, which changes the set of gaps. Neither normalization reproduces $1.08\times10^{-4}$ or $1.36\times10^{-4}$ on CPU. The CUDA float32 route does give larger, grid-dependent tails (below). The ambient shell here exports `PYRITE_MC_BACKEND=cuda`, so the earlier figures were most plausibly float32 GPU evaluations. That attribution is not confirmed.

**Cluster allowance (item b).** At leading order in $1/(u\,s)$, the cross term of two separated jump terms is $\mathcal J_m(X)\mathcal J_n^*(X)\,e^{iXs}/(-is)$, with $s = \Delta\tau/\hbar c$. The ratio $u\lvert\mathcal J(X)\rvert^2/T_J$ is 1 for a single-carrier end, 3 for a near-cancelling joint (a $1/E^2$ dipole), and 4 for a joint whose $\lvert\mathcal J\rvert^2$ vanishes inside the tail. An example of the last is $a = 101$ at 1000 eV against $a = 100$ at 1003 eV, with a zero at 1303 eV for an edge at 1100 eV. Measured pair constants $\lvert\text{cross}\rvert\,u\,s/\sqrt{T_mT_n}$ at $s \in [1, 1.2]\,M/u$ are 1.00, 2.99 and 3.93. So the write-up's lemma fails as stated: "$\lvert\mathcal J\rvert^2$ decreases monotonically beyond the edge, so the cross term is at most $2\hbar c/(u\lvert\tau_m-\tau_n\rvert)\sqrt{T_mT_n}$". The monotonicity is false for two-carrier jumps, and the constant is up to 4. A rigorous integration-by-parts bound gives $\lvert f(X)\rvert + \mathrm{TV}(f)$, which is up to 8.

The coded allowance $4(1+\ln n_C)/M$ (`coherent_windows.py:269`) nevertheless covers the leading-order sum for point-like clusters. The ordered-pair sum is a Hilbert form, bounded by Montgomery–Vaughan by $\pi/M$ per unit power. So $4\pi/M = 12.6/M$ is the bound for $n_C \ge 9$. For $n_C \le 8$, the lattice eigenvalues give $4\lambda_{\max} = 4.0, 6.0, 7.2, 9.4$ against $M$ times the allowance, $6.8, 8.4, 9.5, 12.3$ (at $n_C = 2, 3, 4, 8$). For extended clusters (chains spanning many $\theta$) no proof was found. An adversarial scan reached at most 0.90 of the bound. It used isolated jumps spaced just over $\theta$, dipole and alternating carriers, swept phases, and $n \le 25$; the realized cross terms were at most 1.5% of the diagonal against an allowance of at least 13%.

**Limits (item f) and the step.** The single-segment limit is $(2\sqrt{a^2/u})^2\hbar c/(2\pi a^2\Delta d) = w/(\pi^2 u)$. Beyond $M\hbar c/\Delta d$, the bound $(1 + 4(1+\ln 2)/M)/2$ times that stays above the exact tail, whose oscillating excess is at most $1/(u\Delta d/\hbar c) \le 1/M$. The collinear split cancels exactly. For the step, the span now comes from the captured `d_all_geom` and the vacuum width $2\hbar c\,a_\mathrm{vac}$ of the `good` pieces, the same phase and support the reducer uses for both terms. The decoherence switch is unchanged. Focused suite: 17 passed.

**Budget (item e).** $2\eta_\mathrm{leak} + \eta_F = 3\times10^{-4}$ fits the $5\times10^{-4}$ feature-window row whenever the per-side bound holds, which it does on the float64 reducer. `line-spectrum-error-budget.md:54` reads "`DEFAULT_COHERENT_LEAK_LIMIT` (per side, $2\times10^{-4}$)". The per-side value is $10^{-4}$ (`_line_grid_policy.py:199`); $2\times10^{-4}$ is the two-sided total.

### Discrepancies

1. **Pairwise lemma.** The divergent factor is the "2" in the $2\hbar c/(u\lvert\Delta\tau\rvert)$ pair bound: the write-up §Clusters, the `coherent_edge_leak` docstring (`coherent_windows.py:255-256`) and the `_line_grid_policy.py:214-216` comment. The measured constant is up to 3.93, and the monotonicity premise is false. The coded allowance is supported only at leading order and for point-like clusters, by the Hilbert-form argument above. Either restate the derivation that way, or raise the allowance to a provable constant.
2. **Float32 reducer not covered.** On CUDA float32, the default GPU precision (`_backend.py:361`), the reducer casts $d/\hbar c$ to `REAL` (`lines/_per_hkl.py:385`, `lines/_batched.py:603`). On the anchor's finite footprint $\lvert d\rvert$ reaches $1.49\times10^7$ Å, where one float32 ulp is 1 Å. That is up to 0.68 rad of phase per piece at 2.7 keV. Joint phase continuity, which the envelope derivation assumes, is then lost. An inadvertent float32 GPU evaluation of the same transport, the only GPU run, gave tails of $1.29\times10^{-4}$ for $(0\,0\,2)$ and $3.79\times10^{-4}$ for $(0\,0\,4)$ beyond the post-fix edges. These are $1.3\times$ and $3.9\times$ the bound. On CPU, the model with each $d_j$ rounded to float32 gives $1.60\times10^{-4}$ and $2.43\times10^{-4}$ untruncated, which confirms the mechanism. The fix belongs to the reducer (subtract a per-row or per-electron reference from $d$ before the cast), or else the claim should be scoped to float64.
3. **`sinc_cutoff` not covered.** `sinc_cutoff` is not refused on the coherent windowed path (`runner/line_grid.py::_coherent_rows`), and the hook captures the coefficients before the cutoff. A truncated reducer no longer has the jump spectrum. On CPU float64, `sinc_cutoff = 200` gives a $(0\,0\,4)$ upper tail of $1.80\times10^{-4}$, $1.8\times$ the bound, and `sinc_cutoff = 50` gives $9.5\times10^{-5}$. Either refuse the combination or scope the claim to `sinc_cutoff = None`.

Minimal reproduction for item 3 (CPU): take the anchor case and transport from the test fixture, set `{**case, "hkl_list": [(0, 0, 4)], "coherent_emission": True, "sinc_cutoff": 200.0}`, and call `runner._lines_for_segments(..., coherent=True)` on `np.arange(10, 3700, 0.5 * step_electron_eV)`. The trapezoid fraction above 2717.7 eV is $1.80\times10^{-4}$.

Suggested ledger change: status stays `discrepancy`. Add to Notes that the envelope dominates the float64, `sinc_cutoff = None` reducer on the anchor (untruncated tails $7.6\times10^{-5}$ and $6.1\times10^{-5}$ against $1.0\times10^{-4}$ and $0.98\times10^{-4}$). Record the three open items above, and replace "each about 2.1x" with the truncated and untruncated margins.

## Resolution (task owner, 2026-10-06)

1. **Pairwise lemma.** Restated through the decreasing-piece decomposition in §Clusters: each carrier product is positive and decreasing, so integration by parts holds with constant 2 against the edge moduli $B$, not $\sqrt{T}$, and needs no monotonicity of $\lvert\mathcal J\rvert^2$. The allowance is now $4(1+\ln n_C)\,u_\mathrm{min}/M\cdot\sum_C B_C^2$ (`coherent_edge_leak`, `_RowJumps.edge_moduli`). On the verifier's pair cases the measured constant against $B$ is far below 2 (the exact-$T$ constants 2.99 and 3.93 come from $T \ll B^2u$ there). The anchor's $(0\,0\,2)$ upper edge moves from 2001 to 1930 eV; the bound still dominates the untruncated tail (table above).
2. **Float32.** Scoped to float64: `_coherent_rows` warns under a float32 `REAL` (`test_a_float32_reducer_warns_that_the_bound_needs_float64_phases`). Making the reducer's phases float32-safe is a reducer change (#298 territory), not grid policy; a per-row reference subtraction alone does not suffice, since the per-row span reaches $2\times10^7$ Å.
3. **`sinc_cutoff`.** Refused on the coherent windowed path (`test_a_windowed_policy_refuses_a_sinc_cutoff`).

The budget-note wording (`line-spectrum-error-budget.md`, per side $10^{-4}$) is corrected.

## Fresh-context verification of rebased tip (2026-10-06)

This verification starts from the ledger claim and owner docstrings at
`a16efe252c8618641d26f85cdfe6b2bc04080bc7`, before reading implementation
bodies or this document's earlier derivation. Source: the reducer's stated
finite-piece coherent formation field and decoherence blend; no new external
physical law is asserted. The verifier did not implement this claim.

### Independent expression and filters (recorded before implementation inspection)

Let $H=\hbar c$, let $\tau$ be the observation retardation coordinate in
angstroms, and let $k_j=E_j/H$ be a piece's carrier. The field of electron
$e$ in polarization $p$ is

$$
f_{e,p}(\tau)=\sum_{j\in e}a_{j,p}b_j(\tau)
\exp(-i k_j\tau+i\psi_j)\mathbf 1_{[l_j,r_j]}(\tau),
\qquad S_{e,p}(E)=\int f_{e,p}(\tau)\exp(i E\tau/H)\,d\tau .
$$

Here $a_{j,p}=c_{j,p}/d\!d_j$, $d\!d_j=r_j-l_j$, and
$b_j=\exp(-\tau_{\rm abs}/2)$. The captured coefficient has units of the
square root of photon density; division by the angstrom support length
makes the Fourier integral restore those units. Nonoverlapping pieces of
one track and Parseval imply

$$
\int_{-\infty}^{\infty}\sum_{e,p}|S_{e,p}(E)|^2\,dE
=2\pi H\sum_{j,p}|a_{j,p}|^2d\!d_j\langle b_j^2\rangle .
$$

For constant transmission, integration by parts produces endpoint/joint
terms $H\exp(iE\tau_J/H)J_J(E)/i$, with each joint difference
$J_J=v_1/(E-E_1)-v_2/(E-E_2)$ and the complex endpoint fields $v_s$.
Phase continuity is needed to identify the joint's numerator as a field
difference; matching magnitudes alone is insufficient.

On either tail, put $u_s=|E-E_s|>0$. The diagonal tail of one joint is

$$
T_J=\sum_p\left[
\frac{|v_{1,p}|^2}{u_1}+\frac{|v_{2,p}|^2}{u_2}
-2\operatorname{Re}(v_{1,p}v_{2,p}^*)
\frac{\ln(u_2/u_1)}{u_2-u_1}\right],
$$

with the quotient replaced by $1/u_1$ when the carriers coincide. This is
positive by its definition as an integrated squared modulus. For a common
carrier, $T_J=\sum_p|v_{1,p}-v_{2,p}|^2/u_1$.

A safe decreasing-piece envelope for a joint follows by decomposing its
rational amplitude before taking absolute values:

$$
B_J(E)=\left\{\sum_p\left[
\frac{|v_{1,p}-v_{2,p}|}{u_1}
+\frac{|v_{2,p}|\,|E_2-E_1|}{u_1u_2}\right]^2\right\}^{1/2}.
$$

This does not require $|J_J|^2$ itself to decrease. Every positive scalar
piece in the product of two such envelopes decreases to zero. Integration
by parts therefore bounds a separated pair's oscillatory integral by
$2H B_J B_K/|\tau_J-\tau_K|$ before the factor two for the real cross term.
Inside a cluster, the $L^2$ triangle inequality gives
$(\sum_J\sqrt{T_J})^2$. Between ordered clusters separated by
$M H/u_{\min}$, rank distance supplies the harmonic sum bound

$$
\int_{\rm tail}\sum_{e,p}|S_{e,p}|^2\,dE
\le H^2\left[
\sum_C\left(\sum_{J\in C}\sqrt{T_J}\right)^2
+\frac{4(1+\ln n_C)u_{\min}}{M}
\sum_C\left(\sum_{J\in C}B_J\right)^2\right].
$$

Dividing by Parseval power gives a dimensionless fraction. For an undamped
single segment of length $L$, the two endpoints clustered together give
$2H/(\pi L u)=w/(\pi^2u)$, with $w=2\pi H/L$.

The intensity Fourier support is contained in the difference of field
supports. Span $D$ therefore implies Nyquist spacing $\pi H/D$; selecting
two nodes per Nyquist step is an additional numerical policy. All nonzero
radiating pieces enter $D$, regardless of their amplitude rank. The blend

$$
I=(1-F)\sum_{e,p}|S_{e,p}|^2+F\sum_p\left|\sum_eS_{e,p}\right|^2
$$

obeys Cauchy--Schwarz: its inter-electron term is at most
$FN_e\sum_{e,p}|S_{e,p}|^2$. Thus discarding inter-electron frequency support
has an integrated error at most $F_{\max}N_e$ times the diagonal power;
it does not prove a relative error against arbitrarily cancelled coherent
intensity. The total tail multiplier is $1+F(N_e-1)$. Analytic Gaussian
$F=\exp[-(E\sigma_t/\hbar)^2]$ decreases for positive energy; lower-tail
bounds must use the axis start, not its high-energy edge.

Units, the undamped single-piece limit, coincident-carrier cancellation,
zero field, and $F\to0$ all pass these filters. Two scope checks remain
explicitly open before reading code: linear attenuation creates complex
carrier denominators and a continuous derivative contribution, so an
endpoint-only real-denominator estimate needs a proof; energy-dependent
coefficients or dispersive phase slopes must be distinguished from a truly
band-limited frozen-carrier field. Float64 coordinates do not themselves
prove float64 phase evaluation.

### Comparison with implementation

The revised **undamped** two-carrier and cluster argument matches exactly:
`_RowJumps.edge_moduli` decomposes the rational jump into the two decreasing
pieces above; `coherent_edge_leak` uses the within-cluster triangle bound and
$4(1+\ln n_C)u_{\min}\sum_C B_C^2/M$. Polarization Cauchy--Schwarz and ordered
cluster spacing supply the stated constants. No monotonicity assumption on
an individual jump's squared modulus is needed. Taking more clusters across
different electrons only enlarges this bound; those electrons do not
interfere in the diagonal power. The previous pairwise-lemma discrepancy is
resolved for this undamped representation.

`CoherentRowCollector` captures the actual reducer coefficients after the
formation-valid mask and divides by $2H a_{\rm vac}$. Its endpoint
transmissions are $(a_{\rm pb}-b_{\rm ma})/2$ and
$(a_{\rm pb}+b_{\rm ma})/2$. The mean transmission
$(b_{\rm start}^2-b_{\rm end}^2)/(4q)$, with the zero-slope limit, matches
Parseval for each exponentially damped piece. `CoherentRowField.power`
correctly omits $2\pi H$; `coherent_edge_leak` restores the remaining factor
$H/(2\pi)$ when dividing its energy-denominator tail integral by that power.
There is no missing normalization factor in this comparison.

The capture hook is read-only in the collector. An independent CPU check
used the documented HOPG 30 keV, three-electron anchor transport with a
100 fs bunch: default batched spectrum, capture-forced per-reflection
spectrum, and repeated capture were bit-identical on a 1001-node
10--3700 eV axis. Both captured rows' coefficients, energies, durations,
centres, electron IDs, and three transmission arrays were bit-identical
between that axis and the two-node seeding axis. This establishes neutrality
and axis independence on this anchor, not every material or GPU route.

The reducer's actual phase is $Ed/H-\mathbf g\cdot\mathbf r-L_{\rm esc}\delta\omega(E)$, and its formation argument is
$a_{\rm vac}(E-E_j)-(\Delta L/2)\delta\omega(E)$. For zero dispersion,
the vacuum carrier and geometry phases meet at a true track joint; the
common energy-independent joint phase cancels in the joint's squared
modulus. Capture retains the same $d$ used by the reducer, including its
finite-footprint or offset-free convention. A dispersive field is an
approximation to this frozen-carrier representation: the captured row does
not retain $\Delta L$ or $L_{\rm esc}$, and the neglected slope has no error
charge proved here. Float64 removes the previously identified phase
rounding issue; it does not remove dispersion or absorption-slope terms.

The span calculation includes every valid piece and reproduces the
per-electron and all-electron support spans. `decoherence_bound` uses only
the analytic finite-footprint Gaussian; bins test their lower edge for
$F_{\max}N_e$. Lower-tail multiplication uses the axis start. These match
the independent Cauchy--Schwarz derivation, as an absolute share of diagonal
power. Relative-to-spectrum budgeting still needs care under destructive
inter-electron interference: in general $I\ge(1-F)\sum_e|S_e|^2$, not
$I\ge\sum_e|S_e|^2$. The implementation's stated per-electron denominator
is correct; the earlier write-up's stronger floor assumption is not a
universal theorem.

Budget refusal, float32 warning and `sinc_cutoff` refusal match their stated
contracts. Inspection against rebased main `52675cfb` found the grouped
energy-slice integration preserves complete groups and all phase/formation
terms; it slices only independent photon energies. Focused chunk-invariance
checks pass. The attenuation issue below is already present in the physical
formation law on main, rather than a merge-conflict convention change.

### Remaining discrepancy: attenuation slope at a joint

For a linear optical-depth piece, write
$b_j(\tau)=b_j(l_j)\exp[-\lambda_j(\tau-l_j)]$. Its exact transform has
endpoint denominators

$$
\lambda_j-i(E-E_j)/H,
$$

or equivalently $(E-E_j)+iH\lambda_j$, up to the common factor $iH$.
Consequently a continuous attenuated joint has the exact difference

$$
\frac{v_1}{E-E_1+iH\lambda_1}
-\frac{v_2}{E-E_2+iH\lambda_2}.
$$

`_RowJumps.terms` and `edge_moduli` instead use real denominators and erase
this joint when the carriers and endpoint fields agree, even if
$\lambda_1\ne\lambda_2$. That is the first exact divergent term. Continuous
attenuation adds no zeroth-order field jump, but its derivative contributes
a nonzero spectrum. A single-piece inequality $|F|\le1/|v|$ does not justify
cancelling endpoint bounds across pieces with different complex
denominators.

An independent adversarial field admitted by the collector's representation
makes the failure explicit. One electron, one polarization, two contiguous
pieces with equal unit amplitudes and carrier $E_0=1000$ eV occupy
$[-L,0]$ and $[0,L]$, with $L=1000$ Å. Let $Q=8$,
$s=Q/L$, and $\epsilon=\exp(-Q)$. The transmission is continuous,
$b(\tau)=\exp(-s|\tau|)$: endpoint transmissions are $(\epsilon,1)$ and
$(1,\epsilon)$, with mean intensity transmission
$(1-\epsilon^2)/(2Q)$ for each piece. This is two linear, nonnegative
optical-depth pieces, with slopes of opposite sign. Its carrier phase is
continuous; there is no rounding, cutoff, gap or dispersion in this example.

For $x=E-E_0$, the exact transform and diagonal power are

$$
S(E)=2\operatorname{Re}\frac{1-\exp[(-s+ix/H)L]}{s-ix/H},
\qquad P=2\pi H\frac{L(1-\epsilon^2)}{Q}.
$$

The implementation cancels the central joint, retaining only the tiny
outer endpoint fields. At upper edge $E_0+100=1100$ eV it reports a
full-tail fraction $1.13095394158\times10^{-8}$. Independent integration
of the exact expression on only $[1100,2000]$ eV gives
$8.09864494379\times10^{-4}$: **over 71,600 times the asserted full-tail
bound**, and already above the $10^{-4}$ policy share. The finite interval
is sufficient to refute the bound without estimating an infinite remainder.
The exact expression was also compared pointwise to the production
`formation_factor`, using $q=(-Q/2,Q/2)$ and the two corresponding endpoint
sums/differences; they agree at relative tolerance $10^{-10}$ and absolute
tolerance $10^{-11}$ on 100001 energy nodes. No full material/trajectory
reproduction of this adversarial attenuation profile was performed. The
failure is to the asserted general bound on the very piece-field class it
claims to cover; a narrower physical restriction would need its own proof.

Minimal reproduction of the bound (the reference integral above uses no
jump helpers):

```python
import numpy as np
from pyrite.materials.crystal import HBARC_EV_ANG as H
from pyrite.montecarlo.spectrum.coherent_windows import (
    CoherentRowField,
    _RowJumps,
    coherent_edge_leak,
)

L, Q, E0 = 1000.0, 8.0, 1000.0
s, eps = Q / L, np.exp(-Q)
row = CoherentRowField(
    label="continuous attenuation cusp",
    energy_eV=np.array([E0, E0]),
    amplitude=np.ones((1, 2), complex),
    duration_ang=np.array([L, L]),
    centre_ang=np.array([-L / 2, L / 2]),
    electron=np.array([0, 0]),
    start_transmission=np.array([eps, 1.0]),
    end_transmission=np.array([1.0, eps]),
    mean_transmission=np.full(2, (1 - eps**2) / (2 * Q)),
)
bound = coherent_edge_leak(_RowJumps(row), row.power, E0 + 100, upper=True)
x = np.linspace(100.0, 1000.0, 100001)
reference = 2 * np.real((1 - np.exp((-s + 1j * x / H) * L)) / (s - 1j * x / H))
finite_tail = np.trapezoid(reference**2, x) / (2 * np.pi * H * row.power)
print(bound, finite_tail)
```

A second numerical caution, subordinate to this analytic failure:
`_RowJumps.terms` subtracts nearly equal floating-point numbers when a joint
almost cancels. For $u_1=100$, $u_2=100(1+10^{-8})$, $v_1=1$,
$v_2=u_2/u_1$, independent quadrature gives
$T_J=3.33333325612\times10^{-19}$, while the implementation returns
$-3.46944695195\times10^{-18}$ and `coherent_edge_leak` clips it to zero.
At $10^{-7}$ separation it overestimates the term by about a factor of
three. A strict numerical bound needs a stable small-carrier-separation
formula or a rounding allowance; these tiny terms did not explain the
anchor's macroscopic tails.

### Current verdict and validation evidence

**Verdict: `discrepancy`.** The revised decreasing-piece cross-term bound
resolves the previous undamped pairwise issue; float64 scoping and cutoff
refusal are implemented. The complete exponentially damped-field envelope
still differs by the missing $iH\lambda_j$ term at a joint. Retain the
ledger's `discrepancy` status and record this attenuation-slope example in
Notes. A correction or an explicit narrower scope must be independently
re-verified before suggesting `rederived`; human sign-off remains separate.

Checks on CPU, using the canonical project runner:

- `test_coherent_windowed_line_grid.py` plus `test_chunk_invariance.py`:
  38 passed, five existing batched divide/square-root warnings.
- Independent scratch checks: exact damped formation-transform comparison
  passed; coefficient capture neutrality/axis independence passed; numerical
  near-cancellation comparison recorded. The original infinite-interval
  quadrature hit its subdivision limit; the reported counterexample uses
  the independently sufficient finite integral instead.
- No remote compute, heavy sweeps, code edits, commits, issue edits or ledger
  changes were performed by this verifier.


## Owner correction: attenuation slopes and stable tail norms (2026-10-06)

This corrects the attenuation and near-cancellation examples above. It is
implementation-context work, not independent verification. Historical
measurements above use older windows and are not measurements of this revision.

Write a piece's amplitude transmission as
$b_j(t)=b_j(l_j)\exp[-\lambda_j(t-l_j)]$, with signed slope
$\lambda_j=(\tau_{\rm end}-\tau_{\rm start})/(2\Delta d_j)$ [Å$^{-1}$].
The collector retains $\lambda_j=2q_j/\Delta d_j$ directly from the reducer's
formation constants, even if an endpoint transmission underflows. Endpoints
are recovered from $a_{\rm pb}=b_{\rm start}+b_{\rm end}$ and the ratio
$\exp(-2|q|)$ without subtracting nearly equal endpoint sums/differences.

At a continuous-phase joint the exact damped endpoint difference is

$$
\mathcal J_p(E)=\frac{v_{1,p}}{z_1(E)}-\frac{v_{2,p}}{z_2(E)},
\qquad z_i(E)=E-E_i+iH\lambda_i,\qquad H=\hbar c.
$$

For either tail let $E=X+\sigma s$, $s\ge0$, $\sigma=+1$ above the
carriers and $-1$ below them, and $u_i=\sigma(X-E_i)>0$.
Then $|z_i(E)|\ge u_i+s$, and the constant denominator difference has norm

$$
\Delta_z=|z_2-z_1|=\sqrt{(E_1-E_2)^2+H^2(\lambda_2-\lambda_1)^2}.
$$

The exact algebraic decomposition

$$
\mathcal J_p=\frac{v_{1,p}-v_{2,p}}{z_1}
+\frac{v_{2,p}(z_2-z_1)}{z_1z_2}
$$

therefore has the decreasing positive envelope

$$
b_p(s)=\frac{|v_{1,p}-v_{2,p}|}{u_1+s}
+\frac{|v_{2,p}|\Delta_z}{(u_1+s)(u_2+s)}.
$$

Minkowski and $m=\min(u_1,u_2)$ give an upper bound on the squared tail norm
without subtracting nearly equal integrals:

$$
T_J\le\sum_p\left[
\frac{|v_{1,p}-v_{2,p}|}{\sqrt{u_1}}
+\frac{|v_{2,p}|\Delta_z}{\sqrt{3}\,m^{3/2}}
\right]^2,
\qquad
\int_0^\infty\frac{ds}{(u_1+s)^2(u_2+s)^2}\le\frac{1}{3m^3}.
$$

Swapping indices supplies a second valid decomposition. The implementation
chooses the smaller bound per polarization. At a free end both denominators
are assigned the same carrier and slope, leaving $|v|^2/u$ exactly as a
conservative single-denominator bound. When carriers and slopes match, a
joint becomes $|v_1-v_2|^2/u$; a collinear split still cancels exactly.
Equal fields with different slopes no longer cancel.

For cross terms use $B_J=(\sum_p b_p(0)^2)^{1/2}$, again choosing either
index ordering at the edge and keeping that decomposition fixed for the
proof. Complex denominator phases mean the product is not a positive real
function. Instead $|(1/z_i)'|\le1/(u_i+s)^2$, so each rational piece's
derivative norm is bounded by the negative derivative of its real envelope.
For a product of two pieces, $|g|\le G$ and $|g'|\le-G'$. Integration by
parts then yields

$$
\left|\int_0^\infty g(s)e^{i\sigma s\Delta t/H}ds\right|
\le\frac{H}{|\Delta t|}\left[G(0)+\int_0^\infty(-G')ds\right]
=\frac{2H G(0)}{|\Delta t|}.
$$

Thus the existing cluster allowance $4(1+\ln n_C)u_{\min}\sum_C B_C^2/M$
remains valid for signed attenuation slopes. The diagonal norm and edge
moduli are both formed from positive quantities. This removes the reported
negative tail integral; it does not claim a universal machine-rounding
certificate for arbitrarily ill-conditioned inputs.

Regression anchors integrate the independent attenuation-cusp transform on
a finite interval in both tails, and the near-cancelling joint directly with
a common numerator. All three fail against the previous implementation.
The cusp's finite tail is $8.09865\times10^{-4}$; its previous asserted
infinite-tail bound was $1.13095\times10^{-8}$.

Remaining scope: the frozen-carrier representation omits
$-L_{\rm esc}\delta\omega(E)$ and its intra-piece slope. Neither this fix nor
the earlier measurements prove that approximation's error fits the policy
share. The ledger remains `discrepancy` pending independent re-verification
and resolution of that production-spectrum gap. Coherent coordinate-cache
revision is incremented so earlier windows cannot be reused.

## Fresh-context review of attenuation correction (2026-10-06)

The verifier derived the transform of a piecewise field with signed affine
optical depth before reading implementation bodies. Units, attenuation signs,
the undamped single-piece limit, collinear splitting and $F\to0$ passed.
The signed complex denominators, positive Minkowski tail norms, stable
endpoint capture and derivative-majorant cross-term allowance match that
fixed-carrier field. The full production verdict remains **`discrepancy`**.

Independence boundary: an initial ledger extraction inadvertently displayed
implementation-specific Notes alongside Claim and Source. Implementation
bodies and this existing derivation were read only after the scratch
derivation. The reviewer did not implement the correction, but the input was
not perfectly blind to the owner's earlier account.

### Independent attenuation anchor

For $f(t)=\exp(-s|t|)$ on $[-L,L]$, the transform at frequency detuning $x$
is independently

$$
S(x)=\frac{2\left[s+e^{-sL}\left(-s\cos(xL)+x\sin(xL)\right)\right]}{s^2+x^2}.
$$

With $s=0.03$ Å$^{-1}$, $L=1000$ Å and carrier 1000 eV, integration over
$x\in[0.3,30]$ Å$^{-1}$ gives a finite-tail fraction
$2.0968692556\times10^{-4}$. The corrected infinite-tail bound at either
mirrored edge is $2.9570772063\times10^{-4}$. This independently confirms
the signed attenuation correction for this field, not the complete material
transport spectrum.

### Genuine gaps must retain both endpoints

Ten unit-amplitude pieces of length 1 Å, separated by 1 Å gaps, with one
electron and common carrier 1000 eV, have centres
$10^7+2j+0.5$ Å for $j=0,\ldots,9$. A common translation of the time origin
changes only the overall spectral phase. Removing that phase, their exact
transform is

$$
S(x)=\frac{2\sin(x/2)}{x}\sum_{j=0}^{9}e^{2ijx},
\qquad x=(E-1000\,\mathrm{eV})/H.
$$

The prior join tolerance, $10^{-3}\min(\Delta d_j,\Delta d_{j-1})+
4\times10^{-7}|l_j|$, merged all nine real gaps at this translated origin.
It replaced the separated endpoint terms by a cancelling joint. Integration
on $x\in[100,1000]$ Å$^{-1}$ gives a finite-tail fraction
$2.8471605140\times10^{-3}$, exceeding the previous asserted infinite-tail
bound $3.8590886478\times10^{-4}$ by 7.38 times. This counterexample needs
neither attenuation nor dispersion.

The owner replaced the physical relative tolerance with an allowance for
float64 endpoint reconstruction only,

$$
8\epsilon_{64}\left(|d_j|+|d_{j-1}|+
\tfrac12(\Delta d_j+\Delta d_{j-1})\right).
$$

`test_real_gaps_do_not_cancel_under_retardation_translation` pins the
independent finite integral at origins zero and $10^7$ Å. It fails on the
translated origin before the correction and passes afterward, retaining
all twenty free endpoints. Coherent cache revision 4 invalidates earlier
plans. The verifier independently rechecked this correction: zero merged
joints, finite-tail fraction $2.8471605140\times10^{-3}$ below the corrected
bound $3.9780245401\times10^{-3}$. The collinear-split and both production
reflection-tail anchors also passed in that context. A sub-roundoff gap
still cannot be distinguished from a reconstructed
joint by this allowance; no universal exact snapping-error certificate is
claimed.

### Remaining exact discrepancy: dispersive endpoint denominator

The production formation profile contains the escape-path phase slope. Its
endpoint denominator is

$$
z_j(E)=E-E_{\rm vac,j}
-H\frac{\Delta L_j}{\Delta d_j}\delta\omega(E)+iH\lambda_j,
$$

and the external midpoint phase contains $-L_{\rm mid,j}\delta\omega(E)$.
The collector discards the escape-distance change and does not capture the
midpoint escape distance. Its tail bound and support span therefore describe
a different fixed-carrier field unless these terms vanish or are separately
bounded. A small refractive decrement alone does not control the error: the
ratio of escape-path slope to retardation slope can amplify it.

An independent lossless affine-escape example uses constant
$\delta=10^{-5}$, $\Delta d=100$ Å, $\Delta L=10^6$ Å and
$E_{\rm vac}=1000$ eV. The exact detuning becomes $0.9E-1000$ eV. Its
finite upper-tail fraction on $[1100,1200]$ eV is 0.5098803723, against the
captured field's asserted infinite-tail bound 0.1256222573. The independent
sinc expression agrees with production `formation_profile` pointwise, with
maximum absolute difference $8.24\times10^{-13}$. This is an artificial
affine-escape field, not a material-specific transport reproduction.

For constant $\delta$, the physical Fourier coordinate is
$y=t-\delta L(t)$: piece duration becomes
$\Delta d-\delta\Delta L$ and carrier becomes
$E_{\rm vac}/(1-\delta\Delta L/\Delta d)$. Thus the captured Parseval
normalization and support-based spacing also need re-analysis; padding
window centres alone does not resolve the production claim. The ledger
retains `discrepancy`; no acceptance status is promoted.

## Owner building blocks for dispersive windows (2026-10-06)

The collector now retains $L_{\rm mid,j}$ and signed $\Delta L_j$ alongside
the existing attenuation data. These are the same escape endpoints used by
the production formation integral. The following owner derivation supplies
two building blocks; it is **not independent verification** and does not
yet replace the production seeder's frozen-carrier bound.

### Exact affine phase law

Write $H=\hbar c$ and define an affine surrogate
$\delta\omega(E)=sE+b$, continued over the real energy axis for Parseval.
It equals the material field only where that law is exact; a finite-band
normalization must account separately for its out-of-band power. On a piece,
the change of variable $y=d-HsL(d)$ has Jacobian

$$
k_j=1-Hs\frac{\Delta L_j}{\Delta d_j}>0.
$$

The complete field, including its midpoint phase and its formation slope,
is then a Fourier transform on the new coordinate. The mapped quantities are

$$
\Delta y_j=k_j\Delta d_j,\qquad
y_{\rm mid,j}=d_j-HsL_{\rm mid,j},\qquad
E'_j=\frac{E_{\rm vac,j}+Hb\Delta L_j/\Delta d_j}{k_j},
\qquad a'_{j,p}=\frac{a_{j,p}}{k_j},\qquad
\lambda'_j=\frac{\lambda_j}{k_j}.
$$

The energy-independent midpoint phase also contains $-bL_{\rm mid,j}$;
it remains continuous across a true joint together with the original
susceptibility phase. Endpoint transmissions are unchanged. For disjoint
mapped supports within each electron, Parseval therefore gives

$$
P'=\sum_{j,p}\frac{|a_{j,p}|^2}{k_j}\Delta d_j
\langle e^{-\tau_{\rm abs}}\rangle_j,\qquad
\int\sum_{e,p}|S'_{e,p}(E)|^2dE=2\pi H P'.
$$

The endpoint envelope and support step can now use the mapped row. The
private `_affine_dispersion_row` helper refuses nonpositive Jacobians,
overlapping or reordered mapped supports, and closing a genuine gap into a
joint whose phases have not been shown to agree. Its support check uses the
existing float64
endpoint reconstruction convention, whose sub-roundoff scope remains open.
The vacuum limit $s=b=0$ recovers every original quantity. Constant
$\delta$ is $s=\delta/H$, $b=0$: the counterexample's 100 Å piece becomes
90 Å with carrier $1000/0.9=1111.111\ldots$ eV, amplitude $a/0.9$,
power $P/0.9$, and Nyquist step $\pi H/90$.

### Uniform residual charge on a finite energy band

For a nonlinear law, let $r(E)=\delta\omega(E)-(sE+b)$ and require a
**certified uniform** bound $|r(E)|\le R$ throughout a band of width $W$.
An endpoint sample or fitted residual is insufficient. Inside a piece,
the actual and affine integrands differ by $e^{-ir(E)L(d)}-1$, so

$$
|e^{-ir(E)L(d)}-1|\le\min(2,R L_{\max,j}),\qquad
L_{\max,j}=|L_{\rm mid,j}|+\tfrac12|\Delta L_j|.
$$

Let $m_j=\langle e^{-\tau_{\rm abs}/2}\rangle_j$ be the mean amplitude
transmission. If $q_j=(\tau_{\rm end}-\tau_{\rm start})/4$ and $B_j$ is
the brighter endpoint's amplitude transmission, its stable integral is
$m_j=B_j[-\operatorname{expm1}(-2|q_j|)]/(2|q_j|)$, with limit $B_j$ at
$q_j=0$. Triangle inequality inside an electron and an independent sum
over electrons give the absolute finite-band error bound

$$
\int_{\rm band}\sum_{e,p}|S_{e,p}-S'_{e,p}|^2dE
\le W\sum_{e,p}\left[
\sum_{j\in e}|a_{j,p}|\Delta d_jm_j\min(2,R L_{\max,j})
\right]^2=\mathcal E.
$$

`_dispersion_residual_power_bound` evaluates this positive bound using the
original row. It includes both the midpoint phase and the escape slope
inside the formation integral. It handles signed attenuation and endpoint
underflow, and reduces to zero when $R=0$ or $W=0$. Its units are squared
field times eV; dividing by $2\pi H P'$ gives the affine Parseval charge.
For any sub-band with affine power $Y'$, the actual power lies between
$\max(0,\sqrt{Y'}-\sqrt{\mathcal E})^2$ and
$(\sqrt{Y'}+\sqrt{\mathcal E})^2$. Thus residual power must be combined
with the affine tail bound at the **amplitude** level; adding the two power
fractions directly omits their cross term. A physical finite-band relative
claim also needs a positive lower bound on its actual normalization.

Fast anchors compare the transformed damped exponential against production
formation for both signs of the slope and intercept, reproduce the original
constant-$\delta$ counterexample, and check the nonlinear residual bound
against direct finite integration. They do not supply a material-table
certificate. Remaining work: certify $R$ across Henke interpolation and
absorption edges, combine the charge with envelope and sampling budgets,
establish physical finite-band normalization, and obtain fresh-context
validation before changing the production policy or ledger status.

## Owner material-law enclosure and runtime audit (2026-10-06)

`coherent_dispersion.CoherentDispersionLaw` reconstructs the interpolation
used by the full output axis. Its source is the installed xraydb 4.5.8
`XrayDB._from_chantler`: the real anomalous factor uses an interpolating
cubic spline on the native table truncated three nodes beyond the query's
minimum and maximum; the imaginary factor uses linear interpolation in
log-energy and log-factor. The raw `Chantler` table has energy, f1 and f2
columns ([upstream schema](https://github.com/xraypy/XrayDB/blob/master/xraydb.schema)).
The adapter adds $Z$ to anomalous f1 exactly as `chi_0` does, uses the same
unit-cell volume and constants, and refuses unsupported table domains.

The full-axis extrema are part of the model. Independent local calls to
`refractive_index` can select a different spline: on HOPG, the batch
$[284.8,285.0,285.2]$ eV differs by about $10^{-5}$ Å$^{-1}$ from the
same nodes in an axis spanning $[100,6000]$ eV. The regression pins this
difference and the reconstructed law's agreement with the full-axis call.
The adapter retains the full selection when evaluated on a local subarray.

### Smooth-interval bounds

Every f1 spline knot and every f2 native knot is an interval boundary.
On each interval, translating a cubic to Bernstein form encloses its entire
range by the minimum and maximum Bernstein coefficients, including interior
extrema. The same construction bounds its first two derivatives. For f2,
log-linear interpolation is a positive power law $f_2(E)=cE^p$; hence the
imaginary susceptibility is proportional to $E^{p-2}$ and both derivatives
are bounded analytically with endpoint values and the exponent. Combining
the atom multiplicities gives bounds $C_0,C_1,C_2$ on
$|\chi_0|,|\chi'_0|,|\chi''_0|$ throughout the interval.

Let $m$ be a positive lower bound on $|n|$, obtained by separately enclosing
the real and imaginary parts of $1+\chi_0$ and using
$|n|^2=|1+\chi_0|$. An interval that cannot exclude a square-root zero is
refused. Differentiating $n^2=1+\chi_0$ gives

$$
n'=\frac{\chi'_0}{2n},\qquad
n''=\frac{\chi''_0}{2n}-\frac{(\chi'_0)^2}{4n^3}.
$$

With $w(E)=\delta\omega(E)=E[1-\operatorname{Re}n(E)]/H$,
$|1-n|=|\chi_0|/|1+n|\le C_0$ on the principal square-root branch.
For an interval ending at $E_+$ this supplies

$$
\sup|w'|\le\frac{C_0+E_+C_1/(2m)}{H}=M_1,\qquad
\sup|w''|\le\frac{C_1/m+E_+C_2/(2m)+E_+C_1^2/(4m^3)}{H}=M_2.
$$

The secant $sE+b$ of this **same full-axis law** therefore has uniform
residual $R\le M_2(E_+-E_-)^2/8$. This is a derivative enclosure, not a
sampled-residual estimate. The implementation adds outward float64
allowances for polynomial translation and phase evaluation. These are an
explicit numerical convention; they do not constitute a universal
interval-arithmetic proof for arbitrarily ill-conditioned spline solves.
Nonfinite certificates are refused. The non-Henke Thomson limit
$w(E)=[E-\sqrt{E^2-K}]/H$ above the plasma root supplies a separate analytic
derivative check. Vacuum gives $w=0$ in real arithmetic.

### Absolute excluded-power audit

The runner applies these interval certificates to each captured row and
splits again at its current window edges. Outside the fine window, the
affine row's tail bound supplies an absolute per-electron power upper bound
$Y'$. The finite-band residual charge $\mathcal E$ derived above then gives
$Y\le(\sqrt{Y'}+\sqrt{\mathcal E})^2$. If the affine coordinate map has
nonpositive durations, overlapping support, or closes a genuine gap, the
audit uses the finite-band L1 majorant

$$
Y'\le W\sum_{e,p}\left(\sum_{j\in e}|a_{j,p}|\Delta d_jm_j\right)^2,
$$

which requires no Fourier-map monotonicity or joint cancellation. The
inter-electron term is included through $1+F_{\max}(N_e-1)$ on each
interval. The audit reports absolute excluded-power bounds, their ratio to
the **frozen** reference, residual charges and fallback counts. Its phase
slope diagnostic uses the conservative span $D+HM_1\Delta L$, so the
reported step is $\pi H/[O(D+HM_1\Delta L)]$ with the policy oversampling $O$
(currently eight).
This is a phase-slope diagnostic, not an exact band-limit or sampling proof
for a nonlinear law.

These diagnostics are stored under `coherent_windows.dispersion` and
explicitly mark `relative_production_bound = false`. For the small HOPG
anchor, one row's conservative excluded-power bound is about 183.7 times
the frozen reference; this is **not an observed error**. A warning reports
the bound and phase-slope step on both cold and warm cache paths when the
frozen-reference fraction exceeds the existing two-side share. Coordinate
cache revision 5 includes a fingerprint of the actual trimmed material
tables, cell volume, atom multiplicities, interpolation model/version and
full-axis extrema; coherent coupling weight `B_ang2` is also keyed.

Owner checks cover HOPG and WSe₂ across absorption knots, full/local query
agreement, the analytic Thomson derivative, shifted resonances, a nonlinear
residual, and the nonmonotone-map L1 fallback. The ledger remains
`discrepancy`. A positive physical finite-band normalization and a sampling
error charge are still needed before these absolute bounds can govern the
automatic windows. Fresh-context verification and corrected remote ladders
remain required; current grid coordinates and the point budget are unchanged.

## Independent checkpoint verification (2026-10-06)

This fresh verifier reviewed checkpoint `20c857318bdc949444b003bfc035e7ee6dcddfe2`
only: the material-law enclosure, affine-map parameters, nonlinear residual
charge, and absolute finite-axis excluded-power audit. The source packet
supplied the physical laws, quantities, signatures, units, and limits. The
verifier inspected installed xraydb 4.5.8 `XrayDB._from_chantler` and recorded
a blind derivation in `/tmp/issue350-blind-derivation.md` before reading the
implementation, tests, ledger Notes, or owner derivation. The newer
normalization work is outside this review.

### Independent material derivative chain

With $C=r_e(hc)^2/(\pi V)$ and
$A(E)=\sum_j[Z_j+f'_j(E)+if''_j(E)]$, the supplied physical source gives

$$
\chi=-CA/E^2,\qquad
\chi'=-C(A'/E^2-2A/E^3),\qquad
\chi''=-C(A''/E^2-4A'/E^3+6A/E^4).
$$

Differentiating the principal square root and $w=E(1-\operatorname{Re}n)/H$
gives

$$
n'=\frac{\chi'}{2n},\qquad
n''=\frac{\chi''}{2n}-\frac{(\chi')^2}{4n^3},\qquad
w'=\frac{1-\operatorname{Re}n-E\operatorname{Re}n'}{H},\qquad
w''=-\frac{2\operatorname{Re}n'+E\operatorname{Re}n''}{H}.
$$

The xraydb source selects a table subsection from the full query extrema,
then uses a zero-smoothing cubic spline for anomalous f1 and a positive
power law for f2 between native log-interpolation knots. The implementation
matches that selection, adds $Z$ only to f1's constant term, and splits at
both sets of knots. Its Bernstein convex-hull bounds provide valid
real-arithmetic polynomial enclosures. Its direct differentiation of the
imaginary susceptibility uses powers $p-2$ and $p-3$, matching the chain
above. The positive lower bound on $|n|$, principal-branch inequality
$|1-n|\le|\chi|$, and resulting derivative bounds match independently.
The residual bound $M_2W^2/8$ is the endpoint-secant remainder on each smooth
interval. Units are $w$ in Å$^{-1}$, $w'$ in Å$^{-1}$ eV$^{-1}$, and
$w''$ in Å$^{-1}$ eV$^{-2}$. The explicit float64 padding remains a numerical
convention, not a proof of all interpolation-solve rounding errors.

### Independent affine and residual comparison

For $L=L_m+q\xi$ and $w=sE+b$, substitution into the original integrand gives

$$
\frac{E}{H}(d_m-HsL_m+(1-Hsq)\xi)
-\left(\frac{E_{\rm vac}}{H}+bq\right)\xi-bL_m
-\mathbf g\cdot\mathbf r_m.
$$

Thus $k=1-Hsq$, duration $k\Delta d$, center $d_m-HsL_m$, carrier
$(E_{\rm vac}+Hbq)/k$, amplitude modulus $|a|/k$, and signed attenuation
slope $\lambda/k$ agree with `_affine_dispersion_row` on its positive,
ordered, disjoint-support scope. Vacuum recovers the original row.
Continuous collinear splitting preserves the complete physical integral.

A terminology limitation matters: the returned row alone does **not**
encode the complete complex field. Its `amplitude` omits the factor
$\exp[-i(bL_m+\mathbf g\cdot\mathbf r_m)]$. The one-piece affine test supplies
$\exp(-ibL_m)$ externally. For this checkpoint's power/tail audit, the
energy-independent phase has unit modulus, and its complete endpoint value
is common at a true continuous physical joint; cancellation therefore
remains valid. This argument does not license treating arbitrary touching
rows with discontinuous escape or midpoint phase as continuous physical
joints. Such rows need explicit endpoint phases or the L1 fallback. No
new phase factor is needed in the independently scoped Parseval norm.

Writing $\rho=w-(sE+b)$ with $|\rho|\le R$ gives
$|\exp(-i\rho L)-1|\le\min(2,R|L|)$. Triangle inequality within each electron
and integration over bandwidth $W$ give

$$
\mathcal E\le W\sum_{e,p}\left[
\sum_{j\in e}|a_{j,p}|\int_j e^{-\tau/2}\,d\xi\;
\min(2,R L_{\max,j})\right]^2.
$$

The implementation matches this expression. The audit's looser cap
$W\min(2\sqrt{Q},R\sqrt{Q_L})^2$ also follows, where $Q$ and $Q_L$ are the
squared electronwise L1 totals without and with $L_{\max,j}$ respectively.
The brighter-endpoint transmission integral stays positive and stable for
either absorption slope, including underflow of the darker endpoint.

### Independent absolute audit comparison and adjudication

For each omitted interval, Hilbert-space triangle inequality and Cauchy
inequality across electrons give

$$
Y_{\rm production}\le
[1+(N_e-1)F_{\max}]
\left(\sqrt{Y_{\rm affine}}+\sqrt{\mathcal E}\right)^2.
$$

The audit includes this cross term and uses a finite-band L1 majorant when
the affine map is unsupported. Its use of `decoherence(lo)` is valid for the
production `decoherence_bound`: positive-energy Gaussian longitudinal
suppression is nonincreasing, so the lower endpoint bounds the interval;
`None` supplies $F_{\max}=1$. The private audit docstring should state that
restriction explicitly. An arbitrary nonmonotone callback is outside this
verified contract. The limits $F=0$ and $F=1$ give factors $1$ and $N_e$.

Focused canonical tests selected for dispersion and endpoint capture passed:
32 passed, 30 deselected. Independent scratch calculations additionally
formed the material derivative directly from the interpolants, checked
interior residuals against the certificates, and compared signed-absorption
L1 integrals with numerical quadrature. Across 1306 HOPG/WSe₂ smooth
intervals, the largest sampled derivative-to-bound ratio was 0.973416 and
the largest residual-to-bound ratio was 0.989030. Five absorption slopes,
including both signs and darker-endpoint underflow, matched independent
quadrature. These numeric checks supplement the
symbolic argument; interior sampling is not the certificate itself.

**Narrow verdict:** `rederived`, for the absolute float64 material-phase
power audit under continuous physical joint phases, production monotone
decoherence bounds, and the stated numerical/endpoint reconstruction
conventions. No divergent factor, sign, or exponent was found in that
scoped bound. The affine helper's exact-field terminology and the private
callback's monotonicity restriction should be clarified.

**Full claim verdict:** remains `discrepancy`. The exact unresolved
normalization convention is division by frozen-source $2\pi H P$ in place
of a proved positive finite-axis production-power lower bound. The
phase-slope step is a diagnostic, not a certified nonlinear sampling-error
charge. This report does not promote the complete claim or confer human
sign-off. Suggested ledger edit: retain `discrepancy`; add the independently
rederived scope and its two explicit interface restrictions to Notes.

## Independent conditional normalization review (2026-10-06)

This extension reviews the working normalization building blocks following
checkpoint `20c85731`: `coherent_normalization` and additive midpoint-phase
capture in `coherent_windows`, `lines/_setup`, and `lines/_per_hkl`. The
verifier first recorded `/tmp/issue350-blind-normalization-derivation.md`
from the same physical formation integral and proposed signature, before
reading these implementation bodies or tests. It verifies a **conditional
building block**, with externally certified sample norm uncertainties;
it does not verify a production normalization certificate.

### Sample floor and derivative envelope

Let $G(E)$ have polarization/electron components $S_{e,p}(E)$, and let
$C_p(E)=\sum_e S_{e,p}(E)$. A common phase may be removed from the entire row
without changing either norm. For the grouped norm only, a separate common
phase may also be removed within each electron. Electron-specific gauges
must not be used for the coherent vector $C$.

The physical phase derivative is $d/H-w'(E)L$. Removing the common phase
$Ed_0/H-w(E)L_0$ gives

$$
\partial_E\widetilde\Phi=(d-d_0)/H-w'(E)(L-L_0).
$$

With $|w'|\le M_1$, the affine piece's endpoint maxima give

$$
B_{e,p}=\sum_{j\in e}|a_{j,p}|\int_j e^{-\tau/2}\,d\xi
\left[\frac{\max_j|d-d_0|}{H}+M_1\max_j|L-L_0|\right].
$$

Triangle inequality yields a grouped derivative bound
$K_G=(\sum_{e,p}B_{e,p}^2)^{1/2}$, with electron-specific references allowed.
The coherent derivative bound instead uses row-wide references and
$K_C=[\sum_p(\sum_e B_{e,p})^2]^{1/2}$. The implementation uses midpoint
references of the relevant endpoint ranges, matching this derivation and
preserving common time-origin translation. Signed attenuation enters only
the stable positive L1 piece weights already reviewed above.

At an in-band sample $E_0$, externally certified absolute norm errors
$\epsilon_G,\epsilon_C$ establish
$A_G=\max(0,\|G_{\rm nominal}(E_0)\|-\epsilon_G)$ and the analogous $A_C$.
Reverse triangle inequality gives the squared amplitude floor
$[A-K|E-E_0|]_+^2$. On a side of length $t$, its exact integral is

$$
J(A,K,t)=A^2r\left[1-u+\frac{u^2}{3}\right],\qquad
r=\min(t,A/K),\quad u=Kr/A,
$$

for positive $A,K$. The limits are $J(0,K,t)=0$ and $J(A,0,t)=A^2t$.
Adding the left and right sides gives the interval power floor. The code
uses this stable expression, avoiding subtraction of nearly equal cubes.
The units are squared field times eV.

### Convex production blend

For arbitrary energy-dependent $F(E)\in[F_{\min},F_{\max}]$, let $I_G,I_C$
be the independently integrated grouped and coherent tents. The bound

$$
Y\ge(1-F_{\max})I_G+F_{\min}I_C
$$

is valid pointwise before integration. A second valid floor uses the shared
tent with $A_*=\min(A_G,A_C)$ and $K_*=\max(K_G,K_C)$: it lies below both
vector norms, and hence below every convex blend. Taking the larger of
these two integrated floors remains valid. `_physical_row_power_lower`
implements exactly this construction. It avoids the generally invalid
operation of taking the minimum of two integrated endpoint blends when
$F$ varies with energy. The limits $F=0$ and $F=1$ select the corresponding
individual norm; unknown $F\in[0,1]$ uses the shared tent. A zero individual
sample floor or an uncertainty covering that nominal norm gives no positive
power floor for that individual vector.

### Phase capture and independent checks

`capture_phase_rad` records the same geometric midpoint susceptibility
phase used by the production row, aligned with the row's `idx`/coefficients;
the collector applies the same formation-valid mask. The nominal helper
retains the complete phase $Ed/H-\mathbf g\cdot\mathbf r_m-wL_m$. Its common
reference subtraction preserves both grouped and coherent intensities.
For the affine surrogate, `phase_rad += intercept * escape_mid_ang` has the
required sign: the Fourier integrand contains $-bL_m$. This supplies the
midpoint phase metadata missing from the earlier standalone affine-row
representation. The production tail audit's monotone-callback restriction
is now explicit in its docstring.

Independent scratch calculations integrated the physical exponential
directly, without `formation_factor`: three energies, two polarizations,
four pieces, both absorption signs, and endpoint underflow matched the
nominal complex samples. Direct mixed-power integrals exceeded the returned
lower bound for $F=0$, $F=1$, varying $F\in[0.2,0.8]$, and varying unknown
$F\in[0,1]$. Lower-to-direct-power ratios were 0.169763, 0.137533, 0.079505,
and 0.079596 respectively. Four independently quadrature-integrated tents
covered zero amplitude, zero derivative, compact support, and a small
sample floor. Focused canonical tests passed: 20 passed, 39 deselected.
The scratch numerical allowances are comparison tolerances; they do not
supply production sample-uncertainty certificates.

**Conditional verdict:** `rederived`; no divergent sign, factor, gauge, or
convex-blend convention found. This assumes a matching certified material
derivative bound, correct captured geometry/amplitudes, and certified
absolute errors for the nominal sample norms, including interpolation,
formation, phase, and summation uncertainty. Float64 derivative/envelope
arithmetic retains the previously stated numerical convention.

**Full claim:** remains `discrepancy`. Supplying guessed or zero sample
errors does not certify the nominal physical sample; production sample
uncertainty and the nonlinear sampling-error charge remain unresolved.
Suggested ledger Notes may record this conditional normalization building
block as independently rederived, without promoting the complete claim.

## Owner integration and stricter FWHM gate (2026-10-06)

The nominal field reconstruction now has a real-source regression: the
captured HOPG rows reproduce the production coherent source, including its
grouped/flat blend and electron normalization, on the same trajectories.
The comparison uses an absolute guard from the L1 field, interpolation drift,
argument scales and summation count. Its largest guard stays below the
$10^{-3}$ intrinsic share of the largest source node. A fixed relative
comparison at a nearly cancelling node instead failed at $1.61\times10^{-8}$;
that numerical difference must be accounted for when certifying sample
uncertainty. The regression guard is not a universal production sample
certificate, and `_physical_row_power_lower` still requires supplied
certified absolute norm errors. It does not yet govern the automatic grid.

Issue #350 requests $10^{-3}$ for FWHM as well as yield and centroid. The
previous regression allowed the harness's $10^{-2}$ shape share. Tightening
that assertion on the same-trajectory anchor fails the former two-sample
default: $0.9667565728500449$ eV versus the explicit reference
$0.9654851669308755$ eV, a relative error $1.317\times10^{-3}$.
Changing the numerical policy to four samples per Nyquist step passes the
stricter gate. Peak height remains excluded. Point budgets remain enforced,
so affected large cases may require a larger explicit remote budget; no
coherent grid is silently coarsened to fit it. These checks do not remove
the outstanding production sample-uncertainty and nonlinear sampling charge.

## Owner conditional sampling interval (2026-10-06)

The following extends the conditional normalization construction. This is
an owner derivation with analytic regression evidence; fresh-context
verification of this extension is still required. It does not promote the
full claim or certify automatic grids.

For increasing samples $E_i$ on one smooth material interval $[a,b]$,
partition the whole interval at adjacent sample midpoints. Cell $i$ includes
the nearer band edge when its sample is first or last; samples need not
include $a$ or $b$. Let $A_{X,i}^-=\max(0,\widehat A_{X,i}-\epsilon_{X,i})$
and $A_{X,i}^+=\widehat A_{X,i}+\epsilon_{X,i}$ for $X=G,C$, where the
externally certified absolute errors enclose the nominal grouped and
coherent norms. With the previously derived uniform derivative bounds $K_X$,
triangle and reverse-triangle inequalities give, on each cell,

$$
[A_{X,i}^- - K_X|E-E_i|]_+^2
\le \|X(E)\|^2
\le [A_{X,i}^+ + K_X|E-E_i|]^2.
$$

The lower envelope integrates by $J$ above. For a side of length $t$ the
upper envelope has the positive integral

$$
U(A,K,t)=t\left[A^2+A(Kt)+\frac{(Kt)^2}{3}\right].
$$

Denote the two-sided cell integrals by $L_G,L_C,U_G,U_C$. If
$f_i^-\le F(E)\le f_i^+$ throughout the cell, valid mixed bounds are

$$
L_i=\max\left\{L_*,(1-f_i^+)L_G+f_i^-L_C\right\},\qquad
U_i=\min\left\{U_*,(1-f_i^-)U_G+f_i^+U_C\right\}.
$$

Here $L_*$ integrates the shared lower tent from
$\min(A_G^-,A_C^-),\max(K_G,K_C)$; $U_*$ integrates the shared upper
envelope from $\max(A_G^+,A_C^+),\max(K_G,K_C)$. These shared envelopes
lie below or above both norms pointwise, so they remain valid for any
convex blend. Summing over disjoint cells gives $L\le Y\le U$ for the
physical finite-band row power in squared-field times eV. Any quadrature
estimate $Q$ consequently has the absolute sampling-error charge

$$
|Q-Y|\le\max\{|Q-L|,|Q-U|\}.
$$

This charge does not assume an affine dispersive phase, monotone $F$, or
constant electron interference. It uses bounds on $F$ across each whole
cell, not merely at the endpoints. Separate smooth material intervals
must be handled separately; one derivative certificate cannot span an
uncertified interpolation discontinuity. For production units, apply the
same positive electron normalization to both bounds and the estimate.

`coherent_normalization::_sampled_power_bounds` implements these envelopes;
`::_physical_row_power_bounds` reconstructs nominal physical fields and
uses the material derivative certificate. In the zero-derivative,
zero-uncertainty constant-field limit both bounds coincide when $F$ is
known or the two norms agree. Decreasing maximum cell radii reduces derivative
drift, but adding samples need not monotonically tighten nearest-sample
intervals. Uncertainty in sample norms or cell form-factor bounds remains. Unknown
global $F\in[0,1]$ can leave a nonzero interval width even on a fine grid.

Analytic anchors integrate the two-electron fields
$S_1(E)=1$, $S_2(E)=0.6\exp(i\pi E^2)$ on $[0,1]$, independently of
the production sampler. The grouped norm is constant after electron-specific
gauges and the row-wide coherent derivative is bounded by $1.2\pi$.
Constant $F=0,0.4,1$ and varying $F=0.5+0.4\sin(7E)$ are enclosed;
the varying case uses cell enclosures from $|F'|\le2.8$ rather than an
endpoint monotonicity assumption. Refinement from 9 to 129 samples
contracts the nonconstant integral intervals by more than a factor of 8.
The physical two-piece sinc anchor also checks the complete captured-row
wrapper against a separate analytic formation/phase expression. Additional
limits retain nonzero sample uncertainty and unsampled endpoint strips;
invalid partitions and nonfinite certificate inputs are refused.

As with the conditional floor, envelope arithmetic follows the stated
float64 convention rather than directed interval arithmetic. Certified
production norm uncertainties, rounding charges, automatic-policy
integration, observable-specific centroid/FWHM error control, fresh-context
verification and corrected remote ladders remain required. The complete
ledger claim stays `discrepancy`.

## Independent conditional sampling verification (2026-10-06)

This fresh context derived the norm envelopes from the supplied triangle and
reverse-triangle inequalities and input signature before inspecting the
implementation bodies, tests, owner sampling section, or implementation-specific
ledger Notes. The independent derivation was recorded first in an external
scratch document. The comparison covers checkpoint `6c664de4` and the owner's
subsequent docstring qualifications, which change no executable expression.
The quantity is the finite-band mixed power

$$
Y=\int_a^b\left[(1-F(E))\|G(E)\|^2+F(E)\|C(E)\|^2\right]\,dE.
$$

For either field $X$, its intensity-preserving gauge and certified uniform
derivative bound imply

$$
\left|\|X(E)\|-\|X(E_i)\|\right|
\le \|X(E)-X(E_i)\|\le K_X|E-E_i|.
$$

If the nominal norm is $s_{X,i}$ with certified absolute uncertainty
$\epsilon_{X,i}$, the independently obtained pointwise envelopes are

$$
\ell_{X,i}(E)=\left[s_{X,i}-\epsilon_{X,i}-K_X|E-E_i|\right]_+,
\qquad
u_{X,i}(E)=s_{X,i}+\epsilon_{X,i}+K_X|E-E_i|.
$$

Their squared integrals over nearest-sample cells bound the component power.
The cells reach both band endpoints; an unsampled endpoint strip must not be
dropped. On a side of length $t$, direct integration gives

$$
\int_0^t(A+Kr)^2\,dr=A^2t+AKt^2+\frac{K^2t^3}{3},
\qquad
\int_0^t[A-Kr]_+^2\,dr
=A^2q-AKq^2+\frac{K^2q^3}{3},
$$

where $q=\min(t,A/K)$ for $K>0$ and $q=t$ for $K=0$.
The implementation's positive lower-integral polynomial is algebraically
identical. Clipping the lower nominal norm before subtracting drift is also
identical after the positive-part operation.

For cell-wide bounds $f_i^-\le F\le f_i^+$, positivity proves separately
that the mixed integral lies above $(1-f_i^+)L_G+f_i^-L_C$ and below
$(1-f_i^-)U_G+f_i^+U_C$. The common lower tent formed from the smaller
lower sample norm and larger derivative bound lies below both component
norms pointwise; its square therefore bounds every convex mixture below.
The analogous common upper envelope bounds every mixture above. Taking the
maximum of the two integrated lower bounds and the minimum of the two upper
bounds matches `_sampled_power_bounds`. This remains valid for rapidly varying
$F$; it does not interchange a pointwise minimum with integration.

The wrapper's polarization/electron norm is $\|G\|$, while summing electron
fields before the polarization norm gives $\|C\|$. Its common phase removal
preserves both. Electron-specific derivative gauges preserve grouped power;
the coherent derivative gauge must remain common across electrons. No relative
electron phase is removed before coherent summation. The existing derivative
envelopes and a matching certificate remain prerequisites, not certificates
manufactured by the new wrapper.

Units, positivity, and limiting cases pass: derivative drift has field units;
integrated squares have squared-field eV units; zero amplitude with zero errors
and derivative bounds gives zero; zero derivative with nonzero certified norm
uncertainty retains a nonzero interval. Pure grouped or coherent weights,
unknown form factors, near cancellation, and nonlinear relative phases retain
valid bounds. For any estimate $Q$, the independently obtained error bound is
$|Q-Y|\le\max(|Q-L|,|Q-U|)$.

### Refinement qualification and numerical evidence

Monotonic contraction under arbitrary inserted samples is false. On $[0,1]$,
take $G(E)=C(E)=1-E$, $K_G=K_C=1$, zero sample errors, and $F=1$.
A single sample at zero has lower bound $1/3$, equal to the exact integral.
After adding the exact zero-norm sample at one, the second cell contributes
zero to the lower bound and the result falls to $7/24$. Both remain valid.
Nearest-sample selection does not intersect all available sample cones.
Decreasing maximum cell radii controls derivative drift asymptotically;
convergence to a sharp mixed-power interval additionally requires vanishing
norm uncertainty and sufficiently tight cell-wide form-factor enclosures.
The owner narrowed the docstring accordingly; no bound formula changed.

Independent scratch checks used analytic fields with two electron amplitudes
$1$ and $-0.999\exp(2iE^2)$, comparing against adaptive integration of their
explicit power for $F=0$, $F=1$, $F=0.37$, and
$F=0.5+0.49\sin(31E)$. Interior-only samples exercised both endpoint strips.
A separate physical-row check evaluated an explicit two-piece sinc field with
nonlinear dispersion $\delta\omega(E)=10^{-9}E^2$ and varying form factor,
independently of the production sampler and envelope helpers. Additional exact
checks covered constant fields with zero and small uncertainties, zero power,
the refinement counterexample, and a band one float64 spacing wide whose
rounded midpoint creates a zero-width cell. All eight independent checks and
all 21 maintained normalization checks passed.

Finite arithmetic is conditional on the documented float64 envelope convention.
The midpoint expression avoids summing large same-sign endpoints. Degenerate
zero-width cells are accepted and contribute zero for finite envelope arithmetic;
nonfinite accumulated envelopes are rejected. This is not outward-rounded
interval arithmetic: underflow, rounded sample-cell geometry, and ordinary
polynomial/summation roundoff are not independently charged. No exact real-number
enclosure is certified at arbitrary floating-point scales.

**Verdict:** the conditional finite-band sampling inequalities and wrapper match
the independent derivation, `rederived` within their stated arithmetic and
external-certificate assumptions. The full automatic-grid claim remains
`discrepancy`: production norm-error certificates, rounding charges, automatic
policy integration, centroid/FWHM control, and corrected remote ladders remain
outside this verification. Suggested ledger Notes should record this conditional
verification and refinement qualification without promoting the complete claim.

## Independent captured-row SAMPLE verification (2026-10-06)

This verifier derived the piece integral and norm certificate before inspecting
`coherent_sample_certificates.py`. A subsequent scope extension supplied the
stored material-law expression; its pointwise enclosure was also derived before
inspection of `sample_dispersion_bounds`. The initial handoff left the sign of
the attenuation exponent implicit. Before implementation inspection, the owner
clarified that positive attenuation slope means decreasing transmission toward
increasing piece coordinate. The derivation below uses that explicit convention.

### Captured piece and stable integral

The source is the captured affine-escape piece field, with captured binary64
inputs treated as exact under the reconstruction convention. Let $h_j>0$ be the
piece duration, $H=\hbar c$, $q_j=\lambda_jh_j/2$, and

$$
v_j(E)=\frac{h_j(E-E_j)}{2H}-\frac{\Delta L_j w(E)}2.
$$

With $B_j$ the brighter captured endpoint transmission, the reconstructed
amplitude transmission is $T_j(x)=B_j\exp(-|q_j|-2q_jx)$ for
$-1/2\le x\le1/2$. For $s_j=\operatorname{sign}(q_j)$, taking $s_j=1$ at zero,
the substitution $x=s_j(t-1/2)$ gives

$$
\begin{aligned}
J_j(E)&=\int_{-1/2}^{1/2}T_j(x)\exp(2iv_jx)\,dx\\
&=B_j\exp(-is_jv_j)\operatorname{exprel}(-2|q_j|+2is_jv_j),\\
a_{pj}(E)&=c_{pj}h_jJ_j(E)
\exp\!\left(i\left[Ed_j/H-\phi_j-L_jw(E)\right]\right).
\end{aligned}
$$

Here $\operatorname{exprel}(z)=(\exp z-1)/z$, continuously extended to one at
zero. Its real argument is nonpositive, so the implementation avoids forming
$\exp(|q_j|)$ even when the dimmer endpoint underflows. The equivalent midpoint
form is $B_j\exp(-|q_j|)\sinh(-q_j+iv_j)/(-q_j+iv_j)$.
Units pass: $q_j$, $v_j$ and both phases are dimensionless, and $w$ has
inverse-length units. Signs pass: positive $q_j$ places the brighter endpoint at
$x=-1/2$. Limits pass: zero attenuation gives $B_j\operatorname{sinc}(v_j)$;
zero detuning gives $B_j(1-\exp(-2|q_j|))/(2|q_j|)$, tending to $B_j$;
strong damping decays instead of growing exponentially.

### Directed field and norm enclosure

For $|z|\le\rho\le1/2$, the independently derived series remainder is

$$
\left|\operatorname{exprel}(z)-\sum_{k=0}^{N}\frac{z^k}{(k+1)!}\right|
\le\frac{\rho^{N+1}}{(N+2)!}
\frac{1}{1-\rho/(N+3)}.
$$

The successive tail-term ratios are bounded by $\rho/(N+3)$. The code uses
$N=40$, $\rho=1/2$, and adds the remainder independently to both rectangular
components. Away from zero the interval quotient is valid. For a broad
zero-containing argument, the fallback rectangle follows from
$|J_j|\le\int T_j(x)\,dx\le B_j$; it may be loose but never divides by a
zero-containing interval.

Embedding the original floats before subtracting the first piece's $d_0$,
$\phi_0$, and $L_0$ removes the same unit-modulus gauge from every field.
Consequently both norms remain invariant. Interval substitution of the supplied
pointwise $w(E)$ enclosure includes material-phase uncertainty and all relative
phase and formation effects. Piece sums give $A_{pe}=\sum_{j:e_j=e}a_{pj}$,
and the physical norms are

$$
N_G=\sqrt{\sum_{p,e}|A_{pe}|^2},\qquad
N_C=\sqrt{\sum_p\left|\sum_e A_{pe}\right|^2}.
$$

Directed complex modulus, summation and square root enclose these norms.
For arbitrary finite nominal tensors, the returned binary64 nominal norms
$n_G,n_C$ need not come from any particular field evaluator. Embedding those
returned floats exactly and subtracting them from the independently enclosed
physical norms gives

$$
\epsilon_t\ge\sup_{N\in[L_t,U_t]}|N-n_t|,\qquad t\in\{G,C\}.
$$

This comparison charges nominal formation, phase, field summation and norm
rounding without separately proving those floating operations. Nonfinite
nominal norms and nonfinite error enclosures are refused.

### Stored-law pointwise dispersion

For the same full-axis-selected material data, the supplied source is

$$
\chi(E)=-\frac{P}{E^2}
\left(F_0+\sum_a n_a[f_{1a}(E)+if_{2a}(E)]\right),\qquad
w(E)=\frac E H\left(1-\Re\sqrt{1+\chi(E)}\right).
$$

The exact stored cubic is evaluated by interval Horner arithmetic after selecting
the same PPoly piece, including the right-sided knot convention. The imaginary
factor uses the native positive table ordinates:

$$
f_2(E)=\exp\!\left[(1-r)\log f_{2a}+r\log f_{2b}\right],\qquad
r=\frac{\log E-\log E_a}{\log E_b-\log E_a}.
$$

The selected table brackets contain the full query axis. For
$x=1+\Re\chi$, $y=\Im\chi$, the principal root obeys

$$
\Re\sqrt{x+iy}
=\sqrt{\frac{\sqrt{x^2+y^2}+x}{2}}.
$$

The radicand is analytically nonnegative; intersecting its computed interval
with the nonnegative half-line is therefore justified. The resulting phase
interval encloses the exact real evaluation of the **stored** coefficients,
prefactor and table values. It need not contain the independently rounded
NumPy `law(E)` value; that nominal discrepancy is charged by the norm comparison.
It certifies neither spline construction nor physical interpolation uncertainty.
The zero-susceptibility limit gives zero phase; to first order,
$w=-E\Re\chi/(2H)$, with positive phase for negative real susceptibility.

### Findings, corrections and independent evidence

The first implementation inspection exposed a complex interval constructor
failure for nonreal coefficients. The owner replaced the ambiguous one-argument
constructor with separate real and imaginary inputs. An independent subnormal
case then refuted the initial outward-conversion assumption: a physical coherent
norm of $4.2425934545755737990\times10^{-318}$ received an error rounded inward by
$2.3471722055995\times10^{-324}$ despite `round_ceiling`. The owner added one
outward binary64 successor for positive errors, retaining exact zero, and outward
successors/predecessors for phase endpoints. The corrected subnormal errors
exceed the independent norms. These corrections change interval conversion, not
the physical expression.

Independent direct quadrature first confirmed the piece integral at zero,
both attenuation signs, and $q=1200$. A separate 90-digit integral reference
passed 160 norm comparisons across signed slopes, dim-endpoint underflow,
broad dispersion intervals, large common gauges, zero fields, near cancellation,
subnormal amplitudes and arbitrary biased nominal tensors. A deliberately
rounded-away coherent residual of $10^{-23}$ remained enclosed. Both global
mpmath contexts retained their precision. The stored-law check independently
used polynomial power sums, ratio-form imaginary interpolation and the complex
principal square root: all 55 points passed across HOPG, WSe2, and the HOPG
constant-forward-factor mode, including knots and axis endpoints.

**Scoped verdict:** `rederived` for `_formation_interval`,
`sample_row_norm_certificate`, and `sample_dispersion_bounds` under the stated
exact-captured-input and exact-stored-interpolant conventions. The complete
`coherent-line-grid-windowed-resolution` claim remains `discrepancy`. This
verification does not certify upstream capture/coefficient/geometry uncertainty,
physical material-data error, spline construction, derivative or power-envelope
rounding, GPU reduction arithmetic, automatic policy integration, centroid/FWHM
acceptance, or remote convergence ladders. Suggested ledger Notes should record
this scoped SAMPLE verification without promoting the complete claim. The task
owner performs the final documentation build and rendered-math check.

The subsequent composition review of
`_certified_physical_row_power_bounds` found no API, gauge, or form-factor
mismatch: nominal samples and directed pointwise bounds use the same law and
energy nodes; norms and errors retain grouped/coherent column order; derivative
norms use the same reconstructed brighter-endpoint attenuation law. Different
per-electron or row-wide derivative gauges are permissible because the
corresponding norm is gauge invariant. The mixed-power convention remains
$(1-F)N_G^2+FN_C^2$, and supplied form-factor intervals must cover whole
nearest-sample cells. Strictly interior samples on a knot-free material band
avoid applying a derivative cone across an infinitesimal stored-PPoly joint
mismatch; unsampled endpoint strips are covered by the same open-branch limits,
and isolated endpoint values have zero integral measure. The reviewed regression
compares the composed interval against an independent constant-forward-factor
band integral with $F=0.4$. This composition matches the previously verified
conditional inequalities and needs no guessed norm errors. Its derivative and
power-envelope arithmetic retain their documented float64 qualification;
composition does not extend the scoped verdict to an outward-rounded band
integral or automatic-grid acceptance.

## Independent directed derivative and band-power verification

A fresh verifier derived this continuation from the intended quantity and stored
input conventions before reading its implementation or owner write-up. The
pre-inspection record is `/tmp/issue350-directed-blind.md`. The reviewed owners
are `dispersion_derivative_bound`, `row_derivative_bounds`, and
`_directed_power_bounds` in `coherent_sample_certificates.py`, the optional
`directed=True` path of `_sampled_power_bounds`, and their composition in
`_certified_physical_row_power_bounds`. The target remains

$$
Y=\int_a^b\left[(1-F(E))\|G(E)\|^2+F(E)\|C(E)\|^2\right]\,dE.
$$

Captured binary64 fields, real stored PPoly coefficients, the material
prefactor, and native positive log-linear imaginary-factor tables denote exact
real inputs. Sample norms carry independently certified absolute errors. This
scope excludes physical material-data uncertainty and upstream capture errors.

### Independent expressions and cheap filters

On one knot-free positive-energy band, put $H=\hbar c$,
$\chi=-P(f_1+if_2)/E^2$, and $n=\sqrt{1+\chi}$. Direct differentiation gives

$$
\begin{aligned}
f_1'&=3c_0x^2+2c_1x+c_2,\qquad x=E-E_{\rm knot},\\
f_2'&=\frac{f_2}{E}\frac{\ln(f_{2,b}/f_{2,a})}{\ln(E_b/E_a)},\\
\chi'&=-P\left[\frac{f_1'+if_2'}{E^2}
                  -\frac{2(f_1+if_2)}{E^3}\right],\\
w'&=\frac{1-\Re n-E\Re\{\chi'/(2n)\}}{H}.
\end{aligned}
$$

The implementation differentiates the original coefficients, then uses the
equivalent real-root expression. For $x+iy=1+\chi$,
$R=\sqrt{x^2+y^2}$ and $r=\sqrt{(R+x)/2}$,
$r'=[(xx'+yy')/R+x']/(4r)$. Intervals that cannot exclude $R=0$ or $r=0$
are refused. Units of $w'$ are inverse energy per length. Vacuum gives zero;
constant $f_1$ and weak susceptibility give
$w'=-Pf_1/(2HE^2)$ to leading order, checking the sign and energy power.

For a captured piece with duration $\Delta d$, signed
$q=\lambda\Delta d/2$, and brighter endpoint amplitude $B$, its absolute
integral weight and residual phase-slope majorant are

$$
\begin{aligned}
W_{pj}&=|c_{pj}|\Delta d_j B_j
          \frac{1-e^{-2|q_j|}}{2|q_j|},\\
D_j&=\frac{\max_{\rm endpoints}|d-d_r|}{H}
       +M_1\max_{\rm endpoints}|L-L_r|,\\
B_{pe}&=\sum_{j\in e}W_{pj}D_j,\\
K_G&=\sqrt{\sum_{p,e}B_{pe}^2},\qquad
K_C=\sqrt{\sum_p\left(\sum_e B_{pe}\right)^2}.
\end{aligned}
$$

The ratio has limit one at $q=0$ and is valid for both attenuation signs.
Affine geometry makes endpoint maxima sufficient. Grouped norms permit one
reference gauge per electron; coherent sums require a common row reference.
Subtracting the common captured origin before forming endpoints preserves
short flights at large clocks. The references themselves may be any fixed
real values; all subsequent distances enclose rounding. Zero coupling gives
zero derivative norm, whose units are field amplitude per energy.

Let $S_i$ be a nominal norm, $\epsilon_i$ its certified error,
$A_-=[S_i-\epsilon_i]_+$, and $A_+=S_i+\epsilon_i$. Triangle inequalities give
$[A_--K|E-E_i|]_+^2$ and $(A_++K|E-E_i|)^2$ as power envelopes. For one side
of a cell of width $r$, their integrals are

$$
\begin{aligned}
I_-&=A_-^2t-A_-Kt^2+K^2t^3/3,
       \qquad t=\min(r,A_-/K),\\
I_+&=A_+^2r+A_+Kr^2+K^2r^3/3.
\end{aligned}
$$

For $K=0$, $I_-=A_-^2r$. Interval minimum endpoints enclose $t$ even when
uncertainty straddles the clipping transition. The implementation evaluates
the equivalent factored lower primitive; its interval enclosure includes all
correlations conservatively. A zero lower-amplitude endpoint may safely return
zero. Actual stored midpoint floats define the exact cell partition; cell
width subtraction, sample errors, integration, blending, and totals are
directed. Abstract cone samples may be band endpoints; the composed physical
helper requires strictly interior samples to avoid stored-PPoly joint values.

On a cell where $F\in[F_-,F_+]\subseteq[0,1]$, separate nonnegative coefficient
bounds justify

$$
\begin{aligned}
Y_{\rm cell,-}&=\max\{I_{\rm common,-},
                      (1-F_+)I_{G,-}+F_-I_{C,-}\},\\
Y_{\rm cell,+}&=\min\{I_{\rm common,+},
                      (1-F_-)I_{G,+}+F_+I_{C,+}\}.
\end{aligned}
$$

The common lower cone uses the smaller lower amplitude and larger derivative
bound; the common upper uses the larger upper amplitude and larger derivative
bound. These inequalities hold pointwise before integration, including where
sector envelopes cross. Choosing one constant extreme form factor after
integrating each sector would require an additional ordering argument. Outward
binary64 extraction includes a successor/predecessor correction for subnormal
conversion, while exact zero upper power remains zero.

### Implementation comparison, corrections, and verdict

The initial review found that a rounded band midpoint could select the next
native interpolation segment when the upper endpoint was a knot. The owner
changed selection to the segment immediately right of the lower endpoint.
The owner also added finite aligned-array, positive-duration, nonnegative
transmission, and integer-electron checks before the row's empty return. Both
corrections were inspected; the resulting formulas match the independent
expressions above. The composed helper passes the same law and sample nodes
to the directed sample, derivative, and power paths, preserving grouped and
coherent column order.

Independent 100-digit polynomial power sums, ratio-form imaginary interpolation,
and differentiation of the complex principal square root checked 27 material
derivatives across HOPG, WSe2, and constant-forward-factor HOPG. All lay below
their uniform derivative bounds. Separate 100-digit exact cone primitives
checked 123 cases: one or several samples, endpoint samples, clipping and
adjacent cutoff floats, zero derivatives, norm errors exceeding nominal norms,
fixed or unrestricted form factors, and normal through subnormal powers.
Every returned lower/upper float enclosed the corresponding reference bound.

**Scoped verdict:** `rederived` for the directed stored-law derivative,
captured-row derivative norms, optional directed power-envelope path, and their
composed stored-input band enclosure. Units, limiting cases, and
signs/conventions pass. Suggested ledger Notes should record this scope and
the two corrected findings. The full `coherent-line-grid-windowed-resolution`
claim remains `discrepancy`: automatic policy integration, upstream uncertainty,
production reduction arithmetic, centroid/FWHM acceptance, and remote
convergence ladders are outside this verification. The default float64 cone
path retains its documented qualification. Only a human may mark the complete
claim `signed-off`; the task owner performs the final documentation build and
rendered-math check.

## Owner production-band probe after directed arithmetic

The same-trajectory regression now exercises the composed directed helper
against the production coherent reducer on the small HOPG transport: 30 keV,
tilt 5 degrees, azimuth 45 degrees, 1 micrometre, three electrons, seed zero,
100 fs bunch, one mosaic node. The band is a half-eV neighbourhood of the
first captured row's median vacuum carrier, clipped to one material interval.
At these keV energies the numerical production form factor underflows to zero,
so this comparison isolates its grouped branch. Full-axis endpoints accompany
the fine band query, preserving production's material-spline selection. The
sum of the row enclosures, divided by the production electron count, contains
the 257-node trapezoid result and has a positive lower bound. This comparison
does not prove the production reducer's floating arithmetic or quadrature
error separately.

A separate bounded CPU probe used the same case construction and retained one
transport for both certificate rungs. It captured two rows and 110 pieces;
its selected band was [1167.3158857288324, 1167.8158857288324] eV. The
production fine-band yield was 6.901490066476968e-10 in source units.

| Interior certificate nodes per row | Lower yield | Upper yield | Interval width / fine-band yield | Certificate wall time |
| --- | --- | --- | --- | --- |
| 9 | 1.0440063433516951e-10 | 4.45723912945707e-9 | 6.30710 | 0.521 s |
| 33 | 3.6211737247642883e-10 | 1.2447140735833333e-9 | 1.27885 | 1.543 s |

These are single-run diagnostic timings. The selected band is not a
whole-spectrum normalization or a dominant-line shape benchmark. The
first-derivative norm cones remain conservative: these two rungs do not meet
the intrinsic 1e-3 share for this band. Consequently this helper cannot simply
replace the automatic policy's warning with a passing tolerance certificate.
Automatic use still needs an efficient complete-band error budget, tighter
integration bounds or validated refinement, centroid/FWHM control, and
corrected remote ladders. No coordinate grid is coarsened or acceptance
promoted by this probe.

## Owner curvature enclosure (2026-10-07)

This continuation addresses the slow contraction of the first-derivative
cones. It is an implementation-context derivation, awaiting fresh-context
verification; the complete claim remains `discrepancy`.

### Stored material and captured-field second derivatives

On one open, knot-free stored-law interval, differentiate the same original
cubic and native log-linear imaginary factor. With the notation above,

$$
\begin{aligned}
f_1''&=6c_0(E-E_{\rm knot})+2c_1,\\
f_2''&=f_2\nu(\nu-1)/E^2,
\qquad \nu=\frac{\ln(f_{2,b}/f_{2,a})}{\ln(E_b/E_a)},\\
\chi''&=-P\left[f''/E^2-4f'/E^3+6f/E^4\right].
\end{aligned}
$$

For $x+iy=1+\chi$, $R=\sqrt{x^2+y^2}$ and
$r=\sqrt{(R+x)/2}$, the exact real derivatives are

$$
\begin{aligned}
R''&=\frac{x'^2+y'^2+xx''+yy''}{R}
       -\frac{(xx'+yy')^2}{R^3},\\
r''&=\frac{R''+x''}{4r}-\frac{r'^2}{r},\\
w''&=-\frac{2r'+Er''}{H}.
\end{aligned}
$$

`dispersion_derivative_bound(order=2)` substitutes directed intervals into
these expressions, including original-coefficient differentiation. Selection
uses the branch immediately right of the lower band endpoint. Intervals that
cannot exclude zero refractive roots are refused as on the first-order path.
The bound has units of inverse length per energy squared. Vacuum gives zero.
For constant real forward factor with $A=P f_1$, the independent expression
is $w=(E-\sqrt{E^2-A})/H$ and
$w''=A/[H(E^2-A)^{3/2}]$.

For the gauge-transformed captured piece, write its energy-dependent phase as
$\theta(E,d,L)=E(d-d_r)/H-w(E)(L-L_r)$ plus energy-independent terms.
The first-order endpoint majorant $D_j$ above bounds $|\theta'|$, and
$M_2\max_{\rm endpoints}|L-L_r|$ bounds $|\theta''|$. Thus

$$
\left|\frac{d^2}{dE^2}e^{i\theta}\right|
\le D_j^2+M_2\max_{\rm endpoints}|L-L_r|.
$$

`row_derivative_bounds(order=2)` multiplies this quantity by the same
positive signed-attenuation piece weights and combines polarization/electron
majorants exactly as in the first-order bound. Both orders use identical
fixed references. Separate grouped gauges and a common coherent gauge
preserve the corresponding power and its derivatives. The result bounds the
second derivative of the **field vector**, rather than the second derivative
of its norm. At a zero-attenuation single-piece resonance, the gauge-removed
field is $\Delta d\operatorname{sinc}[\Delta d(E-E_j)/(2H)]$; its second
derivative magnitude is $\Delta d^3/(12H^2)$, covered by the endpoint majorant.

### Directed trapezoid charge and composition

For either sector, let $K_1$ and $K_2$ bound the field-vector first and second
derivative norms. Product differentiation and Cauchy--Schwarz give

$$
P=\|A\|^2,\qquad |P''|\le2(K_1^2+K_0K_2).
$$

On two consecutive sampled energies with exact stored-float distance $h$,
take $K_0=\min(A_{+,i},A_{+,i+1})+K_1h$. The Lipschitz bound from either
endpoint proves this uniform norm upper bound. Certified endpoint norm
intervals provide lower and upper endpoint powers. The classical trapezoid
remainder then yields

$$
\begin{aligned}
J_-&=\max\left(0,\frac h2[A_{-,i}^2+A_{-,i+1}^2]
                      -\frac{h^3}{6}(K_1^2+K_0K_2)\right),\\
J_+&=\frac h2[A_{+,i}^2+A_{+,i+1}^2]
                      +\frac{h^3}{6}(K_1^2+K_0K_2).
\end{aligned}
$$

For global $F\in[F_-,F_+]$, separate nonnegative coefficient bounds give
the same valid mixed-sector lower and upper blends used above. No derivative
of $F$ or constant-$F$ approximation enters. Unsampled endpoint strips use
the directed first-order cones. All power, remainder, strip summation, and
binary64 extraction arithmetic is outward. `_curvature_power_bounds`
intersects this whole-band interval with the original cone interval; hence
it cannot widen the result. `_certified_physical_row_power_bounds(curvature=True)`
composes the stored-law, field, sample, and integral bounds, requiring at
least two strictly interior samples and global form-factor bounds. Its
default path is unchanged. Interior truncation contracts quadratically for
fixed finite derivative bounds; shrinking endpoint strips is also necessary
for full-band quadratic contraction. Sample uncertainty and form-factor
slack remain.

### Owner evidence and practical limit

Analytic regression checks cover affine vector power with a second-order
error charge, nonlinear two-electron phase interference with both endpoint
strips, the physical single-piece sinc limit, common clock translation,
exact zero and subnormal positive powers. Independent high-precision
constant-forward formulas and complex-root differentiation of the stored
HOPG/WSe2 laws are enclosed. These are owner checks, not a fresh-context
verdict.

The same CPU transport and band as the previous probe give:

| Samples per row | Cone interval width / fine yield | Curvature interval width / fine yield | Cone / curvature wall time |
| --- | --- | --- | --- |
| 9 | 6.30710 | 6.30710 | 0.518 / 0.626 s |
| 33 | 1.27885 | 0.670595 | 1.543 / 1.659 s |
| 129 | 0.319699 | 0.0598001 | 5.527 / 5.746 s |

The 129-sample curvature interval is
$[6.731722688933507\times10^{-10},\;7.144432499317292\times10^{-10}]$
and contains the production fine-band yield
$6.901490066476968\times10^{-10}$. The fixed endpoint margin is one percent
of the band on each side; it is included in the interval. Timings are
single-run diagnostics on the three-electron CPU anchor. This rung improves
the width by about a factor of 5.35 but still exceeds the intrinsic $10^{-3}$
share. Complete-band budgeting, tighter bounds/refinement, centroid/FWHM
control, fresh-context verification and corrected remote ladders remain
required before automatic-policy acceptance. No acceptance item or ledger
status is promoted.

## Owner complex-field interpolation enclosure (2026-10-07)

The curvature power charge above contains a squared first-derivative term,
which can remain large when a field's direction rotates even though its norm
changes little. This extension interpolates the complex field vector and
charges its second derivative directly. The derivation and evidence here are
owner checks, awaiting fresh-context verification. The full claim remains
`discrepancy`.

### Exact interpolation integral and remainder

In one fixed sector gauge, let $A_0,A_1$ be the physical field vectors at
energies $a,b$, with $h=b-a$. Their linear interpolant is
$L(x)=(1-x/h)A_0+(x/h)A_1$ for $0\le x\le h$. Direct integration gives

$$
\begin{aligned}
J_L&=\int_0^h\|L(x)\|^2\,dx\\
&=\frac h3\left[\|A_0\|^2+\Re\langle A_0,A_1\rangle+\|A_1\|^2\right]\\
&=\frac h6\left[\|A_0+A_1\|^2+\|A_0\|^2+\|A_1\|^2\right].
\end{aligned}
$$

The last positive sum avoids cancellation for nearly opposite endpoint
fields and is the implemented form. Directed complex endpoint enclosures
bound every term, including sample/material uncertainty and arithmetic.

For a twice differentiable vector field with $\|A''\|\le K_2$, the linear
interpolation Green kernel gives the same vector remainder bound as the
scalar interpolation theorem, because its kernel has one sign. In particular,

$$
\|A(x)-L(x)\|\le\frac{K_2}2x(h-x),\qquad
\|A-L\|_{L^2(a,b)}\le K_2\sqrt{\frac{h^5}{120}}=R.
$$

The second result follows from
$\int_0^h x^2(h-x)^2\,dx=h^5/30$. Minkowski and reverse Minkowski then give

$$
\boxed{
\left[\sqrt{J_{L,-}}-R\right]_+^2
\le\int_a^b\|A(E)\|^2\,dE
\le\left[\sqrt{J_{L,+}}+R\right]^2.
}
$$

`_interpolated_power_bounds` evaluates this enclosure with directed
arithmetic and outward binary64 extraction, retaining exact zero and
subnormal positive powers. The derivative charge has amplitude times
square-root energy units, matching $\sqrt{J_L}$; the resulting power integral
has field-squared times energy units. An affine complex field has zero
remainder and the exact integral above. Opposite endpoint fields give a
positive integral rather than cancelling it. Global form-factor bounds mix
nonnegative grouped/coherent sector integrals as above, including varying
$F(E)$ without a constant-$F$ approximation.

### Matching sector gauges

Complex field interpolation is sensitive to energy-dependent phase gauges,
although each individual sample norm is invariant. A derivative bound in one
gauge cannot be paired with samples in another. `row_phase_gauges` therefore
chooses fixed binary64 $d,L$ offsets relative to the first captured piece.
Its grouped sector has a reference for each electron; its coherent sector
uses one common reference, with electron-dependent coherent references
explicitly refused. Midpoints of relative endpoint ranges are a numerical
choice to reduce phase radii; their exact geometric optimality is unnecessary.
Each chosen stored float is an exact fixed reference for the enclosure.

`sample_row_norm_certificate(field_capture=...)` now optionally exposes its
directed polarization/electron fields and material phase interval **before**
norm reduction, in the same first-piece gauge used by the existing certificate.
The captured nested tuples are immutable; the hook does not alter the default
norm/error computation. For sector references $d_r,L_r$, the composed helper
rotates the captured samples by

$$
\exp\left(-i[Ed_r/H-w(E)L_r]\right).
$$

The grouped sector applies its references before flattening polarization and
electron coordinates. The coherent sector sums electron fields first and
applies the common reference. `row_derivative_bounds(phase_gauges=...)` uses
those same exact offsets for the endpoint slope/curvature majorants, retaining
directed origin subtraction. Thus the second-derivative certificate bounds
the exact field that is interpolated. The material phase interval and its
uncertainty participate in the rotation as well as formation/midpoint capture.

`_certified_interpolated_row_power_bounds` composes these pieces on one open,
knot-free stored-law band with at least two strictly interior samples. Endpoint
strips retain directed norm cones; the whole-band result intersects the
first-order cone interval. This certifies the reconstructed stored-input
field, with the same upstream capture/interpolant exclusions as the existing
helpers. It does not certify the floating production reduction or a spectral
centroid/FWHM.

### Owner regression and production-band probe

Analytic tests pin affine and quadratic field integrals, nonlinear
interpolation contraction, signed absorption with real material dispersion,
grouped and coherent blends, near-opposite/subnormal endpoint fields, and
variable form factors. Independent high-precision piece integrals lie inside
the composed intervals and their relative width is below $10^{-3}$ on the
small synthetic captured-row cases. Tests also pin unchanged norm/error
outputs when the capture hook is used, clock translation invariance, and
refusal of non-common coherent references. The production same-trajectory
regression covers the HOPG band with all three methods and confirms that the
field interpolation enclosure is narrower than the original cones.

The probe reuses the preceding three-electron CPU transport, two captured
rows, 110 pieces, band $[1167.3158857288324,1167.8158857288324]$ eV, and fine
production-band yield $6.901490066476968\times10^{-10}$. Here the endpoint
margin is $h_{\rm band}/(2N)$ for $N$ samples, so its strips shrink with
refinement. Both methods use the same sample coordinates at each rung.

| Samples per row | Curvature width / fine yield | Field interpolation width / fine yield | Curvature / interpolation time |
| --- | --- | --- | --- |
| 33 | 0.671743 | 0.0819596 | 1.668 / 1.777 s |
| 129 | 0.0474996 | 0.00533220 | 5.939 / 6.283 s |
| 513 | 0.00346960 | 0.000337573 | 22.285 / 23.030 s |

The 513-sample field interval is
$[6.900278558067526\times10^{-10},6.902608313285128\times10^{-10}]$.
Its width divided by its positive lower endpoint is about $3.38\times10^{-4}$,
below the intrinsic $10^{-3}$ integration share. It contains the fine
production-band quadrature. The 257-node numerical production quadrature is
a comparison value; the enclosure itself follows from the stored-field
certificate rather than an assumed accuracy of that quadrature. Timings are
single-run CPU diagnostics; this finite three-electron probe is not a remote
thick-target ladder.

This achieves the desired integration precision on the selected band. It
does not establish a full-spectrum normalization or full-axis sampling
budget. Efficient subdivision over material knots and line windows,
form-factor enclosures, centroid/FWHM error control, upstream uncertainties,
fresh-context verification, and corrected remote ladders remain open.
Automatic grid behavior and acceptance/status are unchanged.

## Independent scoped verification of curvature and field interpolation

Fresh verifier, 2026-10-07; checkpoints `447fb5bd` and `bddc6634`, evaluated
at `bddc66345ecb69c72ee05030319680f62a01e10c`. This review covers the optional
second-derivative, curvature, field-capture, gauge, and field-interpolation
building blocks in `coherent_sample_certificates.py` and
`coherent_normalization.py`. It does not promote the full validation claim.

### Independence and source convention

The verifier did not implement either checkpoint. After reading the repository
map, physics-validation skill and methodology, the verifier recorded the blind
source derivation in `/tmp/350-verifier-blind.md` before reading implementation
bodies, maintained tests, this owner write-up, or implementation-specific ledger
Notes. The source was the supplied exact stored-input attenuated piece integral,
principal-root material law, and grouped/coherent power definition. The
comparison then inspected the named definition owners and reconstructed the
source separately in `/tmp/test_350_independent.py`.

The initial blind notes interpreted “log-linear” as an affine logarithm versus
energy. The stored-law docstring clarified that the native convention is affine
logarithm versus **log energy**. The corresponding independent correction is
$f_2(E)=f_{2,a}(E/E_a)^\alpha$,
$\alpha=\ln(f_{2,b}/f_{2,a})/\ln(E_b/E_a)$, hence
$f_2'=\alpha f_2/E$ and $f_2''=\alpha(\alpha-1)f_2/E^2$.
All source-to-code and numerical comparisons below use that stored convention;
this is a clarification of the source convention, not an implementation
 discrepancy.

Captured binary64 coefficients, geometry, constants, phases, damping slopes,
carrier energies, sample energies, and returned gauge offsets are exact inputs.
The scope excludes the physical accuracy of upstream data, interpolant
construction, capture construction, and production floating reduction.

### Independent derivative and curvature derivation

With $S=F_0+\sum_a N_a(f_{1,a}+if_{2,a})$ and
$\chi=-PS/E^2$, differentiate the original stored cubic and native power-law
interpolants:

$$
\begin{aligned}
\chi'&=-P\left(S'/E^2-2S/E^3\right),\\
\chi''&=-P\left(S''/E^2-4S'/E^3+6S/E^4\right),\\
n'&=\frac{\chi'}{2n},\qquad
n''=\frac{\chi''}{2n}-\frac{(\chi')^2}{4n^3},\qquad n=\sqrt{1+\chi},\\
w''&=-\frac{2\operatorname{Re}n'+E\operatorname{Re}n''}{H}.
\end{aligned}
$$

The owner's real-root/radius differentiation is algebraically equivalent.
Rejecting bands where its real root cannot be separated from zero is
conservative; the accepted bands exclude interpolation knots and root
singularities. Vacuum gives zero material derivatives. The derivative units
are inverse length per energy and inverse length per energy squared.

In a fixed reference gauge $(d_r,L_r)$, the piece phase derivative is

$$
\theta_j'(E,x)=\frac{d_j-d_r+\Delta d_j x}{H}
 -(L_{\mathrm{mid},j}-L_r+\Delta L_jx)w'(E),\qquad
\theta_j''=-(L_{\mathrm{mid},j}-L_r+\Delta L_jx)w''(E).
$$

Let $D_j$ bound the absolute first expression and $Q_j$ bound the absolute
second expression uniformly on the cell and $x\in[-1/2,1/2]$. The attenuation
mass is nonnegative and even in signed damping:

$$
m_{pj}=|c_{pj}|\Delta d_j B_j
\frac{1-\exp(-2|q_j|)}{2|q_j|},\qquad
m_{pj}\big|_{q_j=0}=|c_{pj}|\Delta d_jB_j.
$$

The piece first- and second-derivative majorants are $m_{pj}D_j$ and
$m_{pj}(D_j^2+Q_j)$, respectively. Summing pieces within each electron and
then taking the appropriate grouped or coherent vector norm gives $K_1,K_2$.
The owner implements these triangle majorants with directed relative
endpoint geometry. Cancellation can only make them looser. A separate fixed
reference for each electron preserves grouped power; coherent power requires
one common reference. The sampled fields and derivative majorants use the
same exact stored references.

For either sector $P=\|A\|^2$,

$$
|P''|\le 2\left(K_1^2+K_0K_2\right),\qquad
K_0\le\min(\|A(a)\|_+,\|A(b)\|_+)+K_1h.
$$

The trapezoid charge is therefore
$h^3(K_1^2+K_0K_2)/6$, matching `_curvature_power_bounds`.
Endpoint norm uncertainty is charged before squaring; lower amplitudes are
clipped at zero. The optional curvature composition retains the first-order
cones on the endpoint strips and intersects two enclosing whole-band
intervals. The default path continues to use the pre-existing cones.

### Independent interpolation derivation

For the linear interpolant $L$ of a fixed-gauge vector field,

$$
\begin{aligned}
J_L&=\int_0^h\|L\|^2\,dx
=\frac h3\left(\|A_0\|^2+
\operatorname{Re}\langle A_0,A_1\rangle+\|A_1\|^2\right)\\
&=\frac h6\left(\|A_0+A_1\|^2+\|A_0\|^2+\|A_1\|^2\right),\\
\|A-L\|&\le\frac{K_2x(h-x)}2,\qquad
\|A-L\|_{L^2}\le K_2\sqrt{h^5/120}=R,\\
\left[\sqrt{J_{L,-}}-R\right]_+^2
&\le\int_0^h\|A\|^2\,dx
\le\left(\sqrt{J_{L,+}}+R\right)^2.
\end{aligned}
$$

The $1/120$ follows from integrating $x^2(h-x)^2/4$.
The positive sum-of-squares form avoids subtractive cancellation for opposite
endpoints. The owner evaluates endpoint fields as directed complex rectangles,
so endpoint uncertainty is included directly in the interval for $J_L$; no
additional nominal-sample error charge is needed in this interpolation step.
The immutable nested tuple capture precedes norm reduction. The composed
helper rotates those intervals in the matching grouped/common gauges, adds
endpoint-strip cones, and intersects the original whole-band cone result.

For arbitrary $F(E)\in[F_-,F_+]$, nonnegative sector integrals $I_G,I_C$
permit the conservative combination

$$
(1-F_+)I_{G,-}+F_-I_{C,-}
\le I_{\mathrm{physical}}
\le(1-F_-)I_{G,+}+F_+I_{C,+}.
$$

Both new bounds use this combination. They require neither constant $F$ nor
$F'$ or $F''$; taking only the minimum and maximum convex combinations of the
sector integrals would not suffice when $F$ varies and the sector power
difference changes sign. Broad $F$ bounds can prevent either method from
tightening. Directed interval arithmetic, outward extraction plus a successor
for positive upper bounds, and lower clipping preserve zero and subnormal
positive power.

### Independent numerical evidence and verdict

Nine scratch comparisons passed using the canonical project test runner with
`PYRITE_MC_BACKEND=cpu`. References used 75–85 decimal digits, stored-coefficient
power sums, the direct principal complex square root, and an exponential
antiderivative of the attenuated piece integral. They did not call owner
formation, sample-certificate, derivative, or normalization helpers to build
the reference quantity.

- HOPG and WSe2 native interpolants on $[1099.8,1100.2]$ eV: first- and
  second-derivative bounds enclosed independent high-precision differentiation
  at eleven points. Bound/maximum-observed ratios were $1.0014313,1.0033700$
  for HOPG and $1.0018924,1.0197774$ for WSe2. These samples support the
  symbolic interval proof; they are not a substitute for a uniform bound.
- A four-piece, two-polarization, two-electron WSe2 row used signed damping,
  unequal carriers, affine escape, a common $10^{10}$ Angstrom clock,
  and opposing coefficients within one electron. Independent differentiated
  grouped/common-gauge field vectors lay below the supplied second-derivative
  majorants at both band endpoints and its midpoint.
- With seventeen strictly interior samples and $F=0.65$, independent
  power integration gave $15443.1354709906397$. Curvature returned
  $[15443.099404983044,15443.164701808297]$; interpolation returned
  $[15443.109694252631,15443.149608376929]$. Both include endpoint strips.
- With $F(E)=0.5+0.3\sin(11(E-1100))$, independent integration gave
  $14760.2622876937329$, enclosed by
  $[12549.291938827117,17013.536547618798]$ from both compositions with
  global bounds $[0.2,0.8]$. Form-factor uncertainty dominates this case.
- Opposite affine complex endpoints reproduced the exact integrated power
  $2s^2/3$ for $s=1$, a binary64 subnormal $s=10^{-320}$, and $s=0$.
  Positive underflow retained a positive upper bound; zero retained zero.

The maintained five-module scoped suite also passed: **120 tests**, with
12 production/warning diagnostics. Its capture-hook comparison confirms
bitwise identical default norm/error outputs; its directed rectangles exercise
endpoint uncertainty and its coherent-gauge rejection protects the sector
invariance requirement.

**Scoped verdict: rederived.** Units, vacuum/zero/affine limits,
signed-damping symmetry, curvature factors, interpolation constant, gauge
convention, varying-form-factor treatment, and outward extraction agree.
No divergent factor, sign, exponent, unit, or convention was found in the
requested checkpoints. This is independent verification of these stored-input
building blocks. The full claim remains **discrepancy**: automatic integration,
upstream uncertainty, full-axis normalization/sampling, centroid/FWHM, and
corrected remote ladders remain outside this review. The task owner retains
the final documentation build and rendered-math check.

## Owner disjoint-band composition and work budget

`coherent_band_audit.certify_row_band_power` composes the verified smooth-band
field-interpolation primitive over a requested union of disjoint energy bands.
Omitting `bands` requests the material law's complete axis; explicitly supplied
gaps stay omitted. This owner construction does not change the automatic
grid or certify a grid's numerical yield by itself.

Let the requested bands be partitioned at every stored material knot into
intervals $I_i$. For the same captured row, material law and physical form
factor throughout the audit, let $L_i,U_i$ enclose the nonnegative integrated
power on $I_i$. Additivity gives

$$
\sum_i L_i\le Y_B=\sum_i\int_{I_i}P(E)\,dE\le\sum_i U_i.
$$

The implemented sums use directed arithmetic and outward binary64 extraction.
Isolated one-sided spline-knot values have zero integral measure; strictly
interior samples and the primitive's endpoint strips cover each interval.
Overlapping input bands are refused to prevent double counting; touching
bands are allowed. No region between supplied bands enters the integral.
An empty requested set has exact zero power and no evaluations.

With aggregate enclosure $[L,U]$, the audit uses the outward relative width

$$
\eta=\frac{U-L}{L}\quad(L>0)
$$

as its uncertainty charge. For a supplied finite nonnegative numerical yield
$Q$, monotonicity of $Q/Y$ gives the distinct quadrature-error charge

$$
\epsilon_Q=\max\left(\left|\frac QL-1\right|,
                         \left|\frac QU-1\right|\right)
\ge\sup_{Y\in[L,U]}\frac{|Q-Y|}{Y}.
$$

Thus a narrow integration enclosure is insufficient if the production-grid
yield lies far from it. With $L=0<U$, the width charge is infinite; positive
$Q$ has unbounded relative error, while $Q=0$ has upper error one. Certified
zero uses width zero, zero error for $Q=0$ by convention, and unbounded relative
error for positive $Q$. Units of $L,U,Q$ are row-field squared times eV; both
charges are dimensionless. Ratios and their extraction are directed,
including subnormal inputs and near cancellation in the width numerator.

### Refinement and refusal

Initialization evaluates every smooth interval using `initial_samples`
strictly interior midpoint nodes. Every call counts its complete sample set;
no cached or reused sample work is assumed. If the initial sample requirement
exceeds `max_evaluations`, refusal occurs before any partial integral is
claimed. The exception then carries no certificate because full requested
coverage has not been established.

After complete initialization, the largest interval uncertainty is selected
for a larger sample rung. Old and new enclosures of the same unchanged
integral are intersected:

$$
L_i\gets\max(L_{i,\rm old},L_{i,\rm new}),\qquad
U_i\gets\min(U_{i,\rm old},U_{i,\rm new}).
$$

This preserves validity and retains progress even if changed sample locations
produce a looser primitive enclosure. Inconsistent intersections are refused.
The audit returns successfully only when its directed global width charge
meets `relative_tolerance` or the requested integral is certified zero.
Exhausted work budgets raise `BandPowerBudgetError` with the complete but
possibly loose current enclosure, cumulative evaluation count and measured
relative width. No representable interior sample set means refusal rather
than silently treating a positive-width band as empty. The evaluation budget
is separate from an automatic coordinate-grid point budget.

Form-factor bounds may be fixed or supplied by a callback for each whole
smooth interval. Such bounds must enclose the actual form factor everywhere
on that interval. More field samples cannot necessarily reduce this
uncertainty. In particular, the conservative default $F\in[0,1]$ may exhaust
a work budget that suffices for an exactly specified sector; this is a
certificate limit, not permission to assume a constant physical form factor.

### Owner checks and scope

Maintained analytic sinc checks enclose the integral on the full requested
axis and on gapped bands split at synthetic material knots. Their selected
grouped-sector form factor is exact, and their uncertainty and separately
checked numerical-yield error meet the intrinsic $10^{-3}$ share. Small
budgets refuse with work counts; insufficient initial coverage carries no
quantitative certificate. Additional checks cover empty-band zero,
adjacent-float refusal, invalid/overlapping bands, invalid numerical yields,
and directed relative errors on subnormal power intervals. These are owner
checks; fresh-context composition verification is recorded separately.

The helper certifies only its requested stored-input row integral under the
existing primitive assumptions and supplied form-factor enclosures. Full
production normalization requires every relevant row and band to be covered;
omitted spectral regions and a production quadrature need their own charges.
Automatic-policy integration, upstream capture/interpolant uncertainties,
centroid/FWHM control and corrected remote ladders remain open. The full
`coherent-line-grid-windowed-resolution` status stays `discrepancy`.

### Independent verification: disjoint-band assembly and relative bounds

The fresh verifier recorded the blind derivation in
`/tmp/350-band-composition-blind.md` before reading `coherent_band_audit.py`,
its maintained tests, or the owner derivation. This review certifies only the
composition of already valid stored-input interval enclosures. It introduces
no transport law and does not change the full claim's `discrepancy` status.

Let $B$ be the requested finite union of disjoint energy bands and let
$P(E)\geq0$ be the unchanged stored-input row power. Splitting every band at
all native material interpolation knots gives smooth intervals $I_i$.
Their shared endpoints have zero integration measure. Positivity and integral
additivity therefore give

$$
Y=\int_B P(E)\,dE=\sum_i Y_i,
\qquad
L_i\leq Y_i\leq U_i
\quad\Longrightarrow\quad
\sum_i L_i\leq Y\leq\sum_i U_i.
$$

All bounds use the primitive's matching field gauges and physical units.
Endpoint strips remain included by that primitive. Overlapping bands must
be refused or converted into a union before addition; this API refuses them.
Requested gaps remain omitted, so this result is not full-axis power.

For a second valid enclosure of the same unchanged interval, intersection
preserves validity:

$$
[L_i,U_i]\cap[\ell_i,u_i]
=[\max(L_i,\ell_i),\min(U_i,u_i)].
$$

An empty intersection is a contradiction and must refuse. Calls count every
sample evaluation, including samples on earlier refinement rungs. A budget
that cannot initialize every smooth interval supplies no whole-band
certificate; exhaustion after initialization can report a complete loose
certificate. Strictly increasing representable interior samples are required;
an adjacent-float interval that cannot provide them must refuse.

Writing the outward total bounds as $[L,U]$, positivity gives the conservative
relative interval uncertainty

$$
\frac{U-L}{Y}\leq\frac{U-L}{L}\quad(L>0).
$$

For an independently supplied finite nonnegative numerical yield $Q$, the
monotonicity of $Q/Y$ gives a separate error certificate:

$$
\sup_{Y\in[L,U]}\frac{|Q-Y|}{Y}
=\max\left(\left|\frac{Q}{L}-1\right|,
           \left|\frac{Q}{U}-1\right|\right)
=\max\left(0,\frac{Q}{L}-1,1-\frac{Q}{U}\right)
\quad(L>0).
$$

Both expressions require outward arithmetic, including subtraction and float
conversion; positive underflow must never become certified zero. For $L=0<U$,
the width bound is infinite; the error bound is infinite for $Q>0$ and one for
$Q=0$ over positive admissible integrals. At certified zero $L=U=0$, the API
uses the explicit convention of zero error for $Q=0$, infinity for $Q>0$, and
zero width. Relative error at a zero true integral otherwise has no ordinary
definition. A passing width certificate alone places no restriction on an
arbitrary production-grid yield $Q$.

**Implementation comparison.** `certify_row_band_power` splits the unchanged
requested bands at every stored knot, calls the certified primitive on each
whole interval, intersects refinements, and counts all samples.
`_sum_intervals` accumulates directed interval sums and converts outward.
`BandPowerCertificate.relative_width_upper` and `relative_error_upper` match
the expressions and zero conventions above. Overflow and unrepresentable
sample sets refuse. No factor, sign, exponent, unit, or convention discrepancy
was found within this scoped composition review.

**Independent numerical evidence.** Scratch checks used `Fraction` exact
rational arithmetic on 105 enclosures and six supplied yields per enclosure,
including zero, binary64 subnormals, adjacent normal floats, and extreme
dynamic range; 100 additional outward sums enclosed their exact sums.
A separate analytic constant-power oracle checked disjoint gaps, knot splits,
refinement, evaluation accounting, overlap refusal, incomplete-initialization
refusal, complete partial certificates, and adjacent-float sample refusal.
These checks ran independently of maintained tests through the canonical
project runner, with CPU backend selected for the scratch file.

For a single unattenuated support of duration $d=100$ angstrom, unit amplitude,
zero centre and escape distances, and resonance $E_0=1000$ eV, the independent
reference was

$$
P(E)=d^2\operatorname{sinc}^2\!\left(
\frac{(E-E_0)d}{2\hbar c}\right),
\qquad \operatorname{sinc}(x)=\frac{\sin x}{x}.
$$

Using the exact stored float for `HBARC_EV_ANG`, 70-digit independent
quadrature over $[990,994]\cup[997,1003]\cup[1007,1010]$ eV gave
$Y=128939.9108259816345758916$ in row-field-squared times eV units.
With exact grouped-sector form factor bounds, the API returned
$[128672.32481866598,129212.25211307115]$ after 28 sample evaluations,
with relative width upper $0.004196141595841069$, below the requested $0.005$.
The reference is enclosed; the separately checked numerical-yield error bound
is conservative. Three independent scratch tests passed. The latest maintained
band suite passed 13 tests.

**Scoped verdict:** units pass; limiting cases pass; signs and conventions pass;
independent derivation matches; `rederived` for band composition and relative
bounds only. The full windowed-grid claim remains `discrepancy`: this review
does not certify automatic-grid acceptance, omitted bands, production-grid
quadrature, or upstream stored-input exclusions. Only a human may sign off.

## Reduced thick-target remote execution smoke (2026-10-07)

A reduced copy of `hopg_short` completed under automatic windowed resolution
on the lab host through `pyrite remote`: job `issue350_smoke-3`, SLURM 1099,
one uncached case, started 10:23:47 and completed 10:48:01 PDT. The runner
reported **1,447 seconds** for the case. The job requested one GPU and float64
through a task-local `PYRITE_FP64=1` wrapper; this is execution evidence, not
a measured GPU performance comparison.

The isolated catalog selected HOPG at 60 keV, thickness 100,000 angstrom,
polar/azimuth angles 45/135 degrees, a 5 by 5 mm finite footprint, 0.1 mm
beam FWHM, and a Gaussian longitudinal RMS duration of 0.001 fs. Line
transport used 200 electrons, bremsstrahlung one electron, seed 1, four
reflections `(0 0 ±2)` and `(0 0 ±4)`, and `emission="both"`. Windowing was
enabled with an explicit 8,000,000-coordinate budget. Earlier task-local
attempts refused 33,098,177 coordinates for a 1 mm beam and 5,141,939
coordinates for the 0.1 mm beam against a 5,000,000 budget; neither refusal
silently coarsened the grid.

The saved checkpoint `hopg@issue350_smoke-dfda0b618cad` contains one record.
Reading that record on the remote host confirmed **5,141,939 coordinates**
over 10–6,000 eV, below the stated budget, with minimum stored and cast
spacing 0.0005720689655142053 eV. `E_grid`, `spec`, and `spec_coherent` are
matching finite float64 arrays; saved bremsstrahlung and characteristic
arrays are finite as well. Resolved metadata retains the window pieces,
coherent row diagnostics, material fingerprint, and `backend_dtype=float64`.
The complete parameter SHA-256 is
`dfda0b618cad75b51ec1277d53ecaeed8883ed85a0bc9d383b93ca7851be7a80`.

The exported source snapshot was clean revision
`3ab2811ed94eb320e9bbbaf4265dac1e1b3e6fb8`, payload digest
`fee14e7c7fd615433600249dab5d7541125724aea95d5d94de0eea2cad3aab2f`,
synced at 16:54 UTC. This predates the branch rebase and the new disjoint-band
audit; the smoke does not exercise that audit. No separate Monte Carlo runs
were compared to estimate grid error.

**Accuracy remains uncertified.** The stored dispersion audit explicitly
reports `relative_production_bound=False`. Its largest finite-axis
excluded-power bound is 74,534.30595852331 times the frozen reference, for
row 2, and its all-electron material phase-slope step is
0.000030716735157461954 eV. The runner emitted the corresponding precision
warning. These are conservative diagnostics, not measured grid errors.
This smoke demonstrates that a reduced thick-target case runs and persists
both spectra; it does not establish the intrinsic-source tolerance,
centroid/FWHM convergence, or the full envelope claim. Ledger status remains
`discrepancy`.

## Owner continuation: varying Gaussian form-factor bounds

The band audit previously refined only field samples inside each fixed smooth
material interval. A whole-interval form-factor enclosure consequently did
not contract. Even constant sector powers could exhaust the evaluation budget
with a smooth, exactly specified varying form factor. The polynomial-sector
regression reproduced a relative width of 0.0172903 after 500 evaluations
against a requested 0.002 share before correction.

For an active finite-footprint Gaussian longitudinal sector, the unchanged
production law in `lines/_setup.py::_prepare_spectrum` is

$$
F(E)=\exp\!\left[-\left(\frac{E\sigma_z}{H}\right)^2\right],
\qquad H=\hbar c.
$$

On a nonnegative band $[a,b]$, its derivative is nonpositive, so

$$
F(b)\leq F(E)\leq F(a),\qquad a\leq E\leq b.
$$

`coherent_form_factor.py::gaussian_form_factor_bounds` evaluates these endpoint
bounds with directed interval arithmetic and outward binary64 conversion.
The energy endpoints, the stored RMS length in angstrom and the stored
`HBARC_EV_ANG` are exact inputs. The exponent is dimensionless; $\sigma_z=0$
gives exactly one. For an exponent at least 1000, a zero lower bound and the
smallest positive binary64 upper bound safely enclose the positive Gaussian,
including extreme finite inputs. Positive underflow is never certified zero.
This helper does not apply to an empirical infinite-slab form factor or an
inactive decoherence sector. It does not charge construction of the stored
RMS length, evaluation rounding in the production reducer or device casts.

The band audit now bisects its widest uncertain interval when the caller
supplies a form-factor-bound callback. Each child gets its own whole-interval
form-factor enclosure and the initial number of field samples. Material knots
and requested gaps remain boundaries of the partition. Fixed bound pairs
retain sample-count refinement. Both child evaluations must fit the remaining
cumulative budget before a parent is replaced; otherwise the complete current
certificate is returned in `BandPowerBudgetError`. Unrepresentable interior
samples or splits still refuse.

Each partition gives a valid total-power interval $[L_k,U_k]$ for the same
unchanged requested field. Their intersection is also valid:

$$
\max_k L_k\leq Y\leq\min_k U_k.
$$

The audit retains this whole-band intersection across subdivisions. Thus
looser child bounds do not erase a valid parent bound, and inconsistent
intersections refuse. The returned leaf intervals describe the latest
partition; their sum may be wider than the retained total certificate.
Evaluation counts include discarded parents and previous sample rungs.

Owner regressions cover the fixed-interval failure, exact polynomial-sector
integration, budget exhaustion before a complete split, retained parent bounds,
Gaussian endpoint/subnormal bounds and invalid inputs. A physical analytic
two-electron row with opposite fields has coherent sector zero and power

$$
P(E)=2\operatorname{sinc}^2\!\left(\frac{E-1000\,{\rm eV}}{2H}
\,1\,{\rm \mathring A}\right)[1-F(E)].
$$

The subdivided certificate encloses its independent 70-digit integral over
990--1010 eV and separately bounds numerical-yield error within $10^{-3}$.
These implementation-owner checks are complemented by the independent scoped
review below; the full claim stays `discrepancy`. Automatic-policy
integration, actual production-grid comparison, centroid/FWHM and corrected
remote ladders remain open.

## Owner continuation: weighted spectrum-yield audit

Captured row fields previously retained the polarization coefficients,
formation law and phases, but omitted the production mosaic intensity weight.
That is sufficient for a relative bound on an isolated row, but it cannot
reconstruct a spectrum that averages several crystal orientations. Production
adds reflection/orientation powers incoherently and divides by the incident
macro-electron population:

$$
Y=\frac{1}{N_e}\sum_r w_r Y_r,
\qquad w_r\geq0.
$$

The weights multiply intensities, not complex field amplitudes. They are not
renormalized over captured rows, and the incident population is not replaced
by the number of surviving electron identifiers in a row. The coherent
coefficient hook now exposes the current `capture_mosaic_weight` alongside
`capture_phase_rad`; its existing five-argument signature is unchanged.
`CoherentRowCollector` stores that weight in `CoherentRowField.mosaic_weight`,
with a unit default for synthetic or historical single-orientation captures.
No production field arithmetic changes.

`coherent_spectrum_audit.py::audit_captured_spectrum_yield` applies the row
band certificate to every positive-weight row. Directed arithmetic encloses
the weighted sum and division:

$$
\frac{1}{N_e}\sum_r w_r L_r
\leq Y\leq
\frac{1}{N_e}\sum_r w_r U_r.
$$

Zero-weight rows contribute exactly zero. Positive weighted power below
binary64's subnormal range retains a positive upper bound and does not become
certified zero. Extracted Gaussian endpoints and the weighted lower bound are
checked against their original directed endpoints: the underlying float
conversion can round inward at subnormal values despite its requested
rounding mode. The failing Gaussian cases at exponents near 744, 746 and 750,
and the weighted sinc lower-bound case, are now regression anchors. A shared
form-factor enclosure must cover the physical factor
for every supplied row on the whole interval; the analytic Gaussian callback
is applicable to the active finite-footprint sector. The caller must supply
the complete row capture, matching material law and incident population.
These are unchanged exact stored inputs, including each weight.

The cumulative work budget reserves enough initial sample evaluations for
every remaining contributing row. A row that cannot certify its interval
within the remaining budget raises `SpectrumPowerBudgetError`, with its
original row index, total evaluations used and optional **row-only** loose
certificate. No incomplete spectrum certificate is returned. Requested bands
are materialized once, so a generator cannot silently omit bands on later
rows. Material knots and requested gaps retain the row helper's semantics.

The supplied numerical yield $Q$ must integrate exactly those requested bands
and carry the same photons per steradian per incident electron units. The
audit separately evaluates the directed relative error bound

$$
\sup_{Y\in[L,U]}\frac{|Q-Y|}{Y}
=\max\!\left(0,\frac{Q}{L}-1,1-\frac{Q}{U}\right),
\qquad L>0.
$$

`within_tolerance` tests this bound, not just the interval width. Thus an
accurate integral enclosure rejects an inaccurate production-grid yield.
The existing explicit zero conventions apply: certified zero accepts zero
yield, while a positive-power interval without a positive lower floor cannot
certify a zero yield to a subunit relative tolerance.

Owner checks enclose an independent 70-digit weighted sinc integral with two
unequal row weights and three incident electrons, including disjoint bands
provided as a generator; the same enclosure rejects a 10% excess yield.
Budget tests cover initialization refusal and exhaustion on the first and a
later row without mislabelling a row certificate as a spectrum certificate.
Invalid weights/populations and positive weighted underflow are pinned.

On the existing three-electron 30 keV HOPG transport, a two-by-two mosaic
quadrature reconstructs the production spectrum from the weighted captured
fields within the intrinsic $10^{-3}$ share. Separately, one reflection's
0.001 eV band around its median carrier passes the weighted audit against a
257-node production quadrature on those same trajectories. The material-law
axis remains 10--5000 eV during the production evaluation and audit; only
the requested integration band is narrow. This is a band comparison, not
full-axis or automatic-grid acceptance.

These implementation-owner checks are complemented by the independent scoped
validation of the Gaussian/subdivision and weighted-composition extensions
below. Capture reconstruction and material construction uncertainty,
full-axis policy integration, centroid/FWHM acceptance and corrected remote
ladders remain open; the full ledger claim stays `discrepancy`.

## Independent scoped verification of Gaussian and weighted-yield extensions

A fresh verifier reviewed frozen HEAD `59d8d26509103c7607739e92d5a21cf103d2b74d`,
restricted to checkpoints `89e8ebdd` and `59d8d265`. The blind derivation was
recorded in `/tmp/350-weighted-review-blind.md` before inspecting implementation
bodies, this owner write-up, ledger Notes, or maintained tests. The verifier
read the complete validation methodology and changed only this document.
The following verdict concerns the Gaussian endpoint bounds, subdivision and
composition extensions conditional on valid row integral primitives. It does
not promote the full window-envelope claim.

### Source derivation and cheap filters

The supplied source identity is the existing
`coherent-inter-electron-decoherence` Gaussian law. Treat the captured binary64
energy endpoints, RMS length, `HBARC_EV_ANG`, row weights and material
coefficients as exact stored inputs. For $E\geq0$ and $\sigma_z\geq0$,

$$
F(E)=\exp[-(E\sigma_z/H)^2],\qquad H=\hbar c,
\qquad F(b)\leq F(E)\leq F(a)\quad(a\leq E\leq b).
$$

The exponent is dimensionless with energy in eV and length in angstrom.
The zero-length or zero-energy limit gives one; nonzero Gaussian values remain
mathematically positive even when binary64 exponential evaluation underflows.
Therefore a positive upper bound must survive underflow. Since
$\exp(-1000)<2^{-1074}$, a certified exponent of at least 1000 admits the
safe enclosure $[0,2^{-1074}]$. Monotonicity selects the stop endpoint for the
lower bound and the start endpoint for the upper bound.

Integral additivity and positivity give a complete-domain bound by summing
bounds over a disjoint partition. Both children of every subdivision must
cover the unchanged parent before replacing it. If every complete partition
certifies the same integral $I$, intersections remain valid:

$$
\max_k L_k\leq I\leq\min_k U_k.
$$

The retained interval need not equal the sum of the latest leaf intervals.
This is a conditional composition theorem: it does not independently prove
the physical-row interpolation primitive used to construct those intervals.

Production adds reflection/orientation powers incoherently. For exact stored
nonnegative intensity weights and positive incident integer population,

$$
Y=\frac{1}{N_e}\sum_r w_r I_r,\qquad
\frac{1}{N_e}\sum_r w_r L_r\leq Y\leq
\frac{1}{N_e}\sum_r w_r U_r.
$$

Weights multiply intensity, and neither captured row count nor surviving
row electron identifiers replace the incident population. Units are photons
per steradian per incident electron after integrating energy. No additional
weight normalization is permitted. Zero-weight rows contribute exactly zero.

For a separate finite nonnegative numerical yield $Q$ and positive floor,
monotonicity of $Q/Y$ gives

$$
\sup_{Y\in[L,U]}\frac{\lvert Q-Y\rvert}{Y}
=\max\left(\left\lvert\frac{Q}{L}-1\right\rvert,
           \left\lvert\frac{Q}{U}-1\right\rvert\right),\qquad L>0.
$$

With $L=0$ and $Q>0$, the upper bound is infinite. With $Q=0<U$, it is one;
with certified zero and zero numerical yield, it is zero by convention.
An interval-width bound alone cannot accept the supplied numerical yield.
Evaluation work includes every sampled point on every rung and every row.
A loose certificate may describe a complete row domain; it cannot imply
complete spectrum coverage after another row fails.

Units, zero/underflow limits, positivity, incoherent weight placement and
incident-population conventions pass these filters.

### Implementation comparison

`gaussian_form_factor_bounds` uses directed interval arithmetic before
selecting monotone endpoint bounds. Its conversion checks compare the
converted binary64 endpoints with the original directed endpoints and move
outward, including subnormal conversions. The exponent-threshold branch
preserves the smallest positive upper float. This matches the blind result.

`certify_row_band_power` checks that both child evaluations fit before replacing
the parent, counts discarded parents and sample rungs, preserves requested
gaps and material-knot boundaries, and retains intersections of complete-domain
bounds. Exhaustion returns only the last complete row certificate through an
exception. `audit_captured_spectrum_yield` reserves initialization work for all
remaining positive-weight rows; exhaustion returns no partial-spectrum
acceptance. Its directed weighted sum and incident division, including the
lower-endpoint subnormal correction, match the derived composition.

The production path assigns `capture_mosaic_weight` from the same `wm` used
in the CPU coherent power accumulation and the existing GPU intensity
accumulation branches. The five-argument coefficient-hook call is unchanged.
The collector copies this weight without changing amplitudes. The setup
obtains `Ne` from the incident segment population or the explicit electron
limit; `_spectrum.py::_finalize_spectrum` divides accumulated powers by that
same setup population. The checkpoint diff changes only capture metadata and
its description in these production owners, preserving field arithmetic.
The Gaussian helper requires the exact stored RMS length used in setup;
reconstructing it with a different unit-conversion rounding is outside this
agreement.

### Independent numerical evidence

Scratch test `/tmp/test_350_independent.py` supplies seven fresh-context checks;
it is transient evidence, not a maintained regression anchor.

- 262 Gaussian bands: 524 endpoint comparisons against independent
  180-digit Decimal exponentials or the analytic extreme-underflow bound.
  Cases include neighboring floats near exponents 708--750 and 1000,
  zero energy/length, smallest subnormals, maximum-scale finite products,
  and deterministic logarithmic input sampling over 600 decades.
- 2000 scalar error bounds compared with exact rational arithmetic on the
  stored floats, plus five explicit positive/zero-floor conventions.
- 303 positive weighted aggregations checked against exact rational
  multiplication, addition and incident division, including positive results
  below the smallest subnormal, zero weights and populations unrelated to row
  identifiers. Evaluation counts agree with all active-row samples.
- Five subdivision budgets verify retained parent bounds, exact rational
  partition width, both-child coverage and counted samples. Two spectrum
  budget checks verify initial refusal and later-row refusal with cumulative
  counts and a row-only certificate.
- Three physical weighted-yield comparisons use an independent 90-digit sinc
  integral for complex two-polarization rows, unequal weights, 17 incident
  electrons and two disjoint bands. The matched yield passes; yields reduced
  or increased by 20 percent fail against the same valid enclosure.
- A separate four-weight collector check preserves identical captured complex
  fields, phases and electron identifiers, including zero intensity weight,
  and confirms unchanged hook arity.

Maintained same-trajectory production anchors additionally exercise mosaic
capture reconstruction and a production-band weighted-yield comparison.
They are distinguished from the fresh-context Decimal, exact-rational and
analytic-integral evidence above.

### Scoped verdict and exclusions

The independent result is **rederived for these extensions**, conditional on
valid underlying row integral enclosures, complete capture, matching bands and
material law, and the exact stored form-factor inputs. No divergent factor,
exponent, normalization or budget-coverage convention was found in this scope.

The full `coherent-line-grid-windowed-resolution` claim remains **discrepancy**.
Excluded here: validity of the production window envelope and underlying row
primitive, omitted rows or bands, upstream capture/material construction
uncertainty, production floating-point rounding, empirical infinite-slab
form factors, automatic full-axis policy integration, centroid/FWHM,
GPU execution and corrected remote refinement ladders. The owner may append
this scoped `rederived` result and evidence to ledger Notes; no full-claim
status promotion or human `signed-off` is implied.

## Owner continuation: full finite-axis production yield integration

`audit_full_axis_spectrum_yield` now compares the complete resolved finite
axis with the weighted captured-field integral. The energy coordinates must
be strictly increasing, finite, and have endpoints exactly equal to those of
the supplied `CoherentDispersionLaw`. Source samples must be finite,
nonnegative and have the same shape. The audit uses every adjacent interval,
including the coarse backbone between line windows; it cannot select only
the narrow bands that previously passed the audit. The material-law axis
stays fixed throughout refinement.

For stored binary64 coordinates $E_i$ and source samples $s_i$, the numerical
quantity being checked is the exact piecewise-linear integral

$$
Q=\sum_{i=0}^{n-1}
\frac{(E_{i+1}-E_i)(s_i+s_{i+1})}{2}.
$$

Each elementary operation is enclosed by its adjacent binary64 values,
retaining positive underflow. Summation of the nonnegative lower and upper
terms is enclosed using the generic sequential-addition bound, which also
covers a pairwise tree with no more than $n$ additions along a path. With
unit roundoff $u=2^{-53}$ and the smallest positive binary64 number $\eta$,

$$
\gamma_n=\frac{nu}{1-nu},\qquad
a_n=\frac{n\eta}{1-nu},\qquad
|\operatorname{fl}(S)-S|\leq\gamma_n S+a_n.
$$

The absolute term conservatively covers gradual underflow propagated through
the additions. Provided $nu<1/2$, the resulting sum enclosures are

$$
S\geq\max\left(0,\frac{\operatorname{fl}(S)-a_n}{1+\gamma_n}\right),
\qquad
S\leq\frac{\operatorname{fl}(S)+a_n}{1-\gamma_n}.
$$

Directed interval arithmetic and outward conversion produce
$[Q_-,Q_+]$. An exactly zero sample array has exactly zero quadrature.
Nonfinite intermediate sums refuse rather than claiming an enclosure.

The existing weighted spectrum helper bounds the same complete finite-axis
stored field by $[L,U]$, with half the requested relative share reserved for
enclosure width. Even constant form-factor bounds are passed as callbacks,
so refinement can subdivide locally instead of repeatedly sampling the whole
axis. Material knots remain mandatory boundaries; all contributing rows must
finish within the cumulative evaluation budget. The final comparison is

$$
\epsilon_Q=
\max_{q\in\{Q_-,Q_+\}}
\sup_{Y\in[L,U]}\frac{|q-Y|}{Y}.
$$

Convexity in $q$ makes the endpoint maximum sufficient for every quadrature
value in the enclosure. Existing zero-power and no-positive-floor conventions
apply. The audit passes only when $\epsilon_Q$ meets the requested share;
enclosure convergence alone does not accept the production grid.

The runner enables this developer check through the private case mapping
`_coherent_yield_audit`, with optional `relative_tolerance`, `initial_samples`
and `max_evaluations`. Its hook captures rows from the actual coherent
production call, after the automatic grid has resolved, rather than from a
second transport or a separate sub-range evaluation. It reads the production
setup's active coherence sector: inactive decoherence uses $F=1$, an active
finite footprint uses the analytic Gaussian enclosure with the same stored
RMS-duration conversion, and an infinite slab retains the conservative
$0\leq F\leq1$. A loose empirical enclosure may exhaust its budget; it is
never replaced with $F=0$. Multiple radiators and `sinc_cutoff` are refused.
Capture weights and the incident population retain their existing meanings.

Before characteristic and bremsstrahlung evaluation, a failed yield check
raises `LineGridToleranceError` with its relative error, requested share,
axis endpoints and coordinate count. An exhausted integral budget raises
`SpectrumPowerBudgetError`; no partial acceptance is returned. A passing
result carries scalar `line_grid_coherent_yield_audit` provenance: integral
and quadrature bounds, tolerance, evaluations, row and coordinate counts,
axis endpoints and audit wall time. Default production runs do not enable
this instrumentation.

Owner regression checks enclose independent exact-rational nonuniform
quadratures, including subnormal powers; reject truncated axes, invalid
samples, exhausted budgets and a deliberately 10% excessive spectrum; and
cover exact zero power. A one-electron, 1 Ang HOPG track at 30 keV and 5 deg
tilt exercises the full automatic 10--3700 eV axis through the real spectrum
runner. The yield comparison passes the $10^{-3}$ share and capture leaves
the coherent source bit-identical on the same trajectory and coordinates.
This thin-track check is an integration anchor, not the issue's thick-target
or line-shape acceptance.

Scope remains conditional on the row-integral primitives and exact stored
capture/material/F inputs. The quadrature enclosure bounds arithmetic on the
supplied samples, not how production computed those samples, upstream capture
or material uncertainty, power outside the chosen finite axis, centroid or
FWHM. This owner extension requires fresh-context verification before a
scoped `rederived` verdict. The full ledger claim stays `discrepancy` pending
the remaining envelope, production/shape and corrected remote validations.

## Independent full-axis verification (2026-10-07)

Verification target: checkpoint `de828003`, the full finite-axis quadrature,
weighted-integral comparison and opt-in production-runner hook. This context
did not implement the extension. The issue, owner continuation, docstrings,
ledger Notes and selected maintained tests were visible before this derivation;
implementation bodies were not. Thus this is fresh-context verification with
prior exposure to the claimed proof, rather than a blind source-only derivation.
The previously verified row-integral and weighted-composition primitives are
assumed valid only within their recorded stored-input scope.

### Derivation before implementation inspection

Integrating the affine interpolant on each adjacent coordinate interval gives

$$
Q_i=\frac{(E_{i+1}-E_i)(s_i+s_{i+1})}{2},\qquad Q=\sum_i Q_i.
$$

For increasing finite coordinates and nonnegative samples, every term is
nonnegative. Units are photons per steradian per incident electron when the
samples are spectral densities per eV. Constant density recovers density times
axis length; zero density gives zero yield. Windows confer no permission to
omit the intervals connecting them.

Let $u=2^{-53}$ and $\eta$ be the least positive binary64 subnormal. Model an
addition by relative error at most $u$ plus absolute error at most $\eta$.
Propagating through at most $n$ additions gives a conservative positive-sum
bound

$$
|\widehat S-S|\leq\gamma_n S+a_n,\qquad
\gamma_n=\frac{nu}{1-nu},\quad a_n=\frac{n\eta}{1-nu}.
$$

For $nu<1/2$, inversion yields

$$
\max\left(0,\frac{\widehat S-a_n}{1+\gamma_n}\right)
\leq S\leq\frac{\widehat S+a_n}{1-\gamma_n}.
$$

This requires outward arithmetic on the bound constants and inverted endpoints,
not merely outward rounding of the final sum. Neighbor expansion of each
finite elementary operation encloses its exact real result, including positive
underflow; overflow must refuse. Positive products preserve interval ordering.
An exact-zero shortcut is valid only when all samples vanish.

For nonnegative row weights $w_r$ and incident population $N_e>0$, complete
row certificates compose as

$$
L\leq Y=\frac{1}{N_e}\sum_r w_r Y_r\leq U.
$$

Both integral and quadrature must use the same complete finite axis, material
law, captures, weights, population and coherence sector. Exhaustion for any
positive-weight row prevents a complete certificate. For $L>0$, the exact
worst-case relative error for quadrature interval $[Q_-,Q_+]$ is

$$
\epsilon=\max\left(
\left|\frac{Q_-}{L}-1\right|,
\left|\frac{Q_-}{U}-1\right|,
\left|\frac{Q_+}{L}-1\right|,
\left|\frac{Q_+}{U}-1\right|\right).
$$

For fixed positive $Y$, absolute error is convex in $Q$; for fixed nonnegative
$Q$, $Q/Y$ is monotone in $Y$. These properties prove endpoint sufficiency.
When $L=0<U$, a positive quadrature admits no finite relative-error guarantee;
zero quadrature has relative error one for positive yield. Exact zero yield
and quadrature agree; positive quadrature against zero yield does not.
Enclosure convergence is separate from the final error test.

The runner must capture the very call whose samples it audits and reject
before subsequent components run. Inactive decoherence requires $F=1$;
finite-footprint active decoherence permits its analytic Gaussian upper bound;
an unconstrained empirical factor needs the full $[0,1]$ interval. Neither
missing capture nor exhausted work can imply acceptance. These checks concern
finite-axis yield of exact stored inputs, not source-sample construction,
outside-axis power, centroid or FWHM.

### Implementation comparison and independent checks

`_trapezoid_bounds` expands coordinate differences, adjacent sample sums,
products and halving in the correct monotone directions. Its interval context
computes the summation constants and inverted endpoints outward; lower
conversion rounds down and upper conversion rounds up. Positive subnormal
samples are retained. Overflow refuses conservatively even when a rearranged
formula could yield a finite result. This is a supported-scope limitation,
not a false enclosure.

`audit_full_axis_spectrum_yield` requires exact material-law endpoints and
increasing finite coordinates, uses every sample interval, and passes no
restricted bands to the weighted primitive. It reserves half the share for
integral convergence and tests both quadrature endpoints with the independently
derived relative-error expression. Passing the enclosure's lower endpoint as
the primitive's numerical estimate does not license acceptance: the final
full-axis result uses the larger of both endpoint errors.

The runner forwards the capture hook only into the actual coherent call on
the already resolved axis. Coherent spectra are not divided into independent
electron blocks. The same incident count reaches production and the audit;
mosaic weights retain the collector's previously verified convention.
Production and the reconstructed law both use the default dispersive material
convention. The audit's Gaussian duration conversion matches the production
conversion. Active infinite-slab decoherence retains $[0,1]$; inactive
coherence uses $F=1$. Unsupported radiators and `sinc_cutoff` refuse. The
check precedes characteristic/bremsstrahlung work and returns scalar provenance
only after acceptance; budget exceptions propagate without a success record.

Independent temporary checks, run with the project test runner, cover:

- 900 exact-rational nonuniform trapezoid integrals, including adjacent
  representable coordinates, exponents from $-1074$ through $500$, mixed zero
  samples and positive subnormal samples. Every exact integral lies within
  the returned enclosure.
- 1,000 exact-rational scalar relative-error extrema, including very small
  positive yields; every returned upper bound dominates the exact error.
- 100 full-axis composition checks against exact-rational four-endpoint
  extrema, verifying complete-band forwarding, the half-share reservation,
  refinement callbacks and final acceptance comparison.
- Direct coherence-sector probes for inactive, finite-footprint and empirical
  infinite-slab captures; direct failed-yield and exhausted-budget probes,
  both refusing rather than returning scalar provenance.

All five independent checks pass. The maintained spectrum-audit module passes
35 tests. The real automatic-axis HOPG integration anchor passes separately
(1 test, 57 deselected, 48.20 s), including its bit-identical same-trajectory
capture-neutrality comparison. This rerun establishes the thin-track test's
acceptance; it does not independently reproduce the owner's previously
reported numerical bounds or wall time.

### Scoped verdict

- **Claim**: `coherent-line-grid-windowed-resolution` —
  `coherent_spectrum_audit.py::{_trapezoid_bounds,audit_full_axis_spectrum_yield}`
  and `runner/coherent_audit.py::CoherentGridAudit` — affine interpolation,
  positive weighted composition and endpoint relative-error extrema.
- **Filters**: units pass; zero/constant-density limits pass;
  signs/conventions pass.
- **Re-derivation**: matches within the complete finite-axis stored-input
  scope, conditional on the previously verified row-integral primitives.
- **Verdict**: scoped `rederived` for this extension at `de828003`, with
  prior owner-proof exposure explicitly recorded above.
- **Suggested ledger change**: the task owner may add this scoped verdict
  and evidence to Notes. Keep the full row's Status `discrepancy`: the
  general window-envelope claim, upstream construction and production sample
  arithmetic, outside-axis power, centroid/FWHM, scalable reference/thick
  certificates and corrected remote ladders remain outside this verification.
  No `signed-off` transition is proposed.

## Owner continuation: finite-axis centroid enclosure

The full-axis audit now supports a first-moment comparison through
`audit_full_axis_spectrum_centroid`. This is an owner extension awaiting
fresh-context verification. The earlier scoped verification applies to the
frozen yield-only checkpoint `de828003`; it does not certify this new moment
composition or the integration-partition option.

For a positive stored-input spectrum $P(E)$, define

$$
Y=\int_A^B P(E)\,dE,\qquad
M=\int_A^B E P(E)\,dE,\qquad \mu=\frac{M}{Y}.
$$

The energy axis remains positive and fixed. On every latest complete
row-certificate interval $[a_i,b_i]$, positivity implies

$$
a_i L_i\leq\int_{a_i}^{b_i} E P_r(E)\,dE\leq b_i U_i.
$$

With nonnegative captured mosaic weights $w_r$ and the same incident
population $N_e$ as production, outward composition gives

$$
M_- =\frac{1}{N_e}\sum_{r,i}w_r a_i L_{r,i},\qquad
M_+ =\frac{1}{N_e}\sum_{r,i}w_r b_i U_{r,i}.
$$

Whole-axis retained yield intersections may tighten $[L,U]$, but moment
composition uses every interval in the latest complete row partition. For
$L>0$, the centroid enclosure is

$$
[\mu_-,\mu_+]
=\left[\frac{M_-}{U},\frac{M_+}{L}\right]\cap[A,B].
$$

All arithmetic and binary64 endpoint conversions are outward. Units of the
moment are eV times photons per steradian per incident electron; centroid
units are eV. Population normalization cancels in the exact ratio, but it
is retained consistently in both certificates. A zero or uncertified positive
power floor refuses: an empty source has no physical centroid.

The numerical comparison follows `spectrum_observables`, namely the ratio
of the trapezoid of the node values $E_i s_i$ to the trapezoid of $s_i$.
It is not the exact first moment of the affine density interpolant. Each
stored node product is expanded outward before applying the already used
nonuniform trapezoid enclosure. Exact zero samples keep exact zero products;
positive underflow is retained. If the quadrature denominator is enclosed
by $[Q_-,Q_+]$ with $Q_->0$ and its moment by $[R_-,R_+]$, then

$$
[q_-,q_+]
=\left[\frac{R_-}{Q_+},\frac{R_+}{Q_-}\right]\cap[A,B].
$$

Using the same endpoint-extremum argument as for yield, the error charge is

$$
\epsilon_\mu
=\max_{q\in\{q_-,q_+\},\,m\in\{\mu_-,\mu_+\}}
\left|\frac{q}{m}-1\right|.
$$

A result passes only when both yield and centroid comparisons pass their
respective shares. A converged yield cannot substitute for a centroid bound.
A constant or symmetric source recovers its expected centroid as the moment
partition contracts; an unresolved energy partition can still fail even when
the power integral has converged.

`integration_bins` creates a complete uniform integration partition before
splitting at material knots and adaptive refinement. It changes neither the
production coordinates nor the material-law endpoints. A conservative
initialization-budget check occurs before allocating the partition; every
positive-weight row and all knot-induced work remain charged by the existing
cumulative budget. Duplicate, unrepresentable boundaries refuse. This option
is explicit: moment bounds reuse the resulting row certificates without extra
field evaluations, but obtaining a finer partition still costs evaluations.
Scalability to the reference/thick transport is not established here.

The private runner audit enables centroid checking when
`centroid_relative_tolerance` is present in `_coherent_yield_audit`; its
optional `integration_bins` controls the partition. Failed yield, centroid,
or budget checks refuse before characteristic/bremsstrahlung evaluation.
Successful scalar provenance additionally records model and quadrature
centroid bounds, centroid error share and tolerance. Yield-only calls retain
their previous provenance scope and default integration partition.

Owner regressions cover a symmetric sinc source, unequal weighted rows
against an independent 70-digit first-moment integral, positive subnormal
source power, zero-power refusal, unresolved moments, invalid/oversized
partitions and runner refusal. The adversarial source with endpoint samples
$0,1,2$ over 990--1010 eV passes the yield check but has a numerical centroid
of 1005 eV, so the centroid check refuses against the nearly flat symmetric
source centered at 1000 eV. The nine initial centroid cases failed before
this API existed and pass after implementation.

The real thin-track HOPG anchor uses the same 30 keV, 5 degree, one-electron,
1 Ang trajectory and automatic production axis as the previous yield anchor.
With 2048 integration bins and a 16000-evaluation budget, both requested
$10^{-3}$ shares pass, and the unaudited coherent spectrum remains
bit-identical. The focused physical test completes in 72.38 s; this is test
wall time, not the separately recorded audit timer. It is a thin-track
integration anchor, not reference/thick or FWHM acceptance.

The full ledger Status remains `discrepancy`. This owner extension needs a
fresh-context check of moment composition, the numerical-centroid convention,
outward rounding and runner gating. General window-envelope validity,
upstream/sample arithmetic uncertainty, outside-axis power, FWHM and corrected
remote ladders remain outside its scope. A fresh remote job inventory probe
still fails to resolve the configured `qlmc` home with SSH exit 255; no heavy
local ladder or remote job was started.

## Independent centroid-enclosure verification (2026-10-07)

Verification target: commit `35fc0c3c` (parent `de828003`, already scoped
`rederived` for the full-axis yield), namely
`coherent_spectrum_audit.py::audit_full_axis_spectrum_centroid` with
`FullAxisCentroidAudit` and `_lower_float`, the `integration_bins` partition
option of `audit_full_axis_spectrum_yield`, and the centroid gating in
`runner/coherent_audit.py::CoherentGridAudit`. This context did not implement
the extension. Before deriving, it saw the methodology, the ledger row
(including Notes that name centroid/FWHM as open) and the task request. That
request already outlined the moment inequality and the four-endpoint
charge. The owner section, implementation bodies and maintained tests were
read only after the derivation below and after the independent scripts
`scratch/verify_centroid/ref.py` and `pre_checks.py` were written and run.
This is a fresh-context verification with prior exposure to the outline of
the claimed proof, not a blind source-only derivation. The row-integral
primitives, weighted yield composition and `_trapezoid_bounds` are assumed
valid only within the stored-input scope recorded by earlier verifications.

### Derivation before implementation inspection

Let $P(E)\geq0$ be the weighted stored-input spectrum on a fixed finite axis
$[A,B]$ with $0<A<B$:

$$
P(E)=\frac{1}{N_e}\sum_r w_r P_r(E),\qquad w_r\geq0,\quad N_e>0,
$$

$$
Y=\int_A^B P\,dE,\qquad M=\int_A^B E\,P\,dE,\qquad \mu=\frac{M}{Y}.
$$

Let row $r$ carry a complete partition $A=x_{r,0}<\dots<x_{r,n_r}=B$ with
certificates $L_{r,i}\leq Y_{r,i}=\int_{x_{r,i-1}}^{x_{r,i}}P_r\,dE\leq U_{r,i}$.
Positivity of $P_r$ and $0<x_{r,i-1}\leq E\leq x_{r,i}$ on each interval give

$$
x_{r,i-1}\,Y_{r,i}\leq\int_{x_{r,i-1}}^{x_{r,i}}E\,P_r\,dE\leq x_{r,i}\,Y_{r,i},
$$

hence $x_{r,i-1}L_{r,i}\leq\cdot\leq x_{r,i}U_{r,i}$. A negative $L_{r,i}$
only loosens the lower bound because $x_{r,i-1}>0$. Positive weights and
$1/N_e$ preserve order, and additivity over a complete partition gives

$$
M_-=\frac{1}{N_e}\sum_{r,i}w_r x_{r,i-1}L_{r,i}\leq M\leq
M_+=\frac{1}{N_e}\sum_{r,i}w_r x_{r,i}U_{r,i}.
$$

Completeness matters only for $M_+$: an omitted interval keeps $M_-$ valid
but invalidates $M_+$. Rows need not share a partition.

For any valid $Y\in[L,U]$ with $L>0$, positivity of $M$ and $Y$ gives
$M_-/U\leq M/Y\leq M_+/L$. This uses no correlation between the numerator
and denominator, so any valid enclosure of the same $Y$ may be used,
including a whole-axis intersection retained across refinements that is
tighter than the sum of the current interval bounds. The certificates must
describe the same rows, weights, $N_e$, law and axis. Because $\mu$ is a
$P$-weighted mean of $E\in[A,B]$, intersecting with $[A,B]$ is valid. The
enclosure cannot be narrower than about $\mu(U-L)/L$ plus a weighted mean
interval width, so a converged yield does not imply a resolved centroid.

**Numerical centroid.** On increasing nodes $E_0=A<\dots<E_m=B$ with
stored samples $s_j\geq0$, the `spectrum_observables` convention is

$$
\hat\mu=\frac{R}{Q},\qquad
R=\sum_{j=1}^{m}\frac{h_j}{2}\left(E_{j-1}s_{j-1}+E_js_j\right),\qquad
Q=\sum_{j=1}^{m}\frac{h_j}{2}\left(s_{j-1}+s_j\right),
$$

with $h_j=E_j-E_{j-1}$. In exact arithmetic,
$\hat\mu=\sum_j c_jE_js_j/\sum_jc_js_j$ with positive node weights
$c_j=(h_j+h_{j+1})/2$, so $\hat\mu\in[E_0,E_m]$. Under round-to-nearest,
the exact product $E_js_j$ lies within one binary64 neighbour of
$\mathrm{fl}(E_js_j)$, including subnormal results and products that
underflow to zero. An exact zero sample gives an exact zero product. The
nonuniform trapezoid is monotone in its samples. Directed trapezoid bounds
of the lower and upper product arrays, $R\in[R_-,R_+]$, and of $s$,
$Q\in[Q_-,Q_+]$ with $Q_->0$, therefore give
$\hat\mu\in[R_-/Q_+,R_+/Q_-]\cap[A,B]$.

**Error charge.** For $\mu\in[\mu_-,\mu_+]\subset(0,\infty)$ and
$\hat\mu\in[q_-,q_+]$, the function $f=\lvert\hat\mu/\mu-1\rvert$ is
convex in $\hat\mu$ for fixed $\mu$. For fixed $\hat\mu$ it is convex in
$t=1/\mu$, and $t$ ranges over an interval. A convex function on an interval
attains its maximum at an endpoint. Taking the maximum first over $\hat\mu$
and then over $t$ gives

$$
\epsilon_\mu=\max_{q\in\{q_-,q_+\},\,m\in\{\mu_-,\mu_+\}}
\left\lvert\frac{q}{m}-1\right\rvert,
$$

which is a rigorous upper bound on the relative error. It must be rounded
upward.

**Refusals and limits.** $L=0$ or $Q_-=0$ gives no centroid enclosure and
must refuse. If $P$ is constant and the certificates are exact,
$[\mu_-,\mu_+]=(A+B)/2\mp\sum_ih_i^2/(2(B-A))$, which contains $(A+B)/2$
and contracts to it. The trapezoid is exact for $E\,s$ affine, so
$\hat\mu=(A+B)/2$ exactly. A symmetric source has a centroid at its centre.
For a delta-like line at $E_0$ inside interval $k$, the enclosure is
$[x_{k-1},x_k]$ and contracts with the partition. On the numerical side,
the trapezoid on a single nonzero node returns that node energy exactly.

Units: $M$ is in eV times $Y$-units; $\mu$, $\hat\mu$ and the bounds are in
eV. $\epsilon_\mu$ is dimensionless.

### Implementation comparison and independent checks

| Item | Implementation | Agreement |
| --- | --- | --- |
| Moment composition | `coherent_spectrum_audit.py:297-301`: for each active row, in the yield certificates' active order, each latest interval adds `start_eV*lower` and `stop_eV*upper`, scaled by $w_r/N_e$ in the 50-digit interval context | matches $M_\pm$; the rows use the same complete partitions as the yield certificates |
| Denominator | `:302-303` divides by `audit.spectrum.upper/lower` (retained whole-axis intersection, also scaled by $w_r/N_e$) | valid, because no correlation is needed; normalization is consistent |
| Directed conversion and $[A,B]$ | `_lower_float` (round-floor, clipped at 0) and `_upper_float` (round-ceiling plus one ulp); `max/min` with `law.start/stop`, and `law.start == energy[0]` is enforced | matches; $\mu\geq A>0$ keeps the round-floor in the normal range |
| Node products | `:305-311`: `nextafter` bracket of `fl(E*s)`, lower bound clipped at 0, exact zero kept for `values == 0`, non-finite upper bound refused; negative samples are refused upstream | matches, including subnormal and underflow-to-zero cases |
| Quadrature ratio | `:312-317`: `_trapezoid_bounds` on the low/high products, divided by the yield audit's $[Q_-,Q_+]$, intersected with $[A,B]$ | matches |
| Error charge | `:320-321`: `BandPowerCertificate(clo,chi).relative_error_upper` at $q_\pm$, which takes the maximum over both $\mu$ endpoints of `abs(Q/m-1).b` and rounds up | matches the four-corner $\epsilon_\mu$ |
| Refusal | `:292` refuses `spectrum.lower <= 0` or `quadrature_lower <= 0`; an uncertified positive floor exhausts refinement and raises `SpectrumPowerBudgetError` | matches |
| Acceptance | `FullAxisCentroidAudit.within_tolerance` requires both the yield and centroid checks | matches |

`integration_bins` (`:148-169`) is checked as a positive integer. Before
`np.linspace` allocates, `bins*initial_samples*active` is charged against
`max_evaluations`. The `linspace` endpoints are exactly `law.start/stop`, so
the bands are contiguous and complete. Non-increasing edges are refused.
`_smooth_band_spans` then splits at material knots. At `:436`,
`audit_captured_spectrum_yield` charges `initial_samples*len(spans)*active`
(knots included) before any evaluation. The bands enter only the audit
call, so production coordinates are unchanged. With `integration_bins=1`
(the yield default), `bands=None` reproduces the `de828003` partition. The
new precheck is never stricter than the existing downstream check.

Runner: `coherent_audit.py:74-104` selects the centroid audit only when
`centroid_relative_tolerance` is present. It raises
`LineGridToleranceError` if the yield fails, then if the centroid fails.
Refusals propagate as `ValueError`. Provenance is returned only after both
checks pass. The call site `runner/__init__.py:1011-1013` (unchanged since
`de828003`) runs before characteristic and bremsstrahlung evaluation, on
the actual captured rows, `E_grid`, `spec_coherent` and `Ne_lines`. The
yield-only path calls the same function with the same options as before.
One difference: when the centroid check is enabled and no partition is
given, the default is `integration_bins=256`. That can refuse on budget
for many active rows; it fails safe.

Independent numerical evidence (scripts in `scratch/verify_centroid/`, with
exact `Fraction` arithmetic unless noted):

- `pre_checks.py` (written before reading the implementation). Corner
  sufficiency: no interior point of a $9\times9$ grid exceeds the corner
  maximum in 2000 random boxes. Constant density: the exact enclosures
  contain $(A+B)/2$ for 1, 3 and 17 random knots. Random positive densities:
  40 of 40 mpmath (60-digit) centroids lie inside the exact enclosures on
  random nonuniform partitions.
- `check_composition.py`: 300 randomized cases run through the
  implementation, with its yield audit replaced by synthetic certificates.
  Each case has 1–4 rows on independent nonuniform partitions, weights that
  include $0$, $10^{-300}$ and $5\times10^{-324}$, interval powers down to
  $10^{-310}$, exact-zero lower bounds and $N_e$ up to 99. Axes include
  near-degenerate widths. Sources include random values, dynamic ranges of
  $10^{\pm300}$ with positive subnormals, half-zero samples, and a single
  nonzero node. Outcomes:
  - The model enclosure contains the exact rational $[M_-/U,M_+/L]\cap[A,B]$
    in 300 of 300 cases. The maximum outward slack is $1.2\times10^{-12}A$.
  - The quadrature enclosure contains the exact trapezoid ratio $R/Q$ in
    300 of 300 cases. The maximum width is $8\times10^{-11}A$.
  - The returned $\epsilon_\mu$ is at least the exact four-corner maximum
    in 300 of 300 cases.
- `check_end_to_end.py`: 12 cases with real `CoherentRowField` sinc rows.
  Each has 1–3 rows (including zero weight), random line centres and
  durations, $N_e\leq5$, random nonuniform axes of 200–800 nodes on
  990–1010 eV, 64 bins, and 60-digit mpmath $Y$ and $\mu$. In 12 of 12
  cases the yield enclosure contains $Y$ and the centroid enclosure
  contains $\mu$; widths are 0.48–0.92 eV and $\epsilon_\mu$ is
  $2.4$–$4.6\times10^{-4}$. All 12 pass at $10^{-3}$.
- `check_delta.py` (delta-like line through the composition code, line at
  995.3, 1000 and 1009.99 eV, 4, 64 and 1024 uniform intervals): every
  enclosure contains $E_0$ and equals its containing interval. The
  quadrature ratio equals $E_0$. $\epsilon_\mu$ contracts from $5\times10^{-3}$
  to $10^{-5}$. Real-field sinc lines with $d=2\times10^3$ and
  $5\times10^3$ Å exhausted an 8000-evaluation budget in the yield
  primitive and were refused (`SpectrumPowerBudgetError`). That is fail-safe
  behaviour of the previously verified primitive, not an acceptance.
- `check_bins.py` (HOPG with the Henke law, 250–320 eV, 37 interior knots,
  three rows, one with zero weight):
  - Every final row partition is complete and contiguous and contains every
    knot.
  - Counted primitive samples equal the reported evaluations (330).
  - $10^{12}$ bins are refused in 1 ms with a 4 kB peak allocation.
  - A budget that fits `bins*init*active` but not the knot-split spans is
    refused before any primitive call.
  - On a 2-ulp axis, 3 bins are refused as non-increasing boundaries and
    2 bins are refused for having no interior sample set.

Maintained tests:
- `tests/energy-grid/test_coherent_spectrum_audit.py`: 49 passed in 9.7 s.
- `tests/energy-grid/test_coherent_windowed_line_grid.py -k "full_axis or centroid or audit"`:
  5 passed in 66.5 s. This includes the physical HOPG thin-track anchor
  `test_automatic_full_axis_yield_audit_uses_the_production_capture` with
  centroid and 2048 bins, run once.

Not reproduced: the owner's recorded anchor bounds and wall time. The
physical anchor shows only that the test passes. Limitations that remain
recorded: the centroid enclosure is conditional on valid row primitives,
complete matching capture and exact stored inputs. Outside-axis power,
sample arithmetic upstream of storage and the window envelope are outside
its scope.

### Scoped verdict

- **Claim**: `coherent-line-grid-windowed-resolution` —
  `coherent_spectrum_audit.py::{audit_full_axis_spectrum_centroid,FullAxisCentroidAudit,_lower_float}`,
  the `integration_bins` option of `::audit_full_axis_spectrum_yield`, and
  `runner/coherent_audit.py::CoherentGridAudit` centroid gating. The basis
  is the positivity moment bound $a_iY_i\leq\int E P\leq b_iY_i$, the
  outward ratio of independent enclosures, the convex-combination
  intersection, and the four-corner relative-error extremum.
- **Filters**: units pass; the constant-density, symmetric, delta-like and
  zero-power limits pass; signs and conventions pass (the
  `spectrum_observables` trapezoid ratio, relative to the model centroid).
- **Re-derivation**: `matches`. No divergent term was found. The
  whole-axis tightened denominator combined with a per-interval moment
  composition is valid. The node-product brackets, directed division,
  $[A,B]$ intersection, exact-zero and subnormal handling and refusals all
  agree.
- **Verdict**: scoped `rederived` for this extension at `35fc0c3c`, with
  the prior exposure to the proof outline recorded above.
- **Suggested ledger change**: the task owner may add this scoped verdict
  and its evidence to Notes. The full row's Status stays `discrepancy`:
  the general window envelope and row-primitive validity, upstream
  construction and sample arithmetic, outside-axis power, FWHM, scalable
  reference/thick certificates and corrected remote ladders remain open.
  No `signed-off` transition is proposed.


## Physical population continuation (2026-10-08)

Production coherent reduction now uses physical bunch charge and distinct incident
Monte Carlo pairs; see [coherent-physical-bunch-population](../radiation-physics/coherent-physical-bunch-population.md).
Window tail multipliers use the same physical pair scale and include missed
entries in the incident normalization. Infinite-slab capture includes the complete
sampled offsets. Coherent cache revision 7 includes physical charge; coherent
spectrum identities carry their own operator generation. Existing remote ladders
and smoke runs predate this operator and require fresh charge-weighted confirmation.
The full claim remains **discrepancy**. Fresh-context population verification is
**rederived**; charge-weighted remote comparisons are recorded in the linked derivation.
The original convex audit scope is extended for yield below; centroid retains
its convex-weight restriction.


## Signed physical pair-weight yield enclosure

This extension supports finite-axis **yield** with physical pair weights above
one. Fresh-context verification below returns **rederived** for its exact
stored-input scope; the full claim remains **discrepancy**. Existing independently
verified convex scopes retain their original limits. The physical population
operator itself is unchanged.

Let $G(E)\ge0$ be the grouped power and $P(E)\ge0$ the complete coherent
power of the same captured row. For $M\ge2$ incident histories and physical
population $N\ge1$, the raw production estimator is

$$
R(E)=(1-q(E))G(E)+q(E)P(E),\qquad
q(E)=\frac{N-1}{M-1}F(E).
$$

A one-electron or zero-charge bunch retains the grouped self term; its pair
coefficient is zero, including the supported single-history case. The final
source still divides by $M$ and combines rows with their nonnegative mosaic weights. All coefficients here are dimensionless; integrating a raw row
adds the energy unit. This extension assumes the same exact stored fields,
material law and whole-interval coefficient bounds as the existing directed
integration scopes; it does not certify Monte Carlo convergence.

On a band $I$, the existing directed interpolation/endpoint-strip certificates
separately enclose $G_I=\int_I G$ and $P_I=\int_I P$. Suppose
$q(E)\in[q_-,q_+]$ throughout $I$, with $0\le q_-\le q_+$. Since the pure
sector powers are nonnegative,

$$
\int_I R\in
(1-[q_-,q_+])[G_I^-,G_I^+]+[q_-,q_+][P_I^-,P_I^+].
$$

Interval products select all endpoint sign combinations and round outward.
In particular, a negative coefficient multiplies the sector's **upper** bound
for a lower result. The two appearances of $q$ are enclosed independently:
this is conservative, and remains valid when $q$ varies with energy. Treating
$q$ as one constant common to the two integrated sectors would require an
additional proof and is not used. Negative band endpoints and negative sums
are preserved throughout refinement; they are not replaced by zero.

Both pure-sector calls count against the global evaluation budget. Every
signed interval uses twice its per-sector sample count. Initialization,
resampling, both children of a bisection, and reservations for later positive
mosaic-weight rows include this factor. Intersections of successive whole-row
bounds remain valid. Relative acceptance requires a strictly positive complete
row floor, or an exactly zero enclosure; a negative or zero-crossing enclosure
exhausts its budget without returning acceptance. This does not yet support
cancellation between rows whose individual integrals lack positive floors.

Limits and regressions: zero field gives exactly zero; for two identical
aligned histories and $N=12$, $q=11$ gives raw $24$ times the one-track power
and per-incident yield $12$ times that power. Two opposed fields give negative
raw power and refuse acceptance. A varying coefficient crossing one encloses
the independent symmetric sinc integral. A complete finite-axis analytic yield
passes the $10^{-3}$ gate; insufficient initialization budgets refuse before
partial work is accepted. Production capture retains coefficients above one
without altering the emitted spectrum.

The centroid's current first-moment enclosure relies on pointwise positivity.
A signed pair estimator may violate that assumption between production nodes,
even when its integrated yield is positive. Centroid audits therefore still
require convex coefficients in $[0,1]$; a signed first-moment enclosure is a
separate remaining slice. No production envelope, outside-axis, reference/thick
shape or sampling acceptance is promoted by this yield extension.

## Independent signed-yield derivation (2026-10-08)

Fresh verifier; target checkpoint `bf39de6b`. This derivation was written before
reading implementation bodies or the owner's signed-enclosure section. The
handoff supplied the intended operator and scope. A read of the independently
validated population source inadvertently exposed its final brief summary of the
new extension; a ledger text search also exposed historical owner/verifier
Notes. Neither supplied an implementation body. This limited prior exposure is
recorded rather than treated as independent evidence.

The source is the already derived physical distinct-pair population operator in
[coherent physical bunch population](../radiation-physics/coherent-physical-bunch-population.md).
For a fixed stored field, let $G(E)\ge0$ be its grouped power and $P(E)\ge0$ its
complete coherent power. On a finite interval $I$, assume certified bounds
$0\le q_-\le q(E)\le q_+<\infty$, with
$q=(N-1)F/(M-1)$ for $M\ge2$. Zero or one physical electron has zero pair scale.
The required raw integral is

$$
J_I=\int_I\big[(1-q(E))G(E)+q(E)P(E)\big]\,dE.
$$

Let independently valid nonnegative pure-sector integral boxes be
$g_-\le\int_I G\le g_+$ and $p_-\le\int_I P\le p_+$. Since $G$ is nonnegative,
$\int_I(1-q)G$ belongs to the product of the real intervals
$[1-q_+,1-q_-]$ and $[g_-,g_+]$. This remains true when $q$ varies or crosses
one: integrate the pointwise coefficient inequalities first, then allow every
possible grouped integral. Since $q$ and $P$ are nonnegative,
$\int_I qP\in[q_-p_-,q_+p_+]$. Therefore define

$$
\begin{aligned}
C_-&=\min\{(1-q_+)g_-,(1-q_+)g_+,(1-q_-)g_-,(1-q_-)g_+\},\\
C_+&=\max\{(1-q_+)g_-,(1-q_+)g_+,(1-q_-)g_-,(1-q_-)g_+\},\\
L_I&=C_-+q_-p_-,\qquad U_I=C_++q_+p_+.
\end{aligned}
$$

Then $L_I\le J_I\le U_I$. Correlation between sectors can make this loose but
cannot invalidate it. All subtraction, multiplication, addition and interval
sums require outward rounding, including negative subnormal results. Lower
endpoints must retain their sign. Pure-sector zero must be proven, rather than
inferred from rounded products. The units are power times energy; every pair
coefficient is dimensionless. Limits $q=0$ and $q=1$ recover the grouped and
complete coherent integral respectively. Constant $q>1$ reverses the grouped
endpoint order. For $G=1$, $P=0$, $q=2$, the exact unit-interval result is $-1$;
clipping it to zero is invalid.

For disjoint intervals, add signed endpoints. Intersections of independent
whole-domain enclosures remain valid. A positive row floor $L>0$ permits the
relative uncertainty certificate $(U-L)/L$; $L=U=0$ permits the exact-zero
convention. A negative complete integral or a box containing zero must not pass
this positive-yield acceptance rule, even if its width is small. Some individual
intervals may be negative while the complete row integral is positive. Division
by the positive incident population and summation with nonnegative mosaic
weights preserve enclosure; they do not license clipping negative row endpoints.

Both pure-sector primitive evaluations must be charged: $n$ coordinates cost
$2n$ on the signed path, at initialization and every refinement. Global
reservations must cover all remaining rows before spending on a current row.
Callback coefficient bounds may contract under interval subdivision; repeated
sampling of an unchanged interval cannot contract the coefficient range itself.
A fixed bound can instead benefit from denser pure-sector sampling. For the
centroid, positivity of the complete integral alone does not imply pointwise
positivity: the inequality $aJ\le\int_I ER\le bJ$ fails for a signed integrand.
The existing moment proof must therefore retain its convex coefficient scope.

### Source-to-code comparison and adversarial checks

`_signed_row_power_bounds` matches the independent product box above. Its
interval context subtracts the coefficient bounds, multiplies both pure-sector
boxes, and adds outward; floor/ceiling conversion plus a successor retains both
negative and positive subnormal endpoints. `_sum_intervals(..., signed=True)`
retains negative totals. The signed path initializes its retained lower bound
at negative infinity. `BandPowerCertificate.relative_width_upper` accepts
exact zero or a strictly positive complete floor, and returns infinity for
negative or zero-crossing floors. No divergent factor, sign or normalization
was found.

Initialization and refinement charge both sector samples through one fixed
multiplier for the entire row certificate. `_pair_sampling_multiplier` uses
whole-interval bounds before admission. Callback bisections charge both children
before replacing their parent; fixed bounds resample the whole interval and
intersect successive boxes. The weighted audit reserves the doubled initial
cost for every remaining positive-weight row. After each row has passed its
positive-floor or exact-zero gate, the weighted sum and division by the incident
sample count remain nonnegative; the existing positive weighted-sum conversion
is valid under that gate. It is not a license to combine uncertified negative
rows.

The full-axis yield audit forwards the complete axis and retains the established
stored-sample trapezoid enclosure. Its initial allocation check is a preliminary
single-sector lower bound; the subsequent weighted audit computes the actual
doubled cost before any row primitive runs. The capture hook retains physical
pair scales above one, uses zero for zero/one-electron scale, and multiplies
Gaussian whole-band endpoint bounds outward. Centroid entry validates convex
coefficient bounds throughout the same integration path before applying its
pointwise-positive moment proof. Yield acceptance does not bypass that refusal.

Independent scratch evidence uses exact rational reference arithmetic rather
than the signed implementation's interval helpers: 1,200 endpoint product
boxes span binary exponents from $-1074$ through $400$; 100 piecewise-constant
integrals vary the coefficient across one; 300 signed sums include cancellation,
negative totals and subnormals. All are enclosed. A separate instrumented
primitive confirms remaining-row reservations: a budget of 18 permits only the
first row's six initialization evaluations and refuses before an incomplete
split; a budget of 36 charges both sectors of both complete bisections; fixed
bounds charge 32 evaluations for two rows at sample rungs three and five.
These stubbed primitive checks validate composition and accounting only.

Maintained checks: **104 passed** across
`tests/energy-grid/test_coherent_band_audit.py`,
`tests/energy-grid/test_coherent_spectrum_audit.py`,
`tests/montecarlo/test_coherent_physical_population.py` and
`tests/dev/test_docs.py`. They include aligned and opposed analytic fields,
variable coefficients crossing one, exact-zero fields, initialization refusal,
full-axis signed yield and centroid refusal, and capture of physical pair
weights above one.

Scoped verdict: **rederived** for the signed finite-band and complete finite-axis
stored-input yield extension at `bf39de6b`. This trusts the existing pure-sector
primitive only within its already reviewed scope. It does not certify upstream
capture/interpolant construction, production sample arithmetic, outside-axis
power, Monte Carlo convergence, FWHM, thick-target acceptance, GPU execution or
human sign-off. The full ledger claim remains **discrepancy**. Suggested ledger
change: append this scoped evidence to Notes and link this independent section;
retain the claim's Status.

An additional independently constructed two-history row uses phase difference
$1.6$ radians, a 100 Angstrom undamped flight centred at 50 Angstrom, a
1000 eV carrier and the asymmetric interval $[990,1015]$ eV. The coefficient
$q(E)=0.5+3(E-990)/25$ crosses one. Direct 80-digit quadrature of

$$
\left[2+2q(E)\cos(1.6)\right]
\left[100\operatorname{sinc}\left(\frac{100(E-1000)}{2\hbar c}\right)\right]^2
$$

lies inside the returned six-evaluation box; its insufficient budget refuses
acceptance. A stronger refinement attempt also safely refused after 9,990
charged evaluations, with relative width upper 0.00879927 against a requested
0.003. That conservative convergence cost is recorded as a limitation, not a
passing accuracy gate or a counterexample to enclosure.

## Independent verification of frozen-carrier scope (2026-10-08)

Fresh context, commit `f51f2445`. Its content is identical to `220d810f`, the commit originally requested, and to the earlier tip `c7bb000a` for every in-scope path: `src/pyrite/montecarlo/spectrum`, `runner/line_grid.py`, `_line_grid_policy.py`, the anchor test file, the remote harness, this write-up, the ledger part and the check records. The scope is items 1–4 of [Claim](#claim) for the frozen-carrier row field, plus the factual accuracy of item 5. Material-dispersion certification, Monte Carlo convergence, the physical-population operator, the transverse form factor (#370) and CPU/CUDA agreement (#298) are excluded.

**Prior exposure.** Before deriving anything I read the ledger Claim, Code, Source, Checks and Anchor fields and this page's [Claim](#claim) section. These state the bound's structure ($a=c/dd$, the cluster rule, the $B_J$ form and the $4(1+\ln n_C)u_{\min}/M$ allowance), so that structure was known in advance. I fixed the field from `lines/_per_hkl.py::_accumulate_reflection_coherent`, the `lines/_formation.py` docstring and `formation_coefficients`, and `coherent_population.pair_scale`/`mixed_row_power`. I wrote the derivation below before reading `coherent_windows.py`, `runner/line_grid.py` or `_line_grid_policy.py`, and before reading the dated sections of this page.

### Derivation

**Row field.** Write $x=E/\hbar c$ and use lengths in Å. A piece $j$ moves at constant velocity for $t'\in[-T_j/2,T_j/2]$, with frozen coupling $c_j=\sqrt{\alpha\omega_j/(4\pi^2\hbar c)}\,t_{L,j}(A_{\rm PXR}+A_{\rm CBS})_p$. Its reducer phase is $x\,d(t')-\mathbf g\cdot\mathbf r(t')$. Take the retarded time $u=d(t')$ as the variable. It runs over $I_j=[d_j-dd_j/2,\,d_j+dd_j/2]$, with $dd_j=(1-\boldsymbol\beta\cdot\hat{\mathbf n})T_j$. The phase is then $(x-k_j)u+\theta_j$, with $k_j=\boldsymbol\beta\cdot\mathbf g/(1-\boldsymbol\beta\cdot\hat{\mathbf n})=E_{{\rm vac},j}/\hbar c$ and $\theta_j=k_jd_j-\mathbf g\cdot\mathbf r_j$. Since $dt'=du/(1-\boldsymbol\beta\cdot\hat{\mathbf n})$ and $F_j$ is $T_j^{-1}$ times the formation integral,

$$
S_e(x)=\int e^{ixu}f_e(u)\,du,\qquad
f_e(u)=\sum_{j\in e}a_j\,\mathbf 1_{I_j}(u)\,e^{-ik_ju+i\theta_j}\,e^{-\tau_j(u)/2},\qquad
\boxed{a_j=c_j/dd_j}.
$$

On a piece, $\tau_j$ is affine in $u$. Define the amplitude slope $\lambda_j=(\tau_{\rm end}-\tau_{\rm start})/(2\,dd_j)$; it carries a sign. Parseval over the disjoint supports of one electron gives
$\int\lvert S_e\rvert^2dx=2\pi\sum_{j\in e}\lvert a_j\rvert^2dd_j\langle e^{-\tau}\rangle_j$.

**Endpoint form.** For each piece, $\int_{I_j}e^{z_ju}\,du$ with $z_j=i(x-k_j)-\lambda_j$ produces two endpoint terms. At a joint $u_J$ of one electron, $-k_ju_J+\theta_j=-\mathbf g\cdot\mathbf r(t_J)$ on both sides, and $\tau$ is continuous. The phase is therefore common, and the joint term is $e^{ixu_J}e^{i\phi_J}G_J\,[a_-/z_-(x)-a_+/z_+(x)]$. Free ends and genuine gaps keep separate single-denominator terms whose phases are not tied.

**Tail bound.** Inside a cluster, Minkowski gives $\lVert\sum_{J\in C}b_J\rVert\le\sum_J\lVert b_J\rVert$. Across clusters, integration by parts gives $\lvert\int_X^\infty e^{ix\Delta}h\,dx\rvert\le(\lvert h(X)\rvert+\int\lvert h'\rvert)/\lvert\Delta\rvert$. This reduces to $2B_JB_K/\lvert\Delta\rvert$ only if the majorants satisfy $\lvert b\rvert\le B$ and $\lvert b'\rvert\le-B'$.

The majorant $1/\lvert z\rvert$ fails the derivative condition when $\lambda\neq0$, because $\lvert d(1/z)/dx\rvert=1/\lvert z\rvert^2>(x-k)/\lvert z\rvert^3$. The real distance $y=\lvert x-k\rvert$ satisfies both conditions. Write the joint as $(a_1-a_2)/z_1+a_2(z_2-z_1)/(z_1z_2)$. Then

$$
B_J=\frac{\lvert a_1-a_2\rvert}{y_1}+\lvert a_2\rvert\frac{\lvert z_2-z_1\rvert}{y_1y_2},\qquad
\lvert z_2-z_1\rvert=\sqrt{(k_2-k_1)^2+(\lambda_2-\lambda_1)^2}.
$$

Clusters $m$ and $n$ are at least $\lvert m-n\rvert M/y_{\min}$ apart. Applying $\beta_m\beta_n\le(\beta_m^2+\beta_n^2)/2$ and a harmonic sum, the cross terms total at most $4(1+\ln n_C)(y_{\min}/M)\sum_C(\sum_{J\in C}\lVert B_J\rVert)^2$. Polarizations combine by Cauchy–Schwarz in root-sum-square.

**Population factor.** Let $G=\sum_e\lvert S_e\rvert^2$ and $P=\lvert\sum_eS_e\rvert^2$. The historical blend satisfies $(1-F)G+FP\le[1+F(N_e-1)]G$. The physical blend satisfies $G+sF(P-G)\le[1+Fs(H-1)]G$, because $\lvert P-G\rvert\le(H-1)G$. In both cases $F$ must be the supremum over the tail side. $F_z$ decreases in $E$, so that supremum is at the upper edge for the upper tail and at the axis start for the lower tail.

**Step and switch.** $\lvert S_e\rvert^2$ is the transform of an autocorrelation supported on $[-D_e,D_e]$, so the Nyquist step is $h=\pi\hbar c/D_e$. For $P$, use the all-electron span. The coherent excess satisfies $\lvert sF(P-G)\rvert\le F\,[1+s(H-1)]\,G$, so a bin with $F_{\max}N\le\eta_F$ may take $D_e$. $F_z$ varies on a scale of order $E$ wherever it is non-negligible, so multiplying by $F$ does not widen the band at the fringe scale.

**Limits.** One undamped piece whose endpoints share a cluster gives $(\sqrt{T_1}+\sqrt{T_2})^2=4\lvert a\rvert^2/Y$. Over the Parseval power this is $1/(\pi V)$ with $V=dd\,Y/2$, the incoherent $\operatorname{sinc}^2$ tail. A collinear split has equal $a$, $k$ and $\lambda$, so each joint has $b_J\equiv0$.

### Source-to-code comparison

| derived | code | result |
|---|---|---|
| $a_j=c_j/dd_j$, $dd=2\hbar c\,a_{\rm vac}$, reducer's own $c_j$ | `CoherentRowCollector.__call__`, which `_accumulate_reflection_coherent` calls with its `coefs` | matches |
| signed $\lambda=(\tau_e-\tau_s)/(2dd)$, endpoints recovered without cancellation | `attenuation_slope_ang = 2q/duration`; `bright`/`ratio` reconstruction | matches |
| Parseval $2\pi\sum\lvert a\rvert^2dd\langle e^{-\tau}\rangle$ | `CoherentRowField.power`; the $2\pi$ in `coherent_edge_leak` | matches |
| complex denominators $(E-E_i)+i\hbar c\lambda_i$, $\lvert z_2-z_1\rvert$ including $\Delta\lambda$ | `_RowJumps.terms`/`edge_moduli`: `np.hypot(dE, dslope)` | matches; the ledger text writes only $\lvert u_2-u_1\rvert$ (see below) |
| per-jump tail $\lVert b_J\rVert^2$ | `terms`: Minkowski upper bound, $\lvert a_1-a_2\rvert/\sqrt{u_1}+\lvert a_2\rvert\lvert z_2-z_1\rvert/\sqrt{3u_{\min}^3}$ | valid bound; the ledger calls it "exact" |
| cluster and cross allowance | `coherent_edge_leak`: `threshold = M hbar c/u_min`, `4(1+ln n_C) u_min/M sum(B_C)^2` | matches; clusters are counted over all electrons, which is conservative |
| exact gaps | `joint` tolerance $8\epsilon(\lvert c_1\rvert+\lvert c_2\rvert+\bar{dd})$, float64 roundoff only | matches |
| tail $F$ supremum | `factor_at(edge)` for the upper tail, `factor_at(start)` for the lower | matches |
| $N=1+s(H-1)$ | `coherent_case_seeds`: `bound_count`; $H$ = emitting electrons over the direction, at least each row's count | matches, conservative |
| step $\pi\hbar c/D$, $D_e$ or $D_{\rm all}$, divided by 8 | `_support_span`, `COHERENT_NYQUIST_OVERSAMPLING = 8` | matches |
| switch $F(\text{bin lower edge})\,N\le10^{-4}$ per 100 eV | `grouped = decoherence(bins_lo) * count <= limit` | matches |
| float32 warns, `sinc_cutoff` refused, budget refused with count | `_coherent_rows`, `_check_coherent_budget`, `_refuse_coherent_resolution` | matches |

Bisection keeps the invariant that the returned edge satisfies the bound. The bound is not monotone in the edge, because the cluster partition changes with $u_{\min}$. The edge is therefore "an edge meeting the limit", not necessarily the smallest one. That is safe.

### Independent evidence

Scripts are in the git-ignored `scratch/` directory. Each script rebuilds the field from the closed form above. Only `coherent_edge_leak`, `_RowJumps`, `CoherentRowField` and `coherent_window_seeds`, the objects under test, are imported.

- **Random adversarial rows** (`verify_envelope.py`, seed 1). Five families were tested: clustered joints; joints plus gaps plus signed slopes; equal-amplitude joints separated by 400–900 Å; pure attenuation cusps with equal carriers and amplitudes; and $10^{-3}$ Å genuine gaps translated by $10^7$ Å. Each family used 3 trials, 3 halos and 2 sides, 90 comparisons in all. The tail was integrated over 200 keV, plus a coherent remainder majorant. No violations; the worst true/bound ratio is $0.887$. The translated $10^{-3}$ Å gaps were retained (0 joints).
- **Single segment.** Inside the cluster radius the bound equals $1/(\pi V)$ to $10^{-6}$. Beyond it (halo 5 keV, $dd=50$ Å) the bound is $0.553/(\pi V)$, still above the exact Si/Ci tail ($2.78\times10^{-3}$ against $2.53\times10^{-3}$).
- **Threshold scan.** One undamped piece, $dd\in[0.3,4]\,M\hbar c/u$, three halos, exact closed form: worst exact/bound is $0.918$.
- **Phase-aligned trains** of 2–20 pieces with gaps of $1.001$–$3\times$ the threshold, exact closed form: worst ratio $0.584$.
- **Step and switch.** On a two-electron row, `span_all` (540 Å), `span_electron` (50 Å), the Nyquist step and the eighth step match by hand. `electron_step_from_eV` matches the first 100 eV bin with $F_zN\le10^{-4}$.
- **Anchor file.** `uv run pyrite-dev test tests/energy-grid/test_coherent_windowed_line_grid.py`: 62 passed (101.7 s). This includes the production-reducer tail, scope guards, budget, gaps, attenuation and the limit anchors.

### Collinear-split limit: holds only inside one cluster

The limit "a collinear split leaks what the whole flight leaks" is exact when the whole flight's two endpoints share a cluster. Split joints have $T_J=B_J=0$, but `coherent_edge_leak` still forms clusters from all jump positions (`jumps.gap`). Zero-weight joints spaced closer than $M\hbar c/u_{\min}$ can therefore chain separated endpoints into one triangle-inequality cluster, and they also raise $n_C$.

Counterexample: an undamped $dd=50$ Å flight at a 5 keV halo, or a 5000 Å flight cut into 400 equal pieces at a 50 or 500 eV halo. The whole flight gives $0.553/(\pi V)$; the split gives $1/(\pi V)$, a factor $1.81$. Both values bound the true tail, so this affects tightness (wider windows), not validity. The anchor `test_a_collinear_split_leaks_exactly_what_the_whole_flight_leaks` uses one split whose joint chains nothing.

Two fixes are possible. Either drop jumps with $\lVert a_1-a_2\rVert=0$ and $\lvert z_2-z_1\rvert=0$ before forming clusters, or restrict the limit statement to the clustered regime.

### Item 5 accuracy

- **No a-priori dispersive certificate.** Correct. The seeder uses only the frozen-carrier bound. `_dispersion_window_audit` reports a diagnostic `frozen_reference_fraction` with `relative_production_bound: False`.
- **Warning.** `_warn_coherent_dispersion` warns only when the worst row's fraction exceeds $2\eta_{\rm leak}$, on both cache-miss and cache-hit paths. In the anchor run it fired at 183.7 and $9.5\times10^{-3}$. The short-bunch record shows $7.444\times10^{4}$ for row 2, matching item 5. The [60 keV record](../check-records/coherent-physical-population/charge60.json) stores a qualifying fraction of 8.58 (row 1) but an empty `warnings` list. Replaying its `auto_record` through `_warn_coherent_dispersion` does warn. The gap is therefore in harness warning persistence across resumed slices, not in production.
- **Uniform full-axis references.** True. In `checks/coherent_physical_population_remote.py` the reference is `np.linspace` over the automatic axis endpoints at the finest row step divided by 3, on the same fingerprinted transport. On these axes, however, every 30 keV row window is the whole axis $[10,3700]$ eV. At 60 keV only row 1 stops below the axis end (4545 eV), and the union of the other rows' windows still covers $[10,6000]$ eV at the fine step. So the ladders exercise dispersive sampling. They do not exercise power excluded from windows, because none is left unsampled.
- **Tolerance quoted.** The 60 keV automatic FWHM relative error is $2.002\times10^{-4}$, slightly above the "within $2\times10^{-4}$" quoted in item 5. Yield and centroid are within it.
- **Harness step choice.** For a row that switches step partway (`electron_step_from_eV` above the window start), the harness takes `step_electron_eV` as that row's finest step and ignores `step_all_eV`. The cited 30 and 60 keV records are unaffected, because every row is per-electron from the window start.

### Earlier discrepancies at this commit

| item | status |
|---|---|
| per-piece coupling (2026-10-05) | resolved: the capture records the reducer's own `coefs`/$dd$ |
| two-carrier pair lemma, constant 2 (2026-10-06) | resolved: the decreasing real-distance majorants satisfy $\lvert b\rvert\le B$ and $\lvert b'\rvert\le-B'$, including $\Delta\lambda$; worst numeric ratio 0.918 |
| float32 | resolved as scope: `LineShapePrecisionWarning` in `_coherent_rows`; test passes |
| `sinc_cutoff` | resolved: refused; test passes |
| attenuation slopes at joints | resolved: signed slope in $z$, kept before underflow; cusp family passes |
| genuine-gap tolerance | resolved: float64-roundoff tolerance; $10^{-3}$ Å gaps at $10^7$ Å kept |
| dispersive endpoint denominator | out of this scope; moved to item 5 (no certificate claimed) |

### Scoped verdict

**`rederived`** for the frozen-carrier, float64, `sinc_cutoff = None` scope of items 1–4: envelope validity, Parseval normalization, signed attenuation, clusters and cross terms, exact gaps, population factor, step, switch, oversampling, guards and budget. Item 5 is factually accurate, subject to the qualifications above.

Wording corrections are needed. In the ledger $B_J$, use $\lvert z_2-z_1\rvert=\sqrt{\Delta E^2+(\hbar c\,\Delta\lambda)^2}$, not $\lvert u_2-u_1\rvert$. "Exact two-carrier tail" should read "Minkowski upper bound on each jump's tail". "Smallest edge" should read "a bisected edge meeting the limit". The collinear-split limit holds exactly only when the whole flight's endpoints share a cluster; elsewhere the split bound is larger (up to $1.81\times$ here) but still valid. If that limit equality is treated as load-bearing, the divergent term is the cluster partition, which counts zero-weight joints. Sign-off remains a human decision.

## Joint roundoff scale and transverse envelope (2026-10-08, #370)

Finite-footprint coherent rows are now captured with offset-free centres
(`transverse-bunch-form-factor`). The joint test previously scaled float64
roundoff by the captured centre `|d|`. Mm-scale transverse delays had kept
`|d|` near $4\times10^6$ Å, masking ~$10^{-10}$ Å residuals of the upstream
transport coordinates. With offset-free centres near $10^4$ Å, the `tiny`
fixture lost 31 of 49 joints and its windows grew from 86,118 to 164,451
points. Each piece now carries `centre_scale_ang = |d| + 2|r|` of its
complete transport phase ($|t|\le|d|+|r|$ for $d=t-\hat{\mathbf n}\cdot\mathbf r$),
and the tolerance is $8\epsilon$ times the larger of that scale and `|centre|`.
The tolerance still follows the operands' roundoff, not a physical
length: real 1 Å gaps at a $10^7$ Å origin stay gaps, and the fixture
returns to 49 joints and its previous windows. Captured rows also carry the
nonincreasing transverse envelope $F_z(E)\sup_{E'\ge E}F_\perp(E')$ used in
place of $F_z$.

## Independent verification of dispersion handling (2026-10-09)

Fresh context. Scope: the re-scoped item 5 and the dispersion helpers `_affine_dispersion_row`, `_dispersion_residual_power_bound`, `CoherentDispersionLaw.certificate` and `_dispersion_window_audit`. Prior exposure: I read the ledger row (including the owner Notes) and the owner sections of this page before deriving, so the derivation below is not blind to the stated results. I re-derived each result and then compared the code.

### Derivation

With $H=\hbar c$ and a piece's escape distance $L(d)=L_{\rm mid}+(\Delta L/\Delta d)(d-d_j)$, the dispersive phase factor is $e^{-i\,\delta\omega(E)L(d)}$. For an affine law $\delta\omega=sE+b$ the coefficient of $(d-d_j)$ in the exponent is $[kE-E_j-Hb\Delta L/\Delta d]/H$ with $k=1-Hs\Delta L/\Delta d$, which equals $k(E-E'_j)/H$. Substituting $y-y_{\rm mid}=k(d-d_j)$ gives

$$
E'_j=\frac{E_j+Hb\Delta L/\Delta d}{k},\quad
\Delta y=k\Delta d,\quad
y_{\rm mid}=d_j-HsL_{\rm mid},\quad
a'=\frac{a}{k},\quad
\lambda'=\frac{\lambda}{k},
$$

and a constant phase $-bL_{\rm mid}$. Parseval gives $P'=\sum|a|^2\Delta d\,\langle e^{-\tau}\rangle/k$. The constant-$\delta$ example ($k=0.9$) gives 90 Å, carrier $1000/0.9$ eV and step $\pi H/90$.

Residual: at fixed $E$, with $r=\delta\omega-(sE+b)$, $|\Delta S_{e,p}|\le\sum_j|a|\,\Delta d\,m_j\min(2,R\,L_{\max,j})=:G_{e,p}$, where $m_j$ is the mean amplitude transmission. Then $\int_W\sum|\Delta S|^2\le W\sum_{e,p}G_{e,p}^2$. Replacing the sum of per-electron minima by the minimum of sums, $\min(4\sum(\sum w)^2,\ R^2\sum(\sum wL)^2)$, is still an upper bound. Minkowski in $L^2(E,e,p)$ gives $Y\le(\sqrt{Y'}+\sqrt{\mathcal E})^2$. The population factor $1+F(N_e-1)$ multiplies it, with $F$ taken at the lower edge because it is nonincreasing.

Material law: $w=E(1-\mathrm{Re}\,n)/H$, $|1-n|\le|\chi|$ for $\mathrm{Re}\,n>0$. $w'$ and $w''$ follow from $n'=\chi'/2n$ and $n''=\chi''/2n-\chi'^2/4n^3$. The secant residual is at most $M_2(\Delta E)^2/8$. The terms $\chi',\chi''$ come from $F/E^2$ with $F$ a cubic plus a power law, and $f_2/E^2\propto E^{p-2}$.

### Comparison with code

- The code in `_affine_dispersion_row` matches term by term: carrier, amplitude, duration, centre, slope, phase sign (`phase_rad` enters as $-g\cdot r$, so $+bL_{\rm mid}$ is correct), `centre_scale` and the guards.
- `_piece_l1_amplitudes` and `_electron_l1_power` match $G$.
- `_dispersion_residual_power_bound` matches $W\sum G^2$.
- In the audit, the `min(4 l1, R^2 coef)` form matches the minimum-of-sums bound. Per-interval factors, Minkowski, the L1 fallback `proxy <= W*l1`, the edge-leak branches (which lower-tail or upper-tail leak applies only when the interval lies wholly beyond the affine carriers) and the full-power default otherwise are all valid.
- `certificate()` matches $M_1$, $M_2$ and the Bernstein, power-law and $\chi$ derivative expressions.
- Numeric: on HOPG 10--6000 eV, over 658 smooth intervals, sampled secant residual and sampled first derivative are at most 0.973 of the certificate. Dispersion-related focused tests pass.

### Item 5 honesty

The ledger text states that there is no a-priori dispersive certificate and that the audit is a diagnostic: `relative_production_bound=False`, with the warning text "not a measured grid error". The code agrees. The warning fires when the fraction exceeds $2\eta_{\rm leak}$, on cold and warm caches. The record `short-bunch-370.json` shows `excluded_power_bound_eV` 4.667e-3 for row 2 and `frozen_reference_fraction` 64.888 (the ledger's 64.89x), with 2 L1-fallback intervals and an identical warning string. A cheap consistency check gives reference = 7.19e-5 eV. The 14367 s spectrum run was not reproduced, because it is remote-only.

### Qualifications

1. The bound is relative to the reducer's captured field, which freezes coupling and attenuation at the carrier. It is not a bound on the true energy-dependent physical field.
2. The affine-tail step uses the frozen-scope edge-leak proof, which the previous verification covered.
3. The audit gives no sampling-error control. The $\pi H/\mathrm{span}$ step is a diagnostic.
4. The ratio 64.9 comes mostly from the L1 fallback, with $W\cdot L_1^2$ scaling. It is a loose upper bound and is not evidence of error.
5. Window-excluded power is untested empirically: in the ladders, the windows cover the axis.

Verdict: dispersion building blocks `rederived`. The re-scoped claim is honest and matches the code.

## Independent zero-weight-joint derivation (2026-10-09, issue #374)

This scoped derivation was recorded before reading the implementation bodies or
issue patch. Inputs were the claim, source identities, checks, and
`coherent_edge_leak` signature/docstring. A targeted ledger excerpt inadvertently
also exposed historical Notes; this independence-boundary limitation is disclosed
here. No new dispersion, sampling-convergence, or human sign-off is claimed.

Let $k=\hbar c$, with units eV Angstrom. At a joint of continuous endpoint phase,
write the frozen-carrier endpoint contribution for each polarization as

$$
b_J(E)=\frac{a_L}{z_L(E)}-\frac{a_R}{z_R(E)},\qquad
z_i(E)=E-E_i+i k\lambda_i.
$$

The endpoint amplitudes include attenuation at the endpoint; signed attenuation
slopes $\lambda_i$ have inverse-Angstrom units. Both real and imaginary parts of
$z_i$ have eV units. Thus $b_J$ has amplitude/eV units. With retarded positions
$\tau_J$ in Angstrom, the endpoint transform is proportional to
$k\sum_J e^{i E\tau_J/k}b_J(E)$; Parseval normalization converts the
energy-tail integral to the dimensionless fraction $k\int\|\sum_J
 e^{i E\tau_J/k}b_J(E)\|^2\,dE/P$.

Combining the two rational terms gives

$$
b_J(E)=\frac{(a_L-a_R)z_L(E)+a_L[z_R(E)-z_L(E)]}
 {z_L(E)z_R(E)}.
$$

For a collinear subdivision of one unchanged frozen-carrier piece,
$a_L=a_R$, $E_L=E_R$, and $\lambda_L=\lambda_R$. Therefore
$b_J(E)=0$ identically at every interior joint. If both endpoint amplitudes
vanish, the same conclusion holds regardless of carriers or slopes. Equal
amplitudes alone do not imply cancellation when either the carrier or signed
attenuation slope differs. A zero at one energy likewise does not imply an
identically zero endpoint function. No amplitude tolerance is licensed by this
identity: a nonzero joint, however small, must retain its majorant.

For a cluster $C$, Minkowski gives its tail bound

$$
T_C\leq\left(\sum_{J\in C}\sqrt{T_J}\right)^2.
$$

An identically zero endpoint term has $T_J=B_J=0$, so deleting it leaves both
the Fourier field and each fixed cluster's majorant unchanged. Its position has
no physical support in the endpoint sum. Clustering only the remaining nonzero
majorants can split a former chain of nearby positions into separated clusters.
For a retained separation threshold $\delta=Mk/u_{\min}$, rank-separated
clusters remain at least their rank difference times $\delta$ apart. The stated
integration-by-parts pair bound and harmonic rank sum therefore apply to the
new partition with the same coefficients. This operation changes tightness,
not the normalized physical tail.

In particular, an unsplit piece and its collinear subdivision have the same
nonzero endpoint terms, endpoint positions, carrier set, and Parseval power.
After zero-weight-joint removal they must return the same cluster-bound value
on either side of any edge outside the common resonance band. Splitting does
not alter the remaining carrier distance $u_{\min}$ in this limiting case.
If other zero-amplitude pieces introduce unrelated carriers, retaining those
carriers can still make the bound looser; this scoped identity does not require
removing them or changing carrier-distance conventions.

Cheap filters: units pass; whole/split and both-zero-amplitude limits pass;
signed-slope convention passes because the full complex denominators must
coincide. The limits with unequal carrier, unequal slope, or arbitrarily small
nonzero amplitude difference forbid deleting those joints.

### Scoped implementation comparison and independent anchors

The patch removes a captured term exactly when every polarization has equal
left/right amplitude and either the carrier and signed slope match or both
amplitudes vanish. This matches the independently derived sufficient and
necessary rational-function cancellation conditions for finite inputs. It
filters the sorted endpoint positions before recomputing gaps, preserves
per-electron cluster boundaries, and returns zero for an empty endpoint sum.
It does not discard near cancellation or a norm that underflows on squaring.
No coefficient, attenuation law, Parseval power, or separation constant changes.

For an undamped constant-amplitude piece of length $L$, with edge distance $u$
and $x=Lu/k$, the independently integrated exact one-sided tail fraction is

$$
Q_{\rm exact}(L,u)=\frac{k}{\pi L}\left[
 \frac{1-\cos x}{u}+\frac{L}{k}
 \left(\frac{\pi}{2}-\operatorname{Si}(x)\right)\right].
$$

When its two endpoints form separate clusters, their independent tail
majorants and the stated harmonic cross-term allowance give

$$
Q_{\rm bound}(L,u)=\frac{k}{\pi Lu}
 \left[1+\frac{4(1+\ln 2)}{M}\right],\qquad M=64.
$$

These expressions do not call implementation tail helpers. They apply equally
to the upper and lower side of the carrier. Independent scratch checks use the
project test runner to compare the implementation directly against this bound,
its exact tail, and whole/split equality for a 50 Angstrom piece split into four
at a 5000 eV halo and a 5000 Angstrom piece split into 400 at 50 and 500 eV
halos. Additional checks retain unequal carriers, unequal signed slopes, and
nonzero amplitudes of order $10^{-200}$, while a field with both amplitude
sides zero leaves no endpoint terms.

All seven independent scratch checks passed. For the first two length/halo
combinations, the exact fraction is $0.00252927419760506$ and the bound is
$0.00277831636043205$; for the 5000 Angstrom, 500 eV combination they are
$0.000251093036028446$ and $0.000277831636043205$. Both sides and whole/split
representations reproduce the independent bound within $2\times10^{-15}$
relative tolerance. The checks retain all three endpoint terms for unequal
carrier, unequal slope, and $10^{-200}$ nonzero amplitude cases; the all-zero
field returns zero without a division by its zero power.

The task owner's paired CPU lockstep, seed-zero, cache-isolated HOPG anchor
measurement reports unchanged grid size 126021, row point counts 86118 and
105601, and identical windows. This is attributed owner evidence, not a new
independent transport run. That fixture has no removable captured jumps
(57 and 60 before and after), so no speedup is expected there. The independent
straight-piece checks establish the targeted tightness improvement: the prior
zero-weight bridge can no longer change the whole/split bound.

Scoped adjudication: `rederived` for exact zero-endpoint removal and the existing
frozen-carrier clustering proof. Continued `anchored` status is appropriate when
the maintained regression anchors pass. Material dispersion remains uncertified;
this review adds no dispersion certificate, sampling certificate, or human
sign-off. Suggested ledger change: replace the zero-weight-chain qualification
of the collinear limit by equality after exact cancellation, and record this
scoped verifier verdict and retained nonzero-joint anchors.
