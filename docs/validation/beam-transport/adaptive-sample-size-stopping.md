# `adaptive-sample-size-stopping`

Ledger row: [`adaptive-sample-size-stopping`](../ledger-transport-background.md#adaptive-sample-size-stopping). Validation: `adaptive-sample-size-stopping`. Code: `montecarlo/runner/adaptive.py` (`RunningMoments`, `StoppingMonitor`, `run_case_adaptive`, `batch_means`). The user-facing description is in [statistical methods](../../computation/statistical-methods.md#adaptive-electron-counts).

## Claim

For a case transported on the counter-addressed per-electron or CUDA cores, the run stops at the first decision point $n = d_J$ ({eq}`eq-adaptive-decision-points`, v4, #1253) that meets the following conditions for every watched observable $o$, plus the CXR variance-sampling gate below. Here $d_{J-j}$, $j = 1..k$, are the $k$ preceding points; the decisions and the realized count do not depend on how electrons are grouped for transport:

```{math}
:label: eq-adaptive-stop-rule

r_{o,n} = \frac{s_{o,n}}{\sqrt{n}\,\lvert\bar m_{o,n}\rvert} \le \varepsilon,
\quad
\frac{\max_{i\le n} m_{o,i}}{\sum_{i\le n} m_{o,i}} \le c,
\quad
\frac{(\sum_{i\le n} m_{o,i})^2}{\sum_{i\le n} m_{o,i}^2} \ge E_{\min},
\quad
\frac{\lvert\bar m_{o,d_{J-j}}-\bar m_{o,n}\rvert}{\lvert\bar m_{o,n}\rvert} \le \max\!\Big(f\varepsilon,\ z\,r_{o,n}\sqrt{\tfrac{n/d_{J-j}-1}{\delta}}\Big)\ (1\le j\le k),\quad z=0.9 .
```

Otherwise it stops at $n_{\max}$, flagged `statistics_limited`. The terms are:

- $m_{o,i}$ is electron $i$'s grid-independent scalar. For `"line"` it is the line mass $\sum_\ell w_\ell\pi/a_\ell$ over the lines the kernel keeps for a fixed monitor band, the #201 audit quantity. For `"brem"` it is the band integral $\sum_E q_E\, \mathrm{d}N_i/\mathrm{d}E\,\mathrm{d}\Omega$ of its bremsstrahlung density, with trapezoid weights on the fixed brem grid.
- $\bar m_{o,n}$ and $s_{o,n}$ are the sample mean and Bessel-corrected standard deviation of $m_{o,i}$.
- $r_{o,n}$ is the relative standard error of the mean.

The realized result is the fixed-$N$ result at $N=n$, bit for bit. The cited sequential-interval theory concerns the asymptotic regime as the requested width tends to zero, with its regularity and stopping assumptions; it does not certify finite-sample coverage of this guarded implementation. The empirical studies below measure coverage and stopping bias in their stated populations only.


### Decision points and transport chunks (v4, #1253)

The rule is evaluated only at

```{math}
:label: eq-adaptive-decision-points

d_0=n_{\min},\qquad d_{j+1}=\min\big(n_{\max},\ \max(d_j+1,\ \lceil d_j(1+\delta)\rceil)\big),
```

ending at $n_{\max}$, with the pilot count and its projection $N$ added as points of their own; $\delta$ is `decision_growth` (default 0.1), read as the decimal it is written as, so the ceiling is exact integer arithmetic (0.1 is $1/10$, not the binary double). The values $m_{o,i}$ for $d_{j-1}\le i<d_j$ are merged into the moments as one segment, whatever transport chunks delivered them. Transport chunks are sized for device fill and memory only. A stop (or an independent brem freeze) at a point $d$ inside a chunk $[a,b)$ discards the chunk and transports $[a,d)$ again. The realized result is the fixed-$N$ result at $N=d$, bit for bit.

**Chunk independence.** $m_{o,i}$ is a function of (seed, $i$) and the run's table range alone (prefix stability, below), and the per-electron reductions are bincounts over that electron's rows, so a chunk boundary does not change any $m_{o,i}$. On the host brem path each row's band value is a per-row reduction added in a fixed element order, its layered optical depth is summed layer by layer in a fixed order, and the electron sum is one bincount in row order, so neither the transport chunk nor the reducer's row slicing changes it (before the v4 verification fixes, BLAS `gemv`/`gemm` choice by row count and slice-dependent addition order moved brem values by up to $5\times10^{-16}$ relative, single-slab and layered). The CUDA brem band reductions (fused kernel and CuPy `bincount`) accumulate atomically and are not covered by this argument. The grid depends only on the policy and, through the pilot projection, on $m_{o,1..n_p}$. Each segment's moments are reduced from the same values in the same order and merged by the same Chan update in the same sequence, so every moment, guard value and decision is bit-identical under any chunking. Re-transporting $[a,d)$ reproduces the same electrons, by counter addressing, and the existing join contract makes the kept chunks the fixed-$d$ transport.

**Overshoot and cost.** Let $d_{J-1}$ be the last refused point. Then $d_J\le\lceil d_{J-1}(1+\delta)\rceil$, so the stop exceeds the last refused count by at most $\delta$ relative (plus one electron). The block rule's absolute overshoot was up to $B-1$, and its pilot projection rounded up to a block. The run makes at most $\log(n_{\max}/n_{\min})/\log(1+\delta)+1/\delta+3$ checks, since steps of $+1$ occur only while $d_j<1/\delta$. As $\delta\to0$ the grid tends to every electron $\ge n_{\min}$.

**Stability window.** The window $[d_{J-k}, n]$ is a fixed share $1-(1+\delta)^{-k}$ of the sample (25 % at $k=3$, $\delta=0.1$), where the block rule's $kB$ was 75 % of a 12k sample and 1 % of a 1M one. For i.i.d. values,

$$
\bar m_d-\bar m_n=\frac{n-d}{n}\big(\bar m_{[1,d]}-\bar m_{(d,n]}\big),\qquad
\operatorname{Var}\frac{\bar m_d-\bar m_n}{\mu}=\frac{\sigma^2}{\mu^2}\,\frac{n-d}{nd}=r_n^2\Big(\frac nd-1\Big),
$$

so a plain tolerance $f\varepsilon$ over a fixed share refuses at random once $r_n\approx\varepsilon$: at $k=3$, $\delta=0.1$ the drift SD at $d_{J-3}$ is $0.57\,r_n$. The #1252 replays showed this: hBN 30 keV 15° stopped at 321k against 258k for the block rule. The tolerance therefore admits $z$ drift standard deviations scaled by $1/\sqrt\delta$. At $j=1$, $n/d_{J-1}-1\approx\delta$, and the term is $\approx z\,r_n$.

**Single dominant electron under v4.** Let one electron of value $h$ arrive in $(d_{J-1},n]$ among light electrons with mean $a$, share $\sigma=h/(S_1+h)$. The mean moves by $(h-a)/(S_1+h)\approx\sigma$ relative between $d_{J-1}$ and $n$, and $r_n\approx\sigma$ when that electron dominates $S_2$ (to leading order $r^2\approx\sigma^2+(1-\sigma)^2\mathrm{CV}_{\rm light}^2/n$). With $z<1$ the allowance at $j=1$ is $\approx z\,r_n<\sigma$, so the check on which the electron arrives is refused when $\sigma>f\varepsilon$. At later checks the guard may pass. A stop there satisfies $r_n\le\varepsilon$ with $r_n\gtrsim\sigma$, so the reported error covers that electron's effect. For CXR the variance gate additionally refuses while the electron dominates $M_2$ ($M_4/M_2^2\approx1>V_{\max}$). The share cap is unchanged. With $\delta\to0$, $z/\sqrt\delta$ grows and the stability guard reduces to the RSE and share tests, which the decision cadence then supplies.

**Limits of the reparametrisation.** No guard sees an unsampled class. On the #1252 1 MeV 15° captures, $k\le2$ with $\delta=0.1$ stops hBN at 3.3k with every guard passing (VOV 0.008 on the sampled prefix); the 25 % window at $k=3$ refuses it because the first heavy electrons arrive before 4k. That refusal is evidence on these captures, not a guarantee. The earliest stop is $d_k\approx n_{\min}(1+\delta)^k$, a statistical floor set by the policy, not by a transport quantity.

**Batch means.** The realized $n$ electrons are split into $K=\min(20,n)$ contiguous batches of $n_b\in\{\lfloor n/K\rfloor,\lceil n/K\rceil\}$ electrons, independent of the chunks. With batch spectra $\bar S_b$, $S=\sum_b(n_b/n)\bar S_b$ holds exactly by linearity, and $\operatorname{Var}\bar S_b=\sigma^2/n_b$ gives the unbiased estimate $\hat\sigma^2=\sum_b n_b(\bar S_b-S)^2/(K-1)$ and $\widehat{\operatorname{Var}}S=\hat\sigma^2/n$. For equal batches this is $s(\{\bar S_b\})^2/K$. Brem batches partition $[0,N_{\rm brem})$ separately.

The stopping generation `adaptive_sampling_model="decision-points-v4"` re-keys adaptive dataset identities, case content keys and provenance. The deprecated `block_histories` (a chunk-floor hint) is excluded from both identities.

### CXR variance sampling (v2, #1254)

Define central sums $M_j=\sum_i(m_i-\bar m)^j$. For CXR the additional gate is

$$
\frac{s_n}{|\bar m_n|}\le\varepsilon
\quad\text{or}\quad
\frac{M_4}{M_2^2}=\frac1{E_{\mathrm{var}}}\le V_{\max}.
$$

The first branch is a deliberate **empirical near-constant policy exception**;
it is not a population-variance bound. The second counts effective contributors
to the observed squared deviations. A constant positive signal passes this gate (CV = 0); zero signal still
fails the ordinary RSE guard. $V_{\max}$ is
`max_variance_of_variance`; v2 tied it to $1/E_{\min}$, and v3 (#1262) makes
it a separate policy field. Cauchy–Schwarz gives $M_4/M_2^2\ge1/n$, and
non-negative cross terms give $M_4\le M_2^2$, so
$V_{\max}=1$ disables the gate, and `min_effective_histories=0` disables only
the contribution ESS floor. The brem stopping policy is unchanged. Both observables record the dimensionless diagnostic

$$
\mathrm{VOV}=\frac{M_4}{M_2^2}-\frac1n
=\frac1{E_{\mathrm{var}}}-\frac1n.
$$

This is the Geant4/MCNP convention. With independent histories and finite fourth
moment, its square root is an asymptotic plug-in estimate of relative variance
uncertainty. It is not the exact finite-$n$ variance of the unbiased sample
variance. The exact population expression is
$\operatorname{Var}(s^2)=[\mu_4-\frac{n-3}{n-1}\sigma^4]/n$.
The default $V_{\max}=0.01$ (the v2 value at $E_{\min}=100$) limits observed
variance concentration to about 10 percent relative uncertainty, and so the
RSE to about 5 percent. The Geant4/MCNP pass value 0.1 corresponds to about
30 percent. Both thresholds and the CV exception are PyRITE policy, not
prescriptions from either reference; the evidence for the default is under
[threshold selection](#vov-threshold-selection-v3-1262).

Neither branch sees an entirely unsampled class, certifies population moments,
or provides finite-sample or simultaneous sequential coverage. With a true
constant positive signal, CV and VOV vanish, so the original minimum/stability
stop is retained. A constant sampled prefix can also conceal an unseen class.
For a regular variable population with finite fourth moment,
$E_{\mathrm{var}}\simeq n/\kappa$ with kurtosis $\kappa$, so the gate passes
near $n\simeq\kappa/V_{\max}$ without a floor in transport units.

The v3 stopping generation was `adaptive_sampling_model="variance-threshold-v3"`
(#1262) and v2 was `"variance-contributors-v2"` (#1254); v4 (above)
supersedes both. Fixed-count identities retain their generation.

## Sources

- Sequential fixed-width intervals: Y. S. Chow and H. Robbins, "On the asymptotic theory of fixed-width sequential confidence intervals for the mean," *Ann. Math. Statist.* **36**, 457–462 (1965). The rule $N=\inf\{n\ge n_0: s_n^2 + n^{-1}\le n d^2/z^2\}$ is asymptotically consistent ($P(\lvert\bar X_N-\mu\rvert\le d)\to 2\Phi(z)-1$) and efficient as $d\to0$ for any finite-variance i.i.d. population. Here $d=z\varepsilon\lvert\bar m_n\rvert$ is a relative half-width, which is the same theorem applied with a random scale that converges to $\lvert\mu\rvert$.
- Sequential stopping in simulation output analysis: P. W. Glynn and W. Whitt, "The asymptotic validity of sequential stopping rules for stochastic simulations," *Ann. Appl. Probab.* **2**, 180–198 (1992). This covers relative-precision rules and batch means.
- Running moments: B. P. Welford, *Technometrics* **4**, 419–420 (1962); T. F. Chan, G. H. Golub and R. J. LeVeque, "Algorithms for computing the sample variance: analysis and recommendations," *Amer. Statist.* **37**, 242–247 (1983), pairwise merge.
- Effective sample size: L. Kish, *Survey Sampling* (Wiley, 1965), $(\sum w)^2/\sum w^2$, applied here to the contributions $m_i$.

- Geant4 [B1Con documentation](https://geant4-userdoc.web.cern.ch/Doxygen/examples_doc/html/ExampleB1Con.html) and [`G4ConvergenceTester`](https://github.com/Geant4/geant4/blob/master/source/global/HEPNumerics/src/G4ConvergenceTester.cc), inspected 2026-11-17: small relative error and many histories do not establish convergence. Diagnostics include VOV, RSE/VOV trends, largest-score repetition, FOM stability, and a fitted upper-tail density slope (passing threshold 3). The implementation tests VOV against $1/N$ despite a printed label saying $1/\sqrt N$. These are diagnostic checks, not a proof of coverage or the PyRITE stopping policy.
- F. Salvat, *PENELOPE-2024*, NEA/MBDAV/R(2024)1, §§1.3 and 1.5: pp. 24–25, Eqs. (1.89)–(1.90), gives the heuristic skewness condition $|\mu_3|/(\sigma^3\sqrt N)\ll1$ and recommends examining the estimator distribution; the text then explicitly adopts asymptotic error reporting in routine use. Pages 37–39, Eqs. (1.136)–(1.143), accumulate whole-shower scores and their squares, including all particles of each shower before squaring. PENELOPE uses the plug-in $N$ denominator; PyRITE uses Bessel correction. Chapter 7, p. 291 and pp. 317–318, describes fixed shower/time budgets, periodic output, inspection and manual stopping; no analogous automatic tail test is specified there. [NEA report](https://www.oecd-nea.org/jcms/pl_108003/penlope-2024-a-code-system-for-monte-carlo-simulation-of-electron-and-photon-transport); examined the user's Zotero PDF, item `2UVQNJ2B`, attachment `CQ6Q2F87`.

## Derivation

**Independence and prefix stability.** Electron $i$'s draws are a pure function of $(\text{seed}, i)$ ([random number streams](../../computation/random-streams.md)). The per-electron transport cores couple no electrons. So $\{m_{o,i}\}$ are i.i.d. draws from the case's per-electron distribution, and the first $n$ of them are the same whatever $n_{\max}$ is. The rule's decision at $n$ is a function of $m_{o,1..n}$ only, so $N$ is a stopping time for the i.i.d. sequence. That is the setting of Chow–Robbins. The same seed and settings give the same $N$, and on a given backend the same output.

**Running moments.** A block of $b$ values reduces to $(b, \bar x_b, M_{2,b}, \sum x^2, \max x)$. Two sets merge as

```{math}
:label: eq-adaptive-chan-merge

n = n_a + n_b,\quad \delta = \bar x_b - \bar x_a,\quad
\bar x = \bar x_a + \delta\,\frac{n_b}{n},\quad
M_2 = M_{2,a} + M_{2,b} + \delta^2\,\frac{n_a n_b}{n} .
```

This is the exact identity for the centred second moment of the union. The variance is $s^2 = M_2/(n-1)$.

**The ESS floor and the error target.** For non-negative $m_i$ write $S_1=\sum m_i$ and $S_2=\sum m_i^2$. Then $(n-1)s^2 = S_2 - S_1^2/n$, and

```{math}
:label: eq-adaptive-ess-identity

r_n^2 = \frac{s^2}{n\bar m^2} = \frac{n}{n-1}\left(\frac{S_2}{S_1^2}-\frac1n\right) = \frac{n}{n-1}\left(\frac{1}{\mathrm{ESS}_n}-\frac1n\right).
```

The target $r_n\le\varepsilon$ therefore implies $\mathrm{ESS}_n\ge 1/(\varepsilon^2(n-1)/n+1/n)\approx\varepsilon^{-2}$. A floor $E_{\min}$ binds only above that. Its role is a minimum effective sample independent of $\varepsilon$. A rare class of rate $p$ appears among the first $n$ electrons with probability $1-e^{-pn}$, and a light population reaches the floor only at $n\approx E_{\min}(1+\mathrm{CV}^2)$.

**Single dominant electron.** Add one value $h\gg\bar m$ to $n$ light electrons. The share is $h/(S_1+h)$. The running mean jumps by $\Delta\bar m\approx h/n$, so $\Delta\bar m/\bar m\approx h/(S_1+h)$. Also $r_n\approx h/(S_1+h)$ to leading order, because $S_2\approx h^2$ dominates. All three are the same quantity, so with a cap $c<\varepsilon$ and stability tolerance $f\varepsilon$ with $f<1$, a stop on a single dominant electron, or on the block it just arrived in, is refused. The run continues until more of its class are drawn or until $n_{\max}$.

**Pilot projection.** For i.i.d. values $r_n\propto n^{-1/2}$, so the count reaching $\varepsilon$ from a pilot at $n_p$ is $N\approx n_p(r_{n_p}/\varepsilon)^2$. The projection only skips checks and adds $N$ to the decision points. The rule is re-applied at $N$ and every later point, so the projection cannot stop a run that does not meet the rule.

**Per-bin batch means.** (v4 partition and weighting: see the decision-point section above.) With the realized run split into $K$ equal batches $b$ of $B$ electrons, each batch spectrum $\bar S_b$ is an i.i.d. mean of $B$ electron spectra. So $s(\{\bar S_b\})/\sqrt K$ estimates the standard error of the run's mean spectrum per bin, which is the Glynn–Whitt batch-means estimator with independent batches. With normalization by the full count $n$, the batches satisfy $\sum_b (B/n)\bar S_b = S$, the run's spectrum, by linearity of the incoherent reduction.

## Assumptions

- The per-electron values have finite variance. A heavy tail with infinite variance (Pareto index $\le 2$) breaks the CLT and every guard. The guards only act on sampled values.
- The incoherent reduction is a sum over electrons. The coherent line sum couples electrons ([coherent versus incoherent statistics](../../computation/statistical-methods.md#coherent-versus-incoherent-statistics)), so adaptive runs refuse `coherent_emission`. They also refuse the lockstep core, whose shared generator is not prefix-stable, and GDF beams.
- A sample is one primary history including every descendant track. Particle cascades (shell secondaries, pair electrons, transported positrons) satisfy prefix stability because their RNG is lineage-addressed (`CASCADE_RNG_MODEL`, #923): a primary's whole cascade is the same in any block, and the run-wide track cap decides identically at every block size (#1240).
- Without `brem_target_rse`, all watched observables share one realized count, with `Ne == Ne_brem`. Per-electron cutoffs depend on the line/brem split, so a shared count keeps every block's transport equal to the fixed-$N$ run's.
- The monitor band is fixed before transport: the case's policy bandwidth, else its line-grid ends. Under the `resonance-population` policy the final axis may end below the bandwidth ceiling, so the monitor counts tails the final axis truncates. That share is audited separately (`line-grid-resonance-bandwidth`).

## Limiting cases

The numerical coverage/count measurements in this section and the original
anchor descriptions below predate v2. They are historical evidence, not
measurements of the new gate. Current v2 evidence follows separately.

- **Fixed-$N$ limit.** $n_{\min}=n_{\max}=N$ is the fixed-$N$ run. No check precedes $N$, nothing is replayed, and the output is identical (`test_min_equal_max_is_the_fixed_n_run`). For any stop, the output equals the fixed-$N$ run at the realized count, including energy-spread replay and the bunch centroid (`test_adaptive_run_is_the_fixed_n_run_at_its_realized_count`).
- **Gaussian population.** $\mathcal N(1,0.5^2)$, $\varepsilon=5\,\%$, $B=10$, $n_{\min}=50$, 1000 seeds. Coverage of $\pm1.96\,r_N$ is 0.934 with the guards off and 0.943 with the defaults, against 0.95 nominal. The stopping bias is $(+5.4\pm1.6)\times10^{-3}$ and $(+1.3\pm1.4)\times10^{-3}$, at most $0.11\varepsilon$ (`test_gaussian_population_has_near_nominal_coverage`). The small undercoverage and positive bias are the finite-$N\approx100$ sequential effect: $r_n$ falls when $\bar m_n$ is high.
- **Zero signal.** A band with no lines gives $\bar m=0$, $r$ undefined. The rule never converges and the run ends `statistics_limited` at $n_{\max}$.

## Anchors

v4 decision points (#1253):

- `test_records_do_not_depend_on_chunk_size` (rule on a heavy-tailed population at chunks 1, 7, 100, 1000, 6000 and one per point: identical records) and the slow `test_adaptive_output_does_not_depend_on_chunk_size` (live hopg transport with energy spread at chunk floors 1, 7 and 64, including a stop cut inside a 64-electron chunk: identical realized count, statistics and output).
- `test_decision_points_overshoot_at_most_their_growth` pins {eq}`eq-adaptive-decision-points`: end points, spacing $\le\lceil d(1+\delta)\rceil$, and the check-count bound.
- `test_stability_guard_refuses_the_check_a_heavy_electron_arrives_at`: a single electron moving the mean by about 20 % at a 20 % target is refused at its arrival check and allowed at the next; without the guard the run stops at the arrival check.
- `test_pilot_projection_skips_checks_until_the_projected_count`: the pilot count and the unrounded projection are decision points.
- `test_energy_range_replay_rechecks_statistics_before_accepting_stop`: a failed realized-range recheck grows to the next decision point.
- `tests/montecarlo/transport/test_independent_stopping.py` and `test_adaptive_cascades.py` cover the brem freeze (prefix re-transport) and cascades under chunked monitoring.
- `checks/adaptive_decision_cadence_1253.py` (`adaptive_decision_cadence_1253_results.json`) replays the 12 #1252 captures: records are identical at five chunkings. Converged stops lie within $-0.6\,\%$ to $+7.3\,\%$ of a dense $\delta=0.01$ grid with the same window share, i.e. within $\delta$. The 30 MeV 15° stops fall from 12k–192k (block rule, $B=3$k–$48$k) to 3 993 (60°: 32 541 against 31k–192k), against 4 000 at $B=1$k. The 30 keV stops are 265k/625k (hBN 15°/60°) and 321k/625k (HOPG) against 258k–336k and 594k–672k for the block rule. The 1 MeV cases still run to the end of the capture or to $n_{\max}$.

Earlier anchors (block rule, still valid where they do not name block ends):

`tests/montecarlo/runner/test_adaptive_stopping.py`:

- `test_running_moments_equal_direct_formulas` checks the moments, share and ESS against NumPy, and the identity {eq}`eq-adaptive-ess-identity`.
- `test_gaussian_population_has_near_nominal_coverage` covers the Gaussian limit above.
- `test_light_case_meets_its_target_over_20_seeds` runs hopg at 30 keV, 1 µm, watching line mass (skewness about 8, CV about 1). It uses $\varepsilon=10\,\%$, $n_{\min}=200$, $B=20$ and seeds 1–20. Stops fall at 200–440 electrons. All 20 lie within $1.96\varepsilon$ of the pooled 16 000-electron mean, against a threshold of $\ge 17$. The paired stopping bias against each seed's own 800-electron mean is $+0.06\,\%\pm1.5\,\%$, below the smallest reported error.
- `test_stop_inside_transport_equals_the_rule_on_the_population` shows that the live monitor and the rule on the precomputed prefix give the same record.
- `test_guards_turn_a_false_early_stop_into_statistics_limited` and `test_guards_suppress_false_stops_across_seeds` cover heavy tails. On the constructed population, the bare rule stops at 200 electrons 37 % low (mean 1.13 against 1.8; the expected deficit from never meeting the heavy class is 44 %) with a reported 6.5 %. With an ESS floor $E_{\min}=1000$ the run ends `statistics_limited` at $n_{\max}$; the shipped defaults also end it there. Across 40 random seeds, false stops fall from 24 to 1 with $E_{\min}=1000$, and only from 24 to 12 with the default guards ($E_{\min}=100$).
- `test_stability_guard_waits_out_a_late_heavy_electron` and `test_pilot_projection_skips_checks_until_the_projected_count` cover the stability guard and the pilot.
- `test_same_seed_and_target_replay_the_same_count_and_output`, `test_coherent_route_rejects_auto` and `test_lockstep_core_is_rejected_and_auto_avoids_it` cover replay and the refusals.
- `test_energy_range_replay_rechecks_statistics_before_accepting_stop` pins the energy-spread replay: if the realized-range tables change the contributions enough to fail a guard, the count grows and statistics are recomputed before accepting the result. `test_environment_pinned_lockstep_is_rejected` covers the process-wide core pin.
- `test_per_electron_brem_band_integrals_average_to_the_band_integral` and `test_batch_means_reconstruct_the_band_of_the_final_spectra` cover the brem band integral and batch means.
- `test_band_weights_integrate_linear_density_at_off_grid_edges` pins partial trapezoid intervals against an analytic linear-density integral, including a band without interior nodes.

## Cascade extension (#1240)

**Sample unit.** Every row of a descendant track carries its root primary's `electron_id`. The line-mass and brem-band reductions bin rows by that id with `minlength = n`, so $m_i$ is the root-summed contribution of primary $i$ and its cascade, and a primary with no contribution enters as $m_i=0$. The estimator, rule and guards above are unchanged; they see $n$ i.i.d. primary histories.

**Realized transport.** Cascade blocks are never joined (their track tables, cross-references and generation ordering would have to merge). The monitor reads streamed blocks over the population table range, each block drawing on a fresh run-wide track budget of $n_{\max}$ primaries, and keeps none. If the realized range differs (energy spread), streamed passes over $[0, n)$ re-measure on the realized range, growing by one block while the rule fails, exactly as the joined path. The realized $[0, n)$ then runs as one fixed-$n$ cascade call (`block_transport._fixed_cascade`): bunch, table range and track cap of $n$ primaries. Because streamed and joined cascades are the same lineage realization (#923), the returned output is the fixed-$N$ run at the realized count, and the statistics describe it up to summation order. The extra pass bounds the cost at twice the realized transport. Measured on qlmc (jobs 13303 ROCm, 13304 CPU; Si 300 keV shell cascade and Si 3 MeV pair + positron cascade at `DEFAULT_PRECISION`): the second pass is 13–18 % of adaptive wall time on the GPU and below 2 % on the CPU; the rest is the per-block measurement every block-wise adaptive run pays at 100-history blocks (#1252). Adaptive and fixed-at-$N$ spectra agree within 4.0e-16.

**Independent brem stop.** Primaries past $N_{\rm brem}$ run to the line cutoff, as in the fixed case at the realized pair; their descendants keep the cascade threshold. Brem scoring excludes rows with `electron_id >= Ne_brem`, so a descendant belongs to its root's population.

**Anchors.** `tests/montecarlo/transport/test_adaptive_cascades.py`: `test_adaptive_shell_cascade_is_the_fixed_run_at_its_count` (block sizes 1, 4 and 6, launched descendants, CPU fp64 `rtol = 1e-12` on `spec`, `spec_characteristic`, `brem_wide`, `brem`; `eta`, `hit_frac`, `n_segments` exact); `test_converged_shell_cascade_stop_is_the_fixed_run` (an early converged stop, with and without the energy-range replay, and same-seed replay); `test_independent_brem_stop_cascade_is_the_fixed_pair`; `test_monitor_samples_are_root_summed_primaries` (descendants carry most line mass of some primaries; the monitor share and ESS are the root-summed values and differ from a primaries-only measurement); `test_track_cap_fails_iff_the_fixed_run_does`; `test_adaptive_pair_cascade_is_the_fixed_run_at_its_count` (3 MeV Si with boosted synthetic pair and BremsLib rates, launched pair electrons and, with positron transport, launched positrons; block sizes 1, 3 and 6). `test_gpu_adaptive_cascade_is_the_fixed_run_at_its_count` (shell, pair and positron cascades, block sizes 1 and 4, CUDA-path core at `16 eps32`; ROCm qlmc-ace job 13289: 12 passed). The shell test also checks that the streamed-block monitor's mean, relative SE, largest share and ESS equal those of the fixed run's root-summed per-primary values, so the stop decision itself, not only the returned output, is the fixed run's.

**Independent verification (2026-11-17, #1240 extension).** A fresh-context verifier argued the extension from the claim before reading code: lineage-addressed RNG makes each primary's root-summed contribution a function of (seed, $i$) on a fixed table range, so the contributions are i.i.d. and the rule applies unchanged with zeros included; the track cap fails in the monitor only if $D(s) > 1000\,n_{\max}$ at some $s \le N$, which implies the fixed run fails too. Code review matched each step (root-id rewrite in `secondaries._join`, `minlength` bincounts, fresh per-pass `_TrackBudget`, cascade table range under replay, `cut[brem:]` and the `Ne_brem` brem mask, default fallback and identity). An independent NumPy probe reproduced the monitor statistics from the fixed run at blocks 1, 2, 4 and 8 over five bands (20/20 at `rtol <= 1e-10`; e.g. CXR RSE 0.2379 root-summed vs 0.2625 primaries-only) and showed a removed cascade leaves $m_i = 0$ counted. Verdict: matches; CUDA rests on job 13289, not re-run. Human sign-off pending. Profiles with neither fixed counts nor a policy run cascades under `DEFAULT_PRECISION`; a cascade profile's dataset identity changes because the policy enters it, so no marker is bumped and noncascade identities are unchanged.

## Known limits

- **Wald undercoverage on skewed populations.** The light case's interval $\pm1.96\,r_N$ covered 17 of 20 seeds. At fixed $N=400$ two 80-seed sets cover 86–88 %: a property of the sample-SD interval for a skewness-8 population at a few hundred electrons. The stop adds to it at small $n_{\min}$: pooled over 160 seeds at $\varepsilon=5\,\%$, coverage is 0.819, 0.838 and 0.875 at $n_{\min}=100$, 200 and 400, against 0.875 at fixed $N=400$. Treat $r_N$ as a scale, not a calibrated 95 % interval, for such observables.
- **Small $n_{\min}$ on skewed populations.** The same study found a negative stopping bias of $0.3$–$0.4\varepsilon$ at $n_{\min}\le200$, reduced to $0.13$–$0.15\varepsilon$ at $n_{\min}=400$. When the sample mean and SD are low together, the rule stops early. Set $n_{\min}$ near the count the target implies.
- **Unsampled tails.** No guard sees a class that has not appeared. Only $n_{\min}$ and the ESS floor raise the chance of drawing it. The 5 MeV h-BN detector-cone case (#201) is the motivating instance. In the remote measurement (2026-10-08, cluster-head) it reached $n_{\max}=16\,000$ with a line relative SE of 0.39 and ended `statistics_limited`, so the guards did not let it stop early; it needs variance reduction (#203) to converge.

## Independent verification (2026-10-08)

Fresh-context verifier; did not write the implementation. I read the ledger row and this record's claim, sources, assumptions and limiting cases first, rederived the results below, and only then read `montecarlo/runner/adaptive.py`, `montecarlo/runner/block_transport.py::transport_case_blocks`, `montecarlo/spectrum/brem.py::mc_brem_spectrum` (`electron_band_weights`), `montecarlo/spectrum/cxr/_kernels.py::_accumulate_edge_truncation` and `_precision.py::Precision`. The numeric checks use a separate NumPy implementation of {eq}`eq-adaptive-stop-rule`, written from the equations and not from `run_stopping_rule`. The only implementation code reused is `case_measure`, to obtain the hopg per-electron populations.

### Rederivation

**Chan merge.** Take sets $a$ and $b$ with $n=n_a+n_b$ and $\delta=\bar x_b-\bar x_a$. The union mean is $\bar x=(n_a\bar x_a+n_b\bar x_b)/n=\bar x_a+\delta n_b/n$. Then $\bar x_a-\bar x=-\delta n_b/n$ and $\bar x_b-\bar x=\delta n_a/n$. Each part's centred sum about $\bar x$ is its own $M_2$ plus $n_\cdot(\bar x_\cdot-\bar x)^2$, so

$$
M_2=M_{2,a}+M_{2,b}+\delta^2\frac{n_a n_b^2+n_b n_a^2}{n^2}=M_{2,a}+M_{2,b}+\delta^2\frac{n_a n_b}{n}.
$$

This matches {eq}`eq-adaptive-chan-merge`.

**ESS identity.** Use $(n-1)s^2=S_2-S_1^2/n$ and $\bar m=S_1/n$:

$$
r_n^2=\frac{s^2}{n\bar m^2}=\frac{n\,(S_2-S_1^2/n)}{(n-1)\,S_1^2}=\frac{n}{n-1}\left(\frac{S_2}{S_1^2}-\frac1n\right),
\qquad \frac{S_2}{S_1^2}=\frac{1}{\mathrm{ESS}_n}.
$$

This matches {eq}`eq-adaptive-ess-identity`. The identity holds for any $S_1\ne0$; only reading ESS as a count needs $m_i\ge0$. Solving $r_n\le\varepsilon$ gives $\mathrm{ESS}_n\ge[\varepsilon^2(n-1)/n+1/n]^{-1}$, as stated. For a light population, $S_2/n\to\mu^2(1+\mathrm{CV}^2)$, so $\mathrm{ESS}_n\approx n/(1+\mathrm{CV}^2)$, which reaches the floor at $n\approx E_{\min}(1+\mathrm{CV}^2)$. That matches.

**Single dominant electron.** Add one value $h$ to $n$ light values with mean $a$, $S_1=na$, and let the share be $\sigma=h/(S_1+h)$. The mean then moves by exactly $(h-a)/(n+1)$. Relative to the new mean, the move is $(h-a)/(S_1+h)\approx\sigma$, which is the normalization `_stability` uses. Also $1/\mathrm{ESS}\ge\sigma^2$ always, and

$$
r^2\approx\sigma^2+\frac{(1-\sigma)^2\,\mathrm{CV}_{\rm light}^2}{n}+O(\sigma/n),
$$

so $r\gtrsim\sigma$ whether or not $h^2$ dominates $S_2$. The record's conclusion holds: a share in $(c,\varepsilon]$ passes the $r$ test, but the cap refuses it when $c<\varepsilon$, and a jump above $f\varepsilon$ is refused for $k$ block ends. Two points are worth stating. The record's justification "$S_2\approx h^2$ dominates" is stronger than needed and fails at a few hundred electrons when $\sigma\sim\varepsilon$; the lower bound above is the robust form. And with the default $c=0.05$, the cap adds protection only for targets $\varepsilon>0.05$. At tighter targets the $r$ test already implies it, and only the stability window and the ESS floor act.

**Pilot projection.** With $r_n\propto n^{-1/2}$, $N=n_p(r_{n_p}/\varepsilon)^2$. Checks resume at every block end from $N$ on, so the decision is still a function of the prefix, and $N$ is still a stopping time. This matches.

**Batch means.** Let $S=n^{-1}\sum_i S_i$, where $S_i$ is electron $i$'s spectrum and the reduction is linear in electrons. With $K=n/B$ equal batches, $S=\sum_b (B/n)\bar S_b=K^{-1}\sum_b\bar S_b$, and $\operatorname{Var}S=\operatorname{Var}\bar S_b/K$, which $s(\{\bar S_b\})^2/K$ estimates. This matches. It needs equal batches. `Precision` makes $n_{\min}$, $n_{\max}$ and the pilot multiples of $B$, and every stop, including replay growth, falls on a block end, so $B\mid n$ always holds.

**Sources.** Welford (1962), *Technometrics* **4**, 419–420, and Chan, Golub and LeVeque (1983), *Amer. Statist.* **37**, 242–247, are correct for the one-pass and pairwise moments. Chow and Robbins (1965), *Ann. Math. Statist.* **36**, 457–462, prove asymptotic consistency and efficiency of the absolute fixed-width rule for any finite-variance i.i.d. population. Their rule carries an $n^{-1}$ term in the variance that the code omits. Here $n_{\min}\ge2$ plays that term's practical role: it prevents a stop on a zero sample variance. The relative-width rule is covered by Glynn and Whitt (1992), *Ann. Appl. Probab.* **2**, 180–198, which treats relative-precision stopping and batch means, as cited. Kish's $(\sum w)^2/\sum w^2$ is used here as a concentration measure on the contributions, not on sampling weights; the record says so ("applied here to the contributions").

### Code against equations

| Item | Equation | Code | Result |
| --- | --- | --- | --- |
| Chan merge | $\bar x\mathrel{+}=\delta n_b/n$; $M_2\mathrel{+}=M_{2,b}+\delta^2n_an_b/n$ | `RunningMoments.add_block`, lines 83–87 | matches |
| Bessel and RSE | $\sqrt{M_2/(n-1)/n}/\lvert\bar m\rvert$; `None` for $n<2$ or $\bar m=0$ | `relative_se`, lines 94–96 | matches |
| Share and ESS | $\max m/S_1$; $S_1^2/S_2$ | lines 98–102 | matches |
| Stability window | $\max_{1\le j\le k}$ over $\bar m_{n-jB}$, excluding $\bar m_n$ itself | `history[-k-1:-1]`, line 152; needs $k+1$ entries | matches; the stop at 320 in `test_stability_guard_waits_out_a_late_heavy_electron` pins it |
| Pilot rounding | $\lceil n_pN'/B\rceil B$, clipped to $[n_p,n_{\max}]$ | `_project`, line 187 | matches; an undefined RSE projects to $n_{\max}$ |
| $n_{\min}$, $n_{\max}$ | no check below $n_{\min}$; stop at $n_{\max}$ | `should_stop`, lines 192–214 | matches; converging exactly at $n_{\max}$ reports `converged` |
| `statistics_limited` | set exactly when not converged | line 237 | matches |
| Shared count | $N_e=N_{e,\rm brem}$ | lines 439, 472, 235; `realized_case` | matches |
| Band weights | exact integral of a linear density on $[\max(l,g_0),\min(r,g_1)]$: $w_{\rm up}=w\,[(l-g_0)+(r-g_0)]/(2h)$ | `band_weights`, lines 283–289 | matches, including bands with no interior node and edges off the grid |
| Brem scalar | $m_i=\sum_E q_E\,\mathrm{d}N_i/\mathrm{d}E\,\mathrm{d}\Omega$, mean equal to $q\cdot S$ | `brem.py` lines 1060–1075 and 1209–1210, `electron_sum/(4π)`, not divided by $N_e$ | matches |
| Line scalar | $\sum_\ell w_\ell\pi/a_\ell$ per electron | `_kernels.py` lines 106–123, `bincount` of `weight*pi/a_width` | matches |
| Refusals | coherent, lockstep, GDF | `Precision.validate_case`; `adaptive_transport_core` | matches; the code also refuses grooves, secondaries, pair production, positron transport, observation directions and trajectory capture |

### Numeric reproduction

The anchor file `tests/montecarlo/runner/test_adaptive_stopping.py` passed: 45 tests in 33 s. The tests assert bounds, not the quoted numbers, so I reproduced the numbers with the independent rule:

| Quoted | Reproduced |
| --- | --- |
| Gaussian coverage 0.934 rule-only / 0.943 defaults | 0.934 / 0.943 (seeds 0–999). The stopping index $N$ equals the implementation's in 1000/1000 seeds. |
| Gaussian bias $(+5.4\pm1.6)$ and $(+1.3\pm1.4)\times10^{-3}$, at most $0.11\varepsilon$ | $(5.38\pm1.62)$ and $(1.27\pm1.39)\times10^{-3}$, i.e. $0.108\varepsilon$ and $0.025\varepsilon$ |
| hopg skewness about 8, CV about 1 | 8.12 and 1.04. The test docstring at line 182 says "skewness ~ 9". |
| hopg stops at 200–440; 20/20 within $1.96\varepsilon$; 17/20 within $1.96\,r_N$ | 200–440; 20/20; 17/20. $N$ equals the implementation's in 20/20 seeds. |
| paired bias $+0.06\,\%\pm1.5\,\%$, below the smallest reported error | $+0.06\,\%\pm1.53\,\%$; smallest reported error 3.0 % |
| constructed population: bare rule stops at 200, **44 % low**, 7 % reported | stops at 200 with mean 1.129 against 1.8, i.e. **37.3 % low**, 6.5 % reported. 44 % is the heavy class's share of the true mean, which is the expected deficit, not the realized one. |
| guarded run ends `statistics_limited` at $n_{\max}$ | yes at $E_{\min}=1000$, as the test uses, and also with the default guards |
| 40 seeds: false stops $\ge20\to\le1$ | rule-only 24, then 1 **at $E_{\min}=1000$**. With the **default** guards ($E_{\min}=100$), 12 of 40, confirmed through `run_stopping_rule`. |
| 80-seed hopg study at $\varepsilon=5\,\%$: 86–89 % for every $n_{\min}$, stop not responsible | Seeds 1–80, $n_{\max}=1600$, $B=20$: 0.863, 0.863, 0.887 and 0.875 at $n_{\min}=100,200,400,800$; fixed $N$ gives 0.863–0.887. That reproduces the claim. Disjoint seeds 101–180: 0.775, 0.812, 0.863 and 0.925, against 0.863 at fixed $N=400$. Pooled over 160 seeds: 0.819, 0.838 and 0.875 at $n_{\min}=100,200,400$, against 0.875 at fixed $N=400$. |
| negative bias $0.3$–$0.4\varepsilon$ at $n_{\min}\le200$, gone by about 400 | seeds 1–80: $-0.41$, $-0.35$, $-0.13\pm0.10$ and $+0.01$ (in units of $\varepsilon$); seeds 101–180: $-0.45$, $-0.37$, $-0.15\pm0.09$ and $+0.03$. At $n_{\min}=400$ the bias is reduced, not gone. |

### Findings

1. **Misquoted constructed-case error.** The record's Anchors bullet (line 89), the ledger `Checks`, `docs/computation/statistical-methods.md` line 198, and the test docstring at line 256 say "44 % low". The realized error is $-37.3\,\%$; 44 % is the expected deficit $1-1/1.8$. The test asserts only an error $>5\,r$, so it is unaffected.
2. **Guard settings not stated.** The heavy-tail suppression results ("false stops $\ge20\to\le1$", "the guarded run") use $E_{\min}=1000$, not the default 100. The record (line 89) and the ledger `Checks` do not say so. With the shipped defaults, false stops on that population fall only from 24 to 12 of 40.
3. **Coverage attribution overstated.** The Known limits text says "86–89 % for every $n_{\min}$ ... a property of the sample-SD interval, not of the stop", and the ledger `Notes` say "independently of the stop". This holds on seeds 1–80. On a disjoint 80 seeds, and pooled, $n_{\min}\le200$ loses 4–9 points of coverage relative to fixed $N$. That is consistent with the $-0.4\varepsilon$ stopping bias the record already reports. At $n_{\min}\approx400$, coverage equals fixed $N$.
4. Minor; none changes the claim:
   - The single-dominant argument should use $r^2\gtrsim\sigma^2$, and note that the default cap acts only when $\varepsilon>c$.
   - Prefix stability in the Derivation is exact only without an energy spread. With one, the transport tables span the drawn energy range, so $m_i$ depend on $n$ at interpolation level. The replay-and-recheck path in `transport_case_blocks` handles this, and the result is still the fixed-$N$ run at the realized count, but the Derivation paragraph should say so.
   - The Assumptions list of refusals omits grooves, secondaries, pair production, positrons, observation directions and trajectory capture.
   - The test docstring at line 182 says "skewness ~ 9"; the measured value is 8.1.
   - The module docstring of `adaptive.py` points to `statistical-methods.md` for the derivations; they are in this record.

### Verdict

The estimator, the merge, the ESS identity, the guards, the pilot, batch means and band weights rederive and match the code. The anchor tests are green. The Gaussian and hopg acceptance numbers reproduce exactly, and the issue #361 acceptance holds: 20/20 light-case seeds lie within $1.96\varepsilon$, and the paired stopping bias of 0.06 % is below the 3.0 % smallest reported error. This supports `anchored`. Findings 1–3 are errors in quoted evidence and limit statements, not in the implementation. Correct the ledger `Checks` and `Notes` text before applying the status, so the row does not certify a wrong number or an overstated default guard. Human sign-off remains pending (#277).

Human sign-off pending (#277).

## Independent prefix extension verification (2026-10-18, issue #366)

Fresh-context verifier; did not implement the extension. The derivation in this
subsection was written before reading the extension implementation. The prior
verification above is preserved and concerns the shared-count policy.

### Source-first derivation and filters

The intended signature extends `Precision` by `brem_target_rse`,
`brem_min_histories`, and `brem_max_histories`. Sources remain Welford and
Chan–Golub–LeVeque for moments, Kish for concentration, and
[Glynn and Whitt (1992)](https://web.stanford.edu/~glynn/papers/1992/GW92a.html)
for sequential simulation precision. This extension changes population ownership,
not the scalar RSE formula. It supplies separate marginal precision diagnostics;
it does not establish simultaneous confidence coverage for both outputs.

Let $L_i$ and $H_i$ be nonnegative line and bremsstrahlung scalar contributions
of electron $i$ on one counter-addressed transport. Their units are their
respective band-integrated photon yield per incident electron per solid angle;
all RSE, share, ESS, and relative stability tests are dimensionless. Correlation
between $L_i$ and $H_i$ on the same electron is allowed. Independent stopping
means separate policies, not statistically independent outputs.

For component $o$ at its own count $n_o$, define

$$
\bar X_o=\frac{1}{n_o}\sum_{i=1}^{n_o}X_{o,i},\qquad
s_o^2=\frac{\sum_{i=1}^{n_o}(X_{o,i}-\bar X_o)^2}{n_o-1},\qquad
r_o=\frac{s_o}{\sqrt{n_o}\lvert\bar X_o\rvert}.
$$

Each component applies the original four guards using its own target and
bounds. Bremsstrahlung freezes at the first eligible passing block end, or its
maximum, giving $N_b$. Line moments must continue to include every transported
electron while bremsstrahlung remains active. Consequently, if $P_L(n)$ is
the current line passing predicate, the final line count is the first eligible
block end at or after $N_b$ satisfying $P_L(n)$, or the overall maximum.
A prior line pass cannot be latched: its RSE and stability can fail after a late
large contribution. Thus $N_b\le N_L$; equal targets alone do not require equal
counts. Equal fixed bounds give $N_b=N_L=N$, the ordinary fixed-count limit.
Absent `brem_target_rse`, the existing shared policy remains the governing rule.
An all-zero component has undefined relative precision and reaches its cap.

For equal batches of $B$ incident electrons, component $o$ has
$K_o=N_o/B$ batches. If $\bar S_{o,j}$ is a batch's mean spectrum, then

$$
S_o=\frac{1}{K_o}\sum_{j=1}^{K_o}\bar S_{o,j},\qquad
\widehat{\operatorname{SE}}(S_o)=
\frac{s(\bar S_{o,1},\ldots,\bar S_{o,K_o})}{\sqrt{K_o}}.
$$

A scorer normalized by its full population $N_o$ must therefore be rescaled by
$N_o/B$ when scoring one batch. Using $N_L/B$ for the brem prefix would inflate
its batch means by $N_L/N_b$. Fewer than two batches cannot estimate a batch SE.

The first $N_b$ electrons require the minimum of the line and brem transport
cutoffs; subsequent electrons require the line cutoff only. Whole-population
energy tables must match those of the realized fixed-count case. With energy
spread, changing realized counts can change the interpolation table range, hence
the contributions themselves. Replay with realized-population tables and recheck
both policies is required before accepting a tentative stop. These are numerical
identity requirements, not an additional independent-electron theorem. Finite
variance, incoherent linear reduction, and the existing unsupported-route
restrictions remain necessary. Units, zero-signal and equal-fixed-bound limits,
and positivity/concentration conventions pass these source-first filters.

### Extension implementation comparison

After writing the derivation, I inspected `IndependentStoppingMonitor`,
`StoppingMonitor`, `run_stopping_rule`, `case_stopping_monitor`, `batch_means`,
`transport_case_adaptive`, `transport_case_blocks`, `_transport_case`, and
`Precision`. The two scalar monitors reuse the established moments and guards.
Only the brem monitor stops receiving blocks after its decision; the line
monitor receives and checks every block. Brem bounds cannot exceed the overall
maximum, and both policies enforce whole blocks. The absent-policy path still
constructs `StoppingMonitor`; absent optional keys are omitted from serialization.
The transport closure uses the combined cutoff until the brem decision, then
line-only cutoffs; replay redecides the prefix between blocks. Table ranges use
the whole transported population. `batch_means` rescales by $N_L/B$ for line and
$N_b/B$ for brem, and computes each SE using its own number of batches. These
terms and conventions match the derivation; no equation discrepancy found.

### Independent numeric checks and extension verdict

A separate direct NumPy reference recomputed mean, sample SD, share, ESS and
stability from each full prefix, without using implementation moment or guard
helpers. For seeds 0–99 it used 400 lognormal line contributions (log-SD 1),
400 brem contributions (log-SD 0.5), and injected a line contribution of 500 at
index 80 for every third seed. Policies used blocks of 20, line target 0.15 and
bounds 40–400, brem target 0.2 and bounds 60–240, share cap 0.2, ESS floor 10,
and two stability blocks with fraction 0.5. All 100 realized count pairs and
both component RSEs matched; 80 pairs had unequal counts. Equal fixed bounds
80–80 returned exactly 80 line and 80 brem electrons.

The focused `tests/montecarlo/transport/test_independent_stopping.py` anchors additionally
cover late line failure after an earlier pass, a frozen prefix, retry reset,
brem statistics limitation, invalid policies, absent-policy compatibility, and
bitwise equality of final spectra with fixed realized count pairs using different
cutoffs, bunch centering and zero/nonzero energy spread. Batch means reconstruct
the component spectra on their respective populations.

**Verdict:** extension `rederived`. Retain the shared-policy verification above.
The prior equal-count assumption now applies only when `brem_target_rse` is
absent; it is superseded for the opt-in extension by $N_b\le N_L$ and the cutoff
and replay conditions derived here. Existing skew-tail, optional-stopping and
unsampled-tail limitations remain. No new calibrated coverage claim is made.
Human sign-off remains pending (#277).

Verification checks: the focused independent-stopping anchors together with
`tests/dev/test_docs.py` passed (24 tests). Rendered-page inspection found 27
math elements in this extension, all with `class="math notranslate"`, and no
literal dollar delimiters anywhere in the article body.

### Cutoff and joint-stopping assumption audit

Unequal transport cutoffs do not automatically define different line observables.
The built-in CXR adapter calls `_cxr_for_segments`; `_cxr_for_segments_once`
passes the same `E_cut_lines_keV` to every CXR evaluation, including the scalar
monitor. CXR setup calls `_clip_segments_to_cutoff`: it removes rows below the
line floor and shortens the terminal flight using the transport stopping law
and attached stopping tables. It reconstructs the midpoint/end state for
midpoint-energy rows. The intended invariant is that transporting farther for
brem cannot add a low-energy tail to the line contribution. The prefix and
suffix therefore measure the same line observable **provided this clipping
operation reproduces the line-floor transport observable** on the selected
transport/energy-loss route. Equal cutoffs satisfy this condition directly.
The extension reuses that existing physical/numerical contract; this audit did
not independently certify every transport model's cutoff reconstruction.

The fixed-realized-pair equality anchor checks the requested mixed-cutoff
numerical experiment, including its denominators; it does not prove the
clipping invariant or identical distributions across the split. If a route's
cutoff reconstruction fails that invariant, prefix and suffix line contributions
can have different means or variances. The pooled sample-SD RSE would then
remain the implemented descriptive statistic, but its interpretation as an
i.i.d. standard error would be unsupported. Even for a fixed split, differing
means add between-group variation to the pooled sample variance; a random,
correlated split additionally needs its selection effects accounted for.

There is a second, independent limitation: final CXR stopping is constrained by
the brem decision from correlated observations on the same transport. It is
therefore a joint-data stopping time, rather than the standalone scalar stop
used in the earlier Gaussian and hopg coverage studies. Those studies do not
establish interval coverage or stopping-bias bounds for this extension. Nor does
computing batch means at the selected counts remove optional-stopping effects.
Read both scalar RSEs and batch SEs as estimated precision diagnostics; neither
new marginal interval calibration nor simultaneous coverage is certified here.

The extension remains `rederived` for the stated moment formulas, guarded
prefix decision and fixed-realized-pair numerical identity. Passing reference
anchors can support `anchored` for that scoped claim. Any ledger extension must
explicitly exclude a new coverage/bias guarantee for the coupled stopping rule
and state the cutoff-observable equivalence assumption. The earlier shared-rule
coverage evidence remains scoped to its original populations and policy.

Implementation anchors after verification: the equal-minimum hopg case at line target 0.035 and brem target 0.1 stops at 60 and 40 electrons, with RSEs 0.03052 and 0.03132. CPU and CUDA realized-pair tests compare every transport array and final component spectra bit-for-bit with fixed runs, using unequal cutoffs. Seed 8 places the maximum initial energy beyond the realized population and explicitly forces table-range replay. Both CUDA cases passed on the lab RTX 5080 (SLURM job 1588, 2026-10-18). Batch-summary reconstruction uses the reduction precision: float64 on CPU and float32 on CUDA, including brem outputs widened after reduction.


## Independent v2 extension verification (2026-11-17)

A fresh read-only context derived the following identities before inspecting
implementation. For sets $A,B$ with sizes $a,b$, $n=a+b$, and
$\delta=\bar m_B-\bar m_A$, shift the centered values in $A$ by
$-b\delta/n$ and those in $B$ by $+a\delta/n$. Binomial expansion gives

$$
M_3=A_3+B_3+\delta^3\frac{ab(a-b)}{n^2}
+3\delta\frac{aB_2-bA_2}{n},
$$

$$
M_4=A_4+B_4+\delta^4\frac{ab(a^2-ab+b^2)}{n^3}
+6\delta^2\frac{a^2B_2+b^2A_2}{n^2}
+4\delta\frac{aB_3-bA_3}{n}.
$$

The implementation matches and uses the old $M_2,M_3$ before updating them.
Moment units are score to the corresponding power; CV, VOV and both ESSs are
dimensionless and invariant under uniform rescaling. For
$z_i=(m_i-\bar m)^2$, the empirical expression
$\widehat{\operatorname{Var}}_n(z)/(n\bar z^2)$ equals VOV, with denominator $n$ in the empirical variance. Cauchy–Schwarz gives
$M_4/M_2^2\ge1/n$, so VOV is non-negative apart from rounding.

**Verdict: rederived**, limited to algebra, diagnostics and implementation of
the declared empirical policy. Units, constant/zero/sparse-tail limits and
signs/normalization pass. This does not establish coverage. The verifier
independently matched centered NumPy moments for every retrieved capture and
recorded CV/VOV/gate outcomes for 134 realized replay rows. Twelve focused
new regression/diagnostic cases passed. No human sign-off is implied.

### Finite-capture evidence

`checks/adaptive_tail_1254.py` selects the longest saved population for each
material/energy/tilt, records capture SHA-256 and available producer revision,
and replays both policies at 1k, 3k, 6k, 12k, 24k and 48k. The checked-in
`checks/adaptive_tail_1254_results.json` is the numeric record. A timeout stub
for hBN 1 MeV 60° has no producer revision; its byte digest is recorded and
the missing revision is explicit. Capture exhaustion never means convergence
or completion at the requested cap. The reference
$\mathrm{CV}_{\rm full}/\sqrt n$ is a finite-capture proxy, not truth.

The 1 MeV 15° CXR distributions have a near-unit baseline (normalized by their
full mean), but maxima about 285 (hBN) and 664 (HOPG). In HOPG, the large
scores at zero-based primary IDs 40211 and 53961 arrive after every former
small-block stop; the largest score arrives at 893438. The new guard refuses
all former early stops. On 1k prefix checkpoints, the first counts after which
every later CV remains within 20 percent of the full-capture CV are 343k
(hBN) and 344k (HOPG); the ten largest squared deviations contribute 97.3
and 97.7 percent of the captured variance respectively. This is retrospective
finite-reference stability, not a stopping criterion. HOPG reaches the requested cap; hBN's capture ends
at 528k with the gate still failing, so its later outcome is unmeasured.

At 30 MeV 15°, every tested block retains its previous stopping count. At
60°, small-block stops rise from 7k–24k to 31k–48k and larger-block stops
retain their order of magnitude. At 30 keV the captured longer populations
retain their established convergence scale. Across converged replays the
reported/reference CXR RSE ratio must be at least 0.5; capture exhaustion is
reported separately.

### Exact primary attribution

The scalar captures alone contain no energy/depth/line attribution.
`checks/adaptive_tail_attribution_1254.py` closes that gap with eight individual
primary replays on the original capture checkout, revision
`00d62b0ceead5a9cb21829848719400f8fc15481`, with its original catalog overlay,
workspace and CPU environment. Simulation source is unchanged; the added
diagnostic's SHA-256 is recorded in
`checks/adaptive_tail_attribution_1254_results.json`. Every replayed line mass
matches its saved value exactly, including the baseline primary 0 in each
material. Segments and the five largest contributing lines are retained.

For HOPG IDs 40211, 53961 and 893438, the segments hosting the largest
individual lines have energies
997.84–999.21 keV, midpoint depths 3176–4527 Å, and slab-normal direction
cosines 0.061–0.155. Their largest deflections are about 70°, 91° and 71°.
For hBN IDs 150453, 261777 and 270761, the corresponding segment energies are
999.80–1000.00 keV, midpoint depths 1259–4161 Å; the first has a negative
normal direction cosine (back toward the entrance). Their largest turns are
about 97°, 76° and 36°. Each largest turn follows a row tagged `EVENT_ELASTIC` (0). These sampled
outliers arise after substantial elastic scattering at nearly the incident energy; they are not a slowed-to-low-energy
population. No channeling model is selected in these cases.

The largest individual CXR lines carry 21–41 percent of each selected outlier's
mass, with leading-line resonances from 183 eV to 28.35 keV. Thus the large primary scores
are attributable to multiple incoherent line contributions, rather than one
unidentified channel. This is measured attribution for these six outliers,
not a claim that every member of the distribution has the same mechanism.

### VOV threshold selection (v3, #1262)

v2 applied $M_4/M_2^2\le1/E_{\min}$. At the default $E_{\min}=100$ the
heavy-tail case was held to $n\gtrsim E_{\min}\kappa$ whatever its RSE
target. On the #1243 P3 MoS₂ 500 nm 15° 100 keV workload, the gate bound at
30,700 electrons (VOV 0.009916, RSE 0.0024 against target 0.01, ESS 26k). v3
keeps the ESS floor $E_{\min}$ and adds a separate CXR threshold $V_{\max}$
(`max_variance_of_variance`). The default $V_{\max}=0.01$ is the same IEEE
double as `1.0/100.0`, so default-policy stops are unchanged. The P3 workload
therefore still stops at 30,700 electrons, by construction rather than by a
new transport run.

On 2026-11-17, `checks/adaptive_tail_1254.py` was rerun under v3 on the same
captures, copied from qlmc `~/pyrite-1252-s{1,2}`. CPU directories were listed
first, matching the recorded SHA-256 values. All twelve captures carry
$E_{\min}=100$. The report equals `adaptive_tail_1254_results.json` except
for `sampling_model`.

The same replays were then repeated with only $V_{\max}$ changed.
The table lists the 1 MeV 15° rows that v1 stopped early:

| Case | Block | v1 stop | $V_{\max}=0.01$ | $0.03$ | $0.1$ |
|---|---|---|---|---|---|
| hBN | 1k | 4k | capture exhausted | **4k** | **4k** |
| hBN | 3k–48k | 15k–528k | capture exhausted | capture exhausted | capture exhausted |
| HOPG | 1k | 5k | 1.000M (max) | 1.000M (max) | **18k** |
| HOPG | 3k | 12k | 1.002M (max) | 1.002M (max) | **18k** |
| HOPG | 6k | 24k | 1.002M (max) | 1.002M (max) | **24k** |

Bold marks a v3 stop that v1 also took early. The ten largest squared
deviations hold about 97 percent of the captured variance (see finite-capture
evidence above), so those stops precede the scores that set the variance.
At 1 MeV 60° and 30 keV, every threshold gives the same stops. At 30 MeV 15°,
the CV branch decides, so every threshold gives the same stops. At 30 MeV 60°,
blocks of 1k–6k stop at 31k/33k/48k under 0.01, at 12k/12k/24k under 0.03,
and at 7k/12k/24k under 0.1. Blocks of 12k and larger are unchanged.

The default is therefore 0.01. It is the only tested value that refuses every
former 1 MeV early stop. The MCNP/Geant4 pass value 0.1 remains an opt-in for
exploratory runs. $V_{\max}=1$ disables the gate. Since v2 fixed
$V_{\max}=1/E_{\min}$, a profile with an explicit non-default
`min_effective_histories` now changes its stopping rule. The
`variance-threshold-v3` marker re-keys every adaptive identity for this
reason. Anchors: `test_vov_threshold_and_ess_floor_gate_independently`
(each threshold binds alone, `guards` carries both),
`test_vov_threshold_is_compared_to_m4_over_m2_squared`, and
`test_settings_are_validated`.

### VOV trend diagnostic (#1262)

`vov_trend_slope(points)` is recorded for every observable. It fits the
least-squares slope of $\ln\mathrm{VOV}_n$ against $\ln n$ over decision points
(block ends before v4) with $n\ge n_{\rm last}/2$ and $\mathrm{VOV}>0$, and is
`None` below three such points. The #1254 slope values below were computed per block end. For independent histories with a sampled fourth moment,
$\mathrm{VOV}_n\simeq(\kappa-1)/n$, so the slope tends to $-1$; this is
the trend MCNP checks over the last half of a run. A score that dominates
$M_4$ and arrives late raises VOV and flattens or reverses the slope. The
slope is recorded but never gates a stop. Anchors:
`test_vov_trend_slope_limits` (exact $1/n$ gives $-1$, early and non-positive
points are excluded) and `test_vov_trend_slope_is_recorded_and_never_gates`
(equals an independent NumPy fit to prefix VOVs; a late heavy score gives a
positive slope; the stop is unchanged).

Applied to the #1254 captures at their recorded stops, the slope does not
separate the false 1 MeV 15° stops. HOPG at 5k, 12k and 24k gives $-0.97$,
$-0.90$ and $-1.01$, because VOV decays normally until a heavy electron is
drawn. hBN gives $-0.15$ to $+0.38$. Several converged 30 MeV 15° stops give
positive slopes ($+5.43$ for hBN and $+1.18$ for HOPG at 24k blocks), and
30 keV stops give $-0.6$ to $-1.2$. The slope therefore stays a diagnostic.
Like VOV itself, it cannot detect a population class that has not yet been
sampled.

## Independent v3 extension verification (2026-11-17, #1262)

A fresh-context verifier derived the v3 gate from the claim before reading
`StoppingMonitor.observable_status`, then compared it with commit `39603e682`.

**Bounds.** Let $d_i=m_i-\bar m$, so $M_2=\sum_i d_i^2$, $M_4=\sum_i d_i^4$
and $\sum_i d_i=0$. Cauchy–Schwarz on $(1,\dots,1)$ and $(d_i^2)$ gives
$M_2^2\le nM_4$, so $M_4/M_2^2\ge1/n$, with equality for two-point symmetric
samples. Since the $d_i^2$ are non-negative,
$M_2^2=M_4+\sum_{i\ne j}d_i^2d_j^2\ge M_4$, so $M_4/M_2^2\le1$. For $M_2>0$
the zero-sum constraint forces at least two non-zero $d_i$, so the upper bound
is strict. A single spike in $n$ zeros gives $(n^2-3n+3)/[n(n-1)]\to1$. Thus
$\mathrm{VOV}=M_4/M_2^2-1/n\ge0$ in exact arithmetic. The clip at zero only
removes rounding, and $\max(0,\mathrm{VOV})+1/n=\max(1/n,M_4/M_2^2)$.
$V_{\max}\ge1$ therefore admits every sample with $M_2>0$; the explicit
short-circuit guards an ulp of rounding.

**Gate.** For CXR the gate passes iff $s/|\bar m|\le\varepsilon$, or
$V_{\max}\ge1$, or $\mathrm{VOV}+1/n\le V_{\max}$. The contribution floor
$(\sum m)^2/\sum m^2\ge E_{\min}$ is a separate guard. Under v2 the right-hand
side was `1.0 / min_effective_histories`. At $E_{\min}=100$ this is the double
`0x1.47ae147ae147bp-7`, identical to the literal `0.01`. The v2 disable branch
$E_{\min}=0$ is not reached at the default. Default-policy decisions are
therefore bit-identical.

**Interpretation.** For independent histories with finite fourth moment,
$\operatorname{Var}(s^2)/\sigma^4\simeq(\kappa-1)/n$. The plug-in kurtosis
$\hat\kappa=nM_4/M_2^2$ gives $(\hat\kappa-1)/n=\mathrm{VOV}$. The delta method
halves the relative uncertainty for $s$, and hence for the RSE. At
$\mathrm{VOV}=0.01$: $\sqrt{0.01}=0.10$ (variance) and $0.05$ (RSE). At
$0.1$: $0.316$ and $0.158$. The gate passes near $n\simeq\hat\kappa/V_{\max}$.

**Code comparison.** `RunningMoments.variance_of_variance` returns
$\max(0,(\sqrt{M_4}/M_2)^2-1/n)$, $0$ at $M_2=0$ and `None` for $n<2$. The
expression $(\sqrt{M_4}/M_2)^2$ equals $M_4/M_2^2$ up to rounding and avoids
overflow. `cv = rse * sqrt(n)` equals $s/|\bar m|$ with $s^2=M_2/(n-1)$. The
predicate, `guards` keys and `_validate_unit_fractions` all match the claim.
The validator range is $(0,1]$; it rejects `True`, NaN and $\pm\infty$.

**Numeric reproduction.** An independent NumPy probe outside the repository
ran three checks:

- 20,000 random samples (normal, lognormal, Pareto and sparse spikes,
  $2\le n<400$) give $\min(M_4/M_2^2-1/n)=-1.1\times10^{-16}$ (rounding) and
  $\max(M_4/M_2^2)<1$.
- In 1,260 `run_stopping_rule` decisions, on 60 lognormal populations (some
  with injected $10^3\times$ tails), $V_{\max}\in\{10^{-3},0.005,0.01,0.03,0.1,0.5,1\}$
  and $E_{\min}\in\{0,100,1000\}$, the `variance_sampling`,
  `effective_sample_size`, VOV and `guards` values matched the formulas
  recomputed from the realized prefix, with 0 mismatches.
- `1.0/100.0 == 0.01` holds.

The anchors in `test_adaptive_stopping.py` (`vov_threshold`,
`settings_are_validated`, `variance_guard`; 33 passed) and the
`test_profile_precision.py` `variance_threshold` test (1 passed) are green.

**Findings.** Both are minor and do not change the verdict:

- The claim text attributes both bounds to Cauchy–Schwarz. Only the lower
  bound comes from it; the upper bound follows from non-negative cross terms.
- "Zero variance passes this gate" holds for a constant positive signal
  through CV. For a constant zero signal with $n<1/V_{\max}$, the code fails
  the gate ($1/n>V_{\max}$, CV undefined). This was already true in v2. It is
  immaterial because the RSE guard fails zero signal anyway.

**Verdict.** The extension matches and is `rederived`. The anchors
`test_vov_threshold_and_ess_floor_gate_independently` and
`test_vov_threshold_is_compared_to_m4_over_m2_squared` pin the predicate to
an independently computed $M_4/M_2^2$, which supports `anchored` for the
extension. Human sign-off is still pending.

## Independent v4 extension verification (2026-11-17, #1253)

A fresh-context verifier derived the v4 decision-point rule from the claim,
the #1253 issue and the sources before reading the bodies of
`decision_points`, `StoppingMonitor`, `transport_monitored_blocks`,
`IndependentStoppingMonitor`, `AdaptiveChunkPlan` and `_batch_summary`, then
compared them at commit `73a79ed4a` (rebased onto #1262's v3 gate).

- **Claim**: `adaptive-sample-size-stopping` (v4 extension) —
  `montecarlo/runner/adaptive.py::{decision_points, StoppingMonitor, batch_spans, _batch_summary}`,
  `block_transport.py::transport_monitored_blocks`,
  `independent.py::IndependentStoppingMonitor`,
  `stream_sizing.py::AdaptiveChunkPlan` — Chow–Robbins (1965) and
  Glynn–Whitt (1992) sequential stopping and batch means; Chan–Golub–LeVeque
  merge; {eq}`eq-adaptive-decision-points` and {eq}`eq-adaptive-stop-rule`.
- **Filters**: units pass (every guard is dimensionless: ratios of means,
  shares, counts); limits pass ($\delta\to0$, $k=0$, $n_{\min}=n_{\max}$,
  single dominant electron), with the $f=0$ exception below; signs and
  conventions pass, with the floating-point ceiling note below.
- **Re-derivation**: `matches` for the grid, overshoot, check count, drift
  variance, tolerance, earliest stop, dominant-electron refusal, CXR gate and
  batch means. `differs` for chunk independence of the brem statistics at
  the last ulp, described below.
- **Verdict**: `discrepancy` (implementation/test level, not in the
  statistical equations). The anchor test
  `test_adaptive_output_does_not_depend_on_chunk_size` fails at HEAD.
- **Write-up**: `docs/validation/beam-transport/adaptive-sample-size-stopping.md`
  (this section).
- **Suggested ledger change**: given to the task owner; claim status
  unchanged, never `signed-off`.

### Source-first derivation

**Grid.** For integer $d\ge1$ and $\delta>0$, $d(1+\delta)>d$ gives
$\lceil d(1+\delta)\rceil\ge d+1$. The $\max(d_j+1,\cdot)$ therefore binds
only at $\delta=0$, which the validator rejects ($\delta$ must be positive
and finite). For $d<1/\delta$, $\lceil d\delta\rceil=1$, so the steps are $+1$.
Let $d_{J-1}$ be the last refused point. Then
$d_J-d_{J-1}\le\lceil d_{J-1}\delta\rceil<d_{J-1}\delta+1$, so the
overshoot past the last refused count is below $\delta$ relative plus one
electron. Points added for the pilot and its projection only shorten gaps.
The $+1$ phase has at most $1/\delta$ steps. Above it, each step multiplies
by at least $1+\delta$, so the grid has at most
$\log(n_{\max}/n_{\min})/\log(1+\delta)+1/\delta+1$ points, plus 2 for the
pilot and projection. This is $O(\log(n_{\max}/n_{\min})/\delta)$. As
$\delta\to0$, the $+1$ phase covers every count up to $\min(n_{\max},1/\delta)$.
The grid then tends to every electron.

**Drift variance.** Let the $m_i$ be i.i.d. with mean $\mu$ and variance
$\sigma^2$. Write $\bar m_n=[d\,\bar m_d+(n-d)\,\bar m_{(d,n]}]/n$. Then
$\bar m_d-\bar m_n=\frac{n-d}{n}(\bar m_d-\bar m_{(d,n]})$, with independent
terms, so

$$
\operatorname{Var}(\bar m_d-\bar m_n)
=\Big(\frac{n-d}{n}\Big)^2\Big(\frac{\sigma^2}{d}+\frac{\sigma^2}{n-d}\Big)
=\sigma^2\frac{n-d}{nd},
\qquad
\frac{\operatorname{Var}(\bar m_d-\bar m_n)}{\mu^2}=r_n^2\Big(\frac nd-1\Big),
$$

where $r_n^2=\sigma^2/(n\mu^2)$. This matches the record. On the geometric
grid, $n/d_{J-j}-1\approx(1+\delta)^j-1\approx j\delta$. The tolerance
$z\,r_n\sqrt{(n/d-1)/\delta}$ is therefore a test of $z/\sqrt\delta$ drift
standard deviations at every lag: $2.85$ at $z=0.9$, $\delta=0.1$. The
two-sided Gaussian tail is $0.44\,\%$ per comparison, or at most $1.3\,\%$
over $k=3$ correlated lags. The plain floor $f\varepsilon$ binds only when
$r_n\ll\varepsilon$. As $\delta\to0$ at fixed lag, $z/\sqrt\delta\to\infty$,
so the guard becomes vacuous and the RSE and share tests remain. At $k=0$ the
guard is off and the earliest stop is $n_{\min}$.

**Earliest stop.** The guard needs $k$ points before $n$, so the earliest
stop is $d_k\approx n_{\min}(1+\delta)^k$. This depends on the policy alone
and has no block or chunk term.

**Single dominant electron.** Add one electron of share
$\sigma=h/(S_1+h)$ in $(d_{J-1},n]$, over light electrons whose
contribution to $r_n^2$ is $L^2\approx(1-\sigma)^2\mathrm{CV}^2_{\rm light}/n$.
From {eq}`eq-adaptive-ess-identity`, $r_n^2\approx\sigma^2+L^2$. The move
over the arrival segment is $\sigma-O(1/n)$, and the allowance is
$\approx z\,r_n\sqrt{(n/d_{J-1}-1)/\delta}\le z(1+\tfrac{1}{2\delta d})r_n$.
The arrival check is refused when $\sigma>f\varepsilon$ and
$\sigma>z\sqrt{\sigma^2+L^2}$, that is, when $L/\sigma<\sqrt{1/z^2-1}=0.48$
at $z=0.9$. In other words, the heavy electron must hold more than about
81 % of the variance. Otherwise the stop may proceed, but then
$r_n\ge\sigma\sqrt{1+L^2/\sigma^2}>\sigma$, so the reported error covers the
jump. Any $z<1$ keeps the refusal. $z=0.9$ leaves an 11 % margin
($1/z=1.11$), while admitting i.i.d. drift at $2.85\sigma$. The share cap $c$
refuses $\sigma>c$ regardless. The CXR variance gate refuses while the
electron dominates $M_2$, since $M_4/M_2^2\to1>V_{\max}$. The record's
statement holds with the qualifier "while the electron dominates $S_2$",
which it states.

**Batch means with unequal batches.** Let
$\bar S_b=n_b^{-1}\sum_{i\in b}S_i$. Then $S=n^{-1}\sum_iS_i=\sum_b(n_b/n)\bar S_b$
exactly. The weighted ANOVA identity
$\sum_bn_b(\bar S_b-S)^2=\sum_bn_b(\bar S_b-\mu)^2-n(S-\mu)^2$ has expectation
$K\sigma^2-\sigma^2$. Hence $\hat\sigma^2=\sum_bn_b(\bar S_b-S)^2/(K-1)$ is
unbiased, $\widehat{\operatorname{Var}}S=\hat\sigma^2/n$, and equal batches give
$s(\{\bar S_b\})^2/K$. At $K=1$ there is no estimate. These match the record.

### Implementation comparison

| Item | Code | Result |
|---|---|---|
| Grid | `decision_points` (`adaptive.py:175`) | matches, up to the floating ceiling (below) |
| Pilot and projection | `_with_point`, `decide`, `_project` | match; pilot in $[n_{\min},n_{\max}]$ is validated; projection is unrounded and inserted |
| Stability window and tolerance | `_stability` (`adaptive.py:309`), `STABILITY_DRIFT = 0.9` | matches; history includes every folded point, so it needs $k+1$ points; $k=0$ returns `(0, f eps)` |
| CXR gate | `observable_status` | unchanged from v3: `vov + 1/n <= max_variance_of_variance`, CV branch, $V_{\max}\ge1$ disables |
| `vov_trend_slope` | `_fold` appends `(point, VOV)` per decision point | consistent with v4; the record text still says "block ends" |
| Prefix re-transport | `transport_monitored_blocks` | a stop or `cut_at` at $d<$ `stop` truncates the monitor and transports `[start, d)` again; kept chunks join as before |
| Energy-range replay | `transport_case_blocks` | rechecks at `realized` only; on failure it grows to `monitor.next_point()`, a decision point |
| Independent brem | `IndependentStoppingMonitor.should_stop` | both grids are visited in electron order; a freeze at $d<n$ sets `cut_at` |
| Batch means | `batch_spans` (bounds $\lfloor bn/K\rfloor$), `_batch_summary`, `batch_means` (scale $n/n_b$) | match the weighted estimator |
| Chunk plan | `AdaptiveChunkPlan.next_stop` | output-neutral; `size = max(floor, fits)` lets the floor exceed the memory cap (the docstring says "never exceeds") |

### Numeric checks

These are independent probes, outside the repository tests:

- **Grid.** 3,000 random $(n_{\min},n_{\max},\delta)$ with $\delta$ from
  $10^{-3}$ to 1 (2.39M steps). Every step satisfies
  $d_{j+1}-d_j\le\delta d_j+1$ (worst excess $1.4\times10^{-18}$), and every
  grid length is within $\log(n_{\max}/n_{\min})/\log(1+\delta)+1/\delta+1$.
  At $\delta=10^{-6}$ the grid from 2 to 50 is every integer.
- **Drift variance.** 20,000 exponential replicates at $n=1000$, $d=750$
  give $\operatorname{Var}=3.339\times10^{-4}$, against
  $\sigma^2(n-d)/(nd)=3.333\times10^{-4}$.
- **Stability false refusals.** Over 200 i.i.d. lognormal($\sigma=1$)
  populations at $\varepsilon=0.02$ with the other guards off, 44 of 3,561
  RSE-passing checks are refused by stability (1.2 %). This matches the
  $\le1.3\,\%$ estimate.
- **Dominant electron.** The monitor status at the arrival point gives
  move/limit $=1.107$–$1.111$ (refused) with no light variance. It stays
  above 1 at $L/\sigma\le0.24$ and falls to $0.89$ at $L/\sigma=0.59$, with
  $r_n=0.058>\sigma=0.05$. This matches the threshold
  $L/\sigma\approx0.48$.
- **Batch means.** At $n\in\{23,47,101,1003\}$ (batch sizes $\{1,2\}$ to
  $\{50,51\}$, 4,000 replicates), the weighted mean reproduces the full mean
  to $10^{-13}$, and $n\,\mathbb E[\mathrm{se}^2]/\sigma^2=0.98$–$1.01$.
- **#1252 replay.** `checks/adaptive_decision_cadence_1253.py`, re-run on
  the 12 local captures with SHA-256 values equal to the report's, reproduces
  `adaptive_decision_cadence_1253_results.json` exactly except
  `sampling_model` (`decision-points-v3` then, `decision-points-v4` now).
  The default $V_{\max}=0.01$ is the old $1/E_{\min}$ at $E_{\min}=100$, so
  the default decisions are unchanged by the rebase, as claimed. The 30 MeV
  stops at 3,996 are $d_3$ of the grid from 3,000, the earliest stop. This
  replay feeds saved per-electron values, so it does not exercise the brem
  reducer.

### Findings

1. **Anchor test red at HEAD (merge).** At
   `tests/montecarlo/runner/test_adaptive_stopping.py:667`, the slow
   `test_adaptive_output_does_not_depend_on_chunk_size` disables only
   `min_effective_histories`. Under v2 that also disabled the VOV gate; under
   v3/v4 the default $V_{\max}=0.01$ stays active. The case runs to
   `max_electrons` at 200, and line 673 fails (`'max_electrons' ==
   'converged'`). Run with `max_variance_of_variance=1.0` (as
   `test_adaptive_run_is_the_fixed_n_run_at_its_realized_count` does), the
   same case converges at 54 electrons. It replays the realized range, cuts
   inside the 64-electron chunk, and matches across floors 1, 7 and 64 and
   the fixed-54 run bit for bit. The acceptance claim holds in that
   configuration, but its anchor is red.
2. **Brem statistics are not bit-identical across chunkings.** At the
   failing test's settings (default $V_{\max}$, realized 200), floors 1 and
   64 give identical output arrays and CXR values. Yet 32 of 200 per-electron
   brem band integrals differ by up to $4.6\times10^{-16}$ relative. The brem
   `max_electron_share`, `variance_of_variance` and `vov_trend_slope` differ
   in the last digits. The probable cause is in the per-electron path of
   `mc_brem_spectrum` (`spectrum/brem.py:1043`, row loop at `:1158`):
   `density @ band_q` and the `bincount` accumulate over row chunks whose
   boundaries depend on the transport chunk's row count. The record's
   "every moment, guard value and decision is bit-identical under any
   chunking" therefore holds for CXR but only up to rounding for brem. A
   brem guard within an ulp of its threshold could flip a decision. This is
   measure-zero in practice, but it is not the stated guarantee. The CUDA
   fused per-electron reduction was not examined.
3. **$f=0$ crashes.** The validator accepts `stability_fraction = 0`
   ("finite and non-negative"). With $k\ge1$, `_stability` starts from
   `worst = (0.0, floor) = (0.0, 0.0)`, so `worst[0] / worst[1]` raises
   `ZeroDivisionError` at the first evaluated check (`adaptive.py:338`). This
   was reproduced with `run_stopping_rule`. In v3, $f=0$ meant a zero-move
   requirement.
4. **Floating ceiling.** `math.ceil(point * (1.0 + growth))` rounds up when
   $d\delta$ is an integer and $1+\delta$ is inexact. The grid from 200 at
   $\delta=0.1$ is 200, 221, 244, 269, not 200, 220, 242, 267 (10,605
   mismatches in 2.39M steps). The bounds above still hold
   ($\le\delta d+1$). It is deterministic and policy-only, so decisions stay
   chunk-independent. However, {eq}`eq-adaptive-decision-points` and the
   test's float-computed bound do not describe it exactly.
5. **Stale text.** The ledger Claim says "sequential block-wise stopping",
   "first block end" and "last `k` block ends" with tolerance `f*target`.
   The VOV trend section above and `statistical-methods.md` (lines 138 and
   194: "equal electron blocks", "after each block", "over block ends") are
   also stale. The #1254 slope values were computed per block end, not per
   decision point.

### Verdict

The v4 statistical design rederives: the grid, overshoot $\le\delta$ (plus
one), $O(\log n/\delta)$ checks, the drift variance $r_n^2(n/d-1)$, the
$z=0.9$ tolerance, the earliest stop $n_{\min}(1+\delta)^k$, the preserved
dominant-electron refusal (when it dominates $S_2$), the unchanged CXR gate,
and unbiased unequal-batch means. The verdict is still `discrepancy`,
because the #1253 chunk-independence anchor fails at HEAD and the brem
statistics are chunk-independent only to rounding. Fix finding 1, then
either make the brem per-electron reduction chunk-invariant or state "CXR
bit-identical, brem to rounding". After that, the extension can return to
`rederived`. Human sign-off is pending.

## Independent v4 fix verification (2026-11-17, #1253)

A fresh-context verifier checked the five v4 findings against the fixes in
`76eba7d25` and `35cbd04e1`. The source tree is unchanged at `96ada34fa`,
which adds only GPU evidence. The expected behaviour was stated before the
diffs were read.

- **Claim**: `adaptive-sample-size-stopping` (v4 fixes) —
  `adaptive.py::{decision_points, _excess, StoppingMonitor._stability}`,
  `spectrum/brem.py::mc_brem_spectrum` (`electron_band_weights`).
- **Filters**: units, limits and signs pass; nothing new is dimensional.
- **Re-derivation**: `matches` for items 1, 3 and 4. `differs` for item 2 at
  one remaining site, the layered transmission product.
- **Verdict**: `discrepancy`. The scope is narrow: the residual is
  $\le2.2\times10^{-16}$ relative and occurs only for absorbers of two or more
  layers. The statistical design stays rederived.
- **Write-up**: this section.
- **Suggested ledger change**: given to the task owner. Status is unchanged
  and never `signed-off`.

**1. Chunk-independence anchor.** The expected behaviour: at every floor
the realized count, statistics and output must match. At floor 64 the stop
must fall inside the first chunk, and $[0,d)$ must be transported again.
The slow test is green (92 passed with
`test_brem_primary_band.py`). An instrumented copy logged each
`simulate_block` call. At floor 64 the calls were `(0, 64)`, then the prefix
`(0, 54)`, then the realized-range replay `(0, 54)`. Floors 1 and 7 reach 54
through the points 40, 44, 49 and 54. All three floors stop at 54
(`converged`), and floor 64 equals the fixed 54-electron run bit for bit.
The test checks only `realized % 64`. It does not assert that the prefix was
re-transported or compare with the fixed run; the fixed-run comparison is in
`test_adaptive_run_is_the_fixed_n_run_at_its_realized_count`.

**2. Host brem band integrals.** Write electron $e$'s value as
$B_e=\sum_{r\in e}w_r\sum_k\rho_{rk}q_k$. It is chunk-invariant only if
three conditions hold:

- every per-row factor ($T_{\rm abs}$, $\mathrm d\sigma$, the row sum) is
  computed from that row alone;
- the element contributions are added in a fixed order;
- the rows of $e$ are summed sequentially in row order.

The new `_accumulate`/`_electron_band` (`brem.py:1043-1052`) meets the last
two conditions: a per-row `sum(axis=1)` and one `bincount` in row order. An
independent probe ran 2,816 comparisons. It used the eedl, Bethe–Heitler,
BremsLib and partial-BremsLib models, single-slab and `_LAYERS` absorbers,
sorted and interleaved electron ids, 97 and 500 nodes, one and two
elements, and `photon_block` set to `None` or 40. Each case was compared at
row chunks 1–299 and on electron-prefix subsets $c\in\{1,4,11,24\}$. The
results:

- Every single-slab case is bit-identical.
- The pre-fix `brem.py` from `76eba7d25` fails 505 of 704 single-slab
  comparisons, so the probe detects the original defect.
- 42 layered comparisons still differ, by up to $2.2\times10^{-16}$
  relative, and all of them at row chunk 1.

The cause is `brem.py:1170`, `path_start[sl] @ mu_by_layer`. With two or more
layers, a one-row slice takes NumPy's vector–matrix BLAS path, while a
multi-row slice takes `gemm`. In a direct NumPy check, (1, 2) @ (2, 500)
differs from the same row of an (M, 2) product in 17 % of cells; M from 2 to
64 agrees. A singleton slice occurs whenever a call's row count is
$\equiv1\pmod{\texttt{chunk}}$. Production scoring calls `mc_brem_spectrum`
per emitting layer with every absorbing layer included (`emission.py:131`).
The row count therefore depends on the transport chunk, and film-on-substrate
cases are exposed. The new test passes with chunk 1 only because of its data.
The fix should compute $\tau=\sum_\ell p_\ell\mu_\ell$ in a fixed elementwise
order. The alternative is to scope the claim to single-slab absorbers.

The non-per-electron spectrum path (`spec += row_weight @ density`,
`:1046`) is unchanged by the diff, and $\sum_eB_e$ equals
$N_e\,S\cdot q$ to $10^{-12}$ in every probe case. The fused CUDA band
reduction offers no chunk-invariance argument: it splits rows at 65,536 and
accumulates with `jit.atomic_add` (`brem_band_jit.py:132`). A CuPy
weighted `bincount` on the unfused device path is presumably also atomic.
Neither was run.

**3. `stability_fraction = 0`.** The expected behaviour: the limit is
$\max(0,zr_n\sqrt{(n/d-1)/\delta})$, which is zero only at $r_n=0$, so any
move must then be refused. In exact arithmetic a zero-variance sample has
zero moves, so a constant population must still stop. `_excess`
(`adaptive.py:176`) maps a zero move to 0 and a positive move over a zero
limit to $\infty$. The pass test stays `move <= limit` (`:380`). The probe
results:

- Constants 1, 0.1, 1/3, $\pi$ and $7.77\times10^{-5}$ converge at
  $d_3=134$ from 100.
- 600 lognormal runs with $f\in\{0,0.5\}$ and $k=0..4$ raise no exceptions.
- `_excess(1e-300, 0)` returns `inf`.

The anchor's second half (`test_adaptive_stopping.py:577`) asserts only that
a limit is reported, not that a refusal occurs. A positive move with a zero
limit cannot be built from real data, so `_excess` is pinned only
indirectly.

**4. Exact ceiling.** Since $d\delta>0$,
$d+\max(1,\lceil d\,p/q\rceil)=\lceil d(1+p/q)\rceil$ for $d\ge1$, and
`-(-a // b)` is $\lceil a/b\rceil$ for $b>0$. Here
`Fraction(repr(float(g)))` is the shortest round-trip decimal. Reading
$\delta$ as the binary double would not fix the finding: $0.1_{\rm bin}>1/10$
still gives 221. The probe compared 3,312 grids (8.97M steps) against an
independent `Decimal`-based reference. The grids used $\delta$ from
$10^{-4}$ to 10 plus 0.1, 0.3, 0.7, $1/3$, $10^{-6}$, $10^3$, $10^{300}$ and
$5\times10^{-324}$. The results:

- No mismatches.
- No step exceeds $\delta_{\rm bin}d+1$.
- The check-count bound is never exceeded.
- Example grids: 200, 220, 242, 267, 294 and 3000, 3300, 3630, 3993.
  At $10^{-6}$ the steps are $+1$; at $10^{300}$ the grid is
  $[n_{\min},n_{\max}]$; a `np.float64` input works.

The test's floating bound `ceil(d*(1+growth))` fails only for
$\delta<2^{-53}$, where $1+\delta=1$. This is a limit of that check, not of
the code.

**5. Text and replay.** Rerunning `checks/adaptive_decision_cadence_1253.py`
on `scratch/1253-captures/out-{nv-30,cpu-1000,cpu-30000}` reproduces
`adaptive_decision_cadence_1253_results.json` byte for byte. It reports no
failures, chunk independence for all 12 cases, overshoot from $-0.62\,\%$
to $+7.33\,\%$, and 30 MeV 15° stops at 3,993 $=d_3$ from 3,000. The ledger
Claim, Code and Notes and the v4 Claim text match the code, with three
exceptions:

- "every moment … bit-identical under any chunking" and the host-brem
  sentence overstate item 2 for layered absorbers.
- "The 30 MeV stops fall … to 3 993" and the Notes' "30 MeV 3 993" omit the
  60° stops, which are at 32,541 against 31k–192k for the block rule.
- The Claim's "exact rational ceiling" omits that $\delta$ is read as its
  decimal; the derivation page states it.
