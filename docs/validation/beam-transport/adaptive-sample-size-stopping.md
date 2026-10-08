# `adaptive-sample-size-stopping`

Ledger row: [`adaptive-sample-size-stopping`](../ledger-transport-background.md#adaptive-sample-size-stopping). Validation: `adaptive-sample-size-stopping`. Code: `montecarlo/runner/adaptive.py` (`RunningMoments`, `StoppingMonitor`, `run_case_adaptive`, `batch_means`). The user-facing description is in [statistical methods](../../computation/statistical-methods.md#adaptive-electron-counts).

## Claim

For a case transported in equal electron blocks of size $B$ on the counter-addressed per-electron or CUDA cores, the run stops at the first block end $n \ge n_{\min}$ that meets four conditions for every watched observable $o$:

```{math}
:label: eq-adaptive-stop-rule

r_{o,n} = \frac{s_{o,n}}{\sqrt{n}\,\lvert\bar m_{o,n}\rvert} \le \varepsilon,
\quad
\frac{\max_{i\le n} m_{o,i}}{\sum_{i\le n} m_{o,i}} \le c,
\quad
\frac{(\sum_{i\le n} m_{o,i})^2}{\sum_{i\le n} m_{o,i}^2} \ge E_{\min},
\quad
\max_{1\le j\le k}\frac{\lvert\bar m_{o,n-jB}-\bar m_{o,n}\rvert}{\lvert\bar m_{o,n}\rvert} \le f\varepsilon .
```

Otherwise it stops at $n_{\max}$, flagged `statistics_limited`. The terms are:

- $m_{o,i}$ is electron $i$'s grid-independent scalar. For `"line"` it is the line mass $\sum_\ell w_\ell\pi/a_\ell$ over the lines the kernel keeps for a fixed monitor band, the #201 audit quantity. For `"brem"` it is the band integral $\sum_E q_E\, \mathrm{d}N_i/\mathrm{d}E\,\mathrm{d}\Omega$ of its bremsstrahlung density, with trapezoid weights on the fixed brem grid.
- $\bar m_{o,n}$ and $s_{o,n}$ are the sample mean and Bessel-corrected standard deviation of $m_{o,i}$.
- $r_{o,n}$ is the relative standard error of the mean.

The realized result is the fixed-$N$ result at $N=n$, bit for bit. Under the assumptions below, $\bar m_{o,N}\pm z\,r_{o,N}\bar m_{o,N}$ has coverage tending to the nominal $2\Phi(z)-1$ as $\varepsilon\to0$, and the stopping bias $E[\bar m_{o,N}]-\mu_o$ is $o(\varepsilon\mu_o)$.

## Sources

- Sequential fixed-width intervals: Y. S. Chow and H. Robbins, "On the asymptotic theory of fixed-width sequential confidence intervals for the mean," *Ann. Math. Statist.* **36**, 457–462 (1965). The rule $N=\inf\{n\ge n_0: s_n^2 + n^{-1}\le n d^2/z^2\}$ is asymptotically consistent ($P(\lvert\bar X_N-\mu\rvert\le d)\to 2\Phi(z)-1$) and efficient as $d\to0$ for any finite-variance i.i.d. population. Here $d=z\varepsilon\lvert\bar m_n\rvert$ is a relative half-width, which is the same theorem applied with a random scale that converges to $\lvert\mu\rvert$.
- Sequential stopping in simulation output analysis: P. W. Glynn and W. Whitt, "The asymptotic validity of sequential stopping rules for stochastic simulations," *Ann. Appl. Probab.* **2**, 180–198 (1992). This covers relative-precision rules and batch means.
- Running moments: B. P. Welford, *Technometrics* **4**, 419–420 (1962); T. F. Chan, G. H. Golub and R. J. LeVeque, "Algorithms for computing the sample variance: analysis and recommendations," *Amer. Statist.* **37**, 242–247 (1983), pairwise merge.
- Effective sample size: L. Kish, *Survey Sampling* (Wiley, 1965), $(\sum w)^2/\sum w^2$, applied here to the contributions $m_i$.

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

**Pilot projection.** For i.i.d. values $r_n\propto n^{-1/2}$, so the count reaching $\varepsilon$ from a pilot at $n_p$ is $N\approx n_p(r_{n_p}/\varepsilon)^2$. The projection only skips checks. The rule is re-applied at every block end from $N$ on, so the projection cannot stop a run that does not meet the rule.

**Per-bin batch means.** With the realized run split into $K$ equal batches $b$ of $B$ electrons, each batch spectrum $\bar S_b$ is an i.i.d. mean of $B$ electron spectra. So $s(\{\bar S_b\})/\sqrt K$ estimates the standard error of the run's mean spectrum per bin, which is the Glynn–Whitt batch-means estimator with independent batches. With normalization by the full count $n$, the batches satisfy $\sum_b (B/n)\bar S_b = S$, the run's spectrum, by linearity of the incoherent reduction.

## Assumptions

- The per-electron values have finite variance. A heavy tail with infinite variance (Pareto index $\le 2$) breaks the CLT and every guard. The guards only act on sampled values.
- The incoherent reduction is a sum over electrons. The coherent line sum couples electrons ([coherent versus incoherent statistics](../../computation/statistical-methods.md#coherent-versus-incoherent-statistics)), so adaptive runs refuse `coherent_emission`. They also refuse the lockstep core, whose shared generator is not prefix-stable, and GDF beams.
- All watched observables share one realized count, with `Ne == Ne_brem`. Per-electron cutoffs depend on the line/brem split, so a shared count keeps every block's transport equal to the fixed-$N$ run's.
- The monitor band is fixed before transport: the case's policy bandwidth, else its line-grid ends. Under the `resonance-population` policy the final axis may end below the bandwidth ceiling, so the monitor counts tails the final axis truncates. That share is audited separately (`line-grid-resonance-bandwidth`).

## Limiting cases

- **Fixed-$N$ limit.** $n_{\min}=n_{\max}=N$ is the fixed-$N$ run. No check precedes $N$, nothing is replayed, and the output is identical (`test_min_equal_max_is_the_fixed_n_run`). For any stop, the output equals the fixed-$N$ run at the realized count, including energy-spread replay and the bunch centroid (`test_adaptive_run_is_the_fixed_n_run_at_its_realized_count`).
- **Gaussian population.** $\mathcal N(1,0.5^2)$, $\varepsilon=5\,\%$, $B=10$, $n_{\min}=50$, 1000 seeds. Coverage of $\pm1.96\,r_N$ is 0.934 with the guards off and 0.943 with the defaults, against 0.95 nominal. The stopping bias is $(+5.4\pm1.6)\times10^{-3}$ and $(+1.3\pm1.4)\times10^{-3}$, at most $0.11\varepsilon$ (`test_gaussian_population_has_near_nominal_coverage`). The small undercoverage and positive bias are the finite-$N\approx100$ sequential effect: $r_n$ falls when $\bar m_n$ is high.
- **Zero signal.** A band with no lines gives $\bar m=0$, $r$ undefined. The rule never converges and the run ends `statistics_limited` at $n_{\max}$.

## Anchors

`tests/montecarlo/test_adaptive_stopping.py`:

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

## Known limits

- **Wald undercoverage on skewed populations.** The light case's interval $\pm1.96\,r_N$ covered 17 of 20 seeds. At fixed $N=400$ two 80-seed sets cover 86–88 %: a property of the sample-SD interval for a skewness-8 population at a few hundred electrons. The stop adds to it at small $n_{\min}$: pooled over 160 seeds at $\varepsilon=5\,\%$, coverage is 0.819, 0.838 and 0.875 at $n_{\min}=100$, 200 and 400, against 0.875 at fixed $N=400$. Treat $r_N$ as a scale, not a calibrated 95 % interval, for such observables.
- **Small $n_{\min}$ on skewed populations.** The same study found a negative stopping bias of $0.3$–$0.4\varepsilon$ at $n_{\min}\le200$, reduced to $0.13$–$0.15\varepsilon$ at $n_{\min}=400$. When the sample mean and SD are low together, the rule stops early. Set $n_{\min}$ near the count the target implies.
- **Unsampled tails.** No guard sees a class that has not appeared. Only $n_{\min}$ and the ESS floor raise the chance of drawing it. The 5 MeV h-BN detector-cone case (#201) is the motivating instance. In the remote measurement (2026-10-08, qlmc) it reached $n_{\max}=16\,000$ with a line relative SE of 0.39 and ended `statistics_limited`, so the guards did not let it stop early; it needs variance reduction (#203) to converge.

## Independent verification (2026-10-08)

Fresh-context verifier; did not write the implementation. I read the ledger row and this record's claim, sources, assumptions and limiting cases first, rederived the results below, and only then read `montecarlo/runner/adaptive.py`, `montecarlo/runner/block_transport.py::transport_case_blocks`, `montecarlo/spectrum/brem.py::mc_brem_spectrum` (`electron_band_weights`), `montecarlo/spectrum/lines/_kernels.py::_accumulate_edge_truncation` and `_precision.py::Precision`. The numeric checks use a separate NumPy implementation of {eq}`eq-adaptive-stop-rule`, written from the equations and not from `run_stopping_rule`. The only implementation code reused is `case_measure`, to obtain the hopg per-electron populations.

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

The anchor file `tests/montecarlo/test_adaptive_stopping.py` passed: 45 tests in 33 s. The tests assert bounds, not the quoted numbers, so I reproduced the numbers with the independent rule:

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
