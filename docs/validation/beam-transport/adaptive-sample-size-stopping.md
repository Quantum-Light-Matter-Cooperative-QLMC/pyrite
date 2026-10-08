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
- `test_guards_turn_a_false_early_stop_into_statistics_limited` and `test_guards_suppress_false_stops_across_seeds` cover heavy tails. On the constructed population, the bare rule stops at 200 electrons 44 % low with a reported 7 %, while the guarded run ends `statistics_limited` at $n_{\max}$. Across 40 random seeds, false stops fall from $\ge 20$ to $\le 1$.
- `test_stability_guard_waits_out_a_late_heavy_electron` and `test_pilot_projection_skips_checks_until_the_projected_count` cover the stability guard and the pilot.
- `test_same_seed_and_target_replay_the_same_count_and_output`, `test_coherent_route_rejects_auto` and `test_lockstep_core_is_rejected_and_auto_avoids_it` cover replay and the refusals.
- `test_energy_range_replay_rechecks_statistics_before_accepting_stop` pins the energy-spread replay: if the realized-range tables change the contributions enough to fail a guard, the count grows and statistics are recomputed before accepting the result. `test_environment_pinned_lockstep_is_rejected` covers the process-wide core pin.
- `test_per_electron_brem_band_integrals_average_to_the_band_integral` and `test_batch_means_reconstruct_the_band_of_the_final_spectra` cover the brem band integral and batch means.
- `test_band_weights_integrate_linear_density_at_off_grid_edges` pins partial trapezoid intervals against an analytic linear-density integral, including a band without interior nodes.

## Known limits

- **Wald undercoverage on skewed populations.** The light case's interval $\pm1.96\,r_N$ covered 17 of 20 seeds. An 80-seed study at $\varepsilon=5\,\%$ found 86–89 % for every $n_{\min}$, including $n_{\min}=n_{\max}$ (fixed $N$). This is a property of the sample-SD interval for a skewness-8 population at a few hundred electrons, not of the stop. Treat $r_N$ as a scale, not a calibrated 95 % interval, for such observables.
- **Small $n_{\min}$ on skewed populations.** The same study found a negative stopping bias of $0.3$–$0.4\varepsilon$ at $n_{\min}\le200$, gone by $n_{\min}\approx400$. When the sample mean and SD are low together, the rule stops early. Set $n_{\min}$ near the count the target implies.
- **Unsampled tails.** No guard sees a class that has not appeared. Only $n_{\min}$ and the ESS floor raise the chance of drawing it. The 5 MeV h-BN detector-cone case (#201) is the motivating instance; its remote measurement is outstanding in #361.

Human sign-off pending (#277).
