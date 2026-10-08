# Statistical methods

This page describes PyRITE's estimators, uncertainty estimates, and comparisons between sampled runs. The [validation methodology](../validation/methodology.md) defines the evidence required for validation.

## What is being estimated

Every observable is a **per-incident-electron expectation**. The estimator is the sample mean over the run's electrons, and the denominator is always the full incident count $N_e$ — including electrons that missed the target, which contribute a zero path and no rows.

That includes exit fractions (transmitted, backscattered, side-exited, stopped), path-length and energy moments, and spectra per unit energy and solid angle. Details of the normalization conventions live in [transport outputs](../physics/beam-transport/transport-outputs.md#tallies-and-normalization) and [spectral observables](../physics/radiation-physics/spectral-observables.md#units-and-normalization).

Electrons carry no statistical weights. The implementation uses no variance reduction, splitting, or Russian roulette: each electron contributes once to the sample mean. For ordinary sample means, variance falls as $1/N$.

**No absolute flux.** A spectrum is per incident electron until a caller supplies a multiplier. No photons-per-second normalization is applied anywhere.

**No physical electron count.** Macro-particles are not conflated with the electrons in a real bunch. This matters statistically as well as physically: coherent emission scales with the *physical* count, so a coherent observable computed from a few hundred macro-particles is estimating a shape, not a magnitude.

## Error bars

Use binomial errors for fractions and sample standard errors for means.

**Fractions are binomial.** For an exit fraction $p = k/N_e$ with $k$ electrons counted out of $N_e$ incident,

```{math}
:label: eq-statistics-binomial-se

\sigma_p = \sqrt{\frac{p\,(1-p)}{N_e}} .
```

This is the right estimator because each electron either exits through a given face or does not — a Bernoulli trial — and the trials are independent by construction, since each electron's transport reads its own random stream and nothing else.

**Means are sample standard errors.** For a per-electron quantity $x_i$ such as path length, retained energy, or exit clock,

```{math}
:label: eq-statistics-sample-se

\bar{x} = \frac{1}{N_e}\sum_{i=1}^{N_e} x_i,
\qquad
\sigma_{\bar{x}} = \frac{s}{\sqrt{N_e}},
\qquad
s^2 = \frac{1}{N_e - 1}\sum_{i=1}^{N_e}(x_i - \bar{x})^2 .
```

The Bessel-corrected variance is used throughout; at production electron counts the correction is negligible, but it is applied at the small seed counts used for replication studies, where it is not.

**Independent errors combine in quadrature.** Comparing two unpaired quantities uses $\sigma = \sqrt{\sigma_1^2 + \sigma_2^2}$, and the comparison is reported as a shift in units of that combined error.

**Straggled rows are not independent observations.** Energy-loss increments are independent only conditional on their row states and counter addresses. Rows from one electron share the evolving energy and geometry, so treating segment losses as an i.i.d. sample understates uncertainty. Form per-electron observables first, or replicate whole runs by seed. The Urban sampler adds physical variance to path length, terminal channel, clock and spectra; it does not change the denominator, which remains the full incident count.

### Propagating through a spectral reduction

A spectrum bin is a sum over segments, and a coherent bin is a sum of squares of a complex field sum. The incoherent case propagates straightforwardly: bins are sums of non-negative per-segment contributions, so the per-bin relative error falls as $1/\sqrt{N}$ in the *effective* count of segments contributing to that bin — which is why sparsely populated bins at the edges of a window are noisier than the peak even at fixed $N_e$.

The coherent case does not propagate straightforwardly at all, and is treated below.

### Seed replication

For a small number of independent seeds, the mean and its error over seed replicates are used directly:

```{math}
:label: eq-statistics-seed-replication

\sigma_{\bar{x}} = \frac{s(\{\bar{x}_r\})}{\sqrt{R}}
```

over $R$ replicate realizations. Seed-to-seed spread estimates sampling uncertainty without a separate variance model for each observable. This is useful for peak heights, fitted widths, and other quantities that are not simple per-electron averages.

## Comparing two runs

Different transport cores, refinement levels, and step-control rules can produce different realizations. Compare those runs statistically, using sampling errors rather than a numerical tolerance.

### Paired-seed shift

When two arms share a seed, they are **not independent**, and treating them as independent throws away the sensitivity that pairing provides. Refinement decorrelates a trajectory only after enough energy has been lost for the substep grid to move a collision point, so the early history and the incident sampling are common to both arms.

Differencing seed by seed cancels that common variance:

```{math}
:label: eq-statistics-paired-shift

\frac{\Delta}{\sigma} = \frac{\overline{d}}{s(d)/\sqrt{R}},
\qquad d_r = x_r - x_r^{\text{ref}} .
```

This is strictly more sensitive to a systematic bias than combining the two arms' standard errors, and it degrades gracefully to the unpaired test when the arms are fully decorrelated.

Where the unpaired form is used anyway, the denominator treats the two arms as independent, which is **conservative** given a shared seed — a reported shift is if anything understated, never inflated. Stating which direction an assumption errs in is part of reporting the comparison.

Straggling on/off comparisons use this paired design deliberately. The two arms share incident sampling and the pre-existing free-path/scattering draws; after the first sampled loss changes the energy, later rows may decorrelate. Pairing therefore removes the common early-history variance without pretending the whole trajectories are identical. The committed observable check reports the mean and Bessel-corrected SEM of the eight seed-wise on-minus-off differences, not an error inferred from individual segments.

### Reading a shift table

A shift table is read for **sign pattern**, not for individual entries. At a $3\sigma$ flag threshold and a two-sided test, roughly 0.3 % of comparisons trip by chance, so a single starred cell in a large table is expected. What is not expected is a column of same-signed shifts: that is a systematic effect, and it is the thing these tables exist to expose.

The tables print every shift so sign patterns remain visible.

### Welch comparison

Where two arms have genuinely different variances and no shared seed, a Welch test is the appropriate form, with the caveat above about the conservatism of an independence assumption. In practice the paired form covers most of the comparisons that matter, because arms are deliberately run at matched seeds precisely to enable it.

(coherent-versus-incoherent-statistics)=

## Coherent versus incoherent statistics

The incoherent sum is well-conditioned. The global coherent sum is not, and this is a property of the **estimator**, not of any implementation choice.

The incoherent spectrum adds $|A_j|^2$ over segments. Every term is non-negative, so the sum is of the same order as its terms, and ordinary $1/\sqrt{N}$ statistics apply.

The global coherent spectrum squares a *sum of complex amplitudes*:

```{math}
:label: eq-statistics-coherent-conditioning

I \;=\; \Bigl|\sum_j A_j\Bigr|^2 ,
```

where cancellation can leave a small residual field. Relative errors in that residual can be much larger than relative errors in the individual amplitudes.

Energy-loss straggling acts differently on the two reductions. Incoherent intensities have no cross-row phase, so straggling affects them through the realized energy and path distribution. Coherent fields retain the random clock phase. For independent Gaussian phase noise, ensemble averaging would multiply the coherent cross term by $\exp(-\sigma_\phi^2/2)$, but the implemented Urban loss is few-collision and strongly non-Gaussian, and row phases share a trajectory. The Gaussian factor is therefore an interpretation aid, not the estimator. Measure the complex reduction over independent seed replicas.

Coherent results need additional care:

* **The error does not shrink under step refinement.** Refining the numerical step changes each term's phase slightly; when the sum is a residual of cancellation, a small per-term change produces a comparable change in the residual. Convergence in the terms does not imply convergence in the estimator.
* **It is ill-conditioned to *any* per-row change**, including ones that are exactly neutral for the incoherent sum — a reordering, a regrouping, a change of which rows are emitted.
* **Its realization-to-realization scatter is large and does not fall with $N_e$** when a transverse form factor is missing, because what is being sampled is one speckle realization rather than an ensemble average. See [random number streams](random-streams.md#where-inertness-stops).

Report uncertainty for the coherent observable directly. Do not assume it follows the incoherent observable's error scaling.

The structural mitigation — accumulating fields and squaring once, never reducing intensity per block — is described in [execution and acceleration](execution-and-acceleration.md#the-coherent-streaming-kernel).

The phase-sensitive straggling checks and their scope are recorded under `energy-loss-straggling` in the [physics validation ledger](../validation/physics-validation-ledger.md). A free-clock phase shift alone does not replace evaluation of the full coherent kernel.

(adaptive-electron-counts)=

## Adaptive electron counts

An adaptive case chooses its own electron count from a target relative standard error instead of a fixed `Ne`. This is internal API (`montecarlo/runner/adaptive.py::run_case_adaptive`, #361); no profile or CLI key selects it yet. The implementation is ledgered as `adaptive-sample-size-stopping`.

**What is watched.** Transport runs in equal electron blocks $[kB, (k+1)B)$ (`montecarlo/runner/block_transport.py`). After each block, the runner folds a per-electron scalar $m_i$ into running moments for each watched observable:

* `"line"` — the electron's line mass $m_i = \sum_{\ell \in i} w_\ell \pi / a_\ell$ over the lines the kernel keeps for a fixed monitor band. This is the quantity of the #201 line-yield audit. Its mean is the band's line yield per electron.
* `"brem"` — the electron's bremsstrahlung integral over the same band, $m_i = \sum_E q_E\, \mathrm{d}N_i/\mathrm{d}E\,\mathrm{d}\Omega$, with trapezoid weights $q_E$ on the fixed bremsstrahlung grid. Band edges between nodes use linear interpolation; the integral includes the partial intervals at both edges and assumes no density outside the grid.

Both are grid-independent. The monitor band is the case's line-policy bandwidth, else the end points of its line grid. Neither quantity needs the final line grid, which an automatic policy resolves from the transport distribution. The stop therefore never compares spectra on different grids. After the stop, the line grid is resolved once from every realized electron and the spectrum is reduced once, exactly as in a fixed-$N$ run.

**Estimator.** The mean and its relative standard error after $n$ electrons are those of {eq}`eq-statistics-sample-se`:

```{math}
:label: eq-statistics-adaptive-rse

\bar{m}_n = \frac{1}{n}\sum_{i=1}^{n} m_i ,
\qquad
r_n = \frac{s_n}{\sqrt{n}\,\lvert\bar{m}_n\rvert} .
```

Each block is reduced on its own device to a count, sum, centred second moment, sum of squares and maximum. Blocks merge on the host with the Chan–Golub–LeVeque pairwise update, which is algebraically a one-pass Welford over the concatenated values.

**Stopping rule.** Let $n$ run over block ends $\{B, 2B, \dots\}$, with $n_{\min}$, $n_{\max}$ and the pilot count multiples of $B$. Stop at the first $n \ge n_{\min}$ at which, for every watched observable, the error target and all three guards hold:

```{math}
:label: eq-statistics-adaptive-stop

r_n \le \varepsilon,
\qquad
\frac{\max_{i \le n} m_i}{\sum_{i \le n} m_i} \le c,
\qquad
\mathrm{ESS}_n = \frac{\left(\sum_{i \le n} m_i\right)^2}{\sum_{i \le n} m_i^2} \ge E_{\min},
\qquad
\max_{j=1..k} \frac{\lvert \bar{m}_{n-jB} - \bar{m}_n\rvert}{\lvert\bar{m}_n\rvert} \le f\varepsilon .
```

Defaults are $c = 0.05$, $E_{\min} = 100$, $k = 3$ and $f = 0.5$. They are internal settings until the remote measurement in #361 sets profile values. At $n_{\max}$ the run stops regardless. It completes, records `statistics_limited`, and warns `LineYieldStatisticsWarning`.

The optional pilot projects $N = n_{\text{pilot}} \max_o (r_{o}/\varepsilon)^2$, rounded up to a block and clipped to $[n_{\text{pilot}}, n_{\max}]$. No check runs until $N$ is reached. The rule then resumes block by block.

**The ESS guard and the error target are one inequality.** For non-negative $m_i$, the two bound the same pair of sums:

```{math}
:label: eq-statistics-adaptive-ess

r_n^2 = \frac{n}{n-1}\left(\frac{1}{\mathrm{ESS}_n} - \frac{1}{n}\right).
```

So $r_n \le \varepsilon$ already implies $\mathrm{ESS}_n \gtrsim 1/\varepsilon^2$ at large $n$. An ESS floor binds only above $1/\varepsilon^2$. What it adds is a minimum *effective* sample that does not depend on the target: enough electrons for a rare heavy class to have been drawn at all. With a single dominant electron, $r_n$ and the share are of the same order, as is the jump that electron causes in the running mean. The share cap $c < \varepsilon$ and the stability tolerance $f\varepsilon$ therefore bind in that regime.

**What no guard can see.** Every quantity in {eq}`eq-statistics-adaptive-stop` describes only the electrons drawn so far. If a heavy class with rate $p$ has not appeared in $n$ electrons (probability $e^{-pn}$), the run reports a small error around a biased mean. Only a large enough $n_{\min}$ or ESS floor guards against that. The rate-$p$ class is drawn with probability $1 - e^{-pE_{\min}}$ by the time the light electrons alone reach the floor. Variance reduction for such classes is #203.

**Deterministic replay.** Counter-addressed streams ([random number streams](random-streams.md)) make the same seed and settings reproduce the realized count and output on a given backend. The realized case is the fixed-$N$ case at that count, bit for bit. Table energy ranges and the bunch centroid belong to the realized population. When the energy-spread range of the first $n$ electrons differs from that of $n_{\max}$, transport replays $[0, n)$ over its own range and recomputes the statistics. If that replay fails the stopping rule, the count grows by one block and is replayed again until the guards pass or $n_{\max}$ is reached. The reported statistics describe the returned transport.

**Coherent route excluded.** The coherent line sum {eq}`eq-statistics-coherent-conditioning` couples electrons, so a per-electron error does not bound it, and its scatter need not fall with $N_e$. Adaptive runs refuse `coherent_emission`, as they refuse the lockstep core and GDF beams. Coherent cases keep a fixed count.

**Per-bin errors.** Optionally, after the final reduction, each equal block of the realized run is reduced as one batch on the final grids. The per-bin batch-means standard error $s(\{\bar{S}_b\})/\sqrt{K}$ over $K$ batches is reported for `spec` and `brem_wide` in a requested band. It never drives the stop, because the line grid is unknown while transport runs.

**Measured behaviour.** These are the regression tests in `tests/montecarlo/test_adaptive_stopping.py`:

* A Gaussian population $\mathcal{N}(1, 0.5^2)$ at $\varepsilon = 5\,\%$ over 1000 seeds stops at about 100 electrons. The interval $\pm 1.96\, r_N$ covers the mean in 93.4 % of seeds with no guards and 94.3 % with the defaults, at 95 % nominal. The stopping bias is $+0.11\varepsilon$ without guards and $+0.03\varepsilon$ with them. Chow and Robbins show that this coverage tends to nominal as $\varepsilon \to 0$. The small positive bias is the sequential effect of stopping when $\bar{m}$ is high, because the denominator of $r_n$ is then large.
* The light real case is hopg at 30 keV, 1 µm thick, with the line mass watched. Its per-electron distribution has skewness about 8 and CV about 1. At $\varepsilon = 10\,\%$ with $n_{\min} = 200$, 20 seeds stop at 200–440 electrons. All 20 lie within $1.96\varepsilon$ of the pooled 16 000-electron mean. The paired stopping bias is $+0.06\,\% \pm 1.5\,\%$, below every reported error.
* The sample-based interval $\pm 1.96\, r_N$ covers only 17 of the 20 light-case seeds. In a larger 80-seed study it covered 86–89 % at every $n_{\min}$, including $n_{\min} = n_{\max}$, which is a fixed-$N$ run. That undercoverage belongs to the skewed per-electron distribution at a few hundred electrons, not to the stop. In that study, $n_{\min} \lesssim 200$ also showed a negative stopping bias of $0.3$–$0.4\varepsilon$ at $\varepsilon = 5\,\%$, which vanished by $n_{\min} \approx 400$. A skewed population whose sample mean and variance are low together stops early on that pair. Choose $n_{\min}$ near the count the target implies.
* On a constructed heavy-tailed population, the bare rule stops at 200 electrons, 44 % low, with a reported error of 7 %. Exp(1) electrons plus a +400 contribution every 500th electron make that population. With an ESS floor of 1000, the run meets the heavy electrons, the share cap holds, and it ends `statistics_limited` at $n_{\max}$. Over 40 random heavy-tailed seeds at rate $2\times10^{-3}$, false stops beyond 3 reported errors fall from at least 20 to at most 1. The remaining case is one whose tail is still unsampled at the floor.

## Convergence protocol

Check both sampling uncertainty and numerical step refinement.

* **Sample size and seed.** Compare aggregate observables with their standard errors. Ordinary Monte Carlo error falls as $1/\sqrt{N}$; coherent observables need the separate treatment described above.
* **Numerical step refinement.** Refinement can decorrelate trajectories, so changes in a single seed's flight count do not establish convergence. Compare ensemble means with Monte Carlo errors, using paired seeds to retain the shared early-history information.

Increasing sample size cannot remove step bias. Refining the step cannot compensate for insufficient sampling.

## Null results

A null result bounds an effect at the study's resolution; it does not prove exact invariance. Report:

1. The predicted bias and its sign, where known.
2. The measured shift and uncertainty, including the bound the study supports.
3. The tested core, geometry, parameters, and observables.

For example, collision-statistics bounds do not establish radiation invariance, which is a separate ledgered claim. Configurations outside the tested scope require their own evidence.

## Histogramming and binning

Spectra are accumulated onto an energy grid, and the grid is a choice with statistical consequences.

**Bin width trades resolution against noise.** Narrower bins resolve line structure and admit fewer segments each, so per-bin relative error rises. The line grid and the bremsstrahlung grid are separate for this reason: a continuum tolerates coarse binning that would destroy a line.

**Line observables depend on bin width.** A peak height read off a histogram is not a bin-width-independent quantity when the bin is comparable to or wider than the unbroadened source-model linewidth. Comparisons of peak height are only meaningful at matched grids.

**Peak-pick is unstable near a bin edge.** When a resonance falls close to a boundary, a small parameter change can move the argmax by one bin and the reported peak position discontinuously. Two mitigations apply: compare integrated intensity over a window rather than a single bin where the physics permits, and hold the grid fixed across the arms of a comparison so that any edge effect is common-mode.

The grid construction and its schema are documented separately — the derivation workflow under [material energy grids](../guides/sweep-profiles.md) and the schema decisions in [ADR 0005](../adr/0005-energy-grid-schema-decisions.md). Chunking the reduction over segments never affects binning; it partitions a sum and nothing more.

## Validation

The statistical machinery here is exercised by the ledger rows `energy-step-convergence` (binomial and sample standard errors, paired-seed shift, seed replication isolating a real one-signed bias, and the coherent ill-conditioning result), `substep-radiation-invariance` (convergence of a grouped reduction, reported as a bound), and `gpu-transport-core` (multi-seed aggregate comparison across a stream change), plus `energy-loss-straggling` (paired-seed response of terminal fractions, range, clock and spectra), and `adaptive-sample-size-stopping` (the sequential stopping rule, its guards, and its coverage and stopping bias).

The reproducible drivers are `checks/energy_step_convergence_matrix.py` and `checks/collision_statistics_refinement.py`, which print every shift with its error, as described in [reading a shift table](#reading-a-shift-table). The straggling counterpart is `checks/energy_loss_straggling_observables.py`.
