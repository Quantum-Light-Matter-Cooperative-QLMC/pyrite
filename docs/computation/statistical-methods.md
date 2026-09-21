# Statistical methods

This page describes PyRITE's estimators, uncertainty estimates, and comparisons
between sampled runs. The [validation methodology](../validation/methodology.md)
defines the evidence required for validation.

## What is being estimated

Every observable is a **per-incident-electron expectation**. The estimator is
the sample mean over the run's electrons, and the denominator is always the full
incident count $N_e$ — including electrons that missed the target, which
contribute a zero path and no rows.

That includes exit fractions (transmitted, backscattered, side-exited, stopped),
path-length and energy moments, and spectra per unit energy and solid angle.
Details of the normalization conventions live in
[transport outputs](../physics/beam-transport/transport-outputs.md#tallies-and-normalization)
and [spectral observables](../physics/radiation-physics/spectral-observables.md#units-and-normalization).

Electrons carry no statistical weights. The implementation uses no variance
reduction, splitting, or Russian roulette: each electron contributes once to
the sample mean. For ordinary sample means, variance falls as $1/N$.

**No absolute flux.** A spectrum is per incident electron until a caller
supplies a multiplier. No photons-per-second normalization is applied anywhere.

**No physical electron count.** Macro-particles are not conflated with the
electrons in a real bunch. This matters statistically as well as physically:
coherent emission scales with the *physical* count, so a coherent observable
computed from a few hundred macro-particles is estimating a shape, not a
magnitude.

## Error bars

Use binomial errors for fractions and sample standard errors for means.

**Fractions are binomial.** For an exit fraction $p = k/N_e$ with $k$ electrons
counted out of $N_e$ incident,

```{math}
:label: eq-statistics-binomial-se

\sigma_p = \sqrt{\frac{p\,(1-p)}{N_e}} .
```

This is the right estimator because each electron either exits through a given
face or does not — a Bernoulli trial — and the trials are independent by
construction, since each electron's transport reads its own random stream and
nothing else.

**Means are sample standard errors.** For a per-electron quantity $x_i$ such as
path length, retained energy, or exit clock,

```{math}
:label: eq-statistics-sample-se

\bar{x} = \frac{1}{N_e}\sum_{i=1}^{N_e} x_i,
\qquad
\sigma_{\bar{x}} = \frac{s}{\sqrt{N_e}},
\qquad
s^2 = \frac{1}{N_e - 1}\sum_{i=1}^{N_e}(x_i - \bar{x})^2 .
```

The Bessel-corrected variance is used throughout; at production electron counts
the correction is negligible, but it is applied at the small seed counts used
for replication studies, where it is not.

**Independent errors combine in quadrature.** Comparing two unpaired quantities
uses $\sigma = \sqrt{\sigma_1^2 + \sigma_2^2}$, and the comparison is reported
as a shift in units of that combined error.

**Straggled rows are not independent observations.** Energy-loss increments are
independent only conditional on their row states and counter addresses. Rows
from one electron share the evolving energy and geometry, so treating segment
losses as an i.i.d. sample understates uncertainty. Form per-electron
observables first, or replicate whole runs by seed. The Urban sampler adds
physical variance to path length, terminal channel, clock and spectra; it does
not change the denominator, which remains the full incident count.

### Propagating through a spectral reduction

A spectrum bin is a sum over segments, and a coherent bin is a sum of squares of
a complex field sum. The incoherent case propagates straightforwardly: bins are
sums of non-negative per-segment contributions, so the per-bin relative error
falls as $1/\sqrt{N}$ in the *effective* count of segments contributing to that
bin — which is why sparsely populated bins at the edges of a window are noisier
than the peak even at fixed $N_e$.

The coherent case does not propagate straightforwardly at all, and is treated
below.

### Seed replication

For a small number of independent seeds, the mean and its error over seed
replicates are used directly:

```{math}
:label: eq-statistics-seed-replication

\sigma_{\bar{x}} = \frac{s(\{\bar{x}_r\})}{\sqrt{R}}
```

over $R$ replicate realizations. Seed-to-seed spread estimates sampling
uncertainty without a separate variance model for each observable. This is
useful for peak heights, fitted widths, and other quantities that are not
simple per-electron averages.

## Comparing two runs

Different transport cores, refinement levels, and step-control rules can
produce different realizations. Compare those runs statistically, using
sampling errors rather than a numerical tolerance.

### Paired-seed shift

When two arms share a seed, they are **not independent**, and treating them as
independent throws away the sensitivity that pairing provides. Refinement
decorrelates a trajectory only after enough energy has been lost for the substep
grid to move a collision point, so the early history and the incident sampling
are common to both arms.

Differencing seed by seed cancels that common variance:

```{math}
:label: eq-statistics-paired-shift

\frac{\Delta}{\sigma} = \frac{\overline{d}}{s(d)/\sqrt{R}},
\qquad d_r = x_r - x_r^{\text{ref}} .
```

This is strictly more sensitive to a systematic bias than combining the two
arms' standard errors, and it degrades gracefully to the unpaired test when the
arms are fully decorrelated.

Where the unpaired form is used anyway, the denominator treats the two arms as
independent, which is **conservative** given a shared seed — a reported shift is
if anything understated, never inflated. Stating which direction an assumption
errs in is part of reporting the comparison.

Straggling on/off comparisons use this paired design deliberately. The two arms
share incident sampling and the pre-existing free-path/scattering draws; after
the first sampled loss changes the energy, later rows may decorrelate. Pairing
therefore removes the common early-history variance without pretending the
whole trajectories are identical. The committed observable check reports the
mean and Bessel-corrected SEM of the eight seed-wise on-minus-off differences,
not an error inferred from individual segments.

### Reading a shift table

A shift table is read for **sign pattern**, not for individual entries. At a
$3\sigma$ flag threshold and a two-sided test, roughly 0.3 % of comparisons trip
by chance, so a single starred cell in a large table is expected. What is not
expected is a column of same-signed shifts: that is a systematic effect, and it
is the thing these tables exist to expose.

The tables print every shift so sign patterns remain visible.

### Welch comparison

Where two arms have genuinely different variances and no shared seed, a Welch
test is the appropriate form, with the caveat above about the conservatism of an
independence assumption. In practice the paired form covers most of the
comparisons that matter, because arms are deliberately run at matched seeds
precisely to enable it.

(coherent-versus-incoherent-statistics)=

## Coherent versus incoherent statistics

The incoherent sum is well-conditioned. The global coherent sum is not, and this
is a property of the **estimator**, not of any implementation choice.

The incoherent spectrum adds $|A_j|^2$ over segments. Every term is
non-negative, so the sum is of the same order as its terms, and ordinary
$1/\sqrt{N}$ statistics apply.

The global coherent spectrum squares a *sum of complex amplitudes*:

```{math}
:label: eq-statistics-coherent-conditioning

I \;=\; \Bigl|\sum_j A_j\Bigr|^2 ,
```

where cancellation can leave a small residual field. Relative errors in that
residual can be much larger than relative errors in the individual amplitudes.

Energy-loss straggling acts differently on the two reductions. Incoherent
intensities have no cross-row phase, so straggling affects them through the
realized energy and path distribution. Coherent fields retain the random clock
phase. For independent Gaussian phase noise, ensemble averaging would multiply
the coherent cross term by $\exp(-\sigma_\phi^2/2)$, but the implemented Urban
loss is few-collision and strongly non-Gaussian, and row phases share a
trajectory. The Gaussian factor is therefore an interpretation aid, not the
estimator. Measure the complex reduction over independent seed replicas.

Coherent results need additional care:

* **The error does not shrink under step refinement.** Refining the numerical
  step changes each term's phase slightly; when the sum is a residual of
  cancellation, a small per-term change produces a comparable change in the
  residual. Convergence in the terms does not imply convergence in the
  estimator.
* **It is ill-conditioned to *any* per-row change**, including ones that are
  exactly neutral for the incoherent sum — a reordering, a regrouping, a change
  of which rows are emitted.
* **Its realization-to-realization scatter is large and does not fall with
  $N_e$** when a transverse form factor is missing, because what is being
  sampled is one speckle realization rather than an ensemble average. See
  [random number streams](random-streams.md#where-inertness-stops).

Report uncertainty for the coherent observable directly. Do not assume it
follows the incoherent observable's error scaling.

The structural mitigation — accumulating fields and squaring once, never
reducing intensity per block — is described in
[execution and acceleration](execution-and-acceleration.md#the-coherent-streaming-kernel).

The phase-sensitive straggling checks and their scope are recorded under
`energy-loss-straggling` in the
[physics validation ledger](../validation/physics-validation-ledger.md).
A free-clock phase shift alone does not replace evaluation of the full
coherent kernel.

## Convergence protocol

Check both sampling uncertainty and numerical step refinement.

* **Sample size and seed.** Compare aggregate observables with their standard
  errors. Ordinary Monte Carlo error falls as $1/\sqrt{N}$; coherent
  observables need the separate treatment described above.
* **Numerical step refinement.** Refinement can decorrelate trajectories, so
  changes in a single seed's flight count do not establish convergence.
  Compare ensemble means with Monte Carlo errors, using paired seeds to retain
  the shared early-history information.

Increasing sample size cannot remove step bias. Refining the step cannot
compensate for insufficient sampling.

## Null results

A null result bounds an effect at the study's resolution; it does not prove
exact invariance. Report:

1. The predicted bias and its sign, where known.
2. The measured shift and uncertainty, including the bound the study supports.
3. The tested core, geometry, parameters, and observables.

For example, collision-statistics bounds do not establish radiation
invariance, which is a separate ledgered claim. Configurations outside the
tested scope require their own evidence.

## Histogramming and binning

Spectra are accumulated onto an energy grid, and the grid is a choice with
statistical consequences.

**Bin width trades resolution against noise.** Narrower bins resolve line
structure and admit fewer segments each, so per-bin relative error rises. The
line grid and the bremsstrahlung grid are separate for this reason: a continuum
tolerates coarse binning that would destroy a line.

**Line observables depend on bin width.** A peak height read off a histogram is
not a bin-width-independent quantity when the bin is comparable to or wider than
the unbroadened source-model linewidth. Comparisons of peak height are only
meaningful at matched grids.

**Peak-pick is unstable near a bin edge.** When a resonance falls close to a
boundary, a small parameter change can move the argmax by one bin and the
reported peak position discontinuously. Two mitigations apply: compare
integrated intensity over a window rather than a single bin where the physics
permits, and hold the grid fixed across the arms of a comparison so that any
edge effect is common-mode.

The grid construction and its schema are documented separately — the derivation
workflow under [material energy grids](../guides/sweep-profiles.md) and the
schema decisions in [ADR 0005](../adr/0005-energy-grid-schema-decisions.md).
Chunking the reduction over segments never affects binning; it partitions a sum
and nothing more.

## Validation

The statistical machinery here is exercised by the ledger rows
`energy-step-convergence` (binomial and sample standard errors, paired-seed
shift, seed replication isolating a real one-signed bias, and the coherent
ill-conditioning result), `substep-radiation-invariance` (convergence of a
grouped reduction, reported as a bound), and `gpu-transport-core` (multi-seed
aggregate comparison across a stream change), plus `energy-loss-straggling`
(paired-seed response of terminal fractions, range, clock and spectra).

The reproducible drivers are `checks/energy_step_convergence_matrix.py` and
`checks/collision_statistics_refinement.py`, which print every shift with its
error, as described in
[reading a shift table](#reading-a-shift-table).
The straggling counterpart is
`checks/energy_loss_straggling_observables.py`.
