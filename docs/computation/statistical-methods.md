# Statistical methods

The estimators this package forms, how their uncertainty is quantified, and how
two runs are compared when they are — as they usually are — different
realizations rather than different answers.

This page supplies the machinery that [validation
methodology](../validation/methodology.md) assumes. The methodology decides what
counts as evidence; this page decides what the numbers in that evidence mean.

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

Three absences do the statistical work here.

**No statistical weights.** Every electron counts once. There is no variance
reduction, no splitting, no Russian roulette. What this buys is that the sample
mean *is* the estimator, its variance is the ordinary sample variance, and there
is no weight bookkeeping that can silently bias a tally. The cost is that
variance falls only as $1/N$; the package buys its precision with electrons
rather than with cleverness, which is a deliberate trade in favour of
verifiability.

**No absolute flux.** A spectrum is per incident electron until a caller
supplies a multiplier. No photons-per-second normalization is applied anywhere.

**No physical electron count.** Macro-particles are not conflated with the
electrons in a real bunch. This matters statistically as well as physically:
coherent emission scales with the *physical* count, so a coherent observable
computed from a few hundred macro-particles is estimating a shape, not a
magnitude.

## Error bars

Two estimators cover almost everything, and choosing between them is a question
about the observable, not about convenience.

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

over $R$ replicate realizations. The virtue of this form is that it assumes
**no per-observable variance model**: the seed-to-seed spread already carries
every source of sampling error, including ones a binomial or sample-mean model
would miss. It is the tool of choice when the observable is a complicated
functional of the run — a peak height, a fitted width — rather than a simple
per-electron average.

## Comparing two runs

Most interesting comparisons are between configurations that produce **different
realizations**: two transport cores, two refinement rungs, two step-control
rules. Such a comparison is statistical by construction, and there is no
tolerance that makes it numerical.

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

This is why the tables print *every* shift rather than only the flagged ones.

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

and at the energies of interest that sum is a **small residual of a large
cancellation**. Measured at 100 keV into a thick target, the residual is around
47 % even with accurate phases.

Energy-loss straggling acts differently on the two reductions. Incoherent
intensities have no cross-row phase, so straggling changes them only through the
realized energy/path distribution; the paired HOPG check resolves no change in
the integrated bremsstrahlung yield and a $0.334\% \pm 0.053\%$ normalized
spectral total-variation distance. Coherent fields retain the random clock
phase. For independent Gaussian phase noise, ensemble averaging would multiply
the coherent cross term by $\exp(-\sigma_\phi^2/2)$, but the implemented Urban
loss is few-collision and strongly non-Gaussian, and row phases share a
trajectory. The Gaussian factor is therefore an interpretation aid, not the
estimator. Measure the complex reduction over independent seed replicas.

Three consequences follow, and each one breaks an intuition that holds for the
incoherent case.

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

None of this makes the coherent spectrum wrong. It makes it an estimator whose
error must be **stated** rather than assumed to follow the incoherent case's
scaling — and it is why the coherent arm of a core comparison shows median
differences of several percent and maxima of tens of percent where the line
spectrum shows a fraction of a percent, all of it tracking $1/\sqrt{N}$ for its
own population.

The structural mitigation — accumulating fields and squaring once, never
reducing intensity per block — is described in
[execution and acceleration](execution-and-acceleration.md#the-coherent-streaming-kernel).

In the committed 25 keV HOPG (002), zero-bunch-offset pure-geometry check,
straggling lowers integrated coherent-line yield by
$12.52\% \pm 2.54\%$ and peak height by $19.50\% \pm 2.12\%$. This isolates a
phase-sensitive limit; it is not an angle- or bunch-averaged experimental
prediction. The free-clock phase shift reported by the same check is likewise
not a substitute for evaluating the full coherent kernel.

## Convergence protocol

Two axes must be checked, and they behave differently. Nothing in a run's output
signals convergence on either.

**Sample size and seed.** Ordinary Monte Carlo error, falling as $1/\sqrt{N}$.
Aggregate observables are compared with their standard errors.

**Numerical step refinement.** Refining the step **decorrelates** trajectories.
This is the axis that catches people out: per-realization quantities such as
flight counts move for reasons that carry no information at all. A single seed's
flight count wandering down a refinement ladder — 2568, 2882, 2599, 2553, 2605
in one recorded case — measures nothing whatsoever.

The rule that follows: **convergence on the refinement axis must be read from
ensemble means with Monte Carlo errors, never realization by realization.** A
paired-seed comparison is the right instrument, because the arms share their
early history even as they decorrelate.

The two axes are independent and both are required. Refining the step at fixed,
inadequate $N_e$ produces a converged estimate of nothing in particular;
increasing $N_e$ at a fixed, inadequate step produces a precise estimate of a
biased quantity.

## Null results

Most step-control claims in the transport ledger are **null results**, and they
are reported as bounds at a stated resolution rather than as proofs.

The template is explicit about three things:

1. **The predicted bias and its sign.** Freezing the collision rate at the
   flight-start energy understates a rising hazard, so unrefined flights are
   slightly too long. The effect is one-signed and real.
2. **The measured bound.** In the recorded case it sits below about 1 % of the
   mean flight length, with its largest appearance at $2.25\sigma$ on mean
   flight length — in the case slice independently identified as having the
   largest per-flight fractional loss, and with the predicted sign.
3. **The scope.** Which core, which geometry, which observables. A null result
   on collision statistics says nothing about emitted radiation, which is a
   separately ledgered claim.

An invariance measured to be below a threshold is not an invariance proved. The
distinction matters because the predicted bias is *real* — it is merely small
here — so a configuration outside the measured scope has no inherited guarantee.

A useful corollary: when a measured effect has the predicted sign and a
magnitude consistent with the predicted mechanism, that is stronger evidence
than a null at the same significance. Agreement in sign and rough magnitude is
harder to produce by accident than a non-detection.

## Histogramming and binning

Spectra are accumulated onto an energy grid, and the grid is a choice with
statistical consequences.

**Bin width trades resolution against noise.** Narrower bins resolve line
structure and admit fewer segments each, so per-bin relative error rises. The
line grid and the bremsstrahlung grid are separate for this reason: a continuum
tolerates coarse binning that would destroy a line.

**Line observables depend on bin width.** A peak height read off a histogram is
not a bin-width-independent quantity when the bin is comparable to or wider than
the intrinsic linewidth. Comparisons of peak height are only meaningful at
matched grids.

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
error rather than only the flagged ones — for the reason given in
[reading a shift table](#reading-a-shift-table).
The straggling counterpart is
`checks/energy_loss_straggling_observables.py`.
