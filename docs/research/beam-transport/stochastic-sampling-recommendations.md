# Recommendation: phase-space sampling and variance reduction

**Status:** Research recommendation, September 2026. Based on inspection of the implemented samplers and estimators, not an efficiency benchmark. No sampling policy or physical model is changed by this document.

## Recommendation

Improve initial electron phase-space coverage first, preserve existing conditional averages, and introduce explicit statistical weights before any deliberate tail oversampling. Start with incoherent observables. Derive and validate a separate weighted coherent estimator before enabling the same policies for coherent output.

The repository does not establish globally optimal sampling. Some processes use efficient direct samplers; others already avoid random sampling through integration or conditional expectation. The appropriate objective is accuracy of a specified observable at fixed wall time, not the appearance of a sampled distribution or the number of random draws alone.

This concerns particles *within one simulation*. The separate [parameter-space recommendation](../workflows/parameter-space-sampling.md) concerns selection and interpretation of different simulation cases. The [energy-grid recommendation](energy-grid-recommendations.md) concerns a third, independent numerical error source.

## Current sampling inventory

Source paths in this document are relative to `src/pyrite/`.

| Process | Current implementation | Efficiency assessment |
| --- | --- | --- |
| Transverse beam phase space | `montecarlo/transverse.py`: normal draws transformed to correlated position/slope pairs | Reproduces configured Twiss moments statistically; no stratification or tail allocation |
| Longitudinal phase space | `montecarlo/transport/kinematics.py::_sample_bunch_offsets`: Gaussian, uniform, microtrain, or explicit offsets | Direct sampling; no guaranteed tail or microbunch coverage |
| Beam energy spread | `montecarlo/transport/api.py`: independent Gaussian relative energy deviations | No importance sampling; no energy–arrival-time chirp |
| Groove entrance phase | `montecarlo/transport/api.py`: uniform random phase over one period on the applicable path | Candidate for stratification |
| Elastic free paths | `montecarlo/transport/cores.py` and `_jit_kernel.py`: inverse exponential optical-depth sampling | Cheap direct sampler; rare flights still require statistics |
| Collision element | Relative-rate/CDF selection in the transport cores | Standard direct categorical sampling |
| Elastic deflection | `montecarlo/transport/scattering.py`: analytic inverse angular CDF and uniform azimuth | Efficient for the implemented angular law; no large-angle enrichment |
| Energy-loss straggling | `montecarlo/transport/straggling.py`: compound Poisson counts and inverse-CDF ionization losses | Direct physical-law sampling; no importance sampling of large losses |
| Radiation emission and escape | `montecarlo/spectrum/` and `materials/attenuation.py`: expected spectra and attenuation along trajectories | Avoids separately sampling rare emitted photons and their survival |
| Mosaic orientation | `montecarlo/geometry.py::_mosaic_quadrature`: weighted Gauss–Hermite quadrature | Already replaces random orientation sampling with deterministic integration |
| Detector acceptance | `montecarlo/geometry.py::detector_directions`: deterministic rays with solid-angle weights | Avoids randomly waiting for detector hits; angular convergence still required |
| Timepix response construction | `detectors/timepix_response.py::build_response`: condition on absorption, then multiply by absorption probability | Useful conditional sampling already present; position/Fano/threshold/readout draws remain ordinary MC |
| Acquisition noise | Detector Poisson counting and Gaussian electronic noise | Appropriate for synthetic noisy acquisitions; fluctuations are part of the requested output |

These descriptions apply to the implemented models, not to their validity at arbitrary energy. Inverse-CDF sampling of an approximate angular law does not validate that law against physical elastic-scattering data.

Straggling defaults off. When enabled, its Poisson sampler splits large means into bounded-rate chunks and sums their counts; individual ionization transfers use an inverse CDF. This preserves the intended distribution within numerical limits but is not demonstrated to minimize runtime for every mean. An inadmissible Urban parameterization falls back to deterministic mean loss. Mean preservation in that branch must not be described as preservation of fluctuations or tails.

## Beam tails are not reliably covered at small particle counts

The transverse sampler draws two independent normal variables per plane and applies the configured position–slope correlation. The two transverse planes are independent. Arrival time and energy spread use separate RNG streams. Longitudinal samples are recentered to the sampled bunch centroid; explicit time offsets are also recentered. These choices specify the current population and reproducibility conventions, not a variance-reduction policy.

For one standard Gaussian coordinate, the probability outside ±4 standard deviations is approximately 0.00006334. Independent sampling with 1,000 particles has about a 94% probability of seeing none. Approximately 47,300 particles are needed for a 95% chance of seeing at least one. These are elementary Gaussian-tail and binomial calculations, not transport results. Seeing one tail particle is not enough to estimate its contribution accurately. The example is a single coordinate, not a four-dimensional transverse-radius probability.

Tail relevance depends on the response: a rare position may determine aperture losses, and a rare direction may dominate a narrow acceptance or background band. Conversely, allocating many particles to an irrelevant tail wastes work. No sampling improvement can recover a real beam halo, coupling, or chirp that is absent from the configured physical distribution.

## Preserve and extend conditional averaging

The coherent path already does more than square one unprocessed random bunch realization:

- Infinite slabs use an empirical longitudinal/transverse characteristic function in `montecarlo/spectrum/lines/_per_hkl.py::_row_decoherence_factor`. That factor still comes from the sampled population and retains finite-sample error.
- Finite footprints use analytic Gaussian longitudinal averaging in `montecarlo/spectrum/lines/_setup.py`, conditional on the sampled transverse trajectories. Transverse phase, hit/miss behavior, and attenuation remain coupled.

The current assumptions and validation status belong to [`coherent-inter-electron-decoherence`](../../validation/ledger-core-coherent-physics.md#coherent-inter-electron-decoherence) and [`finite-footprint-longitudinal-decoherence`](../../validation/ledger-core-coherent-physics.md#finite-footprint-longitudinal-decoherence). This recommendation does not promote their status or independently validate them.

Prefer further analytic or conditional integration where the relevant variables can actually be separated. Do not replace the joint distribution with independent marginals when Twiss correlations, geometry selection, or a future chirp couples phase and response. Measure the residual error of the empirical form factor, especially where strong cancellation makes a small absolute error significant.

## Proposed source-sampling sequence

### Establish a comparison baseline

Retain ordinary sampling as the reference and compatibility policy. Define a source-sampling owner that produces initial coordinates, stable particle IDs, statistical weights, and provenance. Keep it separate from the physical beam distribution: increasing the proposal's tail coverage must not change the beam's declared emittance or halo fraction.

Separate source randomness from subsequent transport randomness. Preserve seeded legacy results with the old policy selected. A new policy must record its seed, algorithm version, replicate, and any proposal parameters in result identity.

### Trial scrambled Sobol source points

Map a joint scrambled Sobol design through the intended distributions and the existing Twiss transformations. Start with initial phase-space variables only; leave the variable-length sequence of collision events on ordinary independent transport streams. The SciPy implementation supports balanced power-of-two sample sizes and independent scrambling for error estimation {cite:p}`scipysobol`. Do not thin, skip, or arbitrarily truncate the design while claiming those balance properties.

Handle inverse-normal endpoints explicitly without silently clipping away a physical tail. Preserve the intended joint distribution, including discrete microtrain choices. Review sample-centering conventions before combining them with a correlated design. This is a trial for better coverage, not a guarantee that every rare region will be sampled.

### Add deliberate strata where observables require them

Candidate strata include beam core and halo, proximity to footprint boundaries, angular regions contributing to a detector, and resolved microbunch populations. Use disjoint strata with known physical probabilities and sample the conditional distribution within each. Allocate a minimum number to important rare strata, then compare a pilot-informed allocation against that baseline.

For stratum probability $P_h$, sample count $n_h$, and per-trajectory observable $f_{hi}$, the estimator is

```{math}
:label: eq-research-sampling-stratified-estimator

\widehat\mu = \sum_h \frac{P_h}{n_h}\sum_{i=1}^{n_h} f_{hi}.
```

Each stratum must have positive sample count when it can contribute. The weights represent probability mass, not the allocation fraction. This is standard stratified estimation {cite:p}`owenqmc`; it is not yet wired into PyRITE spectra.

### Introduce general source importance weights

For physical source density $p$, proposal density $q$, and trajectories evolved under the unchanged conditional transport law, estimate an incoherent expectation using

```{math}
:label: eq-research-sampling-importance-estimator

\widehat\mu = \frac{1}{N}\sum_{i=1}^{N} w_i f_i,
\qquad w_i = \frac{p(x_i)}{q(x_i)},\qquad x_i\sim q.
```

Require proposal support wherever the target integrand contributes and control large weights. This likelihood-ratio identity is standard importance sampling {cite:p}`owenqmc`. These statistical identities prescribe no new physical law.

For PyRITE, propagate source weights through every segment belonging to a particle, every radiation component, beam diagnostics, and acceptance accounting. Current `spec / Ne` reductions cannot handle deliberate source oversampling without modification. Keep misses as zero-contribution incident particles; normalizing only over survivors changes a per-incident-electron observable.

Do not silently replace division by $N$ with division by the realized weight sum. That creates a different, generally biased finite-sample ratio estimator. Report weight sums and effective sample size as diagnostics, but do not treat effective sample size alone as proof that a rare spectral feature has converged. Keep physical charge normalization distinct from numerical sampling weights.

## Coherent output is a separate estimator problem

Current coherent reductions include self terms and inter-electron cross terms. Nonuniform particle weights change their accounting differently. Multiplying existing intensities by a weight, or inserting square-root weights into an amplitude sum, is not a general solution. Simply squaring an estimated mean amplitude also introduces finite-sample self contributions.

Derive the intended physical bunch expectation, its relation to numerical particles, and the self/cross estimators explicitly. Independent-particle formulas cannot automatically be reused for stratified or Sobol designs, whose points are dependent. Existing physical-charge/macro-particle normalization questions remain open in the coherent ledger. Introduce no weighted coherent production path until these assumptions, limits, and normalization have independent validation.

## Other stochastic processes: priorities

Keep inverse-CDF free-path and elastic-angle sampling as the initial baseline. Their directness is valuable, but it provides no guarantee about rare-event precision. If a pilot demonstrates that large-angle scattering, long flights, or large losses dominate error, investigate transport importance sampling separately. Biasing these events requires the probability ratio of the complete history, including no-collision survival and boundary censoring. Source weights alone cannot correct a biased collision law.

Do not replace straggling by its mean merely to improve precision when the observable depends on fluctuations. Profile large-count Poisson work before changing the sampler, and retain distributional/tail and reproducibility checks.

For Timepix response construction, preserve conditional absorption sampling. Trial stratified subpixel positions and allocate response-column effort where thresholds or charge-sharing tails make the output sensitive. Judge errors after the response is applied to representative source spectra. Avoid introducing coherent artifacts or invalid uncertainties by accidentally sharing noise across events or columns.

Keep synthetic acquisition noise distinct from response calibration uncertainty. To predict an expected count, return the mean directly. To simulate an actual acquisition, preserve the specified Poisson and electronic fluctuations rather than reducing their variance artificially.

## Acceptance evidence and rollout

Compare ordinary sampling, scrambled source designs, and weighted strata at equal wall time. Use independent seed/scramble replicates; individual points within one Sobol design are not independent error replicates. Record both runtime and error for a fixed set of observables, with a separate confirmation set if a pilot tuned the proposal. Do not claim a universal speedup from one geometry.

Representative cases should include broad acceptance, narrow acceptance, footprint clipping, grazing geometry, grooves, compound/layered targets, straggling-sensitive spectra, and short/long/structured bunches. Measure integrated yield, weak background windows, line centroid/width, hit/miss fractions, and tail contributions. Use absolute tolerances near zero. Decompose source-sampling error from transport randomness and energy/angular discretization error; otherwise a better source design can appear ineffective because a different error dominates.

Required implementation checks include equal-weight recovery, known stratum probabilities, weighted beam moments, all-miss/partial-hit normalization, seed and batch reproducibility, and CPU/GPU consistency. Run heavy sweeps and GPU comparisons through the remote compute workflow.

The initial inspection was accompanied by 21 focused passing tests selected from `test_transverse.py`, `test_transport_per_electron.py`, `test_energy_loss_straggling.py`, and `test_straggling_rng_plumbing.py` under `tests/montecarlo/`. They cover selected moments, inverse-CDF behavior, RNG references, and limits. They do not establish variance-reduction efficiency, complete tail correctness, or a fresh GPU validation.

Recommended rollout: preserve existing conditional averages; benchmark improved source coverage for incoherent observables; add explicit weighted strata and complete weight propagation; independently derive coherent estimators; only then consider biased collision histories where measured error justifies the complexity.
