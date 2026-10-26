# Detector-cone variance reduction: pending verification

Validation: `detector-cone-variance-reduction`

**Status: unverified.** This records the intended CPU prototype estimator for
issue #203. It is implementation-context documentation, not an independent
derivation or a production recommendation. The [line-attribution research
record](../../research/beam-transport/detector-cone-line-variance.md) motivates
the study; it does not validate this estimator.

## Proposed estimator and source law

The source angular law is the repository's
[`elsepa-elastic-sampling`](../ledger-transport-background.md#elsepa-elastic-sampling)
claim: the normalized polar quantile interpolant at the current energy, with
uniform azimuth about the incoming direction. No new elastic cross section is
introduced. The proposed change of measure uses the elementary identity
$\mathbb{E}_q[(p/q)H]=\int pH$ wherever the proposal covers the score's support.

For an outgoing direction in the detector cone, `cone_split_event` samples the
analog polar quantile uniformly on a covering interval of mass $P_B$, followed
by the allowed azimuth arc of half-width $h(\theta)$ in radians. Relative to
the analog polar/azimuth law, the proposed likelihood ratio on that cone is

$$
w(\theta)=P_B\frac{h(\theta)}{\pi}.
$$

The prototype first applies roulette with probability
$s=\min(1,U/w_t)$, where $U$ is its computed bound on that likelihood ratio
and $w_t$ is the requested target weight. A surviving copy carries $w/s$.
The analog primary keeps its path before the event and its future outside
the cone; its future is stopped when the analog outgoing direction lies
inside the cone. Copies continue without further splitting.

The intended sample unit is one incident primary, including all its copies:

$$
X_i=H_i+\sum_{c:\,\mathrm{parent}(c)=i}w_c H_c,
\qquad
\bar X=\frac{1}{N}\sum_i X_i,
\qquad
\mathrm{SE}(\bar X)=\frac{s_X}{\sqrt N}.
$$

Weights, quantile probabilities and the reported relative SE are dimensionless;
$X_i$ retains the units of its underlying tally. Copies of one primary are
correlated and are not separate independent observations. The intended
figure of merit is $1/(\mathrm{RSE}^2 T)$ with wall time $T$ in seconds.

## Assumptions and limits requiring verification

- CPU per-electron, ungrooved ELSEPA transport; no secondary cascade, GDF
  source or electron-block admission. One detector direction per run.
- Linear incoherent tallies only. Coherent amplitude cross terms between
  copies are outside this estimator's contract.
- Independent copy keys depend on primary stream key and flight index;
  split decisions must not consume the primary transport stream.
- A disabled request must preserve analog arrays and RNG behavior. Vanishing
  cone mass must create no weighted future; roulette must preserve the
  conditional expectation. Copy-buffer exhaustion raises instead of silently
  dropping contributions.

These are intended contracts, not verified conclusions. In particular, the
proposal support, polar-quantile interval, bound used by roulette, event and
path partition, copy launch state and every supported radiation tally still
need a fresh-context source-to-code check.

## Available evidence and remaining work

`tests/montecarlo/test_cone_split.py` checks primary-stream prefixes, copy-prefix
stability, per-primary aggregation and a three-seed cone/path-length statistic.
Those geometric tallies do not establish the issue's five-seed PXR/CBS,
bremsstrahlung and characteristic-yield acceptance.

`checks/detector_cone_variance_reduction.py` provides remote baseline and
comparison commands. No measured 5 MeV h-BN baseline, radiation-yield agreement,
FOM improvement or method-selection verdict is asserted here. Complete those
measurements through the remote workflow, compare the alternative angular
importance sampler, and independently verify the estimator before advancing
the ledger to `rederived`. Human sign-off remains separate in #277.
