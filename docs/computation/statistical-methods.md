# Statistical methods

:::{note}
**Stub.** Scope is defined below; the content has not been written. Current
authoritative material is linked at the end.
:::

## Intended scope

The estimators this package forms and how their uncertainty is quantified.

* **What is being estimated.** Every observable as a per-incident-electron
  expectation: exit fractions, path-length and energy moments, spectra per unit
  energy and solid angle. Why there are no statistical weights and what that
  buys.
* **Error bars.** Binomial standard errors for fractions, sample standard errors
  for means, and the $1/\sqrt{N}$ scaling the ledger repeatedly checks against.
  How error propagates through a spectral reduction that is a sum of squares.
* **Comparing two runs.** Paired-seed shift/$\sigma$, Welch tests, and when the
  independence assumption in the denominator is conservative rather than wrong.
  Seed replication as the tool that separates a real one-signed bias from noise.
* **Coherent versus incoherent statistics.** Why the incoherent sum is
  well-conditioned and the global coherent sum is not: a squared global sum that
  is a small residual of a large cancellation has an error that does not shrink
  under step refinement, and that must be stated as a property of the estimator.
* **Convergence protocol.** The two independent axes — sample size and numerical
  step — and why refinement decorrelates trajectories, so per-realization
  comparison measures nothing and ensemble means with errors are mandatory.
* **Null results.** How to report a measured invariance as a bound at a stated
  resolution rather than as a proof, which is the form most of the transport
  ledger's step-control claims take.
* **Histogramming and binning.** Energy-grid construction, bin-width effects on
  line observables, and peak-pick instability near a bin edge.

## Deliberately out of scope

Physical models and their derivations. The evidence standard and sign-off
process, which belong to
[validation methodology](../validation/methodology.md); this page supplies the
statistical machinery that methodology assumes.

## Current sources

* [`validation/methodology.md`](../validation/methodology.md)
* ledger rows `energy-step-convergence`, `substep-radiation-invariance`,
  `gpu-transport-core`
* [`physics/beam-transport/transport-outputs.md`](../physics/beam-transport/transport-outputs.md#checking-convergence)
