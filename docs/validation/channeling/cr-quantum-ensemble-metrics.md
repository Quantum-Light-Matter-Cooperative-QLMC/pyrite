# Quantum ensemble evolution metrics

## Independent derivation

Claim: `cr-quantum-ensemble-metrics`. This derivation precedes inspection of the
new metric implementation. Inputs and intended signatures were supplied by the
implementer; the equations below were derived independently from density-matrix
linearity and discrete inner products.

For context on stochastic trace estimation see
[Epperly, Tropp and Webber, XTrace (2024)](https://tropp.caltech.edu/papers/ETW24-XTrace-Making-SIMAX.pdf).
The probe identity used here follows immediately from the covariance condition
below. This implementation uses the elementary estimator, not XTrace.

Let $\rho\succeq0$ be the incident-normalized represented ensemble density and
$T=\operatorname{Tr}\rho$. Surviving MCWF samples have unit norm; removed
samples contribute zero. A physical grid wavefunction is represented by
$u_j=\sqrt{\Delta A}\exp(i\mathbf{k}_t\cdot\mathbf{r}_j)\phi_j$.
The area factor is essential for the dimensionless discrete inner product.

For $T>0$, conditional purity and squared Frobenius off-diagonal content are

$$
\mathcal P=\frac{\operatorname{Tr}\rho^2}{T^2},\qquad
\mathcal C_2^2=\frac{\operatorname{Tr}\rho^2-\sum_j\rho_{jj}^2}{T^2}.
$$

The second quantity depends on the explicitly chosen orthonormal basis. It is
a descriptive coherence diagnostic, not a claimed resource-theory monotone.
If $\rho$ is projected to a proper sector, these describe that conditioned
sector only. Its trace and omitted population must be displayed separately.
One must not combine nonorthogonal localized projectors as a complete basis.

For an orthogonal block representation, the exact numerator is
$\sum_{r,m,n}|\rho_{r,mn}|^2$. The diagonal sum uses that same block basis;
coherence curves in position and energy bases need different labels.

Let independent complex random vectors obey $\mathbb E[zz^\dagger]=I$.
Fourth roots of unity independently sampled in each coordinate satisfy this
condition. Set $v_s=\rho z_s$ and $q_s=\|v_s\|^2$. Then

$$
\mathbb E q_s=\operatorname{Tr}(\rho^\dagger\rho)=\operatorname{Tr}\rho^2.
$$

For $S\ge2$ probes, the standard error of the mean is
$\operatorname{sd}(q_s)/\sqrt S$. Division by $T^2$ gives the conditional
purity probe error. Subtracting the exactly stored diagonal term gives the same
probe error for squared off-diagonal content. This error is conditional on the
supplied empirical density: it excludes trajectory sampling error, numerical
discretization and model error. Finite-probe estimates can be negative for the
off-diagonal diagnostic or exceed one for purity; clipping would hide error.

The actions can be accumulated without storing a dense density matrix:

$$
\rho z_s=\frac1N\sum_{e\,\mathrm{alive}}
u_e(u_e^\dagger z_s).
$$

Probe draws must be independent of trajectory streams. The storage budget is
proportional to recorded depths times probe count times grid dimension; admit
that budget before solving.

## Finite trajectory bias

Write $\widehat\rho=N^{-1}\sum_e X_e$, with independent identically distributed
$X_e=0$ or a unit-rank projector, and $\rho=\mathbb E X_e$. Expanding the
double sum separates equal and unequal electron indices:

$$
\mathbb E\operatorname{Tr}\widehat\rho^2
=\left(1-\frac1N\right)\operatorname{Tr}\rho^2+\frac{T}{N}.
$$

Consequently the plug-in empirical purity has trajectory bias even if the
trace is estimated with infinitely many probes. It is acceptable as a labelled
finite-ensemble diagnostic with $N$ and limitations disclosed. An unbiased
estimator of the unconditioned quadratic numerator for $N>1$ is

$$
\frac{N\operatorname{Tr}\widehat\rho^2-\widehat T}{N-1}.
$$

Dividing by the random $\widehat T^2$ does not generally preserve unbiasedness.
Conditional on $L>1$ surviving independent normalized samples, the unbiased
conditional purity is $(L\widehat{\mathcal P}-1)/(L-1)$. Deterministic
weighted coherent mixtures are a different input and need no trajectory
correction. Partial-norm projected samples need their own diagonal correction,
not the unit-projector formula.

## Transverse energy and empty ensembles

For a self-adjoint transverse Hamiltonian $H$, conditional moments are

$$
\bar E=\frac{\operatorname{Tr}(\rho H)}T,\qquad
\sigma_E^2=\frac{\operatorname{Tr}(\rho H^2)}T-\bar E^2.
$$

The pure-state second moment is $\|Hu\|^2$, including potential/kinetic cross
terms. Squaring an individual energy expectation instead omits intrinsic
quantum variance. In the axial carrier frame, apply the kinetic multiplier
$\hbar^2|\mathbf{k}+\mathbf{k}_t|^2/(2\gamma m)$ to $\phi$, then add
$U\phi$. Ensemble averages must use each sample's carrier consistently.
Mean and standard deviation have units eV; the variance has units eV squared.

For $T=0$ populations and raw moments vanish, but conditional purity,
coherence and energy metrics are undefined. NaN with an explicit empty-state
label is a valid convention. Only roundoff-sized negative exact variances may
be rounded to zero. A pure state has purity one; an equal incoherent mixture of
$d$ orthogonal states has purity $1/d$; an energy eigenstate has zero spread.

## Source-to-code comparison

The verifier inspected `metrics.py` only after writing the independent
derivation. `density_metrics` sums the squared moduli of all supplied Bloch
blocks before conditioning on their total trace. Its energy moments use the
same diagonal populations and energies. `AxialMetricsObservable.__call__`
constructs the physical-grid state with both carrier and square-root area,
forms the correctly conjugated state/probe overlap, and returns the rank-one
density actions. The Hamiltonian uses the tilted kinetic multiplier plus real
potential; the second moment is the norm squared of that full Hamiltonian
application. `finish` squares the ensemble-averaged actions, not the individual
state purities, and subtracts the full-grid diagonal term with the area factor
squared. Probe SEM uses independent probes and sample standard deviation.

An initial implementation silently clamped all negative energy variances.
The verifier requested a cancellation-scaled tolerance; `_energy_spread` now
rejects substantive negative variance and clamps only roundoff. Empty
conditional metrics remain NaN. Upfront positive integer and storage checks
also precede probe allocation. The output explicitly describes empirical
finite-ensemble purity and excludes trajectory error/bias from probe SEM.
No claim of unbiased finite-trajectory conditional purity is made.

The named position-basis and energy/Bloch-basis coherence diagnostics are not
basis-invariant comparators. The app must preserve those labels and disclose
finite-ensemble/probe limitations. The mathematical verdict below does not
validate omitted states, a physical escape model, model energy range, or
statistical convergence of a particular run.

- **Claim**: `cr-quantum-ensemble-metrics` —
  `src/pyrite/montecarlo/channeling/metrics.py::{density_metrics,AxialMetricsObservable}` —
  density-operator trace identities and probe covariance derivation above.
- **Filters**: units pass (dimensionless purity/coherence, eV energy moments);
  limits pass (pure state, orthogonal mixture, energy eigenstate, empty trace);
  signs/conventions pass (conjugated overlaps, carrier and sample-area factors).
- **Re-derivation**: matches — empirical ensemble quadratic numerator, probe
  standard error, basis-dependent off-diagonal subtraction and Hamiltonian
  moments agree with the independent equations.
- **Verdict**: rederived, scoped to the finite represented empirical ensemble.
- **Write-up**: `docs/validation/channeling/cr-quantum-ensemble-metrics.md`.
- **Suggested ledger change**: owner may apply rederived, then anchored after
  independent dense-Hamiltonian/rank-one-action and analytic-mixture regressions
  pass. Preserve explicit sampling and basis limitations; no human sign-off.
