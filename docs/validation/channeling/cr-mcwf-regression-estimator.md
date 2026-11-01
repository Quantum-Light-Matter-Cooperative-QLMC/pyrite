# Four-map MCWF quantum-regression estimator

## Independent derivation before implementation

Proposed claim: `cr-mcwf-regression-estimator`. This derivation precedes any
new implementation body. It concerns a supplied common linear quantum map,
not proof that a particular axial material/escape model is valid.

Starting point is the regression relation in
[Landi et al., PRX Quantum 5, 020201 (2024), Eq. (95), arXiv v3](https://arxiv.org/pdf/2303.04270v3).
For later depth $z$ and earlier depth $z_0$, the selected convention is
$G_{ab}(z,z_0)=\operatorname{Tr}[J_a\Phi_{z-z_0}(\rho(z_0)J_b^\dagger)]$.
The following polarization identity reduces its non-Hermitian initial
operator to four positive rank-one operators and follows directly by expansion.

For a parent pure state $\psi$ and $b=J_b\psi$, define
$\chi_s=\psi+s b$, $R_s=|\chi_s\rangle\langle\chi_s|$, for
$s\in\{1,-1,i,-i\}$. Expansion gives

$$
\boxed{|\psi\rangle\langle\psi|J_b^\dagger
=\frac14(R_1-R_{-1}+iR_i-iR_{-i}).}
$$

The plus-$i$ sign is required for the right-inserted dagger operator. Its
complex conjugate identity would instead produce $J_b|\psi\rangle\langle\psi|$.

Let $n_s=\langle\chi_s|\chi_s\rangle$. For $n_s>0$, evolve the normalized
auxiliary state $\chi_s/\sqrt{n_s}$ with the same supplied Markov model
and take its incident-normalized future expectation $X_{a,s}$ of $J_a$.
Removed auxiliary histories contribute zero. Linearity gives

$$
G_{ab}=\frac14\left(n_1\mathbb E X_{a,1}
-n_{-1}\mathbb E X_{a,-1}
+i n_i\mathbb E X_{a,i}
-i n_{-i}\mathbb E X_{a,-i}\right).
$$

An identically zero auxiliary state contributes zero without drawing a
trajectory. No further normalization of the weighted combination is allowed.
Grid states use $\langle\phi|\phi\rangle=\Delta A\sum_j|\phi_j|^2$;
the same inner product must define $n_s$, auxiliary normalization and
observable expectations. The identity also holds for subnormalized parent
states: their statistical weight is already included in $n_s$.

## Parent ensemble and physical carrier blocks

For MCWF parents, average over the original number of incident parents,
including zero for parents removed before $z_0$. Do not divide by the number
of surviving parents or multiply by survival a second time. For a deterministic
weighted entry mixture, use its declared weights instead of treating components
as equally likely.

If the model is a direct integral over carrier momentum, a parent at carrier
$k$ initially occupies that block. On axis $J_b(k)$ preserves the carrier;
construct all four auxiliary states at that same carrier, then allow the same
physical TDS/electronic transfer rules to move the auxiliary carrier later.
The later observable uses the auxiliary history's current carrier. This
samples the supplied common block-linear map, provided its conditional jump
algorithm is a correct unraveling of that map. Applying a different nonlinear
state-dependent physical model to each branch invalidates the identity.

Sampling an auxiliary entry recipe anew, discarding its relative carrier,
using the parent's future noise as independent noise, or replacing future
current expectation by a population proxy changes the requested correlation.
Sinks remain whatever unresolved or absorbing outcomes the supplied model
declares; the estimator cannot turn numerical loss into physical escape.

## Random streams and uncertainty

Use a versioned namespace addressing the parent incident identity, earlier
recorded-depth index, inserted current component, replica index and one of the
four branch labels. It must be separate from primary MCWF entry/jump streams.
Processing order and admitted batch size must not change any address. A
bounded injective tuple packing with checked field ranges is an acceptable
way to allocate disjoint Philox subsequences; an unspecified small seed hash
is not evidence of nonoverlap.

For replica $r$, define the complex combined estimator
$Y_{a,r}=\sum_s c_s n_s X_{a,s,r}$, with
$(c_1,c_{-1},c_i,c_{-i})=(1,-1,i,-i)/4$. Independent branch simulations make
each replica unbiased; deliberately coupled branches can also be valid if
their marginal laws remain correct and uncertainty is computed on the
combined replica. For $R>1$ independent combined replicas,

$$
\widehat G_a=\frac1R\sum_rY_{a,r},\qquad
\operatorname{SEM}_{\Re G_a}
=\frac{\operatorname{sd}(\Re Y_{a,r})}{\sqrt R},\qquad
\operatorname{SEM}_{\Im G_a}
=\frac{\operatorname{sd}(\Im Y_{a,r})}{\sqrt R}.
$$

The real/imaginary covariance and cross-component/depth dependence matter
when contracting to a spectrum. For a final linear spectral contraction,
contract each independent replica/chunk first and compute the SEM of those
contracted results. Parent-sampling and auxiliary-trajectory uncertainty must
both be represented in the independent replication design. Repeating only
auxiliary paths for one fixed empirical parent set gives conditional
auxiliary error, not the full parent-ensemble error.

Finite-sample correlation estimates need not be positive semidefinite, and
their spectral contractions can fluctuate below zero. Do not apply a
deterministic roundoff-only positivity check to stochastic noise, silently
clip negative estimates, or project the kernel to positive eigenvalues while
claiming unbiasedness. Preserve estimates and statistical errors; any physical
production nonnegativity policy needs separate explicit convergence criteria.

## Independent limiting checks

Direct dense linear-map propagation of $\rho J_b^\dagger$ must equal the
four-map combination, including a non-Hermitian insertion and complex current.
For $J_b=0$ the combination cancels to zero. For a lowering current acting on
the ground state every branch has the same state and the weighted result is
zero. Zero-norm branches must be skipped without affecting other addresses.
Subnormalized parents and mixed parents must scale linearly. With an
undamped two-level Hamiltonian the earlier regression oracle gives
$G(z,z_0)=e^{-i\Delta(z-z_0)/h_v}$ from an initially excited state. Loss and
dephasing must agree with their dense Lindblad oracles. Tests must pin the
complex sign and original-incident normalization, not only a real intensity.

## Source-to-code comparison

Pending signatures, implementation and independent oracle results.

## Fresh-context verification (2026-10-22)

Written by a fresh-context verifier from the ledger row, the module and function
docstrings, and the cited sources only. The derivation in this section was
completed before reading the author sections above or any implementation body;
the comparison subsections that follow it were added afterwards.

### Quantity and convention

For a supplied common linear (completely positive, trace non-increasing) map
$\Phi = \Phi_{z\leftarrow z_0}$ and an earlier, possibly subnormalized, pure
parent $\rho = |\phi\rangle\langle\phi|$, the target is

$$
G_{ab}(z,z_0)=\operatorname{Tr}\!\left[J_a\,\Phi\!\left(\rho\,J_b^\dagger\right)\right].
$$

With the quantum-regression theorem (cited as Landi et al., PRX Quantum 5,
020201 (2024), Eq. (95); the verifier used the standard regression form and did
not consult the paper text in this context), for a
Markovian semigroup $\Phi=e^{\mathcal L (z-z_0)}$ this is the ordered correlation
$\langle J_b^\dagger(z_0)\,J_a(z)\rangle$, insertion on the right of the state.
$G$ carries the units of $J_a J_b^\dagger$ times the parent weight
$\operatorname{Tr}\rho=\|\phi\|^2$ (dimensionless under the inner product
$\|\phi\|^2=A_{\rm cell}\sum\lvert\phi\rvert^2$). It is incident-normalized: no
division by surviving probability.

### Rank-one polarization identity

Write $\beta=B\phi$ with $B=J_b$, so $\rho J_b^\dagger=|\phi\rangle\langle\beta|$.
For $s\in\{1,-1,i,-i\}$ and $R_s=|\phi+s\beta\rangle\langle\phi+s\beta|$,

$$
R_s=|\phi\rangle\langle\phi|+s\,|\beta\rangle\langle\phi|+s^*\,|\phi\rangle\langle\beta|+\lvert s\rvert^2|\beta\rangle\langle\beta|.
$$

With weights $c_s=(1,-1,i,-i)$, the sums $\sum_s c_s=0$,
$\sum_s c_s\lvert s\rvert^2=0$, $\sum_s c_s s=1+1-1-1=0$ and
$\sum_s c_s s^*=1+1+1+1=4$ give

$$
\boxed{|\phi\rangle\langle\phi|B^\dagger=\tfrac14\left(R_1-R_{-1}+iR_i-iR_{-i}\right).}
$$

The sign of the imaginary pair is fixed by the right insertion: with $-iR_i+iR_{-i}$
one would instead isolate $|\phi\rangle\langle\beta|$ with the opposite sign and,
with the conjugate weights $c_s^*$, the left insertion $B|\phi\rangle\langle\phi|$.

### Balanced branches

For $\|\beta\|\neq0$ and $\|\phi\|\neq0$ set $b=\|\beta\|/\|\phi\|$. Since
$|\phi\rangle\langle\beta|=b\,|\phi\rangle\langle\beta/b|$, applying the identity
with $\beta\to\beta/b$ gives
$\chi_s=\phi+s\beta/b$ (both terms of norm $\|\phi\|$) and

$$
\rho B^\dagger=\frac b4\sum_s c_s\,\|\chi_s\|^2\,|u_s\rangle\langle u_s|,\qquad u_s=\chi_s/\|\chi_s\|.
$$

Linearity of $\Phi$ and of the trace then gives

$$
G_{ab}=\frac b4\sum_s c_s\,\|\chi_s\|^2\,g_s,\qquad
g_s=\operatorname{Tr}\!\left[J_a\,\Phi(|u_s\rangle\langle u_s|)\right].
$$

Branches with $\|\chi_s\|=0$ contribute exactly zero and need no evolution. If
$\beta=0$ (including the lost-parent case $\phi=0$), $G_{ab}=0$ exactly.

### Monte Carlo unravelling of each branch

Each $g_s$ is a positive-map expectation from a unit pure state, so any unbiased
unravelling of the same $\Phi$ applies: with $N$ trajectories
$\psi^{(k)}_s(z)$ started from $u_s$, carrying their (non-renormalized) weight so
that $\mathbb E[|\psi\rangle\langle\psi|]=\Phi(|u_s\rangle\langle u_s|)$, or
equivalently renormalized states with lost trajectories counted as zero and the
mean taken over all $N$ started trajectories,

$$
\hat g_s=\frac1N\sum_{k=1}^{N}\langle\psi_s^{(k)}|J_a|\psi_s^{(k)}\rangle,\qquad
\mathbb E\,\hat g_s=g_s.
$$

Dividing by the number of survivors would estimate
$g_s/\operatorname{Tr}\Phi(|u_s\rangle\langle u_s|)$ and is wrong whenever $\Phi$ is
trace-decreasing.

### Subnormalized parent and linearity

Replacing $\phi\to\sqrt p\,\phi$ leaves $b$ and $u_s$ unchanged and multiplies every
$\|\chi_s\|^2$ by $p$, so $\hat G\to p\,\hat G$ sample-by-sample, in agreement with
$G$ being linear in $\rho$. A complex phase $\phi\to e^{i\theta}\phi$ leaves
$\rho$, $b$, $\|\chi_s\|$ and $|u_s\rangle\langle u_s|$ unchanged.

### Uncertainty

The four branch estimates enter $\hat G$ with complex weights. If each replica
$r$ forms $\hat G^{(r)}=\tfrac b4\sum_s c_s\|\chi_s\|^2\hat g^{(r)}_s$ from
independent branch streams, the replicas are i.i.d. and the standard error and the
$2\times2$ covariance of $(\operatorname{Re}\hat G,\operatorname{Im}\hat G)$ follow
from the sample covariance over $r$ divided by the number of replicas. Combining
per-branch SEMs in quadrature is correct only for independent branches and loses
the real/imaginary covariance; doing it from correlated branches (shared seeds)
would be wrong. Pooled joint replicas also preserve covariance across depths and
components.

### Limiting cases derived

- $B=0$ or $\phi=0$: $G=0$.
- Unitary two-level, $H=\tfrac\omega2\sigma_z$, $J_a=J_b=\sigma_-$,
  $\phi=|e\rangle$: $\Phi(\rho\sigma_+)=e^{-iH\tau}|e\rangle\langle g|e^{iH\tau}
  =e^{-i\omega\tau}|e\rangle\langle g|$, so
  $G=\operatorname{Tr}[\sigma_-e^{-i\omega\tau}|e\rangle\langle g|]=e^{-i\omega\tau}$.
  Positive-frequency ordering gives a negative phase for $e^{-iHt}$ evolution.
- Add decay $L=\sqrt\gamma\sigma_-$: $G=e^{-i\omega\tau-\gamma\tau/2}$.

### Comparison with the author derivation

Read after the derivation above was written. The author's pre-code derivation uses
the unbalanced branches $\chi_s=\psi+sJ_b\psi$ and the same weights
$(1,-1,i,-i)/4$; the boxed identity, the $+i$ sign for the right-inserted
$J_b^\dagger$, the $n_s=\langle\chi_s|\chi_s\rangle$ reweighting, the
incident normalization, zero-branch skipping, subnormalized linearity and the
combined-replica errors all agree with this section. The balanced form used by
the implementation (scaling $J_b\psi$ by $1/b$ and restoring the outer factor
$b$) is the same identity applied to $|\psi\rangle\langle J_b\psi/b|$; it is
exact and is a numerical-conditioning choice only. The author's closed form
$e^{-i\Delta(z-z_0)/\hbar v}$ is this section's $e^{-i\omega\tau}$ with
$\omega\tau=\Delta(z-z_0)/\hbar v$.

### Source-to-code comparison

`four_map_regression` accumulates, per replica and branch,

```text
chi = parent + phase * inserted          # inserted = B phi / scale
norm = cell_area * sum(|chi|**2)
replicas[replica] += coefficient * scale * norm * evolve(chi/sqrt(norm), seed, N) / 4
```

with `_BRANCH_PHASES = _BRANCH_COEFFICIENTS = (1, -1, 1j, -1j)` and
`scale = sqrt(||B phi||^2 / ||phi||^2)`. This is term-by-term
$\tfrac b4\sum_s c_s\|\chi_s\|^2\hat g_s$ with $c_s=s$, which is correct because
for $s\in\{1,-1,i,-i\}$ the required weights $c_s$ with $\sum_s c_s s^*=4$,
$\sum_s c_s s=0$ coincide with $s$ itself.

| Item | Derivation | Code | Result |
|---|---|---|---|
| Imaginary-pair sign | $+iR_i-iR_{-i}$ | coefficient `1j` on phase `1j` | match |
| Branch norm | $\|\chi_s\|^2$ with $A_{\rm cell}$ | `norm` uses `cell_area` | match |
| Outer factor | $b=\|B\phi\|/\|\phi\|$ | `scale` | match |
| Zero parent / zero $B\phi$ | $G=0$, no evolution | early `_summarize` of zeros | match |
| Zero branch | skipped | `continue`, no seed consumed | match |
| Incident normalization | divide by started $N$ | `AxialMCWF.run` divides sums by `n_electrons`, removed states contribute zero | match |
| SEM | across combined replicas | `std(ddof=1)/sqrt(R)` of `replicas` | match |
| Re/Im covariance of the mean | $\sum_r\delta\Re\,\delta\Im/[R(R-1)]$ | same | match |

`axial_mcwf_regression` applies the insertion at the parent's carrier
`tilt_inv_ang` (the solver's `base_tilt = momentum * tilt_rad` reproduces the same
value), checks that `from_array` reconstructs the unit auxiliary state on the grid,
and passes `current_expectation` to `AxialMCWF.run`, whose `_observe` evaluates it
on each live renormalized state with that trajectory's current `tilts[b]`
(updated by electronic jumps). The auxiliary seed packing gives local key
$\mathrm{mix}((\mathrm{root}\oplus\mathrm{salt})+G(A+e+1))
=\texttt{mcwf\_stream\_key}(\mathrm{root},A+e)$, and the domain nibble `0x9`
puts every $A\ge2^{63}$, disjoint from primary indices below $2^{63}$.

### Independent numeric oracles

A verifier-written scratch script (not using implementation helpers as reference)
built a three-level GKSL Liouvillian (random Hermitian $H$, two random jump
operators, a trace-decreasing loss term without reinjection) and propagated
$\rho J_b^\dagger$ with `scipy.linalg.expm`, for random non-Hermitian $J_a,J_b$ and a
subnormalized parent with $\|\phi\|^2=0.37$:

- Deterministic branch map supplied to `four_map_regression`: maximum deviation
  from the dense $\operatorname{Tr}[J_a\Phi(\rho J_b^\dagger)]$ is $5.7\times10^{-16}$;
  deviation from the left-insertion $\operatorname{Tr}[J_a\Phi(J_b\rho)]$ is $1.9$, so
  the convention test is discriminating.
- Verifier-written first-order MCWF unravelling ($\Delta z=0.01$, lost trajectories
  counted as zero, mean over started trajectories), compared with the matching
  discrete linear map: over 8 independent addresses with 12 replicas, the 48
  standardized residuals of the real and imaginary parts at nonzero depth have
  mean $0.14$ and RMS $0.98$. The combined-replica SEMs are calibrated and the
  estimator is unbiased under a trace-decreasing map.
- Two-level $H=\tfrac\omega2\sigma_z$ with amplitude damping, $J_a=J_b=\sigma_-$,
  $\phi=|e\rangle$: the estimator reproduces $e^{-i\omega\tau-\gamma\tau/2}$ to
  printed precision (eight digits), including the negative phase.

The anchor module `tests/montecarlo/quantum/test_axial_mcwf_regression.py` passes
(13 tests).

### Zero-scattering axial adapter oracle

The ledger lists this check as pending, but it already exists:
`test_actual_axial_adapter_free_grid_unitary_array_reconstruction_and_carrier`
runs the real CPU `AxialMCWF` (silicon $\langle001\rangle$ dissipator with
`mechanisms=()`, zero potential) through `axial_mcwf_regression` and compares
against an independent FFT free-kinetic propagation of $\phi$ and $B\phi$,
$\langle U B\phi|M|U\phi\rangle$, at the tilted carrier, to $10^{-14}$. This is the right
convention ($\operatorname{Tr}[M\,U\rho B^\dagger U^\dagger]$). Its coverage is limited:
no potential, no jumps, so the carrier never changes, no auxiliary trajectory is
lost, and the replicas carry no noise. The paths where future observables use
an electronically updated carrier, and incident normalization with real sinks
through the actual adapter, are verified here by code reading only, not by an
oracle.

### Verdict

`rederived`. The implemented estimator matches the independent derivation
symbolically and numerically. No discrepancy found. The ledger `Checks`
text is stale: the dense-map and zero-scattering adapter oracles exist in the
anchor module. A carrier-changing or lossy actual-adapter oracle is still absent.
