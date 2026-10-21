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
