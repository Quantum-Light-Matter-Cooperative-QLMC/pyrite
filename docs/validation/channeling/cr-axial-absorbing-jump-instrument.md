# Fixed finite absorbing axial jump instrument

## Independent derivation

Claim: `cr-axial-absorbing-jump-instrument`. This fresh verification context
completed this section before reading the implementation or its tests.
The source is [Castin, Dalibard and Mølmer, arXiv:0805.4002](https://arxiv.org/abs/0805.4002),
Eqs. (2), (4), (8), and the fixed-projector accounting in
[the axial escape note](../../research/physics/axial-escape-instrument.md#three-subspaces-not-two-loss-labels).
The claim is finite supplied-map algebra; no material escape approximation,
matching surface, tail completion, or native continuation follows from it.

Let the retained space have dimension $n$, and the outgoing discrete position
space dimension $m=n_xn_y$ with C-order indices. Supply Hermitian $H$ in eV,
retained jumps $A_a$ and rectangular outgoing jumps $B_b$ in
Å$^{-1/2}$, and a positive unresolved effect $E$ in Å$^{-1}$.
The constant positive scale $h=\hbar v$ has units eV Å. The initial retained
density $\rho_0$ is positive with incident weight
$t_0=\operatorname{tr}\rho_0\leq1$. Its weight must not be normalized away.

The total effect and retained generator are

$$
D=\sum_a A_a^\dagger A_a+\sum_b B_b^\dagger B_b+E,
\qquad K=-iH/h-D/2,
$$

$$
\mathcal L(\rho)=K\rho+\rho K^\dagger+
\sum_a A_a\rho A_a^\dagger.
$$

This follows from the full GKSL equation by retaining the recycling gains
inside the retained block and making outgoing and unresolved outcomes absorbing.
A factorization of $E$ into unresolved jumps exists because $E$ is positive.
The extended evolution is completely positive; its retained compression is
positive and trace decreasing. Retained recycling is essential: including
$A_a^\dagger A_a$ in $D$ without its gain would create spurious escape.
Each term of $\mathcal L$ has units Å$^{-1}$ times density.

For column-major vectorization, the identity
$\operatorname{vec}(X\rho Y)=(Y^{\mathsf T}\otimes X)
\operatorname{vec}\rho$ fixes the conjugations:

$$
L=I_n\otimes K+K^*\otimes I_n+
\sum_a A_a^*\otimes A_a,
\qquad G=\sum_b B_b^*\otimes B_b.
$$

Here $G$ maps $n^2$ retained coefficients into $m^2$ outgoing coefficients.
Define the instantaneous outgoing gain and integrated density by

$$
\sigma(z)=\sum_b B_b\rho(z)B_b^\dagger,
\qquad Q(z)=\int_0^z\sigma(s)\,ds,
\qquad U(z)=\int_0^z\operatorname{tr}[E\rho(s)]\,ds.
$$

All are positive, with $\sigma$ in Å$^{-1}$ and $Q,U$ dimensionless.
Writing $e^\mathsf T\operatorname{vec}\rho=\operatorname{tr}(E\rho)$,
the constant augmented generator is

$$
\frac{d}{dz}\begin{pmatrix}r\\q\\u\end{pmatrix}
=\begin{pmatrix}L&0&0\\G&0&0\\e^\mathsf T&0&0\end{pmatrix}
\begin{pmatrix}r\\q\\u\end{pmatrix},
\qquad (r,q,u)(0)=(\operatorname{vec}\rho_0,0,0).
$$

An augmented exponential avoids inverting $L$, which may be singular.
Tracing $\mathcal L$ cancels the retained recycling terms and yields

$$
-\partial_z\operatorname{tr}\rho
=\operatorname{tr}\sigma+\operatorname{tr}(E\rho),
\qquad \boxed{\operatorname{tr}\rho(z)+\operatorname{tr}Q(z)+U(z)=t_0}.
$$

## Conditional depth and joint Weyl measurement

For a normalized finite window $g$, the Weyl effects
$M_{a,b}=|g_{a,b}\rangle\langle g_{a,b}|/m$ satisfy
$\sum_{a,b}M_{a,b}=I_m$, as independently derived in
[the finite phase-space instrument](cr-axial-phase-space-instrument.md).
Put $C(z)=\operatorname{tr}Q(z)$ and $c(z)=\operatorname{tr}\sigma(z)$.
For an outgoing event in $0\leq z\leq Z$ with $C(Z)>0$, the joint law is

$$
\Pr(dz,a,b\mid F\text{ by }Z)
=\frac{\operatorname{tr}[M_{a,b}\sigma(z)]}{C(Z)}\,dz.
$$

Sample $v$ uniformly on $[0,1)$, invert $C(z)=vC(Z)$, and at that depth
sample the full mixed-density Weyl law of $\sigma(z)/c(z)$. The product is
exactly the joint law above; sampling an integrated outgoing state separately
from depth would generally erase its correlations. The local trace is a rate,
whereas $C(Z)$ is a cumulative incident-normalized event weight. Isolated
zero-rate depths have zero measure. A zero cumulative weight has no conditional
distribution and must refuse before random draws. A numerical root bracket
provides an absolute depth tolerance, not a uniform probability-error bound.

## Filters and independent limits

- Units: pass; $H/h$, $D$, $L$, $G$, $E$ are per Å, and all integrated
  traces preserve the initial incident weight.
- Signs and conventions: pass; $-i[H,\rho]/h$ is the source Hamiltonian
  sign; gain vectorization uses complex conjugation rather than transpose.
- No jumps and $E=0$: unitary retained evolution, no outgoing event.
- Scalar retained model with outgoing rate $\beta$ and unresolved rate
  $\epsilon$: $\rho(z)=t_0e^{-\lambda z}$,
  $C(z)=t_0\beta(1-e^{-\lambda z})/\lambda$,
  $U(z)=t_0\epsilon(1-e^{-\lambda z})/\lambda$,
  where $\lambda=\beta+\epsilon$. Scalar retained recycling cancels.
  The conditional inverse is
  $z=-\ln[1-v(1-e^{-\lambda Z})]/\lambda$.
- Zero depth: $Q(0)=0$, $U(0)=0$, $\rho(0)=\rho_0$; initial outgoing
  gain can already be nonzero, without an integrated event at zero depth.
- Dark retained states can survive forever; eventual unit outgoing probability
  is not an assumption. A rectangular coherent $B_b$ retains its outgoing
  off-diagonal matrix elements; an outgoing loss effect alone cannot supply them.

## Source-to-code comparison

After completing the independent derivation, the verifier inspected
`axial_jump_instrument.py`. The retained block implements the commutator and
anticommutator with the derived column-major factors, then adds exactly the
retained recycling gains. The outgoing block sums the rectangular Kronecker
maps. `unresolved.T.ravel(order="F")` gives the correct trace functional,
including complex off-diagonal effects. `at_depth` exponentiates the augmented
generator rather than dividing by a possibly singular retained generator.
The local gain applies the same outgoing block to the retained state.

`sample_absorbing_outgoing` inverts the cumulative outgoing trace and passes
the instantaneous mixed gain to the Weyl sampler. Its returned cumulative
weight and local trace have the distinct normalization and units derived above.
The Hamiltonian, gain, and loss signs and all factors match. There is no
discarded outgoing coherence or product of independent unconditional marginals.

The budget estimates the square of the full augmented dimension, including
the outgoing density. Its initial refusal precedes complex conversion,
Kronecker construction and eigensolver validation. Further supplied jump
storage is included before complex conversion. This is a deliberately dense
bounded reference; the stated estimate is an admission policy, not measured
peak RSS. Hermiticity and positivity checks use dimension-scaled machine
epsilon; these are numerical tolerances, not source uncertainty or fitted
physics agreement. Subunit initial trace and zero weights remain explicit.

Implementation review initially found a numerical refusal gap: finite jump
entries can overflow the Gram products, and finite Hamiltonian entries divided
by a very small positive supplied scale can overflow the generator. A prepared
model must refuse a nonfinite assembled generator. The task owner added
floating-point overflow/invalid-operation refusal around Gram products and
evolution, and explicit finite checks on the assembled generator and local
gain. The verifier inspected those changes; the finding is resolved.

No numerical reference table, transcribed physical constant, accelerator
comparison or external-code parity claim belongs to this finite algebra.
The citation equation numbers agree with the source; the conversion from time
to supplied per-depth jumps is an explicit assumption, not a fitted parameter.

## Regression evidence reviewed

The verifier independently inspected
`tests/montecarlo/test_channeling_axial_jump_instrument.py`. Its competing-loss
exponential pins incident normalization and separate unresolved accounting.
The retained decay/recycling example would fail if retained jumps were treated
as escape. The coherent analytic integral pins the commutator sign and
preservation of outgoing off-diagonal entries. The independent DOP853 oracle
uses direct matrix products in row-major order, instead of the implementation's
column-major Kronecker construction; it includes complex Hamiltonian terms,
retained recycling, two distinct rectangular outgoing channels and a complex
noncommuting unresolved effect. It checks all augmented components and closure.

The scalar inverse-CDF test fixes the depth law against its closed form.
Recorded conditional arrays in the coherent sampler test directly compare
the instantaneous Fourier carriers at two distinct depths; their resolved
difference would expose substitution of the integrated phase-space marginal.
Incident-addressed streams, zero density, zero depth, invalid data and budget
refusals are also tested. The $10^{-13}$-level ODE agreement threshold is
consistent with its explicitly tighter integration settings; analytic checks
use small-matrix floating-point error scales. The depth test uses the declared
absolute root tolerance. None is tuned to empirical crystal observations.

The task owner reported 20 focused tests passing in 1.00 s on 2026-10-26.
Execution of the added overflow regressions and documentation/render checks
belongs to the task owner and is separate from this source-to-code verdict.

- **Claim**: `cr-axial-absorbing-jump-instrument` —
  `src/pyrite/montecarlo/channeling/axial_jump_instrument.py::{absorbing_axial_model,AbsorbingAxialModel.at_depth,sample_absorbing_outgoing}` —
  Castin, Dalibard and Mølmer Eqs. (2), (4), (8), fixed absorbing compression,
  and finite Weyl resolution of identity.
- **Filters**: units pass; limits pass; signs/conventions pass.
- **Re-derivation**: matches — retained recycling, column-major conjugations,
  integrated positive gain, unresolved trace row and correlated conditional
  depth/phase-space law agree exactly at the supplied finite-model boundary.
- **Verdict**: rederived.
- **Write-up**: `docs/validation/channeling/cr-axial-absorbing-jump-instrument.md`.
- **Suggested ledger change**: task owner may apply rederived, then anchored
  once the focused regression result including overflow checks is recorded.
  No physical handoff claim advances; human sign-off remains in #277.

Task-owner regression record (2026-10-26): all 23 focused tests pass,
including the finite-input overflow refusals, caller-input isolation and an
equal-loss/different-outgoing-momentum counterexample. The latter keeps the
same depth law and cumulative weight while the full outgoing jump map selects
a different Weyl momentum mode.

## Task-owner implementation note (2026-10-31)

Not part of the verdict above. After review of PR #876 the implementation was
restructured: the augmented state is now $(\rho,\kappa X)$ with
$X(z)=\int_0^z\rho(s)\,ds$ and $\kappa=\lVert\Gamma\rVert_2$, so the
generator has dimension $2n^2$ rather than $n^2+m^2+1$. Tallies are
$Q_F=\sum_bB_bXB_b^\dagger$ and $U_O=\operatorname{tr}(E_OX)$, linear
images of the same integral, so the closure argument above carries over.
Densities are Hermitized and eigenvalues within the propagator error bound
are clipped. The verdict above applies to the previous layout; the
re-verification of this form follows.


## Fresh-context re-verification of the occupation form (2026-10-31)

Independent verifier, commit `76f86dc08`. The derivation below was written
from the ledger row and research note before the new implementation body was
read.

### Occupation-integral representation

Let $\mathcal L$ be the retained generator and $X(z)=\int_0^z\rho(s)\,ds$.
Each tally is the depth integral of a map that is linear in $\rho$, so it
commutes with the integral:

$$
Q_F(z)=\int_0^z\sum_bB_b\rho(s)B_b^\dagger ds=\sum_bB_bX(z)B_b^\dagger,
\qquad
U_O(z)=\operatorname{tr}\bigl(E_OX(z)\bigr).
$$

The cyclic trace gives $\operatorname{tr}Q_F=\operatorname{tr}(\Gamma_FX)$
with $\Gamma_F=\sum_bB_b^\dagger B_b$. Retained jumps cancel in the trace
derivative,
$\partial_z\operatorname{tr}\rho=\operatorname{tr}\sum_aA_a^\dagger A_a\rho-\operatorname{tr}\Gamma\rho
=-\operatorname{tr}(\Gamma_F\rho)-\operatorname{tr}(E_O\rho)$. Integrating gives

$$
\operatorname{tr}\rho(z)+\operatorname{tr}Q_F(z)+U_O(z)=\operatorname{tr}\rho(0),
$$

for any $\operatorname{tr}\rho(0)\le1$. The sampled conditional law remains
$\operatorname{tr}\sigma_F(z)\,dz/\operatorname{tr}Q_F(Z)$ for depth, followed by
the Weyl measurement of the instantaneous $\sigma_F(z)/\operatorname{tr}\sigma_F(z)$.

### Scaled augmented system

With $Y=\kappa X$,
$\partial_z(\rho,Y)=(\mathcal L\rho,\kappa\rho)$ and $(\rho,Y)(0)=(\rho_0,0)$.
The exact block exponential is

$$
\exp\!\left[z\begin{pmatrix}\mathcal L&0\\\kappa I&0\end{pmatrix}\right]
=\begin{pmatrix}e^{z\mathcal L}&0\\\kappa\int_0^ze^{s\mathcal L}ds&I\end{pmatrix}.
$$

For every $\kappa>0$, $X=Y/\kappa$ is therefore exact. Here $\kappa$ has
units Å$^{-1}$, which makes $Y$ dimensionless, like $\rho$. Changing the
occupation block's units only conditions the dense exponential. Since
$\operatorname{tr}(\Gamma X)=\operatorname{tr}\rho_0-\operatorname{tr}\rho$ and
$\Gamma\preceq\kappa I$ for $\kappa=\lVert\Gamma\rVert_2$, $Y$ is bounded by
the absorbed weight in the directions that carry loss. It can grow only
along loss-free directions, whose tallies are zero.

### Clipping cannot add bias beyond the propagation error

For Hermitian matrices, eigenvalue clipping is the Frobenius-nearest-point
projection onto the convex positive-semidefinite cone. Hermitization is the
orthogonal projection onto Hermitian matrices. Both projections are
nonexpansive, and the exact $\rho(z)$ and $X(z)$ lie in both sets. Hence, for
the computed $\tilde M$,

$$
\lVert P(\tilde M)-M\rVert_F\le\lVert\tilde M-M\rVert_F.
$$

Clipping and Hermitization therefore never move a density farther from the
exact value than the raw exponential already did. The trace-level change is at
most $\sqrt n\,\lVert\tilde M-M\rVert_F$. The tolerance
$64\epsilon n(1+\lVert G\rVert_1z)\operatorname{tr}\rho_0$ decides only
whether to refuse. It is not a bias the code injects. The CDF and the
instantaneous gain inherit only the realized propagation error, mapped through
$\Gamma_F$ and $B_b$.

The algorithm does **not** guarantee that the window weight
$\operatorname{tr}Q_F(Z)$ exceeds that worst-case bound. The sampler refuses
only when $\operatorname{tr}Q_F(Z)\le0$. In this verifier's probes, the
realized error stayed relative to $\operatorname{tr}Q_F(Z)$, rather than near
the absolute bound: $Z=10^6$ Å, phase about $10^4$, losses scaled by $10^{-16}$,
$\operatorname{tr}Q_F=3.5\times10^{-10}$, error relative to the reference
$3\times10^{-10}$, and the stated bound $n\cdot$tol was $9\times10^{-9}$. This
is an observation, not a discrepancy. A caller that needs a certified
conditional law at such weights cannot obtain the certificate from the
refusal tolerance alone.

### Code comparison

- Column-major $\operatorname{vec}(AXB)=(B^{\mathsf T}\otimes A)\operatorname{vec}X$:
  the commutator is $-i(I\otimes H-H^{\mathsf T}\otimes I)/\hbar v$, the
  anticommutator is $I\otimes\Gamma+\Gamma^{\mathsf T}\otimes I$, and recycling
  is $\bar A\otimes A$. All match the code. The `order="F"` reshape is
  consistent.
- $\kappa I$ fills the lower-left block, and the occupation block is divided
  by $\kappa$ after clipping. $\kappa=1$ when $\Gamma=0$ is harmless, because
  every tally is then zero.
- `_cumulative_outgoing` uses $\operatorname{tr}(\Gamma_FX)$, which equals
  $\operatorname{tr}Q_F$. `at_depth` forms $Q_F$, $\sigma_F$ and $U_O$ exactly
  as derived. Units are Å$^{-1/2}$ jumps, $X$ in Å, $\sigma_F$ in Å$^{-1}$, and
  dimensionless $Q_F$ and $U_O$.
- Root bracket: $X(0)=0$ exactly, so $C(0)-u=-u\le0$, and $C(Z)-u>0$. The
  rate floor $10^{-10}\,\operatorname{tr}Q_F(Z)/Z$ matches the research note.

### Independent numerical evidence

The verifier's scratch script
(`verify_jump.py`, not committed) uses two references. One is the
eigendecomposition of a row-major, probe-built Liouvillian with analytic
$(e^{\lambda z}-1)/\lambda$ occupation. The other is DOP853 integration of
$\rho$ together with the quadrature of $\sigma_F$ and
$\operatorname{tr}E_O\rho$. The model has $n=4$, a $3\times3$ outgoing grid, two
rectangular outgoing jumps, one retained jump and a complex $E_O$.

| case | max $\lvert\Delta\rho\rvert$ | max $\lvert\Delta Q_F\rvert$ | $\lvert\Delta U_O\rvert$ | closure |
|---|---|---|---|---|
| phase $10^4$, $Z=10^4$ Å (vs. DOP853) | $1.3\times10^{-15}$ | $1.2\times10^{-15}$ | $2.5\times10^{-15}$ | $7\times10^{-15}$ |
| weak loss $10^{-8}$ Å$^{-1}$, $Z=10^6$ Å, phase $10^4$ (vs. eigen) | $4.3\times10^{-12}$ | $5.5\times10^{-15}$ | $1.9\times10^{-14}$ | $5\times10^{-13}$ |
| strong loss, $Z=30$ Å (vs. DOP853) | $2.5\times10^{-15}$ | $4.5\times10^{-16}$ | $1.6\times10^{-15}$ | $<10^{-16}$ |

The residual $\rho$ error in the weak-loss case is consistent with
$\epsilon\times$ the phase $10^4$ in either the reference or the
implementation. Sampling 400 events per case with a delta window gave depth
Kolmogorov–Smirnov $p=0.81$ and $0.81$, and position-marginal $\chi^2$
$p=0.22$ and $0.15$, against $\operatorname{tr}(\Gamma_FX(z))$ and
$\operatorname{diag}Q_F(Z)$ from the eigen reference. The focused file
`tests/montecarlo/test_channeling_axial_jump_instrument.py` passed 29/29.

- **Claim**: `cr-axial-absorbing-jump-instrument` —
  `src/pyrite/montecarlo/channeling/axial_jump_instrument.py::{absorbing_axial_model,AbsorbingAxialModel.at_depth,sample_absorbing_outgoing}` —
  Castin, Dalibard and Mølmer Eqs. (2), (4), (8), restricted to the fixed
  absorbing model.
- **Filters**: units pass; limits pass; signs/conventions pass.
- **Re-derivation**: matches. The occupation-integral tallies, $\kappa$
  placement and unscaling, column-major conjugations, closure and conditional
  law agree with the code.
- **Verdict**: rederived.
- **Write-up**: `docs/validation/channeling/cr-axial-absorbing-jump-instrument.md`.
- **Suggested ledger change**: keep `anchored`. Replace "re-verification
  pending" with this dated `rederived` re-verification. A guard or documented
  caveat for window weights below the propagation bound is optional. Human
  sign-off remains in #277.
