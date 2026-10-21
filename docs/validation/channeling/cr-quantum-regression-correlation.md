# Bounded quantum regression correlation

## Independent derivation

Claim: `cr-quantum-regression-correlation`. Root derived these equations before
reading the new implementation. This is a reference for a supplied finite
Markovian model, not a validated axial material model or photon source.

Source for operator ordering: Landi, Kewming, Mitchison and Potts,
[Current fluctuations in open quantum systems, Eq. (95) in arXiv v3](https://arxiv.org/pdf/2303.04270v3),
published in PRX Quantum 5, 020201 (2024). Their regression relation propagates
the operator product inserted at the earlier time, using the same linear
reduced propagator as the one-time state. Its use assumes a suitable Markovian
system/environment model; a one-time fitted semigroup alone does not establish
the validity of a material's two-time regression hypothesis.

Let $z$ be depth in angstrom, $h_v=\hbar v$ in eV angstrom, $H=H^\dagger$
in eV, and $L_\mu$ in inverse square-root angstrom. The generator is

$$
\mathcal L X=-\frac{i}{h_v}[H,X]
+\sum_\mu\left(L_\mu X L_\mu^\dagger
-\frac12\{L_\mu^\dagger L_\mu,X\}\right).
$$

Write $\Phi_s=\exp(s\mathcal L)$ and $\rho(z)=\Phi_z\rho_0$. For supplied
source operators $J_a$, define

$$
G_{ab}(z,z')=\langle J_b^\dagger(z')J_a(z)\rangle
=\operatorname{Tr}\left[J_a\Phi_{z-z'}
\big(\rho(z')J_b^\dagger\big)\right],\qquad z\ge z'.
$$

For the other triangle use
$G_{ab}(z,z')=G_{ba}(z',z)^*$. At equal depths this is
$\operatorname{Tr}(J_b^\dagger J_a\rho(z))$; off-diagonal component indices
need not be real. The auxiliary initial operator $\rho J_b^\dagger$ is
generally neither Hermitian nor normalized. Hermitizing it, projecting its
eigenvalues, or renormalizing its trace changes the desired correlation.

## Spectral quadratic form

Given nonnegative depth quadrature weights $w_i$, complex amplitudinal
transmission $t_i$, and a component analyzer $c_a$, set
$J=\sum_a c_a J_a$. The finite observation-window spectral form is

$$
I(\kappa)=\sum_{i,j,a,b}w_iw_j t_i t_j^*
e^{i\kappa(z_i-z_j)}c_a c_b^*G_{ab}(z_i,z_j).
$$

It is real and nonnegative up to numerical roundoff: it equals the expectation
of $A^\dagger A$, with
$A=\sum_i w_i t_i e^{i\kappa z_i}\sum_a c_a J_a(z_i)$.
Equivalently the block correlation is positive semidefinite and the expression
is its quadratic form with conjugated amplitude coefficients. Both depth and
component indices must be exchanged in the Hermitian extension.

If $J$ is dimensionless, $G$ is dimensionless and $I$ has units angstrom
squared. There is no photon-rate prefactor in this helper. Do not relabel it
photons/eV/sr. A putative straight constant-speed photon map uses
$\kappa=ED/h_v$, $D=1-\boldsymbol\beta\cdot\mathbf n$, but angular-current
normalization, frames, Doppler/recoil approximations and emitting subspace
selection remain separate claims.

Absorption acts at amplitude level. For intensity optical depth $\tau_i(E)$,
$t_i(E)=\exp[-\tau_i(E)/2]$ gives the factor
$\exp[-(\tau_i+\tau_j)/2]$ on a pair of depths. Multiplying a completed
coherent spectrum by an average intensity survival factor is generally a
different model. Complex analyzer coefficients must also be retained; summing
component intensities first discards polarization/azimuth interference.

## Independent analytic oracles

Use $H=\operatorname{diag}(0,\Delta)$, $J=|g\rangle\langle e|$,
$\rho_0=|e\rangle\langle e|$ and a single relaxation jump
$L=\sqrt\Gamma J$. Direct solution of the population and coherence equations
gives, for all depths,

$$
G(z,z')=e^{-\Gamma(z+z')/2}
e^{-i\Delta(z-z')/h_v}.
$$

This is a rank-one positive correlation kernel. With no relaxation and unit
transmission, continuous quadrature on $[0,\ell]$ gives

$$
I(\kappa)=\ell^2\operatorname{sinc}^2
\left[\frac{\ell}{2}(\kappa-\Delta/h_v)\right],
\qquad \operatorname{sinc}(x)=\frac{\sin x}{x}.
$$

The peak is at positive $\kappa=\Delta/h_v$. This pins the complex phase sign,
which an intensity-only or positivity test does not. Adding the pure-dephasing
jump $\sqrt{\eta/2}\,\sigma_z$ multiplies $G$ by
$\exp[-\eta|z-z'|]$. Starting in the ground state gives zero. Mixtures are
linear in $\rho_0$; changing source operators by complex linear combinations
must transform the full component tensor bilinearly.

Input density must be finite Hermitian positive semidefinite with declared
trace. This source definition need not condition on survival. Dense reference
work needs an explicit budget before constructing an $M^2$ by $M^2$
Liouvillian or a recorded-depth component tensor; it is not a scalable
replacement for the production MCWF solver.

## Source-to-code comparison

After completing the independent derivation, the verifier read
`quantum_regression.py` and its analytic oracle tests. `_generator` uses
column-major vectorization: the Hamiltonian commutator is
$-i(I\otimes  H-H^T\otimes  I)/h_v$, and each jump gain is
$L_\mu^*\otimes  L_\mu$. The transpose/conjugation pattern agrees with the
operator equation. `regression_correlation` propagates the full complex
auxiliary operator $\rho(z')J_b^\dagger$ and takes the unsymmetrized trace
with $J_a$. Its opposite triangle exchanges both depth and component indices
under conjugation. Equal-time cross components remain complex.

`spectral_quadratic_form` applies trapezoid depth weights and the positive
spectral phase at amplitude level; complex amplitudes enter once and their
conjugates enter at the opposite depth/component. It retains current cross
terms and checks imaginary/negative residuals against a magnitude-scaled
roundoff bound. The result has the documented current-squared angstrom-squared
units, without photon normalization. The dense generator/recording admission
precedes generator construction.

The independent tests use explicit two-level exponential formulas, zero-ground
emission, pure-dephasing dependence on delay, signed Bohr phase, analytic
finite-window sinc shape and analytic amplitude attenuation. A complex
component test exactly cancels two source components, which pins the analyzer
conjugation convention. The task owner reported all 14 tests passing; the owner
must preserve that exact test result in the anchor evidence.

- **Claim**: `cr-quantum-regression-correlation` —
  `src/pyrite/montecarlo/channeling/quantum_regression.py::{regression_correlation,spectral_quadratic_form}` —
  supplied finite GKSL generator and quantum regression ordering above.
- **Filters**: units pass (inverse-depth generator and current-squared depth-squared
  integral); limits pass (undamped sinc peak, damped two-level, ground/empty
  state); signs/conventions pass (positive Bohr peak, non-Hermitian intermediate,
  complex angular cross terms).
- **Re-derivation**: matches — Hamiltonian/jump vectorization, operator insertion,
  Hermitian extension and spectral amplitude weights match the independent
  equations without missing conjugates or normalization factors.
- **Verdict**: rederived, scoped to the supplied finite Markov reference model.
- **Write-up**: `docs/validation/channeling/cr-quantum-regression-correlation.md`.
- **Suggested ledger change**: owner may apply rederived and, with the reported
  passing analytic oracles, anchored. Do not change the physical axial source
  or handoff claim status; completeness, actual material jumps and photon units
  remain outside this verdict. Human sign-off is not applied.
