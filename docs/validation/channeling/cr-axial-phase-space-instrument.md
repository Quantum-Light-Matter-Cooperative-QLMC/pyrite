# Finite periodic phase-space instrument

## Independent derivation

Claim: `cr-axial-phase-space-instrument`. This derivation was completed by the
root verification context before reading `axial_phase_space.py`. It concerns
positive periodic-grid measurement probabilities, not physical dechanneling.

The continuum motivation is the coherent-state resolution of identity and
positive phase-space measurement described by
[Brody, Graefe and Melanathuru, Physical Review Letters 134, 120201 (2025)](https://doi.org/10.1103/PhysRevLett.134.120201).
That construction defines a measurement model. It does not establish that an
axial crystal implements that measurement. The finite construction below follows
directly from discrete Fourier orthogonality, without approximating a continuum
quadrature.

Let the transverse periodic grid have dimensions $n_x,n_y$ and Hilbert dimension
$M=n_xn_y$. States use the discrete inner product. Physical grid samples must
first be multiplied by the square root of the sample area. Take a normalized
window $g$, so $\sum_j |g_j|^2=1$, and define

$$
g_{a,b}(j)=g(j-a)\exp\left[2\pi i
\left(\frac{b_xj_x}{n_x}+\frac{b_yj_y}{n_y}\right)\right].
$$

Here every position and momentum index is periodic. Summing the matrix element
over $b$ gives $M\delta_{jk}$; summing $|g(j-a)|^2$ over $a$ gives one. Thus

$$
\frac{1}{M}\sum_{a,b}|g_{a,b}\rangle\langle g_{a,b}|=I.
$$

For a supplied, possibly subnormalized state $f$, the joint outcome weight is

$$
q(a,b)=\frac{1}{M}|\langle g_{a,b}|f\rangle|^2,
\qquad \sum_{a,b}q(a,b)=\langle f|f\rangle.
$$

Each weight is nonnegative and dimensionless. No survival normalization is
implicit. For a mixed state, replace the squared overlap by
$\langle g_{a,b}|\rho|g_{a,b}\rangle/M$; linearity then gives exactly the
ensemble mean of the pure-state probabilities.

The joint sampler need not materialize $M^2$ outcomes. Parseval's identity gives

$$
m(a)=\sum_b q(a,b)=\sum_j |g(j-a)|^2|f_j|^2.
$$

Sample $a$ with probability $m(a)/\|f\|^2$. At that position compute
$h_a(j)=g(j-a)^*f_j$. The ordinary unnormalized forward FFT has entries
$\langle g_{a,b}|f\rangle$, so the conditional momentum law is

$$
\Pr(b\mid a)=\frac{|\operatorname{FFT}(h_a)_b|^2}{M m(a)}.
$$

This uses $O(M)$ storage and preserves the joint position/momentum law; sampling
the two unconditional marginals independently generally does not. A complex
window requires the conjugation shown above. A zero state has zero weights and
no conditional sampling distribution.

## Retained/outgoing interpretation and limits

For orthogonal $P$, set $F=I-P$ and supply the outgoing state $f=F\psi$.
The retained map is $P\rho P$ and the outgoing effects are
$F|g_{a,b}\rangle\langle g_{a,b}|F/M$. Their total effect is $P+F=I$,
so this defines a completely positive quantum/classical instrument with exact
trace accounting. It removes retained/outgoing cross coherence as a measurement
does. Algebraic closure alone supplies no material-dependent escape law.

A delta window yields localized position and uniform momentum. A constant
window yields uniform position and the state's momentum distribution. Those
opposite limits expose the configurable resolution. A Gaussian window on a
finite torus is not automatically an exact continuum minimum-uncertainty packet.
Momentum indices use FFT order, and conversion to inverse angstrom, carrier
tilt, lab frames and on-shell directions lies outside this claim.

Physical launch needs independent bounds on discarded coherence, return flux,
window resolution, grid/supercell dependence, electronic jumps and paraxial
kinematics. This claim must not advance `cr-axial-zone-handoff-accounting`.

## Source-to-code comparison

The verifier inspected `axial_phase_space.py` after completing the derivation.
`phase_space_weights` uses the conjugated periodically translated window and
ordinary forward FFT, and divides squared amplitudes by the Hilbert dimension.
This matches the independently derived Fourier sign, complex conjugation and
normalization. `sample_phase_space` computes the position correlation with a
conjugated Fourier transform of the window intensity, then samples the
conditional FFT distribution. The correlation has the correct translation sign.
`state_weight` reports the supplied discrete squared norm rather than a physical
loss rate. The zero-state and allocation-limit conventions agree with scope.

The independent direct-exponential test oracle constructs every translated,
modulated complex window explicitly rather than calling FFT helpers. Tests also
compare both conditional sampling probability arrays against the direct joint
law, and cover mixed-state retained/outgoing trace closure, incident-addressed
measurement streams, zero states and resolution limits. Their execution and
final pass result must be recorded by the task owner before advancing the
ledger from rederived to anchored.

- **Claim**: `cr-axial-phase-space-instrument` —
  `src/pyrite/montecarlo/channeling/axial_phase_space.py::{phase_space_weights,sample_phase_space}` —
  finite Fourier orthogonality; continuum motivation above.
- **Filters**: units pass (dimensionless discrete amplitudes/probabilities);
  limits pass (delta and uniform windows, zero state); signs/conventions pass
  (forward FFT and conjugated complex window).
- **Re-derivation**: matches — identity factor is exactly one over the number
  of grid sites; supplied outgoing norm is preserved.
- **Verdict**: rederived, scoped to the finite periodic measurement.
- **Write-up**: `docs/validation/channeling/cr-axial-phase-space-instrument.md`.
- **Suggested ledger change**: task owner may apply rederived, then anchored
  after the independent direct-exponential regression passes. No change to
  physical handoff status. Human sign-off remains outside this verdict.

## Mixed-density extension: independent derivation (2026-10-26)

This fresh verification context derived the following before reading the new
implementation bodies. The source is the finite Weyl orthogonality above;
the continuum paper supplies motivation only. Let $\rho$ be a finite Hermitian
positive semidefinite density matrix in the discrete C-order position basis,
with possibly subunit trace $t$. A physical position representation must already
have been converted to this discrete orthonormal basis. No cell-area factor
belongs inside the measurement.

Set $w_a(j)=g(j-a)$ and
$\theta_b(j)=2\pi(b_xj_x/n_x+b_yj_y/n_y)$. The convention
$T_aM_bg$ differs from $M_bT_ag$ by a global phase, which cancels in every
effect. Direct expansion gives

$$
q(a,b)=\frac{1}{M}\sum_{j,k}
w_a(j)^*\rho_{jk}w_a(k)
\exp[-i\theta_b(j)+i\theta_b(k)].
$$

Each effect is positive. Finite Fourier orthogonality kills the off-diagonal
terms on summation over $b$, giving

$$
m(a)=\sum_bq(a,b)=\sum_j|w_a(j)|^2\rho_{jj},
\qquad \sum_a m(a)=t.
$$

Thus the position marginal depends only on the diagonal, whereas the conditional
momentum law must retain every off-diagonal entry. For $m(a)>0$ it is
$\Pr(b\mid a)=q(a,b)/m(a)$, with position sampled using $m(a)/t$.
The resulting joint sampling law is exactly $q(a,b)/t$, and the separately
reported input weight must be $t$, not one. Zero trace has identically zero
weights and no sampling law.

Define $H_a(j,k)=w_a(j)^*\rho_{jk}w_a(k)$. Apply the ordinary forward FFT
over the two $j$ axes and the ordinary inverse FFT over the two $k$ axes.
The inverse contributes $1/M$, so the matching momentum diagonal of this
four-axis transform is already $q(a,b)$; no further factor belongs there.
Equivalently, each diagonal is $[F H_a F^\dagger]_{bb}/M$ with the
unnormalized Fourier matrix $F$. This fixes both conjugations and signs.

For any ensemble $\rho=\sum_r p_r|f_r\rangle\langle f_r|$, linearity
gives $q_\rho=\sum_r p_r q_{f_r}$ without requiring a density eigendecomposition
or projecting away coherences. A delta window gives
$q(a,b)=\rho_{aa}/M$; a constant window gives position-independent
$q(a,b)=[F\rho F^\dagger]_{bb}/M^2$. The maximally mixed input
$\rho=tI/M$ gives $q(a,b)=t/M^2$ for every normalized complex window.
For a density operator, weights and traces are dimensionless; a supplied gain
density per unit depth instead transfers its inverse-length units to both
weights and trace. Such a trace is a rate weight, not an integrated event
probability. FFT indices are labels, not physical momenta or escape directions.
Dense input already requires $O(M^2)$ storage;
avoiding all positions at once can bound additional storage by $O(M^2)$.
Explicit resource refusals must precede dense validation workspaces and result
allocations. Resource policy does not alter the measurement law.

### Mixed-density source-to-code comparison

After the derivation, this verifier inspected `density_phase_space_weights`,
`sample_density_phase_space`, `_density_inputs` and
`_density_momentum_weights`. Localization conjugates the row window and leaves
the column window unconjugated, exactly as in $H_a$ above. The forward row FFT
and inverse column FFT produce the required matching diagonal with the inverse
FFT supplying $1/M$. Reshaping in C order agrees with the declared basis and
the output ordering. The full matrix enters each conditional transform, so no
position coherence is discarded or replaced with an independent momentum law.

The sampler scales by the input trace only to construct conditional
probabilities, samples the intensity-window position marginal, then samples
the full localized momentum distribution. It returns the original trace.
Window normalization, finite shape, Hermiticity and positivity checks agree
with the assumptions. Zero density has zero diagnostic weights and refuses
sampling before any random draw. Small negative numerical weights are clipped
under explicit machine-epsilon bounds; these are numerical error tolerances,
not empirical physics tolerances. This qualification limits exact positivity
and linearity statements to exact valid inputs, up to floating-point error.

The declared working-memory estimate is checked before complex density
conversion, eigensolver validation, and dense FFT/result allocations. The
density shape guard precedes complex conversion; nonfinite transform results
refuse. The estimate is an admission policy, not a measured peak-memory bound.
No density diagonalization is used to manufacture sampled ensemble members;
the eigensolver is used only to reject nonpositive inputs.

The direct regression oracle explicitly constructs the complex carrier modes
and evaluates the quadratic form without FFTs. It checks mixed-state linearity,
trace closure and sensitivity to off-diagonal coherence on a rectangular grid.
The sampler test checks both conditional arrays at every position rather than
relying on random histogram agreement. Delta, identity and zero-density limits,
incident stream order, invalid inputs and resource refusals are covered.
The focused phase-space and documentation test command passed all 22 tests on
2026-10-26. No external numerical table, transcription, accelerator-parity
artifact or fitted source tolerance is involved in this algebraic extension.

- **Claim**: `cr-axial-phase-space-instrument` —
  `src/pyrite/montecarlo/channeling/axial_phase_space.py::{density_phase_space_weights,sample_density_phase_space,_density_momentum_weights}` —
  finite Weyl orthogonality and the positive quadratic form above.
- **Filters**: units pass (discrete-basis density units propagate unchanged);
  limits pass (rank one, delta, identity and zero density); signs/conventions
  pass (forward row FFT, inverse column FFT, conjugated row window, C order).
- **Re-derivation**: matches — the inverse FFT supplies precisely $1/M$;
  both off-diagonal coherence and the original trace are preserved.
- **Verdict**: rederived for the mixed-density finite measurement extension.
- **Write-up**: `docs/validation/channeling/cr-axial-phase-space-instrument.md`.
- **Suggested ledger change**: task owner may extend the existing anchored row
  to the mixed-density owners and direct mixed-state/conditional anchors after
  recording the green tests. No physical escape or handoff status changes;
  human sign-off remains outside this verdict.

Task-owner rendered-document check (2026-10-26): the Sphinx HTML contains the
mixed-density expressions inside `class="math notranslate nohighlight"`
elements, including the quadratic form, conditional law and FFT factor.
