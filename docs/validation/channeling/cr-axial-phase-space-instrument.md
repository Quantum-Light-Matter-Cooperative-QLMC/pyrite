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
