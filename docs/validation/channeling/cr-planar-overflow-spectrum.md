# `cr-planar-overflow-spectrum`: resolved finite momentum tail

## Independent derivation, before implementation inspection

Fresh-context verifier, 2026-10-20, issue #522. Scope: a diagnostic conditional
momentum measurement of a supplied positive scattering kernel, not an
irreversible escape model or a production transport change.

The parent supplied the intended signature
`overflow_spectrum(plane, kernel, *, reference_kernel)` and block-Bloch
conventions. Starting point is the phase-covariance jump map
$J(\rho)(x,x')=C(x,x')\rho(x,x')$, equivalently
$J(\rho)=\sum_\alpha L_\alpha\rho L_\alpha^\dagger$.
Its source convention is the ledgered `cr-lindblad-jump-operators`
(Yoshioka 1957; Hall and Hirsch 1965; Bird and King 1990). This review
conditions on that supplied kernel; it does not revalidate those scattering
cross sections.

Let incoming block $s$ have momentum $k_{sa}=(s/n+a)g$ and transfer block
$r$ have $q_{ri}=(r/n+i)g$, with $0\leq s,r<n$. Write
$c=\lfloor(s+r)/n\rfloor$ and $t=(s+r)\bmod n$. Multiplication of plane waves
gives the exact nonwrapped outgoing order $h=a+i+c$ in block $t$.
Consequently its diagonal jump weight is

$$
w_{th}=\sum_{\substack{s,r:\ (s+r)\bmod n=t}}\ \sum_{a,b}
 \rho_{s,ab}\,K_{r,h-c-a,h-c-b},
$$

where a kernel index outside its finite transfer support contributes zero.
The sum must retain incoming coherences: replacing $\rho_{ab}$ by its
diagonal loses interference between transfer amplitudes. No conjugate is
inserted in this contraction: the conjugation already occurs in
$K_{ij}=\sum_\alpha l_{\alpha i}l_{\alpha j}^*$ and
$\rho_{ab}=\sum_\nu p_\nu\psi_{\nu a}\psi_{\nu b}^*$.
Hermiticity makes $w$ real; positivity follows by expanding the two PSD
matrices into amplitudes and summing absolute squares.

The full-loss rate, represented gain, and finite tail are

$$
\ell=\operatorname{tr}(A\rho),\qquad
G_T=\sum_{t,h}w_{th},\qquad
O_T=\sum_{t,\lvert h\rvert>H}w_{th}.
$$

All have units Å$^{-1}$ for dimensionless incident density. The outgoing
support reaches sums of incoming and transfer orders, including the residue
carry; modulo wrapping that support would alias the tail into the particle
band. The signed paraxial angle is $\theta_{th}=\hbar g(t/n+h)/p$.

Let $G_{T,\rm in}$ be the gain inside the original particle band, and
$G_{{\rm ref},\rm in}$ the corresponding gain for the original transfer
kernel. Then

$$
\ell-G_{{\rm ref},\rm in}
=O_T+(\ell-G_T)+(G_{T,\rm in}-G_{{\rm ref},\rm in}).
$$

This is the decisive closure identity. The old overflow equals the resolved
physical tail only if both unresolved full-loss mass and the changed inband
gain are acceptably small. Expanding transfer support can add amplitude
landing inside the original particle band; therefore one cannot silently
identify the old loss residual with the tail. Finite projection can also
change interference, so monotonicity of individual weights with transfer
window is not assumed.

A conditional tail draw uses $w_{th}/O_T$ only after a declared closure test.
It is not valid to rescale $w$ to absorb $\ell-G_T$. With $O_T=0$ there is no
tail distribution to sample. Filters: zero kernel gives zero rates; a single
incoming momentum and single transfer yield their sum with unit residue
carry; a zero-transfer jump preserves the incoming momentum, including its
sign; finite PSD inputs produce nonnegative weights. These are analytic
reference cases independent of implementation helpers.


## Integrated jump-source quadrature, derived before observer inspection

At each Strang step let $\rho_*$ be the state after the first no-jump half
step. The existing second-order jump polynomial books the source

$$
R_* = \Delta z\,\rho_*+\frac{\Delta z^2}{2}J_S(\rho_*).
$$

Every jump flux channel is the corresponding linear functional applied to
$R_*$. If kernels are stored at entrance energy and the current scalar
rate multiplier is $b$, the appropriate source for those stored kernels is
$bR_*$, because $J_S$ in this expression already includes its current
multiplier. The second term therefore contains two scattering factors,
as required by the second-order polynomial. A basis transform into plane
waves before accumulating sources is necessary if energy reprojection
changes eigenvectors. Linearity then permits a single final contraction
with an expanded kernel. This reproduces the existing discrete flux
quadrature; it does not make that quadrature exact in depth or supply a
state-dependent outgoing energy/phase model.

Ordinary density input is dimensionless and gives rates in Å$^{-1}$.
The integrated source has units Å and gives dimensionless integrated jump
weights (expected counts), which can exceed one per incident electron because
retained particles can jump repeatedly. Only absorbing-channel fluxes have
the interpretation of mutually exclusive escape probabilities.
Those two uses must be distinguished even when the API field names contain
`rate` or `per_ang`. Constant-energy accumulation resolves an integrated
momentum marginal, not a depth/energy joint distribution. Sampling its tail
is conditional on the supplied integrated jump current; it does not establish
physical escape or make every counted jump an escaping particle.

## Implementation comparison and verdict

`_outgoing_diagonal` contracts opposite matrix diagonals with a linear
convolution; its `abs(lag) + carry` offset matches $h=a+i+c$. It allocates
one extra order for the residue carry, without circular wrapping. Its
coherence contraction matches the expression above, including complex
phases. `overflow_spectrum` separately contracts the reference kernel and
expanded kernel, retains the reference full-loss diagonal, and records both
inband gains. `sample_angles` tests unresolved mass, inband shift, and total
outside-versus-reference mismatch individually before drawing; it never
fills missing mass by normalization.

The observer in `DensityMatrixSolver.propagate` passes
`_sandwich(self.basis.vectors, dz * rho + 0.5 * dz**2 * first) * self._scale`
after computing `first` and before updating the state. This exactly matches
the independently derived integrated source, including the second power of
the rate multiplier in the second-order contribution. The callback receives
a newly allocated transformed source, so accumulation cannot mutate the
solver state. No callback means no changed default dynamics.

Analytic anchor checks use momentum translation with residue carry,
wavefunction polynomial multiplication with complex amplitudes,
state-dependent full loss, an omitted transfer returning inside the old
particle band, and refusal of an unresolved tail. In particular the
incoming amplitudes $(1,i)/\sqrt2$ and transfer amplitudes $(1,i)$ yield
total gain 3, rather than the diagonal-only value 2; this distinguishes the
required coherence convention. All six tests in
`tests/montecarlo/test_channeling_overflow.py` passed independently (0.73 s).

- **Claim:** `cr-planar-overflow-spectrum` —
  `montecarlo/channeling/overflow.py::{overflow_spectrum,_outgoing_diagonal,OverflowSpectrum.sample_angles}`
  and `montecarlo/channeling/density.py::DensityMatrixSolver.propagate` —
  Fourier transform of the supplied phase-covariance Lindblad jump map.
- **Filters:** units pass with the integrated-source unit qualification;
  limits pass; signs/conventions pass, including complex coherence and
  Bloch residue carry.
- **Re-derivation:** `matches` — no divergent factor, sign, or conjugation
  found in the finite convolution, closure identity, or observer quadrature.
- **Verdict:** `rederived` for the diagnostic instrument conditional on its
  supplied kernels. This does not validate kernel quadrature convergence,
  a physical escape criterion, a depth-angle joint distribution, or the
  paraxial approximation in arbitrarily large tail angles.
- **Suggested ledger change:** owner may record `rederived`, or `anchored`
  on the six analytic tests after its required checks pass. Scope the claim
  to the diagnostic finite momentum measure and closure refusal. No
  production overflow-angle replacement or human sign-off.

### Consequence for the legacy band-edge-angle interpretation

The closure identity and omitted-transfer counterexample disprove any
general inference that the whole legacy overflow residual occupies momenta
outside the original particle band. Thus its production band-edge angle is
a declared assignment, not a proved lower bound on that channel's angles.
Even a bound for an explicitly resolved tail must use its actual signed
Bloch-grid support, rather than assuming the centered expression
$(H+1/2)g\hbar/p$ applies to every residue. This scope correction leaves
the mathematical accounting and the diagnostic convolution verdicts intact.
