# Folded planar spatial density

Validation: `cr-planar-spatial-density-reconstruction`.

Intended owner: `src/pyrite/montecarlo/channeling/spatial.py::spatial_density`.
Independent derivation recorded on 2026-10-17 before inspecting the
implementation body. Implementation comparison followed that derivation.

## Source and contract

[Tokmakoff, MIT OCW 5.74, Spring 2009, Lecture 12, The Density Matrix](https://ocw.mit.edu/courses/5-74-introductory-quantum-mechanics-ii-spring-2009/resources/mit5_74s09_lec12/),
Eqs. (1.6), (1.7), and (1.9) give the density-operator expansion and
expectation-value trace. The Bloch plane-wave convention is the one attributed
to Andersen et al., Phys. Scr. **28**, 308 (1983), and Azadegan (2013),
Eqs. 2–6, in the existing `cr-transverse-eigenproblem` ledger row. The new
claim is the position-projector expectation and folding normalization; it does
not independently revalidate the transverse eigenproblem.

The intended signature is
`spatial_density(x_ang, density, coefficients, orders, kappa_inv_ang, period_ang)`.
The block matrix `density[r,n,m]` is the dimensionless retained density operator
in the band basis; `coefficients[r,h,n]` holds orthonormal eigenvector columns.
The integer `orders[h]` are distinct plane-wave orders. The input position and
period are in angstrom, and `kappa_inv_ang[r]` is in inverse angstrom. All Bloch
blocks already carry their incident-electron ensemble weights: no additional
average over blocks is permitted. Missing trace represents lost particles.
The output is a folded density per incident electron, in inverse angstrom.

## Independent derivation

Write the period as $d$, reciprocal wave number as $g=2\pi/d$, and supercell
length as $L=N d$. Orthonormal supercell Bloch states have position amplitudes

$$
\Psi_{rn}(x)=\frac{1}{\sqrt{L}}\sum_h C_{rhn}
\exp\!\left[i(\kappa_r+h g)x\right].
$$

Here $\kappa_r=r g/N$ modulo the reciprocal lattice. A translation through an
integer number $j$ of cells multiplies the state by
$\exp(i\kappa_r j d)$. Uniform translation averaging removes cross-block terms
because

$$
\frac{1}{N}\sum_{j=0}^{N-1}
\exp\!\left[i(\kappa_r-\kappa_s)j d\right]=\delta_{rs}.
$$

This removal requires distinct Bloch sectors modulo $g$; it does not remove
coherences between bands within the same block. The supplied block-diagonal
operator is

$$
\hat\rho=\sum_{r,n,m}\rho_{rnm}|rn\rangle\langle rm|.
$$

The Born-rule density on the supercell is
$P_L(x)=\langle x|\hat\rho|x\rangle$.
Folding all $N$ cells gives $p(x)=\sum_{j=0}^{N-1}P_L(x+j d)=N P_L(x)$.
Consequently define the cell-normalized amplitudes

$$
\phi_{rn}(x)=\sqrt{N}\Psi_{rn}(x)
=\frac{1}{\sqrt d}\sum_h C_{rhn}
\exp\!\left[i(\kappa_r+h g)x\right],
$$

and obtain

$$
\boxed{p(x)=\sum_{r,n,m}\rho_{rnm}\phi_{rn}(x)\phi_{rm}(x)^*.}
$$

The conjugate belongs on the second band amplitude. Equivalently, if
$R_r=C_r\rho_r C_r^\dagger$, then

$$
p(x)=\frac{1}{d}\sum_{r,h,\ell}(R_r)_{h\ell}
\exp\!\left[i(h-\ell)g x\right].
$$

The common Bloch phase cancels within each block: $p(x)$ is periodic in $d$
and independent of an isolated change in $\kappa_r$ when $C_r$ and $\rho_r$
are held fixed. Physical dependence on Bloch momentum still enters those
coefficients and density matrices.

Distinct integer Fourier orders and $C_r^\dagger C_r=I$ imply

$$
\int_a^{a+d}\phi_{rn}(x)\phi_{rm}(x)^*\,dx=\delta_{nm},
\qquad
\int_a^{a+d}p(x)\,dx=\sum_r\operatorname{Tr}\rho_r.
$$

Thus the required scale is $1/d$, not $1/L$, and neither the number of Bloch
blocks nor survival probability divides the result. A finite retained band
subspace is sufficient provided its columns remain orthonormal.

## Filters and limiting cases

- Units: exponent dimensionless; amplitudes in inverse square-root angstrom;
  $p$ in inverse angstrom; its cell integral dimensionless.
- Zero retained operator gives zero density. Multiplying every block by a
  survival factor multiplies the density by the same factor.
- A single plane wave of weight $w$ gives $p=w/d$, for any Bloch momentum.
- A pure block $\rho_{nm}=w c_n c_m^*$ gives
  $p=w|\sum_n c_n\phi_n|^2$. A diagonal mixture gives
  $p=\sum_n w_n|\phi_n|^2$. A positive semidefinite operator therefore gives
  real nonnegative density, up to floating-point roundoff.
- Conjugation-sensitive example: take two plane-wave columns with orders
  $0,1$ and $c=(1,i)/\sqrt2$. Then
  $p(x)=[1-\sin(gx)]/d$. Transposing the density matrix or conjugating the
  first amplitude instead gives the incorrect opposite sign. At $x=d/4$
  the correct density is zero; at $x=3d/4$ it is $2/d$.
- Equal incoherent populations of those two waves give the uniform $1/d$.
  A real equal superposition gives $[1+\cos(gx)]/d$.
- Adding weighted Bloch blocks adds densities. It cannot create interference
  between distinct blocks in the translation-averaged contract.

## Implementation comparison

The implementation uses positive-phase plane-wave synthesis, right multiplies
the eigenvector columns, and divides the amplitude by the square root of the
cell period. Its contraction is literally

```text
np.einsum("xn,nm,xm->x", psi, rho[block], psi.conj(), optimize=True).real
```

This matches the derived band-index order and conjugation exactly. Summing
blocks without an extra divisor preserves incident-electron weights. Taking
the real part removes roundoff for a Hermitian input; the physical contract
requires a positive semidefinite Hermitian density operator and orthonormal
coefficient columns, which the helper assumes rather than tests numerically.
No clipping or survival normalization is introduced.

The optional sampling hook in `density.py::DensityMatrixSolver.propagate`
calls the helper from its depth-recording closure using the current `rho` and
`self.basis.vectors`. Recording follows energy reprojection, so the matrix and
coefficients share the same instantaneous basis. Only reconstructed scalar
densities are stored, rather than later applying the final basis to earlier
band matrices. This inspection checks the spatial sampling contract, not the
separate physical validity of the propagator.

Independent numerical probes used a uniform 1024-point periodic grid with
$d=2.7$ angstrom and explicit analytic or pure-wavefunction reference values:

- The complex two-wave sine example agrees to $7.78\times10^{-16}$ inverse
  angstrom; its periodic integral is $0.9999999999999996$.
- Two otherwise identical Bloch blocks with weights $0.2$ and $0.3$ integrate
  to $0.4999999999999997$; the difference from half the unit-weight density is
  at most $2.78\times10^{-16}$ inverse angstrom.
- An independent random complex orthonormal five-wave, three-band basis and
  a complex pure state of weight $0.4$ agree with an explicit coherent
  wavefunction sum to $3.34\times10^{-16}$ inverse angstrom; the cell integral
  is $0.4$. This checks complex coefficient columns as well as band coherences.

The periodic-grid integral uses the grid mean times the period; no duplicated
endpoint or trapezoidal endpoint correction is involved. These probes validate
the equation directly; durable regression-test ownership belongs to the task
owner.

## Adjudication

Units, limiting cases, and signs/conventions pass. Independent re-derivation
matches the implementation, including the cell/supercell factor and the
complex-coherence sign. Verdict: `rederived`. The task owner may set the ledger
row to `rederived`, or `anchored` after linking passing regression anchors.
Human sign-off is outside this verification.
