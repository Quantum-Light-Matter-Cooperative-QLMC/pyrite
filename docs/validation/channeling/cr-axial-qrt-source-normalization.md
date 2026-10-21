# On-axis finite-current photon normalization

## Independent derivation before implementation

Claim: `cr-axial-qrt-source-normalization`. Root derived this normalization
before inspecting the new source wrapper. The claim applies to a supplied
finite transverse Hamiltonian and a declared rotating-wave current. It does
not establish a complete material source, physical escape, or standard runs.

Source convention: first-order velocity-coupling emission and paraxial
longitudinal balance as recorded in `cr-dipole-rate`; the channeling Hamiltonian
and radiation interaction are given in J. U. Andersen,
[Notes on Channeling, Eqs. (197)–(199), pp. 122–123](https://phys.au.dk/fileadmin/site_files/publikationer/Lecture_notes/JUA-Channeling20151012total-2.pdf).
Use dimensionless polarization current $A_\lambda=\boldsymbol\epsilon_\lambda^*
\cdot\mathbf v/c$, with the transverse photon phase included away from the
axis. Photon number is resolved per photon energy and solid angle.

Let $E$ be photon energy in eV, $v=\beta c$ the fixed longitudinal speed,
$h_v=\hbar v=\beta\hbar c$ in eV angstrom, and
$D=1-\beta$ on the forward axis. Evaluate the latter without subtracting
nearly equal numbers: $D=1/[\gamma^2(1+\beta)]$. The supplied correlation
uses depth and the convention

$$
G_\lambda(z,z')=
\langle A_\lambda^\dagger(z')A_\lambda(z)\rangle.
$$

Converting the vacuum first-order spectral density from angular frequency
and time to energy and depth gives

$$
\boxed{\frac{d^2N}{dE\,d\Omega}
=\frac{\alpha E}{4\pi^2h_v^2}
\sum_{\lambda=1}^2\int dz\,dz'\,
t(z,E)t(z',E)^*
e^{iED(z-z')/h_v}G_\lambda(z,z').}
$$

The two transverse polarizations are orthogonal photon final states; sum their
intensities, not their amplitudes. An explicit polarizer instead takes the
corresponding complex linear combination before forming the correlation.
$G$ is dimensionless, the double integral is angstrom squared, and the result
has units photons per eV per steradian per incident electron. Incident-density
normalization is already present in $G$; do not multiply survival again.

The factor $E$ remains the actual photon energy, not its line-centre value.
The reference may evaluate a finite positive energy grid; it must neither
extend its normalization claim to all frequencies nor silently renormalize
positive-frequency tails.

## Independent golden-rule and forward dipole checks

For a single transition of transverse gap $\Delta>0$, a stationary initial
population $p_i$ and no damping or absorption,

$$
G_\lambda(z,z')=p_i|A_{\lambda,fi}|^2e^{-i\Delta(z-z')/h_v}.
$$

The long-length distribution is
$\ell^{-1}|\int_0^\ell e^{i(ED-\Delta)z/h_v}dz|^2
\longrightarrow2\pi h_v\delta(ED-\Delta)$.
After integrating positive photon energy the directional rate is

$$
\frac{dN}{dz\,d\Omega}
=\frac{\alpha E_0}{2\pi h_vD}\,
p_i\sum_\lambda |A_{\lambda,fi}|^2,
\qquad E_0=\frac\Delta D.
$$

Since $E_0=\hbar\omega_0$ and $h_v=\hbar\beta c$, this equals the existing
generic golden-rule convention
$\alpha(\omega_0/c)\sum|A_{fi}|^2/(2\pi\beta D)$.
There is no extra factor of $\beta$, $\gamma$, $\hbar$, or Doppler power.

On axis the commutator gives
$\mathbf v_{fi}/c=-i\Delta\mathbf d_{fi}/(\hbar c)$.
Thus the rate becomes
$\alpha\Delta^3|\mathbf d_{fi}|^2/
(2\pi\beta(\hbar c)^3D^2)$, agreeing with the total dipole rate times the
forward angular pattern. This check independently pins both the polarization
sum and the photon Jacobian.

For a finite transparent window, the exact supplied-current result is

$$
\frac{d^2N}{dE\,d\Omega}
=\frac{\alpha E\ell^2}{4\pi^2h_v^2}
p_i\sum_\lambda |A_{\lambda,fi}|^2
\operatorname{sinc}^2\left[\frac{(ED-\Delta)\ell}{2h_v}\right].
$$

This analytic two-level expression is an independent numerical oracle. A
ground state gives zero only for the declared lowering/RWA current. Empty
density and zero depth give zero; density mixtures are linear.

## Global finite-grid current and scope

For periodic transverse coordinates and orthonormal discrete amplitudes,
construct $H$ using the full supplied finite grid. Fourier momentum operators
have eigenvalues $\hbar k_x,\hbar k_y$. Therefore the dimensionless velocity
currents are

$$
A_x=\frac{\hbar c}{\gamma m_ec^2}\,k_x,
\qquad A_y=\frac{\hbar c}{\gamma m_ec^2}\,k_y,
$$

transformed with the same unitary eigenvectors as $H$. The wave-number
operators, not the numbers $k_x,k_y$ alone, appear in position representation.
For the positive-Bohr/RWA version retain matrix element $(f,i)$ only when
$E_i>E_f$ by a declared degeneracy tolerance. Never sum amplitudes of
independent incident electrons, project degenerate multiplets onto an
arbitrary local orbital, or assert that a truncated global grid is physically
complete. All represented downward transitions may include free-to-free
radiation; combining them with another CBS component would need an explicit
no-double-counting contract.

Off-axis emission carries transverse photon momentum and changes the Bloch
sector. A single-periodic-Hamiltonian current correlation is insufficient
there. This on-axis reference must reject off-axis requests instead of
pretending the scalar Doppler substitution supplies that missing physics.

The current RWA, fixed-energy/paraxial motion, recoil omission, finite grid,
supplied Markov bath and energy band are declared approximations. Abrupt
rectangular observation windows can produce ultraviolet tails; the finite
reference does not promise a convergent full-positive-energy photon integral.
Actual material dissipation/carrier sectors, source separation, boundary
switching, attenuation/frame mapping and convergence remain separate gates.

## Source-to-code comparison

After completing the independent derivation, the verifier inspected
`axial_qrt_source.py` and the finite-grid/photon oracles. The global factory
constructs a unitary inverse-FFT plane-wave basis, forms the kinetic plus real
potential Hamiltonian, and transforms both transverse wave-number operators
with the same complete represented eigensystem. The dimensionless current
factor is exactly the derived HBARC divided by gamma times electron rest
energy. Its positive-Bohr mask has the documented energy-index ordering and
absolute tolerance. Incident amplitudes preserve the grid's area-weighted norm.

The spectrum wrapper sums diagonal x/y current correlation intensities
separately and multiplies by actual photon energy with the derived factor.
An initial direct one-minus-beta subtraction was flagged by the verifier and
replaced with the cancellation-free denominator. The revised high-gamma check
pins that correction. The caller's responsibility to match the supplied
correlation's hbar-v to kinetic energy is explicit; arbitrary supplied arrays
cannot be inferred or certified by this wrapper.

The finite-time two-level test independently pins the photon energy prefactor,
Doppler phase, incident normalization and polarization sum. The rest-frame
spontaneous-rate plus aberration oracle independently pins the long-window
normalization. Its resonance-window coverage is corrected analytically using
$(2/\pi)[\operatorname{Si}(2b)-\sin^2(b)/b]$, with $b=10$, so a loose
four-percent truncation tolerance cannot conceal normalization error. Dense
explicit Fourier sums separately check Hamiltonian and current construction;
other tests check free-state zero current, common-potential shift, mixed and
empty density, complex degenerate-state interference, budgets and off-axis
refusal. The owner reports eight focused tests passing before final formatting.

- **Claim**: `cr-axial-qrt-source-normalization` —
  `src/pyrite/montecarlo/channeling/axial_qrt_source.py::{global_axial_source,on_axis_photon_spectrum}` —
  first-order velocity coupling, finite-current correlation and independent
  forward golden-rule normalization above.
- **Filters**: units pass (dimensionless current and photons/eV/sr); limits pass
  (finite-window sinc, golden rule, free/empty state); signs/conventions pass
  (positive-Bohr mask, forward phase, incoherent orthogonal polarization sum).
- **Re-derivation**: matches — no extra beta, gamma, hbar, Doppler or polarization
  factor; global finite-grid transforms agree with the independent expression.
- **Verdict**: rederived, scoped to the supplied finite on-axis RWA reference.
- **Write-up**: `docs/validation/channeling/cr-axial-qrt-source-normalization.md`.
- **Suggested ledger change**: owner may apply rederived, then anchored when
  the reported finite-time/dense-grid/golden-rule tests and final docs checks
  pass. Keep actual material dissipation/carrier closure, energy-band/recoil
  adequacy, complete CR/CBS partition and production source outside this claim.
  No promotion of physical source/handoff claims or human sign-off.
