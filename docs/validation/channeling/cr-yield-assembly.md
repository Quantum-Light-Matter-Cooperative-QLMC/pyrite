# `cr-yield-assembly` re-derivation

Claim: `montecarlo/channeling/emission.py::{EmissionLines.rates,channeling_spectrum}`. Sources: walkthrough §6.4, proposal §3 D5 option A, and `self-absorption` (`_stack_tau`).

## Independent derivation (before reading the bodies)

An electron at depth $z$ in transverse state $\rho(z)$ emits line $c$ at the path rate $\langle d_c^\dagger d_c\rangle_{\rho(z)}$ [Å⁻¹] over $4\pi$ (`cr-dipole-rate`). Its photons are distributed in angle by the normalized lab pattern $P(\hat{\mathbf n}_e)$ [sr⁻¹] about the electron velocity (`cr-boosted-dipole-angular`). In energy they follow the unit-area profile $\mathcal L_c(E-E_c(\theta_e))$ [eV⁻¹] (`cr-line-shape`). A photon leaving depth $z$ along lab $\hat{\mathbf n}$ toward the exit face $z=t$ crosses $(t-z)/n_z$ of material. Its survival is $e^{-\mu(E)(t-z)/n_z}$, or the layer sum of `_stack_tau`. Per incident electron,

$$
\boxed{\frac{d^2N}{dE\,d\Omega}=\sum_c\int_0^t dz\,\langle d_c^\dagger d_c\rangle_{\rho(z)}\,P(\theta_e,\phi_e)\,\mathcal L_c\bigl(E-E_c(\theta_e)\bigr)\,e^{-\tau(z,E,\hat{\mathbf n})}}
$$

averaged over the detector solid angle and the beam's in-plane direction spread. The units are Å⁻¹·sr⁻¹·eV⁻¹·Å, i.e. photons/eV/sr/electron. With no absorption, the integral over $E$ and $4\pi$ is $\sum_c\int dz\,\langle d_c^\dagger d_c\rangle$, the line yield.

$\rho(z)$ must be the solution from the entry state. The estimator is linear in $\rho$, so evaluating it on the ensemble density is unbiased. Zero bound states give no bound→bound pairs, an empty sum, and an identically zero spectrum, so free→free terms are never counted. An in-plane tilt $\psi$ rotates the electron velocity about $\hat{\mathbf x}$. The photon angles must then be taken relative to $(0,\sin\psi,\cos\psi)$ with azimuth from $\hat{\mathbf x}$, which keeps the plane normal fixed.

## Independent numerical checks

**Full assembly.** I assembled my own spectrum and compared it with `channeling_spectrum`. Configuration: Si(110), 17 MeV, TDS + electronic, $n_{\rm cells}=4$, $H=8$, margin 5 eV, 10 µm, entry divergence 1 mrad. Detector at $\theta=0.6/\gamma$, $\phi=0.3$, cone half angle $0.8/\gamma$, in-plane divergence $0.5/\gamma$, energy spread 0.4%, absorption on, grid 4–40 keV.

My assembly takes its own depth history from `DensityMatrixSolver.propagate`. It applies its own attenuation, `linear_attenuation_inv_mm` with path $(t-z)/n_z$. It averages over an equal-area 24×48 cone grid and an 81-node Gaussian in-plane grid, with independently computed electron-frame angles. Its Voigt comes from SciPy and its slope $s$ from the formula verified in `cr-line-shape`.

| quantity | independent | implementation (default quadrature) | implementation ($n_{\rm aperture}=16$, $n_{\rm divergence}=31$) |
| --- | --- | --- | --- |
| line rates on the history | $P_i\,dN/dz$ | `line_rates` | max rel. diff $3.5\times10^{-16}$ |
| $\int dE$ of the spectrum | $7.575897\times10^{-3}$ sr⁻¹ | $7.574703\times10^{-3}$ ($-1.6\times10^{-4}$) | $-1.8\times10^{-4}$ |
| max pointwise difference / peak | — | $1.9\times10^{-2}$ | $8.1\times10^{-4}$ |

The default orders (6 Gauss–Legendre radii × 12 azimuths, 9 Gauss–Hermite) leave a 2% node ripple on the line flanks when the cone and divergence Doppler spread exceed the Lorentzian width. Raising the orders converges to the independent result. This is a quadrature-order caveat for users, not a formula error. Total unabsorbed line yield is $6.72\times10^{-5}$ photons/electron for this 10 µm case.

**Zero bound states.** A flat potential with a 50 eV margin gives an empty catalog and a spectrum and yield that are exactly 0 (test reproduced green).

**Gouanère peaks.** The diamond (110) 80.23 MeV test is green. I could not check the reference values (313.4/213.9/164.6/130.1 keV, Gouanère et al. 1988 Table II col. 10) against the paper: no Zotero tools were available in this session, so they are taken as transcribed.

## Diff against the implementation

- `EmissionLines.rates` computes $\sum$ `amp[a]*conj(amp[b])*rho[upper_a, upper_b]` over members sharing `lower`, which equals $\sum_f\sum_{ii'}a_{fi}a_{fi'}^*\rho_{ii'}$. It matches.
- `_depth_history` chains `solver.propagate(seg, step, initial=(rho, entry))` from `solver.entry_state` and samples at every segment end. The trapezoid rule is applied in `channeling_spectrum`. It matches.
- The electron-frame rotation is `ye = n_y cos s - n_z sin s`, `ze = n_y sin s + n_z cos s` and `phi_e = atan2(ye, n_x)`, which matches the rotation about $\hat{\mathbf x}$.
- Absorption is `_stack_tau(layers, z, n_z, E_c)`, which for $n_z>0$ spans $[z,t]$ with factor $1/n_z$ (`_layer_dz`). One stated simplification: $\mu$ is evaluated at the **line centre** $E_c$ rather than at each $E$ of the profile (`emission.py:842–846`). The error is $O(\Delta\mu/\mu\cdot\tau)$ across a line. At 18.5 keV in 20 µm Si, $\tau\approx0.04$ and the effect is below 1%. For thick absorbers or lines near an absorption edge it is larger. The claim text says "evaluated at each line centre", so this is documented.
- The pattern and the Doppler factor use the mean channel axis (or the tilted in-plane direction), not the instantaneous transverse velocity $\lesssim\theta_L$. This is a documented $O(\gamma\theta_L)$ geometric approximation.

## Verdict

`rederived`. Note in the ledger that peak heights inherit the `cr-line-shape` width discrepancy, though integrated yields do not.

## Resolution (worker, after the verdict)

- The peak-height inheritance from `cr-line-shape` is fixed with the all-block coherence-transfer widths. The quadrature-ripple caveat is now recorded in the ledger Notes and in the `n_aperture`/`n_divergence` parameter description.
- The spectrum is now assembled from the group modes, and capture is reported as a separate component.
- The integral and absorption anchors were re-pinned against the mode form. Each holds to 1e-6 and 1e-10 respectively.
