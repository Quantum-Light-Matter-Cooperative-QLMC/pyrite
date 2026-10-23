# `optical-wavefront-propagation`

## Claim and source

- Claim: passive vector angular-spectrum propagation between reference planes, combined with explicitly modeled polarization/optical elements, can retain CTR phase in finite-distance collection where a far-field angular detector is insufficient.
- Code: `src/pyrite/montecarlo/spectrum/optical_wavefront.py::{propagate_vacuum, forward_power_w}`; `src/pyrite/montecarlo/spectrum/optical_finite_screen.py::propagate_screen_field`; `src/pyrite/montecarlo/spectrum/optical_finite_collection.py::collect_screen_field`; `src/pyrite/detectors/optical_elements.py::apply_optical_elements`; `src/pyrite/montecarlo/spectrum/optical_finite_convergence.py::finite_screen_collection_convergence`. Also checked: the paraxial handoff normalization in `src/pyrite/montecarlo/spectrum/optical_interferometer_handoff.py::coherency_from_screen_fields`.
- Sources: source-free vacuum Maxwell equations in angular-spectrum form ($\mathbf k\cdot\mathbf E=0$, $k_z=\sqrt{k_0^2-k_\perp^2}$, $\operatorname{Im}k_z\ge0$, time factor $e^{-i\omega t}$); Schmidt et al., arXiv:1803.00608, Appendix D (cited, **not consulted** here; see Caveats).
- Units: SI. Monochromatic phasor $\mathbf E$ in V/m (`OpticalWavefront`); two-sided temporal Fourier field $\widetilde{\mathbf E}(\omega)=\int\mathbf E(t)e^{+i\omega t}dt$ in V s/m (`ScreenSpectralField`); outputs W, J/eV and photons/eV.
- Frozen revision: commit `3a98f542` (fresh-context verifier, 2026-10-23).

The derivation below was written from the ledger row, the docstrings and the physics page [optical wavefront and detectors](../../physics/radiation-physics/optical-wavefront-and-detectors.md). The implementation bodies were read afterwards.

## Independent derivation

### Angular-spectrum propagator

In source-free vacuum with $e^{-i\omega t}$, every Cartesian field component obeys $(\nabla^2+k_0^2)\mathbf E=0$, $k_0=\omega/c=E/\hbar c$. A transverse Fourier mode $e^{i\mathbf k_\perp\cdot\mathbf r_\perp}$ has $z$-dependence $e^{\pm ik_zz}$ with $k_z^2=k_0^2-k_\perp^2$. The outgoing (downstream, $+z$) solution with no incoming wave from $z\to\infty$ is the root with $\operatorname{Re}k_z\ge0$ for $k_\perp<k_0$ (phase velocity $+z$ under $e^{-i\omega t}$) and $\operatorname{Im}k_z>0$ for $k_\perp>k_0$ (decay). Hence

$$
\widetilde E_{x,y}(\mathbf k_\perp,z+\Delta z)=H(\mathbf k_\perp,\Delta z)\,\widetilde E_{x,y}(\mathbf k_\perp,z),
\qquad H=e^{ik_z\Delta z}.
$$

Properties: $H(\cdot,0)=1$ (identity); $H(\Delta_1)H(\Delta_2)=H(\Delta_1+\Delta_2)$ (composition); $\lvert H\rvert=1$ for propagating modes; $\lvert H\rvert=e^{-\kappa\Delta z}$, $\kappa=\sqrt{k_\perp^2-k_0^2}$, for evanescent modes. For $\Delta z<0$ the evanescent gain $e^{+\kappa\lvert\Delta z\rvert}$ is unbounded, so refusing negative distance is the correct restriction. Only $E_x,E_y$ need be propagated: Gauss's law $\mathbf k\cdot\mathbf E=0$ fixes

$$
\widetilde E_z=-\frac{k_x\widetilde E_x+k_y\widetilde E_y}{k_z}.
$$

The same operator applies unchanged to a monochromatic phasor and to the temporal Fourier transform at one $\omega$, since both are solutions of the frequency-domain Helmholtz equation.

### Magnetic field (Faraday's law)

With $e^{-i\omega t}$, $\partial_t\to-i\omega$, so $\nabla\times\mathbf E=-\partial_t\mathbf B$ becomes $i\mathbf k\times\mathbf E=i\omega\mu_0\mathbf H$:

$$
\mathbf H=\frac{\mathbf k\times\mathbf E}{\omega\mu_0}=\frac{\mathbf k\times\mathbf E}{Z_0k_0},\qquad
H_x=\frac{k_yE_z-k_zE_y}{Z_0k_0},\quad H_y=\frac{k_zE_x-k_xE_z}{Z_0k_0},
$$

valid for complex $k_z$ (evanescent modes) as well.

### Normal Poynting flux per mode

For one mode, using $\mathbf k\cdot\mathbf E=0$,

$$
E_xH_y^*-E_yH_x^*=\frac{k_z^*\bigl(\lvert E_x\rvert^2+\lvert E_y\rvert^2\bigr)+k_z\lvert E_z\rvert^2}{Z_0k_0}.
$$

Propagating ($k_z$ real): $\operatorname{Re}=k_z\lvert\mathbf E\rvert^2/(Z_0k_0)>0$. Evanescent ($k_z=i\kappa$): the expression is purely imaginary, so a lone evanescent mode carries no net normal power. Over one periodic cell distinct $\mathbf k_\perp$ are orthogonal, so the cell-integrated flux is diagonal in modes. Every propagating mode keeps $\lvert H\rvert=1$, so the propagating flux is conserved between planes.

### Two normalizations and the factor of 2

*Monochromatic phasor.* For $\mathbf E(t)=\operatorname{Re}[\mathbf E e^{-i\omega t}]$, the time-averaged flux is $\langle S_z\rangle=\tfrac12\operatorname{Re}(\mathbf E\times\mathbf H^*)_z$. With discrete Parseval ($\int_{\rm cell}\lvert f\rvert^2dA=\frac{\Delta x\Delta y}{N_xN_y}\sum\lvert\hat f\rvert^2$, $\hat f$ the unnormalized DFT):

$$
P_+=\frac{\Delta x\Delta y}{2Z_0N_xN_y}\sum_{k_\perp<k_0}\frac{k_z}{k_0}
\bigl(\lvert\hat E_x\rvert^2+\lvert\hat E_y\rvert^2+\lvert\hat E_z\rvert^2\bigr)\quad[\mathrm W].
$$

*Two-sided spectral field.* For real fields, $\widetilde{\mathbf E}(-\omega)=\widetilde{\mathbf E}(\omega)^*$ and Parseval gives $\int f(t)g(t)\,dt=\frac1{2\pi}\int_{-\infty}^{\infty}\widetilde f\,\widetilde g^*\,d\omega$. Folding to $\omega>0$:

$$
U=\int dA\int dt\,(\mathbf E\times\mathbf H)_z=\int_0^\infty d\omega\,\frac1\pi\int dA\,\operatorname{Re}(\widetilde{\mathbf E}\times\widetilde{\mathbf H}^*)_z,
\qquad
\frac{dU}{dE}=\frac1{\pi\hbar}\int dA\,\operatorname{Re}\bigl[\widetilde E_x\widetilde H_y^*-\widetilde E_y\widetilde H_x^*\bigr].
$$

With $\hbar$ in J s this is J/J; with $\hbar$ in eV s it is J/eV. The photon density is $dN/dE=(dU/dE)/(E\,e)$ for $dU/dE$ in J/eV and $E$ in eV. In mode form,

$$
\frac{dU}{dE}=\frac{\Delta x\Delta y}{\pi\hbar Z_0N_xN_y}\sum_{k_\perp<k_0}\frac{k_z}{k_0}\lvert\hat{\widetilde{\mathbf E}}\rvert^2.
$$

*Bridge.* A quasi-monochromatic pulse $\mathbf E(t)=\operatorname{Re}[\mathbf E_0e^{-i\omega_0t}]g(t)$ has positive-frequency transform $\tfrac12\mathbf E_0\widetilde g(\omega-\omega_0)$. Then $\frac1\pi\int_0^\infty\frac14\lvert\mathbf E_0\rvert^2\lvert\widetilde g\rvert^2d\omega=\frac{\lvert\mathbf E_0\rvert^2}{4\pi}\cdot2\pi\int g^2dt=\frac{\lvert\mathbf E_0\rvert^2}{2}\int g^2dt$, the phasor result $\tfrac12\lvert\mathbf E_0\rvert^2$ times the duration. The $\tfrac12$ of the time average and the $1/\pi$ of the folded two-sided transform are therefore consistent and must not be mixed:

$$
\boxed{\;P_+:\ \frac{1}{2Z_0}\lvert\mathbf E\rvert^2\ \text{(phasor, V/m)},\qquad
\frac{dU}{d\omega}:\ \frac{1}{\pi Z_0}\lvert\widetilde{\mathbf E}\rvert^2\ \text{(two-sided, V s/m)}.\;}
$$

### s/p basis on propagating modes

For $\mathbf k=(k_x,k_y,k_z)$, $k_\perp>0$, $(c_x,c_y)=\mathbf k_\perp/k_\perp$, $\cos\theta=k_z/k_0$, $\sin\theta=k_\perp/k_0$:

$$
\hat{\mathbf s}=\frac{\hat{\mathbf z}\times\mathbf k}{\lvert\hat{\mathbf z}\times\mathbf k\rvert}=(-c_y,c_x,0),\qquad
\hat{\mathbf p}=\hat{\mathbf s}\times\hat{\mathbf k}=(c_x\cos\theta,\,c_y\cos\theta,\,-\sin\theta).
$$

These are orthonormal with $\hat{\mathbf p}\times\hat{\mathbf s}=\hat{\mathbf k}$, so $(\hat{\mathbf p},\hat{\mathbf s},\hat{\mathbf k})$ is right-handed. Projecting and eliminating $E_z$ with Gauss's law:

$$
E_s=-c_yE_x+c_xE_y,\qquad E_p=\frac{c_xE_x+c_yE_y}{\cos\theta},\qquad
\lvert E_s\rvert^2+\lvert E_p\rvert^2=\lvert E_x\rvert^2+\lvert E_y\rvert^2+\lvert E_z\rvert^2.
$$

The inverse is $\mathbf E_\perp=E_s(-c_y,c_x)+E_p\cos\theta\,(c_x,c_y)$, and $E_z=-\sin\theta\,E_p$. Because each mode's power weight $k_z/k_0$ multiplies the same vector norm, a per-mode $2\times2$ Jones matrix $M$ acting on $(E_s,E_p)$ is power-nonincreasing iff $\lVert M\rVert_2\le1$, and power-preserving if $M$ is unitary. Selected s and p powers sum to the propagating total over the whole cell, though not pointwise in a finite aperture.

### Coherency transfer and passivity

For $\mathbf e'=T\mathbf e$, the coherency $J=\langle\mathbf e\mathbf e^\dagger\rangle$ maps to $J'=TJT^\dagger$. It stays Hermitian PSD: $v^\dagger TJT^\dagger v=(T^\dagger v)^\dagger J(T^\dagger v)\ge0$. The trace is bounded: $\operatorname{tr}J'=\operatorname{tr}(T^\dagger TJ)\le\lambda_{\max}(T^\dagger T)\operatorname{tr}J=\sigma_{\max}(T)^2\operatorname{tr}J$, using $J\succeq0$. Linear interpolation $T_t=(1-t)T_0+tT_1$ satisfies $\lVert T_t\rVert_2\le(1-t)\lVert T_0\rVert_2+t\lVert T_1\rVert_2\le1$ by the triangle inequality, so passive nodes give a passive interpolant. Elements in beam order compose as $T_n\cdots T_1$.

### Paraxial handoff

For $\theta\to0$, $k_z/k_0\to1$ and $\lvert E_z\rvert\to0$, so $\frac{dU}{dE}\to\frac1{\pi\hbar Z_0}\int\lvert\widetilde{\mathbf E}_\perp\rvert^2dA$. The Jones spectral density in photons/eV is

$$
J_{ij}(E)=\frac{1}{\pi\hbar_{\rm eV\,s}Z_0\,(E_{\rm eV}e)}\int_A\widetilde E_i\widetilde E_j^*\,dA.
$$

## Cheap filters

| Check | Result |
|---|---|
| Units | $\Delta x\Delta y\lvert\hat E\rvert^2/(N Z_0)$: (V/m)$^2$m$^2/\Omega$ = W. $\Delta x\Delta y\lvert\hat{\widetilde E}\rvert^2/(N Z_0\hbar_{\rm eV\,s})$: V$^2$s$^2/\Omega$/(eV s) = J/eV. $\mathbf H=\mathbf k\times\mathbf E/(Z_0k_0)$ in A/m (or A s/m). Pass. |
| Limits | $\Delta z=0$ is the identity; composition holds; propagating flux is conserved; evanescent decay is $e^{-\kappa\Delta z}$ with zero net flux; normal incidence gives $E_z=0$ and $S_z=\lvert E\rvert^2/(2Z_0)$; the paraxial limit recovers the $(\pi\hbar Z_0)^{-1}$ Jones normalization. Pass. |
| Conventions | $e^{-i\omega t}$ with $e^{+ik_z\Delta z}$ outgoing; $\operatorname{Im}k_z\ge0$; Faraday sign gives $S_z>0$ for a $+z$ wave; $\hat{\mathbf p}\times\hat{\mathbf s}=\hat{\mathbf k}$ (right-handed in the order $\hat{\mathbf p},\hat{\mathbf s},\hat{\mathbf k}$); the sign of $\hat{\mathbf p}$ affects only the phase convention of $E_p$, not power. Pass. |

## Comparison with the implementation

- `propagate_vacuum` / `propagate_screen_field`: `fft2` → multiply by `exp(1j*kz*distance)` → `ifft2`. `kz = sqrt((k0²-kx²-ky²).astype(complex))` is the principal root: real non-negative for propagating modes, $+i\kappa$ for evanescent ones (real negative argument with $+0$ imaginary part). Zero distance returns a copy; negative distance is refused. Matches.
- `forward_power_w`: `(dx*dy/(2*Z0*nx*ny)) * Σ (kz.real/k0)(|Fx|²+|Fy|²+|Ez|²)` over $k_\perp^2<k_0^2(1-10^{-12})$, with `ez = -(kx*Fx+ky*Fy)/kz`. Identical to $P_+$.
- `collect_screen_field` → `_plane_electric_magnetic` / `_collected_energy`:
  - `hx = (ky*ez - kz*Ey)/(Z0*k0)`, `hy = (kz*Ex - kx*ez)/(Z0*k0)`: the $\mathbf k\times\mathbf E$ components with the correct sign, evanescent modes included.
  - `pz = Re(Ex*conj(Hy) - Ey*conj(Hx))`, then `Σ pz·w·dx·dy/(π·ħ_eVs)` gives J/eV; `photons_per_eV = J/eV / (E_eV·e)`. Identical to the two-sided result.
  - A grazing $k_z\approx0$ mode with content is refused rather than given an infinite $E_z$.
- `_polarized_spectrum`: `es = -cy*Ex + cx*Ey`, `ep = (cx*Ex + cy*Ey)/cos`; reconstruction `(-cy*s + cos*cx*p, cx*s + cos*cy*p)`. Identical to the basis above. Evanescent and grazing modes are zeroed when a Jones matrix is supplied, as documented. The passivity check is $\sigma_{\max}\le1+10^{-12}$.
- `PlaneAcceptance._weights`: rectangles use the exact 1-D overlap product of pixel cells $[x_i\pm\Delta x/2]$ with pixel centres $(i-(n-1)/2)\Delta x$, the same convention as `ScreenSpectralField`. Circles use 8×8 subpixel point sampling. Both apertures must lie strictly inside the window. Matches the docstring.
- `apply_optical_elements`: `j = t @ j @ t.conj().transpose(0,2,1)` per element in order; `transfer` interpolates the real and imaginary parts linearly, which is linear complex interpolation; nodes are checked to $\sigma_{\max}\le1+10^{-12}$; extrapolation and over-angle use are refused. Matches $TJT^\dagger$ and the convexity argument.
- `coherency_from_screen_fields`: `Σ E_a E_b^* w · dx·dy/(π·ħ_eVs·Z0·E_eV·e)`. This equals the paraxial $J_{ij}$ above, gated by angular leakage and by agreement of the trace with the full-vector `collect_screen_field` flux.
- `finite_screen_collection_convergence`: the pitch sequence $(2^kN,\Delta/2^k)$ keeps the window fixed; the window sequence $(2^kN,\Delta)$ keeps the pitch fixed; the relative change between the last two grids is reported. `hemisphere_fft_bias` is the ratio of the FFT Riemann sum to the direct Gauss–Legendre quadrature, minus 1. I checked that quadrature's Jacobian: $d^2k=k_0^2u\,du\,d\phi$ and the $(4\pi^2)^{-1}$ continuous Parseval factor give the same $(\pi\hbar Z_0)^{-1}$ normalization. Matches. It is a numerical diagnostic, not physics.

### Independent numerical checks

The checks used a script with CODATA $\hbar$, $c$, $\mu_0$, $e$ and $Z_0=\mu_0c$, $E=1$ eV, a 64×64 grid and $\Delta=\lambda/3$. The reference expressions were built independently: plane waves on the FFT grid from explicit $\hat{\mathbf s}$/$\hat{\mathbf p}$ vectors, $\mathbf H=\hat{\mathbf k}\times\mathbf E/Z_0$, and the k-space sums above.

1. Oblique mixed s/p plane wave: propagated phase error $1.3\times10^{-8}$ (relative to unit amplitude, from the $\hbar c$ constant). `forward_power_w` against $(k_z/k_0)\lvert\mathbf E\rvert^2L^2/(2Z_0)$: ratio $1+6.9\times10^{-10}$ (rounding of the $Z_0$ constant).
2. Random complex field: zero distance exact (0.0). Composition $0.3\lambda+1.1\lambda$ against $1.4\lambda$: relative difference $9\times10^{-16}$. Propagating power after $5\lambda$: relative change $2\times10^{-16}$.
3. Evanescent mode ($k_x=1.03k_0$): amplitude after $\lambda/2$ divided by $e^{-\kappa\lambda/2}$ is $1-8\times10^{-9}$. Reported forward power is $9\times10^{-41}$, i.e. zero.
4. Faraday sign: reconstructed $H$ against $\hat{\mathbf k}\times\mathbf E/Z_0$, relative difference $7\times10^{-10}$. $S_z>0$, and $S_z=(k_z/k_0)\lvert\mathbf E\rvert^2/Z_0$ exactly.
5. Gaussian two-sided field: `collect_screen_field` (full plane, $z=2\lambda$) and `spectral_energy_per_eV` against the independent $(\pi\hbar Z_0)^{-1}$ k-space sum: both $1+1.4\times10^{-10}$; photons/eV also $1+1.4\times10^{-10}$.
6. Factor-of-2 bridge (time domain, $2^{18}$ samples, Gaussian-modulated carrier): $U_{\rm spec}/U_{\rm time}=1+2\times10^{-16}$ and $U_{\rm phasor}/U_{\rm time}=1.0$.
7. s/p Jones:
   - Pure s and pure p waves pass with fraction 1.000 through their own analyzer and $<10^{-30}$ through the crossed one.
   - Random propagating-only field: $(P_s+P_p)/P=1-2\times10^{-16}$; a random unitary Jones preserves $P$ to $2\times10^{-16}$.
   - Independent $P_s$ from $E_s=\hat{\mathbf s}\cdot\mathbf E$ agrees to $6\times10^{-10}$ once the axial-mode convention (finding 1) is matched; without it the difference is $4\times10^{-4}$, entirely from the $k_\perp=0$ bin.
8. $TJT^\dagger$ with three random contractions on a 3-node grid, evaluated at 7 energies: max deviation from manual linear-interpolated $TJT^\dagger$ is $8\times10^{-17}$. Output minimum eigenvalue is $>0$, the maximum trace ratio is 0.22, and the maximum interpolant $\sigma_{\max}$ is 0.66.
9. Paraxial Gaussian ($w=6\lambda$, $z=3\lambda$): handoff $J_{xx}$ and $J_{xy}$ against the independent $(\pi\hbar Z_0E)^{-1}\int\widetilde E_i\widetilde E_j^*dA$: $1+1.4\times10^{-10}$. The trace against the full-vector flux differs by $7\times10^{-7}$ (expected paraxial residual).

The implementation-context tests `tests/montecarlo/test_optical_wavefront.py`, `test_optical_finite_collection.py`, `test_optical_interferometer_handoff.py`, `test_optical_finite_convergence.py` and `tests/detectors/test_optical_elements.py`: 35 passed at the frozen revision.

## Adjudication

No divergent factor, sign, exponent, unit or convention. The propagator root and sign, the Gauss-law $E_z$, the Faraday $\mathbf H$, both power normalizations, the s/p projection and inverse, aperture weights, the $TJT^\dagger$ map with its passivity argument, and the paraxial $(\pi\hbar Z_0)^{-1}$ handoff all match the independent derivation. They agree numerically to constant-rounding level ($\le 10^{-9}$). Verdict: **rederived** for the vacuum angular-spectrum, Poynting, s/p Jones and coherency-transfer equations.

What this does and does not establish:

- Established: exact source-free vacuum propagation between parallel planes on a periodic grid, and consistent phasor and two-sided spectral energy bookkeeping (the factor of 2). Phase is retained because propagation is linear and unitary on propagating modes.
- Not established: the qualitative clause "where a far-field angular detector is insufficient" (a modelling judgement, not an equation); periodic-window and mesh convergence for any physical case; the Schmidt et al. Appendix D reproduction; measured element data; CuPy/CUDA parity. The PEC source fields fed into the collector belong to `optical-finite-boundary-field` and were not re-derived here.

### Findings

1. Low, convention: at the axial bin $k_\perp=0$ the s/p basis is undefined. `_polarized_spectrum` silently sets $(c_x,c_y)=(1,0)$, i.e. $\hat{\mathbf s}=\hat{\mathbf y}$ and $\hat{\mathbf p}=\hat{\mathbf x}$. A non-diagonal or s-only/p-only `jones_sp` therefore acts on the on-axis mode as a fixed lab-frame polarizer. This choice is not documented in the docstring or on the physics page. The effect is negligible for azimuthally polarized TR (axial content zero) but not for a general field; it accounts for the $4\times10^{-4}$ difference in check 7. Suggest documenting it.
2. Low, wording: the physics page writes the handoff normalization as $1/(\pi\hbar Z_0E\,[\mathrm{J/eV}])$. This is dimensionally ambiguous; it is correct only with $\hbar$ in eV s and $E$ the photon energy in J, which is what the code does. Suggest writing $(\pi\hbar_{\rm eV\,s}Z_0\,E_{\rm eV}e)^{-1}$.
3. Information: with any `jones_sp` (even the identity), evanescent modes are discarded, so identity-Jones collection in the near field differs from unfiltered collection. This is documented; the note is only that identity Jones is not a no-op.

### Caveats

- Schmidt et al., arXiv:1803.00608, Appendix D was not available in this context. The rederivation rests on the first-principles Maxwell angular-spectrum identity, which is the primary cited source.
- FFT periodicity: all identities hold exactly for the periodic cell. Physical aperture collection additionally requires the window/pitch convergence that `finite_screen_collection_convergence` diagnoses but does not guarantee.
