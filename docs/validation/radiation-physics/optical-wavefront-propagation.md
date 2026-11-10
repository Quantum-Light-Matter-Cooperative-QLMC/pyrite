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

The implementation-context tests `tests/montecarlo/spectrum/test_optical_wavefront.py`, `test_optical_finite_collection.py`, `test_optical_interferometer_handoff.py`, `test_optical_finite_convergence.py` and `tests/detectors/test_optical_elements.py`: 35 passed at the frozen revision.

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

## Device coherency-transfer re-derivation (2026-10-23, issues #616–617)

Fresh verifier; scope is `apply_optical_elements` downstream of collection,
with signatures/docstrings and ledger read before implementation. No upstream
vacuum or finite-screen solver, published comparison or hardware calibration
is re-certified by this scoped check.

From linear Jones transfer of a random transverse amplitude $a$, define
$J=\langle aa^\dagger\rangle$. An element $T$ sends $a$ to $Ta$, hence

$$
J'=TJT^\dagger,\qquad
J_{\rm out}=T_n\cdots T_1J_{\rm in}T_1^\dagger\cdots T_n^\dagger.
$$

Thus the first element acts first, with no transpose substituted for the
conjugate transpose. Photon-density units are unchanged; energy-resolved
Jones matrices carry dimensionless amplitudes. For a positive-semidefinite
input and passive $T^\dagger T\preceq I$, positivity is preserved and
$\operatorname{tr}J'\le\operatorname{tr}J$. Identity preserves $J$; a unitary
retarder preserves its trace; an ideal polarizer gives Malus' law. Changing
array backend cannot change these index contractions. The operator remains
restricted to the calibrated spectral and angular domain stated by each
element. This derivation and the unit/limit/sign filters precede inspection
of the implementation body.

### Scoped comparison and verdict

Inspection after derivation found `t @ j @ t.conj().transpose(0, 2, 1)`
in beam order. `CalibratedJonesElement.transfer` stages calibration knots on
the energy backend, interpolates real and imaginary amplitude components and
refuses extrapolation; it introduces no intensity or photon-energy factor.
An independent NumPy example with a complex positive-semidefinite coherency
and two noncommuting complex passive matrices matched the single combined
operator expression within relative tolerance $10^{-14}$ and absolute
tolerance $10^{-10}$ photons/(eV pulse).

- **Filters**: units pass; limits pass; signs/conventions pass.
- **Re-derivation**: `matches`; no transpose, ordering or normalization discrepancy.
- **Verdict**: `rederived` for this scoped coherency-transfer refactor.
- **Suggested ledger change**: retain `rederived`; append this scoped check to
  Checks. No real CUDA execution, published anchor or human sign-off is claimed.

## Direct-Green collection and finite-aperture band convergence (2026-11-10, issue #616)

Fresh-context verifier, frozen revision `729074cb0`. Scope:
`optical_finite_collection.py::collect_screen_green` (and private
`_green_fields`) and
`optical_finite_convergence.py::finite_aperture_spectral_convergence` /
`ApertureSpectralConvergence`. The PEC source `circular_pec_transition_field`
(claim `optical-finite-boundary-field`) is a given input. Order followed: ledger
row and docstrings/signatures only, cheap filters, the derivation below, and
only then the implementation bodies.

### Independent derivation

**Fields of a tangential source sheet.** Time convention $e^{-i\omega t}$,
$k=\omega/c$, $G(R)=e^{ikR}/(4\pi R)$. For a sheet in $z=0$ with outgoing
tangential field $\widetilde{\mathbf E}_s$, the Schelkunoff magnetic current
$\mathbf M=-\hat{\mathbf z}\times\widetilde{\mathbf E}_s$ in front of its PEC
image radiates twice the free-space field. With
$\mathbf A=\hat{\mathbf z}\times\widetilde{\mathbf E}_s=(-E_{s,y},E_{s,x},0)$,

$$
\widetilde{\mathbf E}(\mathbf r)=2\int\nabla\times\big[G\,\mathbf A\big]\,dA'
=2\int G'(R)\,\hat{\mathbf n}\times\mathbf A\,dA',\qquad
G'=\Big(ik-\frac1R\Big)G,
$$

where $\hat{\mathbf n}=(\mathbf r-\mathbf r')/R$. This has exact $1/R$ and
$1/R^2$ terms. Faraday's law with $e^{-i\omega t}$ gives
$\nabla\times\widetilde{\mathbf E}=i\omega\mu_0\widetilde{\mathbf H}$, so
$\widetilde{\mathbf H}=\nabla\times\widetilde{\mathbf E}/(ikZ_0)$. For a
constant $\mathbf A$, away from the sheet
$\nabla\times(\nabla G\times\mathbf A)=(\mathbf A\cdot\nabla)\nabla G-\mathbf A\nabla^2G
=(\mathbf A\cdot\nabla)\nabla G+k^2G\mathbf A$. Using
$\partial_i\partial_jG=(G''-G'/R)\,n_in_j+(G'/R)\,\delta_{ij}$, this gives

$$
\widetilde{\mathbf H}=\frac{2}{ikZ_0}\int G\Big[\Big(-k^2-\frac{3ik}{R}+\frac{3}{R^2}\Big)(\hat{\mathbf n}\cdot\mathbf A)\,\hat{\mathbf n}
+\Big(k^2+\frac{ik}{R}-\frac{1}{R^2}\Big)\mathbf A\Big]dA',
$$

which has $1/R$, $1/R^2$ and $1/R^3$ terms. Units: $\widetilde{\mathbf E}_s$
in V s/m, so $\widetilde{\mathbf E}$ is in V s/m and $\widetilde{\mathbf H}$ in A s/m.

**Spectral energy through an aperture.** For
$\widetilde{\mathbf E}(\omega)=\int\mathbf E(t)e^{+i\omega t}dt$, Parseval and
the reality of $\mathbf E(t)$ give
$U=\int S_z\,dt\,dA=\pi^{-1}\int_0^\infty d\omega\int{\rm Re}[\widetilde{\mathbf E}\times\widetilde{\mathbf H}^*]_z\,dA$.
With $d\omega=dE/\hbar$,

$$
\frac{dU}{dE}=\frac{1}{\pi\hbar}\int_{\rm ap}{\rm Re}\big[\widetilde{\mathbf E}\times\widetilde{\mathbf H}^*\big]_z\,dA,
\qquad
\frac{dN}{dE}=\frac{dU/dE}{E\,e}.
$$

With $\hbar$ in eV s, $dU/dE$ is in J/eV, and $dN/dE$ (with $E$ in eV) is in
photons/eV per electron. The same Parseval argument in the far zone, with
$|\widetilde{\mathbf E}|=2k|\widetilde{\mathbf A}(\mathbf k_\perp)|/(4\pi r)$,
gives the angular yield

$$
\frac{d^2U}{d\omega\,d\Omega}=\frac{k^2\,|\widetilde{\mathbf A}(k\sin\theta)|^2}{4\pi^3Z_0},
\qquad
|\widetilde{\mathbf A}|=2\pi\Big|\int_0^a f(\rho)J_1(k\rho\sin\theta)\rho\,d\rho\Big|
$$

for a radially polarized source $f(\rho)\hat{\boldsymbol\rho}$. For a normal
charge, $f=\frac{e\omega}{2\pi\epsilon_0\gamma v^2}K_1(\omega\rho/\gamma v)$.
As $a\to\infty$ this reduces to the Ginzburg–Frank yield
$e^2\beta^2\sin^2\theta/[4\pi^3\epsilon_0c(1-\beta^2\cos^2\theta)^2]$.

**Sound band-convergence diagnostic.** The band count is
$N=\int_{E_1}^{E_2}dN/dE\,dE$, approximated by composite Simpson on
$n$ odd uniform nodes. Halving the spacing gives $(n-1)2^\ell+1$ nested nodes,
and the change between the last two levels estimates the energy-quadrature
error ($O(h^4)$ for smooth spectra). Each node's $dN/dE$ has two further
discretization errors: the detector quadrature pitch (midpoint rule with
partial-area weights at the aperture edge) and the source pixel pitch. Each
should be refined alone, holding the energy nodes fixed, the physical source
window fixed and the other pitch fixed. For each, the largest relative change
of $dN/dE$ over the fixed nodes between the last two levels should be reported.
This change estimates the error only if the sequence is in its asymptotic
(monotone, power-law) regime. With staircase geometry (pixelated disc edge,
pixel-centre aperture weights) that regime is not guaranteed, and a single
two-level change can understate the error. A sound diagnostic therefore needs
at least three levels, or a check of monotonicity/ratio. A zero or
sign-changing signed near-field $dN/dE$ must not be used as a relative-change
denominator.

### Cheap filters

| Filter | Result |
|---|---|
| Units | pass: V s/m × A s/m × m² / (eV s) = J/eV; ÷ (eV · C) gives photons/eV |
| Large disc, far plane, cone | pass: 6 µm disc, $E=2$ eV, $D=2$ mm, cone half-angles 0.4 and 0.8 rad: `collect_screen_green`/GF $-1.1\times10^{-4}$, $-1.9\times10^{-4}$ (my 6 µm Hankel = GF to $10^{-6}$) |
| Far plane, finite disc | pass: 1 µm disc, 1.5 eV, $D=60$ µm, cone 0.3 and 0.6 rad vs the continuous-disc Hankel integral: $+3.9\times10^{-3}$, $+1.3\times10^{-3}$ at source pitch 0.1 µm; $-6.9\times10^{-4}$, $-2.8\times10^{-4}$ at 0.025 µm |
| Large aperture | pass: at $z=1$ µm, apertures of radius 10 and 20 µm collect the far-field cone $\arctan(R/z)$ to $-3.7\times10^{-4}$ and $-3.4\times10^{-4}$, and approach the hemisphere ($-6.3\%$, $-3.2\%$, deficit $\propto z/R$) |
| Signs/conventions | pass: $e^{+ikR}$ outgoing, $H=\nabla\times E/(ikZ_0)$, $S_z>0$ in every case above |

### Comparison with the implementation

Inspection after derivation:

```text
g1 = (1j*k0 - 1/rr) * g ; g2 = ((1j*k0 - 1/rr)**2 + 1/rr**2) * g
e_out = 2 * sum(cross(g1*rhat, a))                       # a = (-Ey, Ex, 0)
hess_a = (g2 - g1/rr)*(a.rhat)*rhat + (g1/rr)*a
curl = 2 * sum(hess_a + k0**2 * g * a);  H = curl*area/(1j*k0*Z0)
energy = sum(Re(Ex Hy* - Ey Hx*) * w) * pitch**2 / (pi * HBAR_EV_S);  photons = energy/(E e)
```

The code matches the derivation term by term: the factor 2, the sign of
$\mathbf A$, $G'$ and $G''$, the Hessian split, the $k^2G\mathbf A$ term, the
$1/(ik_0Z_0)$ Faraday factor and the $(\pi\hbar)^{-1}$ two-sided
normalization. The detector grid is centred, with
$n=2\lceil{\rm extent}/p\rceil+2$ points of pitch $p$, so it always contains
the aperture strictly. It reuses `PlaneAcceptance._weights`: exact
rectangle-overlap fractions, or 8×8 subsamples for circles, applied as
weights at pixel centres. This is a midpoint rule with fractional edge
weights, with an $O(p)$ edge error. Simpson weights (`h/3`, 4 on odd, 2 on
interior even) are correct, and the nesting $(n-1)2^\ell+1$ is correct. The
detector axis varies only `pitch` (source level 0). The source axis varies
only `nx·2^ℓ, dx/2^ℓ` (fixed window $n_x d_x$) at the base detector pitch.
Both use the fixed base nodes, so the axes are independent. The cache key
`(energy, source_level, pitch)` captures every varying input. Bit-level
`linspace` mismatches between nested node sets can only cause cache misses,
not wrong values. `_MAX_GRID` bounds the refined source grid
($n\,2^{\rm ref}\le4096$), consistent with the only axis that grows the source
array. The detector point count is not bounded, but is time-limited only
because `max_interactions` blocks memory.

### Numerical checks (independent code, outside the repo)

My own $K_1$ source (Jackson moving-charge field, SI) and my own direct
$\mathbf E$, $\mathbf H$ sum with the explicit $1/R^{1,2,3}$ kernels above:

- Source: equal to `circular_pec_transition_field` to $6\times10^{-11}$, with
  316 metal pixels in both.
- Fields at four points, including $\rho=0.03$ µm, $z=0.05$ µm:
  $\mathbf E$ to $2.5\times10^{-10}$ and $\mathbf H$ to $1.0\times10^{-9}$.
- Committed case at 1.5 eV (2 µm square at 0.5 µm, pitch 0.1 µm):
  $2.944466859\times10^{-3}$ photons/eV vs `collect_screen_green`
  $2.944466859\times10^{-3}$, relative $9.5\times10^{-11}$.
- Off-centre, non-aligned rectangle (centre $(0.33,-0.21)$ µm, $1.37\times0.91$ µm,
  pitch 0.07 µm) with my own overlap weights: relative $1.5\times10^{-10}$
  (weights sum to the exact area).
- The error of the same rectangle against a 0.01 µm fine midpoint reference
  is $-3.1\times10^{-4}$, $-7.9\times10^{-4}$ and $+9.1\times10^{-5}$ at pitches
  0.14, 0.07 and 0.035 µm. This is non-monotone; it is the edge-weight effect.

**Diagnostic on the committed case** (1 µm disc, $\beta_z=-0.95$, 1–2 eV,
32×32 at 0.1 µm, plane 0.5 µm, 2×2 µm, detector pitch 0.1 µm, $n_E=3$):

| refinements | energy change | detector pointwise | source pointwise | band photons |
|---|---|---|---|---|
| 1 | $5.89\times10^{-3}$ | $7.63\times10^{-4}$ | $2.92\times10^{-4}$ | $3.0323\times10^{-3}$ |
| 2 | $4.89\times10^{-4}$ | $1.91\times10^{-4}$ | $1.28\times10^{-3}$ | $3.0338\times10^{-3}$ |

The source axis is non-monotone: $2.9\times10^{-4}$ (0.1→0.05 µm), then
$1.28\times10^{-3}$ (0.05→0.025 µm). My own evaluation at source and detector
pitch 0.025 µm gives node values $4.10911$, $2.94360$, $2.18964\times10^{-3}$,
against $4.11894$, $2.94447$, $2.18985\times10^{-3}$ at the base pitches. The
true base-grid error at 1 eV is therefore $\ge2.4\times10^{-3}$. The
refinements=1 report ($\max=7.6\times10^{-4}$ on the pitch axes) understates
it about 3×. The far-plane check above shows the same thing more sharply. At
$D=60$ µm, cone 0.3 rad, the source change 0.1→0.05 µm is $1.5\times10^{-4}$,
while the error against the continuous disc is $3.9\times10^{-3}$, about 25×
larger. The cause is the pixel-centre metal mask $\rho\le a$ on non-nested
centred grids: the staircase disc area and edge phase change erratically with
pitch. The energy and detector axes behaved as honest, decreasing indicators
here.

**Not run (memory).** `check2.py` was also meant to evaluate my own reference
at source pitch 0.0125 µm (192×192 window, ≈20 100 metal pixels) with detector
pitch 0.025 and 0.0125 µm (6 400 and 25 600 points) at the three base nodes.
This would bound the committed-case base error and the source-axis asymptote.
The first attempt used 4 000-point chunks: about 1.3 GB per complex
temporary, ≳10 temporaries, so it was OOM-killed at the 8 GiB cap. A second,
chunked run (≈1 GB) was stopped at the supervisor's request after the
0.025 µm row. Route the 0.0125 µm rows to the remote box.

### Findings

1. **Moderate (diagnostic honesty).**
   `optical_finite_convergence.py::finite_aperture_spectral_convergence`
   (source axis), with `_pointwise` comparing only the last two levels. The
   first divergent statement is the docstring premise that halving
   `dx_m, dy_m` at a fixed window is a convergent refinement axis whose last
   two-level change bounds the error. On the staircased pixel-centre disc
   mask the sequence is non-monotone ($2.9\times10^{-4}$ then
   $1.28\times10^{-3}$), and it understates the continuous-disc error by up to
   25×. Fix: require `refinements >= 2` and report the maximum change over all
   levels (or a ratio/monotonicity flag) for the source axis. Alternatively,
   use area-weighted edge pixels in the source, or state in the docstring that
   the source axis measures the sampled (staircase) disc, not the continuous
   one.
2. **Minor.** `optical_finite_convergence.py::_pointwise` divides by
   `abs(spectra[-1])` without a guard. A zero or sign-changing signed
   near-field node gives inf/nan and a RuntimeWarning. `converged` then
   evaluates False, so this is safe but uninformative. Fix: normalize by
   `max(abs(spectra[-1]))` (or the band value) and refuse/flag sign changes.
3. **Minor.** `ApertureSpectralConvergence.band_photons` is evaluated at the
   base detector/source pitches. The pitch changes are measured only at the
   base energy nodes, and no combined estimate is given. The docstring says
   "finest energy grid", which is true, but it does not say "base pitches".
   Fix: state this, or report the band at the finest pitches.
4. **Minor.** `optical_finite_collection.py::collect_screen_green`. The
   docstring says "Both pitches must resolve min(wavelength, plane distance)",
   but this is not checked. Detector edge weights are applied at pixel centres
   ($O(p)$, non-monotone: $-3.1$, $-7.9$, $+0.9\times10^{-4}$). Fix: warn or
   refuse when `pitch_m` or `dx_m` exceeds $\min(\lambda,z)/2$, and document
   the first-order, non-monotone edge term.
5. **Minor (docs rendering).** In
   `docs/physics/radiation-physics/optical-wavefront-and-detectors.md`, lines
   983 and 985 contain literal control characters: a BS byte before `eta=0.95`
   and a TAB before `imes10^{-3}`. These are `\b` and `\t` swallowed by an
   escape, and they render wrongly. Fix: restore `$\beta=0.95$` and
   `$2\times10^{-3}$`.
6. **Minor (ledger traceability).** The `optical-wavefront-propagation` row's
   Code field lists `finite_screen_collection_convergence` but not
   `finite_aperture_spectral_convergence`, which carries this claim's
   `Validation:` marker. Checks still says `collect_screen_green` is not
   covered by a review. Fix: add the symbol and record this check.
7. **Info.** The cache key and axis independence are correct. Nested-node
   float mismatches cost recomputation only. The `_MAX_GRID` refusal is
   consistent with the source axis.

### Scoped verdict

- **Filters**: units pass; limits pass (GF cone $\le2\times10^{-4}$,
  finite-disc Hankel cone $\le4\times10^{-3}$, converging with source pitch,
  large-aperture cone); signs/conventions pass.
- **Re-derivation**: `matches`. The $\mathbf E$/$\mathbf H$ Green kernels,
  Faraday factor, $(\pi\hbar)^{-1}$ normalization, photons/eV, aperture
  weights and nested Simpson agree to $\le10^{-9}$ with independent code.
- **Verdict**: `rederived` for `collect_screen_green` and the arithmetic of
  `finite_aperture_spectral_convergence`. The diagnostic's source-pitch axis
  is not an honest error bound with two levels (finding 1). That is a
  numerical-diagnostic defect, not a physics-formula discrepancy.
- **Suggested ledger change**: retain `rederived`; add
  `finite_aperture_spectral_convergence` to Code; append this review to Checks
  (replacing "`collect_screen_green` ... not covered"); and add finding 1 and
  the unrun 0.0125 µm reference to Open. No human sign-off is implied.

### Findings applied by the task owner (2026-11-10)

- Finding 1: `_pointwise` now reports the largest change over every pair of
  consecutive levels; `ApertureSpectralConvergence.converged` requires at
  least two refinements; the default is `refinements=2`. The docstring and
  the physics page state that the source axis converges the sampled
  pixel-centre disc, not the continuous one. On the committed case the
  reported source change is now $1.3\times10^{-3}$ instead of
  $1.6\times10^{-4}$, and a single refinement never reports `converged`
  (`test_single_refinement_is_a_lambda_dependent_hemisphere_fraction_not_convergence`,
  slow `test_aperture_band_converges_over_two_refinements_of_each_axis`). The
  source edge itself is unchanged, since that would change the default
  output of `circular_pec_transition_field`.
- Finding 2: the pointwise change is scaled by the largest $\lvert dN/dE\rvert$
  of the finer level; an all-zero level gives an infinite change, so the
  result does not report `converged`.
- Finding 3: documented. `band_photons` uses the finest energy grid at the
  base pitches, and no combined error is formed.
- Finding 4: documented in `collect_screen_green` (pitch resolution is the
  caller's responsibility; the edge term is $O(p)$ and non-monotone). No
  refusal was added.
- Finding 5: control characters on the physics page restored.
- Finding 6: ledger Code and Checks updated.
- Not run: the 0.0125 µm continuous-disc reference (about 8 GiB dense; at
  least 1 GiB chunked). It belongs on the remote box.

## Published propagation anchor: Casalbuoni et al. 2005, Figs. 8 and 15 (2026-11-10, issue #616)

**Scope.** This is a fresh-context review of the proposed published-regression anchor for `optical-wavefront-propagation`. The anchor is `tests/montecarlo/spectrum/test_optical_propagation_casalbuoni_anchor.py` with data in `tests/data/optical/casalbuoni-2005-fig8-15.json`. The propagator equations were rederived earlier (sections above). This section asks only whether the test legitimately pins the claim to published numbers.

**Frozen revision.** The review is frozen at `729074cb0`. The test and data files are **uncommitted** (untracked); their sha256 prefixes are `7ad3231dae4a` for the test and `6b9bc49c56e5` for the JSON. The working tree also carried an uncommitted docstring-only edit to `optical_finite_collection.py::collect_screen_green`, which is not exercised here.

**Source.** S. Casalbuoni, B. Schmidt, P. Schmüser, *Far-Infrared Transition and Diffraction Radiation, Part I*, TESLA Report 2005-15 (DESY 2005). The review used:

- Eqs. (11)–(26), pp. 7–12;
- Fig. 8, p. 15;
- Eqs. (33)–(37) and Figs. 14–15, pp. 20–22.

All of these were read from the local PDF. Every number below comes from scripts written outside the repository. Those scripts use my own quadratures, my own direct Rayleigh–Sommerfeld sum and my own SVG digitization; they do not use the test's helper functions. PyRITE itself is used only as the object under test: `propagate_screen_field`, `collect_screen_field`, and `circular_pec_transition_field` as the shared source.

### Filters

| Check | Result |
|---|---|
| Units | pass. The two-sided field $\tilde E_p=\int E\,e^{i\omega t}dt$ is in V·s/m. $dU/d\omega\,d\Omega=\lvert\tilde E_p\rvert^2R^2/(\pi Z_0)$ is in J·s/sr. Then $\times N^2\times h\,\Delta f$ for the J/eV path, or $\times 2\pi\,\Delta f$ for the J·s path, gives J per 1 GHz. |
| Limits | pass. As $R\to\infty$, my absolute Eq. (25) equals the closed-form Eq. (20)–(21) to $5\times10^{-12}$ at $\gamma\theta=5,20,40$. As $r_b\to0$, the per-electron Eq. (33) reduces to Eq. (11) because $2I_1(x)/x\to1$. |
| Sign/convention | pass. The Fresnel phase uses $R=D/\cos\theta$, as in Eq. (23). The test writes this as $k\rho^2\cos\theta/(2D)$. The FFT and the direct sum both use the $e^{-i\omega t}$ outgoing sign. |

### Derivation of the absolute Eq. (25)

Eq. (11) gives the symmetric-transform radial field of the electron, $\tilde E_r=-e\omega K_1(\omega\rho/\beta\gamma c)/[(2\pi)^{3/2}\varepsilon_0\beta^2\gamma c^2]$. This matches Jackson's $\sqrt{2/\pi}\,(\omega/\gamma v^2)K_1$ form times $1/(4\pi\varepsilon_0)$.

Insert it into Eq. (12). Keep the Eq. (23) expansion $R'\simeq R-\rho\sin\theta\cos\phi+\rho^2/(2R)$, and use $\int_0^{2\pi}e^{-ik\rho\sin\theta\cos\phi}\cos\phi\,d\phi=-2\pi iJ_1(k\rho\sin\theta)$. The result is

$$
\tilde E_x=\frac{e k^2}{(2\pi)^{3/2}\varepsilon_0\beta^2\gamma c}\,\frac{e^{ikR}}{R}\,
\int_0^a J_1(k\rho\sin\theta)\,K_1\!\left(\frac{k\rho}{\beta\gamma}\right)e^{ik\rho^2/2R}\rho\,d\rho .
$$

Eq. (19) gives $dU/d\omega=2\varepsilon_0c\lvert\tilde E\rvert^2$. With $dS=R^2d\Omega$ from Eq. (20),

$$
\boxed{\frac{d^2U}{d\omega\,d\Omega}=\frac{e^2k^4}{4\pi^3\varepsilon_0c\,\beta^4\gamma^2}
\left\lvert\int_0^a J_1(k\rho\sin\theta)K_1\!\left(\frac{k\rho}{\beta\gamma}\right)e^{ik\rho^2/2R}\rho\,d\rho\right\rvert^2 }
$$

This matches the test's `_eq25` prefactor. The report states Eq. (25) only as a proportionality, so the absolute prefactor is a derived quantity. It is certified here by the $R\to\infty$ identity with the report's absolute Eq. (20).

**Convention conversion.** The two transforms are related by $\tilde E_p=\sqrt{2\pi}\,\tilde E_s$. Therefore $2\varepsilon_0c\lvert\tilde E_s\rvert^2=\varepsilon_0c\lvert\tilde E_p\rvert^2/\pi=\lvert\tilde E_p\rvert^2/(\pi Z_0)$. At a sampled pixel, PyRITE's source magnitude equals $\sqrt{2\pi}\times$ Eq. (11) to $1.3\times10^{-12}$.

**Bunch field.** Per electron, Eq. (33) relative to Eq. (11) is $[2I_1(\alpha r_b)/(\alpha r_b)]K_1(\alpha\rho)$ for $\rho>r_b$ and $[2K_1(\alpha r_b)/(\alpha r_b)]I_1(\alpha\rho)$ inside, with $\alpha=k/\beta\gamma$. This matches the test's `scale`.

**Collected energy.** The coherent energy for an infinitely short bunch is $N^2\,dU/d\omega$, with $N=1\,{\rm nC}/e$. Over $\Delta f=1$ GHz this is $N^2(dU/dE)\,h\Delta f$, which is the test's chain.

### Reference data (independent SVG re-digitization)

**Fig. 8 (p. 15).**

- Calibration: the axes were fitted to all six $x$ ticks and seven $y$ ticks. Tick non-uniformity is at most $0.05$ in $\gamma\theta$ and $0.0016$ in ordinate.
- Diamonds: 101 extracted. My diamonds differ from the JSON by at most $0.0029$ of full scale.
- Peak: both data sets peak at $\gamma\theta=28.57$.
- Red line: my extraction is contaminated by diamond paths near $\gamma\theta\approx49.6$, which gives a spurious $0.010$ there. Elsewhere it agrees with the JSON.

**Fig. 15 right (p. 22).**

- Calibration: $x$ from the 1 and 10 THz major ticks, $y$ from eleven ticks (fit residual $10^{-4}$).
- Dashed curve: 460 dash vertices. They differ from the JSON by at most $4.8\times10^{-5}$ µJ/GHz, i.e. $10^{-4}$ relative. Examples: $0.2913$ at 0.5 THz, $0.3657$ at 1 THz, $0.4558$ at 4 THz.

Transcription is faithful.

### Numerical checks (FFT propagator vs my own references)

The Fig. 8 row is the grid row next to the axis, over the lobe $\gamma\theta\in(8,46)$. The figure of merit is the maximum relative deviation of $\lvert E_{xy}\rvert^2R^2/(\pi Z_0)$ from my Eq. (25).

| $W$ [m] | $dx$ [mm] | $n$ | max dev |
|---|---|---|---|
| 0.32 | 0.30 (test base) | 1068 | 0.575 % |
| 0.48 | 0.30 | 1600 | 0.650 % |
| 0.32 | 0.25 | 1280 | 0.559 % |
| 0.32 | 0.60 | 534 | 2.52 % |
| 0.16 | 0.30 | 534 | 3.17 % |
| 0.32 | 0.20 | 1600 | **1.22 %** |

Other Fig. 8 results:

- **Exact distance.** The FFT differs from my exact-distance Eq. (12), a 2-D quadrature over $(\rho,\phi)$, by at most $0.17\%$ at $\gamma\theta=9,27,45$. My Eq. (12) differs from my Eq. (25) by about $0.2\%$. The $0.58\%$ floor is therefore model and pixel-staircase difference, not window error.
- **Complex field.** Compared with my own first Rayleigh–Sommerfeld sum, $\max\lvert E_{\rm FFT}-E_{\rm RS1}\rvert/\max\lvert E_{\rm RS1}\rvert=1.9\times10^{-4}$. The test's threshold is $3\times10^{-3}$.
- **Published curves (peak-normalized).** The FFT differs from the line by $0.0101$ and from the diamonds by $0.0096$ of full scale, both at $\gamma\theta\approx42.1$. My own Eq. (25) differs from the published line by $0.0107$ at the same place. So the $0.01$ is figure versus equation, consistent with an abscissa calibration offset of about $0.07$ in $\gamma\theta$ on a slope of about $0.15$ per unit. It is not propagator error.
- **Far field.** Eq. (26), peak-normalized, differs from the published near-field line by $0.962$ of full scale.

**Perturbations (my own transfer functions applied to the same source).**

| Perturbation | Intensity vs Eq. (25) | Complex vs RS1 | Test outcome |
|---|---|---|---|
| Conjugate propagator $e^{-ik_zd}$ (propagating modes) | $0.575\%$ (identical) | $1.90$ | caught only by the complex check |
| Paraxial transfer $e^{ikd-i(k_x^2+k_y^2)d/2k}$ | $0.83\%$ | $1.6\times10^{-3}$ | **passes every Fig. 8 assertion** |
| Propagation distance $0.99D$ / $1.01D$ | $10\%$ / $11\%$ | — | caught (published curve 0.031 / 0.023 FS) |
| Propagation distance $0.97D$ / $1.03D$ | $30\%$ | — | caught (0.074 FS) |
| Reference Fresnel phase $R=D$ (not $D/\cos\theta$) | $0.79\%$ | — | not discriminated at 1 % |

**Fig. 15 right**, 7-point sunflower bunch, $W=0.48$ m, collected in a circle of radius $0.6\tan(0.1)$ m:

| $f$ [THz] | Published | My Eq. (25)+(33) cone | Far/near | $r_b=0$ vs $150$ µm (equation) | FFT base | FFT vs published | FFT vs my eq. |
|---|---|---|---|---|---|---|---|
| 0.6 | 0.31147 | 0.30897 ($-0.80\%$) | $-0.64\%$ | $+0.17\%$ | 0.30884 ($dx=1$ mm) | $-0.84\%$ | $-0.04\%$ |
| 1.0 | 0.36574 | 0.36538 ($-0.10\%$) | $-0.13\%$ | $+0.40\%$ | 0.36729 ($dx=0.6$ mm) | $+0.42\%$ | $+0.52\%$ |

Convergence and sensitivity for Fig. 15:

- **Window:** going from 0.48 m to 0.72 m changes the result by $\le 3\times10^{-5}$.
- **Pitch at 0.6 THz:** halving to 0.5 mm changes it by $+0.32\%$, to $+0.28\%$ against my equation.
- **Pitch at 1 THz:** halving to 0.3 mm needs $n=1600$ with seven source fields. It exceeded the 1 GiB cap and was **not run**.
- **Bunch radius in the FFT path:** a single central electron differs from the 7-point bunch by only $+0.001\%$ (0.6 THz) and $+0.011\%$ (1 THz).

**Test run.** `pyrite-dev test` on the anchor file: 3 passed, 1 slow test skipped, 3.5 s. The slow test was not run, per a memory restriction (peak about 2 GiB).

### Findings

1. **Medium — the walk-off window is a 1-D bound applied to a 2-D square grid** (`test_optical_propagation_casalbuoni_anchor.py::_walkoff_window` and the module docstring).
   - **Why it fails.** The corner modes $k_x=k_y=\pi/dx$ walk off by $D\,s/\sqrt{1-2s^2}$, where $s=\lambda/2dx$, not by $D\tan(\arcsin s)$. Grazing waves are therefore carried as soon as $dx\le\lambda/\sqrt2$, not only at $dx\le\lambda/2$.
   - **Effect on the test's grids.** The base grid stays converged: the corrected requirement is $W\ge0.210$ m against $W=0.32$ m. The slow test's "fine" point, $dx=0.2$ mm ($s=0.75$), lies in the regime with no converged window, yet `_walkoff_window` declares $W\ge0.317$ m sufficient. I measured $1.22\%$ at $W=0.32$ m and $dx=0.2$ mm, which would fail the 1 % gate. The owner's $0.54\%$ at $W=0.64$ m ($n=3200$) was not reproduced under the memory rule.
   - **Fix.** Use $D\,s/\sqrt{1-2s^2}$ (infinite for $2s^2\ge1$) in `_walkoff_window` and the docstring. Replace the $0.2$ mm refinement point with $0.25$ mm at $W=0.32$ m (corrected requirement $0.316$ m; measured $0.559\%$). Assert the finite and infinite cases with the corrected bound.
2. **Medium — the complex check does not distinguish the exact transfer function from the paraxial one** (`test_fft_propagation_reproduces_fig8_near_field_distribution`, the `3e-3` Green tolerance).
   - **Evidence.** A paraxial (Fresnel) transfer function passes every Fig. 8 assertion: $0.83\%$ absolute and $1.6\times10^{-3}$ complex. The measured FFT-versus-direct-sum residual is $1.9\times10^{-4}$, so a 3e-3 tolerance is not justified by convergence.
   - **Fix.** Tighten to `1e-3` (5× the measured residual). That rejects the paraxial transfer while keeping the sign check, which is the only assertion that catches a conjugated propagator. Alternatively, state in the scope that the anchor covers only NA ≤ 0.05, where exact and Fresnel transfer are equivalent at the gate level.
3. **Medium (scope) — the Fig. 15 bunch radius is not exercised.** At $dx\ge0.6$ mm the 7-point sunflower bunch of radius $150$ µm changes the FFT energy by $\le0.011\%$. The equation's $r_b$ effect is $0.17$–$0.40\%$, also below both tolerances. The "1 nC, $r_b=150$ µm" description therefore pins only the $N^2$, Poynting and aperture normalization of a point-like coherent source. **Fix:** say in the test docstring and the ledger that Eq. (33) (transverse bunch form factor) is not tested, or drop the sunflower average.
4. **Low — the Fig. 8 observable omits $E_z$** (`_fig8_row`).
   - **Effect.** $\lvert E_{xy}\rvert^2$ is smaller than the radiated $\lvert\mathbf E\rvert^2$ by $\cos^2\theta$, a $-0.21\%$ bias at $\gamma\theta=46$. This is inside the tolerance but systematic.
   - **Fix.** Add $E_z=-(k_xE_x+k_yE_y)/k_z$, or divide by $\cos^2\theta$, or document the bias.
5. **Low — tolerance rationale wording.**
   - The comment says digitization is "~0.01 of full scale". Ordinate digitization is in fact $\le0.003$. The $0.01$ is the figure-versus-equation offset on the steep edge, $\gamma\theta\approx42$, which my own Eq. (25) shows too. The 0.02 gate is justified, but by abscissa calibration and slope, not by digitization.
   - The 1 % absolute gate does not discriminate the reference's $R=D$ versus $R=D/\cos\theta$ Fresnel phase ($0.79\%$). The test uses the correct form, so this is only a note.
6. **Low — Fig. 15 pitch convergence is not monotone toward the equation.** At 0.6 THz, halving the pitch moves the FFT from $-0.04\%$ to $+0.28\%$ against Eq. (25)+(33). That is consistent with the owner's "pitch halving 0.3 %" and well inside the 1 % and 1.5 % gates. At 1 THz the pitch-halved run was not reproduced (memory cap).
7. **Info.**
   - The absolute prefactor is derived, not printed in the report. It is certified via the Eq. (20) limit to $10^{-11}$.
   - The figure-versus-equation gap in Fig. 15 ($-0.80\%$ at 0.6 THz) leaves $0.7\%$ headroom inside the 1.5 % published gate.
   - The far/near energies agree to $0.64\%$ and $0.13\%$, which supports the "normalization, not phase" label.

### Verdict (scoped)

The anchor **supports `anchored`** for `propagate_screen_field`/`propagate_vacuum`, with this scope: paraxial-regime (NA ≤ 0.05) propagation of a PEC-disc physical-optics source at a Fresnel number of about 5. The anchor verifies:

- the complex outgoing phase sign, against a direct non-periodic sum;
- the absolute single-electron angular density, against the derived absolute Eq. (25) to 1 %;
- the published Fig. 8 shape, to 0.02 of full scale;
- the absolute collected coherent energy through a circular plane aperture, against Fig. 15 right (dashed) to 1.5 % at 0.6 and 1 THz, for `collect_screen_field`.

The anchor does not verify:

- discrimination between exact and Fresnel transfer (finding 2);
- the transverse bunch form factor (finding 3);
- high-NA propagation, optical elements, `collect_screen_green`, or Schmidt et al. Appendix D.

Findings 1 and 2 should be fixed before the row cites the slow-test convergence sequence as evidence. No discrepancy against the published numbers was found.

### Findings applied by the task owner (2026-11-10)

- Finding 1: `_walkoff_window` and the test docstring now use the per-axis
  corner-mode walk-off $Ds/\sqrt{1-2s^2}$, $s=\lambda/2\Delta x$ (infinite for
  $2s^2\ge1$). The slow test's finer pitch is 0.25 mm at $W=0.32$ m (bound
  0.316 m); the 0.2 mm pitch is now one of the asserted failures (no window
  exists). The window step is $0.32\to0.48$ m, which keeps the slow test near
  1 GiB.
- Finding 2: the complex Green-versus-FFT tolerance is $10^{-3}$ (measured
  $1.9\times10^{-4}$), and the docstring states the paraxial (NA $\le0.05$)
  scope and that exact-versus-Fresnel transfer is not discriminated.
- Finding 3: the docstring and ledger state that the Eq. (33) bunch radius is
  carried along but not tested.
- Finding 4: documented. The comparison is transverse-only, like the report's
  scalar Eq. (12), and the omitted $\lvert E_z\rvert^2$ is 0.2 % at
  $\gamma\theta=46$.
- Findings 5 and 6: tolerance comments corrected (ordinate digitization
  $\le0.003$; the 0.011 is a figure-versus-equation offset; the pitch halving
  is $+0.3$ %).
- After these edits: all four tests pass, including the slow one, in 6.4 s
  serially at 1.04 GiB peak RSS.
- Not run: the 1 THz Fig. 15 pitch halving (seven $1600^2$ sources). It needs
  more than 1 GiB and belongs on the remote box.
