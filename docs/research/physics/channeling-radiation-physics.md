# Channeling radiation: physics implementation walkthrough

**Status:** Proposed physics design; not implemented as a production model. The architecture is recorded in [ADR-0018](../../adr/0018-quantum-channeling-zone.md) (Proposed); delivery is tracked in [#56](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/56) and its sub-issues.

[ADR-0018](../../adr/0018-quantum-channeling-zone.md) fixes the architecture: a quantum entrance zone with one-way handoff to native transport. The earlier Geant4 route in [relativistic electron transport](relativistic-electron-transport.md) is superseded for channeling. This document works through the physics step by step, in the order an implementation would build it: continuum potential → transverse quantum states → initial occupation → depth evolution → photon emission → axial extension. It closes with physics easily missed, and with the ownership map between the channeling zone and native transport.

Conventions used throughout:

- $m$ is the electron rest mass, $\gamma = 1 + T/mc^2$ with $T$ the kinetic energy, $\beta c$ the speed, $p = \gamma m \beta c$ the momentum, and $pv = \gamma m \beta^2 c^2$ the standard channeling energy scale.
- $z$ is beam-longitudinal, $x$ (planar) or $(x, y)$ (axial) transverse.
- Working units are eV and Å: $\hbar^2 / 2 m_e = 3.810\ \mathrm{eV\,Å^2}$, $\hbar c = 1973.3\ \mathrm{eV\,Å}$, $m_e c^2 = 511.0\ \mathrm{keV}$.
- The electron charge is $-e$; "potential" $U$ below always means the *potential energy of the electron*, so wells sit **on** atomic planes and strings, not between them. This one sign flip drives most of the electron-vs-positron phenomenology.

## 0. Regime: why quantum, and what the numbers look like

The transverse motion of a channeled particle is governed by a potential well of depth $U_0$ and width $\sim d_p$ (the interplanar spacing). A WKB count of bound levels,

$$
n_b \sim \frac{d_p}{\pi \hbar} \sqrt{2 \gamma m U_0},
$$

grows like $\sqrt{\gamma}$. Classical/quasiclassical treatments (Baier–Katkov, `G4ChannelingFastSimModel`) require $n_b \gg 1$; at hundreds of MeV that holds, at 1–10 MeV it does not:

| $T$ | $\gamma$ | $pv$ | $2\gamma^2$ | $\psi_c$ Si(110) | $n_b$ Si(110), computed | forward $\hbar\omega$ for $\Delta E_\perp = 5{-}20$ eV |
| --- | --- | --- | --- | --- | --- | --- |
| 1 MeV | 2.96 | 1.34 MeV | 17.5 | 5.7 mrad | 1 | 0.09–0.35 keV |
| 3 MeV | 6.87 | 3.44 MeV | 94.4 | 3.6 mrad | 2 | 0.5–1.9 keV |
| 10 MeV | 20.6 | 10.5 MeV | 846 | 2.0 mrad | 3 | 4–17 keV |

using Si(110): $d_p = 1.92$ Å, $U_0 \approx 21$ eV at 293 K, and the critical angle $\psi_c = \sqrt{2 U_0 / pv}$. The $n_b$ column is the bound-state count from the implemented eigensolve at $k=0$ (`materials.channeling`, ledger `cr-transverse-eigenproblem`); the WKB estimate above overcounts at these energies. It rises to 5 at 30 MeV, 6 at 40 MeV (matching Azadegan 2013) and 9 at 100 MeV.

Three consequences frame the whole implementation:

1. **A handful of levels.** Spectra are discrete lines from transitions between a few bound states — exactly what Watson observed at 1–3 MeV. The states must come from an explicit eigensolve; there is no continuum approximation to fall back on.
2. **The photon energies land in or near the PyRITE detector band.** The Doppler upshift $2\gamma^2$ maps eV-scale transverse spacings to 0.1–20 keV photons, overlapping PXR/CBS lines. Channeling radiation is a physical foreground/background for the coherent spectra, not a separate curiosity.
3. **Electrons dechannel fast.** Because the bound states peak on the atomic planes, channeled electrons sit in the region of maximum nuclear and electronic density. Occupation lengths are of order a micron at these energies (Kephart), so the channeling module governs only the first $\lesssim$ few µm of path. The rest is ordinary native transport, entered through the handoff of §5.4.

## 1. Step 1 — Planar continuum potential

### 1.1 From crystal structure to $U(x)$

Averaging the crystal potential over the two directions in the plane kills every Fourier component except those with $\mathbf{g}$ normal to the plane. For plane $(hkl)$ with spacing $d_p$ and $g_n = 2\pi n / d_p$:

$$
U(x) = \sum_{n \neq 0} U_n\, e^{i g_n x}, \qquad
U_n = -e V_{\mathbf{g}_n},
$$

where $V_\mathbf{g}$ is the electrostatic Fourier coefficient of the full lattice at the reciprocal vector $\mathbf{g}_n$ normal to the plane. In the standard electron-microscopy convention,

$$
V_\mathbf{g} = \frac{h^2}{2 \pi m_e e\, \Omega}
  \sum_j f^{e}_j(s_g)\, e^{-B_j s_g^2}\, e^{-i \mathbf{g} \cdot \mathbf{r}_j},
\qquad s_g = \frac{g}{4\pi} = \frac{\sin\theta}{\lambda},
$$

with $\Omega$ the unit-cell volume and the prefactor equal to $47.878\ \mathrm{V\,Å^3}$ when $f^e$ is in Å and $\Omega$ in Å³. The sum over the basis is exactly the machinery already in `materials/crystal.py::structure_factor` — same extinction rules, same Debye–Waller placement — evaluated with *electron* scattering factors instead of X-ray ones.

### 1.2 Electron scattering factors via Mott–Bethe

The repository already carries source-backed X-ray form factors (`materials/atomic.py::atomic_form_factor`, Waasmaier–Kirfel $f_0$). The electron scattering factor follows from Poisson's equation (Mott–Bethe):

$$
f^{e}(s) = \frac{Z - f_x(s)}{8 \pi^2 a_0 s^2},
\qquad a_0 = 0.5292\ \text{Å},
$$

giving $f^e$ in Å. This reuses validated atomic data instead of introducing a second tabulation (Doyle–Turner would be an independent cross-check, not the primary source). Caveats to encode as tests: the formula is for neutral atoms (fine for our catalog); the $s \to 0$ limit is finite for neutral atoms because $Z - f_x(s) \to O(s^2)$, but it must be evaluated with the analytic limit, not numerically at $s = 0$; only the real part $f_0$ enters — the anomalous X-ray terms $f', f''$ describe photon absorption and do **not** belong in the electrostatic potential.

### 1.3 Thermal averaging and temperature dependence

Each coefficient carries the Debye–Waller factor $e^{-B_j s_g^2}$ with $B = 8\pi^2 \langle u_x^2 \rangle$ and $\langle u_x^2 \rangle$ the one-dimensional projected mean-square thermal displacement. Equivalently: the plane's atomic density profile is a Gaussian of width $u_\text{th} = \sqrt{\langle u_x^2 \rangle}$ ($\approx 0.077$ Å for Si at 293 K, $B \approx 0.47$ Å²), and the potential is the static-lattice potential convolved with it.

Channeling needs an explicit temperature model, because the catalog currently stores one room-temperature isotropic `B_ang2` per crystal. Use a Debye-model $B(T)$,

$$
B(T) = \frac{6 h^2}{m_a k_B \Theta_D}
  \left[ \frac{\phi(x_D)}{x_D} + \frac{1}{4} \right],
\qquad x_D = \frac{\Theta_D}{T},
$$

with $\phi$ the Debye integral and $m_a$ the atomic mass, anchored to the room-temperature $B$ already in the catalog. Well depth and level spacings shift by several percent between 100 K and 400 K, which is comparable to the line-position accuracy the anchors demand — so this is load-bearing, not cosmetic. Note the catalog's placeholder $B = 0.6$ Å² entries (flagged in the validation ledger) are *not* adequate for channeling profiles; the catalog therefore gains per-site thermal displacements, a reference temperature and a Debye temperature for eligible crystals (ADR-0018), while existing consumers keep reading the scalar.

### 1.4 Derived densities

Two more profiles fall out of the same construction and are needed later:

- **Nuclear (plane) density** $n(x)$: Gaussians of width $u_\text{th}$ on each plane position — drives dechanneling transition rates.
- **Electron density** $\rho_e(x)$: from Poisson, $\rho_e \propto \nabla^2 V$, i.e. multiply Fourier coefficients by $g_n^2$ — drives electronic scattering rates and the local stopping correction.

Sanity anchors for step 1: Si(110) depth $\approx 22$ eV and diamond(110) $\approx 25{-}30$ eV at room temperature; symmetric profile for centrosymmetric planes; $U_n \to 0$ for extinct reflections.

## 2. Step 2 — Relativistic transverse eigenproblem

### 2.1 The equation and where $\gamma m$ comes from

Starting from the Dirac equation in a potential $U(\mathbf{r})$ that depends only on $x$, squaring, and dropping terms of order $(U/E)^2$ and the spin coupling (both $\lesssim 10^{-5}$ here — see §9), the longitudinal motion separates and the transverse wavefunction obeys

$$
\left[ -\frac{\hbar^2}{2 \gamma m} \frac{d^2}{dx^2} + U(x) \right] u_k^{(n)}(x)
  = E_\perp^{(n)}(k)\, u_k^{(n)}(x).
$$

This is the Andersen–Bonderup–Pantell master equation: a Schrödinger problem with relativistic mass $\gamma m$. All relativistic kinematics enters through that single $\gamma$ (plus the Doppler map of §4). In working units the kinetic prefactor is $3.810 / \gamma\ \mathrm{eV\,Å^2}$ — nothing else in the solve knows about relativity.

### 2.2 Bloch structure

$U(x)$ is periodic with period $d_p$, so eigenstates are Bloch waves $u_k^{(n)}$ with transverse crystal momentum $k \in (-\pi/d_p, \pi/d_p]$ and band index $n$. Practical structure:

- **Deeply bound bands are flat** (negligible inter-well tunneling): solve at a single $k$ and treat levels as isolated-well states.
- **Near-barrier and above-barrier bands disperse** and must be sampled over the Brillouin zone; they carry the quasi-channeled population and the free→bound capture continuum (§9, item 2).

Numerically: plane-wave diagonalization in the $e^{i g_n x}$ basis (64–128 components converges eV-level spacings to well below linewidths) or central finite differences with Bloch boundary conditions. Both are cheap in 1D; run both once as a cross-check, keep one.

Required invariants (fast tests): orthonormality, parity for centrosymmetric planes, variational monotonicity with basis size, zero potential → free spectrum $\hbar^2(k+g_n)^2/2\gamma m$ and no bound states, and the $\sqrt{\gamma}$ growth of $n_b$.

### 2.3 Electron-specific structure (why this is not the positron problem)

Because wells sit on the planes, the low states are localized on the nuclei with strongly **anharmonic** spacing — nothing like the near-harmonic ladder positrons see between planes. Line identification therefore requires the actual eigenvalues; there is no useful harmonic approximation to check against. The harmonic limit is still a *test* (artificially deep smooth potential → equally spaced levels), just not physics.

## 3. Step 3 — Initial occupation

An electron entering at angle $\psi$ to the plane is a transverse plane wave $e^{i k_x x}$ with $k_x = p \psi / \hbar$ and transverse energy $E_\perp^{inc} = (pv/2)\, \psi^2$. Its decomposition over crystal eigenstates gives the entry populations:

$$
P_n(\psi) = \sum_k \left| \langle u_k^{(n)} | e^{i k_x x} \rangle \right|^2 ,
$$

which for Bloch states collapses onto the $k$ matching $k_x$ modulo $2\pi/d_p$. Averaging over the transverse entry point within one period is equivalent to the Bloch normalization — no separate impact-parameter loop is needed for an ideal surface. On top of this:

- **Beam divergence:** convolve $P_n(\psi)$ with the beam angular distribution (Gaussian $\sigma_\psi$; compare to $\psi_c$ — at 3 MeV, $\psi_c \approx 3.6$ mrad, so a 1 mrad beam populates selectively, a 10 mrad beam mostly feeds the continuum).
- **Surface refraction:** crossing the surface step of the mean inner potential $\bar{U} \sim 10{-}15$ eV changes $E_\perp$ by an amount comparable to level spacings only when $\psi \lesssim \sqrt{2\bar{U}/pv}$; include it as the constant $n=0$ term it is (shift of the energy zero), and document it — it slightly reweights populations near $\psi = 0$.

Unitarity: $\sum_n P_n + P_\text{continuum} = 1$ at entry, exactly.

## 4. Step 4 — Photon kinematics

Energy–momentum conservation for emission $i \to f$ with a photon at lab polar angle $\theta$ gives the resonance condition

$$
\hbar\omega \,(1 - \beta \cos\theta) = \Delta E_\perp
  \left[ 1 + O\!\left( \frac{\hbar\omega}{E} \right) \right],
\qquad \Delta E_\perp = E_\perp^{(i)} - E_\perp^{(f)},
$$

so for small angles

$$
\hbar\omega(\theta) \simeq \frac{2 \gamma^2\, \Delta E_\perp}{1 + \gamma^2 \theta^2}.
$$

The recoil correction is bounded by $\hbar\omega / E \lesssim 2.5 \times 10^{-3}$ at 10 MeV — include it as the first-order multiplicative factor and stop there.

Two consistency points with existing code:

- The Doppler denominator $1 - \beta\cos\theta \equiv 1 - \mathbf{v} \cdot \hat{\mathbf{n}} / c$ is the *same object* as in the ledgered `line-energy-dispersion` row, whose numerator sign was reconciled in #338 as a relabelling: the repository $\mathbf g$ is the photon's momentum transfer with its sign reversed. The channeling implementation must adopt one convention for $\hat{\mathbf{n}}$, the tilt sign (`docs/physics/geometry/tilt-convention.md`), and the frequency sign, and add a cross-check test that the two modules agree on the denominator for the same geometry.
- $\beta$ here is the *longitudinal* velocity. It differs from the total $\beta$ by $O(E_\perp / pv) \lesssim 10^{-5}$ — ignorable, but say so in the derivation docstring.

## 5. Step 5 — Depth evolution

The architecture is fixed by [ADR-0018](../../adr/0018-quantum-channeling-zone.md): an open-system evolution of the transverse state with Lindblad jump operators. Planar production uses a deterministic density matrix; axial, keV and two-electron work use Monte Carlo wave-function (MCWF) trajectories on the same operators. The diagonal rate equation of earlier drafts survives only as a limiting case (§5.3).

### 5.1 Lindblad master equation

Incoherent scattering acts on the transverse density matrix $\rho$ along depth $z$:

$$
\frac{d\rho}{dz}=-\frac{i}{\hbar v}\left[H_\perp,\rho\right]
+\sum_k\left(L_k\rho L_k^\dagger-\tfrac12\left\{L_k^\dagger L_k,\rho\right\}\right),
$$

with $H_\perp$ the transverse Hamiltonian of §2 and $v=\beta c$.

### 5.2 Jump operators

Each jump operator is built from the *fluctuating* part of the lattice potential (thermal displacements) and from electron-gas excitation:

$$
L_q^{\rm TDS}=\sqrt{n(x)\,\frac{d\sigma_{\rm TDS}}{dq}}\;e^{iqx},
\qquad
L_q^{\rm el}=\sqrt{\rho_e(x)\,\frac{d\sigma_{e}}{dq}}\;e^{iqx}.
$$

- $n(x)$ and $\rho_e(x)$ are the profiles of §1.4.
- The thermal diffuse scattering (TDS) cross section uses the Einstein model, $d\sigma_{\rm TDS}/dq\propto|f^e(q)|^2\,(1-e^{-2W(q)})$.
- The electronic cross section is the transverse-momentum-resolved form of the existing PENELOPE shell and plasmon data, in a local-density approximation.

This form is the Markov limit of the frozen-phonon/QEP treatment that is standard in electron microscopy. The limit is benign here because each atom is passed once and Einstein displacements are independent. Correlated-phonon effects are a recorded refinement.

The imaginary part of $H_{\rm eff}=H_\perp-\tfrac{i\hbar v}{2}\sum_k L_k^\dagger L_k$ reproduces the standard absorptive (Yoshioka) potential. That is a limiting-case check.

**Physical consequences.**

- States localized on the planes, which are the low states for electrons, have the *shortest* lifetimes. Near-barrier states live longer.
- The total dechanneling length scales roughly like $\gamma$.
- Small-$q$ (plasmon) jumps barely decohere nearby positions. The decoherence rate of a superposition separated by $\Delta x$ is $\Gamma(\Delta x)\propto\int w(q)\,[1-\cos(q\Delta x)]\,dq$.

Absolute rates are the least first-principles piece of the chain, so anchor them. The Kephart 17 MeV (and 54 MeV) silicon occupation lengths are a direct measurement. If a single global calibration factor proves necessary, it is declared and enters case identity and provenance; it is never buried.

### 5.3 Coherences, backends and the rate-equation limit

A plane wave populates Bloch states *coherently*. The density matrix has off-diagonal elements that oscillate with beat length $\lambda_{nm} = 2\pi \hbar \beta c / \Delta E_{nm}$ (pendellösung-like, tens to hundreds of nm here) and produce depth-oscillating yield in thin crystals. Both backends keep these coherences, so thin crystals stay inside the validity domain.

| Backend | Evolution | Cost and use |
| --- | --- | --- |
| Density matrix (planar production) | Split operator: a coherent step $U\rho U^\dagger$ in the eigenbasis, then an elementwise dissipator kernel on $\rho(x,x')$ in position space. | One noise-free solve also covers the beam-divergence ensemble ($\rho_0$ is the divergence-averaged mixed state). |
| MCWF (axial/keV/two-electron) | Each trajectory evolves under $H_{\rm eff}$ and jumps with probability $\delta z\,\langle L_k^\dagger L_k\rangle$. | $O(M)$ memory per trajectory on a real-space grid. Only ensemble observables are physical. |

When collisional dephasing dominates, the off-diagonal elements vanish, and the populations obey the rate equation

$$
\frac{dP_n}{dz} = \sum_m \left[ W_{m \to n} P_m - W_{n \to m} P_n \right]
  - W_{n \to \text{free}}\, P_n
  + \sum_{m > n} \frac{A_{m \to n}}{\beta c}\, P_m - \sum_{m < n} \frac{A_{n \to m}}{\beta c}\, P_n ,
$$

with golden-rule rates $W_{n\to m}=\sum_k|\langle u^{(m)}|L_k|u^{(n)}\rangle|^2$. That limit, together with MCWF–density-matrix ensemble equivalence, is a regression test. The radiative-cascade terms ($A$) are tiny compared with collisional rates, but they cost nothing and matter for line-ratio anchors.

### 5.4 Dechanneling and handoff

After each jump term, the state is projected onto the bound subspace $\mathcal B$ (bound states plus a declared near-barrier margin) and the free subspace $\mathcal F$. The $\mathcal F$ part leaves the zone:

- In the density-matrix backend it becomes a recorded flux $dP/dz\,dE_\perp$. Per-electron launch states (depth, angle $\sqrt{2E_\perp/pv}$, time $z/v$, energy) are sampled from that flux with keyed RNG streams.
- In MCWF the same projection is applied per trajectory.

Dechanneled electrons continue in the native transport cores; native multiple scattering is off inside the zone, so nothing is double-counted. Rechanneling of quasi-free electrons is omitted and recorded as a validity limit. Report the sensitivity of results to the near-barrier margin.

### 5.5 Conservation

At every depth, bound + handed-off + radiatively-shifted populations, plus the electrons that exited through a face, sum to 1 within solver tolerance. For the density matrix, this is trace + accumulated flux = 1.

## 6. Step 6 — Radiation emission

### 6.1 Dipole rates

Dipole approximation validity: in the average rest frame the photon has $\hbar\omega' = \gamma \Delta E_\perp \sim 10^2$ eV, wavelength $\sim 10^2$ Å, versus a state extent $\lesssim 1$ Å — excellent, and worth a one-line docstring justification. The rest-frame spontaneous rate for $i \to f$ is the textbook

$$
A'_{if} = \frac{e^2\, \omega'^3\, |x_{if}|^2}{3 \pi \epsilon_0 \hbar c^3},
\qquad x_{if} = \langle u^{(f)} | x | u^{(i)} \rangle ,
$$

with $\omega' = \gamma \Delta E_\perp / \hbar$. Parity gives the selection rule: odd $\Delta n$ between opposite-parity bands for centrosymmetric planes. The rule must *emerge* from the computed $x_{if}$, not be imposed. The lab rate per unit path is $dN_{if}/dz = A'_{if} / (\gamma \beta c)$, one factor of $\gamma$ for time dilation.

**Estimator.** The dipole rate is the long-wavelength, bound→bound limit of a generic first-order transition operator $\hat A_{\mathbf k,\epsilon}\propto\hat{\mathbf j}\cdot\epsilon^*e^{-i\mathbf k\cdot\mathbf r}$. Implement the general operator, so the same layer can later serve quantum PXR/CBS. Score the expected value on the evolved state:
$\langle\hat d_\omega^\dagger\hat d_\omega\rangle_{\rho(z)}$ per transition-frequency cluster, where
$\hat d_\omega=\sum_{E_i-E_f\approx\hbar\omega}x_{fi}\,\lvert f\rangle\langle i\rvert$
groups transitions that are degenerate within the linewidth.

- For nondegenerate lines, as in anharmonic electron wells, this reduces to $P_i(z)\,dN_{if}/dz$.
- For an equidistant (harmonic) ladder it stays exact. A harmonic-potential test therefore checks it against the classical dipole limit.
- Squaring per-trajectory MCWF field amplitudes across trajectories is not a valid estimator.

### 6.2 Angular distribution and polarization

Build the lab distribution by boosting the rest-frame dipole pattern rather than transcribing a final formula: rest-frame $dP'/d\Omega' = (3/8\pi) \sin^2\Theta'$ about the dipole axis $\hat{x}$, then the aberration map

$$
\cos\theta' = \frac{\cos\theta - \beta}{1 - \beta\cos\theta},
\qquad
\frac{d\Omega'}{d\Omega} = \frac{1}{\gamma^2 (1 - \beta\cos\theta)^2},
$$

with $\hbar\omega(\theta)$ from §4 tying angle to energy per line. The result is the familiar $1/\gamma$ forward cone with an azimuthal asymmetry about the plane normal. Planar CR is linearly polarized along the projection of $\hat{x}$ — propagate this as a per-component polarization fraction so it can feed the same detector machinery as PXR polarization if/when needed. Cross checks: integrating the boosted pattern over $\Omega$ returns $A'/\gamma$ exactly; the pattern reduces to isotropic-dipole at $\gamma \to 1$.

### 6.3 Line shapes

Physical widths, convolved in this order:

1. **Occupation (dominant for electrons):** Lorentzian with $\Gamma = \hbar \beta c\, (L_i^{-1} + L_f^{-1})$ from the collisional lifetimes of both states; $L \sim 1$ µm gives $\Gamma \sim 0.2$ eV on eV-to-tens-of-eV transitions — percent-level lifetime widths.
2. **Doppler-geometric:** the $\theta$-dependence of $\hbar\omega$ across the detector solid angle and beam divergence (reuse the existing detector-solid-angle machinery; this usually dominates the *observed* width).
3. **Beam energy spread:** $\delta\omega/\omega = s\,\delta\gamma/\gamma$ with $s = 2 + d\ln\Delta E_\perp/d\ln\gamma$ forward: the factor 2 from $\gamma^2$, reduced because $\Delta E_\perp$ falls with the transverse mass ($s = 1.5$ harmonic, 1.73 for Si(110) 1→0 at 17 MeV; `cr-line-shape`).
4. **Band dispersion** for transitions involving dispersive bands: width from the $k$-average of $\Delta E_\perp(k)$.
5. Natural radiative width: negligible; drop after a one-time estimate.

### 6.4 Yield assembly

$$
\frac{dN}{d(\hbar\omega)\, d\Omega}
  = \sum_{i \to f} \int_0^{t} dz\; P_i(z)\,
    \frac{dN_{if}}{dz\, d\Omega}\,
    \mathcal{L}_{if}(\hbar\omega - \hbar\omega_{if}(\theta))\,
    e^{-\mu(\omega)\, \ell_\text{exit}(z, \theta)} ,
$$

normalized to photons/eV/sr/electron at sample exit. In this formula $P_i(z)$ stands for the estimator of §6.1, which equals the population for nondegenerate lines. Photon attenuation reuses the existing stack absorption (`_stack_tau`). The yield is first exposed through a diagnostic API, then persisted as the additive channel `spec_channeling` once anchored (ADR-0018).

## 7. Axial channeling (experimental tier)

The same six steps with dimension bumped to 2:

- **Potential:** $U(x, y)$ from summing string potentials (the same $V_\mathbf{g}$ construction keeping all $\mathbf{g} \perp$ axis), thermally smeared in 2D. Axial wells for electrons are several times deeper than planar (order $10^2$ eV) → more states, harder spectra, photon lines a few times higher in energy at the same beam energy.
- **Eigenproblem:** 2D plane-wave basis over the transverse reciprocal mesh, sparse Lanczos/shift-invert for the lowest ~50 states. For an isolated string the potential is nearly cylindrical, so states classify as $(n, \ell)$ — 2D-atom-like 1s, 2p, 3d… labels with dipole selection rule $\Delta\ell = \pm 1$. The 2D string lattice breaks the symmetry and mixes $\ell$; report the cylindrical labels as *approximate* metadata, let the matrix elements decide the actual rates.
- **Polarization:** $\Delta\ell = \pm1$ transitions radiate circular polarization about the axis; unpolarized beams average to zero net helicity but the azimuthal pattern differs from planar — store polarization per component rather than assuming linear.
- **Evolution/emission:** the same operator layer, but the MCWF backend runs on a real-space grid (~$64^2$ points). At that size the density matrix ($\sim10^7$ entries) and eigenbasis jump tables become too large. Doughnut scattering (azimuthal randomization of quasi-free electrons around the axis) is represented only if the continuum sampling is fine enough. That is the main numerical risk, and a good reason the axial tier stays `experimental` with qualitative peak-order checks only.
- **Vortex (OAM) beams:** a vortex centered on a string projects onto states with $\ell'\equiv\ell \pmod m$, where $m$ is the string lattice's rotational order. An atom-sized $\ell=\pm1$ probe couples to on-column 2p states and keeps its helical phase for >50 nm ([Xin & Zheng 2012](https://arxiv.org/abs/1204.0021)). Off-center or wide vortices spread over many $\ell$, and $\langle L_z\rangle$ changes with impact parameter and depth ([Lubk et al.](https://arxiv.org/abs/1111.6050)). Uncontrolled impact-parameter averaging washes the effect out. A surviving $\ell$ imbalance biases $\ell\to\ell\mp1$ decays, giving net circular polarization and photon OAM in specific lines ([Korotchenko & Kunashenko 2024](https://arxiv.org/abs/2403.01930)). Planar potentials do not conserve $L_z$, so planar channeling carries no OAM mode.

Anchor: Watson's 1–3 MeV silicon axial lines (peak energies and ordering).

## 8. Ownership map: channeling zone and native transport

| Physics | Owner | Notes |
| --- | --- | --- |
| Amorphous/misaligned e⁻ transport | native transport cores | Includes multiple scattering, ionization, bremsstrahlung and secondaries. Range extension is #439/#440. |
| Transverse states, evolution, CR emission | channeling zone | This document; ADR-0018. |
| Incoherent scattering *of channeled electrons* | channeling zone (jump operators $L_k$) | Native multiple scattering is **off** inside the zone. The jump operators *are* the multiple-scattering model there, so running both double-counts. |
| Bremsstrahlung of channeled electrons | existing scorers on tagged straight zone segments | Close-collision bremsstrahlung during the ~µm channeled path is small. Channeling-enhanced bremsstrahlung is folded into the validity budget rather than modeled. |
| Energy loss in the channeled zone | channeling zone, as continuous loss | v1: amorphous stopping power over the µm-scale zone (the error is small because $\Delta T \ll T$). Refinement: scale per state by the $\rho_e(x)$ overlap. |
| δ-ray production in the zone | neither (folded into stopping) | Explicit limitation; acceptable because the channeled path ≪ the total path. |
| Dechanneled / never-channeled electrons | native transport, entered through the secondaries launch seam | One-way handoff of $(\mathbf r,\hat{\mathbf v},T,t)$ sampled from the dechanneling flux; event code `EVENT_DECHANNEL`. |
| PXR/CBS during the channeled zone | existing PXR/CBS kernel on tagged straight mean segments | Channeled-state modulation is an omission (§9, item 10). |
| Characteristic emission | existing scorer | Channeling-modulated (ALCHEMI-type) yields are a research diagnostic (#501). |
| $\gamma(z)$ drift feeding back into line energies | channeling zone, optional | Negligible over µm occupation lengths. |
| ≥200 MeV channeling | quasiclassical (Geant4 `G4ChannelingFastSimModel`/`G4BaierKatkov`) | External validation reference only, never a backend. |

The handoff is one-directional in v1. Once dechanneled, an electron does not re-enter the channeling zone. Rechanneling of near-aligned quasi-free electrons is real but second-order at these dechanneling lengths, and is recorded as a known omission in profile metadata.

## 9. Physics easily missed

Inventory of effects checked for this walkthrough, with include/exclude decisions to encode in metadata:

1. **Transition radiation at the entry/exit surfaces.** Energy per boundary $\sim \alpha \gamma \hbar\omega_p / 3$ with $\hbar\omega_p \approx 31$ eV for Si: ~1.5 eV at 10 MeV, spectrum extending to $\sim \gamma \hbar\omega_p$ (0.1–0.6 keV). That is $10^{-3}$–$10^{-2}$ photons/electron — the *same order as the CR lines* and sitting exactly on top of the 1–3 MeV planar band. Decision needed: add it as a separate additive channel, or compute a bound and document the exclusion (tracked as `cr-transition-radiation-bound`). Silently omitting it risks a real disagreement with soft-X-ray anchors. *#505:* excluded with a bound (`cr-transition-radiation-bound`): a few percent of the strong Si(110) 3 MeV line peaks, but ~10² times their photon count over the band.
2. **Free–free double counting and the free→bound continuum.** Radiative transitions between above-barrier (continuum) transverse states *are* coherent bremsstrahlung, which the PXR/CBS amplitude machinery already produces. The channeling module must **not** integrate free→free matrix elements, or CBS is counted twice. A regression test guards this boundary: a channeling module with zero bound states emits exactly nothing.
   - Bound→free transitions cannot *emit*. Emission lowers $E_\perp$, and free states lie above every bound level.
   - The emitting process is **free→bound radiative capture**. It gives a continuum with a low-energy edge near $2\gamma^2(U_{\rm barrier}-E_n)$ (forward), which does not double-count first-Born CBS, whose final states are plane waves.
   - v1 scores bound→bound lines only. Capture is first estimated as an upper bound, and scored only if it exceeds a few percent of the line yield in an anchor configuration. *#505:* at Kephart's Si(110) 17 MeV, 2.2 mrad the estimate is ~40% of the line yield (`cr-capture-bound`), so the promotion rule fires; scoring awaits a maintainer decision.
3. **Coherent-population (pendellösung) oscillations** in thin crystals (§5.3). Both evolution backends retain coherences, so thin crystals remain valid. The diagonal rate equation is only a limiting case.
4. **Temperature model** (§1.3). A Debye $B(T)$ is anchored to per-site catalog values. Materials with the placeholder $B = 0.6$ Å² are ineligible for channeling profiles.
5. **Beam energy spread** doubles into line width via $\gamma^2$ — §6.3.
6. **Surface refraction** by the mean inner potential — §3; small reweighting near exact alignment.
7. **Doppler-denominator convention coupling** to the `line-energy-dispersion` convention (reconciled in #338) — §4; add the cross-module consistency test *before* anchoring line positions, or the same ambiguity infects two subsystems.
8. **Planar validity requires axis avoidance.** A "planar" run aligned near a low-index axis is silently axial; validate that the beam direction keeps all major axes outside a few times $\psi_c^{axial}$, else refuse/warn.
9. **Spin and quantum-recoil corrections**: spin-flip rates suppressed by $(\hbar\omega/E)^2 \lesssim 10^{-5}$; recoil handled at first order — both documented as dropped with bounds, satisfying the derivation-docstring assumption rule.
10. **PXR/CBS from *channeled* electrons.** Channeled states sample the unit cell non-uniformly, which in principle modulates their PXR emission. v1 scores channeled-zone segments as straight mean segments in the existing PXR/CBS kernel, and records the modulation as an omission. The channeled fraction of the path is µm-scale. Recording the omission keeps the coherent and channeling modules from silently claiming inconsistent electron ensembles.

## 10. Proposed ledger rows

One row per load-bearing equation. `file::symbol` anchors are filled at implementation. Static physics lives under `materials/channeling/`; evolution, emission and handoff live under `montecarlo/channeling/` (ADR-0018).

| id | claim | source | anchor idea |
| --- | --- | --- | --- |
| `cr-mott-bethe-factor` | $f^e(s) = (Z - f_x)/(8\pi^2 a_0 s^2)$, neutral-atom, analytic $s\to0$ | Mott–Bethe; ICRU/IT-C conventions | cross-check vs Doyle–Turner tabulation |
| `cr-planar-continuum-potential` | plane-projected $U(x)$ from $V_\mathbf{g}$ with DW smearing | ABP 1983 | Si(110) depth ≈ 22 eV @ 293 K |
| `cr-debye-b-of-t` | Debye-model $B(T)$ anchored to catalog RT values | standard Debye–Waller theory | published Si $B(T)$ points |
| `cr-transverse-eigenproblem` | $[-\hbar^2/2\gamma m\, \partial_x^2 + U]u = E_\perp u$, Bloch BCs | ABP 1983; Dirac reduction | free limit; parity; $\sqrt{\gamma}$ level count |
| `cr-initial-occupation` | plane-wave → Bloch decomposition, divergence convolution | ABP 1983 | unitarity; selective vs flooded population regimes |
| `cr-lindblad-jump-operators` | Einstein-model TDS and local-density electronic jump operators | ABP 1983; Uggerhøj 2005; QEP/frozen-phonon limit | imaginary part reproduces the absorptive potential; state-ordering of lifetimes |
| `cr-density-matrix-split-operator` | coherent eigenbasis step + position-space dissipator | §5.3 derivation | zero-scattering unitarity; trace + flux = 1; frozen-phonon (abTEM) oracle |
| `cr-mcwf-unravelling` | $H_{\rm eff}$ evolution and jump probabilities | Dalibard–Castin–Mølmer | ensemble mean equals the density-matrix solver |
| `cr-dechanneling-collapse` | bound/free projection, flux, launch-state sampling | §5.4 | Kephart 17/54 MeV occupation lengths; margin sensitivity; calibration declared |
| `cr-zone-handoff-accounting` | energy, population and time continuity across the handoff | §5.5 | closure tests |
| `cr-doppler-line-energy` | $\hbar\omega = \Delta E_\perp / (1-\beta\cos\theta)$ + recoil bound | kinematics | consistency test vs `line-energy-dispersion` denominator |
| `cr-dipole-rate` | $A' = e^2\omega'^3|x_{if}|^2 / 3\pi\epsilon_0\hbar c^3$, $\omega' = \gamma\Delta E_\perp/\hbar$ | standard QED dipole | emergent parity selection rule |
| `cr-boosted-dipole-angular` | aberration-mapped $\sin^2\Theta'$ pattern + polarization | relativistic aberration | solid-angle integral returns $A'/\gamma$; $\gamma\to1$ limit |
| `cr-line-shape` | occupation Lorentzian ⊗ geometric ⊗ $2\delta\gamma/\gamma$ | §6.3 derivation | width ordering vs Watson linewidths |
| `cr-yield-assembly` | depth-integrated exit yield, photons/eV/sr/e⁻ | §6.4 | Watson 1–3 MeV Si peak energies |
| `cr-axial-2d-eigenproblem` | 2D sparse eigensolve, approximate $(n,\ell)$ labels | ABP 1983; Watson axial data | peak ordering only (experimental tier) |
| `cr-transition-radiation-bound` | TR yield bound at boundaries vs CR yield | Ginzburg–Frank | decision record: include or bounded-exclude |

## 11. End-to-end pipeline

```text
per (material, plane/axis, T):                      [cached profile]
  1. V_g from structure_factor machinery + Mott-Bethe f^e + B(T)
  2. U(x) [or U(x,y)], n(x), rho_e(x)
  3. eigensolve at gamma grid -> {E_perp, u, x_if, A_if}; jump kernels L_k

per case (beam E, divergence, tilt):
  4. initial state rho_0 (divergence-averaged; or psi_0 / product state)
  5. evolve rho(z) [planar] or MCWF trajectories [axial/keV] -> dechanneling flux
  6. sample per-electron handoffs -> native transport cores (EVENT_DECHANNEL)
  7. assemble line spectrum -> diagnostic API, later spec_channeling
  8. tagged zone segments + native segments -> existing PXR/CBS, brem, characteristic
     (free-free excluded from 7)
```

Steps 1–3 are per-profile and cacheable; the $\gamma$ grid is the only beam dependence. Steps 4–7 are cheap per configuration. For planar channeling, nothing in the chain needs Monte Carlo sampling except the handoff. The density-matrix module is deterministic given the beam distribution, which keeps it testable to solver tolerance rather than statistical tolerance.
