# Longitudinal quantum coherence and recoil in crystal radiation

**Status:** Research note for [#629](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/629), 8 November 2026, branched from `bc85fc373`. It adds a CPU reference model (`pyrite.quantum._longitudinal`, private), its regression anchors and a regime script. It changes no production model, persisted channel or default output. The three research claims are ledgered as `unverified` until a fresh-context verification.

The question is whether a single electron's *longitudinal* quantum state (a coherent superposition of energies) changes crystal radiation in a way a classical or energy-spread description cannot represent. A second question is where quantum recoil and longitudinal–transverse coupling enter. In short:

- **Static media hide longitudinal coherence from photon counts.** In a static crystal, every frequency-resolved single-electron photon count, and every photon–electron coincidence count that resolves (or integrates) the final electron energy, depends only on the electron's energy distribution $\rho_{nn}$. This holds with exact recoil and with any position-dependent (crystal) interaction. The coherences $\rho_{nm}$ appear only in the photon's two-frequency coherence, i.e. in time-resolved or phase-referenced detection.
- **Pair emission needs coherence spanning the photon energy.** An inter-electron pair term at photon energy $\hbar\omega$ requires single-electron coherence across $\hbar\omega$. Any state obeys $\lvert\operatorname{Tr}(\rho\hat M)\rvert\le\max\lvert A\rvert\sum_n\sqrt{p_np_{n-\ell}}$: the overlap of the beam's *measured* energy distribution with itself shifted by $\hbar\omega$. For a sub-eV-spread beam and photons above a few eV this is zero to any precision, so an ordinary bunch has no $N^2$ term. The issue's decision follows as a theorem.
- **Quantum recoil cuts off harmonic bunching.** Recoil appears as the Talbot phase of the comb. Ideal PINEM bunching at harmonic $\ell$ is $\lvert J_\ell(4\lvert g\rvert\sin(2\pi\ell z/z_T))\rvert$. It reduces to the classical klystron $\lvert J_\ell(\ell X)\rvert$ for $\ell z/z_T\to0$ at fixed $X$. Its maximum over drift is $\max_{x\le4\lvert g\rvert}\lvert J_\ell(x)\rvert$: it reaches the classical maximum once the coherent comb width $4\lvert g\rvert\hbar\Omega$ passes the first maximum of $J_\ell$ ($\approx\ell+0.81\ell^{1/3}$ quanta), and is exponentially suppressed once $\ell-4\lvert g\rvert\gg\ell^{1/3}$. Classical bunching has no such cutoff.
- **Recoil-exact channeling lines.** Forward, the line is $\hbar\omega\simeq E\,\Delta\varepsilon/\bigl(m^2c^4/2E+\Delta\varepsilon+\langle T_\perp\rangle_f\bigr)$. That is Baier–Katkov recoil plus a longitudinal slow-down term that the current `cr-doppler-line-energy` also omits. Off-axis, the recoil depends on photon azimuth. For Si(110) and diamond(110) the shifts are 0.06–0.37 % at 5–54 MeV, small against CR line widths.

## 1. Relation to existing work

| Existing work | What it owns | What this note adds |
| --- | --- | --- |
| [Quantum-coherent emission](quantum-coherent-emission.md) (#501): guard-rail, `quantum-pair-population-normalization` | transverse shaped/entangled states; the general pair expansion $N\operatorname{Tr}(\rho\hat M^\dagger\hat M)+N(N-1)\lvert\operatorname{Tr}(\rho\hat M)\rvert^2$ | the longitudinal (energy) analogue. The guard-rail's M1 loophole (position-dependent interaction) does **not** open the longitudinal channel; there is an explicit $\operatorname{Tr}(\rho\hat M)$ for energy combs with recoil, and a state-independent bound |
| #527 (open): transverse-recoil-resolved CXR, Shi 2024 reproduction | transverse recoil pathways and their observability | none. Longitudinal recoil here enters only as on-shell phase; the transverse basis stays with #527 |
| #249, `line-recoil-resonance`/`-formation`/`-validity-screen` | exact vacuum PXR/CBS root; fixed-momentum time mismatch | the fixed-*energy* (stationary slab) mismatch, which vanishes at the same root (tested) and is the natural kernel for energy eigencomponents |
| `coherent-inter-electron-decoherence`, `finite-footprint-longitudinal-decoherence`, [superradiant PXR](superradiant-pxr.md) | classical point-charge arrival-time form factors | the quantum factor multiplying them, and its bound from the energy spectrum (§3.2) |
| `cr-doppler-line-energy`, `cr-axial-fiber-angular` (no recoil) | no-recoil CR line energies and angular factor | the recoil-exact planar line, its first-order expansion and azimuthal anisotropy (§3.3) |
| [Longitudinal structure](../../physics/beam-transport/longitudinal-structure.md) | classical microtrain/compressed sampling, energy spread drawn independently of arrival | a quantum admissibility condition on that phase space (§6) |

## 2. Reduced model

**State.** One electron on an energy comb $E_n=E_0+n\hbar\Omega$ (orders $n$ consecutive), with a longitudinal density matrix $\rho_{nm}$, $\operatorname{Tr}\rho=1$. The definitions keep three things separate:

- **single-electron longitudinal coherence:** the off-diagonal $\rho_{nm}$, $n\ne m$;
- **classical energy spread:** the diagonal $p_n=\rho_{nn}$ (a mixture over $E_0$ is a further diagonal average);
- **inter-electron coherence:** the two-body $\rho^{(2)}$ of `quantum-pair-population-normalization`, here for separable i.i.d. electrons.

**Propagation.** In vacuum or in a static medium, an energy component is a stationary state. At fixed $z$ it carries $e^{ik_nz}$ with the exact on-shell $k_n=p(E_n)/\hbar$. The density at $(z,t)$ is $\sum_{nm}\rho_{nm}e^{i(k_n-k_m)z-i(n-m)\Omega t}$. Its $e^{-i\ell\Omega t}$ coefficient is the bunching coefficient $B_\ell(z)=\sum_n\rho_{n+\ell,n}e^{i(k_{n+\ell}-k_n)z}$ (`bunching_coefficient`).

**Emission operator.** First order in $e$, one photon in a mode $(\omega,\hat{\mathbf n},\lambda)$, initially empty field, static crystal of infinite mass. Energy is conserved exactly, so $\hat M\lvert E_n\rangle=A_n(\omega)\lvert E_n-\hbar\omega\rangle$. The amplitude $A_n$ is supplied by the caller (coupling, polarization and structure factors), with one physical kernel provided: for a laterally infinite slab of thickness $L$ at normal incidence,

```{math}
:label: eq-lcr-slab

A_n\propto\int_0^Le^{i\Delta_nz}\,dz,\qquad
\Delta_n=\bigl[p_{i}-\hbar g_z-\hbar k_z-p_{f,z}\bigr]/\hbar,\quad
p_{f,z}=\sqrt{p(E_n-\hbar\omega)^2-\hbar^2\lvert\mathbf g_\perp+\mathbf k_\perp\rvert^2},
```

in the `line-recoil-resonance` convention $\mathbf p_f=\mathbf p_i-\hbar\mathbf g-\hbar\mathbf k$ (`slab_mismatch_inv_ang`, `slab_formation_amplitude`). $\Delta_n=0$ exactly at the #249 vacuum root when the final electron moves forward, $p_i-\hbar g_z-\hbar k_z>0$, as always in the PXR/CBS regime (anchor test). This is the fixed-energy counterpart of the fixed-momentum `line-recoil-formation` mismatch.

**Normalization.** $\langle n\rangle$ per electron per photon mode carries the units of $\lvert A\rvert^2$. Absolute coupling, absorption, refraction and polarization sums stay with the existing kernels or with #527.

**Assumptions.** (A1) first-order, one photon, vacuum initial field; (A2) static medium, with no phonon created in the emission vertex (TDS decoherence is separate); (A3) a discrete comb with ideal teeth; a finite tooth width only broadens each line; (A4) emission at normal incidence for the slab kernel; (A5) the reference amplitudes are vacuum plane waves (no refraction); (A6) longitudinal coherences are not decohered between preparation and emission (see §3.4).

## 3. Results

### 3.1 Energy-diagonal emission

`Validation: longitudinal-energy-diagonal-emission` — research claim; status in the [core coherent-physics ledger](../../validation/ledger-core-coherent-physics.md).

The joint state after emission is $\sum_nc_n\int d\omega\,A_n(\omega)\lvert E_n-\hbar\omega\rangle\lvert\omega\rangle$. Tracing the electron gives

```{math}
:label: eq-lcr-photon

\rho_{\rm ph}(\omega,\omega')=\sum_{n,m}\rho_{nm}A_n(\omega)A_m^*(\omega')\,
\delta\bigl(E_n-E_m-\hbar(\omega-\omega')\bigr).
```

At $\omega=\omega'$ only $n=m$ survives: the photon spectrum $\sum_np_n\lvert A_n(\omega)\rvert^2$ is independent of every $\rho_{nm}$. The proof uses only energy conservation, so it holds for any time-independent Hamiltonian, including the crystal's position-dependent potential and exact recoil. It extends to joint photon–electron (EELS-coincidence) counts, which are diagonal in the final electron energy, and to any frequency-integrated count. The coherences live at $\omega-\omega'=(n-m)\Omega$. They appear as photon intensity modulation at harmonics of $\Omega$ in time-resolved detection,

$$I(t)\supset\sum_\ell e^{-i\ell\Omega t}\int d\omega\sum_n\rho_{n,n-\ell}A_n(\omega)A^*_{n-\ell}(\omega-\ell\Omega),$$

or in interference with a field phase-locked to $\Omega$. This is the longitudinal form of the #501 guard-rail and of the photon-state dependence of Zhang et al. (arXiv:2401.05978).

`photon_density_matrix` evaluates {eq}`eq-lcr-photon` on a commensurate grid by an explicit partial trace over final electron energies. It does not assume the result. The tests compare a pure state, its dephased mixture and a re-phased pure state, with exact-recoil slab amplitudes.

### 3.2 Inter-electron pair term and quantum bunching

`Validation: longitudinal-quantum-bunching` — research claim; status in the [core coherent-physics ledger](../../validation/ledger-core-coherent-physics.md).

For separable i.i.d. electrons, photons at $\omega=\ell\Omega$ acquire the pair term $N(N-1)\lvert\operatorname{Tr}(\rho\hat M)\rvert^2$ with

```{math}
:label: eq-lcr-pair

\operatorname{Tr}(\rho\hat M)=\sum_n\rho_{n,n-\ell}A_n,\qquad
\lvert\operatorname{Tr}(\rho\hat M)\rvert\le\max_n\lvert A_n\rvert\sum_n\sqrt{p_np_{n-\ell}} .
```

The bound follows from positivity, $\lvert\rho_{n,n-\ell}\rvert^2\le p_np_{n-\ell}$ (`pair_amplitude_bound`). It is attained by the pure state with amplitudes $\sqrt{p_n}$ when $A$ is constant. A transform-limited Gaussian of coherent rms energy width $\sigma$ attains $e^{-(\hbar\omega)^2/8\sigma^2}$. That equals a classical charge distribution of rms duration $\hbar/2\sigma$, so $\lvert F_1\rvert^2=e^{-\omega^2\hbar^2/4\sigma^2}$. Consequences:

- **Ordinary bunches.** A finite bunch duration from random arrival times is a classical mixture. It multiplies the pair term by the classical form factor already on record. It does not make each electron short. Independently of preparation, a beam with total energy spread $\sigma_E$ has $\lvert F_1\rvert^2\le e^{-(\hbar\omega)^2/4\sigma_E^2}$. For a Gaussian spectrum with $\sigma_E\le1.5$ eV this is $\le2.3\times10^{-121}$ at 50 eV (§5). No $N^2$ term arises at CXR energies.
- **Phase-locked combs (PINEM).** Ideal PINEM amplitudes are $c_n=J_n(2\lvert g\rvert)e^{in\arg g}$. A quadratic expansion of the on-shell phase gives the Talbot length $z_T=4\pi(pc)^3\hbar c/[(mc^2)^2(\hbar\Omega)^2]$. Neumann–Graf addition then gives

  ```{math}
  :label: eq-lcr-bunching

  \lvert B_\ell(z)\rvert=\bigl\lvert J_\ell\bigl(4\lvert g\rvert\sin(2\pi\ell z/z_T)\bigr)\bigr\rvert .
  ```

  With $X=8\pi\lvert g\rvert z/z_T$ held fixed and $\ell z/z_T\to0$, this becomes the classical ballistic bunching $\lvert J_\ell(\ell X)\rvert$, with error $O((z/z_T)^2)$ (tested). The classical $X$ is $\hbar\Omega\,\lvert k''\rvert\,\Delta E\,z$ with $\Delta E=2\lvert g\rvert\hbar\Omega$. Recoil enters through the sine. At the classical optimum $X_{\rm opt}$ the quantum parameter is $2\pi\ell z/z_T=X_{\rm opt}\hbar\omega/(2\Delta E)$, with $X_{\rm opt}\to1$ for large $\ell$ (1.08 at $\ell=32$). Since $\lvert\sin\rvert\le1$, the maximum over drift is $\max_{x\le4\lvert g\rvert}\lvert J_\ell(x)\rvert$. Through the Airy transition near $\ell\approx4\lvert g\rvert$ this falls from the classical maximum $\approx0.68\,\ell^{-1/3}$, then becomes exponentially small for $\ell-4\lvert g\rvert\gg\ell^{1/3}$. So coherent emission at $\hbar\omega$ needs a coherent comb at least about $\hbar\omega$ wide, while classical bunching predicts $\approx0.68\,\ell^{-1/3}$ at any harmonic.

The exact-$k$ sum agrees with {eq}`eq-lcr-bunching` within $2\times10^{-6}$ in the tests. The observed $\le7\times10^{-7}$ is float64 cancellation in $k_{n+\ell}-k_n$ at $z\sim10^8$ Å; the omitted cubic dispersion contributes $\le6\times10^{-9}$ for $z/z_T\le0.05$.

### 3.3 Recoil-exact planar channeling lines

`Validation: cr-recoil-line-energy` — research claim; status in the [channeling ledger](../../validation/ledger-channeling.md).

This uses the paraxial planar continuum model of the #56 engine: plane normal $x$, free in-plane $y$, channel axis $z$. The model has $(p_zc)^2+(p_yc)^2=E^2-m^2c^4-2E\varepsilon(E)$, with $\varepsilon$ an eigenvalue of $H_\perp=p_x^2c^2/2E+U(x)$; this is the mass-$\gamma m$ Hamiltonian of `cr-transverse-eigenproblem`. Energy and the $y$, $z$ momenta are conserved. The static planes absorb $\hbar k_x$ without energy: line positions are exact, and matrix elements stay at transverse dipole order, neglecting $O(k_xa)$ corrections. The final level is evaluated at $E'=E-\hbar\omega$. Hellmann–Feynman gives $\partial\varepsilon/\partial\ln E=-\langle T_\perp\rangle$ ($U$ does not depend on $E$), so $\varepsilon_f(E')\simeq\varepsilon_f-\langle T_\perp\rangle_f\ln(E'/E)$ (linearized; error $O((\hbar\omega/E)^2)$). The conservation equation is then

```{math}
:label: eq-lcr-cr

\hbar\omega\bigl[2(E-p_zc\cos\theta)-2\varepsilon_f(E')\bigr]-\sin^2\theta\cos^2\phi\,(\hbar\omega)^2
=2E\bigl[\varepsilon_i-\varepsilon_f(E')\bigr],
```

with $\theta$ measured from $z$ and $\phi$ from $x$. `channeling_recoil_line_energy_eV` returns its small root, which matches a 50-digit mass-shell solve to $10^{-14}$. To first order, with $\omega_D=\Delta\varepsilon/(1-\beta\cos\theta)$ the current Doppler line,

```{math}
:label: eq-lcr-cr-first

\frac{\omega}{\omega_D}-1\simeq-\frac{\hbar\omega_D}{E}\Bigl[1+\frac{\langle T_\perp\rangle_f}{\Delta\varepsilon}-\frac{\sin^2\theta\cos^2\phi}{2(1-\beta\cos\theta)}\Bigr]+O(\varepsilon/E).
```

- **Recoil.** The leading 1 is Baier–Katkov recoil. Forward with $\langle T_\perp\rangle_f=0$, the exact root equals $\omega_D/(1+\hbar\omega_D/E)$ to $O(\varepsilon/E)$.
- **Longitudinal slow-down.** The $\langle T_\perp\rangle_f$ term is ħ-free when written as $2\gamma^2\langle T_\perp\rangle/E$. It is the channeling analogue of the undulator $K^2/2$, a classical longitudinal–transverse coupling also absent from `cr-doppler-line-energy`.
- **Azimuthal anisotropy.** The last term makes recoil depend on azimuth. A photon emitted toward the plane normal ($\phi=0$) leaves its transverse momentum to the lattice, so the electron does not pay that recoil energy. A photon emitted in the plane ($\phi=\pi/2$) recoils the electron.

Absolute level offsets enter only at $O(\varepsilon/E)$, the paraxial accuracy. The forward fixed-level root differs from the Baier–Katkov form by exactly $\lvert\varepsilon_i\rvert/E$. The off-axis $\phi$ term is of the same order as BK recoil at $\gamma\theta\sim1$, while the standard BK substitution has no $\phi$ dependence. Baier & Katkov (1968) confirm the forward form. Their Eq. (17) carries recoil only through the phase factor $\omega\to\omega E/E'$, $E'=E-\hbar\omega$. For periodic motion this gives $\omega_D(\theta)/(1+\hbar\omega_D(\theta)/E)$ at every angle, with no $\phi$ dependence. It reaches that form after replacing the trajectory operators by their classical values (text after their Eq. 16), so it describes emission from a classical transverse trajectory, not a line between two bound levels. For a free transverse direction the $(\hbar k_\perp)^2$ term cancels exactly in the mass-shell identity. Only the bound plane-normal component survives, which is why the per-level line differs from BK off axis. Whether a level-weighted centroid recovers BK through the $k_x$-kick sum rule is not shown here. The forward result does not depend on it.

### 3.4 Decoherence of longitudinal coherence (assumption, not derived here)

{cite:t}`velasco2026decoherence` (Sec. II) show that in the nonrecoil approximation the electron decoherence probability "is independent of the longitudinal electron wave function". An inelastic event shifts every comb component by the same loss, and so preserves $\rho_{nm}$ up to recoil corrections of relative order $\hbar\Omega/E$. A6 therefore treats energy coherences as robust to inelastic and TDS scattering. Transverse decoherence (`quantum-decoherence-budget`) still applies to transverse degrees of freedom. A quantitative bound on the recoil corrections is not derived here.

## 4. Validation cases

All anchors live in `tests/montecarlo/quantum/test_longitudinal_coherence_reference.py`.

| Case | Limit or reference | Tolerance |
| --- | --- | --- |
| pure vs dephased vs re-phased state, exact-recoil slab amplitudes | photon counts equal and equal to $\sum_np_n\lvert A_n\rvert^2$ | $10^{-13}$ of max |
| photon two-frequency coherence | nonzero only at multiples of the comb, equal to the closed form; zero for mixtures | exact / $10^{-12}$ |
| slab mismatch | zero at the #249 exact root (three geometries), positive slope | roundoff |
| Talbot length | $4\pi/\lvert k_{+1}+k_{-1}-2k_0\rvert$ from exact on-shell $k$ | $10^{-5}$ |
| PINEM bunching | Neumann–Graf closed form, exact $k$ | $2\times10^{-6}$ |
| PINEM attosecond focus | peak density at Priebe et al.'s 2.75 mm ($g=3.95$, 120 keV, 800 nm) | 0.1 mm |
| classical limit | $\to\lvert J_\ell(\ell X)\rvert$ with $O((z/z_T)^2)$ convergence | $2\times10^{-6}$ |
| quantum cutoff | $\max_z\lvert B_{30}\rvert<10^{-5}$ for $\lvert g\rvert=3$, where classical gives $>0.15$ | — |
| Gaussian wavepacket; dephased mixture | $e^{-\ell^2/8\sigma^2}$; zero | $10^{-12}$ |
| pair bounds | $\lvert\operatorname{Tr}\rho\hat M\rvert^2\le\operatorname{Tr}\rho\hat M^\dagger\hat M$; energy-overlap bound, attained | $10^{-12}$ |
| CR root | 50-digit mass-shell solve (five cases, incl. off-axis/azimuth) | $10^{-14}$ |
| CR limits | Baier–Katkov forward; first-order {eq}`eq-lcr-cr-first`; anisotropy; forbidden raising branch | $5\times10^{-6}$ / 1 % of shift |

## 5. Regime estimates

`checks/longitudinal_coherence_regimes.py` (under a second):

- **PINEM Talbot length** at $\hbar\Omega=1.55$ eV: 22 mm at 30 keV, 0.48 m at 200 keV, 0.99 m at 300 keV, 650 m at 5 MeV.
- **Maximum harmonic bunching** over drift. Classical column is $\max_X\lvert J_\ell(\ell X)\rvert$; the quantum columns are $\max_{x\le4\lvert g\rvert}\lvert J_\ell(x)\rvert$ from {eq}`eq-lcr-bunching`:

| $\hbar\omega$ | $\ell$ | $\lvert g\rvert=10$ | $\lvert g\rvert=100$ | $\lvert g\rvert=300$ | classical |
| --- | --- | --- | --- | --- | --- |
| 15.5 eV | 10 | 0.303 | 0.303 | 0.303 | 0.303 |
| 50 eV | 32 | 0.209 | 0.209 | 0.209 | 0.209 |
| 277 eV (C K) | 179 | 7e-96 | 0.119 | 0.119 | 0.119 |
| 1.5 keV | 968 | ≈2e-1213 | 2e-263 | 0.068 | 0.068 |

  Coherent soft-X-ray pair emission from PINEM-modulated electrons needs $\lvert g\rvert\gtrsim\ell/4$: hundreds for the 50 eV–1.5 keV band. That is at the edge of reported strong-coupling PINEM, and it must then survive the decoherence of A6.

- **Recoil-exact CR**, lowest bound–bound line at $k=0$, 293 K. Levels use the solver's $U_0=0$ reference; referring them to vacuum moves each shift by at most 0.5 % of itself (Si, 5 MeV):

| crystal | $T$ | $\Delta\varepsilon$ | $\langle T_\perp\rangle_f$ | $\hbar\omega_D$ | forward shift | of which BK | anisotropy at $\theta=1/\gamma$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Si(110) | 5 MeV | 10.35 eV | 2.77 eV | 2.40 keV | −5.5e-4 | −4.4e-4 | 1.1e-4 |
| Si(110) | 20 MeV | 7.54 eV | 1.96 eV | 24.3 keV | −1.5e-3 | −1.2e-3 | 3.0e-4 |
| Si(110) | 54 MeV | 5.50 eV | 1.40 eV | 125 keV | −2.9e-3 | −2.3e-3 | 5.7e-4 |
| diamond(110) | 20 MeV | 9.47 eV | 2.49 eV | 30.5 keV | −1.9e-3 | −1.5e-3 | 3.7e-4 |
| diamond(110) | 54 MeV | 7.13 eV | 1.84 eV | 162 keV | −3.7e-3 | −3.0e-3 | 7.4e-4 |

  Diamond(110) has fewer than two bound states at 5 MeV. Below 1 MeV the bound–bound lines vanish (#501 §3.2), and recoil is below $10^{-4}$.

## 6. Experimental discriminants

1. **Null test of §3.1.** Spectra of PINEM-modulated electrons and of electrons with the same energy distribution but randomized laser phase must be identical behind a crystal, including in EELS coincidence. A difference would indicate stimulated emission, a time-dependent target or detector timing, not single-electron longitudinal coherence.
2. **Phase-locked photon timing (§3.1).** X-ray intensity modulation at harmonics of $\Omega$, phase-locked to the modulating laser and measured by laser-referenced streaking or cross-correlation. Its amplitude follows the photon coherence of {eq}`eq-lcr-photon`.
3. **Harmonic cutoff and Talbot dependence (§3.2).** For a train of identically modulated electrons, measure coherent emission at $\ell\Omega$ against drift distance and coupling. Quantum theory predicts revivals with period $z_T/(2\ell)$ and a cutoff at $\ell\approx4\lvert g\rvert$; classical klystron theory predicts neither.
4. **CR recoil (§3.3).** A 0.1–0.4 % redshift of MeV CR lines, with the $\langle T_\perp\rangle$ share, and the azimuthal anisotropy at $\theta\sim1/\gamma$. Typical line widths are of order 10 %, so this needs centroid fits at high statistics. The shift is below the current model's other uncertainties.

## 7. Follow-on implementation plan

Proposed only; no production change here.

- **I1 — quantum admissibility screen for coherent pair terms (radiation contract).** The coherent routes using `compressed`/`microtrain` sample arrival times independently of energy spread. They can therefore represent per-electron phase space with $\sigma_t\sigma_E<\hbar/2$, and they credit pair terms at $\hbar\omega\gg\sigma_E$ that {eq}`eq-lcr-pair` forbids. Contract: the pair reducer receives the case's single-electron energy distribution and records `F1_bound = sum sqrt(p_n p_{n-l})` (Gaussian: $e^{-(\hbar\omega)^2/8\sigma_E^2}$) per photon energy, as strict-JSON provenance. It warns, or refuses under a strict policy, where the classical factor exceeds the bound. Identity: no change while record-only. Applying the factor changes default output and bumps the coherent model marker. Owners: `coherent-inter-electron-decoherence` and the `optical-ctr-population` reducers.
- **I2 — recoil-exact CR line centres (radiation contract).** Replace `line_energy_eV` in the planar CR source by {eq}`eq-lcr-cr` with Hellmann–Feynman $\langle T_\perp\rangle_f$ from the solved states, and update the Doppler/Jacobian consistency of `cr-line-shape`. Start record-only, with a shift-fraction screen mirroring `line-recoil-validity-screen`. Default-changing integration bumps the channeling model marker. A low priority given §5.
- **I3 — longitudinal state input (persistence contract), conditional.** Only once a phase-locked modulated source is in scope. The beam/quantum state would carry comb spacing, orders, complex amplitudes or $\rho_{nm}$, and the modulating-field phase reference, persisted with the case identity. The photon output would need a two-frequency or time-resolved channel ($\rho_{\rm ph}(\omega,\omega+\ell\Omega)$), which no current result container represents. Gate: a concrete source specification and a measurement with laser-referenced timing.

## Sources

Inspected: {cite:t}`huang2023quantumrecoil` (Supplementary S1, exact vacuum recoil root, matching `line-recoil-resonance`; S8 recoil in post-selected EELS); {cite:t}`velasco2026decoherence` (nonrecoil decoherence independent of the longitudinal wavefunction); {cite:t}`shi2024transverse` via the #249 record; and the #501 sources.

Not yet inspected; derived independently here, to verify against the papers before `anchored`:

- Feist et al., Nature 521, 200 (2015), Eqs. (13)–(15): $S=\exp(g a^\dagger-g^*a)$, so $\langle N\rvert S\lvert0\rangle=e^{iN\arg g}J_N(2\lvert g\rvert)$ and the cutoff is $2\lvert g\rvert\hbar\Omega$. Methods Eq. (16) propagates with the exact relativistic dispersion, as here.
- Priebe et al., Nat. Photon. 11, 793 (2017): attosecond trains after 1.5 mm drift; shortest pulses expected at 2.75 mm for $g=3.95$ (120 keV, 800 nm). Neither paper states $z_T$ in closed form. Our exact-$k$ propagation puts peak density at 2.75 mm (tested), which fixes both the factor 2 in $J_n(2\lvert g\rvert)$ and the $z_T$ convention.
- Baier & Katkov, Sov. Phys. JETP 26, 854 (1968), Eqs. (16)–(17) and (26): the $\omega\to\omega E/E'$ recoil phase, applied along the classical trajectory.
- Kfir et al., Sci. Adv. 7, eabf6380 (2021): the cathodoluminescence intensity and spectrum do not depend on the electron state; phase correlations with the modulating laser, extending to its harmonics, do.
- Karnieli et al., Sci. Adv. 7, eabf8096 (2021): the diagonal photon spectrum does not depend on the wavefunction; the off-diagonal spectral coherence does.
- Zhang et al., arXiv:2401.05978: spontaneous rate independent of wavefunction shape; photon-state dependence.
