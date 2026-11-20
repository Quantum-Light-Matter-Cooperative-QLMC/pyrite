# Strain and acoustic modulation of CBS, channeling radiation and PXR (#1058)

**Status:** Research note for [#1058](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1058), 20 November 2026. It changes no production model, default or ledger status. Stage-0 numbers come from `checks/strain_ensemble_1058.py`, which averages existing surfaces (`photon_energies`, `chi_g`, `absorption_length_ang`, `channeling_spectrum`) over a frozen distribution of local tilt and lattice parameter. The record is `docs/validation/check-records/strain-ensemble-1058.json` (run at `ee485009f` on local CPU; about 2 min per planar case).

**Question.** Can static or piezo-driven strain create Bragg conditions, raise emission, tune frequency, or select modes and polarization in coherent bremsstrahlung (CBS), planar channeling radiation (CR) and parametric X-ray radiation (PXR)?

**Short answer.**

| Mechanism | Lever at strain or tilt $10^{-4}$–$10^{-3}$ | Verdict |
|---|---|---|
| CBS, local plane tilt | $\delta E/E=\delta\psi/\tan\psi$: 2–20% at $\psi=5$ mrad | **Go** for line steering and envelope broadening. No enhancement: a symmetric tilt ensemble only lowers and widens the peak. |
| CBS, coherent sidebands | Weights $J_m^2(g u_0)$ at spacing $2\gamma^2\hbar c\,q_z$ | **No-go** as a separate observable at MeV. The sidebands carry the same second moment as the tilt ensemble and sit far inside the divergence width. |
| Planar CR, entrance tilt | Order-unity changes in line yields and ratios for tilts $\gtrsim0.1\psi_c$ | **Go** for intensity and line-ratio modulation (§5). |
| Planar CR, line energy | $\delta E/E\approx-1.5\,\varepsilon$ from strain; $(\gamma\delta\psi)^2$ from observation tilt | **No-go**: below every measured width. |
| PXR, d-spacing and tilt | $\delta E/E=-\varepsilon$; $\delta E/E=-\cot\theta_B\,\delta\psi$ | **Fine tuning or lock-in only.** |
| PXR, acoustic satellites | $\sum_m J_m^2=1$ | **No enhancement** in the kinematic limit. Satellites at $\delta E/E\sim d/\Lambda\lesssim10^{-4}$ are unresolved. |
| PXR, extinction relief | Real-photon reflectivity rises when tilt amplitude exceeds the Darwin width | **No-go for PXR yield at MeV.** Only a fraction $\sim\gamma\,\omega_D\approx10^{-4}$–$10^{-3}$ of the PXR cone lies within the Darwin band. Relief applies to diffracted *real* photons (DTR, diffracted bremsstrahlung). |
| Static superlattice (Ashrafi 1991) | Two CR families separated by the kink angle | **Go as a modeling target**; it needs a depth-dependent potential (§7). |
| Resonant hypersound, circular polarization | Helical shear wave at the beat length (≈127 GHz, Si ⟨100⟩ 3.5 MeV) | **No-go now.** It is selective in principle, but needs $\gtrsim4\times10^{-4}$ circularly polarized shear strain at ≈130 GHz to beat dephasing (§8). |

The Mkrtchyan et al. (2013) shifts do not fit a strain mechanism. Their pattern fits a change in the energy scale (§6).

## 1. Regime: the crystal is frozen

A 1 µm path takes 3.3 fs, and a 1 mm path 3.3 ps. Piezo frequencies are ≲ few GHz (period ≳ 100 ps), so each electron sees a static displacement field $\mathbf u(\mathbf r)$. The time dependence only enters through the ensemble of electrons arriving at different phases. Two limits follow from the acoustic wavelength $\Lambda=v_s/f$ (Si shear 5.84 km/s gives Λ = 5.8 µm at 1 GHz; diamond 12.8 µm; quartz 3.7 µm):

- **Wave across the beam** ($\mathbf q\perp\hat u$), or a path much shorter than Λ: each electron sees a locally uniform lattice rotation δψ (for a shear wave, the antisymmetric part of $\nabla\mathbf u$) and strain $\varepsilon$. The observable is an incoherent average over the beam spot (≫ Λ) of the unstrained spectrum at the local $(\psi+\delta\psi,\ a_0(1+\varepsilon))$, with $a_0$ the lattice parameter. For a sinusoidal wave of tilt amplitude $a$, the distribution of $\delta\psi$ is the arcsine law $p(\delta\psi)=1/(\pi\sqrt{a^2-\delta\psi^2})$. Gauss–Chebyshev nodes integrate it exactly for polynomial responses.
- **Wave along the beam**, with path $L\gg\Lambda$: the electron samples a periodic displacement and the lattice acquires satellites (§2).

The single-electron transverse coherence width (Å to nm) is far below Λ. Acoustic orbital-angular-momentum structure on the µm scale therefore cannot imprint photon OAM on single-electron emission; that stays out of scope, as the issue says.

Direct piezoelectric fields are negligible. In quartz, $e_{11}\approx0.17$ C m⁻² at strain $10^{-4}$ gives $E\approx e\varepsilon/\epsilon\approx4\times10^{5}$ V/m. That is $4\times10^{-5}$ eV/Å against planar fields of ≈ 10 eV/Å, a ratio of $4\times10^{-6}$.

**Systematic floor.** Thermal expansion gives $\alpha\Delta T$: $2.6\times10^{-6}$/K (Si), $1.0\times10^{-6}$/K (diamond), $\approx1.3\times10^{-5}$/K (quartz). Beam or RF heating of tens of kelvin therefore matches the achievable acoustic strain in Si and exceeds it in quartz. Thermal bowing of a plate also tilts it.

## 2. Displaced lattice: satellites and the quasi-static limit

**Source equation.** For $\mathbf u(\mathbf r)=\mathbf u_0\sin(\mathbf q\cdot\mathbf r)$ the Fourier component of the lattice potential at $\mathbf g$ becomes (Jacobi–Anger, $e^{iz\sin\phi}=\sum_m J_m(z)e^{im\phi}$)

$$
U_{\mathbf g}\,e^{-i\mathbf g\cdot\mathbf r}\;\to\;U_{\mathbf g}\sum_{m=-\infty}^{\infty}J_m(\mathbf g\cdot\mathbf u_0)\,e^{-i(\mathbf g-m\mathbf q)\cdot\mathbf r}.
$$

Each reflection splits into satellites $\mathbf g-m\mathbf q$ with amplitude weights $J_m(\mathbf g\cdot\mathbf u_0)$. The resonance $E=\hbar\,\mathbf v\cdot\mathbf G/(1-\beta\hat n\cdot\hat u)$ for $\mathbf G=\mathbf g-m\mathbf q$ places them at spacing $\Delta E=\hbar\,\mathbf v\cdot\mathbf q/(1-\beta\hat n\cdot\hat u)$. Kinematic intensities scale as $|U_{\mathbf G}|^2$, so

$$
\sum_m J_m^2(z)=1\qquad\text{(total weight conserved)},\qquad \sum_m m^2J_m^2(z)=z^2/2.
$$

**Equivalence with the tilt ensemble.** For a shear wave along the beam, $\mathbf g\cdot\mathbf u_0=g\,a/q$ with tilt amplitude $a=qu_0$. The sideband second moment, $(\Delta E)^2 z^2/2=(\hbar c\beta g a/(1-\beta))^2/2$, then equals the second moment of the arcsine tilt ensemble, $(\partial E/\partial\psi)^2a^2/2$. The check confirms this to machine precision for every row (`sideband_rms_eV` = `arcsine_rms_eV`). Sidebands are resolvable only when the intrinsic line width is below $\Delta E$, that is, when the coherent radiating length exceeds Λ and the divergence width is below one sideband spacing. Otherwise the satellite comb and the local-tilt ensemble give the same spectrum.

**Limits.** $u_0\to0$ gives $J_0=1$, the unstrained lattice. $q\to0$ at fixed $a$ gives a dense comb and so the quasi-static ensemble. $|\mathbf g\cdot\mathbf u_0|\gtrsim1$ depletes the main line as $J_0^2$, a coherent analogue of a Debye–Waller factor.

## 3. CBS

**Source equation.** Forward planar CBS from a reflection of spacing $d$ at angle ψ between beam and planes (Ter-Mikaelian; Überall; PyRITE's CXR resonance at $\theta_{\rm obs}=0$):

$$
E_1=\frac{\hbar c\beta\,g\sin\psi}{1-\beta}\approx2\gamma^2\hbar c\,g\,\psi,\qquad g=2\pi/d .
$$

Local tilt therefore gives $\delta E/E=\delta\psi\cot\psi$. Normal strain gives $\delta E/E=-\varepsilon$, smaller by the factor $\psi$.

**Stage 0** (beam divergence Gaussian, RMS as stated; arcsine tilt ensemble):

| Case | $E_1$ | Divergence FWHM/E | Sideband spacing at 1 GHz | Peak height ratio at $a=10^{-4}$ / $3\times10^{-4}$ / $10^{-3}$ |
|---|---|---|---|---|
| Si (220), 16.92 MeV, ψ = 5 mrad, σ = 0.2 mrad | 75.1 keV | 9.4% | 494 eV | 0.94 / 0.62 / 0.27 |
| Diamond (220), 9 MeV, ψ = 5 mrad, σ = 0.3 mrad | 34.0 keV | 14% | 67 eV | 0.97 / 0.79 / 0.34 |
| Quartz $(10\bar11)$, 20 MeV, ψ = 1.9 mrad, σ = 0.5 mrad (Mkrtchyan geometry) | 22.7 keV | 62% | 1.08 keV | 0.99 / 0.92 / 0.47 |

At $a=10^{-4}$ the peak-tilt shift is 2% of $E_1$ for ψ = 5 mrad (5% in the quartz geometry); at $10^{-3}$ it is 20% (53%). A symmetric oscillation leaves the centroid fixed to first order and only broadens and lowers the peak. A peak that *rises* cannot come from a symmetric tilt ensemble. It needs a DC tilt toward a region of higher spectral density, or a different mechanism.

**Validity.** This is a single-reflection kinematic line with the classical root and no multiple scattering beyond the divergence. ψ must exceed ψ_c, or CBS and CR mix (true at 5 mrad for both MeV cases; marginal in the quartz geometry).

**Verdict.** Go as a steering and lock-in knob. A modulated tilt is a clean way to sweep the CBS edge, and the line shift is a direct strain sensor at ψ ≫ ψ_c. No intensity enhancement.

## 4. PXR

**Source equation.** $E=\hbar\,\mathbf v\cdot\mathbf g/(1-\beta\hat n\cdot\hat u)$ is linear in $\mathbf g$. A uniform normal strain gives exactly $\delta E/E=-\varepsilon$. A rotation δ of the planes at a fixed detector gives $\delta E/E=-\cot\theta_B\,\delta$, with θ_B the angle between beam and planes.

**Stage 0** (`photon_energies`, exact recoil root, (220), detector at 90°, central 1 mrad step):

| θ_B | Si 16.92 MeV: $E$, $d\ln E/d\delta$ | Diamond 9 MeV: $E$, $d\ln E/d\delta$ | $-\cot\theta_B$ |
|---|---|---|---|
| 75° | 6.23 keV, −0.268 | 6.70 keV, −0.269 | −0.268 |
| 45° | 4.56 keV, −1.000 | 4.90 keV, −1.002 | −1.000 |
| 15° | 1.67 keV, −3.73 | 1.79 keV, −3.74 | −3.73 |

Tilt $10^{-4}$ therefore tunes PXR by $3\times10^{-5}$–$4\times10^{-4}$. That is below the line width set by the detector aperture and the beam divergence, $\Delta E/E\approx\Delta\theta\cot\theta_B$ per mrad. It is usable for lock-in modulation, not for spectral selection.

**Acoustic satellites.** By §2 the integrated kinematic yield is conserved. The satellite spacing is $\delta E/E=(\mathbf q\cdot\hat u)/(\mathbf g\cdot\hat u)\sim d/\Lambda\approx3\times10^{-5}$ (Si (220), 1 GHz), far inside the line. **No enhancement in thin (kinematic) crystals.**

**Extinction relief (thick crystals).** When the extinction depth is shorter than the absorption length, a lattice rotation exceeding the Darwin width ω_D over one extinction depth makes Bragg reflection kinematic. Integrated reflectivity can then rise toward the mosaic limit (`chi_g`, `absorption_length_ang`; symmetric Bragg, σ polarization, extinction depth $\lambda\sin\theta_B/(2\pi|\chi_g|)$):

| Reflection, photon energy | Extinction depth | Absorption length | Ratio | Darwin width |
|---|---|---|---|---|
| Si (111), 2.5 keV | 0.75 µm | 2.75 µm | 3.7 | 173 µrad |
| Si (111), 8 keV | 0.73 µm | 69 µm | 94 | 35 µrad |
| Si (220), 17.5 keV | 1.06 µm | 0.73 mm | 690 | 11 µrad |
| Diamond (220), 8 keV | 2.0 µm | 0.71 mm | 350 | 15 µrad |
| Diamond (220), 30 keV | 2.0 µm | 52 mm | 2.6e4 | 3.3 µrad |

Below the Si (111) cutoff (≈ 2 keV) there is no Bragg reflection at all. The 50 eV–1.5 keV band has no extinction to relieve, as the #527 comment on the issue expected. Above it, tilt amplitudes of $10^{-5}$–$10^{-4}$ exceed ω_D easily.

However, PXR is not a reflected real photon. Its angular distribution has width ≈ $1/\gamma$ (30 mrad at 17 MeV) plus multiple scattering. Only photons within ω_D of the exact Bragg direction re-diffract (dynamical PXR; the fraction scales as $\gamma\omega_D\approx10^{-4}$–$10^{-3}$). Extinction relief therefore cannot raise the integrated PXR yield at MeV. It does raise **diffracted real-photon** channels, namely diffracted transition radiation and diffracted bremsstrahlung, whose yield is proportional to integrated reflectivity. Neither forward CBS nor forward CR is Bragg-reflected, so extinction relief cannot explain the Mkrtchyan CBS behaviour either.

**Driving transverse transitions** (the #527 comment on the issue, item 2). The PXR photon energy comes from $\mathbf v\cdot\mathbf g$, not from the transverse level spacing. Satellites at $\omega\approx(\mathbf v\cdot\mathbf g-\Delta\varepsilon_\perp/\hbar)/(1-\beta\hat n\cdot\hat u)$ can trade weight, but the total is unchanged. At 30 keV the few-eV spacing lies below the 10–25 eV thickness width of a 30 nm film. This is the channeling form of the shaped-electron interference in [#1102](shaped-electron-cxr-1102.md) and should share that derivation. Verdict: fine-structure control in principle only.

## 5. Planar channeling radiation

**Levers.**

- *Line energy.* Planar CR lines are $E_{if}=\Delta\varepsilon_{if}/(1-\beta\cos\theta)$. Strain changes $\Delta\varepsilon$ through the potential. In the harmonic limit $\omega\propto\sqrt{U''/\gamma m}$ with $U''\propto\sum_m|U_m|g_m^2\propto(Z-f)/V_{\rm cell}$, so an isotropic strain gives $d\ln E/d\varepsilon\approx-3/2$, plus the form-factor slope. A tilt of the plane relative to a fixed detector changes the Doppler factor by $(\gamma\delta\psi)^2$, below $10^{-4}$ for δψ ≤ 0.3 mrad and γ ≤ 35. **No-go** as a tuning knob.
- *Populations.* A plane wave entering at angle δψ projects onto Bloch states with weights $|\langle n|e^{ip\delta\psi x/\hbar}\rangle|^2$. At δψ = 0 a symmetric well populates only even states, which suppresses 1→0 relative to 2→1. Over $0.1$–$1\,\psi_c$ the populations change by order unity, as Andersen's Fig. 11 tilt scans and Ashrafi's Fig. 5 show. This is the strong CR lever.

**Stage 0.** `channeling_spectrum`, TDS + electronic kernels, 1 µm, no absorption, no capture, divergence as stated. Yields are bound→bound line yields summed over Bloch blocks. The ensemble applies the arcsine law with four Gauss–Chebyshev nodes per half range (the response is even in δψ). Strain is an isotropic cell scaling of $10^{-3}$ (research-only catalog patch).

Line yields versus entrance tilt (photons per electron, 4π, 1 µm):

| Case | ψ_c | Aligned yields | 1→0 / 2→1 at δψ = 0, 0.26, 0.46, 0.87 mrad |
|---|---|---|---|
| Si(110), 16.92 MeV, σ = 0.2 mrad (well 21.2 eV) | 1.56 mrad | 2→1 1.24e-5 (10.25 keV); 1→0 1.01e-5 (18.42 keV); 3→2 6.2e-6; 3→0 1.6e-6 | 0.82, 1.53, 2.47, 1.61 |
| Diamond(110), 9 MeV, σ = 0.3 mrad (well 23.6 eV) | 2.23 mrad | 1→0 1.89e-5 (7.75 keV) | 1→0 rises to 1.82× aligned at 0.66 mrad, falls to 0.30× at 2.19 mrad |

Arcsine tilt ensembles (yield ratio to the aligned beam):

| Case | Amplitude | 2→1 | 1→0 | 3→2 | 3→0 | 1→0 / 2→1 change |
|---|---|---|---|---|---|---|
| Si(110) 16.92 MeV | 0.1 ψ_c = 0.16 mrad | 0.93 | 1.07 | 1.02 | 1.02 | ×1.15 |
| | 0.3 ψ_c = 0.47 mrad | 0.68 | 1.37 | 0.92 | 0.92 | ×2.0 |
| | 1.0 ψ_c = 1.56 mrad | 0.40 | 0.80 | 0.47 | 0.48 | ×2.0 |
| Diamond(110) 9 MeV | 0.1 ψ_c = 0.22 mrad | | 1.08 | | | |
| | 0.3 ψ_c = 0.67 mrad | | 1.49 | | | |
| | 1.0 ψ_c = 2.23 mrad | | 0.94 | | | |

Isotropic strain $\varepsilon=10^{-3}$ moves the line centroids by $d\ln E/d\varepsilon$ = −1.39 (Si 1→0), −1.73 (2→1), −1.56 (3→2), −1.52 (3→0) and −1.37 (diamond 1→0). This matches the −3/2 harmonic estimate. At $\varepsilon=10^{-4}$ the shifts are 1.4–1.7 × 10⁻⁴, against measured planar widths of several percent ([#1057](planar-line-anchors-1057.md)).

Reading: an acoustic tilt of amplitude ≳ 0.1 ψ_c, that is ≳ 1.5 × 10⁻⁴ rad at these energies, modulates the line ratio at the 15% level. At 0.3 ψ_c it doubles the ratio, while the total bound→bound yield changes by under 10%. The response is even in δψ, so a symmetric drive modulates the ratio at twice the drive frequency, and only a DC tilt offset gives a first-harmonic signal. Tilt amplitudes near ψ_c lower every line by about 2×, by tilting electrons out of the channel.

**Validity.** The crystal is thin (1 µm), so these are entry-population effects. In thick crystals, dechanneling spreads the populations and reduces the contrast. The planar thick-crystal deficit is recorded in [#1057](planar-line-anchors-1057.md). Each electron is treated at one fixed local tilt, which is the transverse-wave limit of §1. A tilt varying along the path within one dechanneling length would instead need the bent-channel term of §8.

## 6. Mkrtchyan et al. (2013)

*Source.* A. R. Mkrtchyan et al., J. Contemp. Phys. (Armenian Acad. Sci.) 48, 154 (2013), DOI `10.3103/S1068337213040026` (Zotero `4CPT3MXM`; PDF read in full). LEA-50 20 MeV ± 20 keV, divergence 0.5 × 1 mrad, 2 mm spot. X-cut quartz, 1 mm thick, $(10\bar11)$ planar channeling. A 1.024 GHz drive at 0, 5 and 10 W (the text also says 0.5 W). Kr–CO₂ proportional counter at 1 m. CR features at 4.5, 8.9 and 10.2 keV; CBS at 22.5 keV.

**What the paper shows (figures read by eye, ±0.5 keV).**

| Feature | 0 W | 5 W | 10 W |
|---|---|---|---|
| Main CR peak (Fig. 3) | ≈10.2–10.5 keV | ≈11.6 keV (×1.13) | ≈13.0 keV (×1.26) |
| CBS peak (Fig. 4, Bethe–Heitler subtracted) | ≈22.5 keV (Fig. 2; left edge of Fig. 4) | ≈26.5 keV (×1.18), higher | ≈31–33 keV (×1.40), lower |

No strain amplitude is quoted. The drive is called both "electromagnetic field" and "hypersonic vibrations". The resonator holder is not characterized acoustically. A 1 mm X-cut plate has its thickness fundamental near 2.9 MHz, so 1.024 GHz is a ≈ 360th overtone.

**Assessment.**

1. *CR shift.* Strain moves planar lines by ≈ 1.5ε (§5). A 13–26% shift would need ε ≈ 0.1, which is impossible. Tilt does not move planar lines at first order, and a detector-angle change lowers them $(1+\gamma^2\theta^2)^{-1}$ rather than raising them.
2. *CBS shift.* $E_1\propto\psi$, so +18% and +40% would need DC tilt changes of ≈ 0.35 and ≈ 0.8 mrad. A symmetric acoustic oscillation cannot give a DC shift (§3), and 0.8 mrad of acoustic tilt would mean shear strain near $10^{-3}$ throughout a 1 mm plate driven off resonance. Thermal or mechanical tilt of the plate or holder under 5–10 W RF is the plausible source. Quartz expands by $1.3\times10^{-5}$/K.
3. *Common pattern.* The exponent ratio $\ln(E_{\rm CBS}'/E_{\rm CBS})/\ln(E_{\rm CR}'/E_{\rm CR})$ is ≈ 1.3 (5 W) and ≈ 1.4 (10 W). A change of the electron energy scale predicts 4/3, since CBS ∝ γ² and planar CR ∝ γ^{3/2}. A detector-gain drift predicts 1, and a pure tilt predicts ∞ (CR fixed). The data favour an energy-scale change, though figure reading alone cannot exclude a gain drift combined with a tilt. In either case the shifts are not a strain observable.
4. *CBS rise then fall.* A symmetric tilt ensemble only lowers the peak, so the rise needs a DC component. Extinction relief does not apply to forward CBS (§4). The behaviour fits the same DC drift, which moves ψ across the CR/CBS crossover (ψ ≈ 1.9 mrad is close to ψ_c ≈ 1.4 mrad here).

**Limits of the comparison.** The crystal is thick (1 mm, so dechanneling and multiple scattering dominate), no strain amplitude or acoustic mode is measured, the spectra are digitized only by eye, and there is no repeat with the drive off after heating. The paper is a qualitative motivation, not a validation target. **No PyRITE anchor is proposed.**

## 7. Static superlattice: Ashrafi et al. (1991)

*Source.* S. Ashrafi, A. W. Saenz, H. Überall and R. Fusina, J. Opt. Soc. Am. B 8, 2513 (1991), DOI `10.1364/JOSAB.8.002513` (Zotero `8TCD7NKT`; PDF read in full).

**Correction to the issue text.** Ashrafi's worked case is *56 MeV planar (110)* channeling in a Ge$_x$Si$_{1-x}$/Si strained-layer superlattice grown along [100], with x = 0.5 and 0.15. Their Fig. 2 shows only the Si ⟨100⟩ 3.5 MeV axial potential as an illustration. The paper contains no calculation at the axis and energy of #994 cases A and B.

**Their model.**

- Commensurate growth sets $a_\parallel=5.484$ Å (x = 0.5). The Poisson effect sets $a_\perp=5.390$ Å (Si) and 5.590 Å (SiGe).
- The [110] channel kinks at each interface by Δψ = 1.05° (x = 0.5) or 0.31° (x = 0.15).
- Si and SiGe [110] directions sit at +8.64 and −9.57 mrad from the 45° direction.
- Pantell–Swent fits: $U_0$ = −25.15 eV (Si), −34.38 eV (SiGe, x = 0.5).
- 1→0 spacings $\Delta\varepsilon$ = 5.62 eV (Si) and 7.01 eV (SiGe), which puts the lab 1→0 lines near 137 and 172 keV (our estimate, $2\gamma^2\Delta\varepsilon$).
- The spectrum is an incoherent per-layer sum. The kink exceeds ψ_c ≈ 0.9 mrad, so each electron channels in only one layer family, and rocking selects it.
- Thin layers give natural widths $\approx3.5\gamma^2\pi\hbar c/L$ (≈ 1 MeV FWHM at L = 200 Å). Layers of several hundred Å are needed to resolve individual lines. The family selection (Fig. 6, peaks at −2.8 and +2.6 mrad for x = 0.15) survives without resolving them.

**What the engines would need.**

- *Planar (Ashrafi's case):*
  - a depth-piecewise profile, with a lattice, composition and kernel per layer (SiGe through a virtual-crystal potential plus alloy-disorder scattering $\propto x(1-x)(Z_{\rm Ge}-Z_{\rm Si})^2$, alongside TDS);
  - a sudden re-projection at each interface, $\rho\to T\,e^{ip\Delta\psi x/\hbar}\rho\,e^{-ip\Delta\psi x/\hbar}T^\dagger$, with $T$ the overlap of adjacent layer bases;
  - per-layer emission catalogs.

  Because Δψ ≫ ψ_c, a coarse version that sums independent per-layer runs at the layer's local incidence already reproduces the two-family rocking curve.
- *Axial analogue (#994 geometry, ⟨100⟩ along the [001] growth axis):* the [001] strings do *not* kink. The beam crosses layers whose string spacing (5.390 vs 5.590 Å) and composition differ. This is a static longitudinal modulation of the axial potential with period $2L_{\rm layer}$, the static counterpart of Grigoryan's longitudinal wave (§8). It couples $\Delta m=0$ states only. The fibre engine would need:
  - a $z$-dependent Hamiltonian and rate operators (piecewise constant);
  - continuity of the transverse wavefunction at each interface;
  - two-time dipole correlations across layers. Layers of 10–50 nm are comparable to the 2p–1s beat length (46 nm) and dephasing length (≈ 35 nm), so the per-zone secular Lorentzian line model does not apply.

  The full-dynamics $G(z)$ path used for planar `cr-line-shape` is the closest existing machinery.

## 8. Resonant hypersound and polarization selection

*Source.* L. Sh. Grigoryan, A. R. Mkrtchyan, A. H. Mkrtchyan, H. F. Khachatryan, W. Wagner and M. A. Piestrup, Radiat. Eff. Defects Solids 153, 221 (2001), DOI `10.1080/10420150108211841` (Zotero `3436K9M2`; PDF read in full).

**What Grigoryan et al. compute.**

- A *longitudinal* wave along the axis modulates the string potential, Eq. (2.4): $U=-(1+\mu\cos k_sz)\,a/\rho+b+\Delta\cos k_sz$.
- The perturbation is cylindrically symmetric. First-order mixing, Eqs. (2.5)–(2.6a), therefore joins states of the *same* $m$ with longitudinal momentum $\pm\hbar k_s$.
- Photon emission keeps $m_f=m_i\pm1$ (Eq. (2.12)) and gains sound-assisted channels ν = ±1, Doppler-shifted by $\nu\beta\hbar k_s$ (Eq. (2.20)).
- ν = ±2 is forbidden at first order (Eq. (3.7)), and ν = 0 is unchanged at first order.
- Near resonance, $\hbar k_s\beta c\approx|\varepsilon_a-\varepsilon_b|$, the ν = ±1 channels grow and inverse (absorptive) transitions such as 1s→2p appear (Figs. 1–2, 10 MeV, μ = 0.01). The authors note that linear perturbation theory fails at resonance.

**Assessment.**

1. *Amplitude.* μ = 0.01 is a 1% modulation of the string potential, a longitudinal strain $\varepsilon_{zz}\approx10^{-2}$ since the string potential scales as $1/d_{\rm string}$. That is an order of magnitude above sustainable bulk hypersonic strain.
2. *Polarization.* The coupling is $\Delta m=0$ and symmetric in ±m, so the 2p(+1) and 2p(−1) populations stay equal. The forward circular polarization of 2p→1s photons then cancels (`cr-axial-helicity-emission`: on-axis V/I is linear in $p_+-p_-$). **A longitudinal drive cannot produce net circular polarization.**

**Transverse helical drive (new derivation).** Take the axis displaced by a static helix, $\mathbf u(z)=u_0(\cos qz,\sin qz)$, the frozen form of a circularly polarized shear wave travelling along the axis. With $z=vt$ in the transverse Schrödinger picture, $U(\boldsymbol\rho-\mathbf u)\approx U-\mathbf u\cdot\nabla U$, and

$$
-\mathbf u\cdot\nabla U=-\tfrac{u_0}{2}\left[e^{-iqvt}(\partial_x+i\partial_y)U+e^{iqvt}(\partial_x-i\partial_y)U\right].
$$

$(\partial_x+i\partial_y)$ raises $m$ by one. The term $e^{-iqvt}(\partial_x+i\partial_y)U$ is resonant for absorption $1s\to2p(+1)$ when $\hbar qv=\varepsilon_{2p}-\varepsilon_{1s}$. The $m\to m-1$ term is off resonance by $2\hbar qv$. The opposite helicity selects 2p(−1). The drive is therefore **chirality-selective**, unlike the longitudinal wave.

*Equivalence with the bent-channel pseudo-force.* The transformation $\boldsymbol\rho'=\boldsymbol\rho-\mathbf u(z)$ replaces the displacement by the inertial force $\mathbf F=-pv\,\mathbf u''(z)$. Its matrix element $pv\,q^2u_0x_0$ equals the displaced-potential element $\gamma m\omega^2u_0x_0$ exactly when $vq=\omega$, which is the resonance condition.

*Size.* In the harmonic limit, with $x_0=\hbar c/\sqrt{2\gamma mc^2\,\Delta\varepsilon}$ and $\gamma m\omega^2x_0=\hbar\omega/2x_0$, the resonant coupling is $V=\Delta\varepsilon\,u_0/(2x_0)$. Full transfer takes $L_\pi=(\lambda_b/2)\,x_0/u_0$, with beat length $\lambda_b=2\pi\hbar\beta c/\Delta\varepsilon$.

For Si ⟨100⟩ 3.5 MeV 2p→1s (rest-frame 212 eV, lab 3.33 keV, $\Delta\varepsilon$ = 27 eV):

- $\lambda_b$ = 46 nm, $x_0$ = 0.13 Å;
- drive at $f=v_T/\lambda_b$ ≈ 127 GHz (Si $v_T$ = 5.84 km/s);
- $L_\pi$ = 0.3 µm (the case thickness) needs $u_0$ ≈ 0.010 Å, a tilt $qu_0\approx1.4\times10^{-4}$.

The 2p→1s line, however, is 0.5–0.95 keV wide in the lab (Andersen Table 2, #992), which is $\Gamma_\perp$ ≈ 4–8 eV. The coherent drive beats this dephasing only if $V>\Gamma_\perp/2$, that is $u_0\gtrsim x_0\Gamma_\perp/\Delta\varepsilon$ ≈ 0.03 Å, or tilt ≳ $4\times10^{-4}$. Below that the drive acts as a rate $\sim2V^2/(\hbar\Gamma_\perp)$ and gives only a partial ℓ imbalance.

Planar Si(110) 17 MeV 1→0 ($\Delta\varepsilon$ = 7.9 eV, $\lambda_b$ = 156 nm) needs a 37 GHz *linear* shear wave, but planar states carry no ±m pair. It modulates intensity, not circular polarization.

**Verdict.** No-go for now. The mechanism is sound and selective, but it needs a sustained, phase-locked, circularly polarized shear wave at ≈ 130 GHz with strain ≳ $4\times10^{-4}$ inside a 0.3 µm membrane. Picosecond-acoustics pulses reach such strain but are not CW and not circularly polarized. A static helical superlattice is not available either.

## 9. Acoustic crystalline undulators

Kaleris et al. (2025) and Korol et al. (2001; Korol, Solov'yov & Greiner 2014) target GeV positrons with bending amplitude above the plane spacing and many periods within the dechanneling length. MeV electrons dechannel within microns, and the periods are µm. This is out of scope here, as the issue decided.

## 10. Prerequisites and proposed follow-ups

Prerequisites for any implementation:

1. Polarization-resolved CBS/PXR output: Stokes parameters per line. Today only the axial fibre source has helicity output (`cr-axial-helicity-emission`).
2. Depth-dependent crystal description: a piecewise profile per layer, with a basis re-projection at each interface, for planar and axial.
3. Local-tilt and strain fields as inputs: a sampled ensemble over the beam spot for the quasi-static limit. This study used a research-only catalog patch.
4. Bent-channel drift on the continuous carrier (#1056), for tilt varying within a dechanneling length.
5. Next multipole order for twisted-photon decomposition (Korotchenko & Kunashenko 2024, 2026) before any OAM claim.

Proposed follow-up issue titles (not created):

- *Strain: quasi-static tilt/lattice-parameter ensemble input for CBS, PXR and planar CR (research API, no default change)*
- *Polarization-resolved Stokes output for CBS and PXR lines*
- *Planar channeling in strained-layer superlattices: depth-piecewise profile and interface re-projection (Ashrafi 1991 reproduction)*
- *Axial fibre engine: z-dependent Hamiltonian for longitudinal superlattice or hypersound modulation*
- *Extinction relief for diffracted real-photon channels (DTR, diffracted bremsstrahlung) under lattice rotation*
- *Helical shear drive of axial 1s→2p(±1): MCWF feasibility of net circular polarization (research)*

## Sources

- A. R. Mkrtchyan et al., J. Contemp. Phys. 48, 154 (2013). Zotero `4CPT3MXM`.
- S. Ashrafi, A. W. Saenz, H. Überall, R. Fusina, JOSA B 8, 2513 (1991). Zotero `8TCD7NKT`.
- L. Sh. Grigoryan et al., Radiat. Eff. Defects Solids 153, 221 (2001). Zotero `3436K9M2`.
- Kaleris et al., Phys. Rev. Accel. Beams 28, 033502 (2025), Zotero `N2WQCWDM`; Korol, Solov'yov & Greiner (2001), Zotero `2F4GBNML`; *Channeling and Radiation in Periodically Bent Crystals* (2014), Zotero `QWQ29V6W`. These were cited for scope only and not re-read for this note.
- Andersen, Bonderup & Pantell, Ann. Rev. Nucl. Part. Sci. 33, 453 (1983): line energies and widths as used in [#992](axial-material-source-992.md).
