# Diffracted characteristic radiation (Kossel lines)

Validation: `diffracted-characteristic-radiation`. Status: **rederived** (first verifier `discrepancy` resolved by scoring the standing-wave term; fresh-context re-check `rederived`). Issue #1176.

## Need

A characteristic line emitted inside a perfect crystal reaches a detector along $\hat n$ partly by Bragg diffraction. When the Bragg energy of a reflection $g$ toward $\hat n$,

$$
E_B=-\frac{\hbar c\,g^2}{2\,\hat n\cdot g},\qquad \hat n\cdot g<0,
$$

falls within a line, the crystal adds the line emitted along $\hat Q=(k\hat n+g)/k$ (diffracted gain) and removes part of the line emitted along $\hat n$ (direct-beam depletion). The detector directions where this happens form Kossel cones. Native characteristic scoring books only plain attenuation toward $\hat n$.

## Derivation

**Mode.** Reciprocity gives the intensity an incoherent source at depth $z$ radiates into vacuum direction $\hat n$. It is the source's emission into the two-beam Laue mode that a plane wave from $\hat n$ excites, time reversed ([`diffracted-bremsstrahlung`](diffracted-bremsstrahlung.md)). Along the inward depth $u=L-z$ from the exit face,

$$
\frac{dP}{du}=iNP,\quad N=\begin{pmatrix}k\chi_0/2n_s & a\\ b & q+k^2\chi_0/2Q_s\end{pmatrix},\quad P(0)=(1,0),
$$

with the symbols of the DBR record. With eigenvalues $\lambda_{1,2}$ ($\operatorname{Im}(\lambda_1-\lambda_2)\ge0$), $x=(\lambda_1-\lambda_2)u$ and $N_{22}=q+k^2\chi_0/2Q_s$, the mode amplitudes are

$$
D_s(u)=b\,\frac{e^{i\lambda_1u}-e^{i\lambda_2u}}{\lambda_1-\lambda_2},\qquad
T_s(u)=\frac{(\lambda_1-N_{22})e^{i\lambda_1u}-(\lambda_2-N_{22})e^{i\lambda_2u}}{\lambda_1-\lambda_2}
=e^{i\lambda_2u}\Big[(\lambda_2-N_{22})\frac{e^{ix}-1}{\lambda_1-\lambda_2}+e^{ix}\Big].
$$

$T(0)=1$ and $T'(0)=iN_{11}$, as the mode equations require when $D(0)=0$.

**Emission.** By Lorentz reciprocity, the amplitude a dipole $\mathbf p$ at $\mathbf r_j$ radiates toward $\hat n$ is $\mathbf p\cdot\mathbf E(\mathbf r_j)$. Here $\mathbf E$ is the field of a plane wave incident from the detector, travelling along $-\hat n$. That wave diffracts with $h=-g$, so $\mathbf E=T\hat e_0e^{-ik\hat n\cdot\mathbf r}+D\hat e_he^{-i(k\hat n+g)\cdot\mathbf r}$, and $D$ is coupled by $\chi_{-g}$ ($e^{-i\omega t}$ fields, $\operatorname{Im}\chi_0>0$). The basis $\hat e_h$ is the normalized projection of $\hat e_0$ normal to $\hat Q$, so $\hat e_0\cdot\hat e_h=C_s\ge0$.

Averaging $|\mathbf p\cdot\mathbf E|^2$ over the orientation of an isotropic, unpolarized line gives

$$
\frac{d^2N}{dE\,d\Omega_n}=Y\,\mathcal L(E)\,\tfrac12\sum_s\Big[|T_s|^2+|D_s|^2+2C_s\operatorname{Re}\big(T_s^*D_s\,e^{-ig\cdot\mathbf r_j}\big)\Big](L-z,E)
$$

($Y$ in photons/sr, $\mathcal L$ the unit-area natural Lorentzian). Different atoms emit incoherently, but each atom's own $T$–$D$ cross term survives. This is the X-ray standing-wave term. It is first order in $\chi_g$, while the intensity terms are second order.

The code computes $D$ for a real coupling $|\chi_g|$, and only $\chi_g\chi_{-g}$ enters $T$. So $D_{\rm true}=D\cdot\chi_{\rm mode}/|\chi_g|$, with $\chi_{\rm mode}=\chi_{-g}^{\rm field}=\overline{\chi(-hkl)}$ in the crystal module's convention, the value `emission_coupling_tables` returns. Averaging over the emitting element's basis sites $\mathbf R_j$ and its thermal motion gives the emitter phase

$$
\phi_Z=\frac{\chi_{\rm mode}}{|\chi_g|}\,\big\langle e^{-2\pi i\,hkl\cdot\mathbf R_j}\big\rangle_{j\in Z}\,e^{-W},
$$

with $e^{-W}$ the amplitude Debye–Waller factor of the case $B$. An element off the crystal basis has $\phi=0$. For Ge (400), $\phi=-0.927$ ($B=0.6$ Å²). For GaAs (200), the Ga and As phases are opposite.

**Limit $\chi_g\to0$.** $a,b\to0$, so $\lambda=(N_{11},N_{22})$ and $T=e^{ik\chi_0u/2n_s}$. Then $|T_0|^2=e^{-k\operatorname{Im}\chi_0\,u/n_s}=e^{-\mu u/\gamma_n}$, the native escape factor, and $D=0$.

**Flux conservation.** For real $\chi_0$ and $\chi_{-g}=\chi_g^*$, $n_s a=(Q_s/k)\,b^*=kC_s|\chi_g|/2$. Hence

$$
\frac{d}{du}\Big(n_s|T|^2+\frac{Q_s}{k}|D|^2\Big)=2\operatorname{Re}\Big[i\,\tfrac{kC_s}{2}\big(\chi_g^*T^*D+\chi_gD^*T\big)\Big]=0,
$$

so $\gamma_n|T|^2+\gamma_h|D|^2=\gamma_n$ ($\gamma_n=n_s$, $\gamma_h=Q_s/k$).

**Scored term.** Native scoring keeps the plain-attenuation term. The Kossel term added to `spec_characteristic` is

$$
S(E)=Y\,\mathcal L(E)\sum_k w_k\Big[\tfrac12\sum_s\big(|T_s|^2+|D_s|^2\big)-|T_0|^2+\operatorname{Re}\big(\phi_ZX\big)\Big](u_k,E),\qquad X=\tfrac12\sum_s2C_sT_s^*D_s,
$$

where $w_k$ is the share of the line emitted in depth bin $k$. Without absorption, flux conservation reduces the intensity part to $(1-\gamma_h/\gamma_n)|D_s|^2$. Symmetric Laue ($\gamma_h=\gamma_n$) therefore moves no flux. An asymmetric geometry with $\gamma_h<\gamma_n$ gives a bright line, and one with $\gamma_h>\gamma_n$ a dark line. Kinematically, with $\int dE\,|D_s|^2=|b_s|^22\pi u\,\hbar cQ_s/|\hat n\cdot g|$ (DBR record) and a line wider than the Bragg width,

$$
\int S\,dE\simeq Y\,\mathcal L(E_B)\Big(1-\frac{\gamma_h}{\gamma_n}\Big)R_E(\bar u),\qquad R_E=\frac{E\,R_\theta}{\tan\theta_B},\quad R_\theta=\frac{\pi k|\chi_g|^2C^2\bar u}{2\gamma_h\sin2\theta_B},
$$

with $\bar u$ the mean remaining depth. For the cross term, kinematically $D_s=b_s(e^{iqu}-1)/q$ with $T=1$. Since $\int dq\,\sin(qu)/q=\pi$ and $(\cos qu-1)/q$ is odd,

$$
\int dE\,\operatorname{Im}X=\pi\sum_sC_sb_s\frac{\hbar cQ_s}{|\hat n\cdot g|},\qquad \int dE\operatorname{Re}(\phi X)=-\operatorname{Im}\phi\int dE\operatorname{Im}X,
$$

independent of depth. A centrosymmetric emitter (real $\phi$) leaves a dispersive profile with no kinematic band. The sign of $\phi$ decides which side of the line is depleted.

## DBR depletion (decision)

The same $|T|^2$ weight applies to the direct brem toward $\hat n$. Its depletion relative to the DBR gain is about $j(\hat n)/j(\hat Q)$. One-ray estimate at the #1166 mirror rays (17 MeV Si, 2 µm, BremsLib, channeling off; `scratch/dbr_depletion_probe.py`, $\int dE$ of the depth-weighted terms):

| reflection | $E_B$ | $\theta_n$ | $j(\hat n)/j(\hat Q)$ | depletion / gain |
|---|---|---|---|---|
| (0 2 −2) | 6990 eV | 57.5° | 3.2e-6 | −5.8e-6 |
| (1 1 −3) | 10003 eV | 47.5° | 6.9e-6 | −1.0e-5 |
| (−1 3 1) | 8588 eV | 51.5° | 3.9e-6 | −6.1e-6 |

DBR therefore keeps no depletion term, and its output is unchanged. The omission is listed in the DBR assumptions.

## Implementation

`spectrum/diffracted_brem.py::laue_mode_intensities` returns $\tfrac12\sum_s|T_s|^2$, $\tfrac12\sum_s|D_s|^2$ and (with `cross=True`) $X$ from one eigen-solution. `bragg_reflections(detail=True)` also returns each reflection's $hkl$ and $\chi_{\rm mode}$. `spectrum/diffracted_characteristic.py` implements the rest:

1. `line_depth_yields`: per line, the unattenuated yield on 64 depth bins, using native rows, cutoff, shell cross sections, cascade yields and channeling flux weights, with no escape factor.
2. `emitter_phases`: $\phi_Z$ per element from the crystal basis, $\chi_{\rm mode}$ and the case $B$.
3. `kossel_term`: the bracket above (with $\phi_Z$) for every depth bin on the reflection's Bragg window (`bragg_window`, step at most FWHM/8 of the nearest line).
4. `diffracted_characteristic_spectrum`: per reflection, the lines within 1.5 keV plus 100 FWHM of $E_B$, then $\mathcal L(E)\,(w\cdot\text{bracket}_Z)$ with the line's element $Z$, deposited with `line_quadrature`.

`runner/diffracted_brem.py::case_diffracted_characteristic` adds the result to `spec_characteristic` in `_spectrum_case_impl` and in streamed blocks, for the same perfect single-slab cases as DBR.

## Assumptions

- Those of DBR: two-beam Laue (both angles above $3\sqrt{|\chi_0|}$), perfect laterally infinite single slab, $\chi_g$ and $\chi_0$ taken at $E_B$ and flat across the window, and $\chi_g\chi_{-g}\to|\chi_g|^2$. The last means there is no anomalous (Borrmann) absorption. It changes the bracket by a relative $O(\epsilon\,\mu L/\gamma)$, with $\epsilon=|\chi''_g|/\chi''_0$ and $\chi''_g$ the Fourier coefficient of $\operatorname{Im}\chi(\mathbf r)$. That is about 0.9 for Ge (400) and 0.86 for GaAs (111); it is origin-independent, unlike $|\operatorname{Im}\chi_g|$ for noncentrosymmetric crystals. Contrast is therefore reliable only for $\epsilon\,\mu L/\gamma\ll1$ (0.037 at the Ge check). No thickness guard enforces this, as for DBR. Thick-crystal Kossel contrast is dominated by Borrmann absorption and is not claimed.
- Unpolarized, isotropic lines with natural Lorentzian shapes, emitted by atoms on the crystal's own basis sites (isotropic Debye–Waller coherent fraction from the case $B$). Emitters off the basis carry $\phi=0$. Site-resolved ionization (electron Bloch-wave effects) is not weighted: every site of an element emits equally.
- The two-beam absorption is photoabsorption ($\chi_0$, Henke), while native escape uses total $\mu$ (EPDL). The subtraction uses the same $\chi_0$, so the term vanishes off Bragg.
- Lines farther than 100 FWHM outside a Bragg window are skipped.
- No Bragg-case (reflection) Kossel lines, mosaic crystals, layer stacks or grooves.
- Electron-side orientation effects on ionization (Hall 1966, `MPCKVBM9`; Josefsson & Allen 1996, `IW7HEZGP`) are outside this claim.

## Checks

| check | result |
|---|---|
| $\lvert T\rvert ^2$, $\lvert D\rvert ^2$ vs `expm` of $N$ (strong coupling, absorbing) | 1e-8 (measured 7e-16) |
| $\chi_g\to0$: $\lvert T\rvert ^2=e^{-k\operatorname{Im}\chi_0u/n_s}$, $D=0$, Kossel term 0 | 1e-12 |
| $n_s\lvert T\rvert ^2+(Q_s/k)\lvert D\rvert ^2=n_s$, real $\chi_0$, strong coupling | 1e-12 |
| symmetric Laue, no absorption: Kossel term 0 | 1e-12 |
| kinematic band vs $Y\mathcal L(E_B)(1-\gamma_h/\gamma_n)R_E$ (textbook $R_\theta$) | 1e-2 |
| standing-wave $\operatorname{Re}(\phi X)$ vs `expm` with complex $\chi_g$ and emitter phase | 1e-8 |
| kinematic $\int dE\operatorname{Im}X=\pi\sum_sC_sb_s\hbar cQ_s/\lvert n\cdot g\rvert$; $\lvert\int\operatorname{Re}X\rvert$ below 2% of it | 3e-3 |
| emitter phases: Ge (400) $=\chi$ phase; GaAs (200) Ga $=-$As, real, $0<e^{-W}<1$ | exact |
| depth yields × native escape vs `mc_characteristic_spectrum` (Ge, 100 keV, 2 µm) | 2e-3 |
| Ge Kα1 on the (−4 0 0) Kossel cone ($\gamma_n-\gamma_h=0.63$, $\phi=-0.927$) | intensity terms alone: +74% of native at the line. With the standing-wave term the profile is dispersive: −44% at the line and up to 2 eV below, +76% 2 eV above, tails ±12–14% at 10 eV. The ±15 eV band is +11% of native (+21% without the cross term). |
| runner adds exactly the Kossel term to `spec_characteristic`; out-of-claim cases add nothing | 1e-9 |

## Anchor

Analytic anchors only (the tests). No measured thin-slab profile with stated thickness has been located; Hall 1966 (`MPCKVBM9`) concerns the electron-side Borrmann effect. Two measured candidates are in Zotero (PXR-CBS collection):

- Faigel, Bortel & Tegze, *Sci. Rep.* **6**, 22904 (2016), doi:10.1038/srep22904 (`JBG7MXA6`). It reports 24 Laue-case Ga/As Kα Kossel profiles of GaAs (synchrotron excitation, 14.4 keV), fitted with Hannon–Trammell dynamical theory; the line shape encodes the structure-factor phase relative to the emitter site. The crystals are thick (Borrmann-dominated), so this checks only the sign and shape of $\operatorname{Re}(\phi_ZX)$ per element, not the magnitude.
- Bortel et al., arXiv:2310.18203 (2023) (`TX6HBKQC`), with GaAs/GaP Kossel patterns from a single XFEL pulse and finite-slab Laue/Bragg formulas.

## Fresh-context verification (2026-11-14)

An independent verifier (`physics-validator`) re-derived the claim before reading the implementation body, then compared it with the code at commit `7929447d1`.

### Independent derivation

The reciprocity mode solves $dP/du=iNP$ with $P(0)=(1,0)$, so $P(u)=e^{iNu}(1,0)^{\mathsf T}$. Sylvester's formula for a $2\times2$ matrix with eigenvalues $\lambda_{1,2}$ gives

$$
e^{iNu}=\frac{(N-\lambda_2)e^{i\lambda_1u}-(N-\lambda_1)e^{i\lambda_2u}}{\lambda_1-\lambda_2}.
$$

With $\lambda_1+\lambda_2=N_{11}+N_{22}$, the $(1,1)$ element is the stated $T_s$ and the $(2,1)$ element is $D_s=b\,(e^{i\lambda_1u}-e^{i\lambda_2u})/(\lambda_1-\lambda_2)$. With $x=(\lambda_1-\lambda_2)u$ and $\operatorname{Im}(\lambda_1-\lambda_2)\ge0$, the stable form is

$$
T_s=e^{i\lambda_2u}\Big[1+(\lambda_1-N_{22})\frac{e^{ix}-1}{\lambda_1-\lambda_2}\Big],
$$

which equals the code's $e^{i\lambda_2u}[(\lambda_2-N_{22})(e^{ix}-1)/(\lambda_1-\lambda_2)+e^{ix}]$ identically. For $\chi_g\to0$, $T=e^{iN_{11}u}$ and $|T_0|^2=e^{-k\operatorname{Im}\chi_0u/n_s}$. Without absorption ($N_{11},N_{22}$ real, $\chi_{-g}=\chi_g^*$), $n_sa=(Q_s/k)b^*$, so $\frac{d}{du}\big(n_s|T|^2+\tfrac{Q_s}{k}|D|^2\big)=0$. The bracket is then $(1-\gamma_h/\gamma_n)|D|^2$: bright for $\gamma_h<\gamma_n$, dark for $\gamma_h>\gamma_n$. The kinematic band follows from $|n\cdot g|=2k\sin^2\theta_B$ and $Q_s=k\gamma_h$:

$$
\int dE\,|D_s|^2=\frac{\pi k\,u\,E\,|\chi_g|^2C_s^2}{4\gamma_h\sin^2\theta_B}=\frac{E\,R_\theta}{\tan\theta_B},\qquad R_\theta=\frac{\pi k|\chi_g|^2C_s^2u}{2\gamma_h\sin2\theta_B}.
$$

**Emission from a lattice site.** Reciprocity gives the amplitude toward $\hat n$ from a dipole $\mathbf p$ at $\mathbf r_j$ as $\mathbf p\cdot\mathbf E(\mathbf r_j)$, with $\mathbf E=T\hat e_0e^{i\mathbf K_0\cdot\mathbf r}+D\hat e_he^{i(\mathbf K_0+\mathbf g)\cdot\mathbf r}$. Averaging over the dipole orientation (isotropic line) and over polarization gives

$$
\tfrac12\sum_s\Big[|T_s|^2+|D_s|^2+2C_s\operatorname{Re}\big(T_s^*D_s\,e^{i\mathbf g\cdot\mathbf r_j}\big)\Big].
$$

Incoherence between different atoms removes cross terms between emitters. It does not remove the cross term of a single emitter. Characteristic lines are emitted by atoms on lattice sites, so $e^{i\mathbf g\cdot\mathbf r_j}$ is fixed by the lattice, not random. Averaged over the emitting sites and thermal motion, $b\,e^{i\mathbf g\cdot\mathbf r_j}$ becomes $b\,F_c e^{i\psi}$, with $F_c$ the coherent fraction (Debye–Waller) and $\psi$ fixed by the sublattice. For monatomic Ge, $\chi_ge^{i\mathbf g\cdot\mathbf r_j}\to-|\chi_g|F_c$. This is the X-ray standing-wave term ("Kossel lines as XSW in reverse"; von Laue's reciprocity treatment of Kossel lines). The ledger's "no $T$–$D$ cross term" holds only for emitters with $F_c=0$, i.e. not on the lattice.

### Comparison with the code

| item | result |
|---|---|
| 1. $T_s$, stable form, $\tfrac12\sum_s(\lvert T\rvert ^2+\lvert D\rvert ^2)$, no $\gamma$ flux factor | $T_s$ and $D_s$ match, including the stable form and the small-$x$ series. The code's $\lvert T\rvert ^2+\lvert D\rvert ^2-\lvert T_0\rvert ^2$ equals an independent `expm` with signed $\chi_g$ to 8e-15. No $\gamma$ factor is correct per steradian of $\hat n$. **Fail**: the omitted cross term, sized in the next table |
| 2. native books $\lvert T_0\rvert ^2$; $\mu_{\rm tot}$ vs $\operatorname{Im}\chi_0$ | pass. Off Bragg the added bracket tends to $\lvert T_0\rvert ^2-\lvert T_0\rvert ^2=0$, so nothing is double counted or omitted. Near Bragg the mismatch is $(\mu_{\rm tot}-\mu_{\rm ph})u/\gamma_n$ times the Kossel term; at the Ge anchor $\mu_{\rm ph}L/\gamma_n=0.040$, so it is below 1% |
| 3. flux conservation, kinematic band, sign | pass. The analytic derivative vanishes; the band equals $Y\mathcal L(E_B)(1-\gamma_h/\gamma_n)R_E$ with $R_\theta$ normalized by $\gamma_h$, matching the DBR record. Bright for $\gamma_h<\gamma_n$ |
| 4. `line_depth_yields` vs native estimator | pass. It uses the same clipping, `elec_id < Ne` rows, `E_repr_keV`, shell cross sections, `_cascade_line_yields` at the default relaxation cutoff, `_shell_flux_weight` and $1/(4\pi N_e)$. Native escape is dropped. The test passes at 2e-3. Minor: `shell_ionization_model=None` maps to `"eedl"` here but passes `None` natively |
| 5. numerics, guards, wiring | pass. The step is the smallest of a quarter of the $2\pi/L$ width in $q$, span/1000 and FWHM/8, floored at span/20000, giving $0.35$ eV for Ge K$\alpha_1$. Lines beyond 100 FWHM contribute a relative $\lesssim10^{-4}$. The bin-mean deposit conserves the band, and there are 64 depth bins. `_in_claim` matches DBR. The term is added once in `_spectrum_case_impl` (separate `phase_cache` key, so it survives OOM retries) and once per streamed block, with the same `Ne_brem` and row filter as native |
| 6. DBR depletion omitted | pass. `scratch/dbr_depletion_probe.py` reproduces −5.8e-6, −1.0e-5 and −6.1e-6. The probe integrates $\int dE\,w_n\cdot(\lvert T\rvert ^2-\lvert T_0\rvert ^2)$ against $\int dE\,w_Q\cdot\lvert D\rvert ^2$, which is the right ratio |
| 7. $\lvert \chi_g\rvert ^2$ for $\chi_g\chi_{-g}$; $\mu L/\gamma\lesssim1$ | concern. Anomalous absorption changes the bracket by a relative $O(\epsilon\,\mu L/\gamma)$, with $\epsilon=\lvert \operatorname{Im}\chi_g\rvert /\operatorname{Im}\chi_0\approx0.9$ for Ge (400). At $\mu L/\gamma\approx1$ that error is order one. The validity statement should read $\epsilon\,\mu L/\gamma\ll1$, not $\lesssim1$ |

**Omitted standing-wave term at the claim's own Ge check** (probe `scratch/verify_kossel_crossterm_probe.py`). The setup is 2 µm Ge, K$\alpha_1$ 9886.0 eV (FWHM 2.78 eV) on the (−4 0 0) cone, $|\chi_g|=1.07\times10^{-5}$, $\gamma_n=0.947$ and $\gamma_h=0.320$. The cross term $X$ is evaluated with $F_c=1$; for Ge (400) at $B=0.6$ Å², $F_c\approx0.93$.

| quantity | kept term $K=$ code bracket | omitted $X$ |
|---|---|---|
| depth-mean peak (line-relative) | +0.340 | ±0.578 (dispersive) |
| $\int\mathcal L K\,dE$ vs $\int\mathcal L X\,dE$, line on $E_B$ | 0.0531 | −0.0119 (−22%) |
| line at $E_B-1$ eV | 0.0350 | −0.0568 |
| line at $E_B+1$ eV | 0.0418 | +0.0500 |
| line at $E_B+3$ eV | 0.0126 | +0.0494 |
| emitter phase $\psi=90°$ (sublattice-selective emitter) | 0.0531 | +0.0926 |
| max of bin mean, 0.5 / 1 / 2.5 / 5 eV bins | 0.061 / 0.045 / 0.021 / 0.011 | 0.086 / 0.034 / 0.0057 / 0.0024 |

The omitted term is first order in $\chi_g$, while the kept term is second order. For $|b|L\lesssim1$ it dominates the line profile, as in the 2 µm Ge check. It sets the bright/dark asymmetry around the cone, which reverses sign within ±1 eV of detuning. In integrated band it is 22% of the kept term with the line on $E_B$, and 1.4–4× when the cone is detuned by 1–3 eV. At the 5 eV bins of the anchor test it is still 23% of the peak.

### Verdict

`discrepancy`. The first divergent term is the omitted interference term $\tfrac12\sum_s2C_sF_c\operatorname{Re}\big(T_s^*D_se^{i\psi}\big)$ in the per-steradian intensity. The ledger's claim that isotropic incoherent emission has no $T$–$D$ cross term is false for emitters on lattice sites. Everything else re-derives: the mode, $T_s$, flux conservation, the kinematic band, no-double-count, depth yields, numerics and wiring.

Two remedies are possible. The first scores the cross term, with $\chi_g$'s phase and the emitting element's sublattice structure factor and Debye–Waller coherent fraction. That needs the signed $\chi_g$, not $|\chi_g|$. The second restricts the claim to $F_c\to0$ emitters and reports the omission as a bound, which would exclude the crystal's own characteristic lines, i.e. the intended use. Also tighten the Borrmann validity statement to $\epsilon\,\mu L/\gamma\ll1$.

### Resolution (task owner, 2026-11-14)

The first remedy is implemented. `laue_mode_intensities(cross=True)` returns $X$, `bragg_reflections(detail=True)` supplies $\chi_{\rm mode}=\overline{\chi(-hkl)}$, and `emitter_phases` forms $\phi_Z$. The reciprocity sign ($h=-g$ for the wave incident along $-\hat n$) and the $e^{-i\omega t}$ conjugation were re-derived above. For Ge (400), $\phi=-0.927$ matches the verifier's $-|\chi_g|F_c/|\chi_g|$ with $F_c\approx0.93$. The Borrmann validity statement is tightened to $\epsilon\,\mu L/\gamma\ll1$. Status stays `filtered` until a fresh-context re-check.

## Fresh-context re-verification (2026-11-14, after `383e55b89`)

A second independent verifier re-derived the standing-wave term before reading `emitter_phases`, `_kossel_modes` or the `cross=True` branch, then compared at commit `383e55b89`. Probes: `scratch/verify2_kossel_xsw_probe.py` (independent field construction), `scratch/verify2_misc.py` (kinematic band, phases, element keys).

### Independent derivation

**Convention.** Fields go as $e^{-i\omega t}$, with $\chi(\mathbf r)=\sum_{\mathbf h}\chi_{\mathbf h}e^{i\mathbf h\cdot\mathbf r}$ and $\operatorname{Im}\chi_0>0$. Tabulated factors $F=f_0+f'+if''$ with $f''>0$ belong to the $e^{+i\omega t}$ convention, so

$$
\chi_{\mathbf h}=-\frac{r_e\lambda^2}{\pi V_{\rm cell}}\sum_l\overline{F_l}\,e^{-W_l}\,e^{-i\mathbf h\cdot\mathbf r_l}.
$$

The crystal module's `chi_g` is $\chi^{\rm cr}(hkl)=-\frac{r_e\lambda^2}{\pi V}\sum_lF_le^{-W}e^{+2\pi i\,hkl\cdot\mathbf R_l}$, so $\chi_{\mathbf g}=\overline{\chi^{\rm cr}(hkl)}$ and $\chi_{-\mathbf g}=\overline{\chi^{\rm cr}(-hkl)}$.

**Reciprocal wave.** By Lorentz reciprocity (symmetric scalar $\varepsilon$, no conjugation of the medium), the amplitude toward $\hat{\mathbf n}$ with polarization $\hat{\mathbf e}$ from a dipole $\mathbf p$ at $\mathbf r_j$ is $\mathbf p\cdot\mathbf E(\mathbf r_j)$. Here $\mathbf E$ is the field of a plane wave incident along $-\hat{\mathbf n}$, $\mathbf K_0=-k\hat{\mathbf n}$. A Bragg partner $\mathbf K_0+\mathbf h$ with $\lvert\mathbf K_0+\mathbf h\rvert=k$ needs $h^2=2k\,\hat{\mathbf n}\cdot\mathbf h$. The reflection condition $g^2=-2k\,\hat{\mathbf n}\cdot\mathbf g$ therefore selects $\mathbf h=-\mathbf g$, and $\mathbf K_h=-k\hat{\mathbf Q}$ is the time-reverse of the emission direction. Insert $\mathbf E=T\hat{\mathbf e}_0e^{i\mathbf K_0\cdot\mathbf r}+D\hat{\mathbf e}_he^{i(\mathbf K_0-\mathbf g)\cdot\mathbf r}$ into $\nabla^2\mathbf E+k^2(1+\chi)\mathbf E=0$, with the inward depth $u$, $\mathbf K_0\cdot\hat{\mathbf u}=k\gamma_n$ and $\mathbf K_h\cdot\hat{\mathbf u}=Q_s$:

$$
T'=i\Big[\frac{k\chi_0}{2\gamma_n}T+\underbrace{\frac{kC\chi_{\mathbf g}}{2\gamma_n}}_{a}D\Big],\qquad
D'=i\Big[\underbrace{\frac{k^2C\chi_{-\mathbf g}}{2Q_s}}_{b}T+\Big(\frac{k^2-\lvert k\hat{\mathbf n}+\mathbf g\rvert^2}{2Q_s}+\frac{k^2\chi_0}{2Q_s}\Big)D\Big].
$$

$D$ is coupled by $\chi_{-\mathbf g}$ (field convention), and $T$ by $\chi_{\mathbf g}$. Here $C=\hat{\mathbf e}_0\cdot\hat{\mathbf e}_h$, and $\hat{\mathbf e}_h$ is any unit transverse vector to $\hat{\mathbf Q}$ in the $\hat{\mathbf e}_0$, $\hat{\mathbf Q}$ plane. The products $Cb\propto C^2$ and $ab\propto C^2$ are independent of the sign choice of $\hat{\mathbf e}_h$.

**Site intensity.** The carrier $e^{i\mathbf K_0\cdot\mathbf r_j}$ is common to both waves, so for an isotropic, unpolarized emitter

$$
\tfrac12\sum_s\big\lvert T_s\hat{\mathbf e}_0+D_s\hat{\mathbf e}_he^{-i\mathbf g\cdot\mathbf r_j}\big\rvert^2
=\tfrac12\sum_s\Big[\lvert T_s\rvert^2+\lvert D_s\rvert^2+2C_s\operatorname{Re}\big(\overline{T_s}D_se^{-i\mathbf g\cdot\mathbf r_j}\big)\Big].
$$

The exponent is $e^{-i\mathbf g\cdot\mathbf r_j}$. A forward first-Born check confirms it: the spherical wave from $\mathbf r_j$, scattered by $\chi_{\mathbf h}e^{i\mathbf h\cdot\mathbf r}$ into $\hat{\mathbf n}$, is stationary for $\mathbf h=-\mathbf g$ along $\hat{\mathbf Q}$, with relative factor $\chi_{-\mathbf g}e^{-i\mathbf g\cdot\mathbf r_j}$. The product $\chi_{-\mathbf g}e^{-i\mathbf g\cdot\mathbf r_j}=-\frac{r_e\lambda^2}{\pi V}\sum_l\overline{F_l}e^{-W}e^{i\mathbf g\cdot(\mathbf r_l-\mathbf r_j)}$ does not depend on the origin. A thermal displacement of the emitter multiplies the cross term by $e^{-W}$, while the coherent field already carries $e^{-W}$ inside $\chi_{-\mathbf g}$.

**Real-coupling solution.** $T$ and the eigenvalues depend on $a,b$ only through $ab\propto\chi_{\mathbf g}\chi_{-\mathbf g}$, and $D=b\,(e^{i\lambda_1u}-e^{i\lambda_2u})/(\lambda_1-\lambda_2)$ is linear in $b$. With $ab\to k^3C^2\lvert\chi_{-\mathbf g}\rvert^2/(4\gamma_nQ_s)$ and $b_{\rm code}=k^2C\lvert\chi_{-\mathbf g}\rvert/(2Q_s)$, the true diffracted amplitude is $D=D_{\rm code}\,\chi_{-\mathbf g}/\lvert\chi_{-\mathbf g}\rvert$. Hence

$$
\phi_Z=\frac{\chi_{-\mathbf g}}{\lvert\chi_{-\mathbf g}\rvert}\big\langle e^{-2\pi i\,hkl\cdot\mathbf R_j}\big\rangle_{j\in Z}e^{-W},\qquad \chi_{-\mathbf g}=\overline{\chi^{\rm cr}(-hkl)},
$$

which is the claim's expression. The modulus in the denominator must equal the `chi_g_abs` the mode uses. In the code both are $\lvert\chi_{\rm mode}\rvert$ (`bragg_reflections` returns `abs(chi_mode)`, and `emitter_phases` divides by `abs(chi_mode)`). For a Bijvoet pair this differs from $\lvert\chi^{\rm cr}(hkl)\rvert$, so using $\lvert\chi_{\rm mode}\rvert$ consistently is required, and it is done.

**Kinematic band.** With $T=1$ and $D=b(e^{iqu}-1)/q$, and $dq/dE=\lvert\hat{\mathbf n}\cdot\mathbf g\rvert/(\hbar cQ_s)$ at $E_B$,

$$
\int dE\,\operatorname{Im}X=\pi\sum_sC_sb_s\frac{\hbar cQ_s}{\lvert\hat{\mathbf n}\cdot\mathbf g\rvert},\qquad \int dE\,\operatorname{Re}(\phi X)=-\operatorname{Im}\phi\int dE\,\operatorname{Im}X.
$$

$\operatorname{Re}X=Cb(\cos qu-1)/q$ is odd in $q$. Since $dq/dE>0$ ($\hat{\mathbf n}\cdot\mathbf g<0$), a real negative $\phi$ (Ge) depletes the line below $E_B$ and enhances it above.

### Comparison with the code

| item | result |
|---|---|
| 1. reciprocal $\mathbf h$, exponent sign, coupling component | $\mathbf h=-\mathbf g$, $e^{-i\mathbf g\cdot\mathbf r_j}$ and $D\propto\chi_{-\mathbf g}=\overline{\chi^{\rm cr}(-hkl)}$, all as in the code. `chi_mode` from `bragg_reflections(detail=True)` equals my $\chi_{-\mathbf g}$, built from `atomic_form_factor` with my own Fourier sum, to $\le4\times10^{-15}$ relative for GaAs (111), $(\bar1\bar1\bar1)$, (200), $(\bar200)$, (311) and (400) at the As K$\alpha_1$ energy 10543.7 eV, where Ga is above its K edge ($f''_{\rm Ga}=3.75$). **Pass** |
| 1. full field, noncentrosymmetric sites | Independent construction: `scipy.linalg.expm` of my $N$ with complex $a,b$ ($ab$ set to the code's $\lvert\chi\rvert^2$ value), with $\lvert p\cdot E\rvert^2$ at every Ga or As basis site, thermally averaged along $\mathbf g$ by 40-node Gauss–Hermite ($B=0.6$ Å²), $\chi$ held at $E_B$. Against the code's $\tfrac12\sum_s(\lvert T\rvert^2+\lvert D\rvert^2)+\operatorname{Re}(\phi_ZX)$, over 27 energies across the window and $u=0.3,1,2$ µm, the maximum absolute difference is $2\times10^{-12}$ to $6\times10^{-12}$ for every reflection and both elements, on brackets of order 0.1–1.4. A deliberate conjugation error ($\chi^{\rm cr}(hkl)$ in place of $\overline{\chi^{\rm cr}(-hkl)}$) on GaAs (111) changes the cross term by $7.4\times10^{-2}$ (As) and $7.3\times10^{-2}$ (Ga), against a cross-term peak of 0.42–0.45, so the test resolves the convention. **Pass** |
| 2. $D_{\rm true}=D_{\rm code}\chi_{\rm mode}/\lvert\chi_g\rvert$, $C_s$ | $T$ and the eigenvalues depend only on $ab$, and $D$ is linear in $b$, so the rescaling is exact under $\chi_{\mathbf g}\chi_{-\mathbf g}\to\lvert\chi_{\rm mode}\rvert^2$. The code's $C=\lvert\hat{\mathbf e}-\hat{\mathbf Q}(\hat{\mathbf Q}\cdot\hat{\mathbf e})\rvert\ge0$ enters both $b$ (through $\sqrt{b^2}$) and $X$. My probe uses the signed basis $\hat{\mathbf e}_h=\sigma\times\hat{\mathbf Q}$, $C_\pi=\hat{\mathbf n}\cdot\hat{\mathbf Q}$, and agrees to $10^{-12}$ (the product $C_sb_s$ is basis-invariant). **Pass** |
| 3. kinematic band, dispersive profile | `laue_mode_intensities` with $\lvert\chi_g\rvert=10^{-8}$, $\chi_0=0$, $u=0.5$ µm, asymmetric geometry: $\int dE\,\operatorname{Im}X$ / prediction = 1.0028 over ±400 eV. Taking $\phi=0.5i$ gives band/prediction $-0.501$, i.e. $-\operatorname{Im}\phi$. $\int\operatorname{Re}X$/prediction is $-1\times10^{-4}$, $-0.007$, $-0.014$ and $-0.061$ over ±10, 40, 100 and 400 eV. The $1/q$ tail of $\operatorname{Re}X$ is only conditionally convergent: $q(E)$ and $Q_s(E)$ are not exactly linear, so a wide window leaves a small net band for real $\phi$. In scoring, the Lorentzian weight confines this, so it is not a finding. Ge $(\bar400)$, $B=0.6$: $\phi=-0.9273+0.0301i$. $\operatorname{Re}\phi$ matches $-e^{-W}\approx-0.93$. The small $\operatorname{Im}\phi>0$ is the $f''$ of $\overline F$ and gives a small negative kinematic band. **Pass** |
| 3. Debye–Waller | Amplitude $e^{-W}$, $W=B(g/4\pi)^2$, from `debye_waller` with the case $B$ (Ge 0.6, GaAs 0.62 Å²). $\chi_{\rm mode}$ also carries $e^{-W}$ (`B_ang2` is passed to `emission_coupling_tables`), so the cross term goes as $e^{-2W}$. My independent thermal average reproduces this at $10^{-12}$. **Pass** |
| 4. no double count, depth yields, wiring | Off Bragg, $D\to0$ makes both $X$ and $\lvert T\rvert^2-\lvert T_0\rvert^2\to0$. The depth-yield and runner tests pass. Element keys agree between the case composition and the crystal basis (`ge`: Ge/Ge; `gaas`: As, Ga/As, Ga), so the phase lookup is not silently dropped. **Pass** |
| 4. Borrmann validity | With full $\chi_{\mathbf g}\chi_{-\mathbf g}$, my bracket differs from the code by up to $2$–$8\times10^{-2}$ for GaAs at 2 µm, 10.5 keV ($\mu=0.064$ µm⁻¹). That matches $O(\epsilon\mu L/\gamma)\approx0.16$, so the stated assumption describes the error. At the Ge check, $\epsilon\mu L/\gamma_n\approx0.93\times0.040=0.037$. **Concern (non-blocking)**: (a) for noncentrosymmetric crystals the record's $\epsilon=\lvert\operatorname{Im}\chi_g\rvert/\operatorname{Im}\chi_0$ depends on the origin (GaAs (111) gives 6.8 by that formula). It should read $\epsilon=\lvert\chi''_{\mathbf g}\rvert/\chi''_0$, with $\chi''_{\mathbf g}$ the Fourier coefficient of $\operatorname{Im}\chi(\mathbf r)$, i.e. $\lvert\sum_lf''_le^{-W}e^{-i\mathbf g\cdot\mathbf r_l}\rvert/\sum_lf''_l\approx0.86$ for GaAs (111). (b) `_in_claim` does not gate on $\epsilon\mu L/\gamma$, so thick crystals are scored outside the stated validity, as for DBR |
| 5. tests | `tests/montecarlo/spectrum/test_diffracted_characteristic.py` with `test_diffracted_brem.py`: 21 passed |

### Verdict

`rederived`. The standing-wave term, its reciprocal-vector sign, the exponent sign, the $\chi_{-\mathbf g}$ coupling and its translation to $\overline{\chi^{\rm cr}(-hkl)}$, the $\lvert\chi_{\rm mode}\rvert$ normalization, the $C_s$ convention and the Debye–Waller usage all match an independent construction to $10^{-12}$, including noncentrosymmetric GaAs reflections on both sublattices. Two non-blocking notes remain: define $\epsilon$ from the absorptive Fourier coefficient, and consider a runtime warning or gate for $\epsilon\mu L/\gamma$ not small. Status changes are for the task owner. Human sign-off is tracked in #277.
