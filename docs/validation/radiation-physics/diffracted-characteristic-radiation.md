# Diffracted characteristic radiation (Kossel lines)

Validation: `diffracted-characteristic-radiation`. Status: **filtered** (verifier `discrepancy` resolved by scoring the standing-wave term; re-verification pending). Issue #1176.

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

- Those of DBR: two-beam Laue (both angles above $3\sqrt{|\chi_0|}$), perfect laterally infinite single slab, $\chi_g$ and $\chi_0$ taken at $E_B$ and flat across the window, and $\chi_g\chi_{-g}\to|\chi_g|^2$. The last means there is no anomalous (Borrmann) absorption. It changes the bracket by a relative $O(\epsilon\,\mu L/\gamma)$, $\epsilon=|\operatorname{Im}\chi_g|/\operatorname{Im}\chi_0$ (about 0.9 for Ge (400)), so contrast is reliable only for $\epsilon\,\mu L/\gamma\ll1$ (0.04 at the Ge check). Thick-crystal Kossel contrast is dominated by Borrmann absorption and is not claimed.
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

None yet. Zotero has no quantitative Kossel-profile dataset: searches for Kossel, Borrmann and fluorescence diffraction, and the Channeling Radiation and PXR-CBS collections, found none. Hall 1966 concerns the electron-side Borrmann effect, not photon Kossel contrast. A measured transmission Kossel profile with stated thickness and geometry would allow `anchored`.

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
