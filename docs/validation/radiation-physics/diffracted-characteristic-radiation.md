# Diffracted characteristic radiation (Kossel lines)

Validation: `diffracted-characteristic-radiation`. Status: **filtered**. Issue #1176.

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

**Emission.** An incoherent source couples to each plane-wave component of the mode with that component's amplitude, and its emissions toward $\hat n$ and $\hat Q$ do not interfere. A line of unattenuated density $Y\,\mathcal L(E)$ ($Y$ in photons/sr, $\mathcal L$ the unit-area natural Lorentzian) emitted isotropically, so that $j(\hat Q)=j(\hat n)$, therefore reaches $\hat n$ as

$$
\frac{d^2N}{dE\,d\Omega_n}=Y\,\mathcal L(E)\,\tfrac12\sum_s\big[|T_s(L-z,E)|^2+|D_s(L-z,E)|^2\big]
$$

for an unpolarized line.

**Limit $\chi_g\to0$.** $a,b\to0$, so $\lambda=(N_{11},N_{22})$ and $T=e^{ik\chi_0u/2n_s}$. Then $|T_0|^2=e^{-k\operatorname{Im}\chi_0\,u/n_s}=e^{-\mu u/\gamma_n}$, the native escape factor, and $D=0$.

**Flux conservation.** For real $\chi_0$ and $\chi_{-g}=\chi_g^*$, $n_s a=(Q_s/k)\,b^*=kC_s|\chi_g|/2$. Hence

$$
\frac{d}{du}\Big(n_s|T|^2+\frac{Q_s}{k}|D|^2\Big)=2\operatorname{Re}\Big[i\,\tfrac{kC_s}{2}\big(\chi_g^*T^*D+\chi_gD^*T\big)\Big]=0,
$$

so $\gamma_n|T|^2+\gamma_h|D|^2=\gamma_n$ ($\gamma_n=n_s$, $\gamma_h=Q_s/k$).

**Scored term.** Native scoring keeps the plain-attenuation term. The Kossel term added to `spec_characteristic` is

$$
S(E)=Y\,\mathcal L(E)\sum_k w_k\,\tfrac12\sum_s\big[|T_s|^2+|D_s|^2-|T_0|^2\big](u_k,E),
$$

where $w_k$ is the share of the line emitted in depth bin $k$. Without absorption, flux conservation reduces the bracket to $(1-\gamma_h/\gamma_n)|D_s|^2$. Symmetric Laue ($\gamma_h=\gamma_n$) therefore moves no flux. An asymmetric geometry with $\gamma_h<\gamma_n$ gives a bright line, and one with $\gamma_h>\gamma_n$ a dark line. Kinematically, with $\int dE\,|D_s|^2=|b_s|^22\pi u\,\hbar cQ_s/|\hat n\cdot g|$ (DBR record) and a line wider than the Bragg width,

$$
\int S\,dE\simeq Y\,\mathcal L(E_B)\Big(1-\frac{\gamma_h}{\gamma_n}\Big)R_E(\bar u),\qquad R_E=\frac{E\,R_\theta}{\tan\theta_B},\quad R_\theta=\frac{\pi k|\chi_g|^2C^2\bar u}{2\gamma_h\sin2\theta_B},
$$

with $\bar u$ the mean remaining depth.

## DBR depletion (decision)

The same $|T|^2$ weight applies to the direct brem toward $\hat n$. Its depletion relative to the DBR gain is about $j(\hat n)/j(\hat Q)$. One-ray estimate at the #1166 mirror rays (17 MeV Si, 2 µm, BremsLib, channeling off; `scratch/dbr_depletion_probe.py`, $\int dE$ of the depth-weighted terms):

| reflection | $E_B$ | $\theta_n$ | $j(\hat n)/j(\hat Q)$ | depletion / gain |
|---|---|---|---|---|
| (0 2 −2) | 6990 eV | 57.5° | 3.2e-6 | −5.8e-6 |
| (1 1 −3) | 10003 eV | 47.5° | 6.9e-6 | −1.0e-5 |
| (−1 3 1) | 8588 eV | 51.5° | 3.9e-6 | −6.1e-6 |

DBR therefore keeps no depletion term, and its output is unchanged. The omission is listed in the DBR assumptions.

## Implementation

`spectrum/diffracted_brem.py::laue_mode_intensities` returns $\tfrac12\sum_s|T_s|^2$ and $\tfrac12\sum_s|D_s|^2$ from one eigen-solution. `spectrum/diffracted_characteristic.py` implements the rest:

1. `line_depth_yields`: per line, the unattenuated yield on 64 depth bins, using native rows, cutoff, shell cross sections, cascade yields and channeling flux weights, with no escape factor.
2. `kossel_term`: the bracket above for every depth bin on the reflection's Bragg window (`bragg_window`, step at most FWHM/8 of the nearest line).
3. `diffracted_characteristic_spectrum`: per reflection, the lines within 1.5 keV plus 100 FWHM of $E_B$, then $\mathcal L(E)\,(w\cdot\text{bracket})$, deposited with `line_quadrature`.

`runner/diffracted_brem.py::case_diffracted_characteristic` adds the result to `spec_characteristic` in `_spectrum_case_impl` and in streamed blocks, for the same perfect single-slab cases as DBR.

## Assumptions

- Those of DBR: two-beam Laue (both angles above $3\sqrt{|\chi_0|}$), perfect laterally infinite single slab, $\chi_g$ and $\chi_0$ taken at $E_B$ and flat across the window, and $\chi_g\chi_{-g}\to|\chi_g|^2$. The last means there is no anomalous (Borrmann) absorption, so contrast is reliable only for $\mu L/\gamma\lesssim1$. Thick-crystal Kossel contrast is dominated by Borrmann absorption and is not claimed.
- Unpolarized, isotropic lines with natural Lorentzian shapes.
- The two-beam absorption is photoabsorption ($\chi_0$, Henke), while native escape uses total $\mu$ (EPDL). The subtraction uses the same $\chi_0$, so the term vanishes off Bragg.
- Lines farther than 100 FWHM outside a Bragg window are skipped.
- No Bragg-case (reflection) Kossel lines, mosaic crystals, layer stacks or grooves.
- Electron-side orientation effects on ionization (Hall 1966, `MPCKVBM9`; Josefsson & Allen 1996, `IW7HEZGP`) are outside this claim.

## Checks

| check | result |
|---|---|
| $|T|^2$, $|D|^2$ vs `expm` of $N$ (strong coupling, absorbing) | 1e-8 (measured 7e-16) |
| $\chi_g\to0$: $|T|^2=e^{-k\operatorname{Im}\chi_0u/n_s}$, $D=0$, Kossel term 0 | 1e-12 |
| $n_s|T|^2+(Q_s/k)|D|^2=n_s$, real $\chi_0$, strong coupling | 1e-12 |
| symmetric Laue, no absorption: Kossel term 0 | 1e-12 |
| kinematic band vs $Y\mathcal L(E_B)(1-\gamma_h/\gamma_n)R_E$ (textbook $R_\theta$) | 1e-2 |
| depth yields × native escape vs `mc_characteristic_spectrum` (Ge, 100 keV, 2 µm) | 2e-3 |
| Ge Kα1 on the (−4 0 0) Kossel cone ($\gamma_n-\gamma_h=0.63$) | gain peak at the line; band ≈ 5% of native K band, peak +14% |
| runner adds exactly the Kossel term to `spec_characteristic`; out-of-claim cases add nothing | 1e-9 |

## Anchor

None yet. Zotero has no quantitative Kossel-profile dataset: searches for Kossel, Borrmann and fluorescence diffraction, and the Channeling Radiation and PXR-CBS collections, found none. Hall 1966 concerns the electron-side Borrmann effect, not photon Kossel contrast. A measured transmission Kossel profile with stated thickness and geometry would allow `anchored`.
