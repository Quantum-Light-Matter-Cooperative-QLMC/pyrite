# `cr-mott-bethe-factor` — Mott–Bethe numerator in the planar potential

Verifier: fresh context. Ledger: `docs/validation/ledger-channeling.md`. Code: `src/pyrite/materials/channeling/potential.py::planar_profile`.

## Claim

The electron–atom potential Fourier component is proportional to $(Z-f_0(g))/g^2$ with only the energy-independent X-ray form factor $f_0$ (Waasmaier–Kirfel, valid to $s\le 6\ \text{\AA}^{-1}$, $s=g/4\pi$). Anomalous $f'$, $f''$ are excluded.

## Independent derivation (before reading the body)

Gaussian units. An isolated atom has charge density $\rho(\mathbf r)=e\,[Z\delta(\mathbf r)-n_e(\mathbf r)]$. Its Fourier transform is $\rho(q)=e\,[Z-f_0(q)]$, because $f_0(q)=\int n_e e^{i\mathbf q\cdot\mathbf r}\,d^3r$. Poisson's equation $\nabla^2\varphi=-4\pi\rho$ gives

$$
\varphi(q)=\frac{4\pi e\,[Z-f_0(q)]}{q^2}.
$$

The electron has charge $-e$, so its potential energy is

$$
u(q)=-\frac{4\pi e^2\,[Z-f_0(q)]}{q^2}.
$$

Cross-check against the Mott–Bethe form. The electron scattering amplitude is $f^e=-\dfrac{m}{2\pi\hbar^2}\,u(q)$ (Born approximation). With $u(q)=-4\pi e^2(Z-f_0)/q^2$, $q=4\pi s$ and $\hbar^2/m=a_0e^2$:

$$
f^e=\frac{m}{2\pi\hbar^2}\frac{4\pi e^2(Z-f_0)}{16\pi^2s^2}
=\frac{Z-f_0}{8\pi^2a_0s^2}.
$$

This is the International Tables expression, so the two forms are the same quantity and $u(q)=-\frac{2\pi\hbar^2}{m}f^e(q)$. ✓

The crystal Fourier coefficient is $U_{\mathbf g}=V^{-1}\sum_j u_j(g)\,e^{i\mathbf g\cdot\mathbf R_j}e^{-B_js^2}$. Thermal motion convolves each atom's total charge, nucleus and cloud alike. The factor $e^{-B s^2}$ therefore multiplies the whole bracket $Z-f_0$, not only $f_0$. With $B=8\pi^2\langle u_x^2\rangle$ and $s=g/4\pi$: $e^{-g^2\langle u_x^2\rangle/2}=e^{-Bs^2}$. ✓

Anomalous $f'$, $f''$ describe photon resonance. They are not charge-density Fourier components, so they do not enter an electrostatic potential. Only $f_0$ applies.

## Filters

- Units: $e^2=14.3996\ \text{eV\,\AA}$, so $e^2/(V g^2)$ is eV. Pass.
- Limit $g\to\infty$: $f_0\to0$ and $U\to -4\pi e^2Z/(Vg^2)$, the bare nuclear Coulomb. Pass. $g\to0$: $Z-f_0\sim g^2$, so $U_0$ is finite. $g=0$ is excluded and $U_0\equiv0$ is a stated convention.
- Sign: $Z-f_0>0$ for $g>0$, so $U_n<0$ for the positive-structure-factor harmonics and wells sit on the atoms. Pass.

## Comparison with the implementation

`planar_profile` forms `(Z_TABLE[el] - f0[1:]) / g_n[1:]**2 * site_sum[1:]` with `site_sum = Σ phase · exp(-B s²)`, and multiplies by `-4π E2_EV_ANG / V`. This is the derived expression exactly. `f0` comes from `cromer_mann_f0`, which reproduces `xraydb.f0` (Waasmaier–Kirfel, no $f'$, $f''$) to all printed digits at $s=0.26,\,0.52,\,1,\,3,\,5.9\ \text{\AA}^{-1}$, e.g. 8.72184106 for Si at $s=0.26$. The harmonic cutoff is $n_{\max}=\lfloor 4\pi s_{\max}/g\rfloor$ with $s_{\max}=6$, and $n=0$ is handled separately.

Numerical point, Si (220) at 293.15 K, with $a=5.431\ \text{\AA}$, $B=0.48166\ \text{\AA}^2$ from `thermal_b_factor('silicon','Si',293.15)`, $s=0.26040$, $f_0=8.71652$ from `xraydb`, structure factor $S=8$:

| quantity | independent | implementation |
| --- | --- | --- |
| $U_{220}$ [eV] | $-4.31575$ | $-4.31589$ |

The $3\times10^{-5}$ relative difference matches the catalog lattice constant: the implementation period is 1.920113 Å against $a/(2\sqrt2)=1.920148$ Å. This is a tolerance-level difference, not a formula difference.

## Adjudication

Re-derivation matches, including factor, sign, units and the Debye–Waller convention. The anomalous terms are absent from `planar_profile`. Verdict: `rederived`.

Out of scope for this claim: `materials.crystal.U_g` adds $f'$ for edge-prone elements, which is the known CBS issue. `planar_profile` does not call `U_g`. The ledger check "diamond (220) equals $-U_g$" is valid only because diamond is not edge-prone.
