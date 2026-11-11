# `cr-planar-exit-angular` re-derivation

Claim: `montecarlo/channeling/exit_angular.py::{coherent_exit_distribution,density_exit_distribution,open_exit_distribution,classical_exit_angles,incidence_nodes,ExitAngularDistribution}`. Sources: Shulga, Shul'ga, Barsuk, Chaikovska & Chehab, NIMB 402, 16 (2017), Eqs. 1–9 and Fig. 4 (Zotero `BCIDCFDA`, read in full); Shul'ga & Shulga, PLB 769, 141 (2017), Eqs. 13–14 (cited by the module; not opened here, because NIMB 402 Eqs. 6–9 restate the same amplitude for the planar case).

## Independent derivation (before reading the body)

**Transverse equation.** NIMB 402 Eq. 8 is $i\hbar v\,\partial_z u=\bigl[\hat p_x^2/2\varepsilon+U(x)\bigr]u$ (with $c=1$), so the kinetic mass is $\varepsilon/c^2=\gamma m$. Here $\varepsilon$ is the *total* energy: Eq. 1 defines $\varepsilon_\parallel=\sqrt{p_\parallel^2c^2+m^2c^4}\approx\varepsilon$, "the full electron energy". Write $U(x)=\sum_hU_he^{ihgx}$ with $g=2\pi/d$ and $U_{-h}=U_h^*$. In the plane-wave basis $\lvert k+mg\rangle$ of Bloch block $k$,

$$
H_{mm'}=\frac{\hbar^2(k+mg)^2}{2\gamma m}\,\delta_{mm'}+U_{m-m'},
\qquad
\langle k+mg\vert U\vert k+m'g\rangle=\frac1d\int_0^d e^{-i(m-m')gx}U(x)\,dx=U_{m-m'} .
$$

Diagonalize $H=C\,\mathrm{diag}(E_n)\,C^\dagger$, where column $n$ of $C$ is eigenvector $n$, so $\psi_n=\sum_mC_{mn}\lvert k+mg\rangle$.

**Exit amplitude.** A plane wave at incidence $\psi$ has $k_x=p\psi/\hbar=k+m_0g$, with $k$ folded into the Brillouin zone. Then $\lvert m_0\rangle=\sum_n\psi_n\,C^*_{m_0n}$. Propagating to $z=L$ multiplies each $\psi_n$ by $e^{-iE_nL/\hbar v}$, and projecting onto $\lvert m\rangle$ gives

$$
\boxed{a_m(L)=\sum_nC_{mn}\,e^{-iE_nL/\hbar v}\,C^*_{m_0n}=\bigl(C\,e^{-iEL/\hbar v}C^\dagger\bigr)_{mm_0}} .
$$

Transverse momentum is conserved across the exit face, so order $m$ leaves at $\theta_m=\hbar(k+mg)/p=\psi+(m-m_0)\lambda/d$. Here $\lambda/d=\hbar g/p$, and the exit angle is measured from the plane, which is the beam axis at $\psi=0$. Because $C$ is unitary in the truncated basis, $\sum_m\lvert a_m\rvert^2=(C C^\dagger)_{m_0m_0}=1$. At $L=0$, $a_m=\delta_{mm_0}$. NIMB Eq. 9 has $d\sigma/d\theta_x\propto(p/2\pi)\lvert\int dx\,e^{-iq_xx}u(x,L)\rvert^2$. For a plane-wave entry this becomes the comb $\sum_m\lvert a_m\rvert^2\delta(\theta-\theta_m)$, a probability per incident electron once divided by the illuminated area (Eq. 4).

**Divergence.** A beam with Gaussian angular spread is an incoherent mixture of plane waves, so the plane waves do not interfere:

$$
\frac{dw}{d\theta}=\sum_m\int d\psi\,\mathcal N(\psi;\psi_0,\sigma)\,\lvert a_m(\psi)\rvert^2\,\delta\bigl(\theta-\theta_m(\psi)\bigr)
=\sum_j\mathcal N(\theta-j\lambda/d;\psi_0,\sigma)\,P_j(\theta-j\lambda/d),
$$

where $j=m-m_0$ and $P_j(\psi)$ is the weight on relative order $j$. The second form is exact and needs no quadrature. I use it for the independent check below.

**Open system.** If $\rho$ has been translation-averaged over the entry point, it commutes with lattice translations. Then $\langle\kappa+hg\vert\rho\vert\kappa'+h'g\rangle=0$ unless $\kappa=\kappa'$, so $\rho=\bigoplus_\kappa\rho_\kappa$. The momentum distribution $\langle q\vert\rho\vert q\rangle$ at $q=\kappa+hg$ is the $hh$ diagonal of block $\kappa$ in the plane-wave basis. If $\rho_\kappa$ is held in the eigenbasis with eigenvector matrix $V_\kappa$, the weight is $(V_\kappa\rho_\kappa V_\kappa^\dagger)_{hh}$ at $\theta=\hbar(\kappa+hg)/p$. The sink is trace-decreasing (absorption above the cutoff, or out of the basis). The resolved weights therefore sum to $\operatorname{tr}\rho(L)=1-F(L)$, where $F$ is the cumulative removed flux. That flux has no angle in the model, so it can only be reported as an unresolved remainder: resolved plus unresolved equals 1, up to the integrator's balance defect.

**Classical reference.** From NIMB Eq. 1, $(\varepsilon/c^2)\ddot x=-U'(x)$. With $z=vt$ this gives $x''=\ddot x/v^2=-U'/(\varepsilon\beta^2)=-U'/(pv)$, with $pv=\gamma mc^2\beta^2$. The deflection function is $\theta(x_0)=x'(L)$ for $x(0)=x_0$ and $x'(0)=\psi$ (Eq. 2). For a beam that is broad compared with $d$, the distribution is $dw/d\theta=d^{-1}\sum_{\rm branches}\lvert dx_0/d\theta\rvert$, which is a uniform average over $x_0\in[0,d)$ (Eqs. 3–5). Its rainbow peaks sit at the extrema of $\theta(x_0)$. Transverse energy is conserved, so $\tfrac12pv\,\theta^2\le\tfrac12pv\,\psi^2+U(x_0)-U_{\min}$.

## Cheap filters

| Filter | Result |
| --- | --- |
| Units | $p/\hbar=\gamma\beta m c^2/(\hbar c)$ in Å$^{-1}$; $E_nL/\hbar v$ is dimensionless (eV·Å/(eV·Å)); $\theta=(k+mg)/(p/\hbar)$ is dimensionless; $U'/(pv)$ has units Å$^{-1}$, matching $x''$. Pass. |
| Limits | $L=0$ returns the incident wave; unitarity; $\theta\to-\theta$ symmetry at $\psi=0$ for an even $U$; classical $L=0$ gives $\theta=\psi$. Pass. |
| Signs/conventions | The overall sign of the phase and the eigenvector gauge cancel in $\lvert a_m\rvert^2$ (any column phase $e^{i\phi_n}$ cancels between $C_{mn}$ and $C^*_{m_0n}$). `conj` must sit on the entry-order row, which it does. Pass. |

## Diff against the implementation

- `coherent_exit_distribution` (`exit_angular.py:141`): $k_x=p\psi$. `transverse_states` folds $k$ into $(-g/2,g/2]$ (`bloch.py:162-164`). The code then sets `m0 = round((k_x - k)/g)` (line 179), which is correct at the $-g/2\to+g/2$ edge shift. Line 185 is `amplitude = c @ (phase * conj(c[m0 + M, :]))`, which is $\sum_nC_{mn}e^{-iE_nL/\hbar v}C^*_{m_0n}$ exactly. `eigh` returns eigenvectors as columns, so the convention is right. The angles are $(k+mg)/p$ (line 186). $\hbar v=\hbar c\beta$ uses the kinetic-energy $\beta$, and $\gamma=1+T/mc^2$. The Hamiltonian (`bloch.py:_hamiltonian`) uses $U_{m-m'}$ with $U_{-j}=U_j^*$. This matches `_synthesize`, which builds $U(x)=U_0+2\,\mathrm{Re}\sum_{m\ge1}U_me^{imgx}$, the same $e^{+imgx}$ convention as my derivation. An entry order outside the basis is counted as unresolved. Matches.
- `incidence_nodes` (line 120): uniform midpoints over $\pm4\sigma$ with Gaussian weights normalized to 1. Truncating at $4\sigma$ reduces the effective RMS by $5\times10^{-4}$ relative, which is negligible. Matches the ledger.
- `density_exit_distribution` (line 198): line 213 is `_sandwich(V, rho)` $=V\rho V^\dagger$ per block, and the code takes the real diagonal. The angles are $(\kappa_r+hg)/p$, with $\kappa_r=rg/n_{\rm cells}\in[0,g)$ (`density.py:212`), and $p$ is taken at the basis $\gamma$. The diagonal is that of the full plane-wave window, so the folding convention does not matter: every component lands at its absolute $k_x$. Matches.
- `open_exit_distribution` (line 228): passes `result.total_flux[-1]` (entry + free + overflow + reprojection) as unresolved. This is the sink weight of the derivation. Matches.
- `classical_exit_angles` (line 256): $pv=\gamma m c^2\beta^2$ (line 281). The slope is $U'(x)=2\,\mathrm{Re}\sum_m imgU_me^{imgx}$ (lines 284–290), consistent with `_synthesize`, and is interpolated linearly with period $d$. Lines 303–307 are a standard kick–drift–kick velocity Verlet with $x'(0)=\psi$ and $\theta=x'(L)$. The impact parameters are uniform midpoints of one period. Matches.

## Independent numeric checks

Script: `scratch/verify_exit_angular.py` (plus `_scan`, `_open`, `_cls_scan`). It builds its own operators from samples of $U(x)$ on 512 points using `profile.potential`, and does not use `potential_eV` or any PyRITE solver.

**(a) Amplitude.** I used two independent propagators: an own plane-wave eigen-solve with $U_h$ obtained by FFT of the sampled $U(x)$, $M=32$; and an own Strang split-step with kinetic half-steps (the reverse of the test's ordering), propagating $u$ with $\psi=e^{ik_xx}u$, so no folding is involved. The fold-edge case is $\psi$ with $k_x=g/2$.

| Case ($T$, $L$, $\psi$) | $P_0$ PyRITE | $P_0$ split-step | max diff vs own eigen | max diff vs split-step | $\sum_m$ |
| --- | --- | --- | --- | --- | --- |
| 3.489 MeV, 800 Å, 0 | 0.81728 | 0.81728 | $9\times10^{-14}$ | $6\times10^{-7}$ | $1-1\times10^{-15}$ |
| 3.489 MeV, 800 Å, 310 µrad | 0.62317 | 0.62317 | $3\times10^{-13}$ | $5\times10^{-7}$ | $1+1\times10^{-15}$ |
| 3.489 MeV, 400 Å, −900 µrad | 0.12071 | 0.12071 | $6\times10^{-14}$ | $3\times10^{-7}$ | $1-2\times10^{-15}$ |
| 20 MeV, 1500 Å, 170 µrad | 0.15318 | 0.15318 | $4\times10^{-14}$ | $9\times10^{-8}$ | $1+1\times10^{-15}$ |
| 3.489 MeV, 800 Å, $k_x=g/2$ | 0.09837 | 0.09837 | $9\times10^{-14}$ | $4\times10^{-7}$ | $1+2\times10^{-15}$ |

The exit angles equal $\psi+j\lambda/d$ to $10^{-14}$ in every case. With the 300 µrad divergence at total 4 MeV and 800 Å, PyRITE's 241-node result (smoothed by 20 µrad) agrees with my exact convolution formula to 0.3% of the peak.

**(b) Open system.** I used a zero-kernel `DensityMatrixSolver` with $n_{\rm cells}=8$ and $H=8$, entering at lattice points $l=-3,5,13,-12$. This covers blocks $r\ge n_{\rm cells}/2$ and negative orders, which the anchored test does not exercise. The angular weights match my own $M=32$ eigen-solve to $\le3\times10^{-6}$, which is the basis truncation at $H=8$, and the angles lie on the $g/n_{\rm cells}$ lattice to $10^{-14}$. With TDS plus electronic scattering (4 MeV, 800 Å, 300 µrad), my own $\mathrm{diag}(V\rho V^\dagger)$ equals the code's weights exactly. Resolved plus unresolved minus 1 is $-6.777\times10^{-6}$, identical to `balance_defect`. The RMS exit angle is 1.595 mrad, matching the ledger's 1.59.

**(c) Classical.** My own integrator is DOP853 with rtol $10^{-10}$, using an FFT derivative of the sampled $U$ and 1500 impact parameters. It agrees with `classical_exit_angles` at $dz=0.5$ Å to $\le0.014$ µrad.

| Case | Own rainbows (mrad) | PyRITE (mrad) | Published (mrad) |
| --- | --- | --- | --- |
| total 4 MeV ($T=3.489$ MeV), 800 Å | 0.716, 1.874, 2.578, 3.234 | identical | 0.70, 1.84, 2.58, 3.21 |
| kinetic 4 MeV, 800 Å | 0.358, 1.635, 2.364, 3.029 | identical | — |
| total 50 MeV, 1700 Å | 0.026, 0.526, 0.853 | identical | 0.05, 0.55, 0.885 |

`pyrite-dev test tests/montecarlo/quantum/test_channeling_exit_angular.py`: 11 passed.

## Energy convention and the 4 MeV quantum panel

The energy convention is total energy. NIMB Eq. 1 says so explicitly. The classical 4 MeV panel is then reproduced only at total 4 MeV and 800 Å: the maximum rainbow error is 0.034 mrad, and there are exactly four rainbows. At kinetic 4 MeV the error is 0.34 mrad at 800 Å and 0.33 mrad at 1300 Å. At total 4 MeV it is 0.77 mrad at 400 Å and 0.21 mrad at 1300 Å. PyRITE's (110) potential is 21.04 eV deep (−14.42 to +6.62 eV), which agrees with the paper's ~21 eV (Fig. 1). The ledger's convention claim is confirmed.

The 4 MeV quantum panel is inconsistent with the source's own stated parameters. I scanned $L$ from 100 to 2600 Å in 50 Å steps, using my exact divergence convolution against the digitized curve and taking the relative RMS residual over the peak height:

| Energy, $L$ | Residual | Peaks (mrad) |
| --- | --- | --- |
| total 4 MeV, 800 Å (stated) | 0.356 (PyRITE 0.355) | −1.34, −0.03, 1.37 |
| total 4 MeV, 390–400 Å | 0.067–0.069 | −3.11, −1.62, −0.03, 1.55, 3.14 |
| kinetic 4 MeV, 1300 Å | 0.017 | −2.92, −1.62, ±0.47, −0.03, 1.65, 2.95 |
| published | — | −3.01, −1.62, −0.50, −0.03, 0.44, 1.65, 2.95 |

No single (energy, $L$) reproduces both panels with this potential. The classical panel requires total 4 MeV at 800 Å; the quantum panel requires kinetic 4 MeV at about 1300 Å, or total 4 MeV at about 400 Å. The quantum peaks at ±1.62 mrad coincide with the total-energy comb spacing $\lambda/d=1.628$ mrad. That alone is not diagnostic: at kinetic 4 MeV and 1300 Å ($\lambda/d=1.441$ mrad) the $\psi$-dependence of $P_{\pm1}$ shifts the divergence-broadened peak centroids to ±1.62 mrad as well. Two independent propagators agree with PyRITE at the stated parameters. The ledger's recorded discrepancy (residuals 0.07 and 0.016) is therefore confirmed as a source-side inconsistency, not an implementation error. Leaving it out as an anchor is the right call.

## Findings (none a discrepancy in the code)

1. The module docstring (`exit_angular.py:9`) writes $dw/d\theta_x=(p/2\pi)\lvert\int dx\,e^{-iq_xx}u\rvert^2$. This holds per unit transverse length (NIMB Eq. 9 carries $L_y$ and divides by the illuminated area in Eq. 4). The implementation never uses this normalization, because it returns probabilities per electron, so the formula is descriptive only.
2. The density-solver distribution is quantized at $\lambda/(d\,n_{\rm cells})$, and for $r\ge n_{\rm cells}/2$ its plane-wave window is offset by one order relative to the folded coherent window. Both effects are at truncation level ($3\times10^{-6}$ at $H=8$) and are documented.
3. I did not independently recompute the ledger's "0.94 coherent / 1.35 classical" RMS values or the 50 MeV residual (0.12). The PLB 769 Eqs. 13–14 citation was not opened.

## Verdict

`rederived`. The coherent amplitude, the divergence average, the open-system plane-wave diagonal with its sink bookkeeping, and the classical Verlet deflection all match an independent derivation and independent numerics to round-off or truncation level. The energy-convention claim (total energy) and the recorded 4 MeV quantum-panel discrepancy, which lies in the source rather than the code, are both confirmed.
