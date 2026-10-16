# `cr-line-shape` re-derivation

Claim: `montecarlo/channeling/emission.py::{line_shape,emission_lines,channeling_spectrum}` (width helper `_coherence_decay`). Sources: Lindblad coherence decay of `cr-density-matrix-split-operator` (Breuer & Petruccione §3.3), the Doppler scaling of a damped emitter, and walkthrough §6.3.

## Independent derivation (before reading the bodies)

**Lorentzian width from the quantum regression theorem.** The line spectrum is the Fourier transform of $G(\tau)=\mathrm{Tr}[d^\dagger e^{\mathcal L\tau}(d\rho)]$, with $d=\sum a_{fi}\lvert f\rangle\langle i\rvert$ summed over **every** term that radiates at the same frequency into the same mode. With $\mathcal L\sigma=-\tfrac{i}{\hbar v}[H,\sigma]+J(\sigma)-\tfrac12\{\hat\Gamma,\sigma\}$, $\hat\Gamma=C(x,x)$, a coherence $\sigma=\lvert f\rangle\langle i\rvert$ obeys

$$
\frac{d\sigma_{fi}}{dz}=\Bigl[i\frac{\Omega_{if}}{v}-\frac{\Gamma_i+\Gamma_f}{2}\Bigr]\sigma_{fi}+\sum_{(f'i')}\langle f'\vert J(\lvert f\rangle\langle i\rvert)\vert i'\rangle\,\sigma_{\ldots}+\text{(non-secular terms)}.
$$

The secular approximation keeps every coupling between coherences of **equal** frequency and drops only the rapidly rotating ones. In a 1-D Bloch problem the bound bands are nearly flat, so $\lvert f_{k'}\rangle\langle i_{k'}\rvert$ in every Bloch block $k'$ has the same frequency as $\lvert f_k\rangle\langle i_k\rvert$ to within much less than the width. $J$ transfers coherence between blocks: a momentum kick $e^{iqx}$ acting on both ket and bra leaves the environment without which-path information. The correct secular decay of the radiating coherence is therefore

$$
\boxed{\kappa_{if}=\frac{\Gamma_i+\Gamma_f}{2}-\mathrm{Re}\sum_{k'}\frac{a_{fi}^{(k')*}}{a_{fi}^{(k)*}}\,\langle f_{k'}\vert J(\lvert f_k\rangle\langle i_k\rvert)\vert i_{k'}\rangle}
$$

(the amplitude ratio removes the eigenvector gauge). This is the familiar result that the coherence width is set by $\tfrac12\int d\sigma\,\lvert f_{ii}(q)-f_{ff}(q)\rvert^2$ plus inelastic loss: small-$q$ scattering, which keeps both states alike, does not broaden. Restricting the gain to $k'=k$ keeps only momentum transfers that are multiples of $g$. That restriction is not the secular approximation, and it tends to the pure-loss width $(\Gamma_i+\Gamma_f)/2$ as the Bloch grid is refined.

Transverse half width: $\hbar v\kappa_{if}$. In the lab the observed phase advances as $\Omega t(1-\beta\cos\theta)$, so the photon centre is $E_c=\Delta E_\perp/D$ and the Lorentzian HWHM is $\hbar v\kappa/D$.

**Beam energy spread.** $E_c(\gamma)=\Delta E_\perp(\gamma)/D(\gamma)$ and $d\beta/d\gamma=1/(\gamma^3\beta)$, so

$$
s\equiv\frac{d\ln E_c}{d\ln\gamma}=\frac{\cos\theta}{\gamma^2\beta D}+\frac{d\ln\Delta E_\perp}{d\ln\gamma}\ \xrightarrow{\theta=0}\ \frac{1+\beta}{\beta}+\frac{d\ln\Delta E_\perp}{d\ln\gamma}.
$$

The Gaussian RMS is $\sigma=E_c\,s\,\sigma_\gamma/\gamma$ with $\sigma_\gamma=\sigma_T/mc^2$. The walkthrough's original $2\delta\gamma/\gamma$ is the kinematic part only. A harmonic well has $\Delta E\propto\gamma^{-1/2}$, giving $s=1.5$; a square well has $\gamma^{-1}$, giving $s=1$; a cusp-like $\lvert x\rvert^p$ well with $p<2$ has $-p/(p+2)$. **The worker's correction is right.**

**Voigt.** The Lorentzian of HWHM $\Gamma_L$ is convolved with a unit Gaussian of RMS $\sigma$. The detector cone and in-plane divergence are added by quadrature over directions, each with its own $\theta_e$.

## Independent numerical checks

**Beam-spread slope** (own plane-wave eigensolve at $\gamma(1\pm10^{-3})$, Si(110), 17 MeV, fixed indices):

| line (block 0) | $d\ln\Delta E/d\ln\gamma$, own | impl `gamma_exponent` | $s$ numeric, $d\ln E_c/d\ln\gamma$ | $s$ formula |
| --- | --- | --- | --- | --- |
| $1\to0$ | $-0.27079$ | $-0.27079$ | 1.72964 | 1.72964 |
| $3\to0$ | $-0.18265$ | $-0.18265$ | 1.81778 | 1.81778 |
| $2\to1$ | $+0.02833$ | $+0.02833$ | 2.02875 | 2.02875 |
| $3\to2$ | $-0.25727$ | $-0.25727$ | 1.74316 | 1.74316 |

At $\theta=1/\gamma$ the numeric $s$ is 0.72936, and the formula gives 0.72936, so the off-axis kinematic term is confirmed. All 22 lines in four blocks agree.

**Voigt:** adaptive-quadrature area is $1.0000000000$, and the value at detuning 0.7 equals a direct convolution integral to $10^{-10}$ for $(\Gamma_L,\sigma)=(0.2,1),(3,0.5),(1,1)$.

**Width — the discrepancy.** Si(110), 17 MeV, TDS + electronic kernel, $H=8$, margin 5 eV. Three independent determinations of the $1\to0$ coherence decay:

1. *Full dynamics, block 0 only.* I seeded $\lvert1\rangle\langle0\rvert$ in block 0 and propagated it with `DensityMatrixSolver.propagate` (validated, #504). The fitted decay of $\lvert\sigma^{(0)}_{01}\rvert$ is 3.26e-4 Å⁻¹, within 3% of the implemented $\kappa$ (3.35e-4). So the implementation computes the same-block coherence decay correctly.
2. *Quantum regression, all blocks.* With the same propagation, $G(\tau)=\sum_{k'}a^{(k')*}\sigma^{(k')}_{01}$ decays at **1.79e-4 Å⁻¹**. A static evaluation of the boxed all-block $\kappa$ using `solver.jump` and `solver.loss_rate` gives 1.785e-4 Å⁻¹ (HWHM 0.352 eV).
3. *Grid convergence.*

| $n_{\rm cells}$ | impl HWHM $1\to0$ (eV) | all-block HWHM (eV) | impl $2\to1$ | all-block $2\to1$ | pure loss $1\to0$ |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.662 | 0.352 | 0.428 | 0.220 | 1.281 |
| 8 | 0.774 | 0.352 | 0.525 | 0.220 | 1.281 |
| 16 | 0.875 | 0.352 | 0.619 | 0.219 | 1.281 |
| 32 | 0.968 | 0.352 | 0.709 | 0.219 | 1.281 |

The implemented width is not converged in the Bloch grid. It grows by 46% from $n_{\rm cells}=4$ to 32 and drifts toward the pure-loss width. The all-block width is grid-independent to $10^{-3}$, and the regression fit converges with it (1.79, 1.78, 1.80 ×10⁻⁴ Å⁻¹ at 4, 8, 16). At the test grid ($n_{\rm cells}=4$) the implementation is 1.9× too wide for $1\to0$ and $2\to1$, and 1.6× for $3\to2$. The ledger's FWHM/E of 16.8% for $1\to0$ becomes about 8.9%. For a line whose $\Delta E_\perp$ disperses across blocks by more than its width ($3\to2$: 0.5 eV spread), the transferred coherence dephases and the true shape lies between the two values, so it needs the block-resolved spectrum. Scaling the kernel by 1/4 and 1/16 leaves the 3% same-block residual of (1) unchanged, so that residual is also an inter-block return effect, not an $O(\kappa/\omega)$ non-secular term.

## Diff against the implementation

- `line_shape` is `scipy.special.voigt_profile(E - E_c, sigma, hwhm)`, which matches.
- In `channeling_spectrum`, `slope = cos(theta_e)/(gamma**2*beta*denom) + gamma_exponent`, `sigma = |centre*slope|*sigma_gamma/gamma` with `sigma_gamma = energy_spread_eV/M_E_EV`, and Lorentzian `hwhm_eV/denom`. These match the derived $s$, $\sigma$ and Doppler scaling. The test name `test_energy_spread_broadens_by_twice_relative_gamma` describes the superseded factor 2, although its body uses $s$.
- **First divergent term:** `emission.py::_coherence_decay` (lines 337–349; the gain is line 343) takes the gain as `inside[block, upper, lower]`, i.e. $\langle i_k\vert J(\lvert i_k\rangle\langle f_k\rvert)\vert f_k\rangle$ of the **same** Bloch block only. The terms $\langle i_{k'}\vert J(\lvert i_k\rangle\langle f_k\rvert)\vert f_{k'}\rangle$, $k'\ne k$, which `solver.jump` returns in the other blocks of `inside`, are dropped. The ledger formula $\kappa_{if}=(\Gamma_i+\Gamma_f)/2-\mathrm{Re}\langle i\vert J(\lvert i\rangle\langle f\rvert)\vert f\rangle$ carries the same per-block restriction, so the claim itself, not just the code, needs the inter-block sum.
- Concrete failing case: `emission_lines(DensityMatrixSolver(planar_profile("silicon",(1,1,0),293.15), 17e6, tds_kernel(...,n_cells=n,max_order=8)+electronic_kernel(...,n_cells=n,max_order=8), margin_eV=5.0))`. The $1\to0$ `hwhm_eV` is 0.662/0.774/0.875/0.968 eV for $n=4/8/16/32$ against a converged 0.352 eV. The ledger check "widths converge in the plane-wave band ($H=8$ vs 16)" tests $H$ only, not $n_{\rm cells}$.
- Suggested fix (for the task owner): accumulate the gain over every block $k'$ whose $\Delta E_\perp^{(k')}$ lies within the width, gauge-aligned by $a^{(k')}/a^{(k)}$, or treat the block-summed coherence as one cluster for the width. Add an $n_{\rm cells}$ convergence check.

The yields, the centres $E_c$ and the beam-spread $\sigma$ are unaffected. Peak heights and the default cluster tolerance inherit the width error.

## Verdict

`discrepancy`, in the Lorentzian width only. The beam-spread factor $s$ (the correction to the walkthrough) and the Voigt and Doppler handling are `rederived`.

## Resolution (worker, after the verdict)

- `_coherence_decay` is replaced. For each transition `(upper, lower)` the coherences of every Bloch block that carries it form a `LineGroup` with generator `K_kk = iΔ_k/ħv − (Γ_i+Γ_f)_k/2 + J_kk`, `K_k′k = ⟨f_k′|J(|f_k⟩⟨i_k|)|i_k′⟩`, taken from `solver.jump` in every block. The reported `hwhm_eV` is the all-block secular value of the boxed formula above. The spectrum uses the eigenmodes of `K` (quantum regression), with complex weights `w_m = (a†V)_m (V⁻¹s)_m`, so dispersive lines such as 3→2 are handled block-resolved rather than with a single width.
- Si(110), 17 MeV, TDS + electronic, `H = 8`, margin 5 eV: the 1→0 half width is now 0.352 eV at `n_cells` = 4, 8 and 16, matching the value above. New tests: `test_line_widths_are_independent_of_the_bloch_grid` (4 vs 16, 2e-3 for 1→0, 2% for 2→1) and `test_group_modes_reproduce_the_propagated_correlation`. The second propagates a coherence seeded in one block with `DensityMatrixSolver.propagate` and compares `G(z)` with the group modes for the flat 1→0 and the dispersive 3→2, within 3% of `|a|²` over 1.2 decay lengths.
- The ledger claim and values are rewritten (FWHM/E for 1→0 is 8.9%, as estimated above). The test name is now `test_energy_spread_broadens_by_the_line_energy_slope`.
- A stated limit: coherence transfer is kept only between the same band-index pair. Transfer into other pairs of similar frequency (dense free bands, capture) is dropped, so capture widths are upper estimates.

## Re-validation (2026-10-15, fresh context, after the resolution)

Claim under review: the `LineGroup` generator, its mode spectrum (`LineGroup.modes`, `EmissionLines.spectrum`) and the reported all-block `hwhm_eV`. I derived the generator and the mode form before reading the new bodies.

**Generator.** With $\mathcal L\sigma=-\tfrac{i}{\hbar v}[H,\sigma]+J(\sigma)-\tfrac12\{\hat\Gamma,\sigma\}$, the element $\sigma_k=\langle f_k\vert\sigma\vert i_k\rangle$ rotates as $e^{-i(E_f-E_i)z/\hbar v}=e^{+i\Delta_k z/\hbar v}$. $J(\sigma)=\sum_r L_r\sigma L_r^\dagger$ maps the block-diagonal coherence $\lvert f_k\rangle\langle i_k\rvert$ into blocks $k'=k+r$, so

$$
\frac{d\sigma_{k'}}{dz}=\sum_k K_{k'k}\sigma_k,\qquad
K_{k'k}=\langle f_{k'}\vert J(\lvert f_k\rangle\langle i_k\rvert)\vert i_{k'}\rangle
+\delta_{k'k}\Bigl[\frac{i\Delta_k}{\hbar v}-\frac{\Gamma_{i,k}+\Gamma_{f,k}}{2}\Bigr],
$$

with the column index the source block. The code builds column `col` from `solver.jump` of `rho[block, lower, upper] = 1`, which is this matrix. The solver's no-jump propagator is $e^{-iH_{\rm eff}dz/2\hbar v}$ on the ket, so the sign $+i\Delta/\hbar v$ agrees with the propagated dynamics.

**Mode spectrum.** By the quantum regression theorem $G(\tau)=\mathrm{Tr}[d^\dagger e^{\mathcal L\tau}(d\rho)]=\sum_k\bar a_k\sigma_k(\tau)$ with $\sigma(0)=s$, $s_k=a_kP_{i,k}=Y_k/\bar a_k$. With $K=V\,\mathrm{diag}(\mu)\,V^{-1}$,

$$
G(\tau)=\sum_m w_m e^{\mu_m\tau},\qquad w_m=(a^\dagger V)_m(V^{-1}s)_m,\qquad \sum_m w_m=G(0)=\sum_k Y_k .
$$

The spectrum is $S(\varepsilon)=\tfrac1\pi\mathrm{Re}\int_0^\infty G\,e^{-i\varepsilon\tau/\hbar v}d\tau/\hbar v=\tfrac1\pi\mathrm{Re}\sum_m w_m/(h_m+i(\varepsilon-\varepsilon_m))$, with $\varepsilon_m=\hbar v\,\mathrm{Im}\,\mu_m$ and $h_m=-\hbar v\,\mathrm{Re}\,\mu_m$. This equals the claimed $\mathrm{Re}\sum_m\bar w_m(i/\pi)/(\varepsilon-\varepsilon_m+ih_m)$, because $-i/(x-ih)=1/(h+ix)$. Convolution with a unit Gaussian of RMS $\sigma$ maps $(i/\pi)/(x+ih)$ to $w(z)/(\sigma\sqrt{2\pi})$, $z=(x+ih)/(\sigma\sqrt2)$ (Faddeeva). I checked that identity by direct quadrature to $10^{-15}$. Normalization: $\int S\,d\varepsilon=\mathrm{Re}\sum w_m$. Positivity is not guaranteed algebraically for a non-stationary source and a truncated generator, so I checked it numerically. The initial-slope decay $-\mathrm{Re}\,\dot G(0)/G(0)$ for a source in block $k$ is the reported $\kappa_k$, the boxed all-block formula above.

**Independent checks** (`scratch/cr505reval/line_corr.py`). Configuration: Si(110), 17 MeV, TDS + electronic, $H=8$, margin 5 eV. The amplitudes are my own commutator dipoles (equal to `_block_dipoles` to $6\times10^{-10}$), and my own $K$ comes from `solver.jump` and `solver.loss_rate`. For each line I seeded $X=d\rho_0$ ($P_i=1$ in every block, Hermitian part) and propagated it with `DensityMatrixSolver.propagate` over six decay lengths. That propagation keeps every coherence, including transfer into other pairs and the off-diagonal $\hat\Gamma$.

| $n_{\rm cells}$ | line | `hwhm_eV` impl | own all-block | eig. diff | max $\lvert G-G_{\rm modes}\rvert/\lvert G_0\rvert$ (complex) | spectrum max diff / peak | min $S$/peak (±4 eV) | $\Delta E$ spread over blocks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1→0 | 0.3522 | 0.3522 | 3e-9 | 0.011 | 0.007 | +7.7e-3 | 0.001 |
| 8 | 1→0 | 0.3518 | 0.3518 | 3e-9 | 0.011 | 0.007 | +7.7e-3 | 0.001 |
| 16 | 1→0 | 0.3517 | 0.3517 | 2e-9 | 0.011 | 0.007 | +7.7e-3 | 0.001 |
| 4 / 16 | 2→1 | 0.2203 / 0.2192 | same | 2e-9 | 0.009 / 0.008 | 0.007 | +3.1e-3 | 0.039 |
| 4 / 16 | 3→2 | 0.2005 / 0.2043 | same | 3e-9 | 0.011 | 0.005 / 0.004 | +2.8e-3 | 0.505 |
| 4 / 16 | 3→0 | 0.4260 / 0.4302 | same | 7e-9 | 0.001 | 0.002 | +1.1e-2 | 0.467 |

- `hwhm_eV` for 1→0 is 0.352 eV at $n_{\rm cells}=4,8,16$ (spread $5\times10^{-4}$ eV). This reproduces the claimed grid independence.
- The complex $G(z)$ (phase included, not only $\lvert G\rvert$) of the full dynamics equals the mode sum within 1.1% of $\lvert G_0\rvert$ for the flat 1→0 and the dispersive 3→2 and 3→0. The numerical one-sided Fourier transform equals the mode spectrum within 0.7% of the peak. So the sign/phase convention and the weights are right, and the dropped transfer into other band-index pairs is below 1% for bound lines.
- $\sum_m\mathrm{Re}\,w_m$ equals the yield to $10^{-6}$. Every scanned spectrum was positive. In the full assembly with capture (`cr-yield-assembly` re-validation) the total spectrum stays positive (minimum $3.4\times10^{-3}$ of the peak).
- For dispersive lines `hwhm_eV` is the initial-slope value, not the shape: the 3→2 mode spectrum has half width ≈ 0.25 eV against a reported 0.20 eV. The claim says this.

**Code reading.** The construction and gauge handling match the derivation. `modes` returns `source = yields/conj(a)` and `weight = (conj(a) @ V)(V^{-1} source)`, and `spectrum` uses `Re(conj(w) * profile)`. For each mode, the lab centre and half width are divided by $D$, and $\sigma=\lvert E_m\rvert s\,\sigma_\gamma/\gamma$, with $s$ from a strength-weighted exponent per group. No group mixes kinds on the 4- and 16-cell grids (`capture_width.py`), so `out[kind of first member]` does not misattribute any group there. Clustered capture rows enter the group through their strongest member's $a$ and pair. That is an approximation for capture shapes only (see `cr-capture-yield`).

**Verdict:** `rederived`. The earlier width discrepancy is resolved. A stated limit stays open: no measured linewidth anchor.


## Azadegan source audit (2026-10-15, independent context)

This source-first check concerns measured planar linewidth anchors for
`cr-line-shape`. The source equations and numerical expectations below were
recorded before inspecting the emission implementation. Sources: Azadegan,
Wagner & Pawelke, [Phys. Rev. B **74**, 045209 (2006)](https://doi.org/10.1103/PhysRevB.74.045209)
(Zotero parent `PQZAK23U`, PDF `KLWUGPNH`), and Azadegan,
Comput. Phys. Commun. **184**, 1064–1069 (2013) (parent `Z93R9VME`, PDF
`DY8WJ8VL`). Indexed full text was checked against the PDF images, including
2006 pp. 045209-3, -8, -9 and 2013 p. 1068.

### Source expectations before code comparison

For a coherence decaying as $\exp[-\kappa z+i\Delta z/(\hbar v)]$,
the one-sided Fourier transform gives a Lorentzian with transverse HWHM
$h=\hbar v\kappa$. With $D=1-\beta\cos\theta$, the photon centre and
FWHM are $E_0=\Delta/D$ and $\Gamma=2h/D$. Units are energy; positive
$\kappa$ gives a positive width. Zero damping gives a delta distribution;
zero angular spread gives one Lorentzian. For two states with population
attenuation lengths $l_i,l_f$, loss alone gives

$$
\kappa=\frac{1}{2}(l_i^{-1}+l_f^{-1}),\qquad
\Gamma_{\rm loss}=\frac{\hbar v}{D}(l_i^{-1}+l_f^{-1})
\simeq\frac{2\gamma^2\hbar v}{l},\qquad
l^{-1}=l_i^{-1}+l_f^{-1}.
$$

This agrees with 2006 Eqs. (6)–(8), up to the stated ultrarelativistic
$v\simeq c$ approximation. It does **not** imply that the optical-potential
loss width equals the linewidth of a trace-preserving scattering model:
correlated scattering can feed the radiating coherence. Nor does it imply
that the experimental FWHM is an intrinsic damping width. The 2006 paper
fits an asymmetric angular convolution, Eq. (23), with independent
$E_0$, Lorentzian parameter $\Gamma_T$, and
$\alpha=\gamma\Theta_{\rm ms,ch}$. It explicitly warns against adding all
partial FWHMs linearly or in quadrature (pp. -6, -9, -10).

**Experimental conditions (2006 Sec. II, pp. -1–2).** Diamond type IIa,
(110), room-temperature treatment (the potential figures use 300 K).
Measured thicknesses are $42.5\pm4.0$, $102.2\pm4.7$,
$168.0\pm3.9$, and $500\pm25$ micrometres. Initial beam divergence is
about 0.1 mrad, relative energy spread $1.3\times10^{-3}$ (the text does
not specify RMS versus FWHM), and energy tuning accuracy 0.2–0.3 MeV.
The detector is on-axis, 3.3 m from the target, behind a 1 mm aperture,
solid angle $7.2\times10^{-8}$ sr. CdTe calibration FWHM is
$470\pm2$ eV at 13.93 keV. At 42.5 micrometres Table III gives detector
FWHM 0.47/0.49/0.60/0.63 keV at 14.6/17/30/34 MeV.
The fitted effective in-plane multiple-scattering RMS angles are much
larger than the incoming divergence: Table II gives
6.03/5.32/2.87/2.60 mrad at those energies and 42.5 micrometres.
A spectrum using only the incoming 0.1 mrad cannot reproduce that response.

**Measured values, not theoretical targets.** The following are direct
transcriptions of Table V, p. -9, all for diamond (110), $1\to0$.
$\Gamma_{\rm exp}$ is the FWHM of the fitted asymmetric experimental
profile. $\Gamma_T$ is its Lorentzian fit parameter; the authors explicitly
state that it still incorporates Bloch dispersion and detector resolution
(p. -6). No errors are supplied for these widths.

| Beam energy (MeV) | Thickness (micrometres) | $\Gamma_{\rm exp}$ (keV) | $\Gamma_T$ (keV) |
| --- | --- | --- | --- |
| 14.6 | 42.5 | 1.51 | 1.09 |
| 14.6 | 168 | 1.80 | 1.07 |
| 14.6 | 500 | 2.20 | 1.31 |
| 17 | 42.5 | 2.00 | 1.45 |
| 17 | 102 | 2.19 | 1.45 |
| 17 | 168 | 2.47 | 1.44 |
| 17 | 500 | 2.77 | 1.26 |
| 30 | 42.5 | 5.88 | 4.25 |
| 30 | 102 | 7.94 | 5.75 |
| 30 | 168 | 6.12 | 3.94 |
| 30 | 500 | 11.98 | 7.99 |
| 34 | 42.5 | 7.84 | 6.00 |

For the same 42.5 micrometre cases, Table IV (p. -8) separates observed
peak maximum $E_{\rm exp}$, fitted zero-angle energy $E_0$, and the
Doyle–Turner calculation $E_{\rm calc}$. These are distinct observables:

| Beam energy (MeV) | $E_{\rm exp}$ (keV) | $E_0$ (keV) | $E_{\rm calc}$ (keV) |
| --- | --- | --- | --- |
| 14.6 | 16.58 | 16.76 | 17.06 |
| 17 | 21.72 | 21.96 | 22.23 |
| 30 | 56.19 | 56.87 | 59.49 |
| 34 | 70.02 | 70.96 | 73.78 |

**Internal source discrepancy: do not convert Table VI into a width anchor.**
At 30 MeV and 42.5 micrometres, Table V has $\Gamma_T=4.25$ keV;
Table III has detector and Bloch FWHMs 0.60 and 0.02 keV. Even using
the paper's approximate quadrature correction gives 4.207 keV. Eq. (6)
then gives $l=0.323$ micrometres with $\gamma=30/0.511$, or
0.334 micrometres with the kinetic-energy convention
$\gamma=1+30/0.511$. Table VI instead gives 0.65 micrometres.
At 17 MeV the analogous values are 0.323 or 0.342 micrometres,
against Table VI's 0.66. This is approximately a factor of two;
it cannot be explained by the small detector/Bloch correction or by
adding the rest energy. In contrast, Table III's *calculated* coherence
FWHMs and Table I's *calculated* lengths are consistent with Eq. (6):
1.51 keV with 0.90 micrometres at 30 MeV, using
$\gamma=30/0.511$. The paper does not expose its inversion procedure,
so the precise origin of the experimental-length discrepancy is unresolved.
Use the directly fitted widths with their response model; do not silently
reinterpret a Table VI length as an independent measured intrinsic width.

**Theoretical example (2013 Sec. 4, p. 1068).** Silicon (110), 40 MeV
*kinetic* energy, thickness 10 micrometres, incidence 0.3 mrad, and
Bloch momentum $k=1.3089$ inverse angstroms. The printed example gives
$E_0(1\to0)=76.2071$ keV, intrinsic FWHM 9.7758 keV, and yield
0.224692 photons/sr/electron. This is an optical-potential calculation,
not a measurement; Eq. (7)'s denominator establishes that the quoted
linewidth is FWHM. Its transverse FWHM target is
$9.7758\,{\rm keV}/(2\gamma^2)=0.7777$ eV with
$\gamma=1+40/0.511$, and the relative FWHM is 12.828%.
The code uses a Doyle–Turner real potential and complex optical potential,
21 Bloch basis functions, and thermal scattering. Its Sec. 4 statement
of agreement with the 2006 experiment does not turn this example into
an experimental linewidth anchor.


### Implementation comparison and scope of the verdict

After recording those expectations I read `emission_lines`,
`EmissionLines.spectrum`, and `channeling_spectrum`. The first constructs
all-block coherence generators with both loss and jump gain. The second
divides mode centres and HWHMs by $D$, exactly as derived above.
The third uses one supplied, depth-independent `divergence_y_rad` in a
Gaussian quadrature. It does not evolve the in-plane angular distribution
with depth, and does not convolve the output with a detector energy
response. Consequently the measured thickness dependence in 2006 Table V
is not predicted by that API from the quoted *initial* divergence alone.
Inserting each experimentally fitted effective multiple-scattering angle
would supply external response information; it would not validate a
prediction of that angle or its thickness dependence.

I ran bounded CPU points with the project runner
(`PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python`),
using `planar_profile(material, (1,1,0), temperature)`,
`tds_kernel(..., n_cells=8, max_order=16)`,
`DensityMatrixSolver(..., margin_eV=5)`, and
`emission_lines(..., include_capture=False)`. The reported values below
select block zero, upper state 1, lower state 0, with the lab FWHM computed
as `2 * hwhm_eV / (1 - beta)`. They are intrinsic selected-kernel values,
not a fit to the observed spectra.

| Material | Energy (MeV) | Temperature (K) | Kernel | Forward energy (keV) | Intrinsic FWHM (keV) |
| --- | --- | --- | --- | --- | --- |
| Diamond (110) | 14.6 | 300 | TDS | 17.9454 | 0.27107 |
| Diamond (110) | 17 | 300 | TDS | 23.1849 | 0.37377 |
| Diamond (110) | 30 | 300 | TDS | 60.5170 | 1.24914 |
| Diamond (110) | 34 | 300 | TDS | 74.7133 | 1.62936 |
| Silicon (110) | 40 | 293.15 | TDS | 76.6893 | 7.99263 |
| Silicon (110) | 40 | 293.15 | TDS + electronic | 76.6893 | 9.17971 |

For Si the 1→0 band is flat across all eight sampled blocks: centres vary
by less than $10^{-6}$ keV and widths by less than $10^{-6}$ keV.
Thus the source's particular $k=1.3089$ inverse angstroms is immaterial
at the precision of this comparison. Relative to the 2013 theoretical
example the model centre is 0.63% high, while the TDS and TDS + electronic
FWHMs are 18.2% and 6.1% low. Different potential/scattering approximations
prevent an exact reproduction claim. Adding electronic scattering improves
that one number but cannot make a theoretical example a measurement.
For diamond, `electronic_kernel` refuses the material because the packaged
shell data lack the required conduction-band entry. No complete collisional
prediction was computed; the TDS-only numbers must stay labelled as such.

The task owner's separately source-checked Gouanère et al., Phys. Rev. B
**38**, 4352 (1988), Table II, p. 4360, provides a cleaner measured
*intrinsic* benchmark: room-temperature diamond (110), 22 micrometres,
80.23 MeV. Column 9 gives intrinsic transverse FWHM after Doppler/multiple-
scattering deconvolution and corrections for finite thickness, detector
resolution, acceptance and energy binning. Its values are transcribed in
`tests/data/channeling/gouanere-1988-diamond110.json`. They are not the
observed photon FWHMs in column 5. The owner reports these CPU convergence
values for $2h$ (eV), `max_order=16`, at the zero Bloch block:

| Transition | Gouanère intrinsic transverse FWHM (eV) | TDS, 8 cells (eV) | TDS, 16 cells (eV) |
| --- | --- | --- | --- |
| 1→0 | 0.32 | 0.197485 | 0.197481 |
| 2→1 | 0.32 | 0.127327 | 0.127333 |
| 3→2 | 0.26 | 0.090963 | 0.090973 |
| 4→3 | 0.31 | 0.070890 | 0.070903 |

Those widths are 38–77% low with TDS alone. The grid refinement does not
explain the discrepancy. Missing electronic scattering is a concrete
limitation, but the present evidence does not establish that adding it
would close the full gap. Accurate peak energies therefore do not validate
linewidths or peak heights. No tolerance is widened to classify this
comparison as agreement.

- **Claim:** `cr-line-shape` — `montecarlo/channeling/emission.py::{emission_lines,EmissionLines.spectrum,channeling_spectrum}` — damped-coherence Fourier spectrum, Doppler width transform; Azadegan 2006 Eqs. (6)–(8) and linewidth Tables III–VI; Azadegan 2013 Eq. (7) and p. 1068.
- **Filters:** units pass; zero-damping/angular-spread limits pass; HWHM/FWHM and Doppler conventions pass. The experimental-length inversion in Azadegan 2006 Table VI has the unresolved approximately factor-two inconsistency documented above.
- **Re-derivation:** matches the selected-kernel Fourier/Doppler relation. Differs from measured-width reproduction: diamond TDS intrinsic widths are too small, electronic inputs are missing, and the assembled response omits depth-dependent in-plane multiple scattering and detector convolution.
- **Verdict:** `rederived` for the stated selected-kernel line-shape equation. Physical measured-width reproduction is incomplete and fails for the TDS-only Gouanère comparison; this evidence does not justify `anchored` for linewidths.
- **Write-up:** `docs/validation/channeling/cr-line-shape.md`.
- **Suggested ledger change:** retain `rederived`; replace the stale source-unavailable note with these transcribed measured values, source convention caveat, missing diamond electronic inputs, and failed TDS-only intrinsic-width comparison. The task owner applies any ledger edit; no human sign-off is claimed.

Reproduce the Azadegan model comparison from the repository root:

```bash
PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python - <<'PY'
from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo.channeling import (
    DensityMatrixSolver, electronic_kernel, emission_lines, tds_kernel,
)

cases = [("diamond", e, 300.0) for e in (14.6, 17, 30, 34)]
cases += [("silicon", 40, 293.15)]
for material, mev, temperature in cases:
    profile = planar_profile(material, (1, 1, 0), temperature)
    thermal = tds_kernel(profile, mev * 1e6, n_cells=8, max_order=16)
    kernels = [("TDS", thermal)]
    if material == "silicon":
        electronic = electronic_kernel(profile, mev * 1e6, n_cells=8, max_order=16)
        kernels += [("TDS + electronic", thermal + electronic)]
    for name, kernel in kernels:
        solver = DensityMatrixSolver(profile, mev * 1e6, kernel, margin_eV=5)
        lines = emission_lines(solver, include_capture=False)
        row = (lines.block == 0) & (lines.upper == 1) & (lines.lower == 0)
        centre_keV = lines.forward_energy_eV()[row][0] / 1000
        fwhm_keV = 2 * lines.hwhm_eV[row][0] / (1 - lines.beta) / 1000
        print(material, mev, name, round(centre_keV, 6), round(fwhm_keV, 6))
PY
```

**Additional 2006 energy-convention caveat.** Table III's calculated widths
numerically use $\gamma=E_e/(mc^2)$: at 17 MeV,
$2\gamma^2\hbar c/(1.06\,{\rm micrometres})=0.412$ keV, matching
its 0.41 keV, whereas $\gamma=1+17/0.511$ gives 0.437 keV. The 2006
experimental section calls its settings electron energies without the
explicit kinetic-energy definition supplied in the 2013 code paper.
This leaves a total-versus-kinetic convention ambiguity (or an omitted
rest-energy correction in the 2006 calculation); it must be resolved
before interpreting all of the nominal-energy peak offset as a potential
error. As a diagnostic only, `transverse_states` at
$\gamma=E_e/(mc^2)$, 300 K, 33 plane waves gives forward 1→0 energies
16.9098/22.0268/58.7938/72.8383 keV at nominal
14.6/17/30/34 MeV. These differ from Table IV's fitted $E_0$ by
0.9%/0.3%/3.4%/2.6%. No beam-energy retuning is assumed in the comparison
table above, and this diagnostic does not validate any linewidth.
