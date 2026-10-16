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
