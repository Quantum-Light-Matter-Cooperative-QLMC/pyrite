# `transverse-bunch-form-factor`

## Claim and where it enters

`mc_spectrum(coherent=True)` phase-sums segment fields against a single fixed far-field direction $\hat{\mathbf n}$ and squares once,

$$
E_j=\sqrt{\tfrac{\alpha\omega}{4\pi^2\hbar c}T_{{\rm abs},j}}\;A_jQ_j\,
\exp\bigl\{i\bigl[\omega t_{{\rm abs},j}
-(\omega\hat{\mathbf n}+\mathbf g)\cdot\mathbf r_j\bigr]\bigr\},
\qquad
\frac{d^2N}{dE\,d\Omega}\propto\Bigl|\sum_jE_j\Bigr|^2 .
$$

With a finite beam spot, `simulate_trajectories` gives each electron a constant transverse entry offset $\Delta\mathbf r_{\perp,e}$ drawn from a Gaussian of r.m.s. width $\sigma_\perp=$ `beam_fwhm_mm`$/(2\sqrt{2\ln2})$, and rigidly translates that electron's whole trajectory by it. The spatial phase reads $\mathbf r_j$ directly, so splitting the sum by electron,

$$
\Bigl|\sum_jE_j\Bigr|^2
=\sum_e\Bigl|\sum_{j\in e}E_j\Bigr|^2
+\sum_{e\neq e'}
\Bigl(\sum_{j\in e}E_j\Bigr)\Bigl(\sum_{j\in e'}E_j\Bigr)^{*}
e^{-i\mathbf q_\perp\cdot(\Delta\mathbf r_{\perp,e}
-\Delta\mathbf r_{\perp,e'})},
\qquad
\mathbf q_\perp=(\omega\hat{\mathbf n}+\mathbf g)_\perp .
$$

The first sum is the intra-electron term, untouched by any rigid per-electron translation. The second is what this row is about.

## The missing average

$\Delta\mathbf r_\perp$ is a *sampled* quantity: the physical observable is the expectation over the spot distribution, not the value at one draw. Taking that expectation for a Gaussian spot is the characteristic function already used for the longitudinal offset,

$$
\bigl|\langle e^{-i\mathbf q_\perp\cdot\Delta\mathbf r_\perp}\rangle\bigr|^2
=\exp\bigl[-(q_\perp\sigma_\perp)^2\bigr],
$$

the exact transverse partner of the $\exp[-(\omega\sigma_z)^2]$ bunch form factor that [Coherent-emission tracking](../../physics/radiation-physics/coherent-emission.md) derives for $t_0$. Since $q_\perp$ is of order $g\sim1\,\text{Å}^{-1}$, the exponent reaches $-10^{8}$ for a spot of only 1 µm: the cross-electron terms are annihilated for any physically realizable beam, leaving

$$
\frac{d^2N}{dE\,d\Omega}\propto\sum_e\Bigl|\sum_{j\in e}E_j\Bigr|^2 .
$$

**The implementation never takes this expectation.** It evaluates one realization of the sampled offsets at one $\hat{\mathbf n}$, so the stored spectrum is a single speckle draw whose cross-electron terms are a zero-mean random residue rather than the zero they average to.

Limiting cases both hold: $\sigma_\perp\to0$ recovers the point source exactly (all offsets vanish, all cross terms constructive), and $\sigma_\perp\to\infty$ gives the intra-electron sum — the same endpoint the implementation reaches only in expectation.

## Measurement

`checks/coherent_transverse_coherence.py`, reference HOPG case: 25 keV, (0,0,±2), $\theta_{\rm obs}=119°$, Timepix face (14 mm chip at 400 mm, $\Delta\theta=2.01°$), $N_e=300$, eight transport seeds.

**Realization scatter of the single-$\hat{\mathbf n}$ peak height.** The incoherent path from the same trajectories is the Monte Carlo counting-noise control.

| `beam_fwhm_mm` | coherent std/mean | coherent min/max | incoherent std/mean |
|---|---|---|---|
| `None` (point source) | 0.126 | 0.63 | 0.074 |
| 0.001 | **0.414** | 0.32 | 0.074 |
| 0.05 | **0.333** | 0.40 | 0.074 |
| 1.0 | **0.307** | 0.37 | 0.074 |

A 30–41% swing on peak height, a 2.5–3× min/max spread, against a 7.4% noise floor. This is not reducible by simulating more electrons: the contrast of a sum of randomly phased emitters is independent of how many there are.

**Seed-averaged enhancement** — the quantity the form factor would return deterministically. Ratio of mean coherent to mean incoherent peak height:

| `beam_fwhm_mm` | enhancement, $N_e=80$ | enhancement, $N_e=300$ |
|---|---|---|
| `None` | 33.2 | 107.5 |
| 0.001 | 3.74 | 3.23 |
| 0.05 | 2.80 | 2.83 |
| 1.0 | 3.19 | 3.25 |

The finite-spot rows are stable in both spot size and electron count — the intra-electron floor, $\approx3\times$. The point-source row is not: it grows linearly with $N_e$, which is the signature of unsuppressed $N^2$ superradiance divided by the per-electron normalization. `beam_fwhm_mm=None` is therefore a degenerate sampling artifact, not a conservative default.

**Angular integration does not recover the average.** L1 error of the tiled face integral against an `n_side=11` reference:

| | n=1 | n=3 | n=5 | n=7 | n=9 |
|---|---|---|---|---|---|
| incoherent (any spot) | 5.47% | 0.71% | 0.13% | 0.04% | **0.01%** |
| coherent, point source | 49.0% | 9.0% | 5.9% | 4.5% | **4.1%** |
| coherent, 50 µm spot | 81.2% | 28.1% | 18.5% | 12.8% | **11.3%** |

The incoherent path converges; the coherent path plateaus, because both the tested grid and the reference are under-sampled. The speckle angular scale is $\lambda/D_\perp\approx2\times10^{-4}$ rad against $\Delta\theta=0.035$ rad — about 180 fringes across the face — so convergence needs $n_{\rm side}\approx180$, roughly 32000 directions per case. Tiling is not the fix; the analytic average is.

## Relation to the far-field curvature term

A separate and much smaller approximation rides on the same fixed $\hat{\mathbf n}$. The far-field expansion keeps the linear retardation $-\omega\hat{\mathbf n}\cdot\mathbf r$ **exactly** and drops only the curvature term $\omega r_\perp^2/2R$; separation along $\hat{\mathbf n}$ carries no error, so slab depth is exact. For this case's 6.5 µm emitting volume:

| working distance | one-radian patch $\sqrt{\lambda R/\pi}$ | neglected phase |
|---|---|---|
| 10 mm | 2.01 µm | 2.62 rad |
| 30 mm (`detector_directions` default) | 3.49 µm | 0.87 rad |
| 100 mm | 6.37 µm | 0.26 rad |
| 400 mm (Timepix) | 12.74 µm | 0.066 rad |

Negligible at the Timepix operating point, not negligible at the library default. This bounds a phase error and so is a coherent-path concern only; the induced observation-angle error $\delta\theta\approx r_\perp/R$ shifts the incoherent line by $\ll0.1$ eV, two orders below the detector-acceptance broadening it sits inside. Recorded here so the two are not conflated: fixing the form factor does not fix the curvature term, and the curvature term is not what drives the numbers above.

## Assumptions and limits

- Measured on one geometry (HOPG basal-plane, 25 keV, $\theta_{\rm obs}=119°$, Timepix acceptance). The form-factor argument itself is geometry-independent — it needs only $q_\perp\sigma_\perp\gg1$ — but the quoted enhancement of $\approx3\times$ is specific to this case's segment statistics.
- The Gaussian result assumes the spot is Gaussian and independent of the longitudinal offset. `transverse_distribution` (Courant–Snyder) and the elliptical `beam_fwhm_y_mm` path need the same average with the appropriate covariance; a correlated transverse–longitudinal phase space would not factorize into $\exp[-(\omega\sigma_z)^2]\exp[-(q_\perp\sigma_\perp)^2]$.
- The finite-footprint branch (`crystal_width_mm` / `crystal_height_mm`) makes the transverse offset affect escape attenuation as well as phase, so the offset is no longer a pure rigid translation and the cross-electron amplitudes are no longer identical. The form factor still applies to the phase; the amplitude variation is a separate effect not measured here.
- Seed scatter is estimated from 8 seeds — enough to establish the effect is many times the noise floor, not enough for a precise contrast.

## Status

`rederived`, as of 2026-08-21. The cross-electron terms are now dropped analytically via the general $\exp[-(q_\perp\sigma_\perp)^2]$ factor for the marginal regime, combined with the longitudinal `coherent-emission` form factor into one joint $F$ — see [`coherent-inter-electron-decoherence`](coherent-inter-electron-decoherence.md) for the combined closed form, its fresh-context re-derivation, and the implementation (an empirical characteristic function of the actual sampled offsets, rather than a closed-form-parametrized $\sigma_\perp$, which extends this row's isotropic-Gaussian scope to elliptical/Courant–Snyder spots for free). The finite-footprint amplitude-coupling case noted above remains excluded, now with an explicit `mc_spectrum` error rather than silent mishandling. Human sign-off remains pending.
