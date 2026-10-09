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

## Tilted face and face-arrival delay (#370)

### Geometry

The lab beam travels along $\mathbf b$ (sample frame). Electron $e$ has lab spot
offset $\mathbf w=(u,v)$, i.e. sample-frame vector
$\mathbf o=u\,\mathbf e_x+v\,\mathbf e_y\perp\mathbf b$, and bunch time $t_0$ at the lab
plane through the sample origin perpendicular to $\mathbf b$. It meets the entrance
face $z=0$ at

$$
\mathbf p_0=\mathbf o+s^*\mathbf b,\qquad s^*=-\frac{o_z}{b_z}=\mathbf p_0\cdot\mathbf b ,
$$

the last form because $\mathbf o\cdot\mathbf b=0$. It arrives $s^*/\beta$ later
(Å, $c=1$). Transport previously launched every electron at the face at $t_0$,
a pulse front parallel to the face; it now adds $s^*/\beta$ to the analytic-spot
$t_0$ (`transport/beam_entry.py::face_arrival_delay_ang`), as GDF beams already did.
For a 0.1 mm spot at 45° the omitted spread was about 300 fs.

### Phase and form factor

On a translation-invariant target, electron $e$'s field is the offset-free field
translated by $(\mathbf p_0,\;t_0+s^*/\beta)$. With the reducer's phase
$\omega d-\mathbf g\cdot\mathbf r=\omega t-\mathbf Q\cdot\mathbf r$,
$\mathbf Q=\omega\hat{\mathbf n}+\mathbf g$,

$$
\varphi_e=\omega t_0-\mathbf Q\cdot\mathbf p_0+\frac{\omega}{\beta}s^*
=\omega t_0-\mathbf K\cdot\mathbf p_{0,xy},\qquad
\mathbf K=\Bigl(\mathbf Q-\frac{\omega}{\beta}\mathbf b\Bigr)_{xy}
=\omega\Bigl(\hat{\mathbf n}-\frac{\mathbf b}{\beta}\Bigr)_{xy}+\mathbf g_{xy},
$$

using $p_{0,z}=0$ and $s^*=\mathbf p_{0,xy}\cdot\mathbf b_{xy}$. The projection is
linear, $\mathbf p_{0,xy}=J\mathbf w$ with columns
$J_{:,i}=(\mathbf e_i-(e_{i,z}/b_z)\mathbf b)_{xy}$, so
$\varphi_e=\omega t_0+\boldsymbol\kappa\cdot\mathbf w$ with
$\boldsymbol\kappa=-J^{\mathsf T}\mathbf K$, matching the issue's
$\boldsymbol\kappa=-\mathbf Q_{\perp,\rm lab}+(\omega/\beta-\mathbf Q\cdot\mathbf b)\nabla s^*$.
For a zero-mean Gaussian spot of lab covariance $\Sigma$ independent of $t_0$,

$$
F=F_z F_\perp,\qquad
F_\perp=\bigl|\langle e^{i\boldsymbol\kappa\cdot\mathbf w}\rangle\bigr|^2
=\exp(-\boldsymbol\kappa^{\mathsf T}\Sigma\boldsymbol\kappa)
=\exp(-\mathbf K^{\mathsf T}\Sigma_f\mathbf K),\qquad \Sigma_f=J\Sigma J^{\mathsf T}.
$$

Limits: zero tilt ($J=I$, $\mathbf b_{xy}=0$) gives
$\exp(-\mathbf Q_\perp^{\mathsf T}\Sigma\mathbf Q_\perp)$, the isotropic
$\exp[-(q_\perp\sigma_\perp)^2]$ above; $\Sigma\to0$ gives one; $\beta\to1$ gives
$\mathbf K=(\mathbf Q-\omega\mathbf b)_{xy}$; $\mathbf K=0$ (phase matched along the
face) is fully coherent for any spot although $\mathbf Q_\perp\neq0$. Tilt enters
only through $\nabla s^*$ and $J$.

### Implementation and scope

Finite-footprint rows pair offset-free fields (face point and full arrival
removed; amplitudes and escape keep sampled positions) with
$F=F_zF_\perp$ per (reflection, orientation) row and in the temporal
characteristic function $\chi=\exp(-\tfrac12\omega^2\sigma_t^2)\exp(-\tfrac12\mathbf K^{\mathsf T}\Sigma_f\mathbf K)$.
Infinite slabs are unchanged in code: complete fields (physical population) or the
empirical characteristic function (historical) now see the delay through $t_0$.
Transport records the spot as plain data (`beam_entry`): lab covariance
($\sigma^2$ from FWHM, or Twiss $\epsilon\beta$), tilt, $E_0$, energy spread and the
delay convention.

Where $F_z>0$ somewhere on the axis and transverse offsets exist, the reducer
refuses: no recorded spot (old artifacts, GDF, groove phase); a Twiss beam with
$\alpha\neq0$ (position–slope correlation makes $S_e$ depend on $\mathbf w$); an
energy spread on a tilted face ($s^*/\beta_e$ couples $\beta$ to $\mathbf w$); any
missed entry; or a face spot whose $6\sigma$ extent plus the offset-free reach of
emission points and their slab exits along $\hat{\mathbf n}$ exceeds a footprint
half-width (Gaussian tail beyond $6\sigma$: $2\times10^{-9}$ per axis). Where
$F_z=0$ on the whole axis only phase-free self terms remain and none of this is
needed.

Window rules need a nonincreasing upper bound. $q(\omega)=\mathbf K^{\mathsf T}\Sigma_f\mathbf K$
is a convex quadratic with minimiser $\omega^*$, so
$F_z(E)\sup_{E'\ge E}F_\perp(E')$, equal to $F_\perp$ above $\omega^*$ and
$e^{-q_{\min}}$ below, is one; each captured row carries it. The full-axis audit
keeps the $F_z$ upper bound and lowers its factor bound to zero. Flat-term
omission keeps its conservative $F_z$ certificate.

On the tilted HOPG (002) geometries ($\mathbf g_{xy}=0$) the delay multiplies the
exponent by 12.0 (45°/135°, 60 keV) and 1.67 (5°/0°, 30 keV). At the
`hopg_short` 0.1 mm spot the exponent is $2.2\times10^{9}$ at 100 eV, so
$F_\perp$ removes the physical-charge cross term across the 10–6000 eV axis.

### Remote smoke (2026-10-09)

The reduced `hopg_short`-class case that #350 refused (SLURM 1150) now
completes on CUDA float64 (SLURM 1169, revision `73ccbfe0`): 5,482,423
coordinates, spectrum 14,367 s, coherent/incoherent yield ratio 1.00012, no
`CoherentSamplingError`. Details in
[`coherent-physical-bunch-population`](coherent-physical-bunch-population.md).

### Checks

`tests/montecarlo/test_coherent_transverse_tilt.py`: the closed form equals an
independent ray-plane construction of $\varphi(\mathbf w)$ to $10^{-10}$ and a
400,000-draw brute-force average of $e^{i\varphi}$ within $5\times10^{-3}$, for
isotropic and elliptical spots at 45°/135°; a phase-matched row has
$\boldsymbol\kappa=0$ and $F_\perp=1$ while $|\mathbf Q_\perp|\sigma>10^3$; zero tilt
and point-spot limits; the delay equals $s^*/\beta$ and vanishes untilted; tilted
identical straight tracks on a finite footprint reproduce
$|S|^2[1+(N-1)F_zF_\perp]$ through `mc_spectrum`; the window envelope is
nonincreasing and bounds $F_\perp$; unsupported beams and edge-reaching spots are
refused.

## Status

`rederived`, as of 2026-08-21. The cross-electron terms are now dropped analytically via the general $\exp[-(q_\perp\sigma_\perp)^2]$ factor for the marginal regime, combined with the longitudinal `coherent-emission` form factor into one joint $F$ — see [`coherent-inter-electron-decoherence`](coherent-inter-electron-decoherence.md) for the combined closed form, its fresh-context re-derivation, and the implementation (an empirical characteristic function of the actual sampled offsets, rather than a closed-form-parametrized $\sigma_\perp$, which extends this row's isotropic-Gaussian scope to elliptical/Courant–Snyder spots for free). The finite-footprint amplitude-coupling case noted above remains excluded, now with an explicit `mc_spectrum` error rather than silent mishandling. Human sign-off remains pending.

**2026-10-08 (#370):** amended for tilted faces, the face-arrival delay and finite footprints (section above). A fresh-context verifier independently re-derived the amended claim (match, no divergent factor, sign or unit; 400k-draw Monte Carlo over random tilts within 2.2e-3); status `anchored` on `tests/montecarlo/test_coherent_transverse_tilt.py`. Human sign-off pending (#277).
