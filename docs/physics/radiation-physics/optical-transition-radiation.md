# Optical planar TR and CTR

Opt-in, source-only transition radiation (TR) and coherent TR (CTR) in a
supported optical band (#615). It reuses the exact planar-stack amplitude and
crossing superposition of [transition radiation](transition-radiation.md) with
an explicit measured permittivity per layer (`OpticalResponse`), and adds a
physical-bunch/sampling contract. It is not a finite-radiator, near-field,
optics or detector model (#616, #617), and it changes no X-ray default: the
production component dispatcher never calls it and the X-ray continuum floor
is untouched.

Entry points: `montecarlo.runner.transition_emission.case_optical_transition`
(transported case) and `montecarlo.spectrum.optical_transition.optical_transition_radiation`
(crossing records). Results are `OpticalTransitionResult`, saved with `save()`
as one `.npz` (arrays plus JSON metadata, no pickles) and read back with
`load_optical_transition`.

## Single-electron source

`Validation: optical-planar-tr`

Each crossing $c$ of a history contributes the vacuum amplitude of the planar
Maxwell boundary problem (Ginzburg–Frank; [exact planar-stack TR](transition-radiation.md#exact-planar-stack-tr)),

$$
\mathbf a_c=G_{i}\,\bigl[-(\mathbf P_{i+1}(\mathbf v_c)-\mathbf P_i(\mathbf v_c))\bigr]\,
e^{i(\omega t_c-\mathbf k_\parallel\cdot\mathbf r_{\parallel,c})/\hbar c},\qquad
\mathbf k_\parallel=\omega\,\hat{\mathbf n}_\parallel ,
$$

with the layer permittivities $\varepsilon_j(E)$ taken from explicit optical
providers (no atomic-table fallback, no extrapolation). Assumptions: classical
point charge, locally straight uniform passage at each crossing, infinite flat
sharp interfaces, local isotropic nonmagnetic passive media
($\operatorname{Im}\varepsilon\ge0$, $e^{-i\omega t}$). Units:
$d^2N/dE\,d\Omega=E\,n_z^2|\sum_c\mathbf a_c|^2/(16\pi^4)$ photons/eV/sr per
electron, summed over polarization.

Limits pinned in `tests/montecarlo/test_transition_radiation_optical.py`: a
matched medium ($\varepsilon=1$) and a vanishing film radiate nothing; a
conductor $|\varepsilon|\to\infty$ at normal incidence gives the backward
Ginzburg–Frank form
$\alpha\beta^2\sin^2\theta/[\pi^2E(1-\beta^2\cos^2\theta)^2]$. A charge entering
bulk gold reproduces the closed finite-$\varepsilon$ Ginzburg–Frank form
(`ginzburg_frank_interface`, $\beta\to-\beta$) at 1.4, 2.5 and 3.3 eV to
$10^{-9}$.

**Published comparison.** Coenen, Vesseur and Polman {cite:p}`coenen2011`
(arXiv:1107.3632v1, withdrawn from arXiv; APL 99, 143103) computed the
hemisphere-integrated TR probability of 30 keV electrons on gold with
Johnson–Christy constants (their Fig. 3). With the pinned Johnson–Christy
record (`gold_johnson_christy_response`), the normalized spectral shape agrees
within 0.006 over 600–900 nm and within 0.04 over the 375–550 nm interband
region, with the dip at 485 nm (`tests/montecarlo/test_transition_radiation_optical_anchor.py`).
The published absolute curve is a constant $0.696\pm0.003$ of this hemisphere
integral over 600–900 nm. That factor is unresolved: it is not a kinematic or
optics difference that changes shape; it may be a collection or
normalization convention of the published figure. Absolute normalization is
therefore anchored by the conductor limit, not by this comparison. The
measured angular patterns (Fig. 4c; text: lobe peaks "around" 60° at 400 nm
and 70° at 900 nm) are not reproduced: per-steradian peaks are 54.9° and
65.8°, $\sin\theta$-weighted peaks 59.6° and 68.7°, and the preprint does not
say which was plotted, nor give the mirror polarization/collection response.

## Bunch population and sampling law

`Validation: optical-ctr-population`

For $N$ independent electrons whose single-electron history fields $\mathbf S$
are identically distributed, the expected bunch power per physical electron is
the squared field-sum expansion (Shkvarunets and Fiorito, arXiv:0712.2848, Eq. 1),

$$
I=\mathrm E|\mathbf S|^2+(N-1)\,|F|^2\,|\mathrm E\mathbf S|^2 ,
$$

with $|F|^2=1$ when $\mathbf S$ already contains each electron's own offset
phase, and $|F|^2=|\langle e^{i\boldsymbol\kappa\cdot\boldsymbol\xi}\rangle|^2$
for offset-free histories translated by $\boldsymbol\xi=(\delta t,\delta x,\delta y)$
at the entrance face, $\boldsymbol\kappa=(\omega,-\omega n_x,-\omega n_y)/\hbar c$
(slab frame). Translation invariance of an infinite planar stack makes the
offset a pure phase common to all crossings of a history and independent of its
scattering, so the factorization needs no further assumption there. $N=Q/e$
comes only from the bunch charge.

**Estimator.** $M$ sampled histories with positive importance weights $w_e$
(unit by default) give

$$
\widehat{\mathrm E|\mathbf S|^2}=\frac{\sum_e w_e|\mathbf S_e|^2}{W},\qquad
\widehat{|\mathrm E\mathbf S|^2}=\frac{|\sum_e w_e\mathbf S_e|^2-\sum_e w_e^2|\mathbf S_e|^2}{W^2-W_2},
$$

$W=\sum w_e$, $W_2=\sum w_e^2$. Both are self-normalized importance-sampling
ratios: consistent for $w\propto p/q$ (bias $O(1/M_\text{eff})$; exactly unbiased only for unit weights), exactly the distinct-pair moments
$G/M$ and $(P-G)/[M(M-1)]$ for unit weights, invariant to a common weight
scale. A weight never multiplies the self term squared, so a macro-particle
weight cannot masquerade as physical self noise. The three moments come from
the existing history sums with crossing charges scaled by $w_e$ and
$\sqrt{w_e}$ (CPU chunked path and the CUDA fused kernel alike). The estimate
is signed and unclipped; negative bins warn. The noise of the pair term relative
to the self term scales with $(N-1)/(M_\text{eff}-1)$,
$M_\text{eff}=W^2/W_2$, recorded as `pair_weight`.

| Law | Histories | $\lvert F\rvert^2$ | Refused |
| --- | --- | --- | --- |
| `sampled` | carry their own sampled offsets and all intra-electron correlations (time–energy chirp, position–direction) | 1 | an analytic factor (would count offsets twice) |
| `gaussian` | offset-free | $\exp(-\boldsymbol\kappa^{\mathsf T}\Sigma\boldsymbol\kappa)$, $\Sigma$ the $(t,x,y)$ covariance at the entrance face | histories with entrance offsets |
| `empirical` | offset-free (resolution = sampling interval, not RMS blur) | transform of a stated profile: `profile` density $|\sum w e^{i\boldsymbol\kappa\cdot\boldsymbol\xi}|^2/W^2$; iid `samples` $(|\sum w e^{i\phi}|^2-W_2)/(W^2-W_2)$ | $\omega\,\delta t>1$ or $k_\parallel\,\delta r>1$ for the stated resolution; missing resolution |
| `realization` | are the physical electrons ($N=M$, unit weights) | — ($|\sum\mathbf S|^2/M$) | weights; $N\ne M$ |

`inter_electron="correlated"` (sub- or super-Poissonian shot noise) is refused
except as an explicit realization. Microbunched sources enter as `empirical`
profiles with their modulation resolved, or as `sampled` histories drawn from
the modulated density; a smooth Gaussian is not evidence of microbunching.

**Limits** (`tests/montecarlo/test_transition_radiation_optical.py`): $N=1$
keeps the self term only; $F\to0$ gives $I_1$ and $F=1$ (in-phase histories)
$N I_1$ per electron; the yield at fixed charge is the same for $M=2,5,17$;
sampled Gaussian offsets reproduce the analytic law within the pair-estimate
error; importance-weighted samples from a wider law reproduce the target law;
a resolved microbunched profile has its harmonic at $E=2\pi\hbar c/\lambda_m$.

## RMS duration convention and coherent window

Catalog `envelope_rms_fs` and `BunchLength.rms_fs` are RMS (standard
deviation) durations, not FWHM; at normal incidence they are the crossing-time
RMS $\sigma_t$ of the `gaussian` law. Then
$|F_t|^2=e^{-(E\sigma_t/\hbar)^2}$, $E_{1/e}=\hbar/\sigma_t$, and the pair term
exceeds a fraction $\epsilon$ of the self term only below

$$
E_w=\frac{\hbar}{\sigma_t}\sqrt{\ln\frac{N-1}{\epsilon}}
\qquad(\texttt{gaussian\_coherent\_band\_eV}).
$$

For `gaussian_200fs` at 1 pC ($N=6.24\times10^6$): $E_{1/e}=3.29$ meV and
$E_w=15.6$ meV at $\epsilon=10^{-3}$. This is a time-only, normal-incidence
window; a tilted front ($t$–$x$ covariance) observed obliquely can make
$\boldsymbol\kappa^{\mathsf T}\Sigma\boldsymbol\kappa<\omega^2\sigma_t^2$ and widen it. The bunch turnover therefore needs
optical support well below 0.01 eV (far-IR/THz data, at least down to a few
meV), which no shipped provider has; across the gold 0.2–5 eV band the CTR
term of that bunch is $e^{-3700}$ or smaller. The X-ray coherent-window
predicates are lattice phases and are not reused here.

## Validity bounds of fixed trajectories

`Validation: optical-planar-tr, optical-ctr-population`

`case_optical_transition` refuses unbounded use (`optical_transition_validity`):

- **Formation.** A medium's own field forms over
  $L=\max(1/|q-\lambda|,1/|q+\lambda|)\,\hbar c$, $q=(\omega-\mathbf k_\parallel\cdot\mathbf v_\parallel)/v_z$,
  $\lambda=\sqrt{\varepsilon\omega^2-k_\parallel^2}$. Crossing records carry
  the transported straight run on each side (`straight_before_ang`,
  `straight_after_ang`; any direction change ends a run). When a run is
  shorter than $L$, that side's linear share of the amplitude $\mathbf A_s$ is
  allowed to change by $2|\mathbf A_s|$; the resulting bound on the self power,
  $\sum_e(2|\delta\mathbf S_e||\mathbf S_e|+|\delta\mathbf S_e|^2)/\sum_e|\mathbf S_e|^2$,
  must not exceed `formation_tolerance` (default 0.05). This is a sensitivity
  bound, not a model of radiation from the deflection (bremsstrahlung). It is
  conservative: condensed-history direction changes end runs.
- **Energy budget.** The radiated energy per physical electron, including the
  pair term, integrated over both hemispheres (Gauss–Legendre in
  $\cos\theta$ × uniform $\varphi$) and the band, must stay below
  `energy_budget` (default $10^{-3}$) of the entrance kinetic energy, and the
  pair-to-self ratio at both band edges must be below $10^{-3}$; otherwise the
  coherent energy just outside the band is unbounded and the case is refused.
  The budget is in-band only: it cannot detect coherent energy below a band
  whose lower edge lies above the coherent window $E_w$, where $|F|^2$ rises
  back toward one (for 30 keV electrons on gold, a 200 fs bunch radiates
  roughly $2	imes10^{-4}$ of the kinetic energy there at 1 pC and
  $2	imes10^{-3}$ at 10 pC). Check $E_w$ against the band before relying on
  fixed trajectories for a high-charge short bunch.

Both checks bound the self (single-history) power; the pair term's relative
error is not bounded by the formation check, and the factor 2 is a heuristic
sensitivity measure, not a proven bound.

Both are recorded in the result's `metadata["validity"]`; passing `None`
skips a check and records it as not evaluated. Space charge, wakefields and
collective bunch evolution are outside this passive-source model.

## Backend

The history sums run on the active backend: CUDA CuPy uses the fused TR kernel
(charges carry the importance weights), other backends the chunked array path;
the optical permittivity is interpolated on the host and uploaded once as
complex128. Validity preflights run on the host.
