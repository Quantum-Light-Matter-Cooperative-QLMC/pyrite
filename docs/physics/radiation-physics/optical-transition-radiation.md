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

Limits pinned in `tests/montecarlo/spectrum/test_transition_radiation_optical.py`: a
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
region, with the dip at 485 nm (`tests/montecarlo/spectrum/test_transition_radiation_optical_anchor.py`).
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

**Limits** (`tests/montecarlo/spectrum/test_transition_radiation_optical.py`): $N=1$
keeps the self term only; $F\to0$ gives $I_1$ and $F=1$ (in-phase histories)
$N I_1$ per electron; the yield at fixed charge is the same for $M=2,5,17$;
sampled Gaussian offsets reproduce the analytic law within the pair-estimate
error; importance-weighted samples from a wider law reproduce the target law;
a resolved microbunched profile has its harmonic at $E=2\pi\hbar c/\lambda_m$.

## RMS duration convention and coherent window

Catalog `envelope_rms_fs` values are RMS (standard
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
  allowed to change by $2|\mathbf A_s|$, so each history's field changes by
  at most $b_e=\sum_c 2\sum_s d_s|\mathbf A_s|$. Propagated through the
  estimators (weights $w_e$, $D=\sum_e w_e b_e$):
  $|\delta\sum w|\mathbf S|^2|\le\sum_e w_e(2b_e|\mathbf S_e|+b_e^2)$ for the
  self term and
  $2D|\sum_e w_e\mathbf S_e|+D^2+\sum_e w_e^2(2b_e|\mathbf S_e|+b_e^2)$ for the
  distinct-pair numerator. The relative bound of the total predicted yield
  (self plus $(N-1)|F|^2$ pair; `coherent_formation_error_bound`) must not
  exceed `formation_tolerance` (default 0.05); the self-only bound is
  recorded as `formation_self_bound`. This is a sensitivity measure, not a
  model of radiation from the deflection (bremsstrahlung), and the premise
  that a deflected term is no larger than the straight one is heuristic. It
  is conservative: condensed-history direction changes end runs.
  In a consistent gold scene (catalog `au` transport tables with
  Johnson–Christy gold optics, 30 keV), the bound is about 1.5. ELSEPA
  scattering deflects the electrons within the optical skin depth, so the
  default tolerance refuses fixed trajectories there; the scene runs only
  with the check disabled. García de Abajo (RMP 82, 209, footnote 17) places
  straight passage over a skin depth only above about 50 keV.
- **Phase formation policy (opt-in).** `formation_kink_phase` (rad) replaces
  "any direction change ends a run" with a phase test. Crossing records also
  carry the direction changes along the path on each side
  (`kinks_before_ang`, `kinks_after_ang`: distance and $|\Delta\hat{\mathbf v}|$,
  16 slots, the last marking where the usable path ends). For a
  piecewise-straight path,
  $|\mathbf r(s)-\mathbf r_0(s)|\le D(L)=\sum_{d_i<L}(L-d_i)|\Delta\hat{\mathbf v}_i|$
  for all $s\le L$, which is exact geometry. A side is short only if the path
  ends inside $L$ or $\|\mathbf K\|D(L)$ exceeds the tolerance, with
  $\mathbf K=(\mathbf k_\parallel,\lambda)$ the side medium's wave vector.
  Treating a smaller phase as straight is the heuristic step; a tolerance of
  0 reproduces the default exactly. With 0.1 rad, the HOPG-transport/Au-optics
  test scene goes from 0.171 to 0: its condensed-history kinks are below the
  phase tolerance within $L$. The gold scene drops from 1.47 to 0.57 and is
  still refused, consistent with real deflection inside the skin depth. The
  value is recorded as `formation_kink_phase` in the validity metadata.
- **Energy budget.** The radiated energy per physical electron, including the
  pair term, integrated over both hemispheres (Gauss–Legendre in
  $\cos\theta$ × uniform $\varphi$) and the band, plus the out-of-band
  coherent envelope below, must stay below `energy_budget` (default
  $10^{-3}$) of the entrance kinetic energy, and the pair-to-self ratio at
  both band edges must be below $10^{-3}$.
- **Out-of-band coherent energy.** Outside the band no optics are supplied, so
  the pair yield is bounded instead of scored: $|\langle\mathbf S\rangle|^2\le
  \langle|\mathbf S|^2\rangle$, $|\mathbf S_e|^2\le n_e\sum_c|\mathbf A_c|^2$
  ($n_e$ crossings), and each crossing by its perfect-conductor image-charge
  energy,

  $$
  \frac{dU_c}{dE\,d\Omega}=\frac{\alpha}{4\pi^2}
  \left|\mathbf n\times\Bigl(\frac{\boldsymbol\beta}{1-\mathbf n\cdot\boldsymbol\beta}
  -\frac{\boldsymbol\beta'}{1-\mathbf n\cdot\boldsymbol\beta'}\Bigr)\right|^2,
  $$

  with $\boldsymbol\beta'$ mirrored in the face (normal incidence: the
  Ginzburg–Frank conductor form), integrated over the vacuum hemisphere (the
  larger of both for an inner face). This energy per eV is flat in $E$, and a
  Gaussian law without time–transverse correlation has
  $|F|^2\le e^{-(E\sigma_t/\hbar)^2}$, so the tails integrate to
  $(N-1)B\,(\hbar/\sigma_t)(\sqrt\pi/2)[\operatorname{erf}x_{\rm lo}+\operatorname{erfc}x_{\rm hi}]$,
  $x=E\sigma_t/\hbar$, $B$ the weighted sample mean of $n_e\sum_c U'_c$ (a
  consistent estimate, not a strict bound). The conductor envelope fails
  where a traversed medium can emit Cherenkov light into vacuum
  ($\beta^2\operatorname{Re}\varepsilon\ge1$, exceeded by up to orders of
  magnitude) and for grazing crossings on low-$|\varepsilon|$ media near
  their plasma edge (up to about 1.6×). Gold below its plasma edge stays at
  0.86–0.99 of it. A layer whose in-band permittivity reaches
  $\beta^2\operatorname{Re}\varepsilon\ge1$ (a proxy for the unscored
  out-of-band values), $\gamma>5$ (where the fixed quadrature is unresolved),
  and sampled, empirical, realization and tilted Gaussian laws are recorded
  as `out_of_band_unbounded`, not refused. The larger-hemisphere rule for
  inner faces is a heuristic. For 30 keV electrons on gold at 200 fs and
  1 pC the envelope is about $1.8\times10^{-4}$ of the kinetic energy below a
  1.4–3.3 eV band.

Both checks bound the self (single-history) power; the pair term's relative
error is not bounded by the formation check, and the factor 2 is a heuristic
sensitivity measure, not a proven bound.

Both are recorded in the result's `metadata["validity"]`; passing `None`
skips a check and records it as not evaluated. Space charge, wakefields and
collective bunch evolution are outside this passive-source model.

## Many-layer stacks

`Validation: optical-stack-fast-solver`

From `FAST_STACK_MIN_LAYERS` layers up, `planar_transition_radiation` and
`interface_response` use an O(N) solve of the same boundary problem (#962). In a
flat isotropic stack $\mathbf k_\parallel$ is conserved and TE and TM decouple.
With up- and down-going amplitudes $a,b$, the tangential fields are $E=a+b$ and
$H=Y(a-b)$, where $Y_{\rm TE}=-\lambda/\omega$ and
$Y_{\rm TM}=\varepsilon\omega/\lambda$. Eliminating the down-going wave below
interface $i$ from the continuity equations with jump $(s_E,s_H)$ gives

$$
D_i=(1+\rho_i)Y_{i+1}+(1-\rho_i)Y_i,\qquad
u_i=\frac{2Y_iA_i+(1-\rho_i)Y_is_{E,i}+(1+\rho_i)s_{H,i}}{D_i},
$$

$$
\rho_{i+1}=p_{i+1}^2\,\frac{(1+\rho_i)Y_{i+1}-(1-\rho_i)Y_i}{D_i},\qquad
A_{i+1}=p_{i+1}u_i,\qquad p_j=e^{i\lambda_jd_j},
$$

with $\rho_0=A_0=0$. The vacuum amplitude is $u$ of the last interface. Because
$|p|\le1$, thick absorbing or evanescent layers cannot overflow, unlike a
transfer-matrix product. Tests compare against the dense solve and pin the
matched-medium, vanishing-film, closed single-interface and opaque-exit limits,
Bloch stop-band decay against the period's transfer trace, and the weak-contrast
resonance $(q-k_z)\Lambda=2\pi m$.

The recursion has a removable pole at $D_i=0$, a bound mode of the lower
half-stack, where the dense matrix is regular. It also rounds $1+\rho$ where
$\rho\to-1$. That rounding reaches the vacuum through $C_i/D_i$, so it is estimated
relative to the pair's terms, as $u\sum_i|C_i/D_i|/\max_i|C_i\beta_i|$ for the
response with unit roundoff $u$, and likewise beside the amplitude. Non-finite
results and estimates above $10^{-8}$ raise `LinAlgError` for the whole call, so
one such pair refuses a batch. A lossless scan across a critical angle can do this;
a small absorption avoids it.

The estimate covers only that rounding. Conditioning near lossless resonances and
phase rounding $u\sum|\lambda d|$ are not estimated and can be larger, up to full
scale at a resonance, where the dense solve is also unreliable. Deep responses can
lose pointwise accuracy in lossless stacks; absorbing stacks keep it. The dense
solve has a limit of its own: on oblique falling-permittivity chains it lost up to
all accuracy in the amplitude where this solve did not, against the dense system
evaluated at 100–800 digits.

These are interference and Bragg effects along $z$. In-plane momentum is
conserved, so this is resonant TR and not diffracted TR; laterally patterned
layers are a separate model (#963).

## Backend

The history sums run on the active backend: CUDA CuPy uses the fused TR kernel
(charges carry the importance weights), other backends the chunked array path;
the optical permittivity is interpolated on the host and uploaded once as
complex128. Validity preflights run on the host.
