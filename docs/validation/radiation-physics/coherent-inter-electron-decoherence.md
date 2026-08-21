# Validation: `coherent-inter-electron-decoherence`

## Claim and where it enters

`mc_spectrum(coherent=True)` sums a complex field per (reflection, mosaic
orientation) row, $E_j$ per segment $j$ of every electron in the row, and
squares once:

$$
E_j=\sqrt{\tfrac{\alpha\omega}{4\pi^2\hbar c}T_{{\rm abs},j}}\;
A_jQ_j\,
\exp\bigl\{i\bigl[\omega t_{{\rm abs},j}-(\omega\hat{\mathbf n}+\mathbf g)\cdot\mathbf r_j\bigr]\bigr\},
\qquad
E_{\rm tot}=\sum_jE_j,
$$

with $t_{{\rm abs},j}=t^{(0)}_j+t_{0,e}$ (intrinsic segment time plus electron
$e$'s longitudinal bunch offset) and $\mathbf r_j=\mathbf
r_j^{(0)}+\Delta\mathbf r_{\perp,e}$ (intrinsic position plus electron $e$'s
transverse entry offset). $t_{0,e}$ and $\Delta\mathbf r_{\perp,e}$ are
sampled once per electron and held constant over that electron's whole
trajectory (`montecarlo/transport/kinematics.py::_sample_bunch_offsets`, the
`pos[:, :2]` assignment in `montecarlo/transport/api.py`).

Both `coherent-emission` and `transverse-bunch-form-factor` already establish
that the physically correct observable is the ensemble average over these
offsets, not one sampled realization — a single realization of a squared
coherent sum is a speckle pattern whose contrast does not fall with electron
count, unlike ordinary incoherent Monte Carlo noise. Those two rows derive the
single-offset closed forms $\exp[-(\omega\sigma_z)^2]$ (longitudinal) and
$\exp[-(q_\perp\sigma_\perp)^2]$ (transverse, $\mathbf
q_\perp=(\omega\hat{\mathbf n}+\mathbf g)_\perp$) separately. This row derives
the **combined** result when both offsets are present simultaneously, and
describes the implementation that replaces the single-realization sum with the
analytic blend.

## Derivation

### Isolating the per-electron constant phase

$t_{0,e}$ and $\Delta\mathbf r_{\perp,e}$ are constants over electron $e$'s
whole trajectory — a structural fact of the transport model, not a
statistical assumption. $\Delta\mathbf r_{\perp,e}$ is transverse by
construction, so $\mathbf q\cdot\Delta\mathbf r_{\perp,e}=\mathbf
q_\perp\cdot\Delta\mathbf r_{\perp,e}$ with $\mathbf q\equiv\omega\hat{\mathbf
n}+\mathbf g$. The segment phase splits exactly:

$$
\omega t_{{\rm abs},j}-\mathbf q\cdot\mathbf r_j
=\underbrace{\bigl[\omega t^{(0)}_j-\mathbf q\cdot\mathbf r_j^{(0)}\bigr]}_{\text{depends on }j}
+\underbrace{\bigl[\omega t_{0,e}-\mathbf q_\perp\cdot\Delta\mathbf r_{\perp,e}\bigr]}_{\equiv\varphi_e,\text{ depends only on }e}.
$$

Define electron $e$'s **intrinsic** coherent sum $S_e\equiv\sum_{j\in
e}a_j\exp\{i[\omega t^{(0)}_j-\mathbf q\cdot\mathbf r_j^{(0)}]\}$ — no $t_0$,
no $\Delta\mathbf r_\perp$, pure geometric/position phase. Because $e^{i\varphi_e}$
is common to every $j\in e$,

$$
\sum_{j\in e}E_j=e^{i\varphi_e}S_e,\qquad
E_{\rm tot}=\sum_e e^{i\varphi_e}S_e
$$

exactly — pure algebra, no approximation yet.

### Ensemble average

$$
|E_{\rm tot}|^2=\sum_e|S_e|^2+\sum_{e\neq e'}e^{i(\varphi_e-\varphi_{e'})}S_eS_{e'}^*.
$$

**Assumption A:** $t_{0,e}$ and $\Delta\mathbf r_{\perp,e}$ are drawn
independently of each other for the same electron (true here — separate RNG
child streams, `SeedSequence(seed).spawn(4)[3]` for the bunch draw vs.
`spawn(2)[1]` for the transverse spot). Both zero-mean Gaussian:
$t_{0,e}\sim N(0,\sigma_z^2)$, $\Delta\mathbf r_{\perp,e}=(x_e,y_e)$ with
$x_e,y_e\sim N(0,\sigma_\perp^2)$ independent (isotropic spot). Then the
per-electron characteristic function factorizes:

$$
\chi\equiv\langle e^{i\varphi_e}\rangle
=\langle e^{i\omega t_{0,e}}\rangle\langle e^{-i\mathbf q_\perp\cdot\Delta\mathbf r_{\perp,e}}\rangle
=\exp\Bigl[-\tfrac12\bigl(\omega^2\sigma_z^2+q_\perp^2\sigma_\perp^2\bigr)\Bigr].
$$

**Assumption B:** offsets are i.i.d. across electrons (no shared jitter, no
deliberately structured bunch). Then for $e\neq e'$,
$\langle e^{i(\varphi_e-\varphi_{e'})}\rangle=|\chi|^2\equiv F$, a single
scalar pulling out of the double sum:

$$
\langle|E_{\rm tot}|^2\rangle
=\sum_e|S_e|^2+F\Bigl[\Bigl|\sum_eS_e\Bigr|^2-\sum_e|S_e|^2\Bigr]
=\boxed{(1-F)\sum_e|S_e|^2+F\Bigl|\sum_eS_e\Bigr|^2},
$$

$$
\boxed{F=\exp\bigl[-(\omega\sigma_z)^2-(q_\perp\sigma_\perp)^2\bigr]
=\exp\bigl[-(\omega\sigma_z)^2\bigr]\cdot\exp\bigl[-(q_\perp\sigma_\perp)^2\bigr]}.
$$

$F$ is **exactly the product** of the two single-offset form factors already
on record — because $\exp(a)\exp(b)=\exp(a+b)$, this falls straight out of
the characteristic-function factorization in Assumption A; Assumption B is
what licenses treating $F$ as one scalar, not what produces the product
structure.

### Limiting cases

- $\sigma_z,\sigma_\perp\to0\Rightarrow F\to1\Rightarrow\langle|E_{\rm
  tot}|^2\rangle\to|\sum_eS_e|^2$, the fully coherent limit, matching the
  `coherent-emission` $N^2$-limit check.
- $\omega\sigma_z\gg1$ **and/or** $q_\perp\sigma_\perp\gg1\Rightarrow
  F\to0\Rightarrow\langle|E_{\rm tot}|^2\rangle\to\sum_e|S_e|^2$, the
  intra-electron floor — "and/or" because $F$ is a *product* of two factors
  each individually $\le1$; either one vanishing alone kills $F$, so a long
  bunch decoheres the sum even at zero transverse spot, and vice versa.

Both endpoints reduce to the `coherent-emission`/`transverse-bunch-form-factor`
single-offset results as $\sigma_\perp\to0$/$\sigma_z\to0$ respectively — an
internal consistency check, not an assumption borrowed from them.

## Implementation

Rather than assume isotropic Gaussian offsets and plug configured
`bunch_length_fs`/`beam_fwhm_mm` into the closed form directly, `F` is
estimated from the **empirical characteristic function** of the actual
per-electron offsets transport already draws:

$$
F=\Bigl|\tfrac1{N_e}\sum_e\exp\bigl[i(\omega t_{0,e}-\mathbf q_\perp\cdot\Delta\mathbf r_{\perp,e})\bigr]\Bigr|^2,
$$

read from `segments["initial_t0_ang"]`/`segments["initial_r_ang"]` — Ne-long
population arrays transport already returns as diagnostic-only fields
(`simulate_trajectories`'s `initial_r_ang`/`initial_t0_ang`, captured before
transport mutates position, "including missed entries"), distinct from the
per-segment gathered/duplicated `t0_ang` the phase itself uses. This needs no
new per-policy $\sigma$-resolution logic: legacy `bunch_length_fs`/
`long_shape`, `long_offsets_fs`, and the resolved `compressed`/`microtrain`
longitudinal policies, plus elliptical (`beam_fwhm_y_mm`) and Courant–Snyder
(`transverse_distribution`) transverse spots, all fall out for free — the
empirical estimator measures whatever distribution was actually sampled,
needing only Assumption A (t0 independent of the transverse offset within one
electron, true here) rather than a specific closed-form shape. It converges to
the boxed closed form via ordinary $1/\sqrt{N_e}$ statistics (ordinary CLT
statistics on a mean of $N_e$ unit-phase terms), not the non-converging
speckle the naive single-realization sum shows.

$\omega t_{0,e}-\mathbf q_\perp\cdot\Delta\mathbf r_{\perp,e}$ is linear in
$\omega$ (since $\mathbf q_\perp=\omega\hat{\mathbf n}_\perp+\mathbf
g_\perp$), so writing $A_e=t_{0,e}-\hat{\mathbf n}_\perp\cdot\Delta\mathbf
r_{\perp,e}$ (row-independent, hoisted once per `mc_spectrum` call) and
$B_e=\mathbf g_\perp\cdot\Delta\mathbf r_{\perp,e}$ (recomputed per row, cheap)
gives $F(E,\text{row})$ as one $(N_e, N_E)$-shaped reduction, chunked the same
way the field reduction already is — comparable cost to, and typically cheaper
than, the existing per-segment reduction since $N_e\le n_{\rm seg}$.

$F$ is applied **per row** (reflection $\times$ mosaic orientation), before
summing rows: $\mathbf q_\perp$ depends on $\mathbf g$, which differs row to
row, so a single scalar $F(E)$ applied to the row-summed spectrum would be
wrong whenever more than one row contributes to the same energy bin. $S_e$ is
obtained by feeding the SAME existing reduction machinery (per-hkl loop and
the batched `(n_seg, N_g)` path alike) the geometric-only phase inputs
(`d_all_geom`/`seg_r_geom`, i.e. $t_0=0$, $\Delta\mathbf r_\perp=0$) in place
of the offset-including ones — a no-op swap when no decoherence-relevant
offset is configured, so every pre-existing test (all of which use the
degenerate zero-offset fixture) is unaffected bit-for-bit. $\sum_e|S_e|^2$ is
a new reduction, grouping the same row's segments by `elec_id` and squaring
each electron's own sum before adding — the same
group-then-reduce-then-square pattern the pre-existing flight-grouped
incoherent path already uses, keyed by electron instead of flight.

The three float32 CUDA-JIT fast paths (`coherent_jit_kernel.py`,
`coherent_stream_jit_kernel.py`) fuse the reduction and the final squaring
into one device kernel with no electron-grouped floor to blend against, so
they fall back to the (already GPU-capable via CuPy) generic path whenever the
blend is active. This trades the fused kernel's speed for correctness on this
path; native kernel support for the blend is an unimplemented performance
follow-up, not a correctness gap.

## Assumptions and limits

- **Assumption A** (t0 independent of the transverse offset within one
  electron) and **Assumption B** (i.i.d. across electrons) are both required;
  neither is checked at runtime (the sampler already satisfies them by
  construction — independent RNG child streams, one draw per electron).
- **Finite-footprint branch excluded, and rejected explicitly.**
  `crystal_width_mm`/`crystal_height_mm` makes the transverse offset also
  perturb escape-path attenuation (an amplitude effect), coupling amplitude
  and phase randomness — $S_e$ itself becomes a random function of
  $\Delta\mathbf r_{\perp,e}$ and can no longer be factored out from under the
  expectation as a fixed quantity multiplying $e^{i\varphi_e}$, breaking the
  very first algebraic step in the derivation. `mc_spectrum` raises rather
  than silently give a physically-incomplete answer.
- **`sinc_cutoff` excluded when the blend is active, and rejected explicitly**
  — an implementation gap (the windowed-energy-range optimization is not
  implemented for the electron-grouped floor), not a physics one.
- **Correlated transverse–longitudinal phase space** (breaks Assumption A) and
  **correlated/structured bunches** (breaks Assumption B) are not checked and
  would invalidate the boxed $F$ if present; not currently reachable given how
  offsets are sampled.
- **Non-Gaussian offsets** do not invalidate the *empirical* estimator (it
  measures the actual characteristic function of whatever was sampled), only
  the closed-form parametrized alternative that was considered and not used.
- **Elliptical/Courant–Snyder transverse spots are not excluded** — this is a
  direct consequence of using the empirical estimator rather than an isotropic
  closed form, not a separate derivation.
- **Nonzero-mean (systematic, not random) offsets** are not addressed: they
  would not change $F$ (a common phase cancels in the modulus) but would shift
  where the coherent peak sits inside $\sum_eS_e$, which the boxed result as
  stated does not cover.
- **The far-field curvature term** ($\omega r_\perp^2/2R$) is a separate,
  smaller, already-bounded effect (`transverse-bunch-form-factor`); retaining
  it would make $\varphi_e$ quadratic rather than linear in $\Delta\mathbf
  r_{\perp,e}$ and the Gaussian characteristic-function step would not apply
  directly. Out of scope here, consistent with both source rows.

## Status

`rederived`. Fresh-context derivation (independent of this implementation)
reproduces the boxed result with no divergent sign, factor, or unit, and both
limiting cases recover the two rows this one combines exactly.
`test_coherent_decoherence_blend_matches_reference_formula` checks the actual
implementation against a reference built from already-validated,
decoherence-inactive `mc_spectrum` sub-calls plus an independently-computed
`F` — not a self-consistency check against the same code path.
`test_coherent_decoherence_inactive_by_default` pins bit-for-bit equality with
the pre-existing path whenever no offset population is supplied. Human
sign-off remains pending.
