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
`coherent_stream_jit_kernel.py`) carry the blend natively. Both entry points
already *accumulate* their squared reduction
into the caller's buffer rather than overwriting it
(`run_coherent_reduction_kernel`'s `spec[k] += wm*(...)`,
`run_coherent_field_accumulation_kernel`'s additive field planes), so
$\sum_e|S_e|^2$ is the same kernel called once per electron over that
electron's own lines, with unit mosaic weight, into one zeroed buffer;
$|\sum_eS_e|^2$ is the single call these paths already made. Both are fed the
geometric (offset-free) phase, and $F$ multiplies outside the kernel. Three
call sites share this shape: `_accumulate`'s per-(reflection, orientation)
reduction, the batched path's per-row reduction fallback, and the batched
streaming field kernel. The streaming path uses a segmented CUDA reduction:
stable electron grouping is packed into whole-electron segment blocks, threads
form each electron field and square it, then each block reduces those
intensities into the persistent per-row result. It deliberately does **not**
use `finalize_coherent_fields`, whose fused collapse would sum the rows before
$F$ could multiply them.

**Historical cost and production fix.** The initial grouped floor spent one
launch per electron per reduction, and
each launch covers the full energy axis regardless of how few segments that
electron contributes, so the blend's cost is set by $N_e$ and is nearly
independent of segment count and row count. Measured on an RTX 5080 (float32,
2000 energy bins, ~60 segments/electron, blend active), streaming path vs. the
generic CuPy fallback it replaces:

| $N_e$ | rows $N_g$ | JIT blend | generic blend | speedup |
| ----: | ---------: | --------: | ------------: | ------: |
|   300 |          2 |    0.24 s |        0.06 s |   0.26x |
|   300 |         18 |    0.25 s |        0.50 s |   2.02x |
|   300 |         50 |    0.26 s |        1.37 s |   5.26x |
|  1000 |          2 |    0.77 s |        0.17 s |   0.22x |
|  1000 |         18 |    0.80 s |        3.43 s |   4.27x |
|  1000 |         50 |    0.84 s |        8.55 s |  10.15x |
|  3000 |          2 |    2.31 s |        0.56 s |   0.24x |
|  3000 |         18 |    2.36 s |       13.58 s |   5.75x |
|  3000 |         50 |    2.47 s |       20.29 s |   8.22x |

These measurements describe the superseded per-electron streaming reducer.
At the production `hopg_hbn` size ($N_e=20{,}000$), it caused ~0.05 cases/s
and ~10% GPU utilization. The segmented streaming reducer makes launch count
$O(N_{seg}/\mathrm{segment\_block})$ instead of $O(N_e)$. An uncached HOPG
profile sample on the same lab RTX 5080 completed 431/5508 cases in 38 s
(~11.3 cases/s), with 56–62% GPU utilization: about 225x the regressed
throughput. CUDA tests cover the segmented kernel against an independent
grouped-field reference for both one and two energies per block and assert one
grouped launch for a two-electron one-block case.

The two per-row routes (`_accumulate` and the
batched reduction fallback) pay $N_e$ launches *per row* rather than $N_e$
total, which does not amortize: measured 0.76 s at $N_e=300$, $N_g=18$ against
the streaming path's 0.25 s. Both remain correct and tested; they are only
reachable when the streaming path is not (the grooved-escape branch, or
`_USE_JIT_COHERENT_STREAM` disabled).

Remaining performance follow-up, not a correctness gap: the two non-streaming
per-row routes could adopt an equivalent segmented reduction if a production
profile begins using them. The existing `_USE_JIT_COHERENT_STREAM` and
`_USE_JIT_COHERENT_REDUCTION` module switches remain test escape hatches.

## Assumptions and limits

- **Assumption A** (t0 independent of the transverse offset within one
  electron) and **Assumption B** (i.i.d. across electrons) are both required;
  neither is checked at runtime (the sampler already satisfies them by
  construction — independent RNG child streams, one draw per electron).
- **Finite-footprint branch: longitudinal average conditional on sampled
  transverse transport.** `crystal_width_mm`/`crystal_height_mm` makes the
  transverse offset perturb escape-path attenuation (an amplitude effect), so
  the combined longitudinal/transverse phase-only factorization above still
  does not apply. Longitudinal arrival time remains independent, however. The
  implementation therefore keeps each electron's sampled transverse phase,
  hit/miss history, and finite-prism attenuation together in $S_e$, and averages
  only the Gaussian arrival time:

  $$
  \left\langle |A|^2\right\rangle_{t_0\mid S}
  = (1-F_z)\sum_e|S_e|^2 + F_z\left|\sum_e S_e\right|^2,
  \qquad F_z=\exp[-(\omega c\sigma_t)^2].
  $$

  This continuously covers both endpoints: a short bunch retains the sampled
  finite-crystal cross-electron enhancement; a long bunch removes every
  cross-electron term and recovers the grouped floor. It does **not** perform an
  additional ensemble average over transverse bunch/transport realizations;
  partially coherent finite-footprint output may therefore retain physical
  single-realization transverse diffraction/speckle. The conditional extension
  is tracked separately as `finite-footprint-longitudinal-decoherence` because
  it has not received the independent fresh-context verification of this row's
  original combined factorization.
- **The blazed-groove escape branch has the same amplitude/phase coupling as
  the finite-footprint branch, and is NOT currently rejected.**
  `escape_distance_ang` places the emission point inside the sawtooth unit
  cell, so a transverse offset moves the escape path length and therefore
  $|S_e|$, exactly the step the finite-footprint exclusion above is about.
  Measured directly while building the GPU parity tests: with a transverse
  offset, the grooved coherent result departs from the boxed blend by ~130% of
  peak, while with longitudinal offsets only it reproduces it to 4e-16.
  The finite-footprint conditional-longitudinal result does not solve this
  transverse ensemble-average problem; the grooved route retains the same
  limitation.
  `test_coherent_decoherence_blend_holds_on_the_per_hkl_route` therefore pins
  the grooved route with longitudinal offsets only.
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
the pre-existing path whenever no offset population is supplied.

The CUDA paths are pinned by
`tests/montecarlo/test_xray_dispersion_cuda.py`: the same closed-form
reference on device for the streaming and per-row-reduction routes, a lockstep
check of all three routes against the generic CuPy path they used to fall back
to (across a 9-row mosaic cone, so each row carries its own $q_\perp$ and
therefore its own $F$), and bit-for-bit identity of the offset-free dispatch.
Those bodies need the CUDA backend active, which `tests/conftest.py` pins off
by design, so `test_coherent_decoherence_device_suite` re-runs them in a child
session with the pin lifted; verified on the lab RTX 5080 (8 passed). Human
sign-off remains pending.
