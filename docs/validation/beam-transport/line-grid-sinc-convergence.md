# `line-grid-sinc-convergence`

Ledger row: [`line-grid-sinc-convergence`](../ledger-core-coherent-physics.md).
Instrument: `src/pyrite/energy_grid/convergence.py` (grid-agnostic harness),
`convergence_case.py` (catalog-case adapter and resumable driver),
`precision_ladder.py` (float32 versus FP64), submitted through
`convergence_job.py`. No production default, kernel, or catalog row changes;
this row records measured grid convergence of the existing line kernels and
the evidence for the resolution policy of issue #109 and the backend-precision
floor of issue #101.

## Claim

1. **Measurement contract.** On one fixed set of transport segments, every
   difference between two line grids is pure quadrature error. The harness
   evaluates only the spectrum phase on those segments, proves identity with a
   segment fingerprint, and never compares separate Monte Carlo runs.
2. **Resolution.** Over hopg and wse2 at 30, 100, and 300 keV, the integrated
   line yield and centroid converge at the $10^{-3}$ intrinsic-source tolerance
   within one factor-two ladder rung of the `sinc_feature_spacing` estimate at
   `aliased_weight_limit = 1e-3`, and in ten of twelve cases at or above it
   (accepted-to-estimate ratio 0.68–5.5). The fixed 3 eV catalog spacing leaves
   0.54–7.7% hopg yield error at $N_e = 2000$, against at most $2.1\times10^{-4}$
   for wse2. Dominant-line FWHM and the line/background ratio at $10^{-3}$ need
   4–32 times finer spacing than yield and centroid where they converge within
   the ladder at all, and
   the literal monotone clause of {eq}`eq-line-grid-richardson-gate` rejects
   observables whose residual changes are round-off or detector-response noise.
   Only two of twelve configurations pass the literal all-observable gate.
3. **Backend precision.** On identical trajectories, float32 line spectra
   deviate from FP64 by at most $3.4\times10^{-6}$ in yield, $2.3\times10^{-5}$ in
   dominant-line FWHM, 0.39 ulp in centroid, and $5.8\times10^{-5}$ pointwise,
   flat in `spacing/ulp(E_max)` from 8 to 3000 in the 5 and 10 keV float32
   binades. The distortion does not scale like ulp/$h$ and grows with ulp at
   fixed features. The 8-ulp collapse floor already holds it two to three orders
   below the policy tolerances, so these data do not support a ulp/rtol spacing
   floor. No catalog line of the measured cases reaches the 20 keV binade.

## Governing equations and where they come from

No new physical law enters. The line kernels evaluate the finite-time lineshape
of the ledgered `finite-time-lineshape` row,

```{math}
:label: eq-line-grid-sinc-lineshape

\frac{d^2N}{dE\,d\Omega} \propto t_L^2\,
\operatorname{sinc}^2\!\left(\frac{a_w\,(E - E_\mathrm{res})}{\pi}\right),
\qquad a_w = \frac{(1 - \boldsymbol\beta\cdot\hat{\mathbf n})\,t_L}{2\hbar c},
```

with $t_L$ the flight time of a segment, $\hat{\mathbf n}$ the observation
direction, and $E_\mathrm{res}$ the resonance energy. The feature is band
limited, and the trapezoid rule on a uniform grid of spacing $h$ integrates it
exactly when $h \le \pi / a_w$ (the first-zero, or Nyquist, step used by
`sinc_feature_spacing`). Above that step the sampled comb aliases.

**Richardson acceptance.** For consecutive rungs of spacings
$h > h' > h''$ (exact halving for the uniform ladder) an observable $q$ passes
when

```{math}
:label: eq-line-grid-richardson-gate

|q(h) - q(h')| \le \mathrm{rtol}\,|q(h'')| + \mathrm{atol}
\quad\text{and}\quad
|q(h') - q(h'')| < |q(h) - q(h')| \;\text{ or }\; |q(h') - q(h'')| \le \mathrm{atol}.
```

The accepted spacing is the coarsest $h$ whose triple, and every finer triple,
pass. `rtol` is the per-observable tolerance of the automatic line-grid policy
(`pyrite._line_grid_policy.DEFAULT_RTOL`): $10^{-3}$ for intrinsic-source
observables and $10^{-2}$ for detected counts. `atol` is an absolute floor;
the near-zero yield floor is $10^{-15}$ photons sr$^{-1}$ electron$^{-1}$.

## Observables

```{list-table} Gated observables of the refinement ladder.
:name: tbl-line-grid-sinc-observables
:header-rows: 1

* - Observable
  - Definition
  - Tolerance class
* - yield
  - trapezoid of the CXR line density over the rung's own coordinates
  - intrinsic source
* - centroid (eV)
  - intensity-weighted mean energy over the line grid
  - intrinsic source
* - dominant-line FWHM (eV)
  - largest-prominence peak; half-maximum crossings interpolated in physical energy (#110 `line_metrics`)
  - intrinsic source
* - line/background ratio
  - dominant line over bremsstrahlung in its three-FWHM window (`line_metrics.line_brem_ratio`)
  - intrinsic source
* - Timepix3 counts
  - trapezoid of the default `Timepix3` read-time response of the line density
  - detected counts
* - EagleXO counts
  - trapezoid of the default `EagleXO` read-time response of the line density
  - detected counts
```

Peak height is excluded: a sampled density maximum depends on where nodes fall
relative to a feature narrower than the grid. Gated observables use the CXR
line density alone. Characteristic lines are deposited as exact Lorentzian bin
masses (`montecarlo/spectrum/characteristic.py::_lorentzian_bin_weights`), a
grid-exact discretization whose sampled FWHM is the bin width by construction;
their yield is recorded per rung as an ungated informational value.

## Assumptions and limits of validity

- **Identical trajectories.** `CaseLadder` runs the runner's own transport phase
  once (`runner._transport_case`) and re-evaluates the production reductions
  (`_lines_for_segments`, `_characteristic_from_segments`, and the under-line
  bremsstrahlung interpolation). At the case's own grid the ladder rung is
  bit-for-bit the runner spectrum phase
  (`tests/energy-grid/test_convergence_case.py::test_ladder_rung_at_the_case_grid_is_the_production_spectrum`).
  A resumed configuration re-runs its fixed-seed transport and continues only
  if `n_segments` and the kinematic digest are identical.
- **Grid does not enter the transport RNG stream (verified in code).**
  `simulate_trajectories` has no photon-grid parameter
  (`montecarlo/transport/api.py:35`). Its draws come from
  `np.random.default_rng(seed)` and `SeedSequence(seed)` children
  (`api.py:538,551,571,584,620`; `transport/kinematics.py:54,117`) and from
  counter-based SplitMix64 streams keyed only by `(seed, electron id, draw index)` (`kinematics.py:190-209`, consumed at `transport/batching.py:513,672`
  and `api.py:728`). The runner decodes the line grid before transport but uses
  it only in `resolve_line_grid`, after the segments exist
  (`montecarlo/runner/__init__.py:493,588,593`). The spectrum kernels draw no
  random numbers. `test_line_grid_does_not_enter_the_transport_rng_stream` pins
  the consequence: a different line grid gives an identical segment fingerprint.
- **One grid-dependent random input remains, in the detector.** The Timepix3
  response matrix is a seeded Monte Carlo over coarse 50 eV input channels whose
  edges start half a fine step below the first node
  (`detectors/timepix_response.py:244,361`). Its channel centres therefore move
  with $h$, so Timepix3 counts carry response-matrix sampling differences between
  rungs in addition to grid error. This bounds how far below the 1% detected-count
  tolerance a Timepix3 difference can be read as grid convergence.
- **Scope.** Incoherent line route only; the coherent-route CBS continuum is
  #117. Uniform windows only; local and nonuniform windows are #101, which reuse
  the same harness through arbitrary coordinate arrays.

## Limiting cases

- **Band-limited sampling.** For a sum of `sinc^2` features of known $a_w$, the
  trapezoid yield equals the analytic $\sum A\,\pi/a_w$ at every rung with
  $h \le \pi/a_w$, and the accepted spacing never exceeds $\pi/a_w$
  (`tests/energy-grid/test_convergence.py::test_accepted_spacing_resolves_the_sinc_nyquist_step`).
  Above Nyquist, the integrated yield of many phase-spread features can remain
  close to exact, because aliases average out by Poisson summation, while the
  sampled shape does not. The gate still rejects the aliased triple through the
  shape observables.
- **Empty spectrum.** Every observable is accepted through the absolute floor
  and undefined shape observables are not treated as failures.
- **One lucky pair.** A zero change followed by a nonzero finer change is
  rejected; so is a coarse pass followed by any failing finer triple.

## Refinement ladder

Configuration: hopg and wse2; 30, 100, and 300 keV; tilts 5 and 85 deg at each
material's first catalog azimuth (95 deg); the 1 mm diagnostic slab of
`energy_grid.derive`; $N_e = 2000$; seed 0; nested uniform ladders with nominal
$h$ = 12, 6, 3, 1.5, 0.75, 0.375, 0.1875, 0.09375 eV over each case's catalog
line bandwidth (actual spacings are up to 0.2% smaller, from
`resolution_num`). Remote job 1738 on one NVIDIA GeForce RTX 5080, CUDA backend,
float32 `REAL`. A six-rung run of the same matrix (job 1737, finest 0.375 eV)
reproduced every shared rung bit-for-bit.

```{list-table} Coarsest nominal spacing (eV) whose Richardson triple and every finer triple pass, per observable, with the estimate on the same segments, the 3 eV yield error, and the finest-rung cost. A dash means no acceptance within the ladder.
:name: tbl-line-grid-sinc-ladder
:header-rows: 1

* - Case
  - Segments
  - Estimate (eV)
  - Yield
  - Centroid
  - FWHM
  - Line/bg
  - Timepix3
  - EagleXO
  - Yield error at 3 eV
  - Finest wall (s)
  - Device peak (MiB)
* - hopg 30 keV, 5 deg
  - 835 792
  - 0.93
  - –
  - 1.5
  - 0.375
  - 0.375
  - –
  - –
  - 5.8e-3
  - 0.20
  - 350
* - hopg 30 keV, 85 deg
  - 147 337
  - 1.10
  - 0.75
  - 1.5
  - –
  - –
  - –
  - 3
  - 5.4e-3
  - 0.06
  - 350
* - hopg 100 keV, 5 deg
  - 2 199 407
  - 0.47
  - 0.75
  - 1.5
  - –
  - –
  - 1.5
  - –
  - 5.3e-2
  - 0.77
  - 953
* - hopg 100 keV, 85 deg
  - 369 412
  - 0.49
  - 0.75
  - 0.75
  - –
  - –
  - –
  - 1.5
  - 1.1e-2
  - 0.15
  - 953
* - hopg 300 keV, 5 deg
  - 4 726 041
  - 0.21
  - 0.75
  - 0.75
  - –
  - –
  - 1.5
  - 0.75
  - 7.7e-2
  - 3.19
  - 2189
* - hopg 300 keV, 85 deg
  - 755 183
  - 0.22
  - 0.375
  - 0.375
  - –
  - –
  - 3
  - 0.375
  - 2.8e-2
  - 0.53
  - 2189
* - wse2 30 keV, 5 deg
  - 1 644 916
  - 7.29
  - 12
  - 12
  - 0.75
  - –
  - 0.75
  - 6
  - 2.8e-5
  - 1.47
  - 2189
* - wse2 30 keV, 85 deg
  - 199 055
  - 7.00
  - 6
  - 6
  - 1.5
  - 0.375
  - 0.75
  - 6
  - 2.1e-4
  - 0.25
  - 2189
* - wse2 100 keV, 5 deg
  - 5 073 142
  - 4.37
  - 12
  - 12
  - 1.5
  - 0.375
  - –
  - 3
  - 5.0e-7
  - 6.93
  - 2670
* - wse2 100 keV, 85 deg
  - 660 170
  - 4.04
  - 12
  - 12
  - 0.75
  - –
  - –
  - –
  - 1.7e-5
  - 1.18
  - 2670
* - wse2 300 keV, 5 deg
  - 12 471 692
  - 2.20
  - 12
  - 12
  - –
  - –
  - –
  - 12
  - 4.1e-6
  - 58.79
  - 5535
* - wse2 300 keV, 85 deg
  - 1 399 732
  - 2.01
  - 6
  - 6
  - 0.75
  - 0.75
  - 0.375
  - 12
  - 5.1e-5
  - 6.89
  - 5535
```

The estimate column is `sinc_feature_spacing` at the intrinsic-source alias
budget $10^{-3}$ on the same segments; yield error at 3 eV is relative to the
finest rung. Transport took at most 1.4 s per case and host peak RSS stayed at
or below 4.1 GiB. Evaluation cost doubles with each halving on the fine rungs:
the largest case (wse2 300 keV, 5 deg) takes 3.6, 15.4, 30.2, and 58.8 s at
3, 0.375, 0.1875, and 0.094 eV.

**The estimator is a sound resolution target for yield and centroid.**
Taking the finer of the yield and centroid acceptances, the accepted spacing
over the estimate is 1.60, 1.54 (hopg 100 keV), 3.56, 1.68 (hopg 300 keV),
1.64, 0.86 (wse2 30 keV), 2.74, 2.96 (wse2 100 keV), and 5.46, 2.99
(wse2 300 keV); hopg 30 keV reads 0.68 (85 deg) and, for centroid alone, 1.61
(5 deg). The two ratios below one are within a ladder factor and are not
estimator failures. For wse2 30 keV at 85 deg, 12 to 6 eV moves the yield by
$2.6\times10^{-3}$ and 6 to 3 eV by $6.4\times10^{-4}$, so the tolerance is crossed
between the rungs and the 7.0 eV estimate is not resolved by this ladder. For
hopg 30 keV the yield has converged by 1.5 eV (it changes by
$4\times10^{-5}$ and then $6\times10^{-8}$ relative) but is rejected there by the
monotone clause below. The estimate never under-resolves the yield by more
than one rung, and the #109 expectation of roughly 0.3–0.5 eV at 300 keV is
confirmed for hopg (0.375–0.75 eV).

**The fixed 3 eV grid is not converged for hopg at production statistics.**
The 0.02% agreement at 30 keV reported in #109 came from $N_e = 10$. At
$N_e = 2000$ the hopg yield at 3 eV is 0.54–0.58% off at 30 keV,
1.1–5.3% at 100 keV, and 2.8–7.7% at 300 keV, where the yield is
non-monotone in $h$ down to 0.375 eV. wse2, whose longest flights are far
shorter, is converged at 3 eV to $2\times10^{-4}$ or better.

**Shape observables are not convergent at $10^{-3}$ on this ladder.** The
largest-prominence peak of an $N_e = 2000$ spectrum is frequently a narrow
feature built from a few long flights, and its crossing-interpolated width
keeps shrinking as $h$ refines: hopg 300 keV at 5 deg reads 1.250, 1.210,
1.203 eV at 0.375, 0.1875, 0.094 eV, still changing by more than its
0.0012 eV tolerance. Where both converge, FWHM needs 4–16 times and the
line/background ratio 4–32 times finer spacing than yield and centroid. FWHM
converges at 0.75–1.5 eV for the broad wse2 lines and at 0.375 eV for hopg
30 keV, 5 deg; for wse2 300 keV, 5 deg it settles at 405.93 eV from 0.375 eV
down but is rejected by the monotone clause.

**The literal monotone clause rejects converged values.** It requires the
finer change to be strictly smaller, with the absolute floor
($10^{-15}$ for yields and counts) as the only escape. Round-off in a
converged yield (hopg 30 keV, 85 deg: changes of $2$ and
$4\times10^{-12}$ on $1.14\times10^{-6}$) fails it, and so does the Timepix3
count. On rungs at or below 1.5 eV the Timepix3 count still moves by up to
$10^{-3}$ relative per halving, ten times inside the 1% tolerance, while the
yield has converged to $4\times10^{-5}$ (wse2 30 keV). At 85 deg the drift keeps
one sign across four halvings, so it is a systematic grid dependence of the
response's shifting coarse input channels rather than sampling noise. Literal acceptance therefore depends on ladder depth:
hopg 30 keV at 85 deg accepts no yield spacing on six rungs but 0.75 eV on
eight. How the gate should treat changes far inside tolerance, and whether
dominant-line FWHM belongs in the $10^{-3}$ gated set, are open policy
questions for #109; they are reported here, not resolved.

## Float32 versus FP64 lineshape distortion

The #101 question: does float32 coordinate precision distort the line spectrum
like ulp/$h$, which would justify a tolerance-derived floor
`spacing >= ulp(E_max)/rtol`; like ulp over the feature width; or negligibly
once kernels evaluate on the post-cast coordinates of #111?

**Method.** `precision_ladder` pickles one FP64 transport: wse2, 300 keV,
tilt 5 deg, azimuth 95 deg, 100 µm, $N_e = 200$, seed 0, 1 219 539 segments.
It evaluates the incoherent lines in a `PYRITE_FP64=1` process (the CuPy
float64 path) and in a float32 process (the CUDA JIT reduction, which receives
the grid, $E_\mathrm{res}$, and $a_w$ cast to float32,
`montecarlo/spectrum/lines/_per_hkl.py:232-259`). Both run on identical
200 eV uniform windows at `spacing/ulp(E_max)` = 8, 30, 100, 300, 1000, and
3000. The windows sit on the strongest line window of each float32 binade:
top 4506 eV (ulp $4.88\times10^{-4}$ eV) and top 8602 eV (ulp
$9.77\times10^{-4}$ eV). The grids are production-like: nodes are moved off
the float32 lattice (top offset 0.382 ulp, spacing stretched by
$1/(1000\pi)$), so the cast moves nodes by up to half an ulp (0.0625 $h$ at
8 ulp) and cast steps span 0.9997–1.125 $h$. No interval collapsed. Remote job
1740, NVIDIA GeForce RTX 5080.

```{list-table} Float32 deviation from FP64 on identical segments. Relative deviations of yield (on nominal and on post-cast integration coordinates), dominant-line FWHM, and the maximum pointwise density difference over the FP64 peak; centroid shift in float32 ulps at the window top.
:name: tbl-line-grid-sinc-float32
:header-rows: 1

* - Window top (eV)
  - spacing/ulp
  - Spacing (eV)
  - Yield
  - Yield, cast coordinates
  - Centroid shift (ulp)
  - FWHM
  - Max pointwise
* - 4506
  - 8
  - 0.00391
  - 9.9e-7
  - 6.8e-7
  - +0.37
  - 1.1e-5
  - 4.2e-5
* - 4506
  - 30
  - 0.0147
  - 7.0e-7
  - 4.0e-7
  - +0.38
  - 1.1e-5
  - 4.2e-5
* - 4506
  - 100
  - 0.0488
  - 8.2e-7
  - 5.0e-7
  - +0.37
  - 1.2e-5
  - 4.2e-5
* - 4506
  - 300
  - 0.147
  - 8.3e-7
  - 5.0e-7
  - +0.37
  - 1.1e-5
  - 4.2e-5
* - 4506
  - 1000
  - 0.488
  - 7.6e-7
  - 7.8e-7
  - +0.35
  - 1.1e-5
  - 3.6e-5
* - 4506
  - 3000
  - 1.465
  - 3.4e-6
  - 3.6e-7
  - +0.39
  - 8.3e-6
  - 3.3e-5
* - 8602
  - 8
  - 0.00782
  - 2.1e-6
  - 1.5e-6
  - −0.16
  - 1.0e-5
  - 5.8e-5
* - 8602
  - 30
  - 0.0293
  - 2.2e-6
  - 1.7e-6
  - −0.17
  - 1.0e-5
  - 5.7e-5
* - 8602
  - 100
  - 0.0977
  - 1.7e-6
  - 1.2e-6
  - −0.14
  - 1.0e-5
  - 5.6e-5
* - 8602
  - 300
  - 0.293
  - 1.5e-6
  - 1.5e-6
  - −0.16
  - 7.6e-6
  - 5.2e-5
* - 8602
  - 1000
  - 0.977
  - 1.9e-6
  - 1.3e-6
  - −0.14
  - 6.9e-7
  - 5.6e-5
* - 8602
  - 3000
  - 2.93
  - 3.1e-7
  - 2.6e-6
  - −0.31
  - 8.0e-6
  - 1.8e-5
```

Per window, FP64 evaluations took 0.09–0.47 s and float32 0.05–0.15 s; device
peaks were 2992 MiB and 356 MiB.

- **No ulp/$h$ scaling.** Over a 375-fold range of spacing/ulp every deviation is
  flat, with fitted log-log slopes between −0.24 and +0.15 where ulp/$h$
  scaling would give −1. At 8 ulp the cast moves quadrature weights by up to
  12.5% of a step, yet the yield on cast coordinates differs from the yield on
  nominal coordinates by less than $10^{-6}$: the kernels already evaluate on
  the post-cast nodes, and the node jitter averages out over the window.
- **Deviation grows with ulp at fixed features.** Doubling the ulp (4.5 to
  8.6 keV) on the same segments raises the yield deviation about 2.2 times
  (median $8\times10^{-7}$ to $1.8\times10^{-6}$) and the pointwise deviation
  about 1.4 times, and leaves FWHM near $10^{-5}$. That is consistent with an
  error set by ulp over the feature width, from float32 detuning and amplitude
  arithmetic. Feature width was not varied independently, so this scaling is
  inferred, not demonstrated.
- **Negligible at the collapse floor.** At 8 ulp the yield deviation is at most
  $2.2\times10^{-6}$, FWHM $1.2\times10^{-5}$, centroid 0.37 ulp
  ($1.8\times10^{-4}$ eV), and pointwise $5.8\times10^{-5}$: about 450 times
  inside the $10^{-3}$ yield tolerance and 80 times inside it for FWHM.
- **The 20 keV binade carries no measurable line.** No 200 eV window in
  [16384, 32768) eV holds CXR line density for wse2 300 keV at tilt 5 deg,
  either at 100 µm with $N_e = 200$ or at 1 mm with $N_e = 2000$
  (12 471 692 segments; jobs 1739 and 1741). The line kernel culls segments
  whose resonance misses the requested window, so only far sinc tails exist
  there. One more ulp doubling on the trend above would give a yield deviation
  near $4\times10^{-6}$ and pointwise near $10^{-4}$; that is an
  extrapolation, not a measurement.
- **On-lattice control.** A first run with every node exactly
  float32-representable (job 1736: top 8850 eV, 57 549 segments at 10 µm and
  $N_e = 100$) shows the same flat dependence: yield $3.3$–$3.6\times10^{-6}$,
  pointwise $3.0\times10^{-4}$, centroid −0.81 ulp. Its larger pointwise
  deviation reflects sparser and narrower features at one percent of the
  statistics.

**Recommended floor rule.** Keep `DEFAULT_BACKEND_SAFETY_ULPS = 8` as the
collapse floor, and do not adopt `spacing >= ulp(E_max)/rtol` as a
lineshape-accuracy floor. In the measured range the float32 error does not
depend on spacing, so a spacing floor cannot control it, and at 8 ulp it is
already two to three orders below both policy tolerances. If #101 needs a
tolerance-tied precision gate, it should compare ulp with the narrowest sinc
feature, requiring FP64 when `ulp(E_max) / w_min` exceeds a calibrated bound
(`w_min` being the estimator's feature step), not ulp with the spacing. These
data do not calibrate that coefficient. Untested: spacings below 8 ulp (refused
by `validate_backend_spacing`), the 20 keV binade, and other feature-width
populations.

```bash
python -m pyrite.energy_grid.convergence_job start-precision --json-out line_grid_precision_109.json
python -m pyrite.energy_grid.convergence_job pull --json-out line_grid_precision_109.json
```

## What this row does not claim

- No default changes. `DEFAULT_RTOL`, `DEFAULT_MAX_SPACING_EV`, and
  `DEFAULT_BACKEND_SAFETY_ULPS` are unchanged.
- It does not claim the line spectra are physically correct, only how they
  respond to grid refinement and to backend precision.
- It does not regenerate catalog `line_by_energy` rows.
