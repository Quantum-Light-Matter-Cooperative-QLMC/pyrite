# `line-grid-sinc-convergence`

Ledger row: [`line-grid-sinc-convergence`](../ledger-core-coherent-physics.md). Instrument: `src/pyrite/energy_grid/convergence.py` (grid-agnostic harness), `convergence_case.py` (catalog-case adapter and resumable driver), `precision_ladder.py` (float32 versus FP64), submitted through `convergence_job.py`. No kernel or numerical-policy constant changes; this row records the reproducible convergence-instrument contract and the sampling argument behind the resolution policy of issue #109. Historical remote measurements are retained below as provenance only; their source artifacts were deliberately retired and they are not validation evidence.

## Claim

1. **Measurement contract.** On one fixed set of transport segments, every difference between two line grids is pure quadrature error. The harness evaluates only the spectrum phase on those segments, proves identity with a segment fingerprint, and never compares separate Monte Carlo runs.
2. **Resolution contract.** The `sinc^2` band limit makes $h\leq\pi/a_w$ a sufficient exact-integration condition on an infinite uniform grid, not a ceiling on empirical Richardson acceptance. A sum of phase-shifted features may pass at a coarser spacing when aliases cancel; the analytic anchor therefore requires the accepted yield to match the independent integral and separately rejects the grossly aliased coarse triple.
3. **Gate and precision instruments.** The Richardson gate implements {eq}`eq-line-grid-richardson-gate`, including its absolute floor, the $0.1\tau$ noise clause, and the all-finer-triples suffix. The precision comparison evaluates float32 and FP64 from one serialized transport and rejects mismatched fingerprints. No backend-error magnitude is claimed without a retained measurement artifact.

## Governing equations and where they come from

No new physical law enters. The line kernels evaluate the finite-time lineshape of the ledgered `finite-time-lineshape` row,

```{math}
:label: eq-line-grid-sinc-lineshape

\frac{d^2N}{dE\,d\Omega} \propto t_L^2\,
\operatorname{sinc}^2\!\left(\frac{a_w\,(E - E_\mathrm{res})}{\pi}\right),
\qquad a_w = \frac{(1 - \boldsymbol\beta\cdot\hat{\mathbf n})\,t_L}{2\hbar c},
```

with $t_L$ the $c=1$ length-equivalent flight time of a segment, $\hat{\mathbf n}$ the observation direction, and $E_\mathrm{res}$ the resonance energy. The feature is band limited, and the trapezoid rule on a uniform grid of spacing $h$ integrates it exactly when $h \le \pi / a_w$ (the first-zero, or Nyquist, step used by `sinc_feature_spacing`). Above that step the sampled comb aliases.

**Richardson acceptance.** For consecutive rungs of spacings $h > h' > h''$ (exact halving for the uniform ladder), write $d_1 = |q(h) - q(h')|$, $d_2 = |q(h') - q(h'')|$, and $\tau = \mathrm{rtol}\,|q(h'')| + \mathrm{atol}$. An observable $q$ passes when

```{math}
:label: eq-line-grid-richardson-gate

\bigl(d_1 < \tau \;\text{ or }\; d_1 = 0\bigr)
\quad\text{and}\quad
\bigl(\, d_2 \le d_1 \;\text{ or }\; \max(d_1, d_2) \le 0.1\,\tau \,\bigr).
```

The first clause bounds the change from $h$ to $h'$. The second rejects one lucky pair, a small change followed by a larger one, unless both changes are noise far inside the tolerance. The factor 0.1 is a harness policy margin, not a derived physical constant. An earlier strictly-smaller form ($d_2 < d_1$, with only the absolute floor as escape) rejected converged values whose residual changes were round-off or detector-response noise, which made the accepted spacing depend on ladder depth (see below). A zero change passes against a zero tolerance. A triple whose finest-rung yield is at or below the near-zero floor of $10^{-15}$ photons sr$^{-1}$ electron$^{-1}$ is accepted.

The accepted spacing is the coarsest $h$ whose triple, and every finer triple, pass. Relative tolerances per class are:

- **Intrinsic source** (yield, centroid): $10^{-3}$, from `pyrite._line_grid_policy.DEFAULT_RTOL`.
- **Detected counts**: $10^{-2}$, from the same production mapping.
- **Shape** (dominant-line FWHM, line/background): $10^{-2}$. This class exists only in the harness (`energy_grid.convergence.SHAPE_RTOL`) and does not change the production policy classes.

Shape observables are also reported, ungated, at $10^{-3}$.

## Independent verification basis

This section records the issue #126 verifier's derivation before inspection of the convergence-harness implementation. The source claim is the ledgered `finite-time-lineshape` result. The intended measurement maps one fixed segment set and a nested sequence of line-grid coordinate arrays to per-rung line spectra, observables, and a segment-identity fingerprint; the precision study maps one serialized transport to paired float32 and FP64 evaluations. The reported yield has units photons sr$^{-1}$ electron$^{-1}$, centroid and FWHM have units eV, line/background is dimensionless, and detected counts retain the detector response's count normalization.

Write $x = E-E_\mathrm{res}$ and use NumPy's normalized sinc convention. A single finite-segment factor is

$$
f(x)
= \operatorname{sinc}^2\!\left(\frac{a_w x}{\pi}\right)
= \left(\frac{\sin(a_w x)}{a_w x}\right)^2.
$$

Here $t_L=L_\mathrm{seg}/\beta$ is the source derivation's $c=1$ length-equivalent flight time, so $a_w=(1-\boldsymbol\beta\mathbin{\cdot}\hat{\mathbf n})t_L/(2\hbar c)$ has units eV$^{-1}$ and the sinc argument is dimensionless. The Fourier transform of $f$ is triangular with support $|k|\leq 2a_w$. Poisson summation places the nearest replicas of that support at $2\pi/h$ for a uniform grid of spacing $h$. Therefore the sampled zero-frequency component, and hence the infinite-grid trapezoid area, is unaliased when

$$
\frac{2\pi}{h}\geq 2a_w
\quad\Longleftrightarrow\quad
h\leq\frac{\pi}{a_w}.
$$

Equality is admissible because the triangular transform vanishes at its boundary. Above this step replicas can contribute to the area, so the band limit alone no longer certifies exactness. It does not forbid a particular sum of phase-shifted features from integrating accurately through cancellation; a convergence gate may therefore accept such a spacing when it is checked against an independent integral tolerance. The checks assume a uniform effectively unbounded grid for the exact area statement; a finite window additionally carries its omitted-tail error, and sums of sinc features retain the strictest individual sufficient band limit.

For the Richardson gate, $d_1$, $d_2$, and $\tau$ all have the units of the observable. The first clause enforces the requested tolerance on the coarse pair, with an explicit exact-zero escape when $\tau=0$. The $d_2\leq d_1$ branch requires non-growing refinement changes. The alternative $\max(d_1,d_2)\leq0.1\tau$ permits non-monotone residuals only when both are a fixed decade inside tolerance. Requiring the candidate triple and every finer available triple to pass prevents a single lucky coarse pair; adding finer rungs may expose a later failure, while already judgeable shared acceptances remain depth-independent only as an empirical property of the stored ladder. The $10^{-15}$ yield floor is dimensionally a yield and is a harness policy for an effectively empty spectrum, not a consequence of Richardson extrapolation.

These filters establish the sinc band limit and the dimensional and logical consistency of the gate. They do not establish the measured convergence or float32 bounds; those require comparison with the implementation, anchor tests, and stored artifacts below.

## Observables

```{list-table} Gated observables of the refinement ladder and their tolerance classes.
:name: tbl-line-grid-sinc-observables
:header-rows: 1

* - Observable
  - Definition
  - Class (rtol)
* - yield
  - trapezoid of the CXR line density over the rung's own coordinates
  - intrinsic source ($10^{-3}$)
* - centroid (eV)
  - intensity-weighted mean energy over the line grid
  - intrinsic source ($10^{-3}$)
* - dominant-line FWHM (eV)
  - largest-prominence peak; half-maximum crossings interpolated in physical energy (#110 `line_metrics`)
  - shape ($10^{-2}$; diagnostic at $10^{-3}$)
* - line/background ratio
  - dominant line over bremsstrahlung in its three-FWHM window (`line_metrics.line_brem_ratio`)
  - shape ($10^{-2}$; diagnostic at $10^{-3}$)
* - Timepix3 counts
  - trapezoid of the default `Timepix3` read-time response of the line density
  - detected counts ($10^{-2}$)
* - EagleXO counts
  - trapezoid of the default `EagleXO` read-time response of the line density
  - detected counts ($10^{-2}$)
```

Peak height is excluded: a sampled density maximum depends on where nodes fall relative to a feature narrower than the grid. Gated observables use the CXR line density alone. Characteristic lines are deposited as exact Lorentzian bin masses (`montecarlo/spectrum/characteristic.py::_lorentzian_bin_weights`), a grid-exact discretization whose sampled FWHM is the bin width by construction; their yield is recorded per rung as an ungated informational value.

## Assumptions and limits of validity

- **Identical trajectories.** `CaseLadder` runs the runner's own transport phase once (`runner._transport_case`) and re-evaluates the production reductions (`_lines_for_segments`, `_characteristic_from_segments`, and the under-line bremsstrahlung interpolation). At the case's own grid the ladder rung is bit-for-bit the runner spectrum phase (`tests/energy-grid/test_convergence_case.py::test_ladder_rung_at_the_case_grid_is_the_production_spectrum`). A resumed configuration re-runs its fixed-seed transport and continues only if `n_segments` and the kinematic digest are identical.
- **Grid does not enter the transport RNG stream (verified in code).** `simulate_trajectories` has no photon-grid parameter (`montecarlo/transport/api.py:35`). Its draws come from `np.random.default_rng(seed)` and `SeedSequence(seed)` children (`api.py:538,551,571,584,620`; `transport/kinematics.py:54,117`) and from counter-based SplitMix64 streams keyed only by `(seed, electron id, draw index)` (`kinematics.py:190-209`, consumed at `transport/batching.py:513,672` and `api.py:728`). The runner decodes the line grid before transport but uses it only in `resolve_line_grid`, after the segments exist (`montecarlo/runner/__init__.py:493,588,593`). The spectrum kernels draw no random numbers. `test_line_grid_does_not_enter_the_transport_rng_stream` pins the consequence: a different line grid gives an identical segment fingerprint.
- **One grid-dependent random input remains, in the detector.** The Timepix3 response matrix is a seeded Monte Carlo over coarse 50 eV input channels whose edges start half a fine step below the first node (`detectors/timepix_response.py:244,361`). Its channel centres therefore move with $h$, so Timepix3 counts carry response-matrix sampling differences between rungs in addition to grid error. This bounds how far below the 1% detected-count tolerance a Timepix3 difference can be read as grid convergence.
- **Scope.** Incoherent line route only; the coherent-route CBS continuum is #117. Uniform windows only; local and nonuniform windows are #101, which reuse the same harness through arbitrary coordinate arrays.

## Limiting cases

- **Band-limited sampling.** For a sum of `sinc^2` features of known $a_w$, the trapezoid yield equals the analytic $\sum A\,\pi/a_w$ at every rung with $h \le \pi/a_w$ (`tests/energy-grid/test_convergence.py::test_sinc_nyquist_is_sufficient_not_an_acceptance_ceiling`). Above Nyquist the integrated observables of many phase-spread features can stay inside tolerance, because aliases average out by Poisson summation. The accepted spacing is therefore not bounded by $\pi/a_w$; the gate guarantees that the accepted rung's yield meets the tolerance against the analytic integral, and that the grossly aliased 12 eV triple is rejected.
- **Empty spectrum.** Every observable is accepted through the absolute floor and undefined shape observables are not treated as failures.
- **One lucky pair and noise.** A change followed by a larger one near the tolerance is rejected; two changes both at most $0.1\,\tau$ are accepted even if the second is slightly larger; a coarse pass followed by any failing finer triple is not accepted (`test_a_larger_second_change_near_the_tolerance_is_rejected`, `test_two_tiny_changes_accept_even_when_the_second_is_slightly_larger`, `test_accepted_spacing_requires_every_finer_triple_to_pass`).

## Historical refinement ladder (non-validating)

The source rung artifacts for this section were retired from the remote store. The values below remain only as a record of the investigation that motivated the current automatic-grid policy. They are not part of this row's validated claim and must not be used as current quantitative evidence.

Configuration: hopg and wse2; 30, 100, and 300 keV; tilts 5 and 85 deg at each material's first catalog azimuth (95 deg); the 1 mm diagnostic slab of `energy_grid.derive`; $N_e = 2000$; seed 0; nested uniform ladders with nominal $h$ = 12, 6, 3, 1.5, 0.75, 0.375, 0.1875, 0.09375 eV over each case's catalog line bandwidth (actual spacings are up to 0.2% smaller, from `resolution_num`). Remote job 1738 on one NVIDIA GeForce RTX 5080, CUDA backend, float32 `REAL`. A six-rung run of the same matrix (job 1737, finest 0.375 eV) reproduced every shared rung bit-for-bit (72 rungs). All acceptances below are recomputed from the stored rung observables with the gate of {eq}`eq-line-grid-richardson-gate`; no ladder was re-run for the gate change.

```{list-table} Coarsest nominal spacing (eV) whose triple and every finer triple pass, per observable, on eight rungs, with the sinc estimate at alias budget 1e-3 on the same segments. Shape columns are gated at 1e-2; the last two columns are the ungated 1e-3 diagnostics. A dash means no acceptance within the ladder.
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
  - FWHM at 1e-3
  - Line/bg at 1e-3
* - hopg 30 keV, 5 deg
  - 835 792
  - 0.93
  - 1.5
  - 1.5
  - 0.375
  - 0.375
  - 3
  - 3
  - 0.375
  - 0.375
* - hopg 30 keV, 85 deg
  - 147 337
  - 1.10
  - 1.5
  - 1.5
  - 0.375
  - 0.375
  - 3
  - 3
  - –
  - –
* - hopg 100 keV, 5 deg
  - 2 199 407
  - 0.47
  - 0.75
  - 1.5
  - –
  - –
  - 1.5
  - 1.5
  - –
  - –
* - hopg 100 keV, 85 deg
  - 369 412
  - 0.49
  - 0.75
  - 0.75
  - –
  - 0.375
  - 6
  - 1.5
  - –
  - –
* - hopg 300 keV, 5 deg
  - 4 726 041
  - 0.21
  - 0.75
  - 0.75
  - –
  - –
  - 1.5
  - 0.75
  - –
  - –
* - hopg 300 keV, 85 deg
  - 755 183
  - 0.22
  - 0.375
  - 0.375
  - –
  - –
  - 3
  - 0.375
  - –
  - –
* - wse2 30 keV, 5 deg
  - 1 644 916
  - 7.29
  - 12
  - 12
  - 3
  - 0.75
  - 6
  - 12
  - 0.75
  - –
* - wse2 30 keV, 85 deg
  - 199 055
  - 7.00
  - 6
  - 6
  - 6
  - 12
  - 6
  - 6
  - 1.5
  - 0.375
* - wse2 100 keV, 5 deg
  - 5 073 142
  - 4.37
  - 12
  - 12
  - 6
  - 6
  - 12
  - 12
  - 1.5
  - 0.375
* - wse2 100 keV, 85 deg
  - 660 170
  - 4.04
  - 12
  - 12
  - 1.5
  - 0.375
  - 12
  - 12
  - 0.75
  - –
* - wse2 300 keV, 5 deg
  - 12 471 692
  - 2.20
  - 12
  - 12
  - 6
  - 6
  - 12
  - 12
  - 0.75
  - 6
* - wse2 300 keV, 85 deg
  - 1 399 732
  - 2.01
  - 6
  - 6
  - 1.5
  - 1.5
  - 12
  - 12
  - 0.75
  - 0.75
```

Transport took at most 1.4 s per case and host peak RSS stayed at or below 4.1 GiB. Evaluation cost doubles with each halving on the fine rungs: the largest case (wse2 300 keV, 5 deg) takes 3.6, 15.4, 30.2, and 58.8 s at 3, 0.375, 0.1875, and 0.094 eV, with a 5.5 GiB device peak.

**The estimator is a sound resolution target for yield and centroid.** Taking the finer of the yield and centroid acceptances, the accepted spacing over the estimate is 1.61 and 1.36 (hopg 30 keV), 1.61 and 1.54 (hopg 100 keV), 3.57 and 1.69 (hopg 300 keV), 1.65 and 0.86 (wse2 30 keV), 2.75 and 2.97 (wse2 100 keV), and 5.45 and 2.99 (wse2 300 keV). In eleven of twelve cases the converged spacing is at or coarser than the estimate. For wse2 30 keV at 85 deg, 12 to 6 eV moves the yield by $2.6\times10^{-3}$ and 6 to 3 eV by $6.4\times10^{-4}$, so the tolerance is crossed between rungs and the 7.0 eV estimate is not resolved by this ladder. The #109 expectation of roughly 0.3–0.5 eV at 300 keV is confirmed for hopg (0.375–0.75 eV).

**Shape is not certified at $10^{-3}$ by this spacing.** The largest-prominence peak of an $N_e = 2000$ spectrum is often a narrow feature built from a few long flights, and its crossing-interpolated width keeps shrinking as $h$ refines: hopg 300 keV at 5 deg reads 1.250, 1.210, 1.203 eV at 0.375, 0.1875, 0.094 eV. Where the $10^{-3}$ diagnostics converge within the ladder, FWHM needs 4–16 times and line/background 2–32 times finer spacing than yield and centroid; for hopg at 100 and 300 keV neither converges at $10^{-3}$ above 0.094 eV. At the $10^{-2}$ shape tolerance hopg 30 keV and every wse2 case converge; hopg at 100 and 300 keV does not.

**Accepted spacing no longer depends on ladder depth.** With the earlier strictly-smaller clause, round-off in a converged yield (hopg 30 keV, 85 deg: changes of 2 and $4\times10^{-12}$ on $1.14\times10^{-6}$) rejected it, so the same data accepted no yield spacing on six rungs but 0.75 eV on eight. Under the current gate both read 1.5 eV.

```{list-table} Gated accepted spacing (eV) on six versus eight rungs of the same stored data, per case, with the yield and centroid acceptances (six / eight) and the reason for any difference.
:name: tbl-line-grid-sinc-depth
:header-rows: 1

* - Case
  - Gated, 6 rungs
  - Gated, 8 rungs
  - Yield
  - Centroid
  - Difference
* - hopg 30 keV, 5 deg
  - –
  - 0.375
  - 1.5 / 1.5
  - 1.5 / 1.5
  - shape converges at 0.375 eV, below the six-rung ladder's finest judgeable triple (1.5 eV)
* - hopg 30 keV, 85 deg
  - –
  - 0.375
  - 1.5 / 1.5
  - 1.5 / 1.5
  - as above
* - hopg 100 keV, 5 deg
  - –
  - –
  - – / 0.75
  - 1.5 / 1.5
  - yield converges at 0.75 eV, not judgeable on six rungs
* - hopg 100 keV, 85 deg
  - –
  - –
  - – / 0.75
  - – / 0.75
  - as above
* - hopg 300 keV, 5 deg
  - –
  - –
  - – / 0.75
  - – / 0.75
  - as above
* - hopg 300 keV, 85 deg
  - –
  - –
  - – / 0.375
  - – / 0.375
  - as above
* - wse2 30 keV, 5 deg
  - –
  - 0.75
  - 12 / 12
  - 12 / 12
  - line/background converges at 0.75 eV, not judgeable on six rungs
* - wse2 30 keV, 85 deg
  - 6
  - 6
  - 6 / 6
  - 6 / 6
  - none
* - wse2 100 keV, 5 deg
  - 6
  - 6
  - 12 / 12
  - 12 / 12
  - none
* - wse2 100 keV, 85 deg
  - 1.5
  - 0.375
  - 12 / 12
  - 12 / 12
  - line/background changes by 3.8e-3 at 0.1875→0.094 eV (0.25 of tolerance, 19 times the previous change), visible only on eight rungs
* - wse2 300 keV, 5 deg
  - 6
  - 6
  - 12 / 12
  - 12 / 12
  - none
* - wse2 300 keV, 85 deg
  - 1.5
  - 1.5
  - 6 / 6
  - 6 / 6
  - none
```

For yield, centroid, and both detected counts, wherever the six-rung ladder accepts a spacing the eight-rung ladder accepts the same spacing, in all twelve cases. Every remaining difference is either a convergence below 1.5 eV, the finest spacing a six-rung ladder can judge, or, once, a change above the noise fraction that only the extra rungs expose.

**Timepix3 counts carry a grid-dependent drift.** On rungs at or below 1.5 eV the Timepix3 count still moves by up to $10^{-3}$ relative per halving, ten times inside the 1% tolerance, while the yield has converged to $4\times10^{-5}$ (wse2 30 keV). At 85 deg the drift keeps one sign across four halvings, so it is a systematic grid dependence of the response's shifting coarse input channels, not sampling noise.

## Historical hopg catalog resolution (non-validating)

`energy_grid.derive` was run remotely for hopg at every row energy of the standard-profile artifact (job 1742: the persisted diagnostic geometry, 1 mm, coarse $N_e = 200$, refine $N_e = 2000$, alias budget $10^{-2}$). Its resolution targets follow the transport, not the catalog bandwidth. Its bandwidth outputs are not usable: the 95% coverage `stop` collapses to 300 eV at 30 keV (catalog 2600 eV) because line coverage integrated the characteristic lines added to the line spectrum (C K near 277 eV); #123 fixes that channel. The 140 keV bremsstrahlung stop is not the same defect: the bremsstrahlung channel carries no characteristic emission. The per-material bremsstrahlung stop is the maximum over all row energies up to 300 keV, and 140 keV agrees with the 140.7 keV override of diamond, which is also carbon. That points to the 40 keV hopg override as stale rather than to a coverage defect; the catalog is not regenerated here. The resolution below therefore keeps each row's catalog `start` and `stop` and sets `num = resolution_num(start, stop, target)`. Every row passes the float32 8-ulp backend check.

Issue #123's separated-component implementation was rerun remotely at the same 1 mm, 5 deg polar, 95 deg azimuth geometry for hopg and wse2 (job 1768, `20260917-105036-f96fc59f`). At hopg 30 keV, the raw PXR/CBS-only 95% coverage is 2470.3 eV and the margin policy rounds it to 2600 eV, exactly the catalog stop; the characteristic-contaminated run had rounded to 300 eV. The same row's bremsstrahlung raw/rounded stops remain 13.55/14.3 keV, inside hopg's 40 keV override. Across matching catalog rows, rounded line stops were at or below the catalog stops except wse2 at 100 keV (3300 vs 3100 eV) and 150 keV (4400 vs 4300 eV). Those two rows require review before any later catalog regeneration; no artifacts or profile references were changed here.

```{list-table} hopg standard-profile rows: catalog num and the num at the derived target spacing, with the derived target.
:name: tbl-line-grid-sinc-hopg-rows
:header-rows: 1

* - Beam (keV)
  - start–stop (eV)
  - Catalog num (spacing, eV)
  - Derived target (eV)
  - num at target (spacing, eV)
* - 30
  - 10–2600
  - 864 (3.001)
  - 1.273
  - 2036 (1.273)
* - 35
  - 10–2900
  - 964 (3.001)
  - 1.170
  - 2471 (1.170)
* - 40
  - 10–3000
  - 998 (2.999)
  - 1.086
  - 2755 (1.086)
* - 50
  - 10–3300
  - 1098 (2.999)
  - 0.969
  - 3397 (0.969)
* - 60
  - 10–3700
  - 1231 (3.000)
  - 0.876
  - 4214 (0.876)
* - 100
  - 50–4600
  - 1518 (2.999)
  - 0.640
  - 7112 (0.640)
* - 150
  - 50–5600
  - 1851 (3.000)
  - 0.489
  - 11351 (0.489)
* - 200
  - 50–6300
  - 2084 (3.000)
  - 0.401
  - 15583 (0.401)
* - 250
  - 50–7700
  - 2551 (3.000)
  - 0.342
  - 22392 (0.342)
* - 300
  - 50–9100
  - 3018 (3.000)
  - 0.298
  - 30399 (0.298)
```

The yield error of both grids was measured on the ladder's own segments (jobs 1743–1745, fingerprints identical to job 1738) against a 0.09375 eV reference.

```{list-table} hopg yield error and centroid shift on the exact catalog grid and on the derived-spacing grid, against a 0.09375 eV reference on identical segments.
:name: tbl-line-grid-sinc-hopg-error
:header-rows: 1

* - Case
  - Catalog num
  - Yield error
  - Centroid shift (eV)
  - Derived num
  - Yield error
  - Centroid shift (eV)
  - Derived-grid wall (s)
* - 30 keV, 5 deg
  - 864
  - 1.2e-3
  - +2.88
  - 2036
  - 1.3e-5
  - −0.002
  - 0.02
* - 30 keV, 85 deg
  - 864
  - 5.7e-3
  - −0.34
  - 2036
  - 1.7e-5
  - +0.007
  - 0.01
* - 100 keV, 5 deg
  - 1518
  - 4.6e-2
  - −7.99
  - 7112
  - 2.6e-5
  - −0.017
  - 0.12
* - 100 keV, 85 deg
  - 1518
  - 1.0e-2
  - −1.29
  - 7112
  - 7.7e-5
  - −0.002
  - 0.03
* - 300 keV, 5 deg
  - 3018
  - 2.1e-2
  - +13.1
  - 30399
  - 9.1e-6
  - −0.009
  - 1.14
* - 300 keV, 85 deg
  - 3018
  - 4.7e-2
  - +30.3
  - 30399
  - 6.8e-6
  - −0.005
  - 0.21
```

At $N_e = 2000$ the fixed 3 eV grid is not converged for hopg, and the error on it is phase-sensitive: at 30 keV, 5 deg the exact catalog grid (3.001 eV) is off by 0.12% while the ladder's 2.998 eV rung is off by 0.58%. The derived-spacing rows bring every measured case inside the $10^{-3}$ yield tolerance with at most $7.7\times10^{-5}$, at 2.4–10 times the points. In this revision the catalog rows are unchanged: the standard hopg artifact is also resolved, through fallback or artifact references, by fourteen other profiles, and moving only one of them would break their shared case identity.

**Other catalog materials.** A transport-only screen compared each material's catalog spacing at 30, 100, and 300 keV with the estimate at derive's alias budget ($10^{-2}$). It used $N_e = 20$, 1 mm, and tilt 5 deg at the first catalog azimuth. 59 of 150 rows, in 41 materials, are coarser than the estimate. The largest non-hopg ratios are hbn (9.2), diamond (6.5), silicon (6.2), black phosphorus (5.2), 4H- and 6H-SiC (4.4), V2O5 (4.0), sapphire (3.8), and TiS2 (3.8), all at 300 keV. The wse2 rows sit at 0.28–0.87 of the estimate, and wse2 yield is measured converged at 3 eV. The screen compares spacings, not yields. The wse2 ladder shows the estimate is conservative for integrals (yield converges at 12 eV against estimates of 2.0–7.3 eV), so a ratio above one does not by itself establish a yield error above tolerance. Only hopg and wse2 are measured.

## Historical float32 versus FP64 study (non-validating)

The #101 question: does float32 coordinate precision distort the line spectrum like ulp/$h$, which would justify a tolerance-derived floor `spacing >= ulp(E_max)/rtol`; like ulp over the feature width; or negligibly once kernels evaluate on the post-cast coordinates of #111?

**Method.** `precision_ladder` pickles one FP64 transport: wse2, 300 keV, tilt 5 deg, azimuth 95 deg, 100 µm, $N_e = 200$, seed 0, 1 219 539 segments. It evaluates the incoherent lines in a `PYRITE_FP64=1` process (the CuPy float64 path) and in a float32 process (the CUDA JIT reduction, which receives the grid, $E_\mathrm{res}$, and $a_w$ cast to float32, `montecarlo/spectrum/lines/_per_hkl.py:232-259`). Both run on identical 200 eV uniform windows at `spacing/ulp(E_max)` = 8, 30, 100, 300, 1000, and
3000. The windows sit on the strongest line window of each float32 binade: top 4506 eV (ulp $4.88\times10^{-4}$ eV) and top 8602 eV (ulp $9.77\times10^{-4}$ eV). The grids are production-like: nodes are moved off the float32 lattice (top offset 0.382 ulp, spacing stretched by $1/(1000\pi)$), so the cast moves nodes by up to half an ulp (0.0625 $h$ at 8 ulp) and cast steps span 0.9997–1.125 $h$. No interval collapsed. Remote job 1740, NVIDIA GeForce RTX 5080.

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

Per window, FP64 evaluations took 0.09–0.47 s and float32 0.05–0.15 s; device peaks were 2992 MiB and 356 MiB.

- **No ulp/$h$ scaling.** Over a 375-fold range of spacing/ulp every deviation is flat, with fitted log-log slopes between −0.24 and +0.15 where ulp/$h$ scaling would give −1. At 8 ulp the cast moves quadrature weights by up to 12.5% of a step, yet the yield on cast coordinates differs from the yield on nominal coordinates by less than $10^{-6}$: the kernels already evaluate on the post-cast nodes, and the node jitter averages out over the window.
- **Deviation grows with ulp at fixed features.** Doubling the ulp (4.5 to 8.6 keV) on the same segments raises the yield deviation about 2.2 times (median $8\times10^{-7}$ to $1.8\times10^{-6}$) and the pointwise deviation about 1.4 times, and leaves FWHM near $10^{-5}$. That is consistent with an error set by ulp over the feature width, from float32 detuning and amplitude arithmetic. Feature width was not varied independently, so this scaling is inferred, not demonstrated.
- **Negligible at the collapse floor.** At 8 ulp the yield deviation is at most $2.2\times10^{-6}$, FWHM $1.2\times10^{-5}$, centroid 0.37 ulp ($1.8\times10^{-4}$ eV), and pointwise $5.8\times10^{-5}$: about 450 times inside the $10^{-3}$ yield tolerance and 80 times inside it for FWHM.
- **The 20 keV binade carries no measurable line.** No 200 eV window in [16384, 32768) eV holds CXR line density for wse2 300 keV at tilt 5 deg, either at 100 µm with $N_e = 200$ or at 1 mm with $N_e = 2000$ (12 471 692 segments; jobs 1739 and 1741). The line kernel culls segments whose resonance misses the requested window, so only far sinc tails exist there. One more ulp doubling on the trend above would give a yield deviation near $4\times10^{-6}$ and pointwise near $10^{-4}$; that is an extrapolation, not a measurement.
- **On-lattice control.** A first run with every node exactly float32-representable (job 1736: top 8850 eV, 57 549 segments at 10 µm and $N_e = 100$) shows the same flat dependence: yield $3.3$–$3.6\times10^{-6}$, pointwise $3.0\times10^{-4}$, centroid −0.81 ulp. Its larger pointwise deviation reflects sparser and narrower features at one percent of the statistics.

**Recommended floor rule.** Keep `DEFAULT_BACKEND_SAFETY_ULPS = 8` as the collapse floor, and do not adopt `spacing >= ulp(E_max)/rtol` as a lineshape-accuracy floor. In the measured range the float32 error does not depend on spacing, so a spacing floor cannot control it, and at 8 ulp it is already two to three orders below both policy tolerances. If #101 needs a tolerance-tied precision gate, it should compare ulp with the narrowest sinc feature, requiring FP64 when `ulp(E_max) / w_min` exceeds a calibrated bound (`w_min` being the estimator's feature step), not ulp with the spacing. These data do not calibrate that coefficient. Untested: spacings below 8 ulp (refused by `validate_backend_spacing`), the 20 keV binade, and other feature-width populations.

```bash
python -m pyrite.energy_grid.convergence_job start-precision --json-out line_grid_precision_109.json
python -m pyrite.energy_grid.convergence_job pull --json-out line_grid_precision_109.json
```

## Historical issue #125 bundled-default audit (non-validating)

On 2026-09-19, jobs `20260919-181223-fc93ad6e` (Slurm 1774) and `20260919-182327-1c1a04ba` (Slurm 1775, resumed as 1776) ran the same fixed-trajectory ladder at 300 keV, 5 degrees, 1 mm, and $N_e=2000$ for the nine suspected worst offenders. The fine ladder used spacings from 12 to 0.09375 eV.

Three materials still had bundled 3 eV rows. Against the 0.09375 eV reference, hBN had 6.48% yield error and a 100.69 eV centroid shift; diamond had 0.154% and 12.17 eV; black phosphorus had 1.26% and 1.88 eV. Their yield/centroid acceptances were 0.375/0.749 eV, 1.500/3.000 eV, and 1.496/1.496 eV, respectively. Silicon, 4H-SiC, 6H-SiC, V2O5, sapphire, and TiS2 already used automatic resolution and accepted approximately 0.375 eV in the aggregate gate.

The audit supports [ADR-0013](../../adr/0013-automatic-line-grids-by-default.md): remove every bundled stored row and artifact reference, and route all bundled profiles through the existing `kinematic-ceiling` plus `sinc-nyquist` policy. The artifact schema remains available as an opt-in performance mechanism.

## What this row does not claim

- No default changes. `DEFAULT_RTOL`, `DEFAULT_MAX_SPACING_EV`, and `DEFAULT_BACKEND_SAFETY_ULPS` are unchanged.
- It does not claim the line spectra are physically correct, only how they respond to grid refinement and to backend precision.
- It does not claim that opt-in stored grids are intrinsically invalid; they require explicit ownership and revalidation when relevant physics changes.

## Independent verifier outcome

The issue #126 fresh-context audit found that the implementation matches the measurement contract, sampling derivation, and Richardson equation. Its original discrepancy exposed an incorrect acceptance criterion in the ledger: Nyquist is a sufficient exactness bound, not an empirical acceptance ceiling. The ledger and analytic anchor now state the same contract explicitly.

- **Measurement identity and RNG isolation: confirmed.** `CaseLadder` transports once and evaluates every rung through the production spectrum reductions. `evaluate_ladder` hashes every spectrum-consumed segment field before the first rung and after every evaluation, and resumed runs compare the new fixed-seed transport with the stored count and digest. The case-grid anchor compares line, characteristic, and bremsstrahlung arrays bit for bit with the runner spectrum phase. `simulate_trajectories` has no photon-grid argument: its stochastic inputs derive from `seed`, `SeedSequence(seed)` children, or SplitMix64 keys formed from seed and electron id with draws addressed by counter. `_transport_case` calls transport before `resolve_line_grid`. The grid-isolation anchor reproduces the segment fingerprint after replacing the line grid.
- **Band limit and acceptance: confirmed.** The independent triangular-support derivation above gives exact infinite-grid trapezoid area for $h\leq\pi/a_w$. Phase-spread aliases may average out above that sufficient bound, so the anchor explicitly requires an above-Nyquist accepted rung to agree with the analytic integral while the grossly aliased 12 eV triple fails.
- **Richardson gate: confirmed.** `_observable_verdict` implements the two clauses of {eq}`eq-line-grid-richardson-gate` literally, including exact-zero acceptance, the $0.1\tau$ escape, and the finest-rung yield floor. `_coarsest_consistent` accepts only a suffix of passing triples, so one lucky coarse pair cannot pass. The focused anchors cover a growing second change, two changes below $0.1\tau$, exact zero, the near-zero floor, and a later failing triple. This algorithm can change an earlier acceptance when newly added finer data fail; the narrower six-versus-eight-rung stability claim is empirical and therefore depends on the stored measurements.
- **One-transport precision instrument: confirmed.** `cmd_transport` serializes one case transport and fingerprint. Each precision process constructs its `CaseLadder` from that transport without rerunning Monte Carlo and rejects a fingerprint mismatch. `compare` requires the float64/float32 pairing and identical fingerprints, windows, ratios, and grids. The CPU control anchor runs both sides through this pipeline and obtains exactly zero deviation.
- **Historical stored-number audit: retired.** No ladder checkpoint JSON, precision JSON/NPZ/pickle, or issue #125 audit artifact is tracked in the repository or present in the validation worktree. The write-up names remote jobs and the remote output `line_grid_precision_109.json`, but does not retain the source rungs alongside the ledger. Consequently the case tables, six-versus-eight acceptances, catalog-grid errors, resource timings, float32 deviation bounds and slopes, 20 keV search, and issue #125 material percentages cannot be recomputed independently from repository evidence. Their internal summaries are not a substitute for artifact comparison. Those values are therefore retained only as non-validating provenance and excluded from this row's claim.

**Verdict: anchored.** The code-level instrument, governing equations, and Nyquist-as-sufficient acceptance semantics match focused regression anchors. Retired remote measurements make no validated quantitative claim. Human sign-off remains pending.
