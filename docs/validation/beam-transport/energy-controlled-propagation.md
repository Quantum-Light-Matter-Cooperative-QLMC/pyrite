# `energy-controlled-propagation`

Ledger row: [`energy-controlled-propagation`](../ledger-transport-background.md#energy-controlled-propagation). Code: `montecarlo/transport/api.py::simulate_trajectories` (`max_dE_frac`); host cores in `montecarlo/transport/cores.py`; exact CUDA core in `montecarlo/transport/_jit_kernel.py`. Measurement: `checks/collision_statistics_refinement.py`.

## Claim

A physical flight may be split into numerical substeps whose left-endpoint predicted mean loss is at most the fraction `f` (`max_dE_frac=f`, which requires `energy_model="midpoint"`) without changing the flight's *physical* identity or the statistics of where its terminating collision occurs.

This is not a strict bound on the realized loss. The deterministic midpoint update is slightly larger when stopping rises as the electron slows, and an Urban compound-Poisson draw is unbounded. The control fixes a deterministic quadrature grid through $s_{\rm cap}=fE_{\rm start}/C(E_{\rm start})$.

The elastic collision is drawn once per physical flight as an optical depth

$$
\tau=-\ln U,
\qquad U\sim\operatorname{Uniform}(0,1].
$$

and consumed across that flight's substeps at each substep's own hazard,

$$
\tau\leftarrow\tau-\frac{ds}{\lambda(E_{\rm substep})}.
$$

the flight closing when the budget is exhausted (collision), at a boundary, at the cutoff, or on termination. A substep that ends because the energy cap bound first keeps the flight's direction, `flight_id`, and remaining `tau`, increments `substep_id`, and does not scatter.

## Why this discretization and not an inversion

The exact statement of the inhomogeneous collision law along a flight is that the collision occurs at the path length `s*` solving

$$
\int_0^{s^*}\frac{ds}{\lambda(E(s))}=-\ln U.
$$

because the survival probability along a path of varying hazard is `exp[-int ds/lambda(E(s))]` and inverting a Uniform(0,1] through it is the standard inverse-CDF sampler. Browning/Mott `lambda(E)` is tabulated, so the integral has no closed form to invert. The optical-depth budget is the left-endpoint (per-substep) discretization of exactly that integral: the *total* optical depth is accumulated over the flight, and only the hazard *within one substep* is held constant. It is therefore consistent with the exact sampler and first order in the substep size, not a bounded-piecewise-constant approximation of a different quantity.

Two properties fall out and are relied on elsewhere:

- **Reduction.** With `f = 0` a flight is one substep, `tau` is consumed in a single term, and the sampled distance is `tau * lambda(E_start)`. The historical draw was `-lambda * ln(U)`. The implementation computes `(-ln(U)) * lambda`, which is the same product in the same floating-point order, so frozen mode is **bit-for-bit unchanged** (verified by hashing `r_mid`, `v_hat`, `L_ang`, `E_keV`, `t_ang`, `elec_id`, `layer` and the exit counts over C/W × 25/100 keV against `main`).
- **No collision resampling.** Refinement never draws another elastic-collision variate. There is one collision $U$ per physical flight at every $f$, so refining the cap cannot create or destroy a collision event by shifting that stream. Optional straggling has a separate per-substep stream.

## Limiting case

A lossless flight has `lambda(E_substep) = lambda(E_start)` for every substep, so the budget telescopes: `sum_k ds_k / lambda(E_start) = s/lambda(E_start)`, identical to the unrefined draw for any subdivision. All refinement error is therefore carried by the variation of `lambda` along the flight and vanishes with the flight's fractional energy loss, not with the substep count as such.

## What refinement is expected to do

Refinement is **not** expected to leave the sampled collision point pointwise unchanged. Holding `lambda` at `E_start` for a whole flight understates the hazard, because the Browning cross section rises as `E` falls, so the unrefined draw places the collision slightly too far away and overstates the mean physical flight length. The bias is first order in the flight's fractional loss.

Refinement also **decorrelates trajectories**. Changing the substep grid changes the energies at which the hazard is evaluated, which moves the sampled collision point, which changes every downstream flight of that electron. A single-seed flight count is consequently not a measurement of anything: the slice-F notes recorded 2568 / 2882 / 2599 / 2553 / 2605 distinct flights down the ladder for one seed (C, 25 keV, 4000 Å, Ne = 200, seed 7), a spread that is pure realization noise.

The claim under test is therefore the ensemble one: at production step sizes the refinement bias is inside Monte Carlo error, so there is no rung at which collision statistics are still moving.

## Measurement

`checks/collision_statistics_refinement.py`, Ne = 1000 × 12 independent seeds per rung (12 000 electrons), `E_cut = 1 keV`, `energy_model="midpoint"`, lockstep core. Ladder `f =` unrefined (one row per flight), 1%, 0.5%, 0.2%, 0.1%, 0.05%.

Observables per electron: physical flights (rows with `substep_id == 0`, which in an ungrooved single-layer slab is the collision count plus the terminal closure), total path, energy and clock at the last row, and the exit-channel fractions. `L_flight` is the intensive view of the same statistic — ensemble total path divided by ensemble total flights, i.e. the sampled elastic mean free path, which is what the optical-depth budget controls most directly.

Each rung's mean is quoted with the standard error of the mean over seed replicates, which assumes no variance model. Rungs are compared by the **paired** per-seed difference, which cancels any variance still common to two rungs sharing a seed and is therefore strictly more sensitive to a systematic bias than combining the two standard errors; it degrades to the unpaired test when the rungs are fully decorrelated.

### C 25 keV, 4000 Å (the slice-F single-seed case)


| `f`  | flights/e      | `L_flight` [Å] | path/e [Å] | `E_ret` [keV]   | trans            |
| ------ | ---------------- | ----------------- | ------------- | ----------------- | ------------------ |
| none | 13.12 ± 0.075 | 321.8 ± 0.66   | 4221 ± 19  | 24.090 ± 0.005 | 0.9942 ± 0.0005 |
| 1e-2 | 13.05 ± 0.040 | 322.3 ± 0.61   | 4205 ± 13  | 24.100 ± 0.003 | 0.9938 ± 0.0006 |
| 5e-3 | 13.13 ± 0.072 | 320.9 ± 1.1    | 4212 ± 16  | 24.100 ± 0.004 | 0.9947 ± 0.0007 |
| 2e-3 | 13.11 ± 0.049 | 321.0 ± 0.95   | 4209 ± 13  | 24.100 ± 0.003 | 0.9948 ± 0.0007 |
| 1e-3 | 13.03 ± 0.047 | 321.5 ± 0.80   | 4187 ± 11  | 24.100 ± 0.003 | 0.9944 ± 0.0010 |
| 5e-4 | 13.06 ± 0.056 | 321.2 ± 0.92   | 4195 ± 15  | 24.100 ± 0.004 | 0.9941 ± 0.0007 |

Largest paired shift against `f = 5e-4`: 1.57σ.

### C 25 keV thick (2×10⁴ Å)


| `f`  | flights/e     | `L_flight` [Å] | path/e [Å]  | `E_ret` [keV] | back             | stop             |
| ------ | --------------- | ----------------- | -------------- | --------------- | ------------------ | ------------------ |
| none | 96.30 ± 0.52 | 278.1 ± 0.88   | 26780 ± 70  | 18.19 ± 0.03 | 0.0453 ± 0.0023 | 0.0488 ± 0.0018 |
| 1e-2 | 96.45 ± 0.80 | 277.9 ± 1.2    | 26790 ± 110 | 18.18 ± 0.05 | 0.0409 ± 0.0014 | 0.0499 ± 0.0028 |
| 5e-3 | 96.29 ± 0.49 | 278.0 ± 0.73   | 26770 ± 74  | 18.20 ± 0.03 | 0.0420 ± 0.0014 | 0.0488 ± 0.0013 |
| 2e-3 | 97.15 ± 0.64 | 276.4 ± 0.98   | 26850 ± 85  | 18.16 ± 0.04 | 0.0443 ± 0.0022 | 0.0513 ± 0.0017 |
| 1e-3 | 96.32 ± 0.50 | 277.3 ± 0.76   | 26700 ± 85  | 18.22 ± 0.03 | 0.0443 ± 0.0021 | 0.0478 ± 0.0015 |
| 5e-4 | 97.45 ± 0.72 | 275.9 ± 0.99   | 26880 ± 110 | 18.14 ± 0.04 | 0.0444 ± 0.0024 | 0.0507 ± 0.0023 |

Largest paired shift against `f = 5e-4`: **2.25σ**, on `L_flight` at the unrefined rung — the matrix maximum, in the case with the largest per-flight fractional loss, with the sign the bias argument predicts (unrefined flights are longer). It is not resolved at 12 000 electrons.

### W 25 keV thick (5×10³ Å), backscatter-dominated


| `f`  | flights/e    | `L_flight` [Å] | path/e [Å] | back             | stop             |
| ------ | -------------- | ----------------- | ------------- | ------------------ | ------------------ |
| none | 610.7 ± 3.1 | 15.60 ± 0.016  | 9528 ± 41  | 0.5573 ± 0.0033 | 0.4007 ± 0.0037 |
| 1e-2 | 604.9 ± 2.7 | 15.64 ± 0.015  | 9460 ± 39  | 0.5636 ± 0.0024 | 0.3931 ± 0.0027 |
| 5e-3 | 605.4 ± 6.3 | 15.62 ± 0.027  | 9453 ± 84  | 0.5589 ± 0.0063 | 0.3968 ± 0.0064 |
| 2e-3 | 607.9 ± 3.8 | 15.61 ± 0.019  | 9490 ± 50  | 0.5584 ± 0.0045 | 0.3970 ± 0.0039 |
| 1e-3 | 600.5 ± 3.3 | 15.64 ± 0.019  | 9390 ± 46  | 0.5704 ± 0.0036 | 0.3891 ± 0.0034 |
| 5e-4 | 607.4 ± 4.3 | 15.61 ± 0.019  | 9480 ± 60  | 0.5619 ± 0.0042 | 0.3958 ± 0.0038 |

Largest paired shift against `f = 5e-4`: 2.21σ (`back`, at `f = 1e-3`, with no monotone pattern down the ladder).

### C 100 keV thick (2×10⁵ Å)


| `f`  | flights/e    | `L_flight` [Å] | path/e [Å]      | `E_ret` [keV] | trans            |
| ------ | -------------- | ----------------- | ------------------ | --------------- | ------------------ |
| none | 216.5 ± 1.9 | 1187 ± 5.5     | 2.568e5 ± 1.2e3 | 80.85 ± 0.15 | 0.9432 ± 0.0024 |
| 1e-2 | 215.0 ± 2.4 | 1190 ± 5.7     | 2.556e5 ± 1.6e3 | 80.97 ± 0.19 | 0.9474 ± 0.0025 |
| 5e-3 | 217.5 ± 1.5 | 1184 ± 3.6     | 2.573e5 ± 9.8e2 | 80.77 ± 0.12 | 0.9436 ± 0.0023 |
| 2e-3 | 218.8 ± 2.3 | 1181 ± 5.5     | 2.582e5 ± 1.6e3 | 80.68 ± 0.18 | 0.9423 ± 0.0023 |
| 1e-3 | 215.2 ± 1.5 | 1187 ± 3.9     | 2.554e5 ± 9.8e2 | 80.97 ± 0.11 | 0.9463 ± 0.0018 |
| 5e-4 | 218.4 ± 2.0 | 1181 ± 5.4     | 2.579e5 ± 1.2e3 | 80.71 ± 0.15 | 0.9439 ± 0.0024 |

Largest paired shift against `f = 5e-4`: 1.74σ.

## Result

Over 4 cases × 5 rungs × 8 observables = 160 paired comparisons, **every** shift against the finest rung is below 2.3σ and none reaches the 3σ flag. Refining `max_dE_frac` by a factor of 20 below the unrefined draw does not move any collision statistic beyond Monte Carlo error at 12 000 electrons per rung.

Two qualifications belong with that number:

- The measurement is a **null result at a stated resolution**, not a proof of invariance. The predicted bias is real and one-signed; it is simply smaller than ~2σ ≈ 1% of `L_flight` here. The case that comes closest (C 25 keV thick, 2.25σ on `L_flight`, unrefined vs finest, correct sign) is the case slice E already identified as having the matrix's largest per-flight fractional loss. A larger ensemble would be expected to resolve it as a small convergent bias, not to contradict it.
- The paired and unpaired tests differ by at most ~0.7σ and the paired shifts are the larger ones, so a little variance does survive the shared seed and the paired test is the more sensitive of the two — which is why it is the one reported. The margin is small, though, which is itself consistent with the decorrelation claim: most of a rung's variance is genuinely independent of the other rung's even at the same seed.

## Assumptions and limits

- Ungrooved, single-layer slab, lockstep core, `energy_model="midpoint"`. The grooved, per-electron, and CUDA cores have since been ported by checklist step H, which also dropped the fail-closed midpoint gates, so none of them raises on a midpoint request; the measurement below remains lockstep.
- `flights/e` equals collisions plus one only because a single-layer ungrooved slab has no internal boundary that closes a flight; in a multilayer stack the boundary crossings enter that count.
- The claim covers *collision statistics*. Substep invariance of the emitted CXR and bremsstrahlung is a separate claim, `substep-radiation-invariance`; this row must not be read as saying anything about the spectra.
- Energy-loss straggling is modeled optionally. The collision optical-depth construction validated here is unchanged; when enabled, the hazard is evaluated along the realized Urban-loss history. The loss stream and its distributional substep semantics are validated by `energy-loss-straggling`.

## Independent verification

Fresh-context rederivation (2026-08-18, verifier context separate from the implementation).

**Filters.** Units pass: $\tau$ and $ds/\lambda(E)$ are both dimensionless, `lam_ang` is a length. Limits pass: at `max_dE_frac = 0` the cap branch (`energy_controlled = max_dE_frac > 0.0`) is never taken, so every flight is a single row and `step_j = tau_left[e] * lam_ang` evaluates $(-\ln U)\cdot\lambda(E_{\rm start})$. IEEE-754 multiplication is exactly commutative and unary negation only flips the sign bit, so $(-\ln U)\cdot\lambda \equiv -(\lambda\cdot(-\ln U)) \equiv -\lambda\ln U$ is bit-identical to the historical draw for any representable $\lambda,\ln U$ — this holds by construction, independent of the doc's reported hash-vs-`main` check. Sign/convention passes: Browning's total elastic cross section (ledger `electron-transport`, screening parameter $\alpha=3.4\times10^{-3}Z^{0.67}/E$) falls with rising $E$, so $\lambda=10^8/\Sigma n\sigma$ *rises* with $E$; since a flight's energy only decreases, freezing $\lambda$ at $E_{\rm start}$ under-weights the true (rising) hazard and lengthens the sampled flight — the claimed bias sign.

**Re-derivation (before reading the implementation body).** Model the elastic collisions along one trajectory as an inhomogeneous Poisson process in path length $s$ with local rate $\mu(s)=1/\lambda(E(s))$. Survival to $s$ is

$$
S(s)=\exp\!\left[-\int_0^{s}\mu(s')\,ds'\right]
=\exp\!\left[-\int_0^{s}\frac{ds'}{\lambda(E(s'))}\right].

$$

$S(s^*)$ is itself $\mathrm{Uniform}(0,1]$ for the random first-collision distance $s^*$ (probability integral transform), so drawing $U\sim \mathrm{Uniform}(0,1]$ and solving

$$
\int_0^{s^*}\frac{ds'}{\lambda(E(s'))}=-\ln U \equiv \tau

$$

for {math}`s^*` samples exactly one collision event, consuming exactly one uniform variate regardless of how the integral is evaluated. Substepping the integral as {math}`\tau\mathrel{-{=}}ds_k/\lambda(E_{{\rm substep},k})`, with $E_{{\rm substep},k}$ the energy at the *start* of substep $k$, is the left-endpoint (piecewise-constant-$\lambda$) Riemann sum for that same integral; it is a pure quadrature refinement of a single fixed draw, not a new sampling event — no substep consumes an RNG call, so the flight's *physical identity* (which $U$ selected it) cannot change under refinement, only the resolved location of {math}`s^*` within it.

**Comparison with the implementation** (the host cores in `montecarlo/transport/cores.py` and the CUDA core in `montecarlo/transport/_jit_kernel.py` share this structure). `tau_left[e] = -1.0` is the "no flight open" sentinel; a fresh draw `tau_left[e] = -np.log(rng.random())` fires only when `tau_left[e] < 0.0`, i.e. once per physical flight, matching $\tau=-\ln U$ exactly. Each iteration computes `total_rate` from `E_j = E_keV[e]`, the substep's own start energy (`E_keV[e]` was last set to the previous substep's `E_end_j`), giving `lam_ang` $=\lambda(E_{\rm substep})$ — the claimed left-endpoint evaluation. `step_j = tau_left[e] * lam_ang` proposes the distance to exhaust the remaining budget at that substep's hazard; boundary, cutoff, and (if `energy_controlled`) the `max_dE_frac` energy cap can each shorten it before it is committed. Every committed row unconditionally consumes `tau_left[e] -= step_j / lam_ang` — the claimed $\tau\mathrel{-{=}}ds/\lambda(E_{\rm substep})$ — clipped at zero for floating-point residue. On a cap-limited row (`limited_j`) the code increments `substep_of[e]` and `continue`s *before* the collision-draw/exit-handling block: `dirs[e]`, `flight_of[e]`, and the carried `tau_left[e]` are all left untouched, and no `rng.random()` call is reachable on that path — direction, `flight_id`, and remaining budget are preserved and no scatter is drawn, as claimed. Only a row that is not cap-limited can close the flight (`flight_of[e] += 1`, `substep_of[e] = 0`, `tau_left[e] = -1.0`), at which point the next iteration's `tau_left[e] < 0.0` check correctly triggers exactly one fresh draw. This is a term-for-term match to the derivation above.

**Statistical evidence.** Recomputing `checks/collision_statistics_refinement.py --quick` (250 electrons × 4 seeds/rung, well below the doc's Ne=1000×12) reproduces the qualitative story at reduced power — no shift over 3σ, several in the low single-digit sigma with mixed sign, consistent with the doc's framing that single-seed/low-`Ne` flight counts are noise-dominated and only the paired, replicated statistic is informative. The doc's headline numbers were re-checked arithmetically: the C-25-keV-thick `L_flight` shift (none $278.1\pm0.88$ vs finest $275.9\pm0.99\,\text{\AA}$) is one-signed with the predicted bias (unrefined longer), and $160=4\ \text{cases}\times5 \text{rungs}\times8\ \text{observables}$ is consistent with the script's full per-case metric set (`flights/e`, `L_flight`, `path/e`, `E_ret`, `clock`, `trans`, `back`, `stop`) even though the published tables display only the non-degenerate exit channels per case for brevity. The "null result at this resolution, not proof of invariance" framing is honest: the one qualification the doc could be faulted for softening — that no committed regression test independently re-derives the $f=0$ bit-for-bit identity against a frozen pre-refactor reference (`test_frozen_energy_model_is_the_bit_for_bit_default` checks default-vs-explicit-`"frozen"` `energy_model`, not the historical single-draw formula against `max_dE_frac=0`) — does not affect the physics verdict, since the identity is guaranteed by IEEE-754 multiplication commutativity independent of any particular commit.

`tests/montecarlo/test_transport_energy_model.py::test_substeps_subdivide_flights_without_scattering_or_redrawing`, `::test_substep_rows_tile_their_flight_in_length_energy_and_clock`, and `::test_step_cap_requires_the_controlled_propagator` (6/6 passing) directly pin the no-redraw/no-scatter/budget-carry bookkeeping checked above.

Verdict: `rederived`. `signed-off` remains a human decision.
