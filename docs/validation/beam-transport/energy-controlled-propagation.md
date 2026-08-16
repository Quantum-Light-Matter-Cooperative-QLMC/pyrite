# `energy-controlled-propagation`

Ledger row: [`energy-controlled-propagation`](../ledger-transport-background.md#energy-controlled-propagation).
Code: `montecarlo/transport.py::simulate_trajectories` (`max_dE_frac`),
`::_transport_core_ungrooved`, `::_transport_core_ungrooved_lut`.
Measurement: `checks/collision_statistics_refinement.py`.

## Claim

A physical flight may be split into numerical substeps of at most `f`
fractional energy loss (`max_dE_frac=f`, which requires
`energy_model="midpoint"`) without changing the flight's *physical* identity or
the statistics of where its terminating collision occurs.

The elastic collision is drawn once per physical flight as an optical depth

```
tau = -ln(U),        U ~ Uniform(0, 1]
```

and consumed across that flight's substeps at each substep's own hazard,

```
tau  <-  tau - ds / lambda(E_substep),
```

the flight closing when the budget is exhausted (collision), at a boundary, at
the cutoff, or on termination. A substep that ends because the energy cap bound
first keeps the flight's direction, `flight_id`, and remaining `tau`, increments
`substep_id`, and does not scatter.

## Why this discretization and not an inversion

The exact statement of the inhomogeneous collision law along a flight is that
the collision occurs at the path length `s*` solving

```
int_0^{s*} ds / lambda(E(s))  =  -ln(U),
```

because the survival probability along a path of varying hazard is
`exp[-int ds/lambda(E(s))]` and inverting a Uniform(0,1] through it is the
standard inverse-CDF sampler. Browning/Mott `lambda(E)` is tabulated, so the
integral has no closed form to invert. The optical-depth budget is the
left-endpoint (per-substep) discretization of exactly that integral: the *total*
optical depth is accumulated over the flight, and only the hazard *within one
substep* is held constant. It is therefore consistent with the exact sampler and
first order in the substep size, not a bounded-piecewise-constant approximation
of a different quantity.

Two properties fall out and are relied on elsewhere:

- **Reduction.** With `f = 0` a flight is one substep, `tau` is consumed in a
  single term, and the sampled distance is `tau * lambda(E_start)`. The
  historical draw was `-lambda * ln(U)`. The implementation computes
  `(-ln(U)) * lambda`, which is the same product in the same floating-point
  order, so frozen mode is **bit-for-bit unchanged** (verified by hashing
  `r_mid`, `v_hat`, `L_ang`, `E_keV`, `t_ang`, `elec_id`, `layer` and the exit
  counts over C/W × 25/100 keV against `main`).
- **No resampling.** Refinement never draws additional randomness. There is one
  `U` per physical flight at every `f`, so refining the cap cannot create or
  destroy a collision event by consuming a different number of variates.

## Limiting case

A lossless flight has `lambda(E_substep) = lambda(E_start)` for every substep,
so the budget telescopes: `sum_k ds_k / lambda(E_start) = s/lambda(E_start)`,
identical to the unrefined draw for any subdivision. All refinement error is
therefore carried by the variation of `lambda` along the flight and vanishes
with the flight's fractional energy loss, not with the substep count as such.

## What refinement is expected to do

Refinement is **not** expected to leave the sampled collision point pointwise
unchanged. Holding `lambda` at `E_start` for a whole flight understates the
hazard, because the Browning cross section rises as `E` falls, so the unrefined
draw places the collision slightly too far away and overstates the mean physical
flight length. The bias is first order in the flight's fractional loss.

Refinement also **decorrelates trajectories**. Changing the substep grid changes
the energies at which the hazard is evaluated, which moves the sampled collision
point, which changes every downstream flight of that electron. A single-seed
flight count is consequently not a measurement of anything: the slice-F notes
recorded 2568 / 2882 / 2599 / 2553 / 2605 distinct flights down the ladder for
one seed (C, 25 keV, 4000 Å, Ne = 200, seed 7), a spread that is pure
realization noise.

The claim under test is therefore the ensemble one: at production step sizes the
refinement bias is inside Monte Carlo error, so there is no rung at which
collision statistics are still moving.

## Measurement

`checks/collision_statistics_refinement.py`, Ne = 1000 × 12 independent seeds
per rung (12 000 electrons), `E_cut = 1 keV`, `energy_model="midpoint"`,
lockstep core. Ladder `f =` unrefined (one row per flight), 1%, 0.5%, 0.2%,
0.1%, 0.05%.

Observables per electron: physical flights (rows with `substep_id == 0`, which
in an ungrooved single-layer slab is the collision count plus the terminal
closure), total path, energy and clock at the last row, and the exit-channel
fractions. `L_flight` is the intensive view of the same statistic — ensemble
total path divided by ensemble total flights, i.e. the sampled elastic mean free
path, which is what the optical-depth budget controls most directly.

Each rung's mean is quoted with the standard error of the mean over seed
replicates, which assumes no variance model. Rungs are compared by the **paired**
per-seed difference, which cancels any variance still common to two rungs
sharing a seed and is therefore strictly more sensitive to a systematic bias
than combining the two standard errors; it degrades to the unpaired test when
the rungs are fully decorrelated.

### C 25 keV, 4000 Å (the slice-F single-seed case)

| `f` | flights/e | `L_flight` [Å] | path/e [Å] | `E_ret` [keV] | trans |
| --- | --- | --- | --- | --- | --- |
| none | 13.12 ± 0.075 | 321.8 ± 0.66 | 4221 ± 19 | 24.090 ± 0.005 | 0.9942 ± 0.0005 |
| 1e-2 | 13.05 ± 0.040 | 322.3 ± 0.61 | 4205 ± 13 | 24.100 ± 0.003 | 0.9938 ± 0.0006 |
| 5e-3 | 13.13 ± 0.072 | 320.9 ± 1.1 | 4212 ± 16 | 24.100 ± 0.004 | 0.9947 ± 0.0007 |
| 2e-3 | 13.11 ± 0.049 | 321.0 ± 0.95 | 4209 ± 13 | 24.100 ± 0.003 | 0.9948 ± 0.0007 |
| 1e-3 | 13.03 ± 0.047 | 321.5 ± 0.80 | 4187 ± 11 | 24.100 ± 0.003 | 0.9944 ± 0.0010 |
| 5e-4 | 13.06 ± 0.056 | 321.2 ± 0.92 | 4195 ± 15 | 24.100 ± 0.004 | 0.9941 ± 0.0007 |

Largest paired shift against `f = 5e-4`: 1.57σ.

### C 25 keV thick (2×10⁴ Å)

| `f` | flights/e | `L_flight` [Å] | path/e [Å] | `E_ret` [keV] | back | stop |
| --- | --- | --- | --- | --- | --- | --- |
| none | 96.30 ± 0.52 | 278.1 ± 0.88 | 26780 ± 70 | 18.19 ± 0.03 | 0.0453 ± 0.0023 | 0.0488 ± 0.0018 |
| 1e-2 | 96.45 ± 0.80 | 277.9 ± 1.2 | 26790 ± 110 | 18.18 ± 0.05 | 0.0409 ± 0.0014 | 0.0499 ± 0.0028 |
| 5e-3 | 96.29 ± 0.49 | 278.0 ± 0.73 | 26770 ± 74 | 18.20 ± 0.03 | 0.0420 ± 0.0014 | 0.0488 ± 0.0013 |
| 2e-3 | 97.15 ± 0.64 | 276.4 ± 0.98 | 26850 ± 85 | 18.16 ± 0.04 | 0.0443 ± 0.0022 | 0.0513 ± 0.0017 |
| 1e-3 | 96.32 ± 0.50 | 277.3 ± 0.76 | 26700 ± 85 | 18.22 ± 0.03 | 0.0443 ± 0.0021 | 0.0478 ± 0.0015 |
| 5e-4 | 97.45 ± 0.72 | 275.9 ± 0.99 | 26880 ± 110 | 18.14 ± 0.04 | 0.0444 ± 0.0024 | 0.0507 ± 0.0023 |

Largest paired shift against `f = 5e-4`: **2.25σ**, on `L_flight` at the
unrefined rung — the matrix maximum, in the case with the largest per-flight
fractional loss, with the sign the bias argument predicts (unrefined flights are
longer). It is not resolved at 12 000 electrons.

### W 25 keV thick (5×10³ Å), backscatter-dominated

| `f` | flights/e | `L_flight` [Å] | path/e [Å] | back | stop |
| --- | --- | --- | --- | --- | --- |
| none | 610.7 ± 3.1 | 15.60 ± 0.016 | 9528 ± 41 | 0.5573 ± 0.0033 | 0.4007 ± 0.0037 |
| 1e-2 | 604.9 ± 2.7 | 15.64 ± 0.015 | 9460 ± 39 | 0.5636 ± 0.0024 | 0.3931 ± 0.0027 |
| 5e-3 | 605.4 ± 6.3 | 15.62 ± 0.027 | 9453 ± 84 | 0.5589 ± 0.0063 | 0.3968 ± 0.0064 |
| 2e-3 | 607.9 ± 3.8 | 15.61 ± 0.019 | 9490 ± 50 | 0.5584 ± 0.0045 | 0.3970 ± 0.0039 |
| 1e-3 | 600.5 ± 3.3 | 15.64 ± 0.019 | 9390 ± 46 | 0.5704 ± 0.0036 | 0.3891 ± 0.0034 |
| 5e-4 | 607.4 ± 4.3 | 15.61 ± 0.019 | 9480 ± 60 | 0.5619 ± 0.0042 | 0.3958 ± 0.0038 |

Largest paired shift against `f = 5e-4`: 2.21σ (`back`, at `f = 1e-3`, with no
monotone pattern down the ladder).

### C 100 keV thick (2×10⁵ Å)

| `f` | flights/e | `L_flight` [Å] | path/e [Å] | `E_ret` [keV] | trans |
| --- | --- | --- | --- | --- | --- |
| none | 216.5 ± 1.9 | 1187 ± 5.5 | 2.568e5 ± 1.2e3 | 80.85 ± 0.15 | 0.9432 ± 0.0024 |
| 1e-2 | 215.0 ± 2.4 | 1190 ± 5.7 | 2.556e5 ± 1.6e3 | 80.97 ± 0.19 | 0.9474 ± 0.0025 |
| 5e-3 | 217.5 ± 1.5 | 1184 ± 3.6 | 2.573e5 ± 9.8e2 | 80.77 ± 0.12 | 0.9436 ± 0.0023 |
| 2e-3 | 218.8 ± 2.3 | 1181 ± 5.5 | 2.582e5 ± 1.6e3 | 80.68 ± 0.18 | 0.9423 ± 0.0023 |
| 1e-3 | 215.2 ± 1.5 | 1187 ± 3.9 | 2.554e5 ± 9.8e2 | 80.97 ± 0.11 | 0.9463 ± 0.0018 |
| 5e-4 | 218.4 ± 2.0 | 1181 ± 5.4 | 2.579e5 ± 1.2e3 | 80.71 ± 0.15 | 0.9439 ± 0.0024 |

Largest paired shift against `f = 5e-4`: 1.74σ.

## Result

Over 4 cases × 5 rungs × 8 observables = 160 paired comparisons, **every** shift
against the finest rung is below 2.3σ and none reaches the 3σ flag. Refining
`max_dE_frac` by a factor of 20 below the unrefined draw does not move any
collision statistic beyond Monte Carlo error at 12 000 electrons per rung.

Two qualifications belong with that number:

- The measurement is a **null result at a stated resolution**, not a proof of
  invariance. The predicted bias is real and one-signed; it is simply smaller
  than ~2σ ≈ 1% of `L_flight` here. The case that comes closest (C 25 keV thick,
  2.25σ on `L_flight`, unrefined vs finest, correct sign) is the case slice E
  already identified as having the matrix's largest per-flight fractional loss.
  A larger ensemble would be expected to resolve it as a small convergent bias,
  not to contradict it.
- The paired and unpaired tests differ by at most ~0.7σ and the paired shifts
  are the larger ones, so a little variance does survive the shared seed and the
  paired test is the more sensitive of the two — which is why it is the one
  reported. The margin is small, though, which is itself consistent with the
  decorrelation claim: most of a rung's variance is genuinely independent of the
  other rung's even at the same seed.

## Assumptions and limits

- Ungrooved, single-layer slab, lockstep core, `energy_model="midpoint"`. The
  grooved, per-electron, and CUDA cores still raise on a midpoint request
  (checklist step H).
- `flights/e` equals collisions plus one only because a single-layer ungrooved
  slab has no internal boundary that closes a flight; in a multilayer stack the
  boundary crossings enter that count.
- The claim covers *collision statistics*. Substep invariance of the emitted CXR
  and bremsstrahlung is a separate claim, `substep-radiation-invariance`; this
  row must not be read as saying anything about the spectra.
- Energy-loss straggling remains unmodeled (see `energy-step-convergence`), so
  the hazard is evaluated along a deterministic CSDA energy history.
