# `transport-midpoint-stopping`

Ledger row: [`transport-midpoint-stopping`](../physics-validation-ledger.md).
Code: `montecarlo/transport.py::simulate_trajectories` (`energy_model`) and
`::_transport_core_ungrooved`.

## Claim

A physical flight of length `s` starting at energy `E_start` ends at

```
E_end = E_start + (dE/ds)((E_start + E_end)/2) · s
```

and advances the relative-age transport clock by

```
Δt = s / β((E_start + E_end)/2)          [Å, c = 1]
```

replacing the left-endpoint ("frozen") rule `E_end = E_start + (dE/ds)(E_start)·s`,
`Δt = s/β(E_start)` that held both coefficients at the flight's start energy.

## Governing equations and where they come from

No new physical law enters. The stopping law is the existing Joy–Luo CSDA
expression already ledgered under `electron-transport`,

```
dE/ds (E) = -(7.85e-4 / E) · Σ_i (n_i/0.602214076) Z_i · ln(1.166 (E + k_i J_i)/J_i)
```

in keV/Å, with `k_i = 0.731 + 0.0688 log10 Z_i`, and the clock is the elementary
kinematic integral

```
t(s) = ∫_0^s ds' / β(E(s')),   β(E) = sqrt(1 - (1 + E/511.0)^-2).
```

Both are initial-value problems in the path length `s`. The frozen rule is the
explicit Euler discretization of each with a single step of size `s`; this claim
replaces it with the midpoint rule for both.

## Discretization and its evaluation

The energy update is the *implicit* midpoint rule, evaluated by one
predictor-corrector pass, which is the standard explicit midpoint (RK2) scheme:

```
E_pred = E_start + (dE/ds)(E_start) · s
E_mid  = (E_start + E_pred)/2
E_end  = E_start + (dE/ds)(E_mid) · s
```

The clock uses the midpoint quadrature of `∫ ds'/β` with the *corrected*
representative energy `(E_start + E_end)/2`, so a flight carries one
representative energy rather than two.

### Cutoff truncation

A flight truncated by the population energy floor `E_cut` has `E_end = E_cut` by
construction, so its midpoint energy is `(E_start + E_cut)/2` exactly and the
truncation distance is not an extrapolation but the exact inverse of the scheme:

```
s_cut = (E_cut - E_start) / (dE/ds)((E_start + E_cut)/2).
```

The frozen rule instead uses `(dE/ds)(E_start)`. Because `|dE/ds|` grows as `E`
falls over the whole range above the cutoff, `|(dE/ds)(E_mid)| ≥ |(dE/ds)(E_start)|`
and therefore `s_cut(midpoint) ≤ s_cut(frozen)`: the frozen rule overshoots the
electron's true residual range.

That same inequality is what keeps the corrector well posed. For any accepted
step `s ≤ s_cut`,

```
E_pred = E_start + (dE/ds)(E_start)·s
       ≥ E_start + [(dE/ds)(E_start) / (dE/ds)((E_start+E_cut)/2)] · (E_cut - E_start)
       ≥ E_cut,
```

so the predictor never falls below the cutoff and the Joy–Luo logarithm is never
evaluated outside its range. No clamp is needed and none is applied.

## Assumptions and limits of validity

- The Joy–Luo law is continuous slowing down: no straggling, no discrete
  inelastic events. Unchanged from `electron-transport`.
- Elastic hazard remains frozen at `E_start`; the collision distance is still
  sampled from the start-energy mean free path. Only stopping and the clock are
  controlled here. Convergent hazard treatment is checklist step F of
  `feature/energy-controlled-electron-transport`.
- `|dE/ds|` monotone in `E` is used only for the cutoff-overshoot direction and
  the predictor bound above, and only over `E > E_cut`. Below
  `E ≈ J/1.166 − kJ` the Joy–Luo logarithm changes sign and the law itself is
  invalid; the cutoff keeps transport out of that region.
- The flight decomposition is untouched, so this is not yet the energy-limited
  substepping of checklist step F.

## Limiting cases

- `s → 0`: `E_mid → E_start`, so both the energy and clock updates reduce to the
  frozen rule, and the difference between the two schemes vanishes as `s³`.
- `dE/ds` constant in `E` (a hypothetical energy-independent stopping power):
  `(dE/ds)(E_mid) = (dE/ds)(E_start)` identically and the two schemes agree
  exactly, as does the clock when `β` is likewise constant.
- `energy_model="frozen"` (the default) is bit-for-bit identical to the
  pre-change core, pinned by
  `tests/montecarlo/test_transport_energy_model.py::test_frozen_energy_model_is_the_bit_for_bit_default`.

## Numerical evidence

One boundary-truncated carbon flight (`composition=[("C", 0.1136)]`,
`elastic_model="sr"`, `E0 = 25 keV`, `seed=11`), so that the row length is the
slab thickness exactly, compared against a 20 000-substep RK4 reference for both
`E(s)` and `t(s)`. Because one flight is one step, the quantity measured is the
*local* truncation error, which is one order higher than the global order.

| flight length | fractional loss | midpoint `E_end` err [keV] | frozen `E_end` err [keV] | midpoint clock err [Å] | frozen clock err [Å] |
|---|---|---|---|---|---|
| 600 Å | 0.51% | 5.785e-07 | 2.668e-04 | 1.161e-04 | 2.347e+00 |
| 300 Å | 0.25% | 7.199e-08 | 6.655e-05 | 1.511e-05 | 5.855e-01 |
| 150 Å | 0.13% | 8.978e-09 | 1.662e-05 | 1.925e-06 | 1.462e-01 |

Error ratios under halving: midpoint 8.04 / 8.02 (energy) and 7.69 / 7.85
(clock) against the expected 8 for an `O(s³)` local error; frozen 4.01 / 4.00
and 4.01 / 4.00 against the expected 4. At 600 Å the midpoint energy error is
0.22% of the frozen error and the midpoint clock error is 5e-05 of it.

Pinned by `tests/montecarlo/test_transport_energy_model.py`:
`::test_midpoint_end_state_error_is_third_order_in_flight_length`,
`::test_midpoint_end_energy_and_clock_beat_the_frozen_rule`,
`::test_midpoint_cutoff_flight_stops_exactly_on_the_energy_floor`,
`::test_midpoint_cutoff_distance_is_shorter_than_the_frozen_extrapolation`.

## What this row does not claim

- Nothing about emitted spectra. Line and bremsstrahlung kernels still read the
  flight-start energy; migrating them to an explicit representative energy and
  proving substep invariance are checklist steps D and G.
- Nothing about the grooved, per-electron, or CUDA cores. A midpoint request on
  any of them raises rather than silently returning the frozen schema; the port
  is checklist step H.
- `_clip_segments_to_cutoff` drops `E_end_keV`/`t_end_ang` when it shortens a
  flight, because its left-endpoint clip rule cannot reconstruct a
  midpoint-integrated end state for the shortened flight. That drop, and the
  fact that the new fields are masked in step with the rows they belong to, are
  pinned by
  `::test_row_transforms_keep_the_new_fields_in_step_with_the_rows`.
