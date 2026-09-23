# `radiation-error-estimators`

Ledger row: [`radiation-error-estimators`](../physics-validation-ledger.md). Code: `montecarlo/spectrum/diagnostics.py` (`cxr_endpoint_resonance_drift`, `brem_endpoint_quadrature_error`); calibration script `checks/radiation_error_estimator_calibration.py`.

## Claim

Two host-side, opt-in estimators bound the error the radiation kernels make by evaluating every per-flight emission coefficient at the flight's start energy, before any spectrum is computed:

1. **CXR endpoint resonance drift.** The line kernel freezes `v = beta(E_start) * v_hat` for the whole flight, so each (flight, reflection) pair radiates at one resonance energy `E_res = HBARC_EV_ANG * v.g / (1 - v.n)`. The estimator evaluates `E_res` at both endpoint speeds -- each with its own Doppler denominator -- and normalizes the sweep by the flight's sinc linewidth, the half-width to the first zero of the kernel's finite-interaction-time factor `t_L sinc(a_width (E - E_res)/pi)`:

   ```
   W = 2 pi HBARC_EV_ANG / (dnm t_L),   dnm = 1 - beta(E_start) v_hat.n,
   t_L = L / beta(E_start).
   ```

A drift of one linewidth means the frozen-energy line is displaced by its own width over a single flight. The per-flight metric is the worst drift over the caller's reflection set.

2. **Bremsstrahlung endpoint quadrature error.** The brem kernel integrates each flight's emission as `n * dsigma/dk(T_start) * L`, a left-endpoint rectangle rule in the electron energy. The estimator re-evaluates the same Bethe-Heitler/Elwert integrand at the flight's midpoint energy and reports the grid-integrated (trapezoid-weighted) relative yield difference and the worst per-bin relative difference, per flight. The Beer--Lambert escape factor and `n * L` cancel in every ratio, so the estimator is geometry-independent.

Both read `E_end_keV` when transport supplied it (`energy_model="midpoint"`) and otherwise predict the end energy with the left-endpoint Joy--Luo stopping rule, the same prediction `_flight_diagnostic_summary` makes, so they apply to frozen runs unchanged. Summaries are the fixed-size p50/p90/p99/max form of the transport diagnostics; each estimator warns when its p99 exceeds a calibrated threshold (`DEFAULT_RESONANCE_DRIFT_WARN`, `DEFAULT_BREM_QUADRATURE_WARN`).

## Governing equations and where they come from

No new physics enters. `E_res` is the kernel's own resonance condition (`mc_spectrum` step 1, Zhai SI Eq. 10); the linewidth is the first-zero scale of the kernel's own `sinc` factor; the brem integrand is the ledgered `brem-spectrum` cross section. The estimators differ from the kernels only in *where along the flight* the same expressions are evaluated. The Joy--Luo end-energy prediction is the rule ledgered under `electron-transport`, applied exactly as in `transport-midpoint-stopping`.

## Assumptions and limits of validity

- The drift estimator measures the resonance *sweep*, not the resulting spectral error directly; the calibration below measures their correlation on the case matrix. Near-grazing observation directions (`dnm -> 0`) amplify both the drift and the linewidth together, which the linewidth units absorb.
- The drift estimator applies only the kernel's hard 10 eV floor, not the kernel's padded spectral-grid window, so it reports drift for flights whose line the caller's grid would skip -- a conservative superset of the radiating set, independent of any particular grid.
- Flights whose start-energy resonance falls below the kernel's hard 10 eV floor radiate nothing and are excluded.
- The brem estimator's per-bin metric excludes bins below 1e-6 of the flight's peak bin; ratios there are dominated by the kinematic cutoff `k -> T`, where both evaluations vanish.
- Both estimators inherit the kernels' approximations (isotropic brem emission, Bethe-Heitler/Elwert validity `Z <~ 30`, frozen couplings across the linewidth). They estimate the *quadrature/evaluation* error of the frozen-energy rule, never the error of the underlying cross sections.

## Limiting cases

- Lossless flight (`E_end = E_start`): both estimators are exactly zero.
- Small loss: the brem integrated error is first order in the flight's fractional energy loss (`dsigma/dk` is smooth in `T` away from the grid endpoint); halving the loss halves the error. Pinned by `tests/montecarlo/test_radiation_error_estimators.py::test_brem_quadrature_error_scales_linearly_with_energy_loss`.
- `segments` carrying a transported `E_end_keV` and the composition-predicted end state give identical estimators when the two end states agree, pinned by `::test_cxr_drift_predicts_the_frozen_end_state_from_composition`.

## Threshold calibration

`checks/radiation_error_estimator_calibration.py` runs a low-Z/high-Z x thin/thick case matrix (C and W, 25 and 100 keV, 2e2--2e4 Ang, 2000 electrons, seed 7, frozen lockstep transport) and compares each estimator against the ACTUAL spectral change when every flight's emission is evaluated at its midpoint energy instead of its start energy -- the same segments with only `E_keV` swapped, so trajectory divergence does not confound the measurement. CXR numbers use the hopg (0,0,2) reflection on the test-suite geometry (`n_hat = (1, 0, 0.01)`, 700--1500 eV grid).

Measured 2026-08-10 (2000 electrons, seed 7, frozen lockstep; actual change = spectrum evaluated at flight-midpoint energy versus start energy on the same segments):

| case | flights | drift p50 | drift p99 | CXR L1 | CXR max bin | quad p50 | quad p99 | brem L1 | brem max bin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C, 25 keV, 2e3 Å | 13 663 | 5.11e-02 | 2.04 | 4.36e-02 | 1.41e-01 | 7.45e-04 | 4.62e-03 | 1.99e-03 | 2.19e-03 |
| C, 25 keV, 2e4 Å | 179 295 | 4.93e-02 | 2.51 | 4.24e-02 | 1.24e-01 | 1.02e-03 | 8.56e-03 | 2.58e-03 | 2.79e-03 |
| C, 100 keV, 2e4 Å | 31 260 | 6.03e-02 | 2.60 | 9.73e-02 | 2.95e-01 | 2.57e-04 | 1.69e-03 | 7.28e-04 | 7.79e-04 |
| W, 25 keV, 5e2 Å | 77 147 | — | — | — | — | 2.32e-04 | 1.83e-03 | 7.48e-04 | 6.55e-04 |
| W, 25 keV, 5e3 Å | 1 107 956 | — | — | — | — | 3.77e-04 | 5.94e-03 | 7.92e-04 | 1.09e-03 |
| W, 100 keV, 5e3 Å | 291 582 | — | — | — | — | 5.98e-05 | 3.98e-04 | 1.68e-04 | 1.77e-04 |

Threshold selection from this table:

- **CXR drift: warn at p99 = 1.0 linewidths.** Every measured case sits at p99 2.0--2.6 linewidths with an actual line-region change of 4--10% (grid L1) and 12--30% (worst bin): when 1% of flights sweep more than one full linewidth, the frozen-energy line spectrum is materially wrong, so all of these cases should warn. The review's 0.05 suggestion matches the measured *typical* (p50) drift, not a warning level; used as a threshold it would fire on half of all flights in every case.
- **Brem quadrature: warn at p99 = 1e-2.** The actual grid-L1 change runs at a stable ~0.4x the p99 integrated estimate (0.30--0.43 in five of six cases; 0.13 for W 5e3 Å, where rare long flights dominate the tail but contribute little yield). A p99 of 1e-2 therefore flags a ~0.4% continuum change and sits above every measured case (p99 <= 8.6e-3), so the default transport does not warn. The review's 1% suggestion as a threshold on the estimator itself would never fire in the matrix despite measured max-bin changes up to 2.8e-3.

The thresholds are therefore measured convergence behavior of these spectra, not the review's a priori 0.05-linewidth / 1% suggestions.

## What this row does not claim

- Nothing changes in any kernel's output: the estimators are read-only diagnostics, and no default call path computes them.
- No substep invariance. Making the spectra invariant to numerical substep refinement is checklist step G of `feature/energy-controlled-electron-transport`; migrating the kernels to a representative energy is decided there, not here.
- The thresholds are warning levels for the frozen rule's reliability, not acceptance tolerances for the convergence matrix of checklist step E.

## Independent verification

Fresh-context rederivation (2026-08-11, verifier context separate from the implementation): units, limits, and signs/conventions all `pass`; source-to-code agreement against `mc_spectrum`'s resonance/sinc definitions, `_brem_dsigma_dk`, and the Joy--Luo stopping rule confirmed term by term; calibration-table arithmetic reproduced. Verdict `rederived`; two robustness observations (the grid-window superset above, and single-composition prediction for multi-layer segments) are addressed in this doc and in `_flight_E_end_keV`. `signed-off` remains a human decision.
