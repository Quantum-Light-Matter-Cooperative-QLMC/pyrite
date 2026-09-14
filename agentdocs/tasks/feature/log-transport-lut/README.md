# feature/log-transport-lut — issue #99

Uniform-in-log-kinetic-energy transport LUT with closed-form indexing.
Parent epic #97; nominally blocked by #98 (semantics/audit), which does not
touch `montecarlo/transport/`, so LUT work proceeds in parallel.

## Problem

`TransportLUTConfig` targets 0.025 keV uniform spacing, 256–16,384 points, and
silently inflates spacing above the cap. Over 0.1–100,000 keV that is 6.1 keV
spacing even at the low-energy cutoff — precisely where a slowing electron
needs resolution. Geometric spacing with adjacent ratio <= 1.01 covers the same
interval in ~1,390 points (arithmetic scaling, not an error guarantee).

## Scope

`montecarlo/transport/lut.py` and every consumer of its index metadata:
`cores.py` (`_lut_index_frac_scalar`, midpoint/cutoff lookups),
`_jit_device.py::_lut_lerp_at`, `_jit_kernel.py` (inline `lut_x`/`alpha_x`
forms), `_jit_launch.py`, `api.py`, `batching.py`, `__init__.py` re-exports,
`tests/montecarlo/test_transport_lut.py`. Photon grids and detectors are owned
by #100/#101.

## Checklist

- [x] Build on a uniform coordinate in log kinetic energy, strictly positive
      bounds; carry `log_E_min` + `inv_dlogE` (or equivalent) in
      `TransportEnergyLUT`.
- [x] Closed-form index: one log, one multiply, one int conversion. Replace the
      inline `(E - E_min) * inv_dE` forms in `_jit_kernel.py` too, not just the
      helpers; update launch metadata and midpoint/cutoff lookups together.
- [x] Convergence study: 128 intervals/decade baseline vs 256 and 512 (trial
      resolutions, not proposed defaults).
- [x] For smooth positive rates, compare interpolating log values against
      interpolating values in log energy. Leave probabilities/CDFs
      untransformed unless positivity, monotonicity, and exact terminal
      normalization are preserved.
- [x] Refine locally / split intervals at model joins; never interpolate across
      a genuine discontinuity.
- [x] Replace silent spacing inflation above the point cap with a reported
      unmet tolerance.

## Acceptance

- [x] CPU and CUDA endpoint behavior identical.
- [x] Off-grid validation against direct physics throughout every decade, plus
      targeted points near cutoffs and model joins.
- [x] `tests/montecarlo/test_transport_lut.py` ~1e-5 relative accuracy over
      5–30 keV preserved; that test does not certify six decades.
- [x] Wall time, memory, accuracy recorded together (CPU build; no GPU sweep needed); GPU numbers via
      `pyrite remote`.
- [x] Adjacent nodes distinct in backend precision (no float32 collapse).

## Decisions

- Table interpolation error only. Flight-integration error (`energy_model`,
  `max_dE_frac`, stopping/cutoff integration) is checked independently.
- Grid change does not extend physics validity toward 100 MeV (#13, #84).

## Results

Grid: uniform in `ln E`. `TransportEnergyLUT` carries `log_E_min` + `inv_dlogE`;
index is `x = (log(E) - log_E_min) * inv_dlogE`, clamp-high tested first so a
non-finite coordinate lands on the low clamp rather than `int(x)`. The same
branch order is spelled out in `_lut_index_frac_scalar`,
`_jit_device::_lut_lerp_at`, and both inline forms in `_jit_kernel`.

Config: `intervals_per_decade=1024`, `min_points=256`, `max_points=16384`,
`rel_tol=1e-4`. `step_keV` is gone. The builder measures the achieved error at
the 1/4, 1/2, 3/4 points of every interval and warns
(`TransportLUTToleranceWarning`) when it exceeds `rel_tol`, naming the energy,
the achieved intervals/decade, and whether `max_points` capped the request.
`max_rel_interp_error` / `worst_energy_keV` / `tolerance_met` /
`intervals_per_decade` are reported on the LUT.

Convergence (synthetic C+Si layer, build-time audit, CPU):

| range (keV) | int/decade | points | max rel err | at (keV) | build (ms) | tables (kB) |
|---|---|---|---|---|---|---|
| 1-30 | 128 | 191 | 6.93e-05 | 4.23 | 0.37 | 10.4 |
| 1-30 | 256 | 380 | 2.68e-05 | 4.22 | 0.39 | 20.8 |
| 1-30 | 512 | 758 | 1.00e-05 | 4.23 | 0.52 | 41.5 |
| 1-30 | 1024 | 1514 | 3.52e-06 | 2.76 | 0.69 | 82.8 |
| 0.1-1e5 | 128 | 769 | 4.76e-05 | 2.76 | 0.50 | 42.1 |
| 0.1-1e5 | 256 | 1537 | 1.93e-05 | 4.23 | 0.67 | 84.1 |
| 0.1-1e5 | 512 | 3073 | 1.17e-05 | 4.23 | 0.99 | 168.1 |
| 0.1-1e5 | 1024 | 6145 | 4.63e-06 | 2.76 | 1.81 | 336.1 |

Every peak sits on a Joy-Luo/Berger-Seltzer stopping crossover. Away from the
joins the smooth tables converge at O(h^2): over 0.1-1e5 keV at 128/256/512
int/decade the dense-probe errors are total_rate 3.97e-05 / 9.92e-06 /
2.48e-06, inv_beta 1.01e-05 / 2.53e-06 / 6.32e-07, alpha (screened Rutherford)
4.05e-05 / 1.01e-05 / 2.53e-06.

Interpolation space: values, not logs. Log-value interpolation was measured
(same ranges, dense probes) and is better per node -- total_rate 5.30e-06 /
1.33e-06 / 3.32e-07, inv_beta 5.92e-06 / 1.48e-06 / 3.70e-07, and exactly zero
for the 1/E screened-Rutherford alpha -- but it costs an exponential per table
read in the innermost loop (~5 reads per step, CPU and CUDA), while the same
accuracy costs only a doubling of nodes at a few hundred kB. It also buys
almost nothing where the error actually lives: at the stopping crossover
5.11e-05 -> 3.13e-05, and for the NIST Mott screening table 3.66e-03 ->
3.56e-03 at 128/decade. The CDF is left untransformed (positivity, monotonicity
and exact terminal 1.0 are asserted).

Joins: no genuine discontinuity exists. The Joy-Luo/Berger-Seltzer splice is
C0 -- the value jump scales linearly with the probe offset (1.4e-09 at 1e-09,
1.4e-12 at 1e-12) with a ~1.1-1.6% relative slope jump -- and the Mott alpha
table is piecewise linear in log10 E, so also C0. Nothing is smoothed across a
jump. Each straddling interval drops to O(h) and is what the audit reports;
the NIST alpha table's own coarse 0.05-0.8 keV interval is the worst such case
(~1e-03 regardless of interpolation space).
