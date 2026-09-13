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

- [ ] Build on a uniform coordinate in log kinetic energy, strictly positive
      bounds; carry `log_E_min` + `inv_dlogE` (or equivalent) in
      `TransportEnergyLUT`.
- [ ] Closed-form index: one log, one multiply, one int conversion. Replace the
      inline `(E - E_min) * inv_dE` forms in `_jit_kernel.py` too, not just the
      helpers; update launch metadata and midpoint/cutoff lookups together.
- [ ] Convergence study: 128 intervals/decade baseline vs 256 and 512 (trial
      resolutions, not proposed defaults).
- [ ] For smooth positive rates, compare interpolating log values against
      interpolating values in log energy. Leave probabilities/CDFs
      untransformed unless positivity, monotonicity, and exact terminal
      normalization are preserved.
- [ ] Refine locally / split intervals at model joins; never interpolate across
      a genuine discontinuity.
- [ ] Replace silent spacing inflation above the point cap with a reported
      unmet tolerance.

## Acceptance

- [ ] CPU and CUDA endpoint behavior identical.
- [ ] Off-grid validation against direct physics throughout every decade, plus
      targeted points near cutoffs and model joins.
- [ ] `tests/montecarlo/test_transport_lut.py` ~1e-5 relative accuracy over
      5–30 keV preserved; that test does not certify six decades.
- [ ] Wall time, memory, accuracy recorded together; GPU numbers via
      `pyrite remote`.
- [ ] Adjacent nodes distinct in backend precision (no float32 collapse).

## Decisions

- Table interpolation error only. Flight-integration error (`energy_model`,
  `max_dE_frac`, stopping/cutoff integration) is checked independently.
- Grid change does not extend physics validity toward 100 MeV (#13, #84).
