# Issue #100: nonuniform photon continua

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/100
Branch: `issue-100-nonuniform-photon-continua`

Status: one slice landed (this branch); issue stays open, remaining scope below.

## Context

Most of #100 already landed as part of the #114 energy-grid rework
(`fd247a8e` conservative Timepix rebinning, `8bb1e087` full-coordinate
`grid_key`), but `detectors/response.py::convolve_detector` and
`detectors/eaglexo_response.py::EagleResponse(resolve_energy=True)` still
called `require_uniform_grid` and rejected nonuniform grids outright -- a
guard added for #98's consumer audit, not the nonuniform support #100 asks
for. See the reconciliation on #97/#100/#101 (2026-09-15) for how that gap
was found.

## This slice

- `detectors/response.py::convolve_detector`: kept the exact prior
  `gaussian_filter1d` sample-space path (bit-for-bit) when the grid is
  uniform; added a direct-quadrature path in physical energy for nonuniform
  grids, weighting each source node by its own local width
  (`node_bin_edges_and_widths`) instead of one shared `dE`. Same physical
  Gaussian energy-resolution model either way, no new physics equation, no
  ledger row.
- `detectors/eaglexo_response.py::EagleResponse`: no code change needed for
  `resolve_energy=True` beyond the above (it already called
  `convolve_detector`); docstring updated to drop the stale uniform-spacing
  caveat.
- Tests: `tests/energy-grid/test_grid_semantics.py` -- replaced the
  "refuses a log grid" test with a mass-conservation check on a log grid away
  from the (documented, still zero-padded) edges, plus a new
  `EagleResponse(resolve_energy=True)` log-grid coverage test.

## Verification

`pyrite-dev lint`, `typecheck`, `docs`: all clean. `pyrite-dev test`: 3919
passed, 79 skipped, 1 known-unrelated failure
(`test_intel_machine_selects_sycl_backend`, missing `dpctl` in this
worktree's `.venv`, not a code issue).

## Remaining on #100 (not touched here)

- Refine the geometric photon-continuum baseline near material edges and
  kinematic endpoints.
- Choose a physically justified positive photon floor; explicit zero-based
  detector-channel edge.
- Out-of-window photon accounting for `response.py`/`eaglexo_response.py`
  (Timepix already reports `outside`; the Gaussian blur here still silently
  drops mass past the grid ends, only documented, not reported).
- Near-zero spectral regions covered by absolute tolerances.
- Continuum observables (yield, centroid, detected counts, line/background
  ratio) shown to converge under refinement.
- CUDA and fallback routes verified for the continuum-scoring path.

Given these, #100 stays open after this PR merges.
