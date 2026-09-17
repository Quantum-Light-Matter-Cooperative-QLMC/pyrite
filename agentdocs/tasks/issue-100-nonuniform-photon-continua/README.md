# Issue #100: nonuniform photon continua

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/100
Branch: `issue-100-nonuniform-photon-continua`

Status: four slices checkpointed and one additional slice implemented on this
branch; issue stays open, remaining scope below.

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

## Source-mesh-independent Timepix channels

- `TimepixResponse` now anchors its coarse input edges to absolute multiples of
  `dE_mc`, with one channel of band padding, instead of anchoring them to the
  first source-bin edge. Refining a source mesh over the same modeled band no
  longer shifts the characterized photon energies.
- All source meshes use the conservative piecewise-constant overlap integral;
  uniform source bins may straddle the fixed detector edges and therefore can
  no longer use the former whole-bin `bincount` shortcut.
- The expensive seeded response matrix is cached by its fixed input/output
  channels, hardware, and MC settings. Grid-specific wrappers remain cached by
  full source-grid identity.
- If midpoint reflection would put the first source-bin edge below zero, the
  Timepix scoring path clamps that edge to its explicit physical 0 eV detector
  boundary. This does not choose the continuum's still-open positive node floor.
- A deterministic regression verifies identical channels, an identical shared
  MC matrix, and detected-count agreement within `1e-3` under source-grid
  halving (`n_mc=64`, seed 7).

## Explicit Gaussian edge-loss accounting

- `convolve_detector(..., return_outside=True)` now returns the unchanged
  blurred density plus `(below, above)` photon masses outside the output
  window. The masses integrate each source-node mass against the Gaussian CDF
  tails at the grid's midpoint-cell edges; the default array-only return stays
  backward compatible and the historical uniform-grid values remain bit-for-bit
  unchanged.
- `EagleResponse.apply(..., return_outside=True)` exposes the same accounting
  after QE and energy resolution. With resolution disabled, no blur crosses the
  window and both reported masses are zero.
- Regressions cover uniform compatibility, lower-edge loss and mass balance on
  a graded grid, and upper-edge loss through Eagle XO.
- The existing convergence harness already carries absolute photon-yield and
  detector-count floors. A new nonzero floor-scale regression pins that these
  floors, rather than relative tolerance alone, accept an almost-dark spectrum.

## Fixed-trajectory continuum convergence

- The convergence harness now reports and gates continuum yield and centroid at
  `1e-3`, and Timepix3/Eagle XO continuum counts at `1e-2`, alongside the
  existing line yield, centroid, detected counts, FWHM, and line/background
  ratio.
- `CaseLadder` evaluates bremsstrahlung directly on every candidate photon grid
  from the same transported segments. It no longer mistakes interpolation of a
  fixed production bremsstrahlung grid for continuum-grid convergence.
- Near-zero classification is component-specific: an empty line spectrum no
  longer causes a nonzero continuum to pass as `near-zero`.
- The resumable convergence checkpoint schema is now 2 because stored reports
  contain the new continuum observables and use direct continuum evaluation.
- A deterministic `seed=0`, three-electron HOPG regression holds transport
  fixed and verifies all four continuum observables converge across nested
  geometric grids of 2,049, 4,097, and 8,193 nodes. The Timepix response uses
  `n_mc=32`, `seed=7`; this seed affects only its fixed response matrix, not
  differences between grid rungs.

## CUDA/fallback parity gate

- Existing CPU coverage already exercises the full EEDL continuum evaluator on
  nonuniform native and geometric grids, while existing CUDA tests cover the
  fused raw kernels on irregular energy coordinates.
- A new hardware-gated regression now drives the full `mc_brem_spectrum` EEDL
  dispatcher over a 257-node geometric continuum and compares the fused CUDA
  reduction with the generic chunked CUDA fallback on identical segments.
- Local collection passes with the CUDA test skipped because this worktree has
  no CuPy/CUDA device. Run it on the lab box with:
  `PYRITE_TEST_BACKEND=cuda UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/test_spectrum_cuda_cheap_hoists.py -k nonuniform_eedl_continuum_matches_chunked_cuda_fallback`.

## Verification

`pyrite-dev lint`, `typecheck`, `docs`: all clean. `pyrite-dev test`: 3919
passed, 79 skipped, 1 known-unrelated failure
(`test_intel_machine_selects_sycl_backend`, missing `dpctl` in this
worktree's `.venv`, not a code issue).

Second slice: affected detector/grid set is 110 passed; lint and typecheck are
clean. The broader core run reached 2128 passed / 37 skipped; its relevant
coarse-near-zero convergence failure motivated the explicit 0 eV edge above
and passes on rerun. Remaining failures require unavailable live database/SYCL
access or sandbox-denied multiprocessing sockets.

Third slice: detector/grid/convergence set is 106 passed; lint and typecheck are
clean. The broader core run reached 2133 passed / 37 skipped / 42 failures;
all failures are the same unavailable live crystallography DB, missing Intel
SYCL dependency, or sandbox-denied multiprocessing socket constraints above.

Fourth slice: affected detector/grid/convergence set is 116 passed; lint,
typecheck, and docs are clean. The broader core run reached 2136 passed / 37
skipped / 42 failures; all failures are the same unavailable live
crystallography DB, missing Intel SYCL dependency, or sandbox-denied
multiprocessing socket constraints above.

Fifth slice preparation: EEDL/fallback set is 9 passed / 1 CUDA-module skip;
lint and typecheck are clean. Hardware execution remains required before the
CUDA acceptance item can close.

## Remaining on #100 (not touched here)

- Refine the geometric photon-continuum baseline near material edges and
  kinematic endpoints.
- Choose a physically justified positive photon floor; explicit zero-based
  detector-channel edge.
- Execute the prepared CUDA/fallback parity gate on a CUDA lab box.

Given these, #100 stays open after this PR merges.
