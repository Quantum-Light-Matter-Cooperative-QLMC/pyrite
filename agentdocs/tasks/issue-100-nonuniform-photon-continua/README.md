# Issue #100: nonuniform photon continua

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/100
Branch: `issue-100-nonuniform-photon-continua`

Status: two slices checkpointed and one additional slice implemented on this
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

## Remaining on #100 (not touched here)

- Refine the geometric photon-continuum baseline near material edges and
  kinematic endpoints.
- Choose a physically justified positive photon floor; explicit zero-based
  detector-channel edge.
- Continuum observables (yield, centroid, detected counts, line/background
  ratio) shown to converge under refinement.
- CUDA and fallback routes verified for the continuum-scoring path.

Given these, #100 stays open after this PR merges.
