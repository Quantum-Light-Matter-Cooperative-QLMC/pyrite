# Issue #100: nonuniform photon continua

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/100
Branch: `issue-100-nonuniform-photon-continua`

Status: five slices checkpointed and one additional slice implemented on this
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

## Derived photon-continuum floor and the detector's own zero edge

- The continuum floor is now derived, not chosen. `photon_continuum_floor_eV`
  (`energy_grid/floor.py`) returns the larger of two independent bounds: the
  medium's bulk free-electron plasma energy, and the lowest energy at which
  every table the continuum pipeline evaluates has real support.
- Modelled band. `materials/attenuation.py::plasma_energy_eV` computes
  `hbar*omega_p = hbar*sqrt(n_e e^2 / (eps_0 m_e))` with `n_e = sum_i n_i Z_i`
  from the medium's own catalog number densities. This is the repository's own
  validity edge rather than an imported convention:
  `crystal.py::optical_constants` writes `delta = (r_e lambda^2 / 2pi) n_a f1`,
  which in the free-electron limit `f1 -> Z` is identically
  `omega_p^2 / (2 omega^2)`, so `delta = 1/2` at `omega = omega_p` and the
  transparent-medium expansion behind photon escape and self-absorption has
  collapsed.
- Data support, measured from the packaged data rather than asserted: EEDL
  MF=26/MT=527 photon spectra reach `0.1 eV` for all 24 transport elements,
  Chantler/FFAST reaches `1.01 eV` for every element (admitted on the strict
  interior, so `1.01 eV` itself reads out of range), and the digitized Eagle XO
  QE curve is documented from `12 eV`. The QE table binds, at `12 eV`.
- For every condensed catalog medium the plasma energy is the larger term, so
  the floor is material-specific and not a round number: `sio2` `30.201 eV`,
  `hbn` `30.248 eV`, **`hopg` `30.661 eV`**, `silicon` `31.050 eV`, `diamond`
  `38.191 eV`, `mose2` `50.048 eV`, `wse2` `56.913 eV`, `ptbi2` `66.248 eV`.
  Data support binds only in the dilute limit, which is the documented limiting
  case (`n_e -> 0` gives `hbar*omega_p -> 0`).
- Independent cross-check, no shared code path: inverting the packaged
  PDG/Sternheimer `C_bar = 2 ln(I / hbar omega_p) + 1` for silicon
  (`materials/_transport_data.py`) gives `31.0482 eV` against the `31.0498 eV`
  computed from the catalog number density -- `5.2e-5` relative.
- `geometric_continuum_grid` builds the geometric baseline from the resolved
  floor and refuses a `floor_eV` below it; a *higher* floor is an ordinary
  bandwidth choice and is allowed. No node redistribution here.
- Zero-based detector channel. `_grid_semantics.py::zero_based_detector_edges`
  always yields a first edge of exactly `0.0 eV` but separates two physically
  different routes. A reflected *negative* half-bin is clamped up to zero --
  `82fea148`'s behaviour, now named and shared rather than inline. A *positive*
  continuum floor instead gets a **separate** empty channel `[0, eps_0)`
  prepended: widening the first source bin down to zero would multiply that
  node's density by the extra width and invent photons. Callers prepend one zero
  to the density; that channel carries no source mass.
- One forced minimal adjustment. Timepix's single `dE_mc` channel of band
  padding assumes the outermost midpoint cell is narrower than `dE_mc`, which a
  uniform mesh satisfies but a coarse geometric one does not (its top half-width
  grows with energy: ~250 eV at 20 keV on a 257-node grid against 50 eV of
  padding), and a floored geometric grid tripped the existing "input-channel
  edges failed to cover the source grid" guard. Channels now widen *only* when
  the source edges actually escape the padded band, so every previously covered
  mesh keeps exactly the channels -- and the seeded MC matrix -- it had.
- Uniform-grid detected density is bit-for-bit unchanged: verified directly
  against the pre-change implementation (identical sum `31.360624834978506`,
  `np.array_equal` true, max abs diff `0.0`), and pinned in-suite by comparing
  `apply` against a rebin on the pre-change *unpadded* edges.
- Physics record: `Validation: photon-continuum-floor`, with source equation,
  assumptions, limiting case, and cross-check in
  `docs/physics/radiation-physics/energy-grid-semantics.md`
  (`eq-grid-plasma-floor`) and a ledger row in
  `docs/validation/ledger-crystallography-atomic-data.md`. Status `rederived`;
  **not** signed off -- only a human does that. Generated ledger views
  regenerated (`0 / 124` claims signed off).

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

Sixth slice (continuum floor): `pyrite-dev lint`, `typecheck`, and `docs` are
clean; `docs` required `pyrite-dev validation-ledger --write` first, because the
new ledger row makes the generated views stale. The affected
detector/grid/materials set is 833 passed / 40 skipped. The broader
`test-suite core` run is **2227 passed / 79 skipped / 0 failures**. The 42
failures earlier slices reported here were environment gaps, not code: this
worktree's `.venv` was missing the `notebooks` and other dependency groups, and
`uv sync --all-groups` resolves them (matplotlib, `dpctl`, and the rest), after
which the live-crystallography-DB and multiprocessing-socket cases skip rather
than fail. Note `pyrite-dev format` reformats seven files unrelated to this
slice; those were reverted rather than committed.

## Remaining on #100 (not touched here)

- Refine the geometric photon-continuum baseline near material edges and
  kinematic endpoints. `geometric_continuum_grid` deliberately builds the
  unrefined baseline only, so this slice is the natural next owner of node
  placement between floor and ceiling.
- Execute the prepared CUDA/fallback parity gate on a CUDA lab box.
- Human sign-off on the `photon-continuum-floor` ledger row (status
  `rederived`); an agent must not mark it.
- No production caller selects `photon_continuum_floor_eV` yet -- the
  convergence ladders still build their geometric grids with a literal
  `100.0 eV` start. Routing them through the derived floor changes their
  numbers, so it belongs with the refinement slice rather than here.

Given these, #100 stays open after this PR merges.
