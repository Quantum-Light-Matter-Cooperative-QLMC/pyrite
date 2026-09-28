# Issue #100: nonuniform photon continua

Issue: #100
Branch: `issue-100-nonuniform-photon-continua`

Status: nine slices implemented and checkpointed on this branch; issue stays
open, remaining scope below.

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

## Derived node placement: edges and the kinematic endpoint

- `energy_grid/refine.py::refined_continuum_grid` takes the geometric baseline
  and refines it only where the modelled continuum is not smooth on the scale of
  its own step: absorption edges, and the bremsstrahlung kinematic endpoint. The
  argument is a midpoint-quadrature error one -- a grid uniform in `u = ln E`
  equidistributes `eps_i ~ (h_i^2/24)|n''/n|` for a locally power-law integrand,
  and both features degrade the straddling interval from `O(h^2)` to `O(h)`,
  which no globally finer geometric spacing fixes at better than first order.
- Edges are **located, never listed**: the steepest adjacent Chantler `f2` ratio
  near each xraydb edge energy, read from the same table the escape model
  attenuates with. `line_seeds.py::absorption_edge_seeds` was split so its
  locator (`absorption_edge_brackets`, returning `EdgeBracket`) is shared with
  the continuum axis rather than copied; the line-window seeds keep their exact
  prior output. Elements come from the medium's own catalog composition plus the
  silicon sensor both detector models share.
- Endpoints need no taper: one anchor pair straddling `E*` at the grid's own
  local spacing puts a midpoint bin *edge* exactly on the cutoff, so the bin
  below carries the tip and the bin above is exactly empty.
- Marks merge into the baseline by displacement, not accumulation: a baseline
  node within half a mark's spacing of a mark node is replaced by it, and marks
  overlapping each other merge on the finer spacing. Without that, float drift
  in a sample lattice opens near-degenerate intervals that
  `validate_backend_coordinates` rightly refuses at float32.
- `CaseLadder.continuum_grids` / `continuum_ladder_grids` are the production
  selectors, so the convergence ladders now start at the medium's own derived
  floor (closing the sixth slice's "no production caller" gap) and carry the
  refinement, with the endpoint read from the case's own incident energy.
- Physics record: `Validation: continuum-node-refinement`, derivation in
  `docs/physics/radiation-physics/energy-grid-semantics.md`
  (`continuum-node-refinement`, `eq-grid-quadrature-error`) and a ledger row in
  `docs/validation/ledger-crystallography-atomic-data.md`. Status `rederived`;
  **not** signed off. Generated views regenerated (`0 / 125` claims signed off).
- Measured on the fixed-trajectory HOPG case (`seed=0`, 3 electrons, 30 keV):
  every gated continuum observable moves far inside budget (yield `2.6e-6`,
  centroid `2.5e-6`, Timepix3 `9.4e-6`, Eagle XO `4.9e-6` at 2049 nodes; the
  centroid rises to `2.5e-5` with the endpoint inside the band, still `40x`
  inside a `1e-3` budget) and every delta shrinks as the baseline refines. Cost
  is `+19`/`+16`/`+10` nodes on 2049/4097/8193. The ledger's earlier
  `+22`/`+19`/`+13` was a stale pre-merge-rule measurement and was corrected in
  this slice; the observable and endpoint-placement figures re-measured
  unchanged.

## Production routing: the brem diagnostic band

- `derive.py::wide_brem_grid(material)` replaces the module-level
  `WIDE_BREM_EV = np.arange(0.0, 40_000.0, 25.0)`. The diagnostic band now
  starts at the medium's own derived continuum floor instead of at `0.0`, which
  sat below the band the escape model is valid over and put a node at an energy
  where it is not defined at all. `--brem-grid-stop` now overrides
  `WIDE_BREM_STOP_EV`; each case's floor comes from its own medium.
- **The lattice stays uniform, deliberately, and this is the measured reason.**
  What this grid measures is a cumulative 95% quantile *in absolute energy*
  near 13-20 keV, returned as a node coordinate and then rounded to 100 eV by
  `margined_stop`. Holding that 100 eV quantum at the 40 keV ceiling costs
  ~2,900 geometric nodes against 1,600 uniform ones, and a coarser geometric
  grid quantizes the quantile outright: measured on hopg/30 keV, a 221-node
  geometric grid moved the coverage energy from 13,575 to 14,136 eV and the
  derived stop from 15,700 to 16,300 eV. Geometric spacing is right for the
  grids the continuum is *evaluated* on and wrong for this one, so the issue's
  "brem diagnostic grid cannot go log" item is answered by routing the floor,
  not by changing the shape.
- Nodes are the same multiples of the step the 0-based grid used, so only the
  sub-floor nodes are dropped. Measured on hopg/30 keV (1 mm slab, 8 electrons,
  `seed=0`): coverage energy and derived stop are **identical** before and
  after, so no derived catalog value moves.
- **Correction (ninth slice).** This slice originally reported the dropped band
  as `6e-7` of total escaping intensity, "because self-absorption has already
  removed it", with densities of order `1e-15`. That does not reproduce and the
  commit message `6fd06a1d` carries the same wrong figure. Re-measured on the
  same case by evaluating once on the zero-based grid and integrating over both
  ranges -- which isolates the grid change from Monte Carlo noise, since the
  floored nodes are a strict suffix -- the dropped 0-50 eV band carries
  `1.3e-3` of the total, and the density at 25 eV is `2.2e-9` against a peak of
  `4.6e-8` near 275 eV, i.e. about 5% of peak, not 1e-15. Self-absorption
  suppresses the low band, it does not empty it. The *conclusion* that no
  derived catalog value moves still holds and was re-verified directly: the
  coverage energy and `margined_stop` are unchanged, because a 0.13% shift in
  the normalization of a cumulative quantile sitting on a steep part of the CDF
  does not survive the 100 eV rounding.
- First node by medium: `hopg`/`silicon`/`diamond` `50 eV`, `wse2`/`ptbi2`
  `75 eV`. The case carries a `_diagnostic_brem_grid` provenance record
  (floor/stop/step/num) beside the existing `_diagnostic_line_grid`.
- Noted while verifying, not changed here: `campaign/sweep.py:825-833` re-spans
  a **uniform** (triple-encoded) brem grid to each case's beam energy
  (`E0*1e3 + step`), but passes a **nonuniform** one through unchanged. A
  floored uniform lattice still encodes as a triple, so that behaviour is
  preserved exactly; a nonuniform production brem grid, however, would keep its
  nodes above the kinematic endpoint rather than being truncated to it. Those
  nodes are zero rather than wrong -- and the refinement slice puts a bin edge
  exactly at the endpoint -- but the asymmetry is real and belongs with whoever
  gives nonuniform grids a sweep-level spelling.

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

Seventh slice (node refinement): `pyrite-dev lint`, `typecheck`, and `docs` are
clean (`docs` again needs `pyrite-dev validation-ledger --write` first, since the
new row makes the generated views stale). The affected
energy-grid/montecarlo/seed set is 134 passed / 2 skipped. The broader
`test-suite core` run is 6 failures, all environment, none code: this shell
exports `PYRITE_ONLINE_TESTS=1` and `PYRITE_RUN_INTEL_SYCL_TESTS=1`, which force
`tests/materials/test_crystal_external_db.py` (needs the `external-db` extra,
`mp_api`) and `test_intel_machine_selects_sycl_backend` (needs `dpctl`) to run
instead of skip; neither extra is installed in this worktree's `.venv`.

Eighth slice (brem diagnostic band): `pyrite-dev lint`, `typecheck`, and `docs`
are clean. `tests/energy-grid` + `tests/scan` + `tests/checkpoint` +
`montecarlo/test_case.py` is 823 passed. Real-case probe (not monkeypatched):
`_build_case("wse2", ...)` carries `(75.0, 30025.0, 25.0)` -- floored start,
and the same beam-energy re-spanning the 0-based grid got.

## Production routing: the installed brem grid's own floor

The eighth slice floored the *diagnostic* band the brem `stop` is measured on.
The band the `stop` was then installed for still began at `0.0`:
`apply.py::_merge_brem` hard-coded `arange["start"] = 0.0`, and every one of the
catalog's `E_grid_brem` rows -- 32 per-material overrides and 15 profile-level
defaults -- carried it. So each production case was evaluated from 0 eV, below
the band the escape model is defined over, while the bound that sized it was
measured from 50 eV.

- **Where the floor is applied, and why not in the catalog.** First attempt was
  to write each medium's derived start into its catalog row and a catalog-wide
  maximum into each profile default. That is wrong, and
  `test_case_content_key_matches_across_profiles_for_shared_cases` caught it:
  `standard` and `sub_100keV` must produce bit-identical case dicts for the
  cases they share, and a *stored per-profile* start makes them disagree about
  the same material as soon as it is not the same number everywhere. The floor
  is a property of the medium, so it is resolved where the band meets the
  material -- `campaign/sweep.py::build_cases` raises a declared start to
  `floored_lattice_start_eV(material, step)`. A profile-level default stays at
  `0.0`, which now reads as "no bound beyond the medium's own".
- `floored_lattice_start_eV` (the lowest multiple of the grid's own step at or
  above `photon_continuum_floor_eV`) is shared with `wide_brem_grid`, so the
  band a bound is measured over and the band it is installed for begin at the
  same energy -- pinned by a regression. `wide_brem_grid`'s output is bit-for-bit
  unchanged.
- It also resolves a catalog *material* to its emitting crystal via
  `continuum_medium_key`. This fixes a regression the eighth slice shipped:
  `wide_brem_grid("mos2-on-sapphire")` raised `unknown filter material`, because
  a film-on-substrate catalog entry is not a medium the attenuation tables know.
- A declared start *above* the floor is kept: narrowing the band is an ordinary
  bandwidth choice, matching `geometric_continuum_grid`. Nonuniform grids pass
  through untouched -- their builders resolve their own floor.
- `_merge_brem` now writes the medium's own derived start, so a future `derive`
  installs a floored row rather than a 0-based one, and the 32 per-material
  override rows were regenerated to match. Those values are redundant with the
  resolver by construction (`max` is idempotent), but they keep the catalog
  self-describing and `show` honest.
- `resolved_show_inputs` reports the *resolved* band, so `pyrite material
  energy-grid show` prints what a case will actually be evaluated over rather
  than the number the row stores. Found and fixed while doing this: the first
  attempt mutated the brem dict in place, and `effective_brem` hands back the
  profile default's own dict, so one material's floor leaked onto every material
  inheriting the default.
- **Import structure.** `campaign` cannot import `energy_grid` -- `energy_grid`
  imports `campaign` -- and the repo map counts function-level imports, so
  deferring did not help either. The floor moved to a package-root leaf,
  `_photon_continuum_floor.py`, exactly as `_energy_grid_encoding.py` and
  `_line_grid_policy.py` did; `energy_grid/floor.py` is now a facade that
  re-exports it and keeps `geometric_continuum_grid`. `docs/repo_map.md` is
  unchanged by this slice, i.e. no new component merges. Ledger and physics-doc
  references follow the function to its new home. A consistency audit caught one
  thing the move broke: `energy_grid/floor.py` had carried the row's only
  `Validation: photon-continuum-floor` marker, so after the split its
  `geometric_continuum_grid` anchor was unmarked; the marker is now on that
  function's own docstring, where it belongs. All three ledgered code anchors
  carry one. The same audit lists 12 **pre-existing** anchor inconsistencies
  elsewhere in the ledger (chiefly `montecarlo/spectrum/lines.py`, split into a
  package by `3c924477`, still named by 23 rows); none is from this branch and
  none is fixed here.
- **Identity.** This deliberately orphans pre-floor records, and the two kinds of
  digest move differently:
  - `case_content_key` **moves** (`4c9114bd` -> `7b3e068b`). A case's
    `E_grid_brem` now starts at the medium's floor, so the band the record was
    computed over genuinely changed; serving a pre-floor record for it would be
    wrong. Re-minted with the reason recorded, as the four prior physics
    re-mintings were.
  - the eight `named_profile_identity` pins and the standard `dataset_identity`
    digest **do not move**: they hash the *declared* sweep payload, which the
    resolver does not touch. An intermediate design did move them; that was
    reverted along with the needless re-minting of the shipped hopg/hbn
    artifacts, whose stored `0.0` is a correct bandwidth request.
  - one digest moves for a third reason: `mose2`'s `survey` dataset digest,
    because `mose2` carries its own override row and that row's `start` now
    records its floor.
- First node by medium at the 25 eV step: 33 of 50 catalog materials at
  `50 eV`, 17 at `75 eV`; `hopg` `50 eV` (floor `30.661 eV`), `mose2` `75 eV`
  (floor `50.048 eV`).
- Measured impact on hopg/30 keV: the dropped 0-50 eV band carries `1.3e-3` of
  total escaping intensity -- see the correction above, which supersedes the
  eighth slice's `6e-7`. This is a real change to absolute yields and is the
  intended one: those nodes sit outside the model's validity.

## Remaining on #100 (not touched here)

- Human sign-off on the `photon-continuum-floor` and
  `continuum-node-refinement` ledger rows (both `rederived`); an agent must not
  mark them.
- A nonuniform *production* grid still has no sweep-level spelling: `apply.py`
  emits line rows as `linspace` inline tables and brem as `arange`, and
  `checkpoints/recompute.py` retunes brem through `(start, stop, step)` only.
  The catalog grid schema already accepts `values`/`logspace`, so this is a
  writer-and-retune gap rather than a schema one. It changes a documented CLI
  contract and needs `cli-reference.md` regenerated.
- Corrected while scoping, not carried forward as work: `results/metrics.py` is
  **not** a uniform-only consumer -- `_peak_sample_spacing` uses the local
  interval and `_integrate_energy_window` uses `np.trapezoid` over explicit
  coordinates. The earlier "`metrics.py:143`" entry pointed into a docstring.
- The CUDA/fallback parity gate is **already executed on hardware** (issue body,
  2026-09-18: RTX 3060 Ti, driver 616.92, CC 8.6, cupy 14.2.0,
  `PYRITE_TEST_BACKEND=cuda`). The earlier "execute on a CUDA lab box" entry
  here was stale.

Given these, #100 stays open after this PR merges.

Ninth slice (installed brem floor): `pyrite-dev lint`, `typecheck`, and `docs`
are clean (`docs` again needs `pyrite-dev validation-ledger --write` first).
`test-suite core` is **2365 passed / 80 skipped / 0 failures**; `cli` 1194
passed / 2 skipped; `apps` 316 passed; `packaging` 301 passed. Goldens
regenerated with `pyrite-dev regen-golden` (node counts drop by 2 or 3 per
material, exactly the sub-floor nodes) and `pyrite-dev repo-map --write`
(unchanged). Real CLI probe, not monkeypatched: `pyrite material energy-grid
brem show hopg` reports `[50, 24900] eV step 25`, and the `mose2` JSON payload
carries `start_eV: 75.0`. `pyrite-dev verify` is green end to end: **4176
passed / 82 skipped, exit 0**. `pyrite-dev format` reformats seven files
unrelated to this slice (the same drift the sixth slice recorded); those were
reverted rather than committed.

Repaired in passing, pre-existing and unrelated to this slice's physics:
`docs/validation/status-summary.md` was stale at `HEAD` -- it undercounted the
ledger at 125 claims and carried duplicated `anchored`/`rederived` rows.
Regenerating from the *untouched* ledger reproduces the same diff, so it is
drift from an earlier slice, not from this one.
