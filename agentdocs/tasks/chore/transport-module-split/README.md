# chore/transport-module-split

## Problem

`src/pyrite/montecarlo/transport.py` is 5050 lines (growing: the in-flight
`feature/energy-loss-straggling` worktree adds ~290 more, including another
large docstring on `_run_per_electron_transport_lut` — see Decisions). It mixes
several separable concerns in one file (see symbol clusters below) and carries
653 lines of docstrings, dominated by `simulate_trajectories`'s 272-line
docstring (`transport.py:3996`).

Much of that docstring's prose duplicates content already maintained as prose
in `docs/physics/beam-transport/*.md` and verified in depth in
`docs/validation/beam-transport/*.md` / `docs/validation/geometry/*.md`
(confirmed by grep: `beam_fwhm_mm`, longitudinal bunch sampling, energy spread,
`transport_core` modes, `energy_model`, groove re-entry, etc. all have a
dedicated docs/physics and/or docs/validation page). Duplication has already
caused drift: `docs/validation/ledger-core-coherent-physics.md:53` records that
a validation run "invalidated the `simulate_trajectories` docstring assertion"
about beam-spot effects on the spectrum — the docstring and the docs page said
different things.

Two related but separable pieces of work:

1. **Split `transport.py` into subfiles** grouped by concern.
2. **Cut the `simulate_trajectories` docstring down** to what
   `docs/validation/methodology.md:110` actually requires (source,
   assumptions, ≥1 limiting case, `Validation: <id>` marker per physics
   function) plus a short pointer to the maintained docs page, migrating any
   content that is NOT already covered in `docs/physics`/`docs/validation`
   before deleting it — never delete a physics claim outright.

These share one branch/task (not two independently-ownable ones): both touch
the same file's structure, order matters (trim before or interleaved with
split, not after — splitting first just relocates the same duplication into a
different file), and running them as parallel branches from the same base
would conflict badly.

## Scope

In scope:
- Restructuring `src/pyrite/montecarlo/transport.py` into a package with
  private internal submodules and a `transport/__init__.py` (or equivalent)
  that re-exports the existing public/tested surface so **no importer changes
  its import** (see "Importers" below).
- Trimming `simulate_trajectories`'s docstring (and any other outsized
  docstring uncovered along the way) to a methodology-compliant derivation
  docstring + pointer, migrating uncovered content into the matching
  `docs/physics/beam-transport/*.md` page first.
- Updating anchors this touches (see Decisions #2).
- Regenerating anything the split invalidates: `docs/_autosummary/*`,
  `docs/repo_map.md` transport entry (`docs/repo_map.md:279`).

Out of scope:
- Any change to transport algorithms, numerics, or RNG stream order. This is
  a structural/documentation move; every existing test must stay bit-for-bit
  green with no seed/output diffs.
- Rewriting `transport_jit_kernel.py` (the CUDA twin) — not requested, and a
  separate file already.
- Retargeting the ~50 `montecarlo/transport.py::<symbol>` anchors in
  `docs/validation/*.md`/`docs/repo-design/*.md` to their new submodule paths
  (see Decisions #2 for why this task keeps `transport.py` as a re-export
  shim instead).

## Current structure (symbol clusters, for the split boundary)

From `get_symbols_overview` on `main`'s `transport.py`, in file order:

- kinematics/RNG streams: `beta_from_keV*`, `_splitmix64`, `_stream_key_scalar`,
  `_stream_uniform_scalar`, `stream_keys`, `_sample_bunch_offsets`
- elastic scattering (Browning/Mott/SR): `_sigma_browning_cm2*`,
  `_alpha_sr_joy*`, `_load_mott_transport`, `_mott_alpha_table`,
  `_scatter_rates_*`, `_sample_cos_theta*`
- stopping power (Bethe/Joy-Luo/spliced) + Sternheimer: `_dEds_*`, `_bs_*`,
  `spliced_stopping_keV_per_ang`, `_element_crossover_keV`,
  `sternheimer_delta`
- Urban energy-loss straggling: `_urban_*` (growing on
  `feature/energy-loss-straggling`)
- energy LUT: `TransportLUTConfig`, `TransportEnergyLUT`, `_lut_*`,
  `build_transport_energy_lut`
- geometry/rotation helpers: `_rotate_direction*`, `_first_prism_exit_scalar`
- transport cores (numba, ungrooved/grooved/per-electron/LUT variants):
  `_transport_core_*`, `_searchsorted_right_scalar`
- batching/dispatch/CUDA availability: `pack_layer_tables`,
  `_percentile_summary`, `_flight_diagnostic_summary`,
  `_cuda_transport_available`, `resolve_transport_core`, `_batch_size`,
  `_capacity_for`, `_batch_electrons`, `_run_per_electron_transport*`,
  `_alloc_scratch`
- public entry point: `simulate_trajectories`, `PerElectronTransportConfig`,
  `DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG`

This is already laid out leaf-to-root (helpers first, orchestration last), so
the split should follow the same dependency order and shouldn't create
import cycles. All the numba `@njit` functions can call jitted functions from
other modules normally — no numba-specific splitting hazard, just ordinary
Python import ordering.

## Importers to leave untouched

Non-test: `api.py`, `campaign/profiles.py`, `campaign/sweep.py`,
`montecarlo/detector.py`, `montecarlo/__init__.py`,
`montecarlo/transport_jit_kernel.py`.
Tests (14 files under `tests/montecarlo/` + `tests/scan/test_public_api.py`).
All of these import from `pyrite.montecarlo.transport` (module-level or
`from .transport import ...`); `docs/api.md`'s autosummary also cites
`pyrite.montecarlo.transport.simulate_trajectories` directly. A package-style
re-export at `transport/__init__.py` keeps every one of these working
unchanged.

## Checklist

1. Rebase onto latest `main`; check whether `feature/energy-loss-straggling`
   has landed (see Decisions #1) and re-run the symbol-cluster scan if the
   file has changed materially.
2. For `simulate_trajectories` (and any other function whose docstring
   exceeds ~2x what methodology.md requires): for each documented parameter,
   confirm whether its claim is already covered by a `docs/physics/*` or
   `docs/validation/*` page (list above / grep `simulate_trajectories` under
   `docs/`). If not covered, migrate the content into the right
   `docs/physics/beam-transport/*.md` page first (small PR-sized additions,
   following `docs/repo-design/documentation.md` conventions — MyST, labeled
   equations, `Validation:`/ledger rules from `docs/validation/methodology.md`
   where the migrated text states a physics claim). Only then trim the
   docstring to source/assumptions/limiting-case/`Validation: <id>` + a link.
3. Check `docs/validation/*.md` rows that quote or paraphrase the docstring
   verbatim (e.g. `finite-beam-size.md`, the coherent-physics ledger note)
   before trimming the sentence they're built on; update the write-up if its
   quoted claim now lives on a docs page instead of in the docstring.
4. Design the package layout (proposed, not fixed):
   `montecarlo/transport/{kinematics,scattering,stopping,straggling,lut,cores,
   batching,api}.py` mapping onto the clusters above; `transport/__init__.py`
   re-exports everything currently importable from `transport.py` (check
   `montecarlo/__init__.py`'s `from .transport import (...)` block and every
   test file's imports for the exact surface, including underscore-prefixed
   names tests reach into).
5. Move code, preserving every `Validation: <id>` marker and ledger-cited
   line verbatim where it isn't being intentionally trimmed per step 2.
6. Regenerate `docs/_autosummary/` and run the docs build; confirm
   `pyrite.montecarlo.transport.simulate_trajectories` still resolves and no
   new (non-baselined) warnings appear.
7. Update `docs/repo_map.md:279` and `docs/repo-design/core-architecture-rfc.md`'s
   line-count mention of this file if still meaningfully wrong afterward.
8. Run `physics-ledger-auditor` to confirm every `Validation: <id>` marker and
   ledger row still resolves.
9. Full verification pass (see Acceptance).

## Decisions / open questions

1. **Sequencing with `feature/energy-loss-straggling`.** That worktree (not
   retouched by this triage) has already added ~290 lines to `transport.py`,
   including expanding `_run_per_electron_transport_lut`'s docstring from one
   line to several. If it lands before this task starts, rebase and redo the
   symbol scan (step 1) — don't tunnel-vision on the line numbers recorded
   here. If it's still open when this task is ready to execute, the owner
   should coordinate rather than let both branches rewrite the same regions
   independently.
2. **Ledger/design-doc anchors stay `transport.py`.** ~50 lines across
   `docs/validation/*.md` and `docs/repo-design/*.md` cite
   `montecarlo/transport.py::<symbol>` as a literal file path (e.g.
   `docs/validation/geometry/finite-beam-size.md:11`,
   `docs/validation/beam-transport/beam-phase-space-injection.md:5`). This
   task keeps `src/pyrite/montecarlo/transport.py` alive as the package's
   `__init__`-equivalent public surface rather than rewriting every anchor to
   a submodule path — smaller blast radius, and the dotted-import anchors stay
   literally true. A follow-up issue can retarget anchors to exact definition
   sites later if the team wants that; flag it in the PR description rather
   than doing it here.
3. **How much docstring content is genuinely uncovered by docs/.** Not fully
   resolved by triage — the implementer needs to actually diff docstring
   prose against the relevant docs/physics and docs/validation pages
   (`beam-phase-space.md`, `longitudinal-structure.md`, `stopping-power.md`,
   `elastic-scattering.md`, `transport-outputs.md`, plus the
   `docs/validation/beam-transport/` and `docs/validation/geometry/` rows
   named in the docstring's own `Validation:` markers). Some sentences (e.g.
   the coherent-emission caveat that was already found wrong once) may need
   to be corrected, not just relocated.
4. **Package layout naming** (step 4) is a proposal; the implementer can
   regroup if the dependency graph doesn't cleanly match the clusters above.

## Delegation

Single `lead-task` owner — not `one-shot`-eligible given the open content
judgment in Decisions #3 and the cross-file blast radius (30 importers, ~50
doc anchors, autosummary, ledger). Suggested internal slices for the lead to
delegate once the docstring-migration content decisions are made:

- Docstring content migration into `docs/physics/beam-transport/*.md`
  (mechanical once decided) — `implement-task`, skills: `documentation-maintenance`,
  `physics-review`.
- Package split + import-surface preservation — `implement-task`, skills:
  `scientific-library`, `monte-carlo` (RNG-stream/bit-for-bit risk).
- Final verification/regen pass (autosummary, repo_map, ledger auditor,
  full test suite) — `implement-task-lite` or done by the lead directly.

Required skills throughout: `monte-carlo` (RNG/bit-for-bit correctness is the
main correctness risk of a pure-structural split), `scientific-library`
(module/export conventions), `physics-review` (docstring content is
physics-claim editing, not just prose), `documentation-maintenance`.

## Acceptance checks

- `uv run pyrite-dev lint`, `format`, `typecheck` clean.
- `uv run pyrite-dev test-suite core`, `test-suite core --numba`,
  `test-suite cli`, `test-suite apps` all green; no seed/output/golden diffs
  (bit-for-bit — this is a structural move, not a physics change).
- `uv run pyrite-dev docs` builds clean; `pyrite.montecarlo.transport.
  simulate_trajectories` still resolves in the built API reference; no new
  (non-baselined) autodoc warnings.
- Every import listed under "Importers to leave untouched" is unchanged at
  its call site.
- `physics-ledger-auditor` reports no orphaned `Validation:` markers and no
  missing ledger rows.
- No physics claim from the original docstring is lost: every sentence
  removed is either already present (linked) on a `docs/physics`/
  `docs/validation` page, or was migrated there in this same change.
