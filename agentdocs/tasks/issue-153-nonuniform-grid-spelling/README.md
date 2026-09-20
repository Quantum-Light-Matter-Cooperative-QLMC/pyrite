# Issue #153: sweep-level spelling for nonuniform photon continua

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/153
Branch: `issue-153-nonuniform-grid-spelling`

Carried forward from #100 (closed by #149), "Remaining on #100" in
`agentdocs/tasks/issue-100-nonuniform-photon-continua/README.md`. The other
item recorded there -- human sign-off on the `photon-continuum-floor` and
`continuum-node-refinement` ledger rows -- is not in scope and an agent must
not mark it.

Status: first slice (read/report) implemented and checkpointed. Write side and
retune side remain.

## Context

`materials/_catalog_decode.py::_grid` has accepted `values`, `arange`,
`linspace` and `logspace` since the schema was written, and
`campaign/sweep.py::build_cases` already passes a nonuniform `E_grid_brem`
through to a case. What was missing was everything *around* that: no reader in
the CLI-facing path resolved a spelling other than the uniform one.

Reproduced by execution on `main` (350f493a), not by reading:

- a `logspace` brem override made `effective_brem` return `None`, so
  `energy-grid brem show` printed `=== hopg ===` and nothing else -- a
  configured grid silently absent from its own report;
- a `values` line row raised `KeyError: 'linspace'` out of `show`.

## This slice (read/report)

- `materials/_catalog_decode.py`: added public `grid_descriptor_kind` and
  `resolve_grid_descriptor`. The latter wraps the existing error-collecting
  `_grid` into one that raises. Deliberately *not* a second decoder: callers
  outside the validating parse now share the one that defines the schema,
  which is what keeps the CLI's idea of a grid from drifting from the
  catalog's.
- `energy_grid/apply.py::resolved_grid_band`: one descriptor -> `kind` +
  resolved `nodes` + the verbatim `payload`, with a mapping payload's keys
  spread in. The spread is what keeps an `arange` band answering to
  `start`/`stop`/`step`, so every uniform caller and every stored digest is
  untouched.
- `effective_brem` resolves all four spellings instead of `.get("arange")`.
- `resolved_show_inputs` floors **only** an `arange` band. This mirrors
  `build_cases`, which raises a declared `start` on a `(start, stop, step)`
  band and passes a nonuniform grid through because that grid's builder
  already resolved its own floor. Flooring one here would report a band no
  case is ever evaluated over. The artifact branch now states
  `"kind": "arange"` explicitly rather than leaving the flooring to key on an
  absent field.
- `show` renders a graded grid as `[first, last] eV x N pts (kind)`: a
  nonuniform grid has no single step to name, so the count and the spelling
  carry what `step` carries for a uniform one. Uniform renderings are
  byte-for-byte unchanged.
- `console/json.py`: the `"kind"` field was already the seam. A graded grid
  reports resolved `start_eV`/`stop_eV`/`points` plus the descriptor echoed
  verbatim under `descriptor`, so the payload round-trips. The resolved
  endpoints matter for `logspace`, whose own `start`/`stop` are *exponents*,
  not energies -- flattening them into `start_eV` would have been wrong.
  `arange` and `linspace` payloads keep their exact legacy shape, with no
  `descriptor` key.
- Two paths that genuinely cannot carry a nonuniform grid now refuse by name
  instead of failing obscurely: `_existing_line_rows` (the writer stores line
  rows as `linspace` only) and `add_file`'s manual-brem branch (artifact
  identity stores a uniform band only, and used to reach `float(None)`).

## Verification

- New `tests/energy-grid/test_nonuniform_grid_spelling.py`, 8 tests: logspace
  and values brem resolution, text and JSON rendering of both bands, the
  no-flooring rule, the by-name merge refusal, and two regressions pinning
  that uniform text and uniform JSON payloads did **not** move.
- `test-suite core` **10 failed / 2410 passed / 83 skipped**; `test-suite cli`
  1204 passed / 2 skipped. Those 10 are the pre-existing energy-grid/materials
  golden-digest failures. Confirmed pre-existing, not assumed: the same ten
  names fail with this branch's changes stashed away, on the same tree.
- `lint`, `typecheck` clean. `cli-reference --check` passes -- no command,
  flag, or help text changed in this slice, only output for grids that
  previously crashed or vanished.
- Real CLI probe, not monkeypatched: `pyrite material energy-grid show hopg
  -o json` still emits
  `{"kind":"arange","start_eV":50.0,"step_eV":25.0,"stop_eV":40000.0}` with no
  `descriptor` key.
- `pyrite-dev format` reformats nine files unrelated to this slice -- the same
  pre-existing drift the ninth slice of #100 recorded. Reverted, not
  committed.

## Second slice (write side)

Route chosen after establishing where a nonuniform continuum can actually
live. Two stores can hold a material's brem grid: the override table
(`[profiles.<p>.overrides.<material>] E_grid_brem`) and an immutable artifact
pinned by `energy_grid_refs`. When a ref exists the artifact **wins** --
`materials/_parse.py:745` overwrites `E_grid_brem` with the artifact's
`arange` -- and artifact identity stores only `{start_eV, stop_eV, step_eV}`,
which a graded grid cannot express.

Measured before choosing: a `logspace` override reaches production today
(hopg `ScanSpec.E_grid_brem` comes back as 300 graded nodes over
`[50.12, 39811]` eV against 1598 uniform ones), the shipped catalog contains
**zero** `energy_grid_refs`, and `data/energy-grid-artifacts/` is empty. So
the override table is both the working path and the only one in use;
extending artifact identity would have moved content-addressed digests for no
present consumer.

- `apply.py::set_brem_geometric` writes the nodes of
  `energy_grid.floor.geometric_continuum_grid` into the override table, and
  refuses by name when the profile pins an artifact for that material, since
  the row would otherwise be silently discarded at load.
- It emits the **`values`** spelling, not `logspace`. A `logspace` descriptor
  hands the floor endpoint back to `np.logspace` on decode and can round it a
  hair *below* the floor -- exactly the placement `geometric_continuum_grid`
  refuses, and the placement the `photon-continuum-floor` ledger row is about.
  Explicit coordinates round-trip bit-for-bit and keep that guarantee.
- `_material_override_table` gained a `profile` argument; `_merge_brem`'s call
  is unchanged.
- CLI: `pyrite-dev energy-grid brem set` gains `--spacing uniform|geometric`,
  `--num`, and `--start`. `uniform` keeps the existing artifact route
  untouched. Each invalid flag combination is refused by name rather than
  reaching the writer.

### Verified

- Real end-to-end probe against the actual shipped catalog, then reverted:
  `pyrite-dev energy-grid brem set hopg --stop 40000 --spacing geometric
  --num 12` wrote the row, `energy-grid brem show hopg` reported
  `[30.6614, 40000] eV x 12 pts (values)  [manual]`, and the catalog decoded
  back to a grid `np.array_equal` to `geometric_continuum_grid`'s own output,
  with the first node exactly the derived floor `30.66142745725104` eV. Note
  the geometric grid starts *at* the floor where the uniform lattice starts at
  `50` eV, because a uniform lattice must snap to a step multiple and a
  geometric one need not.
- `cli-reference --check` passes: `brem set` is a `pyrite-dev` maintainer
  command and is not part of the user-facing `pyrite` reference, so nothing
  needed regenerating. Stated rather than assumed.
- `test-suite cli` 1208 passed / 2 skipped. New tests: 4 flag-consistency
  cases in `test_cli.py`, 4 writer cases in `test_nonuniform_grid_spelling.py`
  (exact nodes, artifact-ref refusal with nothing written, unknown material,
  floor placement).
- `lint` and `typecheck` clean. The new worktree's fresh `.venv` first
  reported 54 unresolved-import diagnostics for optional extras; those cleared
  after `uv sync --group dev --group notebooks --group test --group lint` and
  were an environment artifact, not code.

## Third slice (retune side)

`checkpoints/recompute.py` could only express a retune target as
`(start, stop, step)`. The interesting part was not the missing feature but a
silent one: `recompute_defaults.uniform_bounds` *raises* on a nonuniform grid,
and that raise landed in the same `except (KeyError, TypeError, ValueError)`
branch that exists to tolerate derived stems with no catalog row. So a profile
declaring a graded continuum had its grid quietly discarded and every record
kept its own -- the same shape of silent drop as `effective_brem` in the first
slice.

- `repair_brem_wide` gained `brem_nodes`: an explicit coordinate target, used
  for the at-target comparison and written to the case through
  `encode_energy_grid`, which already stores an exact array. Combining it with
  `brem_start_eV`/`brem_stop_eV`/`brem_step_eV` is refused, as is a degenerate
  array.
- The retuned grid is used as-is for every beam energy rather than extended to
  `E0 + step`, matching `build_cases`, which passes a nonuniform grid straight
  through.
- `rebrem_checkpoints` now separates "no catalog row" from "graded catalog
  row": the `sweep(...)` lookup keeps its own `except`, and a nonuniform
  profile grid resolves to `brem_nodes` instead of falling through. An explicit
  `--brem-*` triple still wins.
- No new `rebrem` flags. The workflow is `brem set --spacing geometric` to
  declare the grid, then `checkpoint recompute brem --profile <p>` to retune
  onto it, so the target has one owner.

### Verified

- 7 new tests in `tests/scan/test_run.py`: graded retune writes exact
  coordinates to record and case, an already-graded record is skipped without
  recomputing, both refusals, and two driver tests pinning that a graded
  profile grid reaches `brem_nodes` while a uniform one still resolves
  `start`/`step` and passes `brem_nodes=None`.
- `lint`, `typecheck`, `format` clean.

### A live profile was already affected

`hopg`/`survey` resolves to a **graded** brem grid on `main` today: 281 nodes
carrying two step sizes (50 eV and 100 eV), the node refinement #100 landed.
So this was not a latent path waiting for someone to declare a geometric grid
-- `rebrem --fidelity survey` on `hopg` was silently keeping each record's own
grid instead of retuning onto the one a live sweep actually builds. It now
retunes, which means such records are correctly seen as not-at-target and are
recomputed.

`test_rebrem_profile_defaults_match_for_explicit_materials_and_all` had to be
made array-aware as a result: it compared whole kwargs dicts with `==`, and
one value is now a coordinate array. Rewritten to compare key by key, still
keyed by material name so it stays order-insensitive -- the original property
it was testing.

## Remaining on #153

- **Artifacts stay uniform-only.** A geometric continuum cannot currently be
  frozen into an immutable artifact. That costs nothing today (no shipped
  profile pins one) but would need an artifact schema v2 if artifacts come
  into use.
