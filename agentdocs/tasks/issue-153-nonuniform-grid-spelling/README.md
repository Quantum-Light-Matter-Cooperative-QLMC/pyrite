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

## Remaining on #153

- **Write side.** `pyrite material energy-grid brem set` has no spelling for a
  nonuniform grid, and `_merge_brem` emits `arange` only. Artifact identity
  stores brem as `{start_eV, stop_eV, step_eV}`, so this is a stored-identity
  change, not only a writer change, and will move digests.
- **Retune side.** `checkpoints/recompute.py` retunes brem through
  `(start, stop, step)` only (`:482`, `:525-526`).
- Both change documented CLI command/help/output contracts and will need
  `docs/repo-design/cli/cli-reference.md` regenerated, which this slice did
  not.
