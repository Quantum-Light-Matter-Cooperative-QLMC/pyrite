# PyRITE agent contract

Keep replies terse; preserve technical substance. Fragments OK. Drop filler,
pleasantries, and hedging. Use normal clarity for security, irreversible
actions, or ambiguity. Code, commits, PR text, quotations, and user-requested
prose keep required format.

## Start here

- Read `docs/repo_map.md` before source exploration.
- Use Serena for symbol-level navigation: definitions, references, callers, and
  call sites across the CLI/domain boundary. Use `rg` or direct reads for exact
  text, non-code, generated files, and targeted searches. `src/` is ~128k LOC;
  prefer Serena for cross-file symbol relationships and avoid broad source dumps.
- `README.md`: science/install/primary workflow. Classify maintained pages under
  `docs/guides/`, `docs/physics/`, `docs/validation/`, `docs/research/`,
  `docs/repo-design/`, or `docs/adr/`; see
  `docs/repo-design/documentation.md`. Backlog: GitHub Issues
  (`gh issue list`), labelled `status:active`/`gated`/`paused`/`backlog`
  and `area:*`.
- Use Context7 only for current external-library docs. Headroom shapes agent and
  tool output; it is not a shell wrapper or code index. Do not use Tokensave or
  RTK.

## Canonical commands

Use project runner; never bare `pytest`, `.venv/bin/python`, or path-hack imports.

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test path/to/test.py -k test_name
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite transport
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite transport-collisions
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite transport-cascades
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite transport-runner
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite spectrum
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite spectrum-coherent
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite spectrum-brem
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite quantum
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite channeling
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite energy-grid
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite materials
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite xsgen
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite core
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite apps
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite packaging
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test --slow path/to/test.py
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev format
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev check-lines
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev complexity --check
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev nbstrip
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev precommit
```

If `uv run` cannot write project environment, add
`UV_PROJECT_ENVIRONMENT=/tmp/pyrite-venv`; do not switch interpreters.

## Choose tests by impact

Start with the changed behavior's test node or owning module. Trace callers and
consumers before selecting neighboring tests; use a domain suite only when the
change spans that domain or its shared fixtures. Never default to the full
suite, `core`, or `integration` for a local edit. `core` is a residual partition,
not an aggregate safety check. Re-run only after a new change, failure, or
unresolved concern. Inert docs/guidance edits need their relevant static checks,
not simulation tests.

| Changed surface | Smallest useful coverage; widen only if affected |
|---|---|
| Flight integration, beam/RNG, geometry | Matching `tests/montecarlo/test_*.py`; add affected runner/segment/spectrum consumers when event or trajectory contracts change. Broader: `transport`. |
| Collision/stopping/straggling physics | Matching atomic, shell, elastic or stopping tests. Broader: `transport-collisions`; radiative flights, secondaries, pair/positron cascades: `transport-cascades`. |
| Runner/resource/trajectory plumbing | Matching adaptive/block/dispatch/artifact tests. Broader: `transport-runner`. |
| Radiation scoring/reduction | Matching spectrum tests; add affected grid/result consumers for changed grid/shape/normalization contracts. Broader: `spectrum` for shared/incoherent CXR, `spectrum-coherent` for coherent/temporal, `spectrum-brem` for bremsstrahlung. These are disjoint, not aggregates. |
| Quantum channeling physics | Bound-state, density-matrix, MCWF, scattering-operator or dipole-emission module. Broader: `quantum`. |
| Channeling integration | Matching selection/handoff/replay/producer tests. Broader: `channeling`; add quantum solver tests only if that physics changed. |
| Energy-grid policy/artifacts | Matching `tests/energy-grid/` modules and affected spectrum consumers. Broader: `energy-grid`; CLI wiring stays in `cli`. |
| Catalog/materials/tables | Matching `tests/materials/` or `tests/xsgen/` modules; use `regen-golden` only for authorized catalog changes and check affected grid/transport consumers. Broader: `materials`/`xsgen`. |
| CLI or app | Affected command/app module, plus owning domain tests only if underlying behavior changed. Broader: `cli`/`apps`. |
| Checkpoints/results/instrument/API | Matching modules under `tests/checkpoint/`, `results/`, `observations/`, `detectors/`, `instrument/`, or `scan/`; add concrete serialization/public callers, not all of `core`. |
| Developer runner/suite ownership | `tests/dev/test_dev_cli.py`; packaging/import changes add corresponding public export/package tests. Broader: `packaging`. |

Select slow tests explicitly by node ID, or put `--slow` before test paths;
do not enable every slow test just to exercise one changed path. Full
`pyrite-dev test --slow` and CPU `test --slow --numba --cov` are explicit
release/audit operations; CI runs every primary domain with slow tests on both
supported Python versions. `verify` runs only types, lint/format checks, source
line budgets, the complexity ratchet, docs, and import-linter; it never runs
tests or changes formatting.

## Work rules

- Make smallest owning change. Preserve unrelated dirty-tree work.
- Prefer reusable logic in `src/pyrite/`; keep marimo apps thin. Keep notebooks
  output-free. Run `uv run marimo check <app.py>` after marimo edits.
- Invoke matching repo skill for CLI, notebooks, Monte Carlo, performance,
  physics, docs, runtime, remote GPU, regression, catalog-golden, or backlog
  work.
- CLI changes preserve documented command/help/output/exit contracts and
  regenerate `docs/repo-design/cli/cli-reference.md`.
- Never bump the version outside a release PR (`docs/repo-design/releasing.md`,
  `release` skill). A PR that changes default numerical output bumps the
  relevant `*_MODEL` marker or `tables-*-N` tag in that PR.
- Heavy sweeps/GPU work use `pyrite remote`; never run locally.
- Add imports with first use. Verify with smallest useful command.
- After editing `src/pyrite/` routines, run `pyrite-dev complexity --check`. On
  a new or worsened CC > 20 routine, simplify it; if the branching is inherent,
  run `complexity --write` and justify each added/raised baseline entry in the
  PR. Never split code only to pass the metric. Policy:
  `docs/repo-design/complexity.md`.
- New tests >5 s wall or with large peak RSS get `@pytest.mark.slow`
  (default runs skip them; CI runs them). Check with `--durations=0`.

## Task dispatch

- Use `triage` for new `>user<` prose, GitHub issues carrying no `status:*`
  label, or `/triage <text>` direct input. It turns the task into a canonical
  GitHub issue (executable body, `area:*`/type labels, `status:backlog`,
  native dependency relations), then stops for review. It never implements and
  creates no branches, worktrees, task docs, commits, or pushes.
- Use `dispatch-task` to route reviewed backlog work. It creates or reuses the
  linked `issue-<n>-<slug>` branch/worktree (`.worktrees/`), moves the issue to
  `status:active`, writes explicit authority/acceptance handoff, then selects:
  `implement-task-lite` for small mechanical slices, `implement-task` for
  normal checklist slices, or `lead-task` for complex/integrating ownership.
  It also owns landed teardown: close the issue, promote durable content, hand
  worktree/ref removal to `repo-cleanup`.
- Example model tiers: Haiku/Luna → lite; Sonnet/Terra → normal;
  Opus/Sol/Fable/K3 → lead. Risk and scope override model label.
- Use `repo-cleanup` for git hygiene. It fans cheap subagents out per item to
  rebase task branches onto `main`, prune stale worktree/tracking entries,
  remove clean worktrees on merged branches, `-d` delete fully merged locals,
  and delete fully merged remote branches. Never pushes otherwise without
  explicit instruction.
- Task type stays separate: workers also invoke matching CLI, notebook,
  physics, performance, docs, remote, regression, or scientific skill.
- Direct user invocation of a worker skill permits task-local checkpoint
  commits unless user says otherwise; never permits push, issue ownership, or
  delegation. Supervisors may pass only authority they hold.
- Checkpoint only independently valid slices: focused checks pass, scoped diff
  reviewed, explicit paths staged. Never `git add .`; never mix unrelated WIP.

## Backlog and physics

Backlog is tracked in GitHub Issues (`gh issue list`/`gh issue view`), not a
repo file. One issue per item, labelled `status:*`/`area:*`; the issue body is
the sole task record (goal, scope, plan checklist, acceptance, decisions).
`dispatch-task` closes an issue when its task lands. Cross-task agent plans
live only in `agentdocs/plans/`, outside the public documentation tree; see
`agentdocs/README.md`. `issue-audit` audits backlog/branch consistency
read-only.

New/edited physics requires source equation, assumptions, limiting case,
`Validation: <id>`, and ledger row. Fresh context verifies it; only human marks
`signed-off`. Agent task "done" for a claim is `rederived` or `anchored` per
the task, never `signed-off`; human sign-off is tracked only in #277 (see
`docs/validation/methodology.md`).
