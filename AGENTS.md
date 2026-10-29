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
  Serena starts with `--project-from-cwd`, so every worktree is pre-activated:
  skip `activate_project`, onboarding, and the instructions manual unless
  troubleshooting Serena. No hook forces Serena use (`serena-hooks remind`
  denied ordinary Read-before-Edit loops); choosing the right tool is on you.
- `README.md` owns science/install/primary workflow. Before writing docs,
  classify the reader's need (tutorial, how-to, reference, or explanation),
  find the canonical page, and follow `docs/repo-design/documentation.md`.
  Use `tutorial-authoring` for `docs/notebooks/`, `documentation-maintenance`
  for other docs; preserve API ownership, units, evidence and stable links.
  Backlog: GitHub Issues
  (`gh issue list`). Status labels (one per open issue): `needs-triage`,
  `backlog`, `active`, `gated`, `paused`. Area labels: `physics`, `cli`,
  `api`, `notebooks`, `tests`, `performance`, `remote`, `geometry`,
  `validation`, `materials`, `infrastructure`, `refactor`. Type: built-in
  `bug`/`enhancement`/`documentation`/`question`. Legacy `status:backlog`/`status:active` equal
  their bare forms; replace them with the bare label when editing an issue.
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

Before the first sandboxed docs build, set writable uv tool directories:

```bash
export UV_TOOL_DIR=/tmp/pyrite-uv-tools
export UV_TOOL_BIN_DIR=/tmp/pyrite-uv-tool-bin
```

Apply these to `pyrite-dev docs`, `verify`, `precommit`, and direct Sphinx
invocations. The marimo HTML-WASM demo export invokes Ruff through uv's tool
runner. `UV_CACHE_DIR` alone does not redirect its default
`~/.local/share/uv/tools` writes; a read-only home causes
`LocalWheelError: Failed to analyze local imports with Ruff` / `Could not create
temporary file`. Set the tool directories up front and keep demo export enabled.

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
supported Python versions. `verify` runs types, lint/format checks, source
line budgets, the complexity ratchet, API checks, docs, and import-linter; it
never runs pytest or changes formatting. HTML docs builds validate and render
committed native MyST-NB tutorial snapshots without executing tutorials. Explicit regeneration uses
`docs/_notebooks.py`; commit affected `.ipynb` captures with their Marimo sources.
Do not add separate tutorial provenance manifests or HTML/session exports.

## Work rules

- Make smallest owning change. Preserve unrelated dirty-tree work.
- Prefer reusable logic in `src/pyrite/`; keep marimo apps thin. Keep notebook
  sources output-free; tutorial snapshot regeneration follows `tutorial-authoring`.
  Run `uv run marimo check <app.py>` after marimo edits.
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
- Public API changes follow [the API checklist](docs/development/api-policy.md#before-approving-a-public-feature):
  owner/status, canonical imports and aliases, compatibility, units/defaults,
  dependencies, docs/tests, and serialization/scientific identity. Update
  `api-surface.toml`; run `pyrite-dev api check` and `api check --imports`.
- After editing `src/pyrite/` routines, run `pyrite-dev complexity --check`. On
  a new or worsened CC > 20 routine, simplify it; if the branching is inherent,
  run `complexity --write` and justify each added/raised baseline entry in the
  PR. Never split code only to pass the metric. Policy:
  `docs/repo-design/complexity.md`.
- New tests >5 s wall or with large peak RSS get `@pytest.mark.slow`
  (default runs skip them; CI runs them). Check with `--durations=0`.
- When a `*_MODEL` marker or `tables-*-N` tag changes, `rg` every identity
  digest/pin derived from it and re-pin all in one pass before committing.

## Shell, runs, and shipping

- Claude Code's tool shell is pinned to bash (`CLAUDE_CODE_SHELL` in
  `.claude/settings.json`), but other agents and humans may run zsh, which
  does not word-split unquoted `$var`:
  `for f in $files` loops once over the whole string. Write commands that
  behave the same in bash and zsh: arrays (`files=(a b)`; `"${files[@]}"`),
  newline loops (`cmd | while IFS= read -r f; do ...; done`), or
  `xargs -0`/`git ... -z`; never rely on splitting a string variable.
  Wrap bash-only scripts in `bash -c '...'`; zsh also aborts on unmatched
  globs, so quote patterns passed to `rg`, `git` and `find`. Run `gh` from
  inside the worktree.
- Never mask a gate's exit code: no `| tail`/`| grep` on lint, test or verify
  without `set -o pipefail`; prefer `cmd > scratch/x.log 2>&1; echo rc=$?`.
- Local tests: `pyrite-dev test` sizes workers by memory; never raise `-n`.
  That budget is a heuristic, not a memory limit. Full `--slow` audits and
  isolated multi-GiB tests belong on the remote box; see the measured
  [slow-test audit](docs/repo-design/compute/slow-test-audit.md). For parallel
  CPU audits, start with `OPENBLAS_NUM_THREADS=1` to avoid multiplying BLAS
  threads by xdist workers; benchmark larger thread counts explicitly.
  Cap heavy local probes with
  `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 ...`.
  After a squash/rebase with no content change, rerun only focused tests;
  leave domain suites to CI.
- Format only changed paths (`uv run ruff format <paths>`); CI formats PRs.
  Repo-wide `pyrite-dev format` only when asked.
- Kill processes by PID (`pgrep -a`, then `kill <pid>`); never `pkill`/`killall`
  patterns, which can match the agent's own shell. Never `git add .`/`-A`.
  `.agents/hooks/guard_bash.sh` blocks these and non-lease force pushes.
- Ship in checkpoints: with push authority, commit, push and open a draft PR
  before any remote/GPU job or long check, so a session limit strands only
  evidence. `ship` owns the verify → rebase → push → PR → issue sequence.
- Remote runs start with the `remote-gpu-jobs` pre-flight (synced SHA, node
  state, `uv` on PATH, explicit catalog env). Record job IDs in the issue's
  `## Progress` section instead of blocking on them.
- Zotero: an empty search or children listing is not proof of absence. Retry,
  search attachments by parent title/DOI, then ask the user before concluding a
  source or SI is missing.

## Task dispatch

- Use `triage` for new `>user<` prose, GitHub issues carrying no status
  label, or `/triage <text>` direct input. It turns the task into a canonical
  GitHub issue (executable body, area/type labels, `backlog`,
  native dependency relations), then stops for review. It never implements and
  creates no branches, worktrees, task docs, commits, or pushes.
- Use `dispatch-task` to route reviewed backlog work. It creates or reuses the
  linked `issue-<n>-<slug>` branch/worktree (`.worktrees/`), moves the issue to
  `active`, writes explicit authority/acceptance handoff, then selects:
  `implement-task-lite` for small mechanical slices, `implement-task` for
  normal checklist slices, or `lead-task` for complex/integrating ownership.
  It also owns landed teardown: close the issue, promote durable content, hand
  worktree/ref removal to `repo-cleanup`.
- Use `ship` (`/ship [checkpoint|final]`) to publish a task branch: verify,
  rebase onto `main`, check remote/PR state, push, open or update the PR, and
  record progress in the issue. Direct invocation grants push/PR/issue-writer
  for the current issue branch only.
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
repo file. One issue per item, with one status label plus area/type labels;
the issue body is the sole task record (goal, scope, plan checklist,
acceptance, decisions).
`dispatch-task` closes an issue when its task lands. Cross-task agent plans
live only in `agentdocs/plans/`, outside the public documentation tree; see
`agentdocs/README.md`. `issue-audit` audits backlog/branch consistency
read-only.

New/edited physics requires source equation, assumptions, limiting case,
`Validation: <id>`, and ledger row. Fresh context verifies it; only human marks
`signed-off`. Agent task "done" for a claim is `rederived` or `anchored` per
the task, never `signed-off`; human sign-off is tracked only in #277 (see
`docs/validation/methodology.md`).
