# PyRITE agent contract

Keep replies terse; preserve technical substance. Fragments OK. Drop filler,
pleasantries, and hedging. Use normal clarity for security, irreversible
actions, or ambiguity. Code, commits, PR text, quotations, and user-requested
prose keep required format.

## Start here

- Read `docs/repo_map.md` before source exploration.
- Use Serena for symbol-level navigation: definitions, references, callers, and
  call sites across the CLI/domain boundary. Use `rg` or direct reads for exact
  text, non-code, generated files, and anything faster to grep — `src/` is ~52k
  LOC, so grep is competitive for most questions.
- `README.md`: science/install/workflow. `docs/`: guides, design, validation.
  `TODO.md`: shared backlog.
- Use Context7 only for current external-library docs. Headroom shapes agent and
  tool output; it is not a shell wrapper or code index. Do not use Tokensave or
  RTK.

## Canonical commands

Use project runner; never bare `pytest`, `.venv/bin/python`, or path-hack imports.

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite core
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite apps
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite packaging
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test path/to/test.py -k test_name
PYRITE_ONLINE_TESTS=1 UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test --cov
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test --numba --cov
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev format
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev nbstrip
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev precommit
```

If `uv run` cannot write project environment, add
`UV_PROJECT_ENVIRONMENT=/tmp/pyrite-venv`; do not switch interpreters.

## Work rules

- Make smallest owning change. Preserve unrelated dirty-tree work.
- Prefer reusable logic in `src/cxr_mc/`; keep marimo apps thin. Keep notebooks
  output-free. Run `uv run marimo check <app.py>` after marimo edits.
- Invoke matching repo skill for CLI, notebooks, Monte Carlo, performance,
  physics, docs, runtime, remote GPU, regression, catalog-golden, or TODO work.
- CLI changes preserve documented command/help/output/exit contracts and
  regenerate `docs/repo-design/cli/cli-reference.md`.
- Heavy sweeps/GPU work use `pyrite remote`; never run locally.
- Add imports with first use. Verify with smallest useful command.

## Task dispatch

- Use `triage` for new `>user<` prose or `/triage <text>` direct input. It
  drafts task docs and local branches/worktrees, syncs TODO pointers, commits
  that setup on `main`, pushes `main` and the task branches, then stops for
  review.
- Use `dispatch-task` to route backlog work. It resolves task worktree/branch,
  verifies approved setup, owns landed teardown, writes explicit
  authority/acceptance handoff, then selects:
  `implement-task-lite` for small mechanical slices, `implement-task` for
  normal checklist slices, or `lead-task` for complex/integrating ownership.
- Example model tiers: Haiku/Luna → lite; Sonnet/Terra → normal;
  Opus/Sol/Fable/K3 → lead. Risk and scope override model label.
- Use `repo-cleanup` for git hygiene. It fans cheap subagents out per item to
  rebase task branches onto `main` (clean or `TODO.md`-only conflicts only;
  real conflicts are aborted and reported), prune stale worktree/tracking
  entries, remove clean worktrees on merged branches, `-d` delete fully
  merged locals, and delete fully merged remote branches. Never pushes
  otherwise without explicit instruction.
- Task type stays separate: workers also invoke matching CLI, notebook,
  physics, performance, docs, remote, regression, or scientific skill.
- Direct user invocation of a worker skill permits task-local checkpoint
  commits unless user says otherwise; never permits push, TODO ownership, or
  delegation. Supervisors may pass only authority they hold.
- Checkpoint only independently valid slices: focused checks pass, scoped diff
  reviewed, explicit paths staged. Never `git add .`; never mix unrelated WIP.

## Backlog and physics

`TODO.md` is authoritative on `main`; branch copies are disposable and
auto-resolve to `main` on merge/rebase via the `TODO.md merge=ours` driver (run
`uv run pyrite-dev bootstrap` once per clone). Edit and drop items on `main`.
Tracked agent plans and handoffs live only in `agentdocs/`, outside the public
documentation tree. Branch detail belongs in
`agentdocs/tasks/<branch-name>/` (full task branch name; entry doc `README.md`);
see `agentdocs/README.md`. `todo-sync` now only checks main's accuracy and that
the driver is installed.

New/edited physics requires source equation, assumptions, limiting case,
`Validation: <id>`, and ledger row. Fresh context verifies it; only human marks
`signed-off`.
