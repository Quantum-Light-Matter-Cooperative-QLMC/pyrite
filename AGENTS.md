# cxr-mc agent contract

Keep replies terse; preserve technical substance. Fragments OK. Drop filler,
pleasantries, and hedging. Use normal clarity for security, irreversible
actions, or ambiguity. Code, commits, PR text, quotations, and user-requested
prose keep required format.

## Start here

- Read `docs/repo_map.md` before source exploration.
- Use Tokensave first for indexed code search, callers/callees, impact, and
  affected tests. Query `.tokensave/tokensave.db` for unsupported structural
  queries. Use `rg` or direct reads for exact text, non-code, generated files,
  and unindexed details.
- `README.md`: science/install/workflow. `docs/`: guides, design, validation.
  `TODO.md`: shared backlog.
- Use Context7 only for current external-library docs. RTK filters verbose shell
  output; do not wrap already-small commands.

## Canonical commands

Use project runner; never bare `pytest`, `.venv/bin/python`, or path-hack imports.

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test path/to/test.py -k test_name
CXR_ONLINE_TESTS=1 rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py lint
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py format
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py typecheck
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py nbstrip
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py precommit
```

If `uv run` cannot write project environment, add
`UV_PROJECT_ENVIRONMENT=/tmp/cxr-mc-venv`; do not switch interpreters.

## Work rules

- Make smallest owning change. Preserve unrelated dirty-tree work.
- Prefer reusable logic in `src/cxr_mc/`; keep marimo apps thin. Keep notebooks
  output-free. Run `uv run marimo check <app.py>` after marimo edits.
- Invoke matching repo skill for CLI, notebooks, Monte Carlo, performance,
  physics, docs, runtime, remote GPU, regression, catalog-golden, or TODO work.
- CLI changes preserve documented command/help/output/exit contracts and
  regenerate `docs/cli-reference.md`.
- Heavy sweeps/GPU work use `cxr remote`; never run locally.
- Add imports with first use. Verify with smallest useful command.

## Task dispatch

- Use `triage` for new `>user<` prose. It drafts task doc and local
  branch/worktree, syncs TODO pointer, then stops before commit/push for review.
- Use `dispatch-task` to route backlog work. It resolves task worktree/branch,
  owns approved setup/landed teardown, writes explicit authority/acceptance
  handoff, then selects:
  `implement-task-lite` for small mechanical slices, `implement-task` for
  normal checklist slices, or `lead-task` for complex/integrating ownership.
- Example model tiers: Haiku/Luna → lite; Sonnet/Terra → normal;
  Opus/Sol/Fable/K3 → lead. Risk and scope override model label.
- Task type stays separate: workers also invoke matching CLI, notebook,
  physics, performance, docs, remote, regression, or scientific skill.
- Direct user invocation of a worker skill permits task-local checkpoint
  commits unless user says otherwise; never permits push, TODO ownership, or
  delegation. Supervisors may pass only authority they hold.
- Checkpoint only independently valid slices: focused checks pass, scoped diff
  reviewed, explicit paths staged. Never `git add .`; never mix unrelated WIP.

## Backlog and physics

`TODO.md` stays identical across branches. Branch detail belongs in
`tasks/<branch-leaf>.md`; see `tasks/README.md`. Work touching tracked items
uses `todo-sync` only for exact cross-branch consistency.

New/edited physics requires source equation, assumptions, limiting case,
`Validation: <id>`, and ledger row. Fresh context verifies it; only human marks
`signed-off`.
