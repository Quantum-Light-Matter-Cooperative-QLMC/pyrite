# cxr-mc AI Agent notes

Keep this file short.

Read `docs/repo_map.md` before exploring source files.
`README.md` has the science-facing overview
`docs/` has the design notes
`docs/physics-validation-ledger.md` tracks which physics is verified; `docs/validation/README.md` is the method
`TODO.md` contains the task backlog

- `TODO.md` on `main` contains full triaged list of tasks + top-level summaries
- `TODO.md` on branches contains details scoped to their specific task; see `TODO.md` for conventions.

## Canonical commands

Run all tests:

```bash
uv run python scripts/dev.py test
```

Run a single test:

```bash
uv run python scripts/dev.py test path/to/test.py -k test_name
```

Lint:

```bash
uv run python scripts/dev.py lint
```

Format:

```bash
uv run python scripts/dev.py format
```

Type check:

```bash
uv run python scripts/dev.py typecheck
```

Notebook cleanup:

```bash
uv run python scripts/dev.py nbstrip
```

Run full verification:

```bash
uv run python scripts/dev.py verify
```

Run all pre-commit hooks:

```bash
uv run python scripts/dev.py precommit
```

Agents should always prefer these commands.

## Working rules

- Prefer edits in `src/cxr_mc/` over notebook logic.
- Notebook changes should stay output-free on commit.
- Do not duplicate README or TODO content here.
- When asked to locate something, use
  `uv run python scripts/dev.py repo-map` first.
- Keep changes surgical and verify with the smallest useful command.
- **Git commits:** the shared `pre-commit` hook may point at a stale worktree
  virtualenv. Run `uv run python scripts/dev.py precommit`, then in PowerShell
  prepend the current worktree's `.venv\Scripts` to `PATH` before `git commit`
  so the hook can find `pre-commit`.
- New/edited physics needs a derivation docstring (source eq, assumptions, a limiting case) + a `Validation: <id>` marker + a row in the validation ledger. Verify physics with a fresh context, never the one that wrote it; only a human marks a claim `signed-off`.
