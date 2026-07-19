# cxr-mc AI Agent notes

File short, keep that way.

Read `docs/repo_map.md` before explore source files.
`README.md` has science-facing overview
`docs/` has design notes
`docs/physics-validation-ledger.md` tracks which physics verified; `docs/validation/README.md` is method
`TODO.md` contains task backlog

- `TODO.md` on `main` contains full triaged list of tasks + top-level summaries
- `TODO.md` on branches contains details scoped to their specific task; see `TODO.md` for conventions.

## Canonical commands

Run all tests:

```bash
uv run python scripts/dev.py test
```

Run single test:

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

Agents prefer these commands always.

## Working rules

- Prefer edits in `src/cxr_mc/` over notebook logic.
- Notebook changes stay output-free on commit.
- Don't duplicate README or TODO content here.
- Locate something: use `uv run python scripts/dev.py repo-map` first.
- `serena` available for semantic search
  - `grep` and other tools last resort
- Keep changes surgical, verify with smallest useful command.

New/edited physics needs derivation docstring:
  - source eq; assumptions; limiting case
  - `Validation: <id>` marker
  - row in validation ledger.
Verify physics with fresh context, never one that wrote it; only human marks claim `signed-off`.