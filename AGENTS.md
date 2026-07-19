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

Run shell commands through `rtk`. For `uv`, always use the shared writable
cache: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv ...`.

Run all tests:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test
```

Run single test:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test path/to/test.py -k test_name
```

Lint:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py lint
```

Format:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py format
```

Type check:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py typecheck
```

Notebook cleanup:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py nbstrip
```

Run full verification:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify
```

Run all pre-commit hooks:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py precommit
```

Agents prefer these commands always.

## Working rules

- Prefer edits in `src/cxr_mc/` over notebook logic.
- Notebook changes stay output-free on commit.
- Don't duplicate README or TODO content here.
- Read `docs/repo_map.md`, then use Tokensave for indexed code search,
  dependency/caller analysis, impact, and affected tests.
- Use Serena for symbol-precise navigation, references, renames, and LSP
  diagnostics when its language server is healthy.
- Query `.tokensave/tokensave.db` for structural questions unsupported by the
  Tokensave tools. Use direct source reads or `rg` for exact text, non-code,
  generated files, or unindexed details.
- Use Context7 only for current external-library documentation, never as a
  repository source. Headroom manages context/model transport; RTK filters
  shell output. Neither replaces repository navigation or verification.
- Treat plugin workflows as optional helpers. Repo safety, physics-validation,
  remote-compute, and verification rules remain authoritative.
- Keep changes surgical, verify with smallest useful command.

New/edited physics needs derivation docstring:
  - source eq; assumptions; limiting case
  - `Validation: <id>` marker
  - row in validation ledger.
Verify physics with fresh context, never one that wrote it; only human marks claim `signed-off`.
