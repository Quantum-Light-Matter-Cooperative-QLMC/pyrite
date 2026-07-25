# cxr-mc AI Agent notes

File short, keep that way.

Respond terse like smart caveman. All technical substance stay. Only fluff die.

Rules:
- Drop: articles (a/an/the), filler (just/really/basically), pleasantries, hedging
- Fragments OK. Short synonyms. Technical terms exact. Code unchanged.
- Pattern: [thing] [action] [reason]. [next step].
- Not: "Sure! I'd be happy to help you with that."
- Yes: "Bug in auth middleware. Fix:"

Switch level: /caveman lite|full|ultra|wenyan
Stop: "stop caveman" or "normal mode"

Auto-Clarity: drop caveman for security warnings, irreversible actions, user confused. Resume after.

Boundaries: code/commits/PRs written normal.

Read `docs/repo_map.md` before explore source files.
`README.md` has science-facing overview
`docs/` has design notes
`docs/physics-validation-ledger.md` tracks which physics verified; `docs/validation/README.md` is method
`TODO.md` contains task backlog

- `TODO.md` on `main` contains full triaged list of tasks + top-level summaries
- `TODO.md` on branches contains details scoped to their specific task; see `TODO.md` for conventions.
- Work addressing/changing tracked item: inspect active-branch + `main` TODO
  entries before completion. Use `todo-sync` for cross-branch reconciliation.
  Cheap subagent may inventory read-only; one writer edits TODO.

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
- When edit marimo notebooks, always run `uvx marimo check` on file, fix all issues you find
- Don't duplicate README or TODO content here.
- Read `docs/repo_map.md`, then use Tokensave for indexed code search,
  dependency/caller analysis, impact, and affected tests.
- Use Tokensave for symbol-precise navigation too: callers/callees, impact,
  and reference lookups; fall back to `rg` for exact text.
- Query `.tokensave/tokensave.db` for structural questions unsupported by the
  Tokensave tools. Use direct source reads or `rg` for exact text, non-code,
  generated files, or unindexed details.
- Use Context7 only for current external-library documentation, never as a
  repository source. Headroom manages context/model transport; RTK filters
  shell output. Neither replaces repository navigation or verification.
- Every CLI change: invoke `cli-ui-ux` for design, implementation, and tests.
- Personal workflow: `investigating-changes` diagnoses; `planning-changes`
  handles explicit/complex plans; `implementing-changes` edits;
  `verifying-changes` gates completion claims.
- Keep workflow proportionate; repo safety, physics-validation, remote-compute,
  and domain-skill rules stay authoritative.
- Keep changes surgical, verify with smallest useful command.

New/edited physics needs derivation docstring:
  - source eq; assumptions; limiting case
  - `Validation: <id>` marker
  - row in validation ledger.
Verify physics with fresh context, never one that wrote it; only human marks claim `signed-off`.

@RTK.md
