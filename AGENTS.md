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

- `TODO.md` is **identical on every branch and on `main`**: the full triaged
  backlog, one summary line per item + pointer. Branch-scoped detail lives in
  `tasks/<branch-leaf>.md` (see `tasks/README.md`), never in `TODO.md` and never
  in `docs/`. Keeping branch `TODO.md` == `main:TODO.md` prevents a
  fast-forward from silently clobbering the backlog; see `TODO.md` header for the
  full convention (incl. the merge-drop step for the `tasks/` file).
- Work addressing/changing tracked item: inspect the `tasks/<branch>.md` detail +
  the shared `TODO.md` before completion. Use `todo-sync` for cross-branch
  reconciliation. Cheap subagent may inventory read-only; one writer edits TODO.

## Canonical commands

This repo is `uv` managed.

Run all (offline) tests:

```bash
uv run python scripts/dev.py test
```

Run single test:

```bash
uv run python scripts/dev.py test path/to/test.py -k test_name
```

Run all tests (including online/external resources):

```bash
CXR_ONLINE_TESTS=1 uv run python scripts/dev.py test
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
- Add import same edit as its first usage, never earlier. Unused import between edits: lint/format autostrip, next edit re-add, loop repeat.

New/edited physics needs derivation docstring:
  - source eq; assumptions; limiting case
  - `Validation: <id>` marker
  - row in validation ledger.
Verify physics with fresh context, never one that wrote it; only human marks claim `signed-off`.

@RTK.md
