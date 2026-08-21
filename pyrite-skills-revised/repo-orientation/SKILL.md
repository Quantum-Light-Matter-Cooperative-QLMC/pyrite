---
name: repo-orientation
description: Use when locating PyRITE code, choosing the smallest owner, tracing dependencies/tests, or updating the repository map.
---

# Repo Orientation

Use the cheapest navigation method that answers the question.

1. Read `docs/repo_map.md` when package/ownership context is needed; skip it for
   an already-known local owner.
2. Use `rg`/direct reads for exact text, filenames, non-code, generated files,
   branch context, and nearby tests. Use Serena when definition/reference/call
   graph navigation is materially better.
3. Prefer `src/pyrite/` implementations, `tests/` fast CPU checks, `checks/`
   physics anchors, and thin marimo apps. Find an existing helper before adding
   a new abstraction.
4. Identify the smallest owning path, affected callers, and focused tests. Do
   not broaden into unrelated README/docs/backlog edits.
5. Regenerate with `pyrite-dev repo-map` only when packages, entry points,
   ownership, or agent-tooling top-levels actually changed.

Use Context7 only for current external-library documentation. Headroom is output
shaping, not a repository index. Do not use Tokensave or RTK.
