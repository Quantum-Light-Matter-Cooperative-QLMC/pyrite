---
name: repo-orientation
description: Use when locating code, choosing owners, assessing cxr-mc structure, updating repository map, or planning surgical changes.
---

# Repo Orientation

1. Read `docs/repo_map.md`.
2. Use Tokensave for indexed search, callers/callees, dependencies, impact,
   branch context, and affected tests.
3. Query `.tokensave/tokensave.db` for unsupported structural queries.
4. Use `rg` or direct reads for exact text, non-code, generated files, runtime
   artifacts, and unindexed details.

Prefer `src/cxr_mc/` implementations, `tests/` fast CPU checks, `checks/`
physics anchors, and thin marimo apps. Legacy
`checks/cxr_analysis_feranchuk.ipynb` is not an app owner. Do not rewrite
README/TODO/docs unless task targets them.

Find smallest owner and existing helper before editing. Prefer focused test over
broad refactor. Context7 is only for current external-library docs; Headroom and
RTK are not code indexes.

Regenerate repo inventory with `scripts/dev.py repo-map` after packages, entry
points, or agent-tooling top-levels change.
