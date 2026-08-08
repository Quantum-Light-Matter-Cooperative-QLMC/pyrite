---
name: repo-orientation
description: Use when locating code, choosing owners, assessing cxr-mc structure, updating repository map, or planning surgical changes.
---

# Repo Orientation

1. Read `docs/repo_map.md`.
2. Use Serena for symbol-level search: definitions, references, callers/callees,
   and dependencies.
3. Use `rg` or direct reads for exact text, non-code, generated files, runtime
   artifacts, branch context, and affected tests. `src/` is ~52k LOC; grep is
   competitive for most questions.

Prefer `src/cxr_mc/` implementations, `tests/` fast CPU checks, `checks/`
physics anchors, and thin marimo apps. Legacy
`checks/cxr_analysis_feranchuk.ipynb` is not an app owner. Do not rewrite
README/TODO/docs unless task targets them.

Find smallest owner and existing helper before editing. Prefer focused test over
broad refactor. Context7 is only for current external-library docs; Headroom and
RTK are not code indexes.

Regenerate repo inventory with `cxr-dev repo-map` after packages, entry
points, or agent-tooling top-levels change.
