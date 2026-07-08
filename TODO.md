# TODO — feature/checkpoint-lifecycle

Checkpoint lifecycle tooling: `cxr slim --grid`, `cxr archive` / `cxr restore` /
`cxr archives`, `remote.py pull` (grid-filtered by default), and `remote.py clear`.
Core landed in `e2bb58a`; the five verification-pass fixes (clear exit-status merge
blocker, `scan --quick --grid` guard, unknown-stem `slim --grid` error, repo-anchored
`archive.DEFAULT_ROOT`, cosmetics) landed in `5fc6751` with regression tests. `clear`
was re-verified live against the qlmc box (dry preview against `hopg`, which lacks a
`_quick.pkl` sibling — exactly the case that used to trip the exit-status bug — now
exits 0). `pull` now defaults to `--grid`; the old whole-file pull moved behind
`-f/--full`. Suite/ruff/pyright green. Ready to merge to `main`.
