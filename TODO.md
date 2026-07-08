# TODO — feature/checkpoint-lifecycle

Checkpoint lifecycle tooling: `cxr slim --grid`, `cxr archive` / `cxr restore` /
`cxr archives`, `remote.py pull --grid`, and `remote.py clear`. Core landed in
`e2bb58a`; the five verification-pass fixes (clear exit-status merge blocker,
`scan --quick --grid` guard, unknown-stem `slim --grid` error, repo-anchored
`archive.DEFAULT_ROOT`, cosmetics) landed in `5fc6751` with regression tests.
Suite/ruff/pyright green.

Remaining on this branch:

1. **Re-verify `remote.py clear` against the qlmc box** post-`5fc6751`. The listing
   loop's `[ -f "$f" ] && echo "$f"` exit-status bug made ssh exit 1 whenever
   `<material>_quick.pkl` was absent; the `|| true` fix is unit-tested against real
   bash but has not been re-driven on the box. Then decide merge to `main`.
2. **Default `remote.py pull <material>` to `--grid`.** Make the grid-filtered pull
   the default and move the current full-pickle pull behind a `-f/--full` flag.
