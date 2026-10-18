# Source complexity baseline

`pyrite-dev complexity` reports source complexity and enforces one narrow
ratchet: no new or worsened routine above cyclomatic complexity (CC) 20. It
exposes new branching without blocking on historical debt or rewarding
metric-only rewrites. Implementation: `pyrite.devtools.complexity`; baseline:
`docs/repo-design/complexity-baseline.json`.

```bash
pyrite-dev complexity                    # report: inventory, CC ranks, hotspots, MI
pyrite-dev complexity --json             # same report as JSON
pyrite-dev complexity --churn 300        # add git churn over the last 300 commits
pyrite-dev complexity --scope tests      # analyze tests separately (report only)
pyrite-dev complexity --check            # ratchet; also run by `pyrite-dev verify`
pyrite-dev complexity --write            # rewrite the baseline, print its changes
```

## Tools and scope

| Measure | Tool | Role |
|---|---|---|
| Per-routine CC | Radon, pinned `radon==6.0.1` in the `lint` group | Gate (`--check`) and report |
| Maintainability index (MI) | Radon `mi_visit` | Informational only |
| Line inventory | `scc` on `PATH`, or `PYRITE_SCC=/path/to/scc` | Informational, optional |
| Churn | `git log --numstat --no-renames`, scope pathspec only | Informational, opt-in |

A missing Radon fails with the `uv sync` command that restores it. A missing or
failing `scc` is reported in the inventory line and never fails a run; point
`PYRITE_SCC` at the installed binary when a launcher wrapper (a snap shim that
fails DBus setup, say) breaks. No local install path is hardcoded.

The source scope is `src/pyrite`; `--scope tests` scans `tests/` separately and
cannot be ratcheted. Paths containing `vendor`, `build`, `dist`, `_build`,
`__pycache__`, `.venv`, `venv`, `site-packages`, `node_modules`, `.worktrees`
or `.git` are skipped. Every `def`, including methods, closures, and defs inside
nested classes, is scored once by file, line, and column; class aggregates are
not reported. A file that does not parse is listed as a parse failure: the
report exits 1, `--check` fails, and `--write` refuses to run.

Each report states the scope, `git` HEAD, Radon and Python versions, threshold,
and exclusions. Output is sorted, so the same tree and tool versions give the
same report. `--check` scans without MI and takes a few seconds.

## Ratchet policy

A *hotspot* is a routine with CC above 20, Radon ranks D–F. Baseline entries
are keyed by module path and dotted lexical name (`Class.method`,
`outer.inner`); a repeated name in one module, such as a property setter, gets
`#2`, `#3` in source order. Line numbers are not part of the key, so edits
elsewhere in a file do not disturb it.

| Current state | `--check` |
|---|---|
| Hotspot at its baseline CC | Pass |
| Hotspot above its baseline CC | Fail: worsened |
| Hotspot with no baseline entry | Fail: new |
| Baseline entry below its CC, or no longer a hotspot | Pass; suggests `--write` |
| Parse failure, or Radon version differs from the baseline's | Fail |

Moves and renames change the key, so a moved hotspot fails as new, and the
message names resolved baseline entries with the same leaf name or CC as
possible origins. Accepting it means `--write`, which prints every added,
raised, lowered, and removed entry; the baseline diff carries the move into
review, so debt cannot disappear silently. Resolved and improved entries
are removed or lowered by the next `--write`, locking in the improvement.

Prefer simplifying a new or worsened routine. When the branching is inherent,
run `--write` and justify each added or raised entry in the PR. Do not split a
routine only to pass the gate.

## Kernel exceptions

Entries may carry an `exception` note, which `--write` keeps while the routine
remains a hotspot. Exceptions are capped at their recorded CC like every other
entry. Three numerical kernels carry one: the CPU transport body
(`make_cpu_transport_core.body`, CC 227), the exact CUDA kernel
(`_transport_kernel`, 140) and the LUT kernel (`_transport_lut_kernel`, 88).
Restructuring them needs RNG-stream, event-order, CPU/GPU parity, and
performance evidence; parity work stays in #344.

Secondary cleanup candidates (`perf/performance_analysis.py` and
`console/dashboard/render.py`, MI near 0 with CC 44 hotspots) are recorded in
the baseline as ordinary debt. Their refactors need separate scope.

## Limitations

- CC counts decision paths. It does not measure bugs, and it is not the number
  of tests a routine needs.
- MI depends on comments and line counts, so it says nothing about correctness.
  There is no aggregate MI or mean-CC gate.
- `scc` complexity is a per-file count of branch tokens. It does not equal Radon
  CC, and `scc` code-line totals differ from physical line counts.
- Test lines of code are not coverage; coverage is #538's scope.
- Churn counts moves as full deletes plus adds and is not a gate.
- Radon skips lambdas, and its CC can change between releases, which is why
  the baseline pins its version.
- The 1,250-line source budget (`pyrite-dev check-lines`) remains a separate
  gate. The report also counts modules at 1,100 lines or more.
