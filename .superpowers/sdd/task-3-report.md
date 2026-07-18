# Task 3 Report: `scan.py --max-minutes`, exit 75, `paused` progress state

## Status: DONE

Commit: `14512b2` — "feat(scan): --max-minutes budget with EX_TEMPFAIL and paused progress state"
(branch `worktree-remote-chunked-jobs`, on top of Task 1 `371655d` and Task 2 `bf4cd3f`).

## What was implemented

`src/cxr_mc/scan.py`:
- `import time` added.
- `_build_parser`: new `--max-minutes FLOAT` (default `None`) — "soft wall-clock budget; exit 75 if work remains (chained remote slices)".
- `run(args)`: computes **one** deadline (`time.monotonic() + max_minutes*60`) once, before the material loop — so `--all` spans the whole invocation rather than resetting the budget per material. Each iteration passes `max_seconds = max(0.0, deadline - time.monotonic())` (clamped to `>= 0`) to `_run_material`; `None` when `--max-minutes` wasn't given. Tracks `incomplete` across materials and raises `SystemExit(75)` at the end if any material didn't finish. Hard failures (exceptions from `run_sweep`) are untouched — they still propagate through `_run_material`'s existing `except BaseException` re-raise, before `run()` ever sees a return value.
- `_run_material(args, material, max_seconds=None)`: forwards `max_seconds=max_seconds` into `run_sweep(...)`, captures the boolean result, writes the final progress record as `state="done" if complete else "paused"`, prints `"paused (budget) -> ..."` instead of `"done -> ..."` on the incomplete path, and returns `complete` to `run()`.
- The `complete` value is computed as `run_sweep(...) is not False` rather than bare truthiness — see "Design decision" below.
- The `except BaseException` → `state="failed"` path is byte-for-byte unchanged.

`src/cxr_mc/remote.py` (~line 1186, `_parse_progress_records`):
- State whitelist widened from `{"running", "done", "failed"}` to `{"running", "done", "failed", "paused"}`.

## Design decision: `is not False` guard (deviation from the brief's literal sketch)

The brief's Step 3 sketch has `_run_material` treat `run_sweep`'s return via plain truthiness. Implementing it that way broke 5 pre-existing tests in `tests/test_sweep.py` (a file I was explicitly told never to touch): `test_scan_checkpoints_under_registry_name`, `test_run_material_applies_penetration_watchdog`, `test_scan_progress_record_tracks_running_and_done`, `test_scan_forwards_n_families_and_beam_uvw_overrides`, `test_scan_all_runs_every_material_in_toml_manifest`. Those tests' `fake_run_sweep` stubs predate Task 2/3 and don't return anything (implicit `None`), so `if not _run_material(...)` read them as incomplete and raised `SystemExit(75)` where none was expected, and the "done" progress-state assertion failed.

I verified this wasn't caused by my changes by stashing them and confirming `tests/test_sweep.py` passes clean against HEAD (78 passed). Real `run_sweep` (Task 2, already committed) always returns an actual `bool`, so `complete = run_sweep(...) is not False` is behaviorally identical to `bool(complete)` for every real caller, while treating legacy test doubles that return `None` as "complete" rather than "paused". This is documented inline in `scan.py`. No test_sweep.py edits were made or needed.

## TDD evidence

**RED** — `uv run python scripts/dev.py test tests/test_scan_budget.py tests/test_remote.py -k "budget or paused"` (before implementation):
```
python3 -m pytest: error: unrecognized arguments: --max-minutes 5.0
...
FAILED tests/test_scan_budget.py::test_scan_budget_incomplete_exits_75
FAILED tests/test_scan_budget.py::test_scan_budget_complete_exits_normally
FAILED tests/test_scan_budget.py::test_scan_budget_writes_paused_progress_state
FAILED tests/test_scan_budget.py::test_scan_no_max_minutes_passes_none_max_seconds
FAILED tests/test_remote.py::test_parse_progress_records_accepts_paused_state - KeyError: 'hopg'
5 failed, 112 deselected in 1.33s
```
Failures matched expectation: argparse rejects the unrecognized `--max-minutes` flag (4 scan-budget tests), and the whitelist test fails because `_parse_progress_records` silently drops `state: "paused"` records (`KeyError: 'hopg'` — nothing was recorded).

**GREEN** — `uv run python scripts/dev.py test tests/test_scan_budget.py tests/test_remote.py tests/test_run.py`:
```
........................................................................ [ 51%]
...................................................................      [100%]
139 passed in 1.68s
```
Full suite: `uv run python scripts/dev.py test` → `839 passed, 1 warning` (the warning is pre-existing matplotlib figure-count noise from an unrelated test, not introduced by this change). `lint` → "All checks passed!". `typecheck` → "0 errors, 0 warnings, 0 informations".

## Files changed
- `src/cxr_mc/scan.py` — flag, deadline/budget wiring, `paused` state.
- `src/cxr_mc/remote.py` — whitelist widened (single-line change; see note on unrelated reformatting below).
- `tests/test_scan_budget.py` (new) — 4 tests: budget-incomplete → exit 75, budget-complete → normal exit, incomplete → `paused` progress record written, and (added beyond the brief) no `--max-minutes` → `max_seconds=None` passed through to `run_sweep` untouched.
- `tests/test_remote.py` — 1 new test, `test_parse_progress_records_accepts_paused_state`, placed immediately after the existing `test_parse_progress_records_ignores_malformed_snapshots`.

## Self-review

- **Completeness**: flag name (`--max-minutes`), exit codes (0 complete / 75 incomplete / hard failures still raise), one-deadline-spans-whole-invocation semantics (including `--all`), clamp-to-`>=0`, `paused` progress state, and the `remote.py` whitelist all match the brief's binding contract exactly — these are the pieces Task 4's generated SLURM script depends on.
- **No overbuilding**: didn't add retry logic, didn't touch `run_cases`/`run_sweep` (Task 1/2's surface), didn't add CLI docs beyond the one help string, didn't add a `time_fn` seam to `scan.py` (not required by the brief; `run.py`'s `run_sweep` already owns that testing seam and `scan.py`'s tests monkeypatch `run_sweep` wholesale instead — consistent with how the file already tests `_run_material`).
- **Test quality**: RED confirmed with real failures before writing any implementation code; tests stub `scan.run_sweep` and `scan.build_cases` (and, since I found the brief's fixture triggers the real penetration watchdog against a case dict missing `composition`, also `scan.gate_cases_by_penetration` as a passthrough) so no real Monte Carlo runs. The added 4th test (`max_seconds=None` propagation when `--max-minutes` is omitted) covers the "budget disabled" path the brief's 3 tests don't exercise.
- **Pristine output**: full suite, lint, and typecheck all clean; no xfail/skip markers added.
- **Housekeeping caught mid-task**: running `uv run python scripts/dev.py format` reformats 12 files unrelated to this task (pre-existing repo-wide formatting drift in this worktree, including `src/cxr_mc/sweep.py` — a file I was told never to touch). I reverted every file outside my task's scope (`git checkout --`) both times this happened and confirmed `scan.py`/`remote.py` are idempotent under `format` (a second `format` run produces zero diff on them). Only `src/cxr_mc/scan.py`, `src/cxr_mc/remote.py`, `tests/test_remote.py`, and the new `tests/test_scan_budget.py` are in the commit. Note the harness's own PostToolUse auto-format hook did reformat quote-style/line-wrapping on the rest of `remote.py` and `test_remote.py` beyond my touched lines (visible in `git diff` as many `'...'` → `"..."` changes) — this was not something I initiated and is behaviorally inert (verified via `lint`/full-suite pass), but it is a larger diff footprint on those two files than the brief's "single-line"/"append one test" framing implied.

## Concerns

- The auto-format-triggered quote-style churn across most of `remote.py` and `tests/test_remote.py` (see above) is larger than strictly necessary for this task's stated scope, though it's mechanical, semantically inert, and unavoidable given the Edit-tool's PostToolUse formatting hook. Flagging for visibility in case a reviewer wants a narrower diff.
- This worktree's uncommitted-elsewhere formatting drift (12 files off-canonical, one of which is the forbidden `sweep.py`) is pre-existing and outside this task's scope — noted here only because it surfaced while sanity-checking `format`, not because I touched it.
- Note: this report file previously held an unrelated stale "Task 3: Germanium-family crystals" write-up from a different plan run/numbering; it has been overwritten with this task's content.

## Review fixes (follow-up commit)

Reviewer verdict "Needs fixes" — one Important, one Minor. Both taken.

### Important: multi-material deadline-threading test added

New `test_scan_all_threads_one_deadline_across_materials` in `tests/test_scan_budget.py`.
Setup: `scan.time.monotonic` monkeypatched to a controllable fake clock (dict counter);
`scan.load_all_materials` stubbed to `["hopg", "hbn", "mose2"]`; `run_sweep` stub records
`kwargs["max_seconds"]` per call and advances the fake clock by 200 s; args parsed from
`["--all", "--max-minutes", "5"]` (300 s budget). Asserts `seen == [300.0, 100.0, 0.0]`, i.e.:
(a) the second material's `max_seconds` (100.0) <= the first's (300.0) — one deadline is
shared, not reset per material; and (b) after the deadline fully elapses (clock at 400 s > 300 s
deadline), the third material is still CALLED with `max_seconds=0.0` rather than skipped, so
fully-cached materials can resolve to complete. The test discriminates real bugs: a fresh
per-material deadline would yield `[300.0, 300.0, 300.0]`; skipping on elapsed budget would
drop the third recorded call.

### Minor: guard narrowed

`src/cxr_mc/scan.py` `_run_material`: `complete = run_sweep(...) is not False` replaced with

```python
result = run_sweep(...)
complete = True if result is None else bool(result)
```

Only a bare `None` (the five legacy `tests/test_sweep.py` stubs that predate the budget
feature) is read as complete; any other unexpected falsy return (0, "", etc.) now fails loud
as incomplete instead of silently reading as complete. `tests/test_sweep.py` untouched.

### Re-run evidence

`uv run python scripts/dev.py test tests/test_scan_budget.py tests/test_remote.py tests/test_sweep.py`
-> `196 passed in 23.99s`. `lint` -> "All checks passed!". `typecheck` -> "0 errors, 0 warnings,
0 informations".

Follow-up commit: see `git log` — "test(scan): cover --all deadline threading; narrow run_sweep completeness guard".
