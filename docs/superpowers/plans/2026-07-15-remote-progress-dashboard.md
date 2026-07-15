# Remote Progress Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make newly submitted concurrent remote scans persist independent per-material case progress and make `cxr remote attach` render that progress locally.

**Architecture:** `run_sweep` reports an initial snapshot and each completed new case through a callback. `scan.py` owns atomic JSON record persistence and terminal state transitions. `remote.py` wires one record path into each queued scan, parses records defensively, and uses local `tqdm` bars while polling SLURM; legacy jobs without the dashboard metadata marker retain log-following attach behavior.

**Tech Stack:** Python 3.11+, argparse, JSON, pathlib/os atomic replacement, tqdm, pytest, generated Bash/SLURM scripts.

## Global Constraints

- Only jobs submitted after this change use progress records.
- `cxr remote logs` remains the raw diagnostic stream.
- Malformed or temporarily unreadable records are ignored for that poll.
- Ctrl-C disconnects only the local viewer.
- New queue scans disable terminal `tqdm` output in the shared job log.

---

### Task 1: Sweep Progress Events and Atomic Scan Records

**Files:**
- Modify: `src/cxr_mc/run.py`
- Modify: `src/cxr_mc/scan.py`
- Test: `tests/test_run.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: the existing `run_cases(..., callback=...)` per-case callback.
- Produces: `run_sweep(..., on_progress: Callable[[int, int, int], None] | None)` where arguments are completed new cases, total cases, and cached cases; scan progress JSON fields `material`, `total_cases`, `cached_cases`, `completed_new_cases`, and `state`.

- [x] **Step 1: Write failing orchestration and persistence tests**

Add tests that require an initial `(0, total, cached)` event, one event after every newly completed case, atomic replacement without leftover temporary files, and `running`/`done`/`failed` scan records.

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run python scripts/dev.py test tests/test_run.py tests/test_sweep.py -k progress`

Expected: failures because `run_sweep` has no `on_progress` callback and `scan.py` has no progress writer or internal CLI flags.

- [x] **Step 3: Implement the minimal callback and writer**

Call `on_progress(0, len(cases), cached_cases)` after resume filtering and call it with the incremented completed count from the existing case callback. Add an atomic one-line JSON writer in `scan.py`, hidden `--progress-file` and `--no-progress` arguments, and terminal writes around `run_sweep`.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run python scripts/dev.py test tests/test_run.py tests/test_sweep.py -k progress`

Expected: all selected tests pass.

### Task 2: Queue Wiring and Local Dashboard

**Files:**
- Modify: `src/cxr_mc/remote.py`
- Test: `tests/test_remote.py`

**Interfaces:**
- Consumes: progress JSON records produced by `scan.py` and existing job metadata/SLURM helpers.
- Produces: `_parse_progress_records(text)`, `_read_progress_records(jobid)`, and dashboard attach behavior; legacy attach remains available for metadata without the new marker.

- [x] **Step 1: Write failing queue, parser, renderer, and lifecycle tests**

Require queue scripts to create `JOBDIR/progress`, pass a material-specific `--progress-file`, add `--no-progress`, and persist a dashboard metadata marker. Require malformed record lines to be skipped, bars to show cached plus newly completed cases independently, terminal SLURM state to close bars and print persisted state, and Ctrl-C to leave the job running.

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run python scripts/dev.py test tests/test_remote.py -k 'progress or attach or queue_script'`

Expected: failures because the queue has no progress arguments and attach still streams the shared log.

- [x] **Step 3: Implement queue wiring, defensive parsing, and local bars**

Add the metadata marker and per-material record paths to generated Bash. Parse one JSON object per line with strict field validation. For new jobs create one local `tqdm` per metadata material, update each bar from its latest record, poll SLURM until it leaves the queue, perform one final record read, close bars, and print the persisted final state. Route old jobs through the existing log-following viewer.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run python scripts/dev.py test tests/test_remote.py -k 'progress or attach or queue_script'`

Expected: all selected tests pass.

### Task 3: Focused and Repository Verification

**Files:**
- Modify only files needed to resolve failures introduced by Tasks 1 and 2.

**Interfaces:**
- Consumes: completed dashboard implementation.
- Produces: verified CLI help, focused tests, lint, and type-check evidence.

- [x] **Step 1: Run the full focused regression set**

Run: `uv run python scripts/dev.py test tests/test_run.py tests/test_sweep.py tests/test_remote.py`

Expected: all tests pass.

- [x] **Step 2: Check the CLI surfaces**

Run: `uv run cxr scan --help`

Run: `uv run cxr remote attach --help`

Expected: both commands exit successfully; internal scan flags remain hidden.

- [x] **Step 3: Run static checks**

Run: `uv run python scripts/dev.py lint`

Run: `uv run python scripts/dev.py typecheck`

Expected: both checks pass, or any unrelated baseline issue is identified with exact output.

- [x] **Step 4: Review the final patch**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors and only the planned files are modified.
