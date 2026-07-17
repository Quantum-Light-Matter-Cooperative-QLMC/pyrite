# Chunked Remote GPU Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break remote SLURM sweeps into ~10-minute self-resubmitting slices so the single-GPU lab box yields to other users at every slice boundary, per `docs/superpowers/specs/2026-07-17-remote-chunked-jobs-design.md` (the spec; read it for rationale — this plan is the how).

**Architecture:** A soft wall-clock budget threads from `run_cases` (`should_stop` hook) through `run_sweep` (`max_seconds`, `complete` flag) to `scan.py` (exit 75 = budget hit). `remote.py` generates a chained batch script whose slices resume from checkpoint, run `scan.py --max-minutes`, and `sbatch` the next slice with `--nice`; reservations release only at terminal states; attach follows the whole chain.

**Tech Stack:** Python 3 (stdlib argparse/subprocess), bash payloads generated as f-strings, pytest with monkeypatch. No new dependencies.

## Global Constraints

- Canonical commands only: test with `uv run python scripts/dev.py test <path> -k <name>`, lint with `uv run python scripts/dev.py lint`, typecheck with `uv run python scripts/dev.py typecheck`.
- Work in this worktree; never touch `mats_to_sim.toml`, `src/cxr_mc/sweep.py`, `tests/test_sweep.py` (spec Scope).
- Exit code semantics (spec Component 1): `0` = complete, `75` = budget hit / work remains, any other nonzero = hard failure. Exactly these three classes, everywhere.
- Chunked defaults (spec Component 2): `--chunk-minutes` default `10`; `0` = monolithic escape hatch; chunked slices get `#SBATCH --time` = ceil(3 × chunk-minutes) minutes; monolithic keeps `UNLIMITED`; every chunked sbatch (slice 0 included) carries `--nice=10000`.
- Chain states are prefix-matched strings in `$JOBDIR/state`: handoff = `queued slice ...`; terminal = `done*` / `FAILED*` / `cancelled*`. Reservations release ONLY at terminal (spec 3a).
- Handoff ordering is fixed (spec Component 2 step 4): write `queued slice k+1` state → `sbatch` (fail-closed, numeric-SID validated) → append `slurm_job_id:` to meta → exit 0.
- Marker lines in `$JOBDIR/log` are exact: `completed: <m>` and `failed: <m>`.
- New progress-file state `paused` (budget exit); `_parse_progress_records` whitelist becomes `{"running", "done", "failed", "paused"}`.
- This is not physics code: no derivation docstrings or ledger rows are needed. Follow existing docstring style in each file.
- Commit after every task; commit messages follow the repo's `feat:`/`test:`/`docs:` convention and end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `run_cases` gains a `should_stop` hook

**Files:**
- Modify: `src/cxr_mc/montecarlo/runner.py:495` (`run_cases`)
- Test: `tests/test_montecarlo.py` (append)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `run_cases(cases, max_workers=None, progress=True, callback=None, should_stop=None)` where `should_stop: Callable[[], bool] | None`. Checked **before each new case starts**; once it returns True, no new case begins, already-started work drains normally (callbacks still fire for drained cases), and `run_cases` returns the results list with `None` for never-started cases. Task 2 relies on exactly this.

**Semantics per dispatch path** (all three live inside `run_cases`):

1. `_serial` (lines ~547-557): check at the top of the loop; `break` when True.
2. GPU pipeline (lines ~578-599): the prefetch window is the in-flight set. When `should_stop()` fires, stop submitting new transport futures (`inflight[j] = ...`); keep consuming futures already in `inflight` (their GPU phase + callback still run); exit the loop when `inflight` is empty. Concretely: guard the `if j < n:` submission with `and not stopped`, where `stopped` is latched by checking `should_stop` once per iteration before the submission; after latching, `break` once `inflight` is empty (`if stopped and not inflight: break`). Iterate over `sorted(inflight)` remainder rather than `range(n)` after the latch — or simplest correct form: keep the `range(n)` loop but `if i not in inflight: break` after the latch.
3. CPU pool (lines ~610-621): all futures are pre-submitted, so "stop launching" means cancelling unstarted futures. In the `as_completed` loop, `continue` on `fut.cancelled()`; after each completed case, if `should_stop()` fires (latch once), call `f.cancel()` on every future — cancelled ones are skipped as they surface.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_montecarlo.py`; monkeypatch `run_case` — only the serial path is unit-testable because pool workers re-import the real module; the pool paths are covered by review):

```python
def test_run_cases_should_stop_halts_new_dispatch(monkeypatch):
    from cxr_mc.montecarlo import runner

    ran = []
    monkeypatch.setattr(runner, "run_case", lambda case: {"name": case["name"]})
    calls = {"n": 0}

    def stop_after_two():
        return calls["n"] >= 2

    seen = []

    def cb(i, case, out):
        calls["n"] += 1
        seen.append(case["name"])

    cases = [{"name": f"c{i}"} for i in range(5)]
    results = runner.run_cases(
        cases, max_workers=0, progress=False, callback=cb, should_stop=stop_after_two
    )
    assert seen == ["c0", "c1"]
    assert results[0] is not None and results[1] is not None
    assert results[2] is None and results[3] is None and results[4] is None


def test_run_cases_should_stop_none_runs_everything(monkeypatch):
    from cxr_mc.montecarlo import runner

    monkeypatch.setattr(runner, "run_case", lambda case: {"name": case["name"]})
    cases = [{"name": f"c{i}"} for i in range(3)]
    results = runner.run_cases(cases, max_workers=0, progress=False, should_stop=None)
    assert all(r is not None for r in results)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run python scripts/dev.py test tests/test_montecarlo.py -k should_stop`
Expected: FAIL — `TypeError: run_cases() got an unexpected keyword argument 'should_stop'`

- [ ] **Step 3: Implement** — add the parameter, document it in the docstring (one short paragraph in the existing style: checked before each new case; in-flight work drains; unstarted slots stay `None`), and wire all three paths per the semantics above. The serial path:

```python
    def _serial():
        for i in _maybe_bar(range(n)):
            if should_stop is not None and should_stop():
                break
            out = run_case(cases[i])
            ...
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run python scripts/dev.py test tests/test_montecarlo.py -k should_stop`
Expected: PASS. Then run the whole file: `uv run python scripts/dev.py test tests/test_montecarlo.py` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/montecarlo/runner.py tests/test_montecarlo.py
git commit -m "feat(montecarlo): add should_stop hook to run_cases"
```

---

### Task 2: `run_sweep` soft budget (`max_seconds`, `complete` flag, final save)

**Files:**
- Modify: `src/cxr_mc/run.py` (`run_sweep`, lines ~108-237)
- Test: `tests/test_run.py` (append; reuse its existing fixtures/stubs for cases and results — read the file first and follow its patterns)

**Interfaces:**
- Consumes: Task 1's `run_cases(..., should_stop=...)`.
- Produces: `run_sweep(cases, results, *, ..., max_seconds=None, time_fn=time.monotonic) -> bool`. Returns `complete` — True iff every requested `(name, E0_keV)` is now in `results`. When the deadline passes, no new cases start, in-flight drain, then one **final `_save()`** runs so partially-finished configs persist at case granularity (resume filtering is per `(name, E0)` — see spec Key enabling fact). Task 3 relies on the bool.

- [ ] **Step 1: Write the failing tests.** Drive a fake clock; stub `run.run_cases` with a fake that honors `should_stop` and calls the callback like the real one (this task tests `run_sweep`'s budget plumbing, not Task 1 again):

```python
def test_run_sweep_budget_stops_early_and_resumes(tmp_path, monkeypatch):
    from cxr_mc import run as run_mod

    clock = {"t": 0.0}

    def fake_run_cases(todo, max_workers=None, progress=True, callback=None, should_stop=None):
        for i, c in enumerate(todo):
            if should_stop is not None and should_stop():
                break
            clock["t"] += 10.0  # each case takes 10 "seconds"
            callback(i, c, {"out": c["name"]})
        return []

    monkeypatch.setattr(run_mod, "run_cases", fake_run_cases)
    monkeypatch.setattr(
        run_mod, "store_result",
        lambda results, case, out: results.setdefault(case["name"], {}).__setitem__(
            case["E0_keV"], {"case": case}
        ),
    )
    cases = [
        {"name": f"cfg{i}", "E0_keV": 30, "crystal": "hopg", "thickness_ang": 1e4,
         "tilt_deg": 0.0}
        for i in range(4)
    ]
    ckpt = tmp_path / "hopg.pkl"

    results = {}
    complete = run_mod.run_sweep(
        cases, results, checkpoint_path=str(ckpt),
        max_seconds=25.0, time_fn=lambda: clock["t"],
    )
    assert complete is False
    assert len(results) == 3  # budget passed after case 3 (t=30 > 25)
    # the final save persisted the partial state: a fresh resume skips them
    results2 = {}
    complete2 = run_mod.run_sweep(
        cases, results2, checkpoint_path=str(ckpt),
        max_seconds=1000.0, time_fn=lambda: clock["t"],
    )
    assert complete2 is True
    assert len(results2) == 4


def test_run_sweep_no_budget_returns_complete(tmp_path, monkeypatch):
    # same stubs; max_seconds=None must behave exactly as today and return True
    ...  # same fake_run_cases/store_result setup, all 4 cases, assert returns True
```

(Write the second test out in full with the same stubs — no literal `...` in the committed file.)

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run python scripts/dev.py test tests/test_run.py -k budget`
Expected: FAIL — `run_sweep() got an unexpected keyword argument 'max_seconds'`

- [ ] **Step 3: Implement** in `run_sweep`:

```python
def run_sweep(cases, results, *, ..., on_progress=None, max_seconds=None, time_fn=None):
    ...
    if time_fn is None:
        time_fn = time.monotonic
    deadline = None if max_seconds is None else time_fn() + max_seconds
    should_stop = None if deadline is None else (lambda: time_fn() >= deadline)
    ...
    run_cases(todo, max_workers=max_workers, progress=progress, callback=_cb,
              should_stop=should_stop)
    print(f"{len(todo)} cases in {time.perf_counter() - t0:.0f} s")
    complete = all(
        c["name"] in results and c["E0_keV"] in results[c["name"]] for c in cases
    )
    if not complete:
        _save()  # persist partially-finished configs: resume is per (name, E0)
    return complete
```

Extend the docstring's parameter list (`max_seconds`, `time_fn`, return value) in the existing style. Note `_save()` on the incomplete path also runs when `max_seconds is None` and something failed to store — that is fine and harmless (atomic, idempotent).

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run python scripts/dev.py test tests/test_run.py`
Expected: PASS, no regressions in the file.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/run.py tests/test_run.py
git commit -m "feat(run): soft wall-clock budget and complete flag for run_sweep"
```

---

### Task 3: `scan.py --max-minutes`, exit 75, `paused` progress state

**Files:**
- Modify: `src/cxr_mc/scan.py` (`_build_parser`, `run`, `_run_material`)
- Modify: `src/cxr_mc/remote.py:1186` (`_parse_progress_records` state whitelist)
- Test: create `tests/test_scan_budget.py`; append one whitelist test near the existing `_parse_progress_records` tests in `tests/test_remote.py`

**Interfaces:**
- Consumes: Task 2's `run_sweep(..., max_seconds=...) -> bool`.
- Produces: CLI flag `--max-minutes FLOAT` (default None). Process exit `0` when every selected material is complete, `SystemExit(75)` when any is budget-incomplete; hard failures keep raising as today. The budget spans the whole invocation (`--all` included): one deadline computed at `run()` start; each material gets the remaining time (clamped to ≥ 0 — `max_seconds=0` still resolves fully-cached materials to complete). Progress record state `paused` written when a material is incomplete. Task 4's generated script relies on the flag name and the exit codes.

- [ ] **Step 1: Write the failing tests** in `tests/test_scan_budget.py` (monkeypatch `scan.run_sweep` — never run real MC):

```python
import pytest

from cxr_mc import scan


def _args(material, max_minutes, progress_file=None):
    ap = scan._build_parser(__import__("argparse").ArgumentParser())
    argv = [material]
    if max_minutes is not None:
        argv += ["--max-minutes", str(max_minutes)]
    if progress_file is not None:
        argv += ["--progress-file", str(progress_file)]
    return ap.parse_args(argv)


def test_scan_budget_incomplete_exits_75(monkeypatch, tmp_path):
    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: False)
    monkeypatch.setattr(scan, "build_cases", lambda *a, **kw: [
        {"name": "cfg0", "E0_keV": 30, "crystal": "hopg", "thickness_ang": 1e4,
         "tilt_deg": 0.0, "beam_uvw": (0, 0, 1), "hkl_list": [(0, 0, 2)]}
    ])
    args = _args("hopg", 5.0)
    args.checkpoint_dir = str(tmp_path)
    with pytest.raises(SystemExit) as exc:
        scan.run(args)
    assert exc.value.code == 75


def test_scan_budget_complete_exits_normally(monkeypatch, tmp_path):
    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: True)
    monkeypatch.setattr(scan, "build_cases", lambda *a, **kw: [
        {"name": "cfg0", "E0_keV": 30, "crystal": "hopg", "thickness_ang": 1e4,
         "tilt_deg": 0.0, "beam_uvw": (0, 0, 1), "hkl_list": [(0, 0, 2)]}
    ])
    args = _args("hopg", 5.0)
    args.checkpoint_dir = str(tmp_path)
    scan.run(args)  # no SystemExit


def test_scan_budget_writes_paused_progress_state(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: False)
    monkeypatch.setattr(scan, "build_cases", lambda *a, **kw: [
        {"name": "cfg0", "E0_keV": 30, "crystal": "hopg", "thickness_ang": 1e4,
         "tilt_deg": 0.0, "beam_uvw": (0, 0, 1), "hkl_list": [(0, 0, 2)]}
    ])
    progress = tmp_path / "hopg.json"
    args = _args("hopg", 5.0, progress_file=progress)
    args.checkpoint_dir = str(tmp_path)
    with pytest.raises(SystemExit):
        scan.run(args)
    record = json.loads(progress.read_text())
    assert record["state"] == "paused"
```

And in `tests/test_remote.py`, next to the existing `_parse_progress_records` tests:

```python
def test_parse_progress_records_accepts_paused_state():
    payload = '{"material":"hopg","total_cases":4,"cached_cases":1,"completed_new_cases":1,"state":"paused"}'
    records = remote._parse_progress_records(payload)
    assert records["hopg"]["state"] == "paused"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run python scripts/dev.py test tests/test_scan_budget.py tests/test_remote.py -k "budget or paused"`
Expected: FAIL — unrecognized argument `--max-minutes`; the whitelist test fails because `paused` is dropped.

- [ ] **Step 3: Implement.**

`_build_parser`: add

```python
    ap.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="soft wall-clock budget; exit 75 if work remains (chained remote slices)",
    )
```

`run()`: compute one deadline, thread remaining time, exit 75 on any incomplete:

```python
def run(args):
    ...
    validate_materials(materials)
    deadline = None
    if getattr(args, "max_minutes", None) is not None:
        deadline = time.monotonic() + args.max_minutes * 60.0
    incomplete = False
    for material in materials:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not _run_material(args, material, max_seconds=remaining):
            incomplete = True
    if incomplete:
        raise SystemExit(75)  # EX_TEMPFAIL: budget hit, work remains
```

(`import time` at the top of `scan.py`.)

`_run_material(args, material, max_seconds=None)`: pass `max_seconds=max_seconds` into `run_sweep`, capture `complete = run_sweep(...)`, return it, and write the final progress record with `state="done" if complete else "paused"`. The `except BaseException` failed-state write is unchanged.

`remote.py` `_parse_progress_records`: `if state not in {"running", "done", "failed", "paused"}: continue`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run python scripts/dev.py test tests/test_scan_budget.py tests/test_remote.py tests/test_run.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/scan.py src/cxr_mc/remote.py tests/test_scan_budget.py tests/test_remote.py
git commit -m "feat(scan): --max-minutes budget with EX_TEMPFAIL and paused progress state"
```

---

### Task 4: chained submission in `remote.py`

**Files:**
- Modify: `src/cxr_mc/remote.py` — `_slurm_batch_script`, `_submit_slurm_command`, `start_queue`, `_queue_metadata`, `_cli_scan`/`_cli_start` + both argparse parsers; add `_chunked_queue_script`.
- Test: `tests/test_remote.py` (extend the dry-run tests around lines 252 and 1179-1222)

**Interfaces:**
- Consumes: Task 3's `scan.py <m> --max-minutes <float>` contract and exit codes 0/75/other.
- Produces: `start_queue(..., parallel_materials=None, chunk_minutes=10.0)`; `_slurm_batch_script(..., time_limit=SLURM_TIME)`; `_submit_slurm_command(..., nice=False)`. Task 5 relies on the state strings (`queued slice ...`, `FAILED (slice resubmission) ...`) and the `STOP` sentinel path `$JOBDIR/STOP`.

- [ ] **Step 1: Write the failing dry-run tests** (model on `test_remote_dry_run_describes_sbatch_submission` at `tests/test_remote.py:252` — same monkeypatching that forbids ssh/sync):

```python
def test_chunked_dry_run_emits_chain_script(monkeypatch, capsys):
    # same no-ssh/no-sync monkeypatching as test_remote_dry_run_describes_sbatch_submission
    ...
    remote.start_queue(["hopg"], dry_run=True)  # chunked is the default
    out = capsys.readouterr().out
    assert "--max-minutes" in out
    assert "sbatch --parsable --nice=10000" in out          # resubmit + slice 0
    assert 'queued slice' in out                            # handoff state written BEFORE sbatch
    assert 'FAILED (slice resubmission)' in out             # fail-closed resubmit
    assert '[ -f "$JOBDIR/STOP" ]' in out or '"$JOBDIR/STOP"' in out
    assert "failed: $m" in out                              # hard-failure marker, never retried
    assert "#SBATCH --time=30" in out                       # 3 x 10 min backstop
    assert '"queued slice"*' in out                         # trap handoff case
    # handoff must not release reservations: the trap's release happens only in
    # terminal branches -- assert the handoff case body is empty (';;' right after)


def test_chunk_minutes_zero_emits_monolithic_script(monkeypatch, capsys):
    ...
    remote.start_queue(["hopg"], dry_run=True, chunk_minutes=0)
    out = capsys.readouterr().out
    assert "#SBATCH --time=UNLIMITED" in out
    assert "parallel_materials=2" in out
    assert "--max-minutes" not in out


def test_parallel_materials_rejected_in_chunked_mode(monkeypatch):
    with pytest.raises(SystemExit, match="chunk-minutes 0"):
        remote.start_queue(["hopg"], dry_run=True, parallel_materials=2)


def test_cli_start_chunk_flags(monkeypatch, capsys):
    ...
    remote.main(["start", "hopg", "--dry-run", "--chunk-minutes", "0",
                 "--parallel-materials", "3"])   # legal: monolithic
    with pytest.raises(SystemExit):
        remote.main(["start", "hopg", "--dry-run", "--parallel-materials", "3"])  # illegal: chunked default
```

(Fill each `...` with the standard dry-run monkeypatch block copied from the existing test; no literal `...` committed.) Also update any existing dry-run tests that assert monolithic content on default flags: they must now pass `--chunk-minutes 0` / `chunk_minutes=0` **only if** they assert monolithic-only content; tests asserting mode-independent content (job name, partition, reservations) stay as they are.

- [ ] **Step 2: Run tests, verify the new ones fail**

Run: `uv run python scripts/dev.py test tests/test_remote.py -k "chunk or parallel_materials"`
Expected: new tests FAIL (no `chunk_minutes` parameter yet).

- [ ] **Step 3: Implement.**

Add `import math` to the imports. New payload generator, next to `_queue_script`:

```python
def _chunked_queue_script(jobid, materials, quick, workers, chunk_minutes):
    """One SLURM slice of a self-resubmitting chain (spec: chunked remote jobs).

    Reused verbatim by every slice: it resumes from checkpoint, does about
    chunk_minutes of work via scan.py --max-minutes, and either terminates the
    chain (all materials completed:/failed:) or hands off: write the
    'queued slice' state FIRST, then sbatch fail-closed, then append the SID.
    State-first ordering keeps the EXIT trap from releasing reservations while
    the next slice is already pending (spec Component 2 step 4).
    """
    flags = ""
    if quick:
        flags += " --quick"
    if workers is not None:
        flags += f" --workers {workers}"
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    chunk_seconds = int(round(chunk_minutes * 60))
    return f"""JOBDIR="{jobdir}"
cd "{REMOTE_DIR}" || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{REMOTE_UV} sync >> "$JOBDIR/log" 2>&1 || {{ echo "FAILED (uv sync) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
mats=({mats})
total=${{#mats[@]}}
chunk_seconds={chunk_seconds}
slice_start=$(date +%s)
n=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && continue
  now=$(date +%s)
  remaining=$((slice_start + chunk_seconds - now))
  [ "$remaining" -gt 0 ] || break
  remaining_min=$(awk "BEGIN {{ printf \\"%.2f\\", $remaining / 60 }}")
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  rc=0
  {REMOTE_UV} run --no-sync python scan.py "$m"{flags} --max-minutes "$remaining_min" \
    --progress-file "$JOBDIR/progress/$m.json" --no-progress >> "$JOBDIR/log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "completed: $m" >> "$JOBDIR/log"
  elif [ "$rc" -ne 75 ]; then
    echo "WARNING: scan failed for $m (exit $rc); will not retry" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    echo "failed: $m" >> "$JOBDIR/log"
  fi
done
unresolved=0
failures=0
for m in "${{mats[@]}}"; do
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && {{ failures=$((failures + 1)); continue; }}
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  unresolved=1
done
if [ "$unresolved" -eq 0 ]; then
  if [ "$failures" -gt 0 ]; then
    echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
  else
    echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
  fi
  exit 0
fi
[ -f "$JOBDIR/STOP" ] && exit 0
k=$(grep -c "^slurm_job_id: " "$JOBDIR/meta" 2>/dev/null)
echo "queued slice $((k + 1)) $(date -Is)" > "$JOBDIR/state"
SID=$(sbatch --parsable --nice=10000 "$JOBDIR/run.sh") || {{ echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
SID=${{SID%%;*}}
case "$SID" in ''|*[!0-9]*) echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1 ;; esac
printf 'slurm_job_id: %s\\n' "$SID" >> "$JOBDIR/meta"
"""
```

`_slurm_batch_script` gains `time_limit=SLURM_TIME` (keyword-only) — replace the hardcoded `#SBATCH --time={SLURM_TIME}` with `{time_limit}` — and the `finish` trap becomes terminal-only release with a handoff case (spec 3a; note the existing trap releases on *every* exit — that is the bug being fixed):

```bash
finish() {{
  status=$?
  current=$(cat "$JOBDIR/state" 2>/dev/null || true)
  case "$current" in
    "queued slice"*) ;;
    done*|FAILED*|cancelled*|cancelling*) release_reservations ;;
    *) echo "FAILED (exit $status) $(date -Is)" > "$JOBDIR/state"; release_reservations ;;
  esac
}}
```

`_submit_slurm_command(jobid, reservation_stems=None, nice=False)`: when `nice`, emit `sbatch --parsable --nice=10000` (slice 0 carries the courtesy too).

`start_queue`:

```python
def start_queue(materials, quick=False, workers=None, no_sync=False, dry_run=False,
                parallel_materials=None, chunk_minutes=10.0):
    _check_materials(materials)
    chunked = chunk_minutes > 0
    if chunked and parallel_materials is not None:
        raise SystemExit(
            "--parallel-materials only applies to a monolithic allocation; "
            "pass --chunk-minutes 0 to use it"
        )
    if not chunked:
        parallel_materials = _validate_parallel_materials(
            DEFAULT_PARALLEL_MATERIALS if parallel_materials is None else parallel_materials
        )
    ...
    if chunked:
        payload = _chunked_queue_script(jobid, materials, quick, workers, chunk_minutes)
        time_limit = str(max(1, math.ceil(chunk_minutes * 3)))  # minutes: hard backstop
    else:
        payload = _queue_script(jobid, materials, quick, workers, parallel_materials)
        time_limit = SLURM_TIME
    script = _slurm_batch_script(jobid, payload, job_name=f"cxr-{jobid}",
                                 reservation_stems=stems, time_limit=time_limit)
    ...
    submit = _submit_slurm_command(jobid, stems, nice=chunked)
```

`_queue_metadata` gains a `chunk_minutes: <value>` line (after `parallel_materials`; for monolithic record `0`). Both `start` and `scan` parsers gain `--chunk-minutes` (`type=float, default=10.0`, help mentions `0` = whole-box monolithic run) and change `--parallel-materials` to `default=None` (keep `choices`); `_cli_start`/`_cli_scan` thread both through. `remote_scan()` and `start_zhai_queue` are untouched — Zhai stays monolithic (`time_limit` default, `nice` default False); the new trap changes its script text, which is fine (its states are already terminal-only).

- [ ] **Step 4: Run tests, verify everything passes**

Run: `uv run python scripts/dev.py test tests/test_remote.py`
Expected: PASS — new chunked tests plus all pre-existing dry-run/submission tests (updated where they asserted monolithic-only content).

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/remote.py tests/test_remote.py
git commit -m "feat(remote): chunked self-resubmitting SLURM chain (default 10 min slices)"
```

---

### Task 5: stop sentinel + chain-aware attach with watchdog

**Files:**
- Modify: `src/cxr_mc/remote.py` — `_stop_jobid`, `_attach_progress_dashboard`, `_attach_log_stream`; add `_poll_chain`, `_is_terminal_state`.
- Test: `tests/test_remote.py` (append)

**Interfaces:**
- Consumes: Task 4's state strings and `$JOBDIR/STOP` sentinel.
- Produces: `_poll_chain(jobid) -> (state: str, slurm_live: bool, records: dict)` in ONE ssh round-trip; `_is_terminal_state(state)` = `state.startswith(("done", "FAILED", "cancelled"))`; module constant `_POLL_GRACE_POLLS = 15` (~30 s at the 2 s poll).

- [ ] **Step 1: Write the failing tests.** Monkeypatch `remote._ssh_capture` with scripted responses and `remote.time.sleep` with a no-op:

```python
def _poll_payload(state, sid="123", squeue="RUNNING", progress=""):
    return (
        f"@@STATE\n{state}\n@@SID\n{sid}\n@@SQUEUE\n{squeue}\n@@PROGRESS\n{progress}\n"
    )


def test_poll_chain_parses_sections(monkeypatch):
    monkeypatch.setattr(
        remote, "_ssh_capture",
        lambda cmd: _poll_payload(
            "running hopg [1/1] since now", progress=
            '{"material":"hopg","total_cases":4,"cached_cases":1,'
            '"completed_new_cases":2,"state":"running"}'
        ),
    )
    state, live, records = remote._poll_chain("20260717-abc")
    assert state.startswith("running")
    assert live is True
    assert records["hopg"]["completed_new_cases"] == 2


def test_attach_dashboard_survives_slice_gap_and_ends_terminal(monkeypatch):
    responses = iter([
        _poll_payload("running hopg [1/1] since now"),
        _poll_payload("queued slice 2 now", squeue=""),   # inter-slice gap: not live
        _poll_payload("running hopg [1/1] since now"),    # next slice picked up
        _poll_payload("done [1/1] now", squeue=""),
    ])
    monkeypatch.setattr(remote, "_ssh_capture", lambda cmd: next(responses))
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    ok = remote._attach_progress_dashboard(
        "20260717-abc", "job: 20260717-abc\nmaterials: hopg\nslurm_job_id: 123\n"
    )
    assert ok is True


def test_attach_dashboard_watchdog_exits_on_broken_chain(monkeypatch, capsys):
    monkeypatch.setattr(
        remote, "_ssh_capture",
        lambda cmd: _poll_payload("queued slice 2 now", squeue=""),
    )
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    ok = remote._attach_progress_dashboard(
        "20260717-abc", "job: 20260717-abc\nmaterials: hopg\nslurm_job_id: 123\n"
    )
    assert ok is False
    assert "chain appears broken" in capsys.readouterr().out


def test_stop_writes_stop_sentinel_before_scancel(monkeypatch):
    commands = []
    monkeypatch.setattr(remote, "_slurm_job_id", lambda jobid: "123")
    monkeypatch.setattr(remote, "_slurm_state", lambda sid: "RUNNING")
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: commands.append(cmd[-1]))
    remote._stop_jobid("20260717-abc")
    (cmd,) = commands
    assert cmd.index("STOP") < cmd.index("scancel")
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run python scripts/dev.py test tests/test_remote.py -k "poll_chain or watchdog or slice_gap or stop_sentinel"`
Expected: FAIL — `_poll_chain` does not exist; the dashboard breaks out on the first not-live poll; `_stop_jobid`'s command has no STOP.

- [ ] **Step 3: Implement.**

```python
_POLL_GRACE_POLLS = 15  # ~30 s at the 2 s poll: covers normal inter-slice latency


def _is_terminal_state(state):
    """Chain-terminal persisted states (spec 3c): done / FAILED / cancelled."""
    return state.startswith(("done", "FAILED", "cancelled"))


def _poll_chain(jobid):
    """State + latest-SID squeue liveness + progress records in ONE round-trip."""
    _check_shell_tokens([jobid])
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    remote_cmd = (
        f'D="{jobdir}"; '
        'echo "@@STATE"; cat "$D/state" 2>/dev/null; '
        'echo "@@SID"; SID=$(sed -n "s/^slurm_job_id: //p" "$D/meta" 2>/dev/null | tail -1); '
        'printf "%s\\n" "$SID"; '
        'echo "@@SQUEUE"; case "$SID" in \'\'|*[!0-9]*) ;; *) '
        + _squeue_state_command("$SID", retired="STATE=")
        + 'printf "%s\\n" "$STATE" ;; esac; '
        'echo "@@PROGRESS"; for f in "$D"/progress/*.json; do '
        '[ -f "$f" ] || continue; cat "$f" 2>/dev/null || true; printf "\\n"; done'
    )
    out = _ssh_capture(remote_cmd)
    sections = {"STATE": [], "SID": [], "SQUEUE": [], "PROGRESS": []}
    current = None
    for line in out.splitlines():
        if line.startswith("@@"):
            current = line[2:]
        elif current in sections:
            sections[current].append(line)
    state = sections["STATE"][0].strip() if sections["STATE"] else ""
    live = any(line.strip() for line in sections["SQUEUE"])
    records = _parse_progress_records("\n".join(sections["PROGRESS"]))
    return state, live, records
```

`_attach_progress_dashboard`: keep the bar setup and the metadata guard; replace the poll loop body:

```python
    interrupted = broken = False
    missed = 0
    try:
        while True:
            state, live, records = _poll_chain(jobid)
            _update_progress_bars(bars, records)
            if _is_terminal_state(state):
                break
            missed = 0 if live else missed + 1
            if missed >= _POLL_GRACE_POLLS:
                broken = True
                break
            time.sleep(2)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for bar in bars.values():
            bar.close()
    if interrupted:
        _disconnect_hint(jobid)
        return False
    if broken:
        print(
            f"\nchain appears broken: no live SLURM job for ~30 s and job state is "
            f"not terminal -- check `cxr remote status {jobid}`"
        )
        return False
    print("\n--- job finished ---")
    print(_job_state(jobid))
    return True
```

(The final `_job_state` call can reuse the last polled `state` instead — do that, it saves a round-trip.) The dashboard no longer needs `scheduler_id` for the loop; keep the metadata guard that requires materials + a recorded SID as-is.

`_attach_log_stream`: replace the single-SID squeue wait with the same shell-side logic — terminal-state prefix check plus a missed-counter watchdog (15 misses at `sleep 2`); the up-to-30 s wait-for-first-SID loop collapses into the same grace counter:

```bash
tail -n 50 -F --retry "$D/log" 2>/dev/null & TP=$!
missed=0
while :; do
  ST=$(cat "$D/state" 2>/dev/null)
  case "$ST" in done*|FAILED*|cancelled*) break ;; esac
  SID=$(sed -n "s/^slurm_job_id: //p" "$D/meta" 2>/dev/null | tail -1)
  LIVE=
  case "$SID" in ''|*[!0-9]*) ;; *) <_squeue_state_command with retired="STATE="> LIVE="$STATE" ;; esac
  if [ -n "$LIVE" ]; then missed=0; else missed=$((missed + 1)); fi
  if [ "$missed" -ge 15 ]; then echo "chain appears broken -- check cxr remote status" >&2; break; fi
  sleep 2
done
sleep 1; kill "$TP" 2>/dev/null
printf "\\n--- job finished ---\\n"; cat "$D/state" 2>/dev/null
```

`_stop_jobid`: in the remote command, write the sentinel before scancel (spec 3b):

```python
    remote = (
        f'D="{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"; '
        ': > "$D/STOP"; '
        f'scancel {scheduler_id} || exit $?; '
        ...  # rest unchanged
    )
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run python scripts/dev.py test tests/test_remote.py`
Expected: PASS, including all pre-existing attach/stop tests (update any that assert the old single-SID dashboard loop).

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/remote.py tests/test_remote.py
git commit -m "feat(remote): STOP sentinel and chain-aware attach with broken-chain watchdog"
```

---

### Task 6: the `remote-gpu-jobs` skill

**Files:**
- Create: `.agents/skills/remote-gpu-jobs/SKILL.md`
- Generated mirror: run `uv run python scripts/dev.py sync-skills` (never hand-edit `.claude/skills`)

**Interfaces:** none consumed by code; content documents Tasks 1-5's user-facing behavior.

- [ ] **Step 1: Write the skill** (frontmatter style: copy an existing `.agents/skills/*/SKILL.md` header):

```markdown
---
name: remote-gpu-jobs
description: Use when about to run a sweep, heavy Monte-Carlo compute, or any GPU-bound cxr-mc workload -- routes it to the lab box via cxr remote instead of running locally.
---

# Remote GPU jobs

**Never run a sweep on the laptop.** Local multiprocessing sweeps crash WSL.
All heavy Monte-Carlo compute belongs on the lab GPU box (`qlmc`), submitted
through `cxr remote`.

## Workflow

1. `cxr remote sync` -- push the current working tree (start/scan do this
   automatically unless `--no-sync`).
2. `cxr remote start <materials...>` (or `--all`) -- submit and walk away.
   Survives ssh disconnect.
3. `cxr remote status` / `logs --follow` / `attach` -- observe any time.
4. `cxr remote pull <stems...>` when state is `done`.

## Chunking (the default)

`start`/`scan` submit a self-resubmitting chain of ~10-minute SLURM slices
(`--chunk-minutes`, default 10). Each slice boundary is a scheduler decision
point and every slice carries `--nice=10000`, so other users' jobs jump ahead
roughly every 10 minutes. The sweep still runs unattended to completion:
slices resume from the per-material checkpoint.

- `--chunk-minutes 0` deliberately takes the whole box in one allocation.
  Only do this when nobody else needs it.
- To free the box for someone: `cxr remote stop <material>`. The chain
  cancels cleanly (no orphan resubmission) and a later `start` resumes from
  checkpoint with nothing lost beyond the in-flight config.
```

- [ ] **Step 2: Mirror and verify**

Run: `uv run python scripts/dev.py sync-skills`
Expected: `.claude/skills/remote-gpu-jobs/SKILL.md` appears, identical content.

Run: `uv run python scripts/dev.py lint`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add .agents/skills/remote-gpu-jobs .claude/skills/remote-gpu-jobs
git commit -m "docs(skills): remote-gpu-jobs -- always route heavy compute to the box"
```

---

## Final verification (whole branch)

- `uv run python scripts/dev.py verify` — full test + lint + typecheck must pass.
- Spec cross-check: every design-doc component (1, 2, 3a-3c, 4) maps to a landed task; the Testing section's cases all exist.
