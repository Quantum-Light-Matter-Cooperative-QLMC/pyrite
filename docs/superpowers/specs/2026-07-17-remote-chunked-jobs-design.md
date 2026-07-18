# Design: chunked, queue-yielding remote GPU jobs

## Problem

`cxr remote start`/`scan` submit **one** SLURM allocation (`_slurm_batch_script`)
with `--time=UNLIMITED` that internally loops over every material with bounded
concurrency. On the single-GPU lab box (`qlmc`) that allocation holds the GPU for
the whole sweep — often 16–20 h. SLURM only re-evaluates the queue when an
allocation *ends*, so one monolithic job gives the scheduler zero decision points:
fair-share never runs, and another user (e.g. a postdoc) cannot slot in until the
entire sweep finishes.

A secondary, standing problem this addresses: heavy sweeps must never be run on the
laptop. Local multiprocessing sweeps crash WSL. All heavy Monte-Carlo compute
belongs on the remote box.

## Goal

Break a sweep into small slices — **~10 minutes each** — where every slice is a
*separate SLURM allocation*. Each slice boundary becomes a scheduler decision
point, and slices carry `--nice` so any normal-priority job jumps ahead of the
next slice. The result: the box yields to other users roughly every 10 minutes,
while the sweep still runs to completion unattended and survives ssh disconnect.

## Key enabling fact

Scans already resume from checkpoint. `run_sweep` (`src/cxr_mc/run.py`) skips every
case already present in `checkpoints/<material>.pkl` (`resume=True`), and
`_checkpoint_save` is atomic and per-case ("resumable on crash"). Re-invoking a
scan therefore continues exactly where the previous one stopped, losing at most the
single in-flight case. This is what makes fine-grained (~10 min) time-slicing both
possible and safe: a slice does a bounded amount of work, exits, and the next
allocation resumes.

## Approach (chosen)

**Self-resubmitting SLURM chain.** The laptop submits slice 0 and walks away. Each
slice: resume checkpoint → do ~10 min of work → checkpoint → exit. If the sweep is
not yet complete, the slice `sbatch`s the next slice (with `--nice`) and ends; when
the sweep is complete it stops. Exactly one job sits in the queue at a time, the
chain handles an unknown number of slices, and it preserves the module's
launch-and-walk-away property.

Alternatives considered and rejected:

- **Throttled array** (`--array=0-N%1`): must guess `N`, and litters the queue with
  pending tasks. No advantage over the chain here.
- **Laptop-driven submit-poll-resubmit**: requires the laptop awake and connected
  for the whole run — exactly the WSL-crash failure mode we are moving away from.

## Component 1 — `--max-minutes` soft budget

`run_sweep` gains a soft wall-clock budget:

- New parameter `max_seconds: float | None = None` (default `None` = today's
  unbounded behavior) plus an injectable `time_fn` (default `time.monotonic`) so
  tests drive a fake clock and avoid sleep-based flakiness.
- The budget is checked **between completed cases**. Once the deadline passes, the
  loop stops launching new cases, lets in-flight cases drain, saves, and returns a
  completion flag `complete: bool` (True when every requested case is now cached).
- Per-case transport parallelism is unchanged: `run_cases` still uses the worker
  pool across a material's cases. Only the decision of *when to stop starting new
  cases* is bounded.

`scan.py` (`src/cxr_mc/scan.py` `run()`):

- New CLI flag `--max-minutes FLOAT` (converted to `max_seconds`), threaded into
  `run_sweep`.
- Maps the returned `complete` flag to a process **exit code**:
  - `0` — the material is fully complete.
  - `75` (`EX_TEMPFAIL`) — the budget was hit; work remains.
- This exit code is the chain's per-material completion signal.

## Component 2 — chained submission in `remote.py`

New user-facing flag on `start` and `scan`:

- `--chunk-minutes FLOAT`, **default 10**.
- `--chunk-minutes 0` selects the current single monolithic allocation (escape
  hatch for when someone deliberately wants the whole box).

Chunked mode is the **default** so the box yields by default.

**One logical job, one `jobdir`** (unchanged bookkeeping shape). The batch script
(`run.sh`, reused verbatim by every slice since it just resumes from checkpoint):

1. Sync happens once at chain start on the laptop side (not per slice).
2. The slice walks the material list in order, running
   `scan.py <m> --max-minutes <remaining>` for each, decrementing a shell-tracked
   remaining-wall budget. Fully-cached materials return instantly.
   - Materials run **sequentially** within a slice (chunked mode uses
     `parallel_materials = 1`). We drop cross-material overlap to keep the wall
     budget and resume accounting simple; per-case parallelism is retained, so the
     throughput cost is modest.
3. If every material exited `0` → set state `done`, release reservations, stop.
4. Otherwise → `sbatch --nice=10000 <jobdir>/run.sh` for the next slice, append the
   new scheduler ID to `meta` (`slurm_job_id: <sid>`), set state to
   `queued slice k+1 ...`, and exit.

**Why tracking helpers barely change:** every existing helper already reads the
scheduler ID with `sed -n "s/^slurm_job_id: //p" ... | tail -1`. Appending a new
`slurm_job_id:` line per slice means `_slurm_job_id`, `_live_jobs`, and
`_submission_outcome` automatically follow the *current* slice with no change.

## Component 3 — the three lifecycle subtleties

These are where the chain differs from a single job and where bugs would hide.

### 3a. Reservation lifecycle

Today `_slurm_batch_script`'s EXIT trap (`finish` → `release_reservations`) frees
the checkpoint reservation on *every* exit. In a chain that would release the stem
mid-run and let another job claim it. Fix: reservations are released **only on a
terminal state** (`done` / `FAILED` / `cancelled`), never on a handoff exit. The
`finish` trap gains a "handoff" case (state set to `queued slice ...`) that neither
marks failure nor releases reservations. The reservation therefore spans the whole
chain and is freed exactly once, at terminal.

### 3b. Stop must not zombie the chain

If `stop` scancels the running slice, that slice's EXIT trap must not resubmit the
next one. Fix: `_stop_jobid` writes a `STOP` sentinel file into the jobdir **before**
issuing `scancel`. The slice's resubmit step checks `[ -f "$JOBDIR/STOP" ]` and
skips resubmission when present. This is race-free, unlike depending on the
`cancelling` state string landing before the trap fires. On stop we also release
reservations (terminal) as today.

### 3c. Attach / status terminal condition

The progress dashboard (`_attach_progress_dashboard`) reads `slurm_job_id` once and
exits when that SID leaves `squeue` — which in a chain happens at *every* slice
boundary. Fix: attach loops until `_job_state()` is terminal
(`done` / `FAILED` / `cancelled`), re-reading progress each poll. A between-slices
gap shows a non-terminal state (`queued slice k`), so the viewer keeps tracking
across the whole chain. The raw-log viewer (`_attach_log_stream`) gets the same
terminal-state-based termination instead of single-SID `squeue` liveness.

## Component 4 — the skill

Create a repo skill so future agents always route heavy compute to the box.

- Canonical source: `.agents/skills/remote-gpu-jobs/SKILL.md`.
- Mirror into `.claude/skills` via `uv run python scripts/dev.py sync-skills`
  (never hand-edit `.claude/skills`).
- Description must trigger whenever an agent is about to run a sweep / heavy
  Monte-Carlo / GPU compute.
- Content:
  - **Always** submit via `cxr remote start` (chunked by default, survives
    disconnect); **never** run a sweep on the laptop — it crashes WSL.
  - The workflow: `sync` → `start` → `status`/`logs`/`attach` → `pull` when state
    is `done`.
  - The chunking default (~10 min slices) and its courtesy rationale (`--nice`
    yields the box to other users every slice).
  - How to `stop` a chain to free the box for someone else, and that the chain
    resumes cleanly from checkpoint when restarted.

## Testing

- **`run_sweep` budget** (`tests/`): with an injected fake clock, a tiny budget
  stops early and returns `complete=False`; a second call over the same checkpoint
  finishes and returns `complete=True`, proving resume. Deterministic, no sleeps.
- **`scan.py` exit code**: `--max-minutes` maps an incomplete sweep to exit `75`
  and a complete one to exit `0`.
- **`remote.py` chain script** (extend the existing `--dry-run` snapshot tests):
  the chained `run.sh` contains the resubmit `sbatch --nice`, the `STOP`-file gate,
  and terminal-only reservation release; `--chunk-minutes 0` still emits the
  monolithic script.
- **Chain state machine**: handoff / terminal / stop transitions, and the attach
  terminal-exit condition, exercised against fabricated job-dir state.

## Scope

Touches `src/cxr_mc/remote.py`, `src/cxr_mc/scan.py`, `src/cxr_mc/run.py`, the new
`.agents/skills/remote-gpu-jobs/` skill (+ its `.claude/skills` mirror), and their
tests. It leaves the currently-uncommitted `sweep.py` / `test_sweep.py` /
`mats_to_sim.toml` changes untouched.
