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
`_checkpoint_save` is atomic. Saves happen at **config granularity** — `_save()`
fires when a config's last energy completes — not per case, so a slice *killed*
mid-config loses that config's completed energies. A slice that exits *gracefully*
via the budget below performs one final save after draining, so the graceful path
resumes at case granularity (resume filtering is per `(name, E0)`) and loses
nothing. Re-invoking a scan therefore continues where the previous one stopped.
This is what makes fine-grained (~10 min) time-slicing both possible and safe: a
slice does a bounded amount of work, checkpoints, exits, and the next allocation
resumes.

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
  loop stops launching new cases, lets in-flight cases drain, does a **final
  save** (so completed cases of partially-finished configs persist — see Key
  enabling fact), and returns a completion flag `complete: bool` (True when every
  requested case is now cached).
- **`run_cases` must cooperate.** `run_sweep` hands the entire todo list to
  `run_cases` in one call (`run.py`), so there is no between-cases loop in
  `run_sweep` where a deadline check could live. `run_cases`
  (`src/cxr_mc/montecarlo.py`) gains a `should_stop: Callable[[], bool] | None`
  hook, checked before dispatching each case; when it fires, no new cases start
  and in-flight cases drain. `run_sweep` builds the callable from
  `max_seconds`/`time_fn`. (Alternative rejected: mini-batching `run_cases` calls
  from `run_sweep` — it would defeat the worker pool's pipelining and change
  progress-bar behavior.)
- Per-case transport parallelism is otherwise unchanged: `run_cases` still uses
  the worker pool across a material's cases. Only the decision of *when to stop
  starting new cases* is bounded.

`scan.py` (`src/cxr_mc/scan.py` `run()`):

- New CLI flag `--max-minutes FLOAT` (converted to `max_seconds`), threaded into
  `run_sweep`.
- Maps the returned `complete` flag to a process **exit code**:
  - `0` — fully complete.
  - `75` (`EX_TEMPFAIL`) — the budget was hit; work remains.
  - Any other nonzero exit keeps meaning **hard failure**, exactly as today.
    The three codes are semantically distinct and the chain treats them
    differently (Component 2).
- With `--all`, the budget spans the whole invocation: exit `75` if *any*
  material remains incomplete, `0` only when all are done. (The chain always
  invokes one material at a time, but the flag exists on the shared parser, so
  the multi-material semantics are defined rather than accidental.)
- **Progress-file state:** a budget exit must not write `state="done"` — the
  dashboard would show an incomplete material as finished. `_run_material`
  writes a new `state="paused"` when `complete` is False, and
  `_parse_progress_records` (`remote.py`) adds `"paused"` to its state
  whitelist so the record isn't silently dropped.

## Component 2 — chained submission in `remote.py`

New user-facing flag on `start` and `scan`:

- `--chunk-minutes FLOAT`, **default 10**.
- `--chunk-minutes 0` selects the current single monolithic allocation (escape
  hatch for when someone deliberately wants the whole box).

Chunked mode is the **default** so the box yields by default.

Flag interactions:

- `--parallel-materials` is only meaningful in monolithic mode. Chunked mode runs
  materials sequentially (see below), so combining `--parallel-materials` with a
  nonzero `--chunk-minutes` is a **hard error** — never silently ignored. The
  flag remains legal with `--chunk-minutes 0`.
- Every slice — **including slice 0**, submitted from the laptop — carries
  `--nice=10000`. The courtesy applies from the start: if someone is already
  waiting at submit time, they go first.
- Chunked slices get a **hard `#SBATCH --time` backstop of 3× chunk-minutes**
  instead of `UNLIMITED`. The soft budget only checks between completed cases, so
  a hung or pathologically long case would otherwise hold the box indefinitely —
  the exact monopolization this design exists to fix. The 3× margin exists
  because checkpointing is config-granular: the cap must comfortably exceed the
  worst-case *config* time, or a config that never fits inside the cap re-runs
  from scratch every slice (a livelock the checkpoint cannot break). Materials
  with configs longer than the cap need a larger `--chunk-minutes`; the default
  (10 min chunk → 30 min cap) is far above current per-config times. Monolithic
  mode keeps `UNLIMITED`.

**One logical job, one `jobdir`** (unchanged bookkeeping shape). The batch script
(`run.sh`, reused verbatim by every slice since it just resumes from checkpoint):

1. Sync happens once at chain start on the laptop side (not per slice).
2. The slice walks the material list in order, **skipping** any material with a
   `completed:` or `failed:` marker already in the log, running
   `scan.py <m> --max-minutes <remaining>` for each and decrementing a
   shell-tracked remaining-wall budget. Fully-cached materials return instantly.
   - Materials run **sequentially** within a slice (chunked mode uses
     `parallel_materials = 1`). We drop cross-material overlap to keep the wall
     budget and resume accounting simple; per-case parallelism is retained, so
     the throughput cost is modest.
   - Per-material exit codes are handled three ways:
     - `0` → append `completed: <m>` to the log (today's marker; `remote scan`'s
       auto-pull already keys off it).
     - `75` → incomplete; the material stays eligible for the next slice.
     - **any other nonzero → append `failed: <m>` to the log and never retry
       it.** Without this, a persistently crashing material (bad input, OOM, a
       bug) fails fast every slice and the chain resubmits forever — an infinite
       loop of short SLURM jobs that never reaches a terminal state. The
       marker-in-log mechanism persists across slices for free, mirroring
       `completed:`.
3. If every material now has a `completed:` or `failed:` marker → terminal: set
   state `done` (or `done with N warning(s)`, matching today's warn-and-continue
   summary), release reservations via the trap, stop.
4. Otherwise, **hand off — in this order**:
   1. Set state to `queued slice k+1 ...` *first*. (Ordering rationale: if the
      handoff state were written after `sbatch` and the script died in between,
      the EXIT trap would see a non-handoff state, mark the job `FAILED`, and
      release reservations — while slice k+1 is already pending and would then
      run unreserved. State-first closes that window; the residual failure mode
      — state says queued but nothing was submitted — is caught by the attach
      watchdog in 3c.)
   2. `sbatch --nice=10000 <jobdir>/run.sh`, **fail-closed** exactly like
      `_submit_slurm_command`: validate the returned SID is numeric; on any
      sbatch failure write `FAILED (slice resubmission) ...` to state and exit
      nonzero, so the trap releases reservations and attach terminates. Without
      this the chain dies silently with state stuck at `queued slice ...`,
      attach polls forever, and the reservation leaks.
   3. Append the new scheduler ID to `meta` (`slurm_job_id: <sid>`) and exit.

**Why tracking helpers barely change:** every existing helper already reads the
scheduler ID with `sed -n "s/^slurm_job_id: //p" ... | tail -1`. Appending a new
`slurm_job_id:` line per slice means `_slurm_job_id`, `_live_jobs`, and
`_submission_outcome` automatically follow the *current* slice with no change.
(The chain's `queued slice ...` state also matches `_submission_outcome`'s
existing `queued*` → pending classification.)

**The Zhai path stays monolithic.** `_zhai_queue_script` is a single
run-to-completion payload; it gains no chunking. It shares `_slurm_batch_script`,
so the trap's new handoff case (3a) touches it — but Zhai never writes a
`queued slice` state, so its behavior is unchanged.

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
`cancelling` state string landing before the trap fires. (If `stop` lands during
the inter-slice window, the recorded SID is the *pending* next slice — `scancel`
kills it before it ever runs, so no resubmit can happen; the sentinel covers the
running-slice case and doubles as defense if `scancel` itself fails.) On stop we
also release reservations (terminal) as today.

### 3c. Attach / status terminal condition

The progress dashboard (`_attach_progress_dashboard`) reads `slurm_job_id` once and
exits when that SID leaves `squeue` — which in a chain happens at *every* slice
boundary. Fix: attach loops until `_job_state()` is terminal
(`done` / `FAILED` / `cancelled`), re-reading progress each poll. A between-slices
gap shows a non-terminal state (`queued slice k`), so the viewer keeps tracking
across the whole chain. The raw-log viewer (`_attach_log_stream`) gets the same
terminal-state-based termination.

Two refinements:

- **Broken-chain watchdog:** state-only termination hangs forever if the chain
  zombies (e.g. a slice crashed between writing the handoff state and `sbatch` —
  the residual window from Component 2 step 4). Each poll therefore also re-reads
  the latest SID from `meta`; if the state is non-terminal but no recorded SID has
  been live in `squeue` for a grace window (~30 s, covering normal inter-slice
  latency), attach exits with an explicit "chain appears broken — check
  `cxr remote status`" warning instead of polling forever.
- **One round-trip per poll:** the dashboard already makes two ssh calls per
  2-second poll (`_read_progress_records`, `_slurm_state`); naively adding
  `_job_state` makes three. Fold state, latest SID, and progress records into a
  single remote command per poll.

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
  stops early, **force-saves the partial config**, and returns `complete=False`;
  a second call over the same checkpoint finishes and returns `complete=True`,
  proving case-granular resume. Deterministic, no sleeps.
- **`run_cases` `should_stop` hook**: no new case dispatches after the hook
  fires; in-flight cases drain and their callbacks run.
- **`scan.py` exit codes**: `--max-minutes` maps an incomplete sweep to exit `75`
  and a complete one to exit `0`; a hard failure keeps its non-75 nonzero exit;
  a budget exit writes progress state `paused`, never `done`.
- **`_parse_progress_records`** accepts the new `paused` state.
- **`remote.py` chain script** (extend the existing `--dry-run` snapshot tests):
  the chained `run.sh` contains the resubmit `sbatch --nice`, the state-before-
  sbatch handoff ordering, the fail-closed SID validation, the `STOP`-file gate,
  the `failed:` marker skip, terminal-only reservation release, and the 3×
  `--time` backstop; `--chunk-minutes 0` still emits the monolithic
  `UNLIMITED`-time script; `--parallel-materials` with nonzero `--chunk-minutes`
  is rejected.
- **Chain state machine**, against fabricated job-dir state:
  - handoff / terminal / stop transitions;
  - **persistent failure**: a material with a `failed:` marker is skipped and,
    once every material is completed-or-failed, the chain terminates
    `done with N warning(s)` — no infinite resubmission;
  - **resubmit failure**: a failed `sbatch` yields `FAILED (slice resubmission)`
    and reservations are released;
  - attach's terminal-exit condition, including the broken-chain watchdog
    (non-terminal state + no live SID past the grace window → warn and exit).

## Scope

Touches `src/cxr_mc/remote.py`, `src/cxr_mc/scan.py`, `src/cxr_mc/run.py`,
`src/cxr_mc/montecarlo.py` (the `run_cases` `should_stop` hook), the new
`.agents/skills/remote-gpu-jobs/` skill (+ its `.claude/skills` mirror), and their
tests. It leaves the currently-uncommitted `sweep.py` / `test_sweep.py` /
`mats_to_sim.toml` changes untouched.
