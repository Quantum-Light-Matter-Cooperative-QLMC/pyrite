# Remote progress dashboard

Branch: `feature/remote-progress-dashboard`

TODO scope: former P1 items 2–3, one canonical remote-status command, pending
SLURM queue context, and compute-time estimates in its dashboard.

## Goal

When a job is pending, show its scheduler consideration position and the top
pending item in the same queue view. Beside overall compute progress, show
elapsed, estimated remaining, and estimated total compute time.

Combine one-shot and continuous monitoring under:

```text
cxr remote status [JOB]
cxr remote status [JOB] -a
cxr remote status [JOB] --attach
```

Without `--attach`, `status` prints one snapshot and exits. With
`-a/--attach`, it continuously monitors the same dashboard until interrupted.

Keep these as structured dashboard/status information. Do not add them to raw
job logs.

## Decisions

- `status` is the only public command for the dashboard. Remove the standalone
  `attach` command, including registration, help, completion, contract
  snapshots, tests, and current documentation. Do not retain a hidden alias.
- `status` preserves one-shot behavior by default. `-a/--attach` selects the
  existing continuous, reconnecting monitor over the same job-selection and
  verbosity options.
- Interrupting `status --attach` disconnects only the viewer, reports that the
  remote job remains active, and preserves the established interruption exit
  behavior.
- One-shot and attached status use one acquisition and rendering path and
  receive queue/timing features at every useful verbosity without extra
  per-frame SSH sessions.
- Define queue position explicitly from Slurm's pending priority order, scoped
  to the relevant partition/cohort. It is scheduler consideration order, not a
  guaranteed start order: priority can change and backfill can run lower-ranked
  jobs first.
- Show compact sanitized top-item identity and pending reason. Never imply an
  exact start time when Slurm does not provide one.
- Fetch the target and pending cohort in one bounded `squeue` call per status
  snapshot. Preserve transport errors; never display stale or invented rank.
- ETA uses compute progress, not Slurm allocation `%L`. Extend progress records
  with explicit-second timing fields needed to survive polling and chunked
  resume.
- Prefer cost-weighted progress when present. Cached work, zero progress,
  parallel materials, pauses, restarts, failures, completion, and legacy
  records must have defined behavior. Render `—` until a finite estimate is
  defensible.
- Treat ETA as an estimate, label it accordingly, and smooth or stabilize it
  enough that two-second polling does not produce distracting oscillation.
- Preserve terminal sanitization, `NO_COLOR`, redirected output, and existing
  status/attach failure semantics.

## Owning paths

- Remote snapshot acquisition/streaming: `src/cxr_mc/_remote/viewer.py`
- Scheduler state and shared query helpers: `src/cxr_mc/_remote/state.py`,
  `src/cxr_mc/_remote/scripts.py`
- Dashboard parsing/rendering: `src/cxr_mc/_remote/presentation.py`
- Atomic progress records: `src/cxr_mc/scan.py`; component writers in
  `src/cxr_mc/blaze.py`, `src/cxr_mc/rebrem.py`, and `src/cxr_mc/reline.py`
- CLI contract: `src/cxr_mc/_remote/cli.py`
- Tests: `tests/test_remote.py`, `tests/test_scan_cli.py`, and component-specific
  progress tests

## Implementation path

1. Move continuous-monitor dispatch behind `status -a/--attach`; remove the
   standalone `attach` command and stale public references while preserving
   job selection, verbosity, reconnection, interruption, and disconnect hints.
2. Specify the machine payload for queue rank/top item, including partition
   scope, ordering, unavailable fields, and hostile scheduler text.
3. Extend the existing marked status snapshot with one bounded queue query.
   Parse and sanitize it at the presentation boundary.
4. Define additive progress timing fields and units. Accumulate active compute
   time correctly across checkpoint resume and chunked self-submission without
   breaking legacy record parsing.
5. Derive job-wide elapsed/remaining/total estimates from validated aggregate
   progress, including parallel-material and cost-weighted cases.
6. Render queue context only while pending and timing beside the overall bar.
   Keep case rows compact and one-shot/attached frames identical.
7. Regenerate CLI contract/reference, update current examples, and search for
   stale `remote attach` command references.

## Verification

- Pure tests for pending ranks: first/middle/last, ties, partition scope,
  priority changes, backfill caveat text, malformed/ANSI/control data, retired
  IDs, and `squeue` failure.
- Pure ETA tests: no progress, one tick, cached-only start, cost weighting,
  multiple parallel materials, pause/resume, chunk restart, failed/done,
  zero-total, legacy records, and non-finite/hostile values.
- CLI tests proving `status` remains one-shot by default, both `-a` and
  `--attach` select continuous monitoring, standalone `attach` is absent, and
  job/detail options compose with attached mode.
- Snapshot tests for narrow/redirected output, `NO_COLOR`, detail levels,
  interruption/disconnect behavior, and identical one-shot/attached frames.
- Confirm one SSH status stream and one bounded scheduler query per frame.
- Regenerate `tests/data/cli_contract.json` and `docs/cli-reference.md`.
- Focused remote/scan/component suites, lint/type checks, then real
  `cxr remote --help`, `cxr remote status --help`, and one-shot/attached
  subprocess probes. No live submission required for unit acceptance.

## Non-goals

- Changing Slurm priority, partition, or backfill policy.
- Promising exact execution order or scheduler start time.
- Writing dashboard metadata into simulation logs.
- Performance-kernel or checkpoint-throughput optimization.
- New polling connections or unbounded full-cluster history queries.
- Retaining `cxr remote attach` as a public or hidden compatibility alias.

## Dispatch

Worker skill: `implement-task`

Required skills: `cli-ui-ux`, `remote-gpu-jobs`, `regression-testing`.

Authority: checkpoint commits; no push; no TODO writing; no delegation.

Stop on an undefined queue cohort, inability to distinguish active compute from
queue/pause time, a proposed extra SSH connection per refresh, unrelated dirty
work, or output changes lacking explicit compatibility tests.

## Implementation status

- [x] Canonicalize dashboard monitoring under `status -a/--attach`; remove
  standalone `attach` command and current references.
- [x] Define queue cohort as all active jobs in the configured SLURM partition;
  rank pending jobs by descending scheduler priority then numeric job ID.
- [x] Acquire target allocation and partition cohort through one bounded
  `squeue` snapshot inside the existing framed SSH stream.
- [x] Persist additive `active_compute_seconds`, `measured_new_cases`, and
  optional `measured_new_cost` across worker-process/chunk resumes.
- [x] Derive stable compute elapsed/ETA/estimated-total values without counting
  cached work, queue time, or paused time as measured throughput.
- [x] Cover priority/rank/ties/partition changes, hostile/retired/error data,
  cached/cost/parallel/pause/restart/failure/legacy timing, CLI migration,
  interruption, frame reuse, and generated CLI artifacts.
- [x] Refresh current cluster guidance, repository map, CLI contract, and CLI
  reference.

Verification on 2026-07-29:

- `scripts/dev.py test tests/test_remote.py tests/test_scan_budget.py
  tests/test_scan_beam_options.py tests/test_scan_coherent.py tests/test_run.py
  tests/test_blaze.py` — 429 passed.
- `scripts/dev.py lint` — passed.
- CLI contract/reference `--check` — passed.
- `scripts/dev.py typecheck` — changed paths clean; whole check remains blocked
  by unresolved optional `imageio.v3` and `kaleido` imports in
  `src/cxr_mc/plots/render_trajectories.py`.
- Real `cxr remote --help`, `cxr remote status --help`,
  `status --attach --json`, and removed `remote attach` probes matched the
  documented command contract.
