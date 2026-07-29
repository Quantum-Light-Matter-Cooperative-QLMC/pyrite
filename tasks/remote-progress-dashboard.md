# Remote progress dashboard

Branch: `feature/remote-progress-dashboard`

TODO scope: former P1 items 2–3, pending SLURM queue context and compute-time
estimates in `cxr remote attach` and related status dashboards.

## Goal

When a job is pending, show its scheduler consideration position and the top
pending item in the same queue view. Beside overall compute progress, show
elapsed, estimated remaining, and estimated total compute time.

Keep these as structured dashboard/status information. Do not add them to raw
job logs.

## Decisions

- `status` and `attach` share one acquisition and rendering path; both receive
  the feature at every useful verbosity without extra per-frame SSH sessions.
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

1. Specify the machine payload for queue rank/top item, including partition
   scope, ordering, unavailable fields, and hostile scheduler text.
2. Extend the existing marked status snapshot with one bounded queue query.
   Parse and sanitize it at the presentation boundary.
3. Define additive progress timing fields and units. Accumulate active compute
   time correctly across checkpoint resume and chunked self-submission without
   breaking legacy record parsing.
4. Derive job-wide elapsed/remaining/total estimates from validated aggregate
   progress, including parallel-material and cost-weighted cases.
5. Render queue context only while pending and timing beside the overall bar.
   Keep case rows compact and status/attach output identical.
6. Update CLI output contracts/reference only if public help or snapshots
   change; search for duplicated status renderers.

## Verification

- Pure tests for pending ranks: first/middle/last, ties, partition scope,
  priority changes, backfill caveat text, malformed/ANSI/control data, retired
  IDs, and `squeue` failure.
- Pure ETA tests: no progress, one tick, cached-only start, cost weighting,
  multiple parallel materials, pause/resume, chunk restart, failed/done,
  zero-total, legacy records, and non-finite/hostile values.
- Snapshot tests for narrow/redirected output, `NO_COLOR`, detail levels, and
  identical `status`/`attach` frames.
- Confirm one SSH status stream and one bounded scheduler query per frame.
- Focused remote/scan/component suites, lint/type checks, then real
  `cxr remote status --help` and `cxr remote attach --help` probes. No live
  submission required for unit acceptance.

## Non-goals

- Changing Slurm priority, partition, or backfill policy.
- Promising exact execution order or scheduler start time.
- Writing dashboard metadata into simulation logs.
- Performance-kernel or checkpoint-throughput optimization.
- New polling connections or unbounded full-cluster history queries.

## Dispatch

Worker skill: `implement-task`

Required skills: `cli-ui-ux`, `remote-gpu-jobs`, `regression-testing`.

Authority: checkpoint commits; no push; no TODO writing; no delegation.

Stop on an undefined queue cohort, inability to distinguish active compute from
queue/pause time, a proposed extra SSH connection per refresh, unrelated dirty
work, or output changes lacking explicit compatibility tests.
