# Compute performance optimization

Branch: `feature/compute-performance-optimization`

TODO scope: Active item "Compute performance optimization," plus locally
adopted and reviewed Inbox item "CPU Performance Flag." Per the local-only
triage override, its Inbox marker remains on authoritative `main:TODO.md` for
later reconciliation on `main`.

## Goal

Use representative remote measurements to find and remove material
bottlenecks. Improve end-to-end throughput while keeping CPU, GPU, RAM, and
VRAM within stable limits. Keep the GPU fed only where feed-wait evidence
identifies host transport as the constraint.

The requested roughly 85% utilization is a capacity target, not an independent
success metric: unused RAM is not a defect, and forcing every resource to the
same percentage can reduce throughput or OOM margin.

## Current evidence

- `--performance-profile` emits `cxr.performance.v1` NDJSON and the analyzer
  excludes incomplete sessions.
- The GPU path intentionally uses one CUDA context with CPU transport
  pipelined behind serial GPU spectrum/bremsstrahlung work.
- `Sweep.spec_chunk` and `Sweep.brem_chunk` are the existing fine-grained VRAM
  controls.
- Historical `compute_test_300keV` runs found 20,000 line segments stable and
  substantially faster than 22,000-30,000 because larger chunks repeatedly
  entered OOM fallback. Those artifacts are diagnostic history, not a current
  control/candidate acceptance baseline.
- The branch previously carried phase-specific adaptive OOM fallback and
  terminal-session classification fixes. Fresh remote validation remains
  required before calling them an end-to-end improvement.

## Existing CPU-profiler behavior and bug

Remote `--nsys` currently always appends a second, serial CPU-only pass in
`src/cxr_mc/_remote/scripts.py:_cpu_profile_block`:

- forced `CXR_MC_BACKEND=cpu`, `--workers 0`, `--quick`, and
  `--max-minutes 10`;
- fresh job-local checkpoints so cache hits do not erase the profile;
- `cProfile` and `pstats` artifacts at
  `$JOBDIR/performance/<profile>/<material>.cpu.{prof,txt}`;
- no performance sampler flag, preventing the CPU pass from overwriting the
  GPU run's telemetry;
- runtime opt-out only through `CXR_MC_NO_CPU_PROFILE=1`.

This makes the slow CPU pass implicit whenever `--nsys` is requested. After
the combined CPU/GPU Nsight run completes, the CPU-only subprocess starts but
is not represented correctly by the attached dashboard. It writes
`progress/<material>.cpu.json`; `_status_remote_command` sends both progress
files, while `_parse_progress_records` keys them by material. The records can
therefore overwrite each other, and job state/metadata still describe the
Nsight phase rather than the active CPU-only phase.

## CPU profiling CLI contract

Scope is `cxr remote run`; no local-run flag is added by this slice.

- `-c/--cpu`: opt into the existing serial CPU cProfile pass after the selected
  normal performance run. It implies performance mode, uses the selected
  catalog profile as the artifact profile, and remains compatible with
  `--nsys`. `--nsys` alone stops launching the CPU pass.
- `--cpu-only`: run only the forced-CPU cProfile pass. Skip the normal GPU run,
  Nsight capture, GPU checkpoints, and GPU performance sampler. It implies
  performance artifact ownership for the selected catalog profile.
- `--cpu` and `--cpu-only` are mutually exclusive. `--cpu-only` and `--nsys`
  are also incompatible because "CPU only" must not start a GPU/Nsight phase.
- Both CPU modes remain monolithic (`--chunk-minutes 0`) and use isolated
  job-local checkpoints. Preserve current `--quick`, serial backend, and hard
  time-limit safeguards unless measurements justify a different bounded
  workload.
- No CPU profiler runs without either flag. Remove the environment-variable
  opt-out as a public control; retaining it as an emergency internal kill
  switch is acceptable but must not change CLI truth.
- CPU-profiler failure is terminal for `--cpu-only`. For `--cpu`, retain the
  completed primary artifacts, mark the CPU phase failed, and return a job
  failure rather than silently reporting full profiling success.

## Dashboard contract

- Metadata and mode summaries expose `cpu: true/false` and
  `cpu_only: true/false` distinctly from `nsys`.
- Job state changes before the CPU subprocess starts, e.g. `profiling CPU
  <material>`, and reaches a CPU-specific completed/failed terminal event.
- Progress records carry a phase/run identity. Dashboard aggregation must not
  collapse GPU and CPU records solely because both name the same material.
- During `--cpu`, dashboard shows the current phase and current CPU progress;
  completed GPU progress remains visible without replacing the active record.
- During `--cpu-only`, no GPU phase or GPU-progress row is synthesized.
- `cxr remote status`, `--attach`, logs, job listing, performance inventory,
  pull, and prune agree on phase and terminal state.

## Owning paths

- CLI options, implications, exclusions, help:
  `src/cxr_mc/_remote/cli.py`
- Submission validation, state, pull hints:
  `src/cxr_mc/_remote/lifecycle.py`
- queue script, CPU subprocess, metadata:
  `src/cxr_mc/_remote/scripts.py`
- progress collection and rendering:
  `src/cxr_mc/_remote/viewer.py`, `src/cxr_mc/cli/_dashboard.py`
- performance artifact lifecycle:
  `src/cxr_mc/cli/performance.py`, remote lifecycle helpers
- profiler/script/CLI regressions: `tests/test_remote.py`,
  `tests/test_remote_click.py`, CLI contract snapshot

## Stepwise checklist

- [ ] Add `-c/--cpu` and `--cpu-only` to `cxr remote run`; encode mutual
      exclusions, implied performance mode, monolithic allocation, and help.
- [ ] Thread explicit CPU mode through Click invocation, lifecycle validation,
      queue metadata, script builders, mode summaries, and dry-run output.
- [ ] Stop coupling `_cpu_profile_block` to `nsys`; launch it only for `--cpu`
      or as the sole workload for `--cpu-only`.
- [ ] Refactor CPU pass into a first-class queue phase with explicit start,
      completion, failure, exit status, and artifact paths.
- [ ] Make progress aggregation phase-aware; preserve simultaneous historical
      GPU and active CPU records for one material without filename-order wins.
- [ ] Update status/attach/jobs/logs and artifact list/pull/prune behavior for
      CPU-only and combined profiling jobs.
- [ ] Add focused CLI/script/metadata/dashboard tests, including shell syntax
      validation and failure propagation.
- [ ] Regenerate `docs/cli-reference.md` and
      `tests/data/cli_contract.json` after the CLI change.
- [ ] Run focused CPU checks locally; exercise the slow profiler only through
      an authorized remote quick-profile job.

## Broader optimization path

1. Reproduce a complete matched baseline, including a controlled 300 keV
   cohort with identical profile, material grid, seed/backend, allocation,
   workers, chunks, cache state, and checkpoint state.
2. Validate activity labels against direct timings. Separate checkpoint I/O,
   CPU transport, GPU spectrum/bremsstrahlung, feed wait, synchronization, and
   OOM retry time.
3. Profile remaining material hotspots; change only measured bottlenecks.
4. Sweep bounded worker/chunk settings and report throughput, peak RSS/VRAM,
   utilization distributions, failures, repeats, and spread.
5. Preserve checkpoint correctness, resumability, physics outputs, seeds,
   grids, and fidelity across comparisons.

## Decisions and open questions

Decided:

- CPU profiling becomes explicit; `--nsys` no longer pays its cost by default.
- `--cpu-only` means no primary GPU/Nsight run, not "run GPU then hide it."
- Dashboard needs phase identity; renaming the progress file alone cannot fix
  material-keyed aggregation.
- Heavy profiling remains remote-only.

Open during implementation:

- Exact phase-aware progress schema: extend the existing record with a stable
  `phase`/`run_id`, or represent phases as nested records. Preserve backward
  compatibility with existing job progress files.
- Whether combined `--cpu` should retain both phases in one dashboard table or
  show completed-primary plus active-secondary summaries. Choose the smallest
  representation that preserves phase, status, and failure truth.

## Delegation / required skills

- Worker: `lead-task` because CLI, generated contract, remote scripts,
  dashboard state, and artifact lifecycle cross subsystem boundaries.
- Required: `performance`, `remote-gpu-jobs`, `cli-ui-ux`,
  `regression-testing`, `documentation-maintenance`.
- Add `scientific-library` or `monte-carlo` only if implementation changes the
  runner/profiler target rather than orchestration.
- Authority: task-local checkpoint commits only; no push, TODO editing, or
  remote mutation without explicit supervisor/user authority.

## Acceptance checks

- `cxr remote run PROFILE --nsys` performs only the Nsight/primary run.
- `cxr remote run PROFILE --nsys --cpu` performs primary then CPU phases and
  reports both accurately while attached.
- `cxr remote run PROFILE --cpu-only` starts no GPU/Nsight scan and produces
  `.cpu.prof` plus `.cpu.txt` from an uncached bounded CPU workload.
- Invalid combinations fail before sync/staging/submission with exact Click
  usage errors.
- CPU phase appears live in `status --attach`; its progress is never hidden by
  the completed primary record.
- CPU-only failure fails the job. Combined CPU failure preserves primary
  artifacts but leaves an unambiguous failed terminal state.
- Existing performance list/pull/prune includes CPU artifacts and remains
  preview-by-default/revalidation-safe.
- Focused remote CLI/script/dashboard tests and CLI contract checks pass;
  generated CLI reference is current.

## Non-goals

- Maximizing allocation percentages without throughput benefit.
- Heavy local WSL profiling.
- Changing physics, seeds, grids, or fidelity to improve profile runtime.
- Unbounded GPU process multiplication.
- Redesigning the entire dashboard outside phase-aware CPU tracking.

## Stop conditions

Stop on unrelated dirty work, incompatible workload comparisons, unavailable
remote allocation for runtime acceptance, telemetry ambiguity that prevents
phase attribution, or any proposed speedup that changes physics outputs.
