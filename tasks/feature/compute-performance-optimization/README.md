# Compute performance optimization

Branch: `feature/compute-performance-optimization`

TODO scope: Active item "Compute performance optimization." The formerly
adopted Inbox item "CPU Performance Flag" is implemented (below) and its
marker is already dropped from `main:TODO.md`; reconciliation is done.

Current open slice (2026-08-08): the Active item's own text — review whether
the `cupyx.jit.rawkernel`/Numba `@njit` optimizations from
[`docs/compute-performance-optimization.md`](../../../docs/compute-performance-optimization.md)
Rounds 1-3 are optimally executed, review any critical physics changes they
carry, add bit-for-bit/toleranced validation for the new paths, and confirm
non-NVIDIA fallbacks. Plus fresh remote evidence of a transport-side stall on
heavy materials at large `--ne-line`. See "Next slice" below.

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
- profiler/script/CLI regressions: `tests/remote/test_remote.py`,
  `tests/remote/test_click.py`, CLI contract snapshot

## Stepwise checklist

- [x] Add `-c/--cpu` and `--cpu-only` to `cxr remote run`; encode mutual
      exclusions, implied performance mode, monolithic allocation, and help.
- [x] Thread explicit CPU mode through Click invocation, lifecycle validation,
      queue metadata, script builders, mode summaries, and dry-run output.
- [x] Stop coupling `_cpu_profile_block` to `nsys`; launch it only for `--cpu`
      or as the sole workload for `--cpu-only`.
- [x] Refactor CPU pass into a first-class queue phase with explicit start,
      completion, failure, exit status, and artifact paths.
- [x] Make progress aggregation phase-aware; preserve simultaneous historical
      GPU and active CPU records for one material without filename-order wins.
- [x] Update status/attach/jobs/logs and artifact list/pull/prune behavior for
      CPU-only and combined profiling jobs.
- [x] Add focused CLI/script/metadata/dashboard tests, including shell syntax
      validation and failure propagation.
- [x] Regenerate `docs/cli-reference.md` and
      `tests/data/cli_contract.json` after the CLI change.
- [x] Run focused CPU checks locally without heavy compute.
- [ ] Exercise the slow profiler through an authorized remote quick-profile
      job; not run in this slice because remote mutation was explicitly out of
      authority.

## Implementation evidence

- `-c/--cpu` now implies the selected profile's performance mode and a
  monolithic allocation, then runs the bounded serial CPU cProfile phase after
  the primary phase. `--nsys` alone no longer starts cProfile.
- `--cpu-only` starts no primary scan, performance sampler, Nsight command, or
  primary checkpoint reservation. It rejects `--cpu`, `--nsys`, primary
  repetition/sampling knobs, GPU chunk pins, and nonzero chunk allocations at
  the Click boundary before lifecycle submission.
- Queue metadata records `nsys`, `cpu`, and `cpu_only` separately. Primary and
  CPU progress records carry `phase: primary|cpu`; dashboard keys and labels
  retain both records for one material and select the active CPU phase for the
  headline while preserving the completed primary row.
- CPU phase state records start, per-material completion/failure, and final
  `done CPU profile` / `FAILED CPU profile` truth. CPU failures return nonzero;
  combined runs retain and automatically pull primary performance artifacts.
- Existing performance inventory/pull/prune support already included
  `.cpu.prof` and `.cpu.txt`; focused lifecycle tests remained green.
- Verification: 459 neighboring remote/dashboard/scan tests passed; full CLI
  suite 879 passed; `cxr-dev lint` and `cxr-dev typecheck` passed; generated CLI
  reference and frozen contract both passed their `--check` commands.

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

## Next slice: rawkernel/physics verification + MoSe2 stall

Not started. Two parts of the same Active TODO item. Editing/dispatch
environment for this slice is an Intel-dGPU laptop with no CUDA device and no
locally-visible SYCL device — Part A's CUDA compile/compare/golden/A-B-timing
work is not runnable here at all and must go through `cxr remote` on
`qlmc`/`ALEX-DESKTOP` per `AGENTS.md`; only the non-NVIDIA-fallback spot check
below is local.

### A. Required GPU verification (gates enabling Round 3)

`docs/compute-performance-optimization.md` Round 3 landed the fused line
prologue (`line_prologue_jit_kernel.py`) behind `_USE_JIT_LINE_PROLOGUE =
False` in `spectrum.py`, plus the brem launch-config change and the streamed
reduction kernels — all still gated because no CUDA device was available
while implementing. No test references `_USE_JIT_LINE_PROLOGUE` or
`line_prologue` today (`rg` confirms zero hits under `tests/`). Round 3's own
"Required GPU verification" list is the checklist:

- [ ] Compile each JIT specialization on the minimum and current supported
      CuPy versions.
- [ ] Compare the prologue's `E_r`, `aw`, and nonzero `w` against the eager
      path on one-block synthetic inputs, edge-bracketing inputs, and full
      MoS2 cases.
- [ ] Run the line golden suite and repeated-run determinism checks; record
      max absolute error, max error/peak, significant-bin relative error, and
      integral drift.
- [ ] Re-run the interleaved burn-in A/B harness at `Ne=450`, 2000, and 10000
      on both `ALEX-DESKTOP` and `qlmc`.
- [ ] Re-measure transport/GPU overlap before touching process counts or
      starting a `prange`/CUDA transport project.
- [x] Confirm the non-NVIDIA (CPU/NumPy, no CuPy) fallback path still runs
      and is exercised by CI — `_USE_JIT_LINE_PROLOGUE` and the raw kernels
      are CUDA/CuPy-only branches; the eager/CPU path must be untouched and
      covered. Spot-checked 2026-08-08 on an Intel-dGPU laptop (no CUDA, no
      visible SYCL device in this dev environment — `dpctl.get_devices()`
      returns `[]`): `tests/integration/test_intel_sycl_backend.py` +
      `test_backend_selection.py` are 11 passed / 1 skipped, i.e. backend
      auto-probe already falls through past CUDA/SYCL to CPU cleanly. This
      only confirms general backend fallback machinery, not the new prologue
      kernel specifically (it has no dedicated test either way).
- [ ] If any of the above passes clean, flip `_USE_JIT_LINE_PROLOGUE = True`
      as its own reviewed change with a `Validation:` id and ledger row
      (Round 2 measured non-bit-for-bit `fma` contraction on the earlier
      gather-EK prototype; confirm Round 3's fixed-order kernel's actual
      bit-for-bit/tolerance status here, not by assumption).

### B. MoSe2 / large `--ne-line` transport stall (TODO.md 2026-08-07)

Fresh remote-box symptom report, not yet reproduced or diagnosed:
`--ne-line=20_000` on `MoSe2` (and possibly other heavy materials) —
transport runs tens of seconds with GPU at 0%, CPU inconsistent (often
<10%, sometimes 60-70%), GPU bursts to 100% instantaneously then drops to
0%, host RAM 50-80%, VRAM 20-50%. Pattern reads as transport-side
(CPU/Numba) serialization or scheduling stall feeding the GPU, not a GPU
compute or memory-pressure problem — but unconfirmed.

- [ ] Reproduce with `cxr remote run` at matched `--ne-line=20_000` on
      `MoSe2`, `--perf` telemetry on, per `docs/performance-profile-analysis.md`.
- [ ] Compare against a lighter material/`--ne-line` at the same profile to
      isolate whether the stall scales with `n_seg` (as Round 2's linear
      transport/lines/brem split predicts) or is a distinct discontinuity at
      high segment counts (e.g., OOM-retry/chunk-halving churn, or `njit`
      compilation/cache-eviction stalls under `numba` parallel dispatch).
      `Sweep.spec_chunk`/`Sweep.brem_chunk` and the adaptive chunk/OOM-retry
      path in `runner.py` are the first suspects for the observed GPU
      100%-then-0% burstiness.
- [ ] Attribute the "GPU 0% for 10s+" window to a specific phase (transport
      `njit` compile/dispatch, checkpoint I/O, host-side chunk sizing) using
      NVTX ranges per the Round 1/2 method, not by inference from utilization
      alone.
- [ ] Decide fix vs. document as expected (compute-bound transport at very
      high `Ne`) once attributed; do not change chunk defaults without a
      controlled A/B per the "Broader optimization path" above.

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

**Next slice (rawkernel/physics verification + MoSe2 stall) delegation:**
`lead-task` — crosses `montecarlo/` JIT kernels, physics validation, and
remote profiling. Required: `physics-review`, `physics-validation`,
`performance`, `remote-gpu-jobs`, `regression-testing`,
`monte-carlo`/`scientific-library`. Part A touching `_USE_JIT_LINE_PROLOGUE`
needs a `Validation:` id and ledger row per `AGENTS.md`; do not flip the flag
without it.

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
