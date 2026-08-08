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

- [x] Compile each JIT specialization on the minimum and current supported
      CuPy versions. **PASS** (2026-08-08, `qlmc` = DESKTOP-QNIHO3D, RTX 5080
      16 GiB, driver 610.47, SM 12.0, Python 3.14.6, CUDA runtime 13.2,
      NVRTC 13.3, NumPy 2.4.6, Numba 0.65.1). *Minimum and current supported
      CuPy are the same version*: `pyproject.toml` pins the NVIDIA extra to
      `cupy-cuda13x[ctk]>=14.1.1`, and 14.1.1 is also the newest release on
      PyPI (release tail: 13.6.0, 14.0.0, 14.0.1, 14.1.0, 14.1.1), so there is
      exactly one supported CuPy for the CUDA path and it is the installed
      one. The `amd` extra (`cupy>=13.4.0`, ROCm) is not testable on an NVIDIA
      box and CuPy 13.x does not support CUDA 13, so it is out of scope here.
      16/16 launch paths ran clean, and a `_cached_codes` audit over every
      `_JitRawKernel` in the five kernel modules confirms **14/14 global
      kernels hold a compiled specialization, 0 uncovered** (device=True
      helpers are inlined and correctly show 0). The audit caught a real gap
      on the first pass: `coherent_stream_jit_kernel._field_kernel_1e` is
      unreachable from the production path because
      `CoherentStreamKernelConfig` pins `energies_per_block=2`; it is now
      driven explicitly. Harness: `verify_prologue.py compile`.
- [x] Compare the prologue's `E_r`, `aw`, and nonzero `w` against the eager
      path on one-block synthetic inputs, edge-bracketing inputs, and full
      MoS2 cases. **PASS with a quantified, non-bit-for-bit `w`.** Method: the
      reference is recomputed from `spectrum.py`'s *own* eager helpers
      (`_line_kin_core`, `_interp_index`, `_interp_gather_line_tables`,
      `_line_amp_sq_core`, `_line_weight_core`) assembled exactly as the
      batched incoherent branch assembles them — not a reimplementation. For
      the full-material cases the prologue is wrapped so **every production
      launch** is checked pairwise against eager on identical inputs.

      | case | pairs | kept | mask disagreements | `E_r` | `aw` | `w` max rel | `w` max ULP | `w` bitwise |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | one-block 256x1 | 256 | see note | 0 | bit-for-bit | bit-for-bit | — | — | — |
      | one-block 64x4 | 256 | 63 | 0 | bit-for-bit | bit-for-bit | 2.83e-6 | 34 | 33/63 |
      | edge-bracketing | 168 | 70 | 0 | bit-for-bit | bit-for-bit | 5.26e-7 | 8 | 56/70 |
      | full MoS2 (Ne=200) | 7,661,016 | 3,767,496 | 0 | bit-for-bit | bit-for-bit | 7.97e-6 | 124 | 3,044,265 |
      | full MoSe2 (Ne=200) | 9,260,658 | 4,567,449 | 0 | bit-for-bit | bit-for-bit | 7.82e-6 | 104 | 4,242,447 |

      Key results:
      - **Keep-mask agreement is exact everywhere** (0 kernel-only and 0
        eager-only survivors across 17.2 M pairs), including the
        edge-bracketing case that deliberately places `E_res` exactly on
        `lo_keep`, `hi_keep`, the 10 eV floor, both `E_tab` endpoints, interior
        bracket nodes, and outside both table ends. The kernel's negated
        early-return branch is equivalent to the eager boolean mask.
      - **`E_r` and `aw` are bit-for-bit identical** (0 ULP) on all 8.3 M
        compared survivors.
      - **`w` is NOT bit-for-bit**, max relative 7.97e-6 (~67 float32 eps),
        max 124 ULP. This is the Round 2 `fma` finding reproduced on Round 3's
        fixed-order kernel, and it was predicted by static audit before
        measuring: the eager gather kernel
        (`_INTERP_GATHER_LINE_TABLES_F32`) deliberately spells the blend with
        `__fadd_rn`/`__fmul_rn`/`__fsub_rn` round-to-nearest intrinsics to
        *prevent* NVCC contracting `f0 + frac*(f1-f0)` into an FMA, while
        `line_prologue_jit_kernel._interp_row`/`_interp_shared` use plain
        `cupyx.jit` arithmetic with no such guard. `_polarization_amplitude_sq`
        is likewise FMA-contractible where `_line_amp_sq_core` is not.
      - Divergence is confined to `w` exactly as that mechanism predicts:
        `E_r` and `aw` never pass through an interpolated table value.
      Harness: `verify_prologue.py numeric` / `verify_prologue.py material`.
- [x] Run the line golden suite and repeated-run determinism checks; record
      max absolute error, max error/peak, significant-bin relative error, and
      integral drift. **PASS.** Spectrum-level A/B on identical transported
      segments (flag off vs on, same case, same seed):

      | material | Ne | n_seg | bins | peak | max abs err | max err/peak | significant-bin max rel | integral drift |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | MoS2 | 200 | 116,076 | 698 | 3.352e-8 | 7.11e-15 | 2.12e-7 | 3.53e-7 | 5.00e-8 |
      | MoSe2 | 200 | 140,313 | 598 | 2.773e-8 | 3.55e-15 | 1.28e-7 | 2.95e-7 | 4.07e-8 |

      All bins are "significant" (>1e-3 of peak) in both cases. For scale, the
      repo's own accepted cross-path latitude for the batched line path is
      `BATCH_RTOL = 500 * eps_f32 = 5.96e-5`
      (`tests/montecarlo/test_coherent_emission.py`); the measured
      significant-bin error is ~170x tighter than that already-accepted bound.
      **Determinism: 3 consecutive prologue runs on identical inputs were
      bitwise identical (max repeat diff exactly 0.0)** for both materials —
      the fixed-order zero-weight design does deliver the run-to-run
      determinism it was chosen for, unlike the atomic-compaction alternative.
      Observed keep fraction is ~49% for both materials, so the zero-padded
      reduction input is only ~2x the compacted eager input, not the
      order-of-magnitude blowup a low keep fraction would have caused.

      Suite half of this item: `cxr-dev test-suite core` on `qlmc` (GPU
      backend live, so the flag is actually reached) gives **identical results
      with the flag off and on — 1075 passed, 3 failed, 42 skipped in both
      arms, the same 3 tests**. The prologue changes zero test outcomes. Those
      3 failures are `cxr remote sync` artifacts, not regressions: sync omits
      `docs/` and `scripts/`, so
      `test_refresh_retains_cached_entry_after_configured_mp_failure` cannot
      import `scripts/refresh_external_cif.py`,
      `test_bundled_crystal_validation_ids_are_ledgered` cannot open
      `docs/physics-validation-ledger.md`, and the energy-grid golden
      comparison lacks its checked-in inputs. Verified: the same three test
      files pass locally (76 passed) where those paths exist. Note the test
      tree itself is not synced by `cxr remote sync` either and had to be
      copied to `qlmc` separately to run this at all.
- [~] Re-run the interleaved burn-in A/B harness at `Ne=450`, 2000, and 10000
      on both `ALEX-DESKTOP` and `qlmc`. **PARTIAL: `qlmc` done, `ALEX-DESKTOP`
      not reachable.** `ALEX-DESKTOP` is absent from `~/.ssh/config` (which
      defines only `qlmc`) and from `known_hosts`, and this editing box is not
      it (16 cores / 30 GiB / no NVIDIA driver, vs `ALEX-DESKTOP`'s 24 cores /
      23.4 GiB / RTX 3060 Ti). There is no route to that box from this
      environment, so its arm cannot be run here and remains open.

      `qlmc` arm (2026-08-08), Round 2 method reproduced exactly: transport
      once, 4 s GPU burn-in, explicit `Device().synchronize()` around each
      timed call, 7 reps, 3 rounds with the arms interleaved across separate
      processes. Workload: `mos2`, standard catalog profile, 30 keV,
      thickness 2e4 A, tilt 45 deg, azimuth 180 deg, `N_g = 66`, 698 line
      bins, 1201 brem bins, incoherent, `REAL = float32`.

      | `Ne` | `n_seg` | base `lines_min` (3 rounds) | prologue `lines_min` (3 rounds) | best-vs-best |
      | --- | --- | --- | --- | --- |
      | 450 | 256,206 | 0.0292 / 0.0289 / 0.0293 | 0.0278 / 0.0277 / 0.0278 | **-4.2%** |
      | 2000 | 1,138,773 | 0.0763 / 0.0760 / 0.0762 | 0.0717 / 0.0717 / 0.0717 | **-5.7%** |
      | 10000 | 5,693,865 | 0.3360 / 0.3370 / 0.3353 | 0.3054 / 0.3056 / 0.3056 | **-8.9%** |

      The win is real, monotonic in `Ne`, and far outside the noise (the
      prologue arm's round-to-round spread is 0.07% at `Ne=10000` vs 0.5% for
      base), but it is **much smaller than Round 2's extrapolated 3x**. Round 2
      projected 0.83 s -> 0.25-0.30 s at `Ne=2000` from the measured phase
      split; the delivered figure is ~6% at that `Ne`. Two reasons, both
      measured here rather than assumed:
      - Round 2's extrapolation was taken on `ALEX-DESKTOP` (RTX 3060 Ti) with
        `N_g = 110`. On an RTX 5080 with `N_g = 66` the eager prologue is a
        much smaller share of a much faster GPU phase, so removing it wins
        less.
      - The fixed-order design feeds **all** `n_seg * N_g` pairs to the
        reduction kernel instead of the compacted survivors. At the measured
        ~49% keep fraction that roughly doubles the reduction kernel's input
        stream, which eats part of what the fused prologue saves.
      Brem is unaffected as expected (0.1513-0.1563 s in both arms at
      `Ne=10000`). Spectrum sums are identical within each arm across all
      three rounds (determinism) and differ between arms only in the last few
      digits (3.6586502832614e-06 base vs 3.6586501439145325e-06 prologue),
      consistent with the ~1e-7 drift recorded above.
- [x] Re-measure transport/GPU overlap before touching process counts or
      starting a `prange`/CUDA transport project. **PASS — and the balance has
      flipped since Round 2.** Single-process phase wall times on `qlmc`
      (transport once, then min-of-3 timed GPU calls on the same segments;
      same workload definition as item 4):

      | material | `Ne` | `n_seg` | transport s | lines s | brem s | GPU phase s | **transport / GPU** | transport us/seg |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | mos2 | 450 | 255,273 | 0.127 | 0.0292 | 0.0077 | 0.0370 | **3.45** | 0.499 |
      | mos2 | 2000 | 1,143,574 | 0.310 | 0.0777 | 0.0299 | 0.1076 | **2.88** | 0.271 |
      | mos2 | 10000 | 5,693,865 | 1.273 | 0.3356 | 0.1564 | 0.4920 | **2.59** | 0.224 |
      | mose2 | 500 | 369,522 | 0.145 | 0.0349 | 0.0102 | 0.0450 | **3.22** | 0.392 |
      | mose2 | 2000 | 1,455,375 | 0.378 | 0.0939 | 0.0374 | 0.1313 | **2.88** | 0.260 |
      | mose2 | 5000 | 3,640,419 | 0.867 | 0.2236 | 0.0996 | 0.3232 | **2.68** | 0.238 |
      | mose2 | 10000 | 7,332,101 | 1.671 | 0.4341 | 0.2128 | 0.6469 | **2.58** | 0.228 |
      | mose2 | 20000 | 14,662,634 | 3.153 | 0.8561 | 0.4296 | 1.2857 | **2.45** | 0.215 |
      | hopg | 500 | 56,981 | 0.088 | 0.0049 | 0.0024 | 0.0074 | **11.98** | 1.551 |
      | hopg | 2000 | 230,913 | 0.124 | 0.0070 | 0.0049 | 0.0118 | **10.51** | 0.538 |
      | hopg | 10000 | 1,127,800 | 0.290 | 0.0226 | 0.0200 | 0.0426 | **6.80** | 0.257 |
      | hopg | 20000 | 2,290,190 | 0.510 | 0.0441 | 0.0415 | 0.0856 | **5.96** | 0.223 |

      Round 2 measured single-core transport at **≈0.98x** the whole GPU phase
      on `ALEX-DESKTOP` (RTX 3060 Ti) and concluded that "the `gpu-pipeline`
      engine hides transport completely at two or more transport workers".
      **That conclusion no longer holds on `qlmc`.** On an RTX 5080 the GPU
      phase shrank while single-core transport did not, so transport is now
      **2.4-3.4x** the GPU phase for the TMDs and **6-12x** for hopg. Even
      after scaling the line phase by `110/66` to match Round 2's `N_g`, the
      mose2 `Ne=20000` ratio is still `3.153 / 1.857 = 1.70`.

      Implication for the pending decisions this item was meant to gate:
      keeping the GPU fed on `qlmc` needs roughly **3-4 concurrent transport
      workers for TMDs and 6-12 for light materials**, not the two Round 2
      implied. Conversely, further optimization of the *line* phase has
      limited end-to-end value on this box — which is the main reason the
      prologue's measured 4-9% line-phase win translates to very little
      wall-clock benefit per case. A `prange`/CUDA transport project is now
      better motivated than it was in Round 2, but process count is the
      cheaper lever and should be measured first.
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
