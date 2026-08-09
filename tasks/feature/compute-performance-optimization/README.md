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

**Status 2026-08-08: worked. Part A 5/6 items closed (item 4 partial),
Part B 4/4 closed.** Headline outcomes:

- `_USE_JIT_LINE_PROLOGUE` **stays `False`** — not because the numerics
  failed (they passed comfortably) but because item 4's `ALEX-DESKTOP` arm is
  unreachable, the measured win is ~1-3% of case wall time, and the line path
  already carries four unledgered `Validation:` markers.
- Round 3's fixed-order prologue is **not bit-for-bit** — confirmed by
  measurement, mechanism identified (FMA contraction the eager gather kernel
  explicitly blocks). Divergence is confined to `w`; `E_r` and `aw` are
  bit-for-bit and the keep-mask agrees exactly.
- The MoSe2 "stall" is **not a stall**: it is the pipeline-fill transient plus
  compute-bound transport, scaling linearly in `n_seg`. Decision is
  document-as-expected; no chunk or worker default changes.
- Transport/GPU balance has **flipped since Round 2** — transport is now
  2.4-3.4x the GPU phase on `qlmc` (RTX 5080) vs ~0.98x on `ALEX-DESKTOP`
  (3060 Ti). Round 2's "two transport workers hide transport" no longer holds.

All measurements below are on `qlmc` unless stated. Harness scripts were run
from the remote scratch tree and removed afterwards; the box was returned to
its original catalog (the temporary `stall_repro` profile was deleted).

Two parts of the same Active TODO item. Editing/dispatch
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
      as its own reviewed change with a `Validation:` id and ledger row.
      **NOT FLIPPED. `_USE_JIT_LINE_PROLOGUE` remains `False`.** Three
      independent reasons, in order of weight:

      1. **The checklist is not clean: item 4 is only half done.** It requires
         the interleaved A/B on *both* `ALEX-DESKTOP` and `qlmc`.
         `ALEX-DESKTOP` is unreachable from this environment, so the second
         arm does not exist. The gate says flip only if everything passes.
      2. **The measured benefit does not justify the cost.** The prologue wins
         4.2-8.9% of the *line phase*. Item 5 measured transport at 2.4-3.4x
         the whole GPU phase on `qlmc`, so that is roughly **1-3% of case
         wall time** — and the 6-case steady-state run above spent 30-34% of
         its time waiting on transport, not on lines. Against that, flipping
         permanently accepts a non-bit-for-bit change to `w` (max 7.97e-6
         relative, 124 ULP) plus a `Validation:` id, a ledger row, and golden
         regeneration.
      3. **It would be the fifth unbacked physics claim on this code path.** A
         ledger audit of `src/cxr_mc/montecarlo/` found the incoherent line
         path already carries four `Validation:` markers with **no ledger
         row**: `line-hkl-batch`, `line-amplitude-fusion`,
         `line-gemv-elementwise`, and `line-absorption-tabulation` (the last
         self-describes as "a Physics-METHOD change, not a reassociation").
         No ledger row's `code` column names *any* of the five
         `*_jit_kernel.py` modules, and those modules carry zero `Validation:`
         markers. Adding a fifth claim on top of four unsettled ones deepens
         debt that `AGENTS.md`'s physics governance exists to prevent. Those
         four should land before a new one is stacked on them.

      **The numerics themselves are not the blocker** — they passed
      comfortably (items 2 and 3: exact keep-mask agreement, bit-for-bit `E_r`
      and `aw`, significant-bin error ~170x inside the repo's own accepted
      cross-path bound, and bitwise run-to-run determinism). Round 2's `fma`
      question is now answered by measurement rather than assumption: **Round
      3's fixed-order kernel is NOT bit-for-bit**, and the mechanism is
      identified — the eager gather kernel blocks FMA contraction with
      explicit `__fadd_rn`/`__fmul_rn`/`__fsub_rn` intrinsics, the prologue's
      `_interp_row`/`_interp_shared` do not.

      For whoever picks this up:
      - Matching the eager kernel's rounding discipline (the `__f*_rn`
        intrinsics, or compiling the prologue with `-fmad=false`) would remove
        the `w` divergence, but **would not make the end-to-end spectrum
        bit-for-bit**: the prologue feeds the reduction kernel all
        `n_seg * N_g` fixed-order pairs while the eager path feeds compacted
        survivors batched to `_JIT_LINE_BATCH_TARGET`, so each reduction
        thread accumulates a different set in a different order. Tolerance,
        not bit-for-bit, is the right frame for this change.
      - Proposed identity if it is taken: `Validation: line-prologue-fusion`,
        anchored on
        `montecarlo/line_prologue_jit_kernel.py::run_line_prologue_kernel`,
        status `filtered`, same class as `coherent-line-hkl-batch` (an
        evaluation-order claim, no new equation), with the item 2/3 numbers
        above as its `checks` column. It needs a real regression test too:
        **no test references `_USE_JIT_LINE_PROLOGUE` or the prologue module
        today.**

### B. MoSe2 / large `--ne-line` transport stall (TODO.md 2026-08-07)

Fresh remote-box symptom report, not yet reproduced or diagnosed:
`--ne-line=20_000` on `MoSe2` (and possibly other heavy materials) —
transport runs tens of seconds with GPU at 0%, CPU inconsistent (often
<10%, sometimes 60-70%), GPU bursts to 100% instantaneously then drops to
0%, host RAM 50-80%, VRAM 20-50%. Pattern reads as transport-side
(CPU/Numba) serialization or scheduling stall feeding the GPU, not a GPU
compute or memory-pressure problem — but unconfirmed.

- [x] Reproduce with `cxr remote run` at matched `--ne-line=20_000` on
      `MoSe2`, `--perf` telemetry on, per `docs/performance-profile-analysis.md`.
      **REPRODUCED (2026-08-08, `qlmc`).**

      Two CLI corrections found while setting this up, both of which make the
      checklist text as written unrunnable:
      - `cxr remote run` no longer exists. `cxr remote` now exposes only
        `gc | performance | prune-jobs | pull | rm | sync`; runs go through
        `cxr run PROFILE -R/--remote`. `docs/performance-profile-analysis.md`
        still documents the old `cxr remote run ...` form throughout, and its
        example profile `compute_test_300keV` is not in the catalog either.
      - `--ne-line` is not a `cxr run` flag. It is a **catalog edit** on
        `cxr profile set PROFILE -l/--ne-line N`, which rewrites that
        profile's `n_electrons` grid. So "running with `--ne-line=20_000`"
        means the profile's electron count was changed and then run.

      Reproduction used a scratch profile created **on `qlmc` only** (so the
      git worktree keeps a clean `data/materials.toml`):
      `cxr profile create stall_repro --from promising_low_ne` then
      `cxr profile set stall_repro --thickness 50000 --energy 30 --polar 45
      --azimuth 180 -l 20000 --material mose2`, then
      `cxr run stall_repro -m mose2 -p -i 1 --checkpoint-dir /tmp/stall_ckpt`.
      One case, MoSe2, 5 um, 30 keV, tilt 45 deg, azimuth 180 deg,
      22 reflections, `emission = both`.

      GPU utilization sampled independently at 0.5 s alongside the run:
      ```
      0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 24 66 24 100 100 100 100 100 100 100 41 0
      ```
      **~8.5 s of GPU at 0-1%, then a burst to 100% for ~3.5 s, then 0** —
      the reported pattern, from a single case. Host CPU sat at 3.4-3.9% of
      the 32-core box during the idle window while `process_cpu_percent` was
      ~100%, i.e. exactly one core busy. That is the reported "CPU inconsistent,
      often <10%". VRAM peaked at 18.6% and host RAM at 16.8% for one case;
      the reporter's 50-80% RAM is consistent with more cases in flight
      (`transport_prefetch_count = 4`) and/or several materials.
      With the full `promising_low_ne` grid this one-case pattern repeats
      **450 times per material** (5 thicknesses x 3 energies x 5 polars x
      6 azimuths), which is what makes it read as a sustained stall.
- [x] Compare against a lighter material/`--ne-line` at the same profile to
      isolate whether the stall scales with `n_seg` or is a distinct
      discontinuity at high segment counts. **RESULT: purely linear in
      `n_seg`. No discontinuity. Not material-specific.** See the item-5 table
      above for the full phase scan; the relevant reductions:

      MoSe2 across a 40x span of `Ne` (500 -> 20000, `n_seg` 369 k -> 14.66 M):
      from `Ne=2000` to `Ne=20000` the segment count grows **10.07x** while
      transport grows **8.34x**, lines **9.12x**, and brem **11.5x**. The
      transport/GPU ratio moves smoothly and *monotonically* from 3.22 to
      2.45 with no step anywhere. Per-segment transport cost **falls**
      monotonically (0.392 -> 0.215 us/seg) and flattens — that is fixed
      `njit` compile cost amortizing over more segments, which is the exact
      opposite signature of a compilation or cache-eviction stall that would
      grow with segment count.

      Cross-material at matched `Ne=20000`: per-segment transport cost is
      **0.2226 us for hopg vs 0.2150 us for MoSe2 — within 3.5% of each
      other.** MoSe2 is "heavy" only because it produces **733 segments per
      electron vs hopg's 114** (14.66 M vs 2.29 M segments at the same
      electron count). There is no material-specific slow path; there is only
      more work. Note hopg's transport/GPU ratio is *worse* (5.96 vs 2.45)
      because it has far fewer reflections, so its GPU phase is tiny.

      The named first suspects are **ruled out by direct telemetry** on the
      reproduction run: `gpu_oom_retry_count_total = 0`,
      `line_gpu_oom_retries = 0`, `brem_gpu_oom_retries = 0`,
      `generic_gpu_oom_retries = 0`, and
      `effective_spec_chunk == attempted_spec_chunk == 100000` — no chunk
      halving occurred at all, so `Sweep.spec_chunk`/`Sweep.brem_chunk` and
      the adaptive chunk/OOM-retry path are not involved. VRAM peaked at
      18.6%, nowhere near the pressure an OOM-retry loop implies (and
      consistent with the reporter's own 20-50% VRAM observation).
- [x] Attribute the "GPU 0% for 10s+" window to a specific phase (transport
      `njit` compile/dispatch, checkpoint I/O, host-side chunk sizing) using
      NVTX ranges per the Round 1/2 method, not by inference from utilization
      alone. **ATTRIBUTED TO TRANSPORT — but not via NVTX, because the NVTX
      ranges the method assumes do not exist.**

      `docs/compute-performance-optimization.md` (Round 1 method section)
      refers to `cxr.transport.line` / `cxr.transport.brem` ranges. **Those
      ranges are not in the tree** — `transport.py` contains zero
      `_nsys_push`/`_nsys_range` calls, and there is no NVTX range around the
      transport wait or the worker result transfer either. Every existing
      range (`cxr.spectrum_case:*`, `cxr.lines*`, `cxr.brem`,
      `cxr.interpolate`) is on the GPU side, in `runner.py` and `spectrum.py`.
      An `nsys` capture would therefore show the GPU-idle window only as an
      *unlabelled gap*, which cannot discriminate the three candidate phases —
      so NVTX alone could not have answered this item. That doc claim should
      be corrected.

      Used instead the shipped `cxr.performance.v1` activity labels, which
      *do* carry per-tick phase identity (`on_activity` emits `transport_wait`
      before the blocking `fut.result()` and `spectrum` after it). One-second
      telemetry on the reproduction run:

      | t (s) | phase | GPU % | box CPU % | process CPU % |
      | --- | --- | --- | --- | --- |
      | 1.04 | `transport_wait` | 1 | 0.0 | 1.0 |
      | 2.09 | `transport_wait` | 0 | 3.5 | 101.8 |
      | 3.13 | `transport_wait` | 0 | 3.7 | 98.9 |
      | 4.16 | `transport_wait` | 0 | 3.5 | 100.7 |
      | 5.20 | `transport_wait` | 0 | 3.8 | 99.9 |
      | 6.25 | `transport_wait` | 0 | 3.9 | 123.9 |
      | 7.30 | `spectrum` | 1 | 4.1 | 110.9 |
      | 8.34 | `spectrum` | 66 | 3.4 | 100.3 |
      | 9.40-12.51 | `spectrum` | 100 | ~3.5 | ~100 |

      Terminal counters: `gpu_feed_wait_fraction = 0.5297`,
      `driver_wait_seconds_total = 6.981 s`,
      `transport_seconds_total = 3.454 s`,
      `spectrum_seconds_total = 6.198 s`,
      **`checkpoint_seconds_total = 0.0043 s`**, `engine = gpu-pipeline`,
      `effective_workers = 2`.

      Discriminating the three candidates the item names:
      - **transport: OWNS the window.** Every GPU-0% tick is labelled
        `transport_wait`, with exactly one core saturated.
      - **checkpoint I/O: ruled out.** 4.3 ms total, 0.03% of a 13.3 s run.
      - **host-side chunk sizing: ruled out.** Chunk never changed and zero
        OOM retries fired (see the item above).
      - **`njit` compile: ruled out as the *repeating* cause.** It is a
        one-time cost, and the per-segment scan shows it amortizing away
        (0.392 -> 0.215 us/seg) rather than growing.

      One measured detail worth carrying forward: `driver_wait_seconds_total`
      (6.98 s) is **twice** `transport_seconds_total` (3.45 s). The driver
      blocked ~3.5 s longer than transport actually computed, which is the
      cost of returning the full per-case segment payload from the worker
      process to the driver (the segment dict is pickled through the pool
      pipe). So the idle window is roughly half transport compute and half
      transport result transfer — the transfer half is invisible to
      `transport_seconds_total` and to any transport-side NVTX range that
      might be added.
- [x] Decide fix vs. document as expected (compute-bound transport at very
      high `Ne`) once attributed. **DECISION: DOCUMENT AS EXPECTED. No code
      change, no default change.**

      The single-case reproduction overstates the problem because one case
      has nothing to overlap with. Steady-state measurement, 6 MoSe2 cases at
      `ne_line = 20000` (2 polars x 3 azimuths, same 5 um / 30 keV):

      | configuration | wall | `gpu_feed_wait_fraction` | GPU ticks at 0% | driver wait s | transport s | spectrum s |
      | --- | --- | --- | --- | --- | --- | --- |
      | 1 case | 13.3 s | **0.530** | 17 / 29 | 6.98 | 3.45 | 6.20 |
      | 6 cases, default workers (resolved 6) | 52.1 s | **0.344** | 17 / 48 | 17.86 | 23.06 | 33.99 |
      | 6 cases, `--workers 6` | 49.9 s | **0.299** | 11 / 45 | 14.82 | 23.51 | 34.73 |

      Two things settle it:
      - **The zeros are one contiguous block at the start of the run, not a
        recurring stall.** The 6-case GPU timeline is
        `1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 15 100 100 54 100 100 100 100 100 56 ...`
        — 16 consecutive zero ticks while the pipeline fills, then the GPU
        holds 100% with only brief inter-case dips. Per-case wall time falls
        **22.6 s -> 6.7 s** over the six cases (the progress bar's own
        `22.58s/it -> 6.73s/it`). The long GPU-0% window is the pipeline-fill
        transient: the first case's transport must finish before any GPU work
        can exist, and at `Ne = 20000` that first transport is ~8-20 s.
      - **Feed-wait falls monotonically as cases become available to overlap**
        (0.530 -> 0.344 -> 0.299), and across 6 cases
        `transport_seconds_total` (23.06 s) now *exceeds*
        `driver_wait_seconds_total` (17.86 s), i.e. transport is genuinely
        running concurrently with GPU work. The `gpu-pipeline` engine is
        behaving as designed.

      Why "fix" is not warranted:
      - Scaling is linear in `n_seg` with no discontinuity (item 2), so there
        is no pathology to remove — only more work.
      - Every suspected mechanism is measurably absent: zero OOM retries,
        unchanged effective chunk, 4-42 ms checkpoint I/O, VRAM at 18.6%,
        engine never demoted to serial.
      - The default worker count already resolves to 6 here; forcing
        `--workers 6` explicitly changed wall time only 52.1 s -> 49.9 s
        (~4%), so there is no default worth changing. Per this task's own
        non-goals and the "Broader optimization path" above, chunk/worker
        defaults must not be changed without a controlled A/B, and the
        measurement says such an A/B has nothing to win here.

      Follow-ups worth opening separately (NOT done in this slice):
      1. **Add NVTX ranges to `transport.py`.** `docs/compute-performance-optimization.md`
         already claims `cxr.transport.line` / `cxr.transport.brem` exist;
         they do not. Adding them would turn this multi-run diagnosis into a
         single `nsys` capture.
      2. **Correct `docs/performance-profile-analysis.md`**: it documents the
         removed `cxr remote run ...` form and a `compute_test_300keV` profile
         that is not in the catalog.
      3. **Transport result transfer.** `driver_wait_seconds_total` was 2x
         `transport_seconds_total` on the single case, i.e. roughly half the
         idle window is returning the per-case segment payload from the worker
         to the driver rather than computing it. Reducing that (payload dtype
         or shared memory) is a real lever, but it is a separate task with its
         own A/B and its own correctness surface.
      4. Optionally document the expected fill transient in the user-facing
         performance guide so "GPU at 0% for the first ~10-20 s at high
         `--ne-line`" is not re-reported as a bug.

## Next slice: GPU transport JIT module (started 2026-08-08)

Opened on the trigger the perf doc's "GPU transport: not yet" section named:
Part A item 5 measured transport at 2.4-3.4x the GPU phase for the TMDs and
6-12x for hopg, compute-bound and linear in `n_seg`. Design and rationale live
in `docs/gpu-transport-rawkernel.md`.

Landed (CPU-verified only; no GPU was available in the authoring session):

- [x] `transport.py`: counter-based per-electron RNG (SplitMix64 keyed by
      `(seed, electron)`, indexed by draw counter). Pure integer arithmetic
      plus one exact `uint64 -> double` conversion, so host and device agree
      bit-for-bit. Explicit numba signatures are load-bearing — without them
      numba unifies to int64, every `>>` becomes an arithmetic shift, and draws
      silently lose their top bit (caught by moments, not by range checks).
- [x] `transport.py`: `_transport_core_ungrooved_perelectron`, the executable
      specification the CUDA kernel ports. Run-to-completion per electron,
      slot addressing `i * cap + s`, no atomics.
- [x] `transport.py`: `_run_per_electron_transport` — batching, measured-capacity
      replay, mask compaction. Shared by both cores so the two cannot drift.
- [x] `transport_jit_kernel.py`: the `cupyx.jit` port. fp64, textually
      identical arithmetic, `while`-only control flow (no reliance on
      transpiler `break`/`continue`), flattened arrays per house style.
- [x] `simulate_trajectories(transport_core=...)`, default `"lockstep"`.
      Default path is unchanged and still bit-for-bit — asserted by test.
- [x] 34 CPU tests + 10 CUDA-gated tests in
      `tests/montecarlo/test_transport_per_electron.py`. Core suite: 1113
      passed, 51 skipped.

Measured on CPU: per-electron core vs lockstep core agree on backscatter
fraction, transmit fraction, segments/electron, mean segment length, mean
segment energy, and mean depth across 8 seeds at Ne=3000 (all within 4 sigma;
backscatter specifically 0.15367 +- 0.00156 vs 0.15429 +- 0.00140 over 12 seeds,
Welch p=0.77). Results are invariant to batch size and to segment capacity
including forced replays at cap=4.

Not bit-for-bit, by construction and unavoidably:

- Per-electron vs lockstep: differently-ordered streams, so different samples of
  the same distribution.
- CPU vs CUDA per trajectory: CUDA's `log`/`exp`/`pow`/`sin`/`cos` are a few ulp
  from the host libm and transport is chaotic. Even the first segment passes
  through `log`. Verifiable claims are identical RNG streams, identical control
  flow/addressing, first-step agreement at `rtol=1e-12`, and aggregate agreement.

Therefore: enabling either new core changes numerical output and needs a
`Validation:` id, a physics ledger row, and a golden regen before the default
moves. Not done, and the default was not moved.

Open — needs a GPU session, in this order:

- [x] Compile each specialization on min and current pinned CuPy. **Done
      2026-08-08** on qlmc (RTX 5080, driver 610.47, CuPy 14.1.1 — the
      `cupy-cuda13x` floor and the current pin are the same version, so one run
      covers both). One real defect found and fixed: `e = e_start + i` mixed the
      int32 launcher scalar with the uint32 `blockIdx`/`threadIdx` product, and
      `cupyx.jit` in CUDA mode promotes that to uint32 and then refuses the
      `same_kind` cast of the signed operand (`TypeError: Cannot cast from
      'int32' to uint32`). Fixed by making the thread index signed once at its
      source. Every listed transpiler assumption held: `**` (ast.Pow),
      `xp.abs`/`xp.log10` on device scalars, int16 narrowing on store, and
      zero-size `internal_bounds` passed as an argument all transpile.
- [x] Run the 10 CUDA-gated tests. **Done 2026-08-08**: 37 passed in 54.81s on
      qlmc, all 10 CUDA tests included — determinism, launch-geometry
      independence at nthreads 32/128/512, capacity replay at cap 4/256/4096,
      first-step agreement with the CPU reference at `rtol=1e-12`, and aggregate
      agreement. The two slow tests are CPU-side: the 8-seed lockstep comparison
      (31.15s) and the CUDA-vs-CPU aggregate (14.61s).
- [x] `Ne` sweep of transport wall time, GPU core vs CPU core. **Done
      2026-08-08** on qlmc (RTX 5080, CuPy 14.1.1, NumPy 2.4.6, numba 0.65.1,
      Python 3.14.6). Workload: 30 keV into a 1e6 Ang stopping slab, catalog
      compositions from `build_cases(material_sweep(m))[0]`, `E_cut=5 keV`,
      seed 1, 3 repeats, medians below, compile excluded by a warm-up call.
      hopg runs 354 segments/electron and MoSe2 880 in this slab (not the 114 /
      733 quoted from the earlier profile, which was a thinner, transmitting
      case — the ratio between the two materials is what carries over).

      | material | Ne | lockstep | per-electron CPU | cuda | cuda vs lockstep |
      | --- | --- | --- | --- | --- | --- |
      | hopg | 250 | 0.020 s | 0.022 s | 0.018 s | 1.11x |
      | hopg | 1000 | 0.096 s | 0.091 s | 0.041 s | 2.32x |
      | hopg | 4000 | 0.404 s | 0.963 s | 0.099 s | 4.10x |
      | hopg | 16000 | 2.387 s | 4.416 s | 0.151 s | 15.86x |
      | mose2 | 250 | 0.039 s | 0.082 s | 0.086 s | 0.46x |
      | mose2 | 1000 | 0.163 s | 0.347 s | 0.102 s | 1.60x |
      | mose2 | 4000 | 0.766 s | 2.838 s | 0.187 s | 4.10x |
      | mose2 | 16000 | 6.154 s | 11.459 s | 1.269 s | 4.85x |

      Three findings:

      1. **Crossover is Ne ~ 500-1000 for both materials**, so launch overhead
         stops mattering well below production `Ne`. The 450-electron default
         sits right at it; `--ne-line=20000` is far past it.
      2. **The CPU per-electron core is a regression, 0.27-1.05x of lockstep.**
         The restructuring costs on CPU (counter-addressed SplitMix64 per draw
         instead of batched `Generator` draws, and worse locality); the win is
         entirely the device. The deferred `numba.prange` idea therefore starts
         from a ~2x deficit and needs more than 2 cores just to break even.
      3. **MoSe2's 4.85x at Ne=16000 is a capacity defect, not divergence** —
         see the item below. Note also that lockstep's own ns/segment degrades
         with `Ne` (226 -> 421 for hopg, 180 -> 433 for MoSe2), so part of the
         15.86x is the CPU baseline getting worse, not the GPU getting better.

- [x] **Fix the first-batch capacity gamble.** **Done 2026-08-08.** `cap` started
      at `config.seg_capacity` = 512 and was carried
      across batches, but MoSe2's per-electron tail exceeds it, so batch 0 —
      12787 of the 16000 electrons at that capacity — runs to completion,
      overflows, and is replayed at cap=2048 with a batch of 3196. Roughly 80%
      of the run is thrown away, once. Measured at Ne=16000, MoSe2:

      | cap | budget | batch | median | ns/seg |
      | --- | --- | --- | --- | --- |
      | 512 (default) | 512 MB | 12787 | 1.981 s | 140.9 |
      | 1024 | 512 MB | 6393 | 1.451 s | 103.2 |
      | 2048 | 512 MB | 3196 | 0.927 s | 65.9 |
      | 2048 | 4 GB | 25575 | 0.863 s | 61.3 |
      | 4096 | 512 MB | 1598 | 1.368 s | 97.3 |

      Raising the static default is the wrong fix: `cap` and `batch` trade off
      against a fixed byte budget, so cap=4096 starves the GPU, and hopg's own
      optimum is 1024 (0.153 s) against MoSe2's 2048.

      Two changes, both resting on the cores reporting the true `seg_count` even
      for an electron that overflowed. **(1)** The replay used to jump to
      `max(needed, 4 * cap)`; `needed` is exact, so one replay at `needed` always
      suffices and the `4x` only over-provisioned. `capacity_growth` is replaced
      by `capacity_headroom` (1.25) and `cap` is reset after *every* batch from
      the running maximum, in both directions. **(2)** `probe_electrons` (1024)
      shortens the first batch — the only one still sized by the guess. Its
      segments are kept, so it is just a short batch.

      Measured at Ne=16000, 3 repeats, medians; `full`/`probed` both start from
      the unchanged default 512:

      | material | arm | median | launches | electrons | settled cap |
      | --- | --- | --- | --- | --- | --- |
      | hopg | old default | 0.139 s | 2 | 16000 | 512 |
      | hopg | tuned (1024) | 0.129 s | 2 | 16000 | 544 |
      | hopg | full | 0.146 s | 2 | 16000 | 560 |
      | hopg | probed | 0.150 s | 3 | 16000 | 560 |
      | mose2 | old default | 0.727 s | 7 | 28787 | 2048 |
      | mose2 | tuned (2048) | 0.467 s | 5 | 16000 | 1593 |
      | mose2 | full | 0.491 s | 5 | 28787 | 1593 |
      | mose2 | probed | 0.499 s | 6 | 17024 | 1593 |

      MoSe2 −32% with no tuning, settling tighter (1593) than the hand-swept
      optimum. **The probe is worth nothing on the GPU and a lot on the CPU**: a
      GPU batch costs about one electron lifetime whatever its width (16000
      electrons fit in one wave here), so a discarded batch and a probe cost the
      same launch — the probe is a ~2% premium when the guess was already right.
      On the serial CPU core a discarded batch costs its electrons: MoSe2 at
      Ne=8000 goes 2.571 s -> 1.712 s (−33%), 9024 electrons instead of 16000.
      Kept as one shared default rather than branching the shared driver.

      7 new CPU tests: probe-invariance of the segments at four probe widths,
      the replayed batch shrinking to the probe, and `cap` settling within
      headroom of the true maximum with and without a probe. 44 pass on the GPU
      box, 1113 in the core suite.
- [x] Warp divergence and occupancy. **Done 2026-08-08.** `ncu` is unusable on
      qlmc — GPU performance counters are admin-only (`ERR_NVGPUCTRPERM`) and
      sudo needs a password; enabling them is a driver-param change on a shared
      box and was not attempted. Both quantities were obtained without counters.

      Occupancy, from the CUDA occupancy API: 92 registers/thread, 40 B local,
      register-limited to 640 of 1536 threads/SM = **41.7%**, flat for
      `nthreads` <= 128 and 33.3% above. Wall time tracks it (MoSe2 medians:
      1.08 s at 64, 1.12 s at 128, 1.27 s at 256, 1.64 s at 512), so the
      default 128 is already optimal — no change.

      Divergence, computed exactly from the per-electron segment counts (a warp
      owns 32 contiguous electrons and costs its longest-lived member):

      | material | mean | p99 | max | warp-wasted | ceiling |
      | --- | --- | --- | --- | --- | --- |
      | hopg | 353 | 406 | 448 | 11.9% | 1.14x |
      | mose2 | 879 | 1220 | 1274 | 27.4% | 1.38x |

      **Verdict: the persistent-thread work queue is not worth adding.** Perfect
      balancing recovers 27.4% of a kernel that is 24% of MoSe2's wall — ~7% end
      to end, less for hopg — and would break the index-addressed output slots
      the reproducibility guarantees rest on. Lifetimes at fixed energy in a
      stopping slab are too tightly distributed for it to pay.

- [x] **Share one device copy of the segments across a case's spectrum
      kernels.** **Done 2026-08-08.** The cheap half of the handoff measurement
      below, and it needed nothing from transport: every kernel already reached
      for its segment arrays through `xp.asarray(segments[k], dtype=REAL)`, and
      that returns a device array of the right dtype untouched. So
      `_segments_on_device` stages one copy and `_spectrum_case` hands it to all
      three kernels; the kernels are unchanged, and the callers that still pass
      host segments (the reline/repair paths) behave exactly as before. The
      `REAL` cast each kernel used to do independently now happens once, so the
      spectra are bit-for-bit identical. `_line_pair_for_case` stages too — it
      runs the same two kernels over one transport.

      hopg, Ne=16000, 5.65M segments, RTX 5080; arms interleaved after a
      discarded warm-up round so neither pays pool growth alone; medians of 5:

      | arm | three kernels | H2D | copies |
      | --- | --- | --- | --- |
      | per-kernel upload | 0.176 s | 656 MB / 0.113 s | 24 |
      | one shared copy | 0.108 s | 283 MB / 0.047 s | 16 |

      −39% on the spectrum phase. Bytes land where the accounting predicted:
      116 → 50 B/segment, i.e. the 48 B/segment union plus the two arrays the
      staging uploads whether or not a given case reads them — small enough not
      to be worth making the helper case-aware. The 16 remaining copies are
      energy tabulations, not segments.

      Gated by 9 tests in `tests/montecarlo/test_segment_staging.py`: identical
      spectra from staged vs host segments (lines both coherences, and brem),
      dtypes and scalar fields carried through, staging idempotent, layer
      masking identical, and a case staging exactly once with all three kernels
      handed that copy. Full local suite 2747 passed; on the GPU box those 9 run
      against real CuPy arrays and `tests/montecarlo tests/detectors tests/scan`
      exits 0. The remaining remote core-suite failures are all
      `FileNotFoundError` on `docs/`, `scripts/`, `notebooks/` paths that
      `cxr remote sync` does not ship.

- [ ] **Keep segments device-resident** — the remaining half, and still the
      largest item on the list. Time budget at Ne=16000, cap=2048, six
      launches (kernel/compaction from `nsys`, wall from unprofiled runs since
      `nsys` inflates host time; kernel median is identical at both capacities,
      37.8 vs 37.9 ms, which cross-checks the device numbers):

      | | hopg | mose2 |
      | --- | --- | --- |
      | wall | 0.188 s | 0.927 s |
      | transport kernel | 0.050 s (27%) | 0.224 s (24%) |
      | device compaction | 0.005 s (3%) | 0.072 s (8%) |
      | D2H payload | 0.061 s (32%) | 0.146 s (16%) |
      | host-side remainder | 0.072 s (38%) | 0.485 s (52%) |

      The kernel is no longer the bottleneck; the driver around it is. Payload
      is 82 B/segment (0.46 GB hopg, 1.15 GB MoSe2) over pageable memory at a
      measured 7.6-7.9 GB/s. The remainder is scratch allocation, mask
      construction, and NumPy output assembly — unattributed, which is what
      NVTX ranges in `transport.py` would fix.

      The spectrum side was measured too (hopg, Ne=16000, 5.65M segments,
      `REAL=float32`), by counting the bytes each kernel pushes back up:

      | phase | wall | segment H2D | share |
      | --- | --- | --- | --- |
      | transport (cuda) | 0.348 s | — (82 B/seg down) | — |
      | lines, incoherent | 0.364 s | 226 MB / 0.207 s | 57% |
      | lines, coherent | 0.088 s | 271 MB / 0.046 s | 53% |
      | brem | 0.069 s | 158 MB / 0.026 s | 38% |

      Bytes account exactly: 40 B/seg incoherent (`r_mid`, `v_hat`, `L_ang`,
      `E_keV`, `elec_id`), 48 coherent (adds `t_ang`, `t0_ang`), 28 brem. A case
      therefore pushes **116 B/segment up** on top of the **82 B/segment pulled
      down**, and `_spectrum_case` runs all three over the same `segs` dict
      without sharing a device copy — 24 uploads whose union is 48 B/segment.
      Re-upload alone is 0.279 s, 80% of a transport (~0.12 s of it bus time at
      ~6 GB/s pageable; the rest is first-touch pool growth).

      So there are two fixes, and the cheap one does not need transport at all:
      **(a)** cache one device copy per case across the three spectrum kernels —
      **done, see the item above**; **(b)** never come down, which is what is
      left here. Pinned staging is *not* the cheap partial it looked like —
      copying into a pinned buffer measured 3.8-4.0 GB/s against 6.1-6.3 GB/s
      pageable, since the staging copy costs more than the overhead it removes.
      It would only pay if the arrays were produced into pinned memory.

Deliberately deferred: keeping segments on the device end to end, NVTX ranges in
`transport.py`, `numba.prange` over the per-electron core for the core-starved
CPU case, and grooved transport.

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
