# Portable GPU backends

Branch: `feature/portable-gpu-backends`

TODO scope: support non-NVIDIA GPU architectures without regressing the current
CUDA or NumPy execution paths.

## Goal

Replace CUDA/CuPy-specific backend assumptions with a small accelerator
contract, then provide maintained paths for NVIDIA CUDA, AMD ROCm, and Intel
SYCL GPUs. Keep CPU NumPy as the universal fallback. AMD ROCm is the primary
non-NVIDIA deployment target; the available Intel Arc A370M is the practical
hardware target for proving the abstraction on a second vendor. Apple Metal is
out of scope.

“Supported” means installation, deterministic backend selection, full
line-plus-bremsstrahlung execution, memory/OOM handling, telemetry, numerical
comparison, and documented hardware verification—not merely importing an array
module. Resource admission must also keep small local devices and host RAM
usable; accelerator discovery alone must never authorize consuming nearly all
available memory.

## Current evidence

- `src/cxr_mc/montecarlo/_backend.py` probes only
  `cupy.cuda.runtime.getDeviceCount()`, exports `xp`, `cp`, `_GPU`, `REAL`, and
  `_to_cpu`, and silently falls back to NumPy on any probe failure.
- The spectrum hot path mostly uses a compact NumPy-like surface: array
  creation/conversion, indexing, reductions, `einsum`, interpolation,
  transcendental functions, sorting, and complex dtypes.
- `runner.py`, `run.py`, and `materials/attenuation.py` bypass `xp` for
  CuPy-specific array typing, allocator limits/statistics, pool release, and
  `cupy.cuda.memory.OutOfMemoryError`.
- Performance and remote paths assume NVIDIA tooling (`nvidia-smi`, NVTX,
  Nsight Systems), CUDA modules, CUDA contexts, and CuPy-named metrics.
- Packaging unconditionally installs `cupy-cuda13x[ctk]`; `dpctl` is also a
  default dependency but has no current source use.
- Current CuPy documentation describes NVIDIA CUDA and AMD ROCm support. ROCm
  uses AMD-hosted `amd-cupy` wheels rather than the repository’s CUDA wheel,
  so mutually exclusive accelerator distributions cannot remain one
  unconditional dependency.
- User has an Intel Arc A370M in this laptop but no AMD GPU. Host-level checks
  confirm the Intel stack is ready: `clinfo` sees discrete `0x5693` and
  integrated `0xa7a8`; `dpctl 0.22.1` sees both through Level Zero 1.15 and
  OpenCL 3.0. The discrete device reports 128 compute units and 3.85 GiB, the
  integrated device 64 compute units and 14.45 GiB shared memory; neither
  advertises fp64.
- Codex's normal sandbox hides the GPU device nodes, so hardware checks must run
  with explicit host-device access. The earlier empty `dpctl` result measured
  sandbox isolation, not missing drivers.
- Current `dpctl` provides device/runtime management but no `dpctl.tensor`
  module. The Intel NumPy-like candidate is the separate `dpnp` package. A
  `dpnp 0.20.0` host smoke on `level_zero:gpu:0` completed a float32 allocation
  and reduction correctly. Direct probes also found and executed the sampled
  hot-path surface, including `interp`, `einsum`, and `sinc`; full
  line/bremsstrahlung parity remains untested.

## Decisions

- Keep transport on CPU NumPy. Port only the spectrum/bremsstrahlung device hot
  path and its lifecycle plumbing.
- Preserve `run_cases(engine="auto"|"gpu"|"cpu")`: `gpu` means any supported
  accelerator, not specifically CUDA. Backend/vendor selection is separate
  from execution topology.
- Introduce one backend object/protocol owning array namespace, platform/device
  identity, availability probe, host/device conversion, device-array
  detection, OOM exceptions, allocator limit/release/statistics, precision,
  synchronization, and optional profiling ranges.
- Keep `xp`, `_GPU`, `REAL`, and `_to_cpu` compatibility exports during
  migration. Remove internal reliance on the CuPy-specific `cp` global.
- Selection must be deterministic and inspectable. Proposed surface:
  `CXR_MC_BACKEND=auto|cpu|cuda|rocm|sycl`; explicit unavailable selections
  fail clearly, while `auto` follows a documented priority and may fall back to
  CPU.
- Make accelerator stacks optional, vendor-specific extras:
  `cxr-mc[nvidia]`, `cxr-mc[amd]`, and `cxr-mc[intel]`. Base
  `cxr-mc` remains CPU-capable and installs none of the heavy accelerator
  stacks. NVIDIA owns the CUDA CuPy distribution, AMD owns the ROCm CuPy
  distribution/source configuration, and Intel owns `dpnp` plus `dpctl`.
  Document these extras as mutually exclusive within one environment unless a
  tested combination is later proven safe; never co-install conflicting CuPy
  distributions.
- Preserve existing CUDA defaults, single-device/single-context runner
  topology, chunk controls, checkpoint contents, result ordering, and public
  physics outputs.
- Rename telemetry schema fields only through an additive compatibility
  migration: retain readers for `cupy_pool_*` while emitting backend-neutral
  allocator/device fields with backend and vendor identity.
- Keep the existing lab-box CUDA/SLURM path working. Add capability hooks for
  non-NVIDIA telemetry/profilers/modules; do not invent cluster defaults for
  hardware not present.
- Hardware-specific claims require execution on that vendor’s device. Mocked
  unit tests prove dispatch and lifecycle contracts, not real support.
- Prioritize the ROCm adapter and packaging contract. Use the Arc A370M to
  validate that the abstraction is genuinely vendor-neutral through Level Zero;
  lack of AMD hardware keeps ROCm support provisional until an AMD acceptance
  run is arranged.
- Evaluate `dpnp` as the SYCL array namespace and retain `dpctl` only for
  device/queue/runtime capabilities it owns. Keep both in a SYCL-specific
  optional install rather than the CPU base.
- Treat fp64 as a backend capability. On the Arc A370M,
  `CXR_FP64=1` under automatic selection falls back to CPU NumPy; an explicitly
  selected SYCL backend fails clearly. It must not silently execute reduced
  precision while claiming a reference run.
- Call named machine-safety presets **execution resource policies**, not
  compute profiles: `SweepProfile` already owns simulation fidelity and
  performance profiles are telemetry artifacts. Proposed selector:
  `CXR_MC_RESOURCE_POLICY=auto|conservative|balanced|throughput`.
- `auto` chooses `conservative` for devices below 8 GiB, including the 3.85 GiB
  Arc A370M. Initial conservative device budget is the smaller of 50% total
  memory and total memory minus 2 GiB. If that leaves no viable chunk, do not
  attempt the allocation. `balanced` and `throughput` retain larger,
  documented budgets; `throughput` may preserve the current 85% single-process
  CuPy pool default.
- A resource policy owns device-memory budget/reserve, initial and minimum
  chunks, OOM retry count, pool-release cadence, CPU worker count, and host-RAM
  admission for fallback. Existing expert environment knobs remain supported
  as explicit overrides, with resolved values reported in the runtime plan.
- Enforce budgets in backend-neutral chunk admission before allocation.
  Adapters additionally apply allocator/pool hard limits when their runtime
  exposes them. `dpnp` documents queue/device-directed USM allocation but no
  portable CuPy-equivalent pool-fraction cap; SYCL safety therefore cannot
  depend on an allocator cap.
- CPU fallback is deliberate and observable:
  `CXR_MC_BACKEND=auto` plus `engine="auto"` may fall back before accelerator
  work for unavailable/incompatible hardware, unsupported requested precision,
  or an infeasible resource budget, and may rerun the complete spectrum/brem
  phase on NumPy after bounded catchable OOM retries. Emit one warning plus a
  structured fallback reason in telemetry/runtime diagnostics.
- Explicit intent fails instead of changing devices:
  `engine="gpu"` with no viable accelerator, or explicit
  `CXR_MC_BACKEND=cuda|rocm|sycl` with an unavailable/incompatible/over-budget
  backend, raises an actionable backend/resource error. `engine="cpu"` or
  `CXR_MC_BACKEND=cpu` always selects NumPy. Never hide per-operation
  host-array fallback inside an accelerator case.
- Do not implement or evaluate an Apple Metal adapter in this task.

## Open decisions

- Where can an AMD ROCm device be obtained for the required hardware run?
- Should delivery land the contract plus locally validated Intel support while
  ROCm remains explicitly provisional, or wait to publish non-NVIDIA support
  until both vendors pass hardware acceptance?
- Should `auto` prefer the discrete Arc over the integrated Intel GPU, and what
  stable selector exposes vendor/device/backend choice without relying on
  enumeration order?
- What measured `balanced` budget and reserve should ship after conservative
  Arc validation? The conservative formula above is the required safe default,
  not a performance conclusion.
- Should explicit backend selection remain environment-only at import time, or
  also gain a `cxr` option/config surface? Any CLI addition needs defined scope,
  precedence, help, JSON, and subprocess behavior.
- Can CI provide real accelerator runners, or must vendor validation be a
  documented periodic/manual matrix with CPU-hosted contract tests in normal
  CI?

## Owning paths

- Backend contract and adapters:
  `src/cxr_mc/montecarlo/_backend.py`, likely new
  `src/cxr_mc/montecarlo/_backends/`
- Device hot path:
  `src/cxr_mc/montecarlo/spectrum.py`,
  `src/cxr_mc/montecarlo/geometry.py`,
  `src/cxr_mc/materials/attenuation.py`
- Runner, memory, OOM, and compatibility:
  `src/cxr_mc/montecarlo/runner.py`, `src/cxr_mc/run.py`,
  `src/cxr_mc/montecarlo/__init__.py`
- Telemetry and profiling:
  `src/cxr_mc/performance_profile.py`,
  `src/cxr_mc/performance_analysis.py`
- Remote capability detection:
  `src/cxr_mc/_remote/config.py`, `src/cxr_mc/_remote/scripts.py`,
  `src/cxr_mc/_remote/viewer.py`, `src/cxr_mc/_remote/cli.py`
- Packaging and documentation:
  `pyproject.toml`, `uv.lock`, `README.md`,
  `docs/running-on-a-cluster.md`, `docs/performance-profile-analysis.md`,
  `docs/repo_map.md`
- Tests:
  `tests/test_montecarlo.py`, `tests/test_gpu_oom_retry.py`,
  `tests/test_pool_cadence.py`, `tests/test_run.py`,
  `tests/test_performance_profile.py`, `tests/test_performance_analysis.py`,
  and affected remote/output tests

## Implementation checklist

1. Freeze the backend capability contract and selection/fallback semantics.
   Inventory every `xp` operation and every direct CuPy/CUDA/NVIDIA dependency;
   build a per-candidate compatibility matrix before choosing adapters.
2. Split packaging into CPU base plus vendor-specific accelerator installs.
   Verify clean installs and lock strategy on supported OS/Python combinations;
   expose the exact extras `cxr-mc[nvidia]`, `cxr-mc[amd]`, and
   `cxr-mc[intel]`; move `dpctl` and `dpnp` into the Intel extra if its
   operation matrix passes.
3. Refactor `_backend.py` into discovery plus adapters. Keep compatibility
   exports and make probe failures observable without noisy default imports.
4. Route device-array checks, host transfers, allocator policy/statistics,
   OOM retry, synchronization, and precision through the selected backend.
   Preserve the current CUDA behavior first. Add execution-resource-policy
   resolution with legacy environment-knob compatibility and explicit
   precedence.
5. Make spectrum/geometry/attenuation use only the frozen array/capability
   surface. Add small compatibility shims only for operations missing from a
   chosen backend; avoid host transfers inside hot loops.
6. Add backend-neutral memory estimation and admission before device
   allocations. Resolve chunks against the selected device budget; enforce
   allocator caps where supported; make bounded OOM retry shrink chunks; then
   perform policy-controlled whole-phase CPU fallback or raise the documented
   error.
7. Implement the ROCm adapter as the primary non-NVIDIA target and the SYCL
   adapter against the same contract. Validate SYCL on the Arc A370M once the
   runtime exposes it; keep ROCm provisional until matching AMD hardware is
   available. Fail explicitly for unsupported operations instead of silently
   producing CPU/GPU mixtures.
8. Generalize runtime plans, diagnostics, performance telemetry, analysis
   tables, and allocator fields. Maintain backward reads for existing
   `cxr.performance.v1` CuPy fields or version the schema additively. Record the
   requested/resolved resource policy, budgets, reserves, chunk changes, OOM
   attempts, and CPU-fallback reason.
9. Generalize remote device monitoring/profiling as capability-gated paths
   while preserving current `nvidia-smi`/Nsight behavior and lab defaults.
10. Add backend contract tests using fakes plus CPU numerical reference tests.
   Add vendor-marked smoke, OOM, allocator, and representative physics parity
   tests runnable only on matching hardware. Include low-memory admission tests
   that cannot allocate beyond a fake 4 GiB device budget, and fallback/error
   matrix tests covering automatic versus explicit selections.
11. Document installation matrices, selection precedence, resource-policy and
   override precedence, fallback/failure behavior, supported operations, known
   limitations, profiler availability, and hardware validation evidence.
   Update repo map if backend modules change package structure.

## Acceptance checks

- A CPU-only base install imports and runs without CUDA/ROCm/SYCL packages.
- Clean CPU, NVIDIA, AMD, and Intel environments resolve through
  `cxr-mc`, `cxr-mc[nvidia]`, `cxr-mc[amd]`, and `cxr-mc[intel]`
  respectively. Each vendor environment installs only its selected heavy
  accelerator stack; CPU installation remains lightweight.
- CUDA installation retains existing automatic selection, single-context
  pipeline, pool limiting/release, catchable OOM retries, fp32 default,
  `CXR_FP64=1`, checkpoint identity, result ordering, and output-noise behavior.
- Explicit unavailable backend selection exits with an actionable diagnostic;
  `auto` selection and CPU fallback are deterministic and tested.
- The Arc A370M resolves `CXR_MC_RESOURCE_POLICY=auto` to `conservative`,
  admits no more than the conservative formula, reports the resolved byte
  budget, and completes a representative case without destabilizing the
  desktop. A fake 4 GiB device proves allocations above budget are rejected
  before backend allocation.
- Automatic selection falls back to a host-admitted NumPy spectrum/brem phase
  for unsupported fp64, infeasible device budget, or exhausted catchable OOM
  retries, and records why. The corresponding explicit GPU/backend selections
  raise actionable errors without beginning a hidden CPU run.
- CPU fallback respects policy-resolved host-RAM and worker limits. If host
  admission also fails, execution stops with an actionable resource error
  rather than risking system OOM.
- At least one approved AMD ROCm device and one approved Intel SYCL device
  complete a representative line-plus-bremsstrahlung case without hidden
  host-array fallback. If staged delivery is approved, the first landed slice
  names the second hardware target as an explicit unfinished gate.
- Each fp64-capable backend agrees with the NumPy float64 reference under
  existing numerical tolerances. Backends without fp64 fail or explicitly
  route `CXR_FP64=1` to CPU. Every GPU backend's float32 output has separately
  justified tolerances; chunk-invariance and finite-output checks pass.
- Each supported backend demonstrates device selection, array round-trip,
  allocator reporting/release where available, catchable OOM behavior or an
  explicit capability limitation, pre-allocation budget admission, and clean
  shutdown.
- Performance events identify backend, vendor, device, precision, device
  utilization/memory when available, and backend-neutral allocator metrics.
  Existing CuPy-profile artifacts remain analyzable.
- Remote CUDA dry runs and tests remain unchanged. Non-NVIDIA remote behavior
  either generates validated vendor commands or clearly reports unsupported
  capabilities; it never emits NVIDIA commands for another vendor.
- Focused backend/runner/spectrum/run/performance/remote suites, lint, typecheck,
  docs, and real CPU/CUDA smoke paths pass. Vendor hardware evidence records
  device/runtime versions and exact commands.

## Non-goals

- Moving stochastic electron transport to GPU.
- Multi-GPU scheduling, distributed arrays, or simultaneous mixed-vendor
  execution.
- Rewriting physics kernels solely for performance before parity is proven.
- Claiming support from import success, mocks, or CPU emulation.
- Changing profile grids, seeds, checkpoint physics identity, or scheduler
  policy unrelated to backend capability.

## Dispatch

Worker skill: `lead-task`

Required skills: `scientific-library`, `performance`, `regression-testing`,
`monte-carlo`, `remote-gpu-jobs`, `documentation-maintenance`, and
`cli-ui-ux` if a public selector is added.

Suggested slices after hardware access is approved:

1. operation/capability matrix, backend contract, packaging design;
2. compatibility-preserving CUDA refactor and CPU/CUDA regression suite;
3. AMD ROCm adapter and CPU-hosted contract tests; hardware acceptance remains
   a release gate;
4. Intel `dpnp`/SYCL adapter, explicit discrete-Arc selection, and Level Zero
   hardware validation;
5. backend-neutral telemetry, remote capability hooks, and documentation;
6. AMD hardware validation, then independent cross-vendor parity/performance
   review.

Task-local checkpoint commits are allowed after dispatch. No worker may push,
edit canonical TODO ownership, run heavy work locally, or claim untested vendor
support. Delegation is appropriate for independent vendor adapter
investigation and parity review after interfaces are frozen.

Stop on unavailable acceptance hardware, mutually incompatible dependency
resolution, a candidate backend missing required complex/interpolation/einsum
semantics, unexplained numerical drift, checkpoint identity changes, or
unrelated dirty work outside explicit paths.

## Implementation evidence (2026-07-29)

- Ported compute-performance commits `002b8e8`, `ee26601`, and `bfa1381` as
  scoped cherry-picks before backend edits. This preserves the learned
  non-increasing successful line-chunk cap, phase-specific line/brem/generic
  OOM retries and telemetry, and failed-session exclusion without merging the
  other task's divergent `TODO.md` or branch-local docs.
- Base dependencies are CPU-only. Vendor extras are `nvidia` (CUDA CuPy),
  `amd` (Python-3.13-compatible upstream CuPy ROCm source build), and `intel`
  (`dpnp` + `dpctl`). AMD's `amd-cupy 13.5.1` index wheel was rejected because
  it exposes only `cp310`, while cxr-mc requires Python 3.13 or newer.
- Intel host validation selected Level Zero `gpu:0`,
  `Intel(R) Graphics [0x5693]`, 4,128,911,360 bytes, 128 compute units, no
  fp64. `auto` resolved `conservative` with a 1,981,427,712-byte device budget
  and 2,147,483,648-byte reserve.
- Tiny Arc checks passed both finite line and bremsstrahlung chunk-invariance
  kernels, followed by one end-to-end 1-electron line-plus-bremsstrahlung
  `run_case`; all returned spectra were finite. This is functional evidence,
  not a performance conclusion.
- AMD hardware validation remains an explicit unfinished acceptance gate. No
  ROCm support claim or cross-vendor performance claim is made without an
  approved AMD device.
