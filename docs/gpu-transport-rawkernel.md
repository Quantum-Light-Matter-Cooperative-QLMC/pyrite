# GPU electron transport RawKernel

## Why now

`docs/compute-performance-optimization.md` ("GPU transport: not yet") deferred
this on the grounds that single-core transport was ≈1× the GPU phase per case,
so the `gpu-pipeline` engine hid it at two or more transport workers. It named
the condition for revisiting: *the prologue rawkernel lands and a fresh
end-to-end profile shows the available transport workers no longer keep the GPU
fed.*

The Round 3 remeasurement recorded that condition. Transport is now **2.4–3.4×
the GPU phase for the TMDs and 6–12× for hopg**, per-segment cost is
material-independent and linear in `n_seg` with zero OOM retries, and the MoSe2
`--ne-line=20000` investigation attributed the long GPU-0% window to transport
compute plus transport result transfer, in roughly equal parts.

Two consequences shaped this design:

- Transport is genuinely compute-bound, not scheduling-bound, so moving the
  compute is a real lever rather than a reshuffle.
- Roughly half the idle window is *returning the segment payload* to the driver.
  Transport that already runs on the device puts the segments where the line and
  brem kernels want them, which attacks the transfer half as well. This module
  does not yet exploit that (see "Not done"), but the design leaves room for it.

## Scope

Ported: `_transport_core_ungrooved`, including the multilayer stack, the finite
`(width, height)` prism footprint, both elastic models (Browning/Mott and
analytic screened Rutherford), and per-electron energy cutoffs.

Not ported: grooved transport. `_transport_core_grooved` carries the periodic
facet exit/re-entry machinery and its own event budget; it stays on the CPU and
`simulate_trajectories` rejects the combination rather than silently degrading.

## The ordering problem, and the RNG

`_transport_core_ungrooved` is written lockstep over electrons — an outer step
loop, an inner electron loop — for one reason: a single NumPy `Generator` serves
every electron, so the draws must be consumed in a fixed interleaved order.
That order is exactly what a GPU cannot reproduce. One thread per electron runs
each electron to completion, so it consumes its stream contiguously.

The usual fixes are per-thread cuRAND states (which makes results depend on how
threads were assigned) or an atomic output counter (which makes segment order
depend on scheduling). Both would make the port unverifiable except
statistically, and both would need a full golden regen on every launch-config
change.

Instead, randomness is *addressed* rather than *streamed*. Draw `c` of electron
`e` is

```text
key_e   = splitmix64(seed + PHI * (e + 1))
u(e, c) = (splitmix64(key_e + PHI * (c + 1)) >> 11) * 2**-53
```

with `PHI = 0x9E3779B97F4A7C15` and SplitMix64's finalizer (Steele, Lea & Flood,
OOPSLA 2014), which is a bijection on 64 bits and passes BigCrush in counter
mode. Keys are built on the host so device code needs no 64-bit integer casts,
and so both cores provably address the same streams.

This is pure integer arithmetic plus one exact `uint64 -> double` conversion (the
shifted value is below `2**53`), so the generator is **bit-for-bit identical on
host and device**. It also makes a run independent of:

- CUDA launch geometry (`nthreads`);
- batch size;
- capacity replays.

`tests/montecarlo/test_transport_per_electron.py` asserts all three, plus the
generator against an independent pure-Python SplitMix64.

## Output addressing

Segment `s` of local electron `i` is written to slot `i * cap + s` — a pure
function of the electron index, with no atomics. After the kernel, a boolean
mask built from `seg_count` compacts the batch into electron-major, step-minor
order.

`seg_count[i]` records the *true* count even when it exceeds `cap`; the thread
keeps transporting but stops writing. The driver then grows `cap` and replays
the batch. Replay is exact because streams are counter-addressed, so an
overflow costs time and nothing else. This is what lets `cap` default to a
modest 512 slots instead of the `max_steps = 20000` worst case: the CPU path
gets away with `np.empty((Ne * max_steps, 3))` only because Linux never faults
in the untouched pages, and `cupy.empty` has no such luxury.

`PerElectronTransportConfig` sets `seg_capacity` and a `scratch_budget_bytes`
that bounds the dense `(batch, cap)` grid; neither changes results.

## What is and is not bit-for-bit

| Property | Status | How it is checked |
| --- | --- | --- |
| RNG stream, host vs device | bit-for-bit | integer-only; asserted against a pure-Python reference |
| Control flow, branch structure, output addressing | identical | shared algorithm, signature-drift test |
| CPU per-electron core vs CUDA kernel, per trajectory | **not** bit-for-bit | first-step agreement to `rtol=1e-12` |
| CPU per-electron core vs CUDA kernel, in aggregate | agrees | 8-seed observable comparison at 4σ |
| Per-electron core vs lockstep core | **not** bit-for-bit by construction | 8-seed observable comparison at 4σ |

The CPU/GPU trajectory difference is not a defect and cannot be engineered away.
CUDA's `log`, `exp`, `pow`, `sin` and `cos` are accurate to a few ulp but are not
the host libm's implementations, and transport is chaotic: a last-bit difference
in one scattering angle diverges over the several hundred events an electron
undergoes. The first recorded segment already passes through `log`, so even it
agrees only to rounding rather than exactly.

The per-electron core differing from the lockstep core is likewise structural —
they draw from differently-ordered streams, so they realize different samples of
the same distribution.

**Consequence for validation.** Enabling either new core changes numerical
output. That needs a `Validation:` id, a physics ledger row, and a golden regen,
per `AGENTS.md`. The default remains `transport_core="lockstep"`, which stays
bit-for-bit; nothing in this change alters it.

One deliberate measure-zero difference is documented in the code: the elastic
element selection recomputes per-element rates rather than buffering them, so
the kernel needs no per-thread local array. A recomputation landing exactly on
the sampled cumulative boundary could select a neighbouring element. The CPU
reference recomputes identically, so the two cores agree with each other; only
a buffered implementation would differ, and only at a boundary.

## Kernel structure

One thread per electron; the outer lockstep loop is gone. Control flow uses only
`while` with an explicit `running` flag, so nothing depends on `break` /
`continue` support in the `cupyx.jit` transpiler. Device helpers return a single
scalar each; `_first_prism_exit_scalar` and `_rotate_direction_scalar` return
tuples on the CPU and are therefore inlined.

Arithmetic is fp64 and textually identical to the CPU core, so any divergence is
attributable to libm alone. fp32 would roughly double throughput on consumer
cards but would change trajectories qualitatively rather than at rounding scale;
it is not offered.

Arrays are flattened to 1-D and indexed manually, matching the house style set
by `line_prologue_jit_kernel`. Per-layer element data is padded to
`(n_layers, max_elements)` with live lengths in `L_nel`.

## Expected behaviour, and what is not yet measured

The kernel compiles and its tests pass on an RTX 5080 (driver 610.47, CuPy
14.1.1, which is simultaneously the `cupy-cuda13x` floor and the current pin):
37 tests in 54.81s, including determinism, invariance to `nthreads` 32/128/512,
capacity replay at cap 4/256/4096, first-step agreement with the CPU reference
at `rtol=1e-12`, and aggregate agreement.

One porting note worth keeping. `cupyx.jit` follows C promotion in CUDA mode, so
mixing the uint32 launch indices with a signed int32 scalar yields uint32 and
then fails the `same_kind` cast of the signed operand. The index is made signed
once at its source; the alternative — carrying uint32 through the whole kernel,
as `line_prologue_jit_kernel` does — would have collided with the int32 layer
tables and int8 exit codes.

**Throughput is still unmeasured**, and divergence is the open question: threads
in a warp run until the *last* electron in that warp terminates, and electron
lifetimes vary widely — hopg's ~114 segments/electron and MoSe2's ~733 are very
different regimes. No throughput number is claimed here. Still to establish:

1. an `Ne` sweep of transport wall time against the CPU core, to find where the
   port pays off and whether launch overhead dominates at low `Ne`;
2. warp-divergence and occupancy from `nsys` / `ncu`, which decide whether a
   persistent-thread work queue is worth adding.

## Not done

- **Segments never leave the device.** The driver copies each batch's compacted
  segments back to host because `simulate_trajectories` returns NumPy. Handing
  device arrays straight to the line/brem kernels is the change that would
  recover the transfer half of the MoSe2 idle window, and it is a larger,
  separate piece of work with its own correctness surface.
- **NVTX ranges in `transport.py`.** `docs/compute-performance-optimization.md`
  already claims `cxr.transport.line` / `cxr.transport.brem` exist; they do not.
  Adding them is a prerequisite for attributing GPU transport phases in a single
  capture.
- **`numba.prange` over the per-electron core.** The same restructuring makes
  the CPU core trivially parallel, which is the cheap answer for the core-starved
  case the perf doc flagged. Not wired up here.
- **Grooved transport.**

## Reproducing the CPU-side checks

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test \
  tests/montecarlo/test_transport_per_electron.py
```

27 CPU tests; the 10 CUDA tests skip without a device.
