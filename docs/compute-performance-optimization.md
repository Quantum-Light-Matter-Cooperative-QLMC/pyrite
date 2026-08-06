# Compute performance optimization

Broad-strokes summary of the GPU/throughput work. Method, the levers found,
what landed, and what is left. Written in rounds; each round records its own
hardware, workload, and numbers.

- [Round 1 (2026-07-31)](#round-1-2026-07-31) — `xp.fuse` line-chain fusion and
  incremental checkpoint shards.
- [Round 2 (2026-08-06)](#round-2-post-rawkernel-re-profile-2026-08-06) —
  re-profile after the `cupyx.jit.rawkernel` reductions landed; the eager
  prologue is now the constraint.
- [Round 3 (2026-08-06)](#round-3-fused-line-prologue-implementation-2026-08-06)
  — fused deterministic line prologue, combined interpolation gather, and the
  measured brem launch adjustment.

## Review verdict

The central diagnosis is correct: after the line and brem reduction kernels
landed, the next material GPU lever is the incoherent line prologue, not more
Numba around transport and not CuPy's reduction autotuner. The checkpoint-shard
design and the decision to keep transport on the CPU for now are also sound.

Three corrections were made during implementation:

1. `__fmaf_rn` does **not** reproduce a non-contracted
   `f0 + frac * (f1 - f0)` sequence; it deliberately performs one rounding
   after the multiply-add. Preserving the former blend requires separate
   `__fsub_rn`, `__fmul_rn`, and `__fadd_rn` operations.
2. Atomic survivor compaction makes output order scheduler-dependent. Since the
   downstream float32 reduction is order-sensitive, the implementation keeps a
   deterministic `(segment, g)` order and marks rejected pairs with zero
   weights; the reduction skips those slots before evaluating `sin`.
3. A 3x line-only speedup does not imply six transport workers. Using the
   measured line+brem totals gives roughly two workers at `Ne=2000` and three at
   `Ne=450`, before headroom. Re-profile the completed kernel before setting a
   new default.

## Round 1 (2026-07-31)

Work on `feature/compute-performance-optimization`. Numbers are from `nsys`
profiles on the lab box (`qlmc`, RTX-class single GPU) over the `hopg` test
profile (189 cases) unless noted; see `performance-profiles/hopg_test-*/`.

### Method

Profiling drove every change — no speculative optimization. Each candidate was
isolated with `nsys` NVTX ranges (`cxr.lines`, `cxr.brem`,
`cxr.transport.line/brem`, `cxr.lines.tab/accum`) plus `cuLaunchKernel` counts
from `nsys-stats.txt`, then verified bit-for-bit or tolerance-bounded against
the golden suite before landing. See `docs/performance-profile-analysis.md` for
the analysis playbook. Guiding rule: **GPU utilization is evidence, not the
target — compute-weighted wall time is.**

### Where the time was going

Two dominant, unrelated levers surfaced on the `hopg` profile:

1. **Launch-bound spectrum kernels.** The batched incoherent line path fired a
   storm of tiny elementwise CuPy kernels — `cupy_multiply`, `cupy_add`,
   `take`, `where`, `searchsorted`, `clip` — ~82k `cuLaunchKernel` calls total.
   At single-digit GPU utilization the wall was dispatch-bound, not
   compute-bound: the GPU was starved, not busy.
2. **O(N²) checkpoint I/O.** `run_sweep` re-serialized the *entire growing*
   per-material store to `line.pkl`/`brem.pkl` on **every** finished config, so
   total bytes written over a sweep grew quadratically — ~16 s of a ~28 s wall
   (~55%), all serial CPU I/O with the GPU idle.

`cxr.brem` is genuinely compute-bound and was deliberately left alone.

### What landed

#### 1. Line-chain kernel fusion (`montecarlo/spectrum.py`)

Collapsed the batched incoherent line launch-storm into a handful of fused
kernels, all **bit-for-bit** with the prior op-by-op path:

- `_line_kin_core` (`xp.fuse`) — resonance frequency + photon kinematics
  (~15 elementwise launches → 1). Exact op order preserved (`x*x`, not `x**2`).
- **Shared interpolation index** — `chi`/`U` (real+imag) and `mu` all sample the
  same `E_res` on the same `E_tab_g` grid, so the bracket is computed **once**
  and each table is gathered off the shared index. Removes 4× redundant
  `searchsorted`/`clip`.
- `_line_weight_core` (`xp.fuse`) — escape factor + PXR prefactor with the
  absorption term folded in.

**Impact:** `cuLaunchKernel` 82,457 → 63,765 (−23%); `cxr.lines` 4.43 s →
3.61 s on `hopg`. Bigger on high-reflection materials (e.g. `mos2`, where
`cxr.lines` was ~18 s). `cxr.brem` unchanged, as expected. No new validation
debt — pure kernel-merge, existing `Validation: line-hkl-batch` marker intact.

This builds on the earlier GPU line-absorption tabulation fix (already on
`main`), which had already cut `cxr.lines` from ~70 s to ~8 s by ending a
per-hkl CPU `xraydb` sync stall.

#### 2. Incremental checkpoint shards (`run.py`, `_checkpoint_store.py`)

Records are immutable once a config completes, so there is no reason to rewrite
the whole store each time. Each finished config now writes exactly one
crash-safety **shard** (`<stem>/parts/<hash>.pkl`) — O(1) per config, O(N) per
sweep. At the end of the sweep (clean finish *or* budget stop) the shards are
folded **once** into the authoritative `<stem>/{line,brem}.pkl` monolith and the
`parts/` directory is cleared, so `slim`/`prune`/`archive`/`remote` and the
analysis app still see the unchanged on-disk contract.

Crash-safety improves rather than regresses: a hard kill leaves shards behind,
and `load` unions them over any stale monolith (shards win per `name`/`E0`).
Legacy `<stem>.pkl` callers keep the old whole-file path.

**Impact:** per-sweep checkpoint write goes from O(N²) whole-store re-pickle to
O(N) shards + one consolidation — an estimated 16 s → ~1–2 s on `hopg`
(pending an `nsys` rerun on the box to confirm).

### Verification

- Line fusion: 122/122 line goldens (chunk-invariance, montecarlo, mosaic,
  multilayer, coherent, faceting).
- Checkpoint: `test_run`/`profiles`/`prune`/`archive`/`slim`/`analyze`/`remote`
  green, incl. new shard-clear-on-consolidate, unconsolidated-shard recovery,
  and shard-over-stale-monolith override tests.
- `lint` + `typecheck` clean on both.

### Remaining levers (not yet taken)

- **`cxr.lines.tab`** — per-hkl `chi`/`U` `f1`/`f2` tabulation (~5.5 s on
  `hopg`). Next target, but carries the `f1` edge cusp + coherent goldens, so
  trickier than the absorption (`f2`/`mu`-only) fix.
- **Transport aggregate** — large in wall-clock terms but runs pipelined across
  the CPU worker pool; watch the memory-aware worker cap on high-beam-energy
  refine steps (OOM risk on the single-GPU box).
- **`cxr.brem`** — compute-bound; leave unless the kernel math itself changes.

Round 2 supersedes the first two of these: `cxr.lines.tab` was fixed by the
stacking prologue (host→device uploads hoisted out of the reflection loop), and
the transport/GPU balance is now quantified.

## Round 2: post-rawkernel re-profile (2026-08-06)

Investigation only; nothing landed. Re-profile of the incoherent line + brem
path after the `cupyx.jit.rawkernel` reductions
(`spectrum_jit_kernel.py`, `brem_jit_kernel.py`, `coherent_jit_kernel.py`)
became the default (`_USE_JIT_*_REDUCTION = True`). Question asked: is there
more to win from (a) more raw/elementwise/reduction kernels or wider fusion,
(b) `cupyx.optimizing.optimize`, (c) more Numba JIT or a GPU transport port?

Answer: (a) yes, ~3× on the line phase, but the target has moved — the
rawkernels are no longer the cost, the **eager-CuPy prologue that feeds them
is**. (b) and (c) are dead ends, with numbers below.

### Environment and workload

Local box `ALEX-DESKTOP` (not `qlmc`): RTX 3060 Ti 8 GiB, driver 610.47,
24 cores, 22 GiB RAM, `nsys` 2026.4.1. Single case from the `compute_test`
catalog profile, `mos2`, `MoS2 1mm pol=45 az=120 footprint=5x5mm`:
22 reflections × 5 mosaic-MC nodes (`N_g = 110`), 1018 line bins, 4001 brem
bins, incoherent path, `REAL = float32`, no layers, finite 5×5 mm footprint.

Method: `_transport_case` once, then repeated `_lines_for_segments` /
`_brem_wide_from_segments` on the same segments with explicit
`Device().synchronize()` around each call. **The 3060 Ti idles at 210 MHz SM
and takes seconds to reach boost**, which alone produced ±25% spread; every
timing below is preceded by a 4 s GPU burn-in, reported as min and median of
7–9 reps, with A/B arms interleaved across separate processes.

### Where the time goes now

Phase split, warm, seconds per case:

| `Ne` | `n_seg` | transport (1 core) | lines | brem |
| --- | --- | --- | --- | --- |
| 450 | 1.07 M | 0.35 | 0.200 | 0.055 |
| 2000 | 4.50 M | 1.05 | 0.79 | 0.27 |
| 10000 | 22.6 M | 5.46 | 4.08 | 1.50 |

All three scale linearly in `n_seg`, and single-core transport holds at
≈0.98 × the whole GPU phase at every `Ne` — the ratio is `Ne`-independent, so
there is no electron count at which the balance flips on its own.

`nsys` at `Ne = 450` (3 reps of lines + brem): **3703 `cuLaunchKernel` calls per
lines call**, 166 ms/rep GPU-busy against 270 ms/rep wall. Kernel time share:

| Kernel | Share | What it is |
| --- | --- | --- |
| `_kernel_2e` (brem) | 24.3% | brem reduction rawkernel |
| `_kernel_3e` (lines) | 22.1% | line reduction rawkernel |
| `cupy_prepare_array_indexing` | 11.1% | fancy indexing in `_interp_gather2d` |
| `cupy_add` | 6.5% | `A2`/`A2_pxr`/`A2_cbs` accumulation |
| `cupy_take` | 6.0% | fancy indexing in `_interp_gather2d` |
| `cupy_multiply` | 4.7% | polarization dot products |
| `cupy_subtract__int64` | 3.9% | `idx - 1` interp bookkeeping |
| `_line_amp_sq_core` | 3.9% | fused amplitude core |
| `cupy_where` | 3.3% | interp endpoint clamps |
| `cupy_subtract__float32` | 3.0% | interp blend |

**The two rawkernels are only 46% of GPU-busy; ~50% is the eager per-seg-block
prologue in `mc_spectrum`'s batched incoherent branch** — interp gathers,
polarization dots, and boolean-mask compaction, none of which existed as a
target when round 1 was written.

### Confirmed lever: fuse the prologue

Prototype (in-memory source patch, not landed): replace `_interp_gather2d` /
`_interp_gather1d` with one `cupy.ElementwiseKernel` doing flat-index
gather+blend, eliminating all fancy indexing. Three interleaved A/B runs at
`Ne = 2000`, min of 7 reps:

| Arm | Run 1 | Run 2 | Run 3 |
| --- | --- | --- | --- |
| base | 0.837 s | 0.826 s | 0.805 s |
| gather EK | 0.642 s | 0.607 s | 0.596 s |

**−25% on the line phase**, reproducible. Not bit-for-bit: 4.8e-9 relative,
from `nvcc` contracting `a + b*c` into an `fma`. Either force separate
`__fsub_rn`/`__fmul_rn`/`__fadd_rn` operations to retain the former rounding,
or carry it as the same class of debt as
`Validation: line-amplitude-fusion` (ledger row + golden regen).

Re-profiled with the gather EK in place (`Ne = 2000`, per rep, burn-in
excluded): the line reduction rawkernel is 158 ms of 352 ms GPU-busy (45%),
and the remaining 194 ms is still prologue — `cupy_add` 30 ms,
`_line_amp_sq_core` 26 ms, `cupy_multiply` 18 ms, residual `_interp_index`
bookkeeping (`searchsorted` + `clip` + `take` + int64 subtract) ~35 ms, mask
compaction (`scan_naive` + `getitem_mask` + `bitwise_and` + `greater` +
`isfinite`) ~30 ms.

**Recommended next change:** one `cupyx.jit.rawkernel` covering the *whole*
prologue — per `(segment, g)` pair compute kinematics → interpolation → both
polarizations → weight and feed the existing `E_r`/`aw`/`w` reduction interface.
The original prototype proposed atomic survivor compaction; Round 3 instead
uses deterministic fixed-order buffers with zero-weight rejection. That
collapses the eager prologue to one launch per segment block and removes its
full-size intermediates without introducing run-to-run ordering noise.
Extrapolating the measured split, the earlier target was 0.83 s → 0.25–0.30 s
at `Ne = 2000` (≈3×), but that remains a hypothesis until the implemented
fixed-order variant is timed on the profiling hosts.

### Measured non-levers

Each of these was prototyped and timed; none is worth taking.

- **`seg_block` sizing.** The path is memory-traffic-bound, not launch-bound,
  so the obvious "batch harder" knob goes the wrong way. Against a base median
  of 0.827 s, a 16 M-element budget costs **+27%** (1.05 s) and 250 k costs
  **+23%** (1.01 s). The current `max(1, 1_000_000 // N_g)` sits at the
  optimum — leave it alone.
- **Dropping the `components=False` PXR/CBS dead work.** `_line_amp_sq_core`
  always computes and returns `a2_pxr`/`a2_cbs`, and the caller always
  accumulates them. A `components`-gated variant measured 0.806/0.804 s against
  a base of 0.795/0.914 s — inside the noise. A naive skip-the-accumulate patch
  was 13% *slower* (memory-pool reuse). Not worth the branch.
- **`cupyx.optimizing.optimize`.** It only tunes CuPy's own
  `ReductionKernel`-backed routines, and this path has almost none: `cub_any`
  0.1%, `argmin` 0.2%, `min` 0.2%, `bsum_shfl` ~0.2% — **under 1% of GPU-busy
  in total**, so the ceiling is ~0.5%. It also needs `optuna`, which is not a
  dependency. Skip.
- **More Numba JIT.** `cProfile` of `_transport_case` at `Ne = 2000`: 0.911 s of
  0.912 s inside `simulate_trajectories`, 645 Python calls for the whole phase.
  Coverage is already complete; there is no interpreter overhead left to
  remove. The only levers are `prange` or a GPU port, below.

### Marginal: rawkernel launch configs

12-point sweep of `nthreads × energies_per_block` at `Ne = 2000`, min of 5:

- Lines (`SpectrumKernelConfig(512, 3)` today): best 256/3 at 0.717 s vs 0.747 s
  — ~4%, at the noise floor. Not worth touching.
- Brem (`BremKernelConfig(256, 2)` today): best 512/3 at 0.219 s vs 0.244 s —
  ~10% of the brem phase, ≈2% of a case. Real but small.

Caveat: changing `nthreads` changes the shared-memory reduction tree and is
**not** bit-for-bit; `energies_per_block` alone is. If only one is taken, take
brem `energies_per_block` 2 → 3.

### GPU transport: not yet

Not warranted now, and the reason is the pipeline rather than the kernel.
Single-core transport is ≈1× the GPU phase per case, so the `gpu-pipeline`
engine hides it completely at two or more transport workers; the
`_PIPELINE_WORKER_MEM_MB` split (transport-only workers no longer charged the
full-case `_WORKER_MEM_MB` budget) is what makes that reliably reachable. It
becomes worth revisiting only when either

1. the prologue rawkernel lands and a fresh end-to-end profile shows the
   available 2–3 transport workers no longer keep the GPU fed; or
2. a run is core-starved (single-core SLURM allocation, or an interactive
   single case, where transport is ~50% of wall).

If it is taken: `_transport_core_ungrooved` is already written lockstep over
electrons, so a `numba.cuda` port maps directly (one thread per electron,
lockstep loop dropped). The costs are per-electron cuRAND streams replacing the
single shared `Generator`, and an atomic `nseg` counter for segment output.
Both change segment order, so neither is bit-for-bit — ledger row plus a full
golden regen. `numba.prange` over the electron loop buys the core-starved case
far more cheaply, with the identical RNG-stream problem and the caveat that it
oversubscribes against the process pool.

### Reproducing

No new tooling was added. The harness is `_transport_case` once followed by
repeated `_lines_for_segments` / `_brem_wide_from_segments` on the returned
`tp["segs"]`, with `coherent=False`. Anyone repeating this must keep the GPU
burn-in and the interleaved A/B arms — without them the clock ramp swamps every
effect reported above. Prototype variants were applied by re-`exec`-ing a
patched `spectrum.py` source into the imported module before importing
`runner`, so the working tree stayed clean.

### Suggested order

1. Gather `ElementwiseKernel` (−25% on lines, small and isolated) — use separate
   round-to-nearest arithmetic intrinsics if the existing blend order must be
   retained.
2. Full prologue rawkernel (est. ≈3× on lines) — the actual prize.
3. Brem `energies_per_block` 2 → 3 (~2% of a case, bit-for-bit).
4. Re-measure the pipeline balance; only then reconsider transport `prange` or
   a GPU port.

### Scope limits

Measured on one material, one geometry, one GPU, and the incoherent line path
only — the coherent path was under concurrent work and deliberately left out.
The prologue result is shared code and should generalize; the −25% and ≈3×
figures are for this workload and should be re-measured on `qlmc` before being
quoted as production numbers.

## Round 3: fused line prologue implementation (2026-08-06)

Implementation pass following the Round 2 review. No CUDA device was available
in the editing environment, so this round records code structure and static
verification only. Performance and GPU numerical validation remain release
gates; none of the Round 2 estimates are promoted to measured results here.

### What changed

#### 1. Deterministic full line prologue (`line_prologue_jit_kernel.py`)

The new float32 CUDA JIT kernel assigns one thread to each `(segment, g)` pair
and evaluates, in one pass:

- resonance and photon kinematics;
- the shared energy bracket plus chi/U/mu interpolation;
- both polarization amplitudes;
- Beer-Lambert attenuation, finite-time width, and mosaic-weighted line weight.

It emits three fixed-order arrays (`E_r`, `aw`, `w`). Rejected pairs retain
their deterministic slot with `w=0`; no atomic counter or mask compaction is
used. This costs three arrays of at most the existing one-million-pair block
budget, but eliminates the roughly forty eager temporaries and preserves stable
pair ordering. The path is staged behind `_USE_JIT_LINE_PROLOGUE = False` until
the GPU release gates below pass. When opted in, it is further gated to the
existing fast-path domain: CuPy, float32, incoherent, `components=False`,
`sinc_cutoff=None`, and the single-slab absorber branch (finite transverse
footprint remains supported). Every other case stays on the prior
implementation.

#### 2. Reduction kernel streaming (`spectrum_jit_kernel.py`)

The line reduction kernels now load the weight first and skip zero-weight pairs
before loading `E_r`/`aw` or evaluating `sin`. They can accumulate directly
into a caller-provided output, avoiding one extra CuPy add launch per segment
block. Fresh calls still allocate and return a zero-initialized spectrum, so
the previous API remains valid. The impossible 2048-thread launch option was
removed; CUDA blocks are capped at 1024 threads.

#### 3. One-launch interpolation fallback (`spectrum.py`)

The eager batched path now gathers chi/U real+imag and mu in one
`ElementwiseKernel`. Separate round-to-nearest subtract, multiply, and add
intrinsics retain the pre-existing interpolation operation order. This path is
still useful for the coherent batch branch, which deliberately does not use the
new incoherent prologue.

#### 4. Brem launch setting (`brem_jit_kernel.py`)

`DEFAULT_BREM_KERNEL_CONFIG.energies_per_block` changed from 2 to 3, the
bit-for-bit launch-only improvement measured in Round 2. The thread count stays
at 256, so the reduction tree is unchanged.

#### 5. Shared-transport repair consistency (`runner.py`)

The live runner already transports `max(Ne, Ne_brem)` once and selects the line
and brem populations by electron id, so the earlier duplicate-transport concern
has been resolved. The brem-only repair helper still used the obsolete
`seed + 1` convention, however; it now uses `case["seed"]` to reproduce the
shared live transport's brem population.

### Required GPU verification

Before enabling this in a release branch:

1. Compile each JIT specialization on the minimum and current supported CuPy
   versions.
2. Compare the prologue's `E_r`, `aw`, and nonzero `w` against the eager path on
   one-block synthetic inputs, edge-bracketing inputs, and full MoS2 cases.
3. Run the line golden suite and repeated-run determinism checks. Record max
   absolute error, max error/peak, significant-bin relative error, and integral
   drift.
4. Re-run the interleaved burn-in A/B harness at `Ne=450`, 2000, and 10000 on
   both ALEX-DESKTOP and `qlmc`. Capture line wall time, kernel count, survivor
   fraction, and fixed-order zero-slot overhead.
5. Re-measure transport/GPU overlap before changing process counts or starting
   a `prange`/CUDA transport project.
