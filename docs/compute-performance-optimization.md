# Compute performance optimization (2026-07-31)

Broad-strokes summary of the GPU/throughput work on
`feature/compute-performance-optimization`. Method, the levers found, what
landed, and what is left. Numbers are from `nsys` profiles on the lab box
(`qlmc`, RTX-class single GPU) over the `hopg` test profile (189 cases) unless
noted; see `performance-profiles/hopg_test-*/`.

## Method

Profiling drove every change — no speculative optimization. Each candidate was
isolated with `nsys` NVTX ranges (`cxr.lines`, `cxr.brem`,
`cxr.transport.line/brem`, `cxr.lines.tab/accum`) plus `cuLaunchKernel` counts
from `nsys-stats.txt`, then verified bit-for-bit or tolerance-bounded against
the golden suite before landing. See `docs/performance-profile-analysis.md` for
the analysis playbook. Guiding rule: **GPU utilization is evidence, not the
target — compute-weighted wall time is.**

## Where the time was going

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

## What landed

### 1. Line-chain kernel fusion (`montecarlo/spectrum.py`)

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

### 2. Incremental checkpoint shards (`run.py`, `_checkpoint_store.py`)

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

## Verification

- Line fusion: 122/122 line goldens (chunk-invariance, montecarlo, mosaic,
  multilayer, coherent, faceting).
- Checkpoint: `test_run`/`profiles`/`prune`/`archive`/`slim`/`analyze`/`remote`
  green, incl. new shard-clear-on-consolidate, unconsolidated-shard recovery,
  and shard-over-stale-monolith override tests.
- `lint` + `typecheck` clean on both.

## Remaining levers (not yet taken)

- **`cxr.lines.tab`** — per-hkl `chi`/`U` `f1`/`f2` tabulation (~5.5 s on
  `hopg`). Next target, but carries the `f1` edge cusp + coherent goldens, so
  trickier than the absorption (`f2`/`mu`-only) fix.
- **Transport aggregate** — large in wall-clock terms but runs pipelined across
  the CPU worker pool; watch the memory-aware worker cap on high-beam-energy
  refine steps (OOM risk on the single-GPU box).
- **`cxr.brem`** — compute-bound; leave unless the kernel math itself changes.
