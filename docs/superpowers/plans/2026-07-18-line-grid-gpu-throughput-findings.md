# Line-grid rerun: GPU throughput findings (2026-07-18)

Follow-up to `2026-07-17-line-grid-bounds-scan-findings.md`. During the targeted
rerun (energies 150/200/250/300, grid ceiling 30000 eV) the GPU was badly
underutilized. This documents what was measured, the root cause, the fix
applied, and what remains.

## Symptom (as reported)

- Coarse phase (the overnight "980 runs"): GPU sat at a **steady 5–25%**.
- Later (refine phase): GPU **stuttered 0% ↔ 96%** — long idle, brief spikes.

## Workload / setup

- Box `qlmc` (`DESKTOP-QNIHO3D`), single GPU (~16 GB), SLURM partition `gpu`,
  Python 3.14, `uv run`.
- Script: `scripts/analyze_line_grid_bounds.py`. Per energy it runs a **coarse**
  scan (`COARSE_NE=500`, ~980 cases over near-zero tilts × azimuths + spot
  checks) then **refines** the top-5 candidates at `REFINE_NE=5000`.
- Engine: `run_cases` (`montecarlo/runner.py`). With a GPU present, CPU transport
  pipelines across a worker pool while the spectrum/brem phase runs **serially in
  the main process on a single CUDA context**.

## Measurements

Serial single-case probe (`CXR_MC_TIMING=1`, ne=1000), **CPU spectrum path**.
Note `max_workers=0` alone does not force this: in the code it only disables
the transport worker pool, and on a GPU box the serial path still runs the GPU
spectrum (`run_cases`, `montecarlo/runner.py`). That the probe ran the CPU
path is confirmed by its absolute times — ~180 s/case against ~1.2 s/case on
the live GPU pipeline below, a gap the 2× ne difference cannot explain.

| grid step | nbins | transport | spectrum (CPU, main) | wall |
| ---: | ---: | ---: | ---: | ---: |
| 5 eV | 5998 | 3.0 s | 185.3 s | 188.4 s |
| 25 eV | 1200 | 2.4 s | 176.3 s | 178.6 s |

Single-run probes (no repeat spread measured); the probe's job is the
*relative* grid-step comparison, not absolute GPU-path times. The **~14.9 GB
peak reserved CuPy pool, freed per case** under the default
`CXR_MC_FREE_EVERY=1`, is a GPU-run figure (the runner only reports the pool
peak in GPU mode) from the refine-phase regime — the CPU probe has no CuPy
pool.

Live sampling of the real GPU pipeline, coarse phase (job 212, ne=500):

- Throughput: **~1.2 s/case** steady (better than the ~3.4 s/case feared).
- GPU utilization: **steady 2–4%**, GPU memory **494 MiB**.

## Root cause — two distinct regimes

1. **Coarse phase is CPU-bound, not GPU-bound.** At ne=500 each spectrum is a
   tiny burst of GPU kernels; only 494 MiB is resident and the GPU idles waiting
   on main-process Python orchestration. The low utilization here is
   **structural** — the GPU is barely involved. This is the "steady 5–25%"
   regime (the 2–4% above was sampled on job 212, the 5–25% reported overnight
   on the earlier job — different job and sampling window, same regime).

2. **Refine-phase stutter is the pool-free cadence.** At ne=5000 the spectrum
   arrays are large enough to actually saturate the GPU in bursts. The default
   `CXR_MC_FREE_EVERY=1` frees the ~15 GB CuPy pool **after every case**, forcing
   a full device sync + realloc between cases → the 0% ↔ 96% stutter.

Grid step size is **not** a useful lever: 5998 → 1200 bins changed spectrum time
by only ~5% because spectrum cost scales with electrons/segments, not grid bins.

## Fixes applied

- **`CXR_MC_FREE_EVERY=40`** in the sbatch (the sbatch lives on `qlmc`, not in
  this repo): frees the pool every 40 cases instead of every case, removing the
  per-case device-sync that caused the refine stutter. Env-only; **no
  correctness/physics change** — it only changes *when* the allocator is freed.
  **Memory caveat:** the ~14.9 GB peak was measured *with* per-case frees; with
  frees every 40 cases the pool can grow/fragment across mixed-shape refine
  cases, and headroom on the 16 GB card is only ~1 GB. The runner's designed
  safety net for a stretched cadence is `CXR_MC_FREE_WATERMARK_MB` (frees when
  the reserved pool crosses the watermark — see the A2 comment in
  `montecarlo/runner.py`), which job 212 was submitted **without**. Set e.g.
  `CXR_MC_FREE_WATERMARK_MB=15000` alongside `FREE_EVERY=40` on future runs; if
  212 hits an out-of-memory in refine, resubmit with it (the per-energy
  checkpoint below preserves completed energies).
- **Per-energy resumable checkpointing** in `analyze_line_grid_bounds.py`: the
  `--json-out` file is written atomically after each energy, and a restart skips
  energies already present. Makes the run stop/resume-safe, so the box can be
  yielded (scancel + resume) without losing completed energies. This is the real
  enabler behind "chunking" this script — it is not a `cxr remote scan` sweep, so
  `--chunk-minutes` does not apply.

## Still open (not chased)

- **Coarse phase is structurally CPU-bound.** No env knob raises GPU utilization
  there; it dominates wall time. Making it GPU-bound needs a real code change —
  batch the spectrum across cases, or overlap transport and spectrum instead of
  running spectrum serially in the main process. Larger, separate task.
- **The `FREE_EVERY=40` fix is not yet verified.** When job 212 reaches the
  refine phase, re-sample GPU utilization to confirm the 0% ↔ 96% stutter is
  gone, and watch the reserved pool (`CXR_MC_TIMING` prints the peak) to
  confirm it stays under 16 GB without the watermark.

## Job state

- Job 211 killed (was stuck in refine with the stutter).
- Job 212 cancelled at 2026-07-18T~06:30 UTC (6h 9m wall). Its checkpoint holds one
  **complete** energy: 150 keV (raw_eV=10740.0, stop_eV=12400.0, coarse+refine done —
  confirmed by the later run's resume line "1 energ(ies) already done: [150.0]").
  Confirms the old job-207 value (9540 eV) was truncated low.
- **Merge file** (`line_grid_bounds_merged.json`, in the worktree): the three
  trustworthy rows from job 207 (30/50/100 keV, `source: job_207_trustworthy`) + the
  completed 150 keV row (`source: job_212_complete`). 200/250/300 keV remain missing.
- **2026-07-18 16:42 rerun (energies 200/250/300, `--coarse-engine cpu`,
  FREE_EVERY=40 + WATERMARK=15000) crashed after ~5.5 min** at case 199/980 of the
  200 keV coarse phase with `BrokenProcessPool`. Root cause per dmesg: the kernel
  OOM-killer killed a pool worker (global OOM on the 45 GiB box; slurmd cgroup).
  The forced-CPU engine's `ncpu*3//4 = 24` workers each hold multi-GB full-case
  state at 200 keV — the pool oversubscribes RAM. The tqdm trace shows the classic
  precursor: throughput collapsed from ~30 it/s to 20–40 s/it (swap thrash) before
  the kill. **Fix for resubmit: cap workers, e.g. `--max-workers 8`** (the CLI
  passthrough already exists), or add a memory-aware worker cap to the cpu engine.
- The `CXR_MC_FREE_EVERY=40` refine-stutter fix is still unverified (no run has
  reached a monitored refine phase since it was applied).
