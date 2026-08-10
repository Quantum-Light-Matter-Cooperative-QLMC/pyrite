# Running on a cluster (SLURM)

`pyrite run [PROFILE] -m MATERIAL` is the headless entry point for one profile
member. Canonical full runs write `checkpoints/<material>/{line,brem}.pkl`;
survey and overridden runs use identity-qualified directories. This makes the
command a clean fit for any batch scheduler without the optional lab-box helper
below. Install once, submit one job per material, then pull checkpoints back for
local analysis or static-HTML export. See [Sweep fidelity and dataset
identity](sweep-profiles.md) for the complete naming contract.

> The scripts below are **templates** — partition names, the CUDA module, account
> strings, and resource limits are site-specific. Adapt them to your cluster.

## 1. Install on the cluster

The project is uv-managed with a committed lockfile. On the login node:

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git
cd pyrite
uv sync                       # CPU-only base environment
# or exactly one: --extra nvidia | --extra amd | --extra intel
uv run pyrite --help          # sanity check
```

No GPU stack is installed by default. NVIDIA nodes use
`uv sync --extra nvidia` and a CUDA runtime matching `cupy-cuda13x`. Intel
nodes use `uv sync --extra intel`. AMD nodes currently require a ROCm toolchain
and `CUPY_INSTALL_USE_HIP=1 uv sync --extra amd`; AMD-hosted wheels do not yet
support PyRITE's Python version. Keep ROCm deployment provisional until
validated on the target cluster.

Set `PYRITE_MC_BACKEND` explicitly in production jobs when silently changing
hardware would be wrong. Automatic selection may fall back to CPU; explicit
`cuda`, `rocm`, or `sycl` errors if unavailable or over budget.

## 2. One material per job

Submit with `sbatch run_cxr.sh mose2`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pyrite-scan
#SBATCH --partition=gpu          # <-- your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=pyrite-%x-%j.out

set -euo pipefail
module load cuda/13.x            # <-- match the cupy-cuda13x wheel (omit for CPU)
cd "$SLURM_SUBMIT_DIR"

MATERIAL="${1:?usage: sbatch run_cxr.sh <material>}"
uv run pyrite run standard -m "$MATERIAL"
uv run pyrite run standard -m "$MATERIAL" --fidelity survey
```

On an accelerator node one main-process device context handles spectrum/bremsstrahlung while
a process pool prepares CPU electron transport, so `--cpus-per-task` supplies
those transport workers. For a **CPU-only** partition, drop `--gres` and the CUDA
module; `run_cases` uses a full-case worker pool capped by both core count and
available memory. Pass `--workers $SLURM_CPUS_PER_TASK` to request the allocation's
CPU count; the memory cap still applies. The two pools carry different per-worker
RAM budgets: `PYRITE_MC_WORKER_MEM_MB` (default 6144) for full-case CPU workers,
`PYRITE_MC_PIPELINE_WORKER_MEM_MB` (default 1536) for the transport-only workers
behind a GPU. A pinned `--workers` clamped by either budget now warns.

## 3. Several materials as a job array

One array task per material — they run independently and write their own pickles:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pyrite-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-3              # <-- 0..N-1 for the N materials below
#SBATCH --output=pyrite-%x-%A_%a.out

set -euo pipefail
module load cuda/13.x
cd "$SLURM_SUBMIT_DIR"

MATERIALS=(mose2 wse2 mos2 hopg)            # indexed by $SLURM_ARRAY_TASK_ID
uv run pyrite run standard -m "${MATERIALS[$SLURM_ARRAY_TASK_ID]}"
```

## 4. Retrieve and visualize locally

The checkpoints are the only output you need off the cluster:

```bash
rsync -avz login-node:~/pyrite/checkpoints/ ./checkpoints/
```

Then run `pyrite app analysis launch <material>` (the
`src/pyrite/apps/analysis_app.py` marimo app) or
run `pyrite app analysis export` locally —
all interactive visualization and static-HTML export stay on your workstation.

## Notes

- **`__main__` guard:** `pyrite run` (and the `python -m pyrite._entry.scan` shim) are properly
  guarded, so the `spawn` / `forkserver` transport workers are safe. Don't wrap the
  sweep in an unguarded `python -c "…"`.
- **`--quick`** runs a tiny smoke grid into `<material>_quick.pkl` — use it to
  validate your sbatch script cheaply before submitting the full sweep.
- **fp64:** set `PYRITE_FP64=1` for double-precision reference runs (the GPU path
  defaults to fp32). Devices without fp64 error when selected explicitly;
  automatic selection routes the reference run to CPU.
- **small devices:** `PYRITE_MC_RESOURCE_POLICY=auto` selects `conservative` below
  8 GiB and admits chunks before allocation. Use `balanced` or `throughput`
  only after measuring headroom on the target node.

## Lab-box remote helper

`pyrite remote` syncs the current working tree to the configured lab host and
submits every CXR compute run through SLURM. Its fixed lab allocation requests
the `gpu` partition, one node, one task, and one GPU (`--gres=gpu:1`). Default
`--chunk-minutes 10` runs one material at a time in bounded, self-resubmitting
slices. `--chunk-minutes 0` selects a monolithic `UNLIMITED` allocation; only
that mode accepts `--parallel-materials`, defaults to one material,
and caps concurrency at four. This helper remains an NVIDIA lab-box path: its
generated batch job starts with `module purge`, then loads `cuda`, `openmpi`,
and `hdf5`; it uses the synced project's configured `uv`
environment, not the WarpX-specific `jrozells` Conda environment.
`PYRITE_REMOTE_GPU_VENDOR` defaults to `nvidia`; setting `amd` or `intel`
fails before batch-script generation until those lab-box module/profiler paths
are validated, so NVIDIA commands are never emitted for another vendor.

Review the exact batch script and `sbatch --parsable` submission command without
contacting the lab box:

```bash
pyrite run standard -m hopg --remote --dry-run
```

`pyrite run standard -m hopg --remote` syncs, submits, follows the SLURM job, and
pulls the checkpoint. Add `--detach` to return after submission. The hidden
compatibility command retains advanced `--chunk-minutes` / `--parallel-materials`
controls during migration; use concurrent materials only for workloads measured
to fit. Use `pyrite job status`, `pyrite job logs --follow`, or `pyrite job attach` to
monitor the allocation. Attached status shows an independent
case-progress bar for each material; `logs --follow` shows the raw shared job log.
While pending, status ranks the target among all pending jobs in the configured
SLURM partition by scheduler priority (descending, then numeric job ID) and
shows the current queue leader and reason. This is a consideration-order
snapshot, not a start-time promise: priority can change and backfill can run a
lower-ranked job first.

Progress timing uses additive active worker-process seconds persisted in each
atomic progress record. Cached cases reduce remaining work but do not inflate
measured throughput; chunk queue time and pauses do not count as compute time.
Elapsed compute is always shown when valid. ETA and estimated total remain `—`
until at least one new unit of work supplies a finite rate. Cost-weighted work
is preferred when available; parallel-material process seconds are folded back
to approximate wall time using the submitted parallelism.

Use `--perf` to enable resource sampling for the selected profile:

```bash
pyrite run sub_100keV --remote --perf
```

Each five-second NDJSON sample records host and process-tree CPU/RAM, CPU
affinity/frequency/iowait, swap activity, GPU utilization, clocks, performance
state, and VRAM (`null` when no NVIDIA GPU is available), power, temperature,
resolved beam parameters, active driver phase/case, most recently completed
case, in-flight work, progress, worker topology, electron counts, grid widths,
adaptive spectrum/brem chunk sizes, and child-process max/mean RSS. Rolling
counters include CPU transport, spectrum, GPU feed-wait, checkpoint time, GPU
OOM retries, and CuPy pool used/reserved/peak memory. These phase counters are
enabled by `--perf`; `PYRITE_MC_TIMING` is not required. Logs go to
`performance-profiles/NAME/<material>.ndjson`; remote logs appear beside case
progress in attached status. Fetch every remote job matching the catalog profile name
with:

```bash
pyrite remote performance pull sub_100keV
```

Pulled files land under `performance-profiles/NAME/<job>/<material>.ndjson`.
Use the
[performance-profile analysis playbook](performance-profile-analysis.md) to
validate sessions, derive phase/resource metrics, classify bottlenecks, and
design controlled tuning runs.

For a bursty, spectrum-dominated GPU run, add `--nsys` to a
single-material, single-repetition performance submit. This runs an uncached
job-local session under Nsight Systems and writes CUDA/NVTX/Python-stack trace
artifacts beside the NDJSON. `pyrite remote performance pull NAME` fetches the
`.nsys-rep`, `.sqlite`, and `.nsys-stats.txt` files too; see the playbook's
Nsight section for the exact command and interpretation limits.

`pyrite job stop ...` cancels an active allocation with `scancel`.
`pyrite run --preset zhai --remote` follows the same submit-and-wait workflow for
the Zhai reproduction; add `--detach` to return after submission. Retrieve an
existing cache with `pyrite remote pull --preset zhai`.

With no job id, `pyrite job attach`, `pyrite job logs`, and `pyrite job status` resolve
to the most recently active job. Profile submissions name their job after the profile
(`sub_100keV`, then `sub_100keV-2` once the bare name is taken), so
resubmitting a profile leaves the earlier, now-terminal jobs on the box.
`pyrite job attach` warns on stderr when it defaults to a job that is no longer running,
so a stale default never masks the live resubmission.

`pyrite remote prune-jobs` deletes terminal (done/failed/cancelled) job
directories, previewing exact targets unless `--yes`. Scope it to one profile
family with `--profile NAME` or sweep every profile with `--all`. A live chain
is never removed, so it is safe to prune old runs of a profile while a fresh
submission of the same profile is still going:

```bash
pyrite remote prune-jobs --profile sub_100keV        # preview
pyrite remote prune-jobs --profile sub_100keV --yes  # delete
```
