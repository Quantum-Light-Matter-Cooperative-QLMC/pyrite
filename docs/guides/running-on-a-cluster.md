# Running on a cluster (SLURM)

`pyrite run [PROFILE] -m MATERIAL` is the headless entry point for one profile member. Canonical full runs write `pyrite-output/checkpoints/<material>/{line,brem,characteristic}.h5`; overridden runs (and existing `--survey` datasets) use identity-qualified directories. This makes the command a clean fit for any batch scheduler without the optional lab-box helper below. Install once, submit one job per material, then pull checkpoints back for local analysis or static-HTML export. See [Sweep fidelity and dataset identity](sweep-profiles.md) for the complete naming contract.

> The scripts below are **templates** — partition names, the CUDA module, account
> strings, and resource limits are site-specific. Adapt them to your cluster.

## 1. Install on the cluster

The project is uv-managed with a committed lockfile. On the login node:

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git
cd pyrite
uv sync                       # CPU-only base environment
# or exactly one: --extra nvidia | --extra amd | --extra intel
source .venv/bin/activate
pyrite --help                 # sanity check
```

No GPU stack is installed by default. NVIDIA nodes use `uv sync --extra nvidia` and a CUDA runtime matching `cupy-cuda13x`. Intel nodes use `uv sync --extra intel`. AMD nodes currently require a ROCm toolchain and `CUPY_INSTALL_USE_HIP=1 uv sync --extra amd`; AMD-hosted wheels do not yet support PyRITE's Python version. Keep ROCm deployment provisional until validated on the target cluster.

Set `PYRITE_MC_BACKEND` explicitly in production jobs when silently changing hardware would be wrong. Automatic selection may fall back to CPU; explicit `cuda`, `rocm`, or `sycl` errors if unavailable or over budget.

## 2. One material per job

Submit with `sbatch run_pyrite.sh mose2`:

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
source .venv/bin/activate        # or install once with `uv tool install .`

MATERIAL="${1:?usage: sbatch run_pyrite.sh <material>}"
pyrite run standard -m "$MATERIAL"
```

On an accelerator node one main-process device context handles spectrum/bremsstrahlung while a process pool prepares CPU electron transport, so `--cpus-per-task` supplies those transport workers. For a **CPU-only** partition, drop `--gres` and the CUDA module; `run_cases` uses a full-case worker pool capped by both core count and available memory. Pass `--workers $SLURM_CPUS_PER_TASK` to request the allocation's CPU count; the memory cap still applies. The two pools carry different per-worker RAM budgets: `PYRITE_MC_WORKER_MEM_MB` (default 6144) for full-case CPU workers, `PYRITE_MC_PIPELINE_WORKER_MEM_MB` (default 1536) for the transport-only workers behind a GPU. A pinned `--workers` clamped by either budget now warns.

## 3. Several materials as a job array

One array task per material — they run independently and write their own component checkpoints:

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
source .venv/bin/activate        # or install once with `uv tool install .`

MATERIALS=(mose2 wse2 mos2 hopg)            # indexed by $SLURM_ARRAY_TASK_ID
pyrite run standard -m "${MATERIALS[$SLURM_ARRAY_TASK_ID]}"
```

## 4. Retrieve and visualize locally

The checkpoints are the only output you need off the cluster:

<!-- verify: skip (rsync to a login node, not a pyrite/pyrite-dev command) -->
```bash
rsync -avz login-node:~/pyrite/pyrite-output/checkpoints/ ./pyrite-output/checkpoints/
```

Then run `pyrite app analysis launch <material>` (the `src/pyrite/apps/analysis_app.py` marimo app) or run `pyrite app analysis export` locally — all interactive visualization and static-HTML export stay on your workstation.

## Notes

- **`__main__` guard:** `pyrite run` (and the `python -m pyrite._entry.scan` shim) are properly guarded, so the `spawn` / `forkserver` transport workers are safe. Don't wrap the sweep in an unguarded `python -c "…"`.
- **`--quick`** runs a tiny smoke grid into the `<material>_quick/` component directory — use it to validate your sbatch script cheaply before submitting the full sweep.
- **fp64:** set `PYRITE_FP64=1` for double-precision reference runs (the GPU path defaults to fp32). Devices without native fp64 (e.g. Intel Arc/integrated GPUs) are refused regardless of `PYRITE_FP64`: explicit selection errors and automatic selection runs on CPU.
- **small devices:** `PYRITE_MC_RESOURCE_POLICY=auto` selects `conservative` below 8 GiB and admits chunks before allocation. Use `balanced` or `throughput` only after measuring headroom on the target node.
- **runtime pins on remote runs:** `pyrite run --remote` copies `PYRITE_MC_TRANSPORT_CORE`, `PYRITE_MC_RESOURCE_POLICY`, and `PYRITE_MC_MIN_CHUNK` from the submitting shell into the job script; other variables do not reach the job. For example, `PYRITE_MC_TRANSPORT_CORE=cuda pyrite run PROFILE --remote` forces device transport below the automatic electron-count threshold.

## Lab-box remote helper

`pyrite remote` syncs the current working tree to the configured lab host and submits every CXR compute run through SLURM. Each allocation requests one node, one task, and the configured SLURM target profile: `remote.partition` (default `gpu`), optional `remote.nodelist` (default `any`, no `--nodelist`), and `remote.gres` (default `gpu:1`). Default `--chunk-minutes 10` runs one material at a time in bounded, self-resubmitting slices. `--chunk-minutes 0` selects a monolithic `UNLIMITED` allocation; only that mode accepts `--parallel-materials`, defaults to one material, and caps concurrency at four. It uses the synced project's configured `uv` environment, not the WarpX-specific `warpx-user` Conda environment. `remote.gpu_vendor` (`PYRITE_REMOTE_GPU_VENDOR`, default `nvidia`) selects the job prelude. NVIDIA jobs start with `module purge`, then load `cuda`, `openmpi`, and `hdf5`. AMD jobs load no modules. They sync into their own `.venv-amd` with `CUPY_INSTALL_USE_HIP=1 uv sync --extra amd`, which builds CuPy against the node's ROCm toolchain (`ROCM_HOME`, default `/opt/rocm`). They also pin `PYRITE_MC_BACKEND=rocm`, so a missing HIP stack fails the job instead of falling back to CPU, and they log `rocminfo` gfx and `rocm-smi` output in the job-log header. `--nsys` needs an NVIDIA target. `intel` still fails before batch-script generation. Every compute node must see the head node's `PYRITE_REMOTE_DIR` at the same path on a shared filesystem: the batch script, job bookkeeping (`jobs/<id>/`), reservations, progress, and checkpoints all live there, and `job status`, `job logs`, and `remote pull` read them from the head node. `job status` ranks each job within the partition its own `run.sh` requested.

For example, to target the AMD node `qlmc-ace` behind the `qlmc` head node:

```bash
pyrite config set remote.target qlmc
pyrite config set remote.gpu_vendor amd
pyrite config set remote.partition gpu-amd
pyrite config set remote.nodelist qlmc-ace
pyrite config set remote.gres gpu:radeon8060s:1
pyrite run --quick -m hopg --remote --dry-run
```

Switch back with `remote.gpu_vendor nvidia`, `remote.partition gpu`, `remote.nodelist any`, and `remote.gres gpu:1`, or override one command with the matching `PYRITE_REMOTE_*` variables.

Remote configuration. There is no built-in host: set the SSH-config alias with `pyrite config set remote.target HOST` (or `PYRITE_REMOTE_HOST`, or `-R HOST` per command). The checkout directory and `uv` executable on that host default to `~/pyrite` and `~/.local/bin/uv`; `sync` creates the checkout directory if it is missing. A leading `~` is resolved to the remote login home with one cached, non-interactive `ssh` call per process, so everything downstream (scripts, `scp` paths) sees an absolute path; if that call fails the command stops and asks for absolute paths. Override with `PYRITE_REMOTE_DIR` and `PYRITE_REMOTE_UV` (absolute POSIX paths, `~/...` paths, or for `uv` a bare executable name on the remote non-interactive `PATH`). The precedence table is in [configuration resolution](../repo-design/configuration-resolution.md).

Cross-section tables. `sync` also guarantees the box holds every xsgen table it needs under `<REMOTE_DIR>/xsgen/tables`: the pinned BremsLib and ELSEPA releases plus any tables generated locally. It inventories the box with `sha256sum` in the same ssh round trip as the energy-grid artifacts and ships only missing or differing `.npz`/`.json` pairs, so a second sync sends no table bytes and a truncated or edited table is replaced. Tables found in the box's legacy `~/.local/share/pyrite/xsgen/tables` are copied into the synced tier first, without overwriting. (Locally, `pyrite tables migrate` does the same for an explicit workspace; the legacy tier stops being searched in 0.5.0.) A sync refuses, naming `pyrite tables fetch CODE --archive PATH`, when a pinned table is missing locally.

Fetched datasets. The EPICS2025 EEDL and EADL files and PyRITE's derived EPDL photon table (`pyrite tables fetch eedl`, `eadl`, `epdl`, or a bare `pyrite tables fetch` for everything) no longer ride in the code payload. The box needs no GitHub token: PyRITE's private release archives reach it only through `sync`, never by a fetch on the box. `sync` ships them once, byte for byte, to `<REMOTE_DIR>/datasets/<name>/`, in the same inventory round trip and by the same content-digest rule, so a box without internet access gets them from the laptop. A box synced before #263 still holds copies under `src/pyrite/data/characteristic_cross_sections/`; the inventory copies those into place first, without overwriting, and the digest check replaces a bad copy. A sync refuses, naming `pyrite tables fetch NAME`, when a dataset is missing or differs locally. The table-set digest, stamped as `code_tables_digest`, covers the datasets too.

SBETHE reference data. Every run of the default shell model reads SBETHE's `sdbase/` (`pdatconf.p14`), installed locally by `pyrite tables fetch sbethe`. It is upstream-only (no PyRITE release hosts it), so `sync` copies your own fetched copy, all ~600 files (18 MB), to `<REMOTE_DIR>/xsgen/reference-data/sbethe/sdbase/` by the same content-digest rule, in the same round trip; a second sync sends nothing. A box fetched by hand earlier holds it in `~/.local/share/pyrite/xsgen/reference-data/sbethe/sdbase/`; when the synced location is absent, the inventory copies that whole directory into place first, so only differing files ship. A sync refuses, naming `pyrite tables fetch sbethe`, when the local copy is missing or its `pdatconf.p14` differs from the pin. The table-set digest covers these files too.

Every job runs `pyrite tables verify --require bremslib,elsepa,sbethe-tables,eedl,eadl,epdl,sbethe` before its sweep and ends in `FAILED (tables)` when a pinned table, dataset, or `sdbase/` is absent or differs.

One shared checkout, one code identity. `sync` unpacks into a single directory on the box, so a second working tree syncing mid-run would otherwise rewrite `src/pyrite` and `checks/` underneath a job that is already running — silently mixing two revisions into one set of results. Each sync therefore identifies its payload by content (every synced file's arcname plus its content hash, after the CRLF→LF normalization, so a Windows and a Linux checkout of the same code agree) and records that identity in `<REMOTE_DIR>/.pyrite-sync` as the last step of a successful extraction, together with the local `git` revision, a dirty flag, a timestamp, and the source host and worktree. Submission copies those `code_` fields into each job's `jobs/<id>/meta`, so a job's code identity is fixed when it is queued, and evidence collected on the box can name a revision even though the remote checkout is an exported tree rather than a repository.

Syncs are serialized. `sync` holds an exclusive `flock` on `<REMOTE_DIR>/.pyrite-sync.lock` (through one long-lived ssh session, so it works for any transfer transport and is released if the client dies) from the live-job check through the stamp; a second sync waits up to five minutes, then fails with a clear message. Each upload goes to its own `/tmp/pyrite_code.<digest>.<id>.tgz`, which only that sync extracts and removes. The tarball is unpacked into a staging directory first, so a truncated or corrupt upload fails before the live tree or `.pyrite-sync` is touched; only then are stale `.py` files removed and the staged tree copied over. A job's batch script takes the same lock shared, briefly, before its payload, so it never starts importing mid-sync.

Code transfer uses rsync when this machine and the box both have it, so a sync after a one-file edit sends about that file instead of the whole tree. Each sync path (`src/`, `checks/`, `vendor/`, and the top-level files) is mirrored on its own with `--checksum --delete --delay-updates`, so a file deleted locally is deleted on the box too, and updated files are renamed into place only at the end of each transfer. `--delete` only ever targets one sync path, or `external-catalog/`, so `pyrite-output/checkpoints/`, `jobs/`, `.venv`, `.pyrite-sync` and the xsgen tables are never in its reach. The box's own `__pycache__` survives inside live directories. Energy-grid artifacts and the legacy pre-#263 dataset copies stay on the content-delta path above. The tar+scp path described above is the fallback, used on native Windows, when either end lacks rsync, and when a synced text file has CRLF line endings (rsync cannot normalize them). `PYRITE_SYNC_TRANSPORT=tar` forces it; `PYRITE_SYNC_TRANSPORT=rsync` fails instead of falling back. The payload digest and stamp are the same for both transports. `pyrite remote sync -v` prints the chosen transport (and, for tar, why), plus rsync `--stats` for each transfer.


A sync then refuses, before transferring anything, when a live job recorded a different payload identity — naming the job, its recorded digest and revision, and the incoming one. Syncing the *same* payload is silent, so starting a second material during a long run keeps working. Jobs queued before code stamping record no identity; they are reported as unverifiable rather than treated as a conflict. Override with `pyrite remote sync --force`, which proceeds and warns, naming every live job whose remaining steps may now import a different revision than they started with:

```bash
pyrite remote sync --force
```

Review the exact batch script and `sbatch --parsable` submission command without contacting the lab box:

```bash
pyrite run standard -m hopg --remote --dry-run
```

`pyrite run standard -m hopg --remote` syncs, submits, follows the SLURM job, and pulls the checkpoint. Add `--detach` to return after submission. The hidden compatibility command retains advanced `--chunk-minutes` / `--parallel-materials` controls during migration; use concurrent materials only for workloads measured to fit. Use `pyrite job status`, `pyrite job logs --follow`, or `pyrite job attach` to monitor the allocation. Attached status shows an independent case-progress bar for each material; `logs --follow` shows the raw shared job log. While pending, status ranks the target among all pending jobs in the configured SLURM partition by scheduler priority (descending, then numeric job ID) and shows the current queue leader and reason. This is a consideration-order snapshot, not a start-time promise: priority can change and backfill can run a lower-ranked job first.

Progress timing uses additive active worker-process seconds persisted in each atomic progress record. Cached cases reduce remaining work but do not inflate measured throughput; chunk queue time and pauses do not count as compute time. Elapsed compute is always shown when valid. ETA and estimated total remain `—` until at least one new unit of work supplies a finite rate. Cost-weighted work is preferred when available; parallel-material process seconds are folded back to approximate wall time using the submitted parallelism.

Use the developer performance command to enable resource sampling for the selected profile:

```bash
pyrite-dev perf sub_100keV --remote
```

Each five-second NDJSON sample records host and process-tree CPU/RAM, CPU affinity/frequency/iowait, swap activity, GPU utilization, clocks, performance state, and VRAM (`null` when no NVIDIA GPU is available), power, temperature, resolved beam parameters, active driver phase/case, most recently completed case, in-flight work, progress, worker topology, electron counts, grid widths, adaptive spectrum/brem chunk sizes, and child-process max/mean RSS. Rolling counters include CPU transport, spectrum, GPU feed-wait, checkpoint time, GPU OOM retries, and CuPy pool used/reserved/peak memory. These phase counters are enabled by `pyrite-dev perf`; `PYRITE_MC_TIMING` is not required. Logs go to `pyrite-output/performance/NAME/<material>.ndjson`; remote logs appear beside case progress in attached status. Fetch every remote job matching the catalog profile name with:

```bash
pyrite remote performance pull sub_100keV
```

Pulled files land under `pyrite-output/performance/NAME/<job>/<material>.ndjson`. Use the [performance-profile analysis playbook](performance-profile-analysis.md) to validate sessions, derive phase/resource metrics, classify bottlenecks, and design controlled tuning runs.

For a bursty, spectrum-dominated GPU run, add `--nsys` to a single-material, single-repetition performance submit. This runs an uncached job-local session under Nsight Systems and writes CUDA/NVTX/Python-stack trace artifacts beside the NDJSON. `pyrite remote performance pull NAME` fetches the `.nsys-rep`, `.sqlite`, and `.nsys-stats.txt` files too; see the playbook's Nsight section for the exact command and interpretation limits.

`pyrite job stop ...` cancels an active allocation with `scancel`. `pyrite run --preset zhai --remote` follows the same submit-and-wait workflow for the Zhai reproduction; add `--detach` to return after submission. Retrieve an existing cache with `pyrite remote pull --preset zhai`.

With no job id, `pyrite job attach`, `pyrite job logs`, and `pyrite job status` resolve to the most recently active job. Profile submissions name their job after the profile (`sub_100keV`, then `sub_100keV-2` once the bare name is taken), so resubmitting a profile leaves the earlier, now-terminal jobs on the box. `pyrite job attach` warns on stderr when it defaults to a job that is no longer running, so a stale default never masks the live resubmission.

`pyrite remote prune-jobs` deletes terminal (done/failed/cancelled) job directories, previewing exact targets unless `--yes`. Scope it to one profile family with `--profile NAME` or sweep every profile with `--all`. A live chain is never removed, so it is safe to prune old runs of a profile while a fresh submission of the same profile is still going:

```bash
pyrite remote prune-jobs --profile sub_100keV        # preview
pyrite remote prune-jobs --profile sub_100keV --yes  # delete
```
