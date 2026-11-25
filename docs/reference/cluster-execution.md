# Cluster execution profiles

PyRITE can schedule independent simulation cases locally, as native SLURM job
arrays, or through Dask. **Execution profiles** select *where* cases run;
`Numerics` and `ExecutionConfig` select numerical methods and local kernel
settings. Changing the executor does not change the physics case identity or
random seeds.

Distributed execution is **experimental**: validate a small workload on your
cluster before relying on multi-node results or scaling estimates.

## Choose an execution mode

| Mode | Select with `--execution` | When to use it |
| --- | --- | --- |
| Normal local run | omit the flag, or `local` | One workstation/node; retains PyRITE's usual CPU/GPU scheduling |
| Native SLURM | `slurm-cpu` or a custom SLURM profile | Independent cases as job-array tasks; no Dask required |
| Local Dask | `dask-local` | Test Dask dispatch on one machine |
| Dask on SLURM | custom Dask profile with `cluster = "slurm"` | Provision packed CPU or one-GPU Dask worker jobs with `dask-jobqueue` |
| Existing Dask scheduler | custom Dask profile with `cluster = "existing"` | Join a scheduler you started, by address or scheduler file |

The separately configured `pyrite run --remote` workflow synchronizes to an
SSH host and submits jobs through that host's SLURM setup. **Do not combine
`--remote` with `--execution`.**

## Prepare the environment

On the coordinator and every worker, use compatible Python/PyRITE versions,
physics tables and material data. From a source checkout, install the locked
environment with `pixi install`; add Dask only when using its backend:

```bash
pixi install       # includes distributed and dask-jobqueue
pixi run pyrite tables verify
```

For an installed tool rather than a checkout, the Dask extra is
`uv tool install 'pyrite-mc[dask]'`. Add the appropriate accelerator extra
only where needed. Set `PYRITE_MC_BACKEND=cpu` for CPU-only experiments, or
select an explicitly supported GPU backend on an allocated GPU node.

Distributed runs require:

- A **shared POSIX filesystem at the same absolute path** for the coordinator
  and workers, with compatible locking and atomic-rename behavior. Case input,
  work-unit status and immutable output artifacts are staged there.
- SLURM CLI tools (`sbatch`, `squeue`, `scancel`) for native arrays; working
  scheduler-to-worker TCP connectivity for Dask.
- A Python executable importable on workers, available physics data and enough
  allocated RAM/VRAM for each work unit. The coordinating process must remain
  running until work finishes; on an HPC site run it as a compute-node job
  (see [Coordinator job](#coordinator-job)), never on a login node.

## Run and inspect a profile

Built-in smoke-oriented profiles need no TOML file:

```bash
pyrite run standard -m hopg                          # unchanged default
pyrite run standard -m hopg --execution dask-local
pyrite run standard -m hopg --execution slurm-cpu --execution-plan
pyrite run standard -m hopg --execution slurm-cpu
```

`--execution-plan` resolves the cases and prints *requested* CPU, GPU, memory,
work-unit and retry counts. It does not submit, reserve resources or predict
runtime. `slurm-cpu` requests only one CPU per array task by default; use a
site-specific profile for substantial runs.

Create `execution.toml` for a named cluster configuration, then select its
name with `--execution NAME --execution-config execution.toml`.

### Native SLURM array (CPU)

```toml
[execution.cpu-array]
backend = "slurm"
max_attempts = 3

[execution.cpu-array.resources]
cpu_cores = 8
memory_bytes = "32GiB"
accelerator = "cpu"

[execution.cpu-array.slurm]
partition = "cpu"                  # adapt to your site
walltime = "02:00:00"
max_concurrent = 8
cases_per_task = 2
# python_executable = "/shared/venv/bin/python"
```

```bash
pyrite run standard -m hopg --execution cpu-array \
  --execution-config execution.toml --execution-plan
pyrite run standard -m hopg --execution cpu-array \
  --execution-config execution.toml
```

For a native **GPU** array, use a separate profile with `gpu_count = 1`,
`accelerator = "cuda"` (or `"rocm"`), a GPU partition, and an appropriate
`gres`, such as `gpu:1`. One GPU is supported per work unit. SLURM and the
site's device settings must enforce real hardware isolation.

## Dask on an HPC cluster

The Dask backend moves whole work units, so it changes neither physics
identity nor numerical output. Choose one of three deployment modes with
`cluster` in the profile's `dask` table:

| `cluster` | Scheduler | Workers | Use it when |
| --- | --- | --- | --- |
| `"local"` | Coordinator process | Local processes | Testing Dask dispatch on one node |
| `"slurm"` | Coordinator process, run as a compute-node job | Worker jobs that dask-jobqueue submits with `sbatch` | Compute nodes may run `sbatch` |
| `"existing"` | You start it, or the site does | You start them (for example with `srun`) | `sbatch` is not allowed from compute nodes, or workers need a custom launcher such as dask-cuda |

A single-allocation batch runner (`SLURMRunner`, `dask-mpi`) is not supported.

### Worker packing and profile fields

With `cluster = "slurm"`, each worker **job** holds the profile's `resources`:
`cpu_cores` cores and `memory_bytes` of memory (`4GiB` when unset and the site
config gives none). A CPU job runs `cpu_cores` single-threaded Dask worker
processes. Each process runs one work unit at a time, serially, with BLAS
threads limited to one, so `workers` jobs run `workers × cpu_cores` work units
at once. dask-jobqueue divides the job's memory evenly among its processes.
`--execution-plan` reports `worker_processes_per_allocation`, and each run's
`execution.json` records the same layout under `worker_topology`.

| Field | Default | Meaning |
| --- | --- | --- |
| `cluster` | `"local"` | `"local"`, `"slurm"`, or `"existing"` |
| `workers` | `2` | `slurm`: SLURM worker jobs (the adaptive minimum); `local`: worker processes; `existing`: planning value only |
| `max_workers` | `workers` | Adaptive maximum, in the same unit as `workers` |
| `adaptive` | `false` | `slurm` only: scale between `workers` and `max_workers` jobs with load |
| `cases_per_task` | `1` | Cases per work unit |
| `max_in_flight` | 2 × worker processes | Bound on queued plus running Futures |
| `partition` | site config | SLURM partition (dask-jobqueue `queue`) |
| `walltime` | site config | Worker-job time limit, `[days-]HH:MM:SS` |
| `config_name` | `"slurm"` | dask-jobqueue config section to read site settings from |
| `jobqueue` | `{}` | `SLURMCluster` keyword arguments forwarded verbatim |
| `scheduler_address` | none | `existing` only: `tcp://host:port` |
| `scheduler_file` | none | `existing` only: path to a scheduler JSON file on shared disk |

A fixed-size SLURM cluster keeps `workers` jobs: it adapts between equal
bounds, so a job that leaves at its walltime is replaced. Profiles written
before this layout still parse. A profile without `walltime` now uses the
site config's walltime (dask-jobqueue's built-in default is 30 minutes)
instead of one hour, so set `walltime` explicitly to keep the old limit.

### Site settings

Site details stay in dask-jobqueue's own configuration, not in new PyRITE
fields. Each setting resolves in this order:

1. PyRITE fields: `resources` (cores, processes, memory), `partition`, `walltime`.
2. The profile's `jobqueue` table.
3. The dask config section `jobqueue.<config_name>` from `~/.config/dask/*.yaml`
   (user), `/etc/dask/*.yaml` (admin) or `$DASK_CONFIG`.
4. dask-jobqueue's built-in defaults.

`jobqueue` keys are `SLURMCluster` keyword names; hyphenated config spellings
such as `job-extra-directives` work too. Before anything is staged, PyRITE
checks the keys against the installed dask-jobqueue and refuses `cores`,
`processes` and `config_name`, which PyRITE sets itself. A list given in
`jobqueue` replaces the config section's list rather than extending it.

A starter `~/.config/dask/jobqueue.yaml` overrides only the keys a site needs;
dask merges it with the built-in `jobqueue.slurm` section:

```yaml
jobqueue:
  slurm:
    account: MY_ACCOUNT
    walltime: "04:00:00"
    python: /shared/pyrite/.venv/bin/python    # same path on every node
    interface: null           # e.g. ib0 if the default route is not the cluster network
    local-directory: /tmp     # node-local disk for worker spill files
    log-directory: /shared/pyrite-output/dask-logs
    death-timeout: 120
    job-extra-directives:
      - "--qos=MY_QOS"
    job-script-prologue:
      - "cd /shared/pyrite"
      - "export PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
```

dask-jobqueue reads every key of a section named by `config_name`, so build
a named section (for one of several partitions, say) from a full copy of the
`jobqueue.slurm` section in dask-jobqueue's `jobqueue.yaml`.

```toml
[execution.site-dask]
backend = "dask"
max_attempts = 2

[execution.site-dask.resources]
cpu_cores = 32
memory_bytes = "64GiB"
accelerator = "cpu"

[execution.site-dask.dask]
cluster = "slurm"
partition = "MY_PARTITION"
walltime = "08:00:00"
workers = 4
cases_per_task = 1
# config_name = "mysite"           # a named dask-jobqueue section
# [execution.site-dask.dask.jobqueue]  # or per-profile keywords
# job_extra_directives = ["--constraint=avx512"]
```

### Worker lifetime

Unless `worker_extra_args` (from `jobqueue` or the config section) already
has `--lifetime`, PyRITE adds `--lifetime` and `--lifetime-stagger` from the
worker walltime. The stagger is 5 % of walltime, at most 5 minutes. Workers
retire 10 % of walltime (between 1 and 15 minutes) plus the stagger before
SLURM's limit: an 8-hour job gets a 7 h 40 min lifetime and a 5-minute stagger.
Walltimes under about 2 minutes get no lifetime. A retiring worker hands its
running unit back to the scheduler, which runs it again elsewhere, and the
cluster submits a replacement job. Cases a worker already published are kept.

### Coordinator job

With `cluster = "slurm"` the Dask scheduler runs inside the coordinating
`pyrite run` process. Run it as a compute-node batch job, never on a login
node; its walltime bounds the whole run. If it ends early, submit the same
script again: `pyrite run` resumes, and published case artifacts are reused.

<!-- verify: skip (sbatch batch script template, submitted with sbatch) -->
```bash
#!/bin/bash
#SBATCH --job-name=pyrite-coordinator
#SBATCH --account=MY_ACCOUNT
#SBATCH --partition=MY_PARTITION
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=coordinator-%j.log
set -euo pipefail
cd /shared/pyrite                       # same absolute path on every node
export PYRITE_MC_BACKEND=cpu
pixi run --as-is pyrite run standard -m hopg \
  --execution site-dask --execution-config execution.toml
```

The coordinator needs 1-2 cores; the work runs in the worker jobs. Each
worker job counts against the same per-user job and CPU limits as the
coordinator.

### Existing scheduler

Use `cluster = "existing"` when compute nodes may not call `sbatch`, or to
start workers yourself. Start a scheduler that writes a scheduler file on
shared disk, start workers against the same file, then point the profile at it:

<!-- verify: skip (site-side dask scheduler/worker launch, not a pyrite CLI invocation) -->
```bash
# inside one allocation, e.g. sbatch --nodes=3 --ntasks-per-node=1 --cpus-per-task=32
srun --ntasks=1 --nodes=1 dask scheduler --scheduler-file /shared/dask/scheduler.json &
srun --ntasks=2 --nodes=2 dask worker --scheduler-file /shared/dask/scheduler.json \
  --nworkers "$SLURM_CPUS_PER_TASK" --nthreads 1 --local-directory /tmp &
```

```toml
[execution.manual-dask]
backend = "dask"
max_attempts = 3

[execution.manual-dask.resources]
cpu_cores = 1
accelerator = "cpu"

[execution.manual-dask.dask]
cluster = "existing"
scheduler_file = "/shared/dask/scheduler.json"   # or scheduler_address = "tcp://host:8786"
max_in_flight = 128
```

Each existing worker should run one work unit at a time per thread; start
them with `--nthreads 1`. `workers` is a planning value here, so set
`max_in_flight` to about twice the worker processes actually available.

### GPU workers

`cluster = "slurm"` provisions one-GPU worker jobs when the site's GPU
directive is in `job_extra_directives`. PyRITE never writes GRES syntax,
because sites spell it differently (`gpu:1`, `gpu:a100:1`, `--gpus-per-node`).
Each GPU job runs one Dask worker labelled `GPU=1`, and SLURM sets its
`CUDA_VISIBLE_DEVICES`:

```toml
[execution.gpu-dask.resources]
cpu_cores = 8
memory_bytes = "32GiB"
gpu_count = 1
accelerator = "cuda"        # or "rocm"

[execution.gpu-dask.dask]
cluster = "slurm"
partition = "MY_GPU_PARTITION"
walltime = "04:00:00"
workers = 2

[execution.gpu-dask.dask.jobqueue]
job_extra_directives = ["--gres=gpu:1", "--constraint=A100|H200"]
job_script_prologue = ["cd /shared/pyrite", "export PYRITE_MC_BACKEND=cuda"]
```

A GPU profile whose effective directives request no GPU (`--gres`, `--gpus*`)
fails before staging. Every GPU worker also refuses to start work without
explicit device visibility (`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
or `HIP_VISIBLE_DEVICES`); a Dask resource label alone does not isolate a
GPU. For several GPUs per node, start
[dask-cuda](https://docs.rapids.ai/api/dask-cuda/stable/) workers yourself
(`dask cuda worker --scheduler-file FILE --resources GPU=1`) and use the
existing-scheduler mode. PyRITE runs the existing single-GPU numerical path
in each GPU task, without an outer multiprocessing pool.

### Site checklist

- **Account and QoS limits:** `sacctmgr -nP show assoc user=$USER format=account,qos`,
  then `sacctmgr show qos NAME` for running-job, CPU, GPU and walltime caps.
  Keep `workers × cpu_cores` plus the coordinator within them.
- **`sbatch` from compute nodes:** from a short allocation, run
  `sbatch --test-only --wrap=true`. If refused, use the existing-scheduler mode.
- **Network:** workers must reach the scheduler port on the coordinator's
  node. Set `interface` when the default interface is not the cluster network.
- **Local disk:** set `local-directory` to node-local storage, not shared home.
- **Walltime:** match the partition's `MaxTime`; lifetimes derive from it.
- **Shared paths and Python:** the checkout, `python`, physics tables and
  output directory at the same absolute paths on every node. Build the venv
  inside a compute-node job.

### Worked example: Hoffman2

On UCLA's Hoffman2 (2026-11), compute nodes may call `sbatch`, workers reach a
compute-node scheduler over the default interface, and worker jobs connected
about 65-75 s after submission. The per-user `campus24` QoS on the `campus`
partition allows 1024 CPUs, 100 running jobs and 24 hours, so 30 jobs of 32
cores plus a 2-core coordinator fit:

```yaml
# ~/.config/dask/jobqueue.yaml
jobqueue:
  slurm:
    account: MY_ACCOUNT
    python: /u/home/USER/pyrite/.venv/bin/python
    local-directory: /tmp
    job-script-prologue:
      - ". /u/local/Modules/default/init/modules.sh"
      - "cd /u/home/USER/pyrite"
      - "export PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
```

```toml
[execution.hoffman2.resources]
cpu_cores = 32
memory_bytes = "64GiB"
accelerator = "cpu"

[execution.hoffman2.dask]
cluster = "slurm"
partition = "campus"
walltime = "23:00:00"
workers = 30
```

Submit the coordinator script above with `--partition=campus` and
`--time=24:00:00`. Hoffman2 login nodes are for editing and submission only:
build the environment (`pixi install`) inside a compute-node job, and
copy data through the `hoffman2-dtn` transfer node. A `pyrite remote` target
for the same checkout needs `--extras dask` (see
[remote targets](../guides/running-on-a-cluster.md#set-up-a-new-remote-host)), or each
remote job's exact `pixi install` removes Dask from the shared environment.

## Checkpoints, retries and outputs

Distributed execution partitions the case list into coarse work units; Dask
sends Futures, **not** electron tracks or spectrum arrays, over the scheduler
network. Workers read staged HDF5 inputs and publish immutable per-case
artifacts. Only the **coordinator** updates the normal checkpoint and detector
observation store. Both native SLURM and Dask support directional detector
work units.

Completed case artifacts can be reused after a failed/preempted task;
`max_attempts` bounds scheduler-level retries. Each distributed run also writes
operational `execution.json` under
`<checkpoint-root>/.execution-jobs/<run-id>/`. Allocation information is
*not* included in the scientific case hash. See
[Working with results](../guides/working-with-results.md) for checkpoints and analysis.

Dask's `max_in_flight` caps queued **plus** running Futures, limiting staging
pressure. For an existing scheduler, set it in line with the workers actually
available to the campaign; `workers` is a planning value, not a remote cluster
reservation.

## Native SLURM Python API

For advanced Python orchestration, supply resolved `cases`, an existing
`results` mapping and the campaign's table-aware content-key function. This
example wires the experimental executor to the normal sweep coordinator; it
is not a standalone simulation. The coordinator and compute nodes require
compatible Python/PyRITE installations and the same absolute shared paths.

```python
from pyrite.storage import CaseArtifactStore
from pyrite.execution.slurm import SlurmArrayConfig, SlurmArrayExecutor
from pyrite.execution import ResourceRequest, RetryPolicy
from pyrite.runs.run import run_sweep

# Supply `cases` from the existing campaign builder, and a content-key
# function that includes exactly the same physics/table markers as the sweep.
store = CaseArtifactStore("/shared/pyrite-output/checkpoints")
key_fn = ...  # existing resolved case identity function for this campaign
executor = SlurmArrayExecutor(
    store,
    key_fn,
    resources=ResourceRequest(cpu_cores=8, memory_bytes=32 * 1024**3),
    config=SlurmArrayConfig(partition="cpu", cases_per_task=4, max_concurrent=16),
    retry_policy=RetryPolicy(max_attempts=3),  # 1 (no automatic retries) by default
)
run_sweep(
    cases,
    results,
    executor=executor,
    artifact_store=store,
    content_key_fn=key_fn,
    checkpoint_dir="/shared/pyrite-output/checkpoints",
)
```

GPU tasks use one process; CPU tasks may use local multiprocessing bounded
by the allocated `cpu_cores`. The GPU request must match the scheduler's real
allocation and device visibility; see [GPU allocation](#native-slurm-array-cpu).

The operational files are under
`<artifact-root>/.execution-jobs/<run-id>/{inputs,logs,status}`. Immutable case
artifacts live under `<artifact-root>/<material>/.execution-artifacts/`.
A worker can publish valid results before a failure prevents its task completion
marker; the coordinator recovers those results before surfacing the failure.
See [worker artifact integrity](distributed-artifacts.md) for publication rules.

## Retry, cancellation and resumption

`RetryPolicy(max_attempts=N)` bounds the **total** number of SLURM array
submissions. The default is `N=1` (no automatic retries); opt in with e.g.
`max_attempts=3` for preemption-prone allocations. A failed or abruptly
terminated job submits only incomplete array indices with a new `sbatch
--array=...` override. Each attempt tags its status JSON so a stale failure
report cannot masquerade as a retry result. Before running numerical work,
each worker validates committed artifacts and skips already-completed cases;
coordinator callbacks run **at most once per case** within one `execute` call.
A corrupt committed artifact raises rather than being overwritten.

While `execute` is active, `executor.cancel()` can be called from another
thread or an event callback. It requests `scancel` for this executor's current
SLURM allocation, recovers any published case artifacts, and raises
`ExecutionCancelledError` from the coordinating `execute` call. Cancellation
does not delete the artifact inbox; resume the sweep to reuse completed work.
It requires permission to cancel the submitted SLURM job and a running
coordinator; it is **not** a background job-management service after the
coordinator itself exits. The `submitted_job_ids` and `last_run_dir` attributes
provide basic debugging/provenance for a live or finished invocation.

If all retry attempts fail, `execute` raises with the remaining failed task
indices. Successfully published case results stay in the artifact store and
are recovered by a subsequent `run_sweep(..., resume=True)`, including after
the coordinator or login session was interrupted. No recomputation is done for
committed and valid cases. Standard checkpoint metadata is always consolidated
by the coordinator rather than compute-node workers.

The native coordinator is synchronous: keep it alive while tasks run. It does
not reattach to an old allocation after coordinator termination. Resume reuses
committed artifacts through a new invocation. The common
[distributed limitations](#validation-and-limitations) also apply; no cluster
scaling or physical-validation claim follows from successful dispatch.

Implementation: {source}`src/pyrite/execution/slurm.py::SlurmArrayExecutor`.

## SSH remote helper and manual SLURM

For a cluster reached through PyRITE's SSH/SLURM helper:

```bash
pyrite config set remote.target MY_SSH_ALIAS
pyrite remote sync
pyrite run standard -m hopg --remote --dry-run
pyrite run standard -m hopg --remote --detach
pyrite remote job status
pyrite remote job logs --follow
pyrite remote pull hopg          # after the job completes
```

This workflow handles code/data synchronization and remote job bookkeeping;
it is separate from execution profiles. Consult `pyrite remote --help` for
site-specific target, partition and GPU configuration. For a scheduler not
integrated with these tools, a standard `sbatch` script can invoke
`pyrite run standard -m MATERIAL` directly within an allocated job; separate
materials can run as ordinary SLURM job-array elements. Remote trajectory
captures are not included in checkpoint pulls: inspect or transfer them with
`pyrite remote trajectories ls`, `info`, and `pull` (see the
[remote CLI reference](../repo-design/cli/cli-reference-remote.md)).

## Validation and limitations

Before scaling up, run a small Dask connectivity and correctness smoke test:

<!-- verify: skip (standalone developer smoke script, not a pyrite CLI invocation) -->
```bash
pixi run python checks/dask_execution_smoke.py --mode local \
  --output "$PWD/pyrite-output/dask-smoke-local"
```

On SLURM, the same check supports `--mode slurm` with site-provided scheduler
and worker addresses; use `--help` for its arguments. Results compare numerical
outputs and checkpoint reuse, **not** cluster scaling or physical accuracy.
Run GPU smoke tests only in appropriately isolated allocations.

Current distributed limitations: trajectory capture, transport-only dispatch,
explicit wall-clock-bounded sweeps and performance-profile wrappers are not
supported; `run_sweep` rejects them before touching any checkpoint or cache, even
when every case is already complete. CPU and GPU resources are homogeneous within
a work-unit group.
Check network reachability, shared-directory permissions, worker imports and
scheduler logs when tasks cannot start; use native SLURM arrays to isolate a
Dask-specific failure. Real multi-node throughput and reliability remain
site-dependent and require verification.
