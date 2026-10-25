# Running on a cluster (SLURM and Dask)

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
| Dask on SLURM | custom Dask profile with `cluster = "slurm"` | Provision CPU Dask workers using `dask-jobqueue` |
| Existing Dask scheduler | custom Dask profile with `cluster = "existing"` | Join an available CPU or properly isolated GPU cluster |

The separately configured `pyrite run --remote` workflow synchronizes to an
SSH host and submits jobs through that host's SLURM setup. **Do not combine
`--remote` with `--execution`.**

## Prepare the environment

On the coordinator and every worker, use compatible Python/PyRITE versions,
physics tables and material data. From a source checkout, install the locked
environment with `uv sync`; add Dask only when using its backend:

```bash
uv sync --extra dask       # includes distributed and dask-jobqueue
uv run pyrite tables verify
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
  running until work finishes; run it in an appropriate allocation when site
  policy requires one.

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

### Dask with SLURM-provisioned CPU workers

Dask can request CPU worker allocations through `dask-jobqueue`:

```toml
[execution.cpu-dask]
backend = "dask"
max_attempts = 2

[execution.cpu-dask.resources]
cpu_cores = 8
memory_bytes = "32GiB"
accelerator = "cpu"

[execution.cpu-dask.dask]
cluster = "slurm"
partition = "cpu"                  # adapt to your site
walltime = "01:00:00"
workers = 2
max_workers = 6
adaptive = true
cases_per_task = 2
max_in_flight = 12
```

```bash
pyrite run standard -m hopg --execution cpu-dask \
  --execution-config execution.toml
```

This mode creates a Dask scheduler in the **coordinator process**, then
launches SLURM CPU workers. The workers must be able to connect back to that
scheduler; cluster network policies may require the coordinator to run on a
compute node instead of a login node. Automatic GPU worker provisioning is
**not supported** here.

### Dask with an existing scheduler (CPU or GPU)

```toml
[execution.existing-dask]
backend = "dask"
max_attempts = 3

[execution.existing-dask.resources]
cpu_cores = 4
memory_bytes = "8GiB"
accelerator = "cpu"

[execution.existing-dask.dask]
cluster = "existing"
scheduler_address = "tcp://scheduler.internal:8786"  # site-specific
workers = 2
cases_per_task = 2
max_in_flight = 4
```

Use the same CLI pattern with `--execution existing-dask`.

GPU Dask requires a **preconfigured existing scheduler**. Change the resource
request to `gpu_count = 1` and `accelerator = "cuda"` or `"rocm"`, configure
workers with `GPU=1` Dask resources, and ensure each worker actually receives
an isolated device (`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, or
`HIP_VISIBLE_DEVICES`). A Dask resource label alone does **not** isolate a GPU.
PyRITE runs the existing single-GPU numerical path in each GPU task, without
adding an outer multiprocessing pool.

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
[Working with results](working-with-results.md) for checkpoints and analysis.

Dask's `max_in_flight` caps queued **plus** running Futures, limiting staging
pressure. For an existing scheduler, set it in line with the workers actually
available to the campaign; `workers` is a planning value, not a remote cluster
reservation.

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
uv run python checks/dask_execution_smoke.py --mode local \
  --output "$PWD/pyrite-output/dask-smoke-local"
```

On SLURM, the same check supports `--mode slurm` with site-provided scheduler
and worker addresses; use `--help` for its arguments. Results compare numerical
outputs and checkpoint reuse, **not** cluster scaling or physical accuracy.
Run GPU smoke tests only in appropriately isolated allocations.

Current distributed limitations: trajectory capture, transport-only dispatch,
explicit wall-clock-bounded sweeps and performance-profile wrappers are not
supported. CPU and GPU resources are homogeneous within a work-unit group.
Check network reachability, shared-directory permissions, worker imports and
scheduler logs when tasks cannot start; use native SLURM arrays to isolate a
Dask-specific failure. Real multi-node throughput and reliability remain
site-dependent and require verification.
