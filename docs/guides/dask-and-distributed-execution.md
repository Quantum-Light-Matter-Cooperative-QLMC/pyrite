# Distributed execution: SLURM and optional Dask (#681)

`pyrite run --execution NAME` chooses scheduling **without changing `Numerics`,
physics case identity, or RNG seeds**. Native SLURM remains functional without
Dask. The `dask` optional extra installs `distributed` and `dask-jobqueue`:

```shell
uv pip install 'pyrite-mc[dask]'
# Or: uv sync --extra dask
```

## Dry-run resource inspection

```shell
pyrite run standard -m hopg --execution qlmc-nvidia \
  --execution-config execution.toml --execution-plan
```

The JSON plan shows case count, work-unit count, requested cores/GPUs/memory,
concurrency limit, and retry limit. **These are requested resources**, not
verified reservations or throughput predictions. This command does not submit
jobs, allocate a Dask cluster, or write checkpoints. The existing campaign
case builder runs to discover the actual case count.

At the Python level:

```python
from pyrite.execution import inspect_execution, resolve_execution_profile

p = resolve_execution_profile("slurm-cpu")
print(inspect_execution(p, case_count=320))
```

## Dask local, remote scheduler, and SLURM workers

Built-in quick check (requires the optional extra):

```shell
pyrite run standard -m hopg --execution dask-local
```

Explicit remote scheduler, with pre-configured workers and a **shared POSIX
filesystem available at the same absolute path** from coordinator and workers:

```toml
[execution.qlmc-dask]
backend = "dask"
max_attempts = 3
[execution.qlmc-dask.resources]
cpu_cores = 4
memory_bytes = "8GiB"
accelerator = "cpu"
[execution.qlmc-dask.dask]
cluster = "existing"
scheduler_address = "tcp://scheduler.internal:8786"
cases_per_task = 2
```

```shell
pyrite run standard -m hopg --execution qlmc-dask \
  --execution-config execution.toml
```

For automatic SLURM worker provisioning via `dask-jobqueue`:

```toml
[execution.qlmc-jobqueue]
backend = "dask"
max_attempts = 2
[execution.qlmc-jobqueue.resources]
cpu_cores = 8
memory_bytes = "32GiB"
[execution.qlmc-jobqueue.dask]
cluster = "slurm"
partition = "cpu"
workers = 2
max_workers = 6
adaptive = true
walltime = "01:00:00"
cases_per_task = 2
```

Dask uses Futures and stages case inputs as bounded immutable HDF5 batches
on shared disk; workers publish numerical outputs there. Only small Futures
and artifact references cross the scheduler network. The coordinator alone
updates PyRITE's standard checkpoint and observation store. A failed or
preempted task leaves completed case artifacts available for recovery/retry.

### GPUs and nested parallelism

GPU Dask execution is restricted to **an existing scheduler** with workers
configured for resource annotations (`GPU=1`) *and real device isolation*
(`CUDA_VISIBLE_DEVICES` on CUDA or `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`
on ROCm). Dask resource annotations do not isolate the hardware. PyRITE does
not auto-launch GPU `dask-jobqueue` workers because site-specific allocation,
module loading and visibility differ. Use native SLURM GPU arrays or connect
to an already-configured GPU Dask cluster. Each GPU task runs its normal
single-GPU numerical routine without an outer process pool. CPU units use
local serial dispatch to avoid multiplying nested process pools.

### Directional detector observations

Both native SLURM and Dask support direction-dependent detector work units.
Scalar results retain their established physics content key; raw directional
spectra use an additional key combining that physics key with the sampled
`(N, 3)` directions. Their artifacts live in a **separate directory**, never
inside scalar CAS. The coordinator verifies both before calling the existing
`ObservationPlan` assembly and checkpoint lifecycle. Detector response and
gain are assembled by the usual coordinator path, not by the worker.

### Provenance and limitations

Each distributed execution writes `execution.json` under its unique
`<checkpoint-root>/.execution-jobs/<run-id>/` directory. This contains backend,
requested resources, Python/host, and (for SLURM) submitted job IDs. It does
**not** enter a scientific case hash, CAS key or dataset identity.

Still unsupported: distributed trajectory capture, transport-only dispatch,
explicit wall-clock-bounded runs and performance-profile wrappers. The
coordinator must remain alive while Futures/job arrays complete. Ensure
worker Python installations and physics data are compatible, and verify cluster
filesystem locking/hardlink semantics. These backends have not yet been
validated with full multi-node workloads; benchmark against native SLURM
before using Dask for production.
