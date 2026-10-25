# Named execution profiles (issue #681, experimental)

Execution *profiles* select where independent cases are scheduled. They are
separate from `pr.ExecutionConfig` (#680), which controls the local numerical
kernels and chunk sizes, and are **not** part of a case's physics identity.

Available backends are `local`, native `slurm`, and optional Futures-based
`dask` (installed through `pyrite-mc[dask]`). Profiles use a small explicit TOML
file; migration into the unified named object catalog in #680/ADR-0021 is
future work. Selecting a profile does not change material presets.

## Built-in profiles

The old invocation retains its exact default behavior:

```shell
pyrite run standard -m hopg
```

An explicit local executor is also available:

```shell
pyrite run standard -m hopg --execution local
```

For a **CPU-only** SLURM array with default one-core tasks:

```shell
pyrite run standard -m hopg --execution slurm-cpu
```

For local Dask testing, use `pyrite run standard -m hopg --execution dask-local`.
Both distributed backends require a shared filesystem visible at the **same
absolute path** from the coordinator and compute nodes. SLURM needs
`sbatch`/`squeue`/`scancel` and a compatible Python installation. Workers must
be able to import PyRITE and access its physics datasets. The built-in profiles
are smoke-test configurations, not site-portable resource recommendations.
See [Dask and distributed execution](dask-and-distributed-execution.md).

## Named resources and retry controls

Example `execution.toml`:

```toml
[execution.cluster-nvidia]
backend = "slurm"
max_attempts = 3

[execution.cluster-nvidia.resources]
cpu_cores = 8
memory_bytes = "32GiB"
gpu_count = 1
accelerator = "cuda"
gpu_memory_bytes = "16GiB"

[execution.cluster-nvidia.slurm]
partition = "gpu"
walltime = "02:00:00"
max_concurrent = 4
cases_per_task = 2
gres = "gpu:1"
# python_executable = "/path/visible/on/compute/nodes/bin/python"
```

Run:

```shell
pyrite run standard -m hopg --execution cluster-nvidia --execution-config execution.toml
```

The same selection works from Python:

```python
from pyrite.execution import build_executor, resolve_execution_profile

profile = resolve_execution_profile("cluster-nvidia", config_path="execution.toml")
executor, artifact_store = build_executor(
    profile,
    case_key=content_key_fn,  # use pyrite.api.source_content_key_fn()
    artifact_root="/shared/checkpoints",  # absolute path shared by every node
)
run_sweep(
    cases,
    results,
    executor=executor,
    artifact_store=artifact_store,
    content_key_fn=content_key_fn,
    checkpoint_path=checkpoint_path,
)
```

The `content_key_fn` passed to the executor and the coordinator **must be the
same canonical physics/table-aware function**. The executor never hashes the
profile name or substitutes allocation metadata for physics identity.

## Current limitations

- One homogeneous resource request per array, with at most one GPU per task.
- Both distributed backends support independently keyed detector-direction
  artifacts. Trajectory capture, time-budgeted sweeps and performance-profile
  wrappers remain unsupported and are rejected before submission.
- A GPU selector requires `accelerator = "cuda"` or `"rocm"`; device isolation
  and affinity remain SLURM/site responsibilities, not promises from a profile.
- The CLI's existing `-R/--remote` submission mode is a **different** workflow
  and cannot be combined with `--execution`.
- The TOML file is explicit per invocation; it is not yet a persistent catalog
  of execution presets. The `pyrite run` default remains unchanged.
- Execution provenance now lives beside each distributed staging directory,
  not inside scientific case identities. Cluster-end-to-end validation and
  scaling benchmarks remain outstanding. The local executor still owns its
  existing multiprocessing, GPU fallback and kernel scheduling policies.
