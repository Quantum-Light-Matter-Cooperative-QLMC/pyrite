# Experimental native SLURM job arrays (#681)

The first native SLURM backend partitions a **single** already-resolved sweep
into independent work units. Every task stages a small input HDF5 file on a
**shared POSIX filesystem**. Workers use PyRITE's existing `run_cases` numerical
engine and publish only immutable case artifacts. The coordinator alone executes
`run_sweep` callbacks, the standard content-addressable cache and authoritative
checkpoint writes. Dask is **not** required.

Python example (from a **SLURM login node** with the same installed PyRITE
version and Python path on compute nodes):

```python
from pyrite.execution.artifacts import CaseArtifactStore
from pyrite.execution.slurm import SlurmArrayConfig, SlurmArrayExecutor
from pyrite.execution import ResourceRequest
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
)
run_sweep(cases, results, executor=executor, artifact_store=store,
          content_key_fn=key_fn, checkpoint_dir="/shared/pyrite-output/checkpoints")
```

For one GPU per SLURM array task, request
`ResourceRequest(cpu_cores=8, gpu_count=1, accelerator="cuda")` and set a site
appropriate `partition` and (if necessary) `gres="gpu:a100:1"`. For ROCm use
`accelerator="rocm"` and the matching site-specific `gres`. SLURM **must**
enforce GPU allocation and visibility; Dask resource labels and PyRITE's
resource descriptors do not provide isolation. GPU workers use a **single
process**, avoiding nested pools and competing CUDA contexts. CPU tasks can
use local multiprocessing, bounded by `cpu_cores`.

The array files are under
`<artifact-root>/.execution-jobs/<run-id>/{inputs,logs,status}`. Each task
writes an atomic per-task completion marker. Individual case artifacts live
under `<artifact-root>/<material>/.execution-artifacts/` and remain reusable
through standard sweep resume. A crashed worker may leave completed case
artifacts even when it never writes its task-status marker; the coordinator
will deliver those successful references before surfacing the task failure.
The next `run_sweep(..., resume=True)` recovers already-published work.

Current limits: synchronous coordinator, homogeneous resource requests per
array, one GPU per array task, no job submission from a non-SLURM workstation,
no observation-direction/trajectory-capture jobs, no bounded sweep slices,
no automatic cancellation, no SLURM retry/resubmission, and no named CLI
execution profiles. Use existing `pyrite remote` for SSH-based workflows.
Neither this implementation nor the current tests demonstrate multi-node
scaling efficiency; benchmark on a real cluster before enabling by default.
