# Run on a cluster

Preview a campaign, then run it locally, through Dask, or as a SLURM array.
Execution profiles change scheduling while retaining the resolved physics case
identity and random seeds. **Distributed execution is experimental**; verify a
small workload before scaling up.

(choose-an-execution-mode)=
(prepare-the-environment)=
## Prepare once

The coordinator and workers need compatible PyRITE/Python versions and the same
material data and reference tables. For execution profiles they also need a
**shared POSIX filesystem at the same absolute path**. Keep the coordinator
running until completion; use an allocation if your site requires it.

In a source checkout, install the Dask extra only if using Dask:

```bash
uv sync --extra dask
uv run pyrite tables verify
```

### Reuse compiled kernels across jobs

Kernels compile once per cache directory and are reused by later jobs. By
default each user's cache is under the PyRITE user cache directory on the node
running `pyrite` or importing `pyrite`; `pyrite cache status` prints it. To share one cache across
SLURM jobs, export the same `NUMBA_CACHE_DIR` (and, for GPU jobs,
`CUPY_CACHE_DIR`) on a shared filesystem before starting `pyrite`. Numba stores
entries per CPU model and feature set, and CuPy per GPU architecture and
toolkit, so mixed-node pools keep one entry per kind rather than reusing an
incompatible one; a shared directory therefore grows with node heterogeneity.
Warm the cache with one small job before submitting an array, so the array's
tasks do not all compile concurrently.

(run-and-inspect-a-profile)=
(native-slurm-array-cpu)=
(dask-with-slurm-provisioned-cpu-workers)=
(dask-with-an-existing-scheduler-cpu-or-gpu)=
## Preview, then run

Use a small campaign such as `quickstart` first. CPU examples:

```bash
PYRITE_MC_BACKEND=cpu pyrite run quickstart -m hopg --execution slurm-cpu --execution-plan
PYRITE_MC_BACKEND=cpu pyrite run quickstart -m hopg --execution local
PYRITE_MC_BACKEND=cpu pyrite run quickstart -m hopg --execution dask-local
PYRITE_MC_BACKEND=cpu pyrite run quickstart -m hopg --execution slurm-cpu
```

The preview reports **requested** resources, work units and retries. It does
not submit jobs or predict runtime. `slurm-cpu` requires SLURM tools and requests
one CPU per array task by default. Replace it with a site-specific execution
profile for partitions, memory and concurrency; complete
[SLURM and Dask templates](../reference/cluster-execution.md#native-slurm-array-cpu)
include GPU allocation rules. Dask SLURM provisioning supports CPU workers;
GPU Dask requires an existing scheduler with actual device isolation.

(checkpoints-retries-and-outputs)=
Only the coordinator updates the normal checkpoints; completed immutable case
artifacts can be reused after failure. Inspect the resulting dataset with
`pyrite checkpoint list`, then [plot saved results](working-with-results.md).
[Retry and output contracts](../reference/cluster-execution.md#checkpoints-retries-and-outputs)
explain operational metadata and queue limits.

(ssh-remote-helper-and-manual-slurm)=
## Submit from a workstation over SSH

For an already configured SSH host with PyRITE's SLURM helper, replace
`MY_SSH_ALIAS` with your host alias:

```bash
pyrite config set remote.target MY_SSH_ALIAS
pyrite remote sync
pyrite run quickstart -m hopg --remote --dry-run
pyrite run quickstart -m hopg --remote --detach
pyrite remote job status
pyrite remote job logs --follow
pyrite remote pull hopg
```

Pull after completion. This helper synchronizes code/data and manages remote
jobs; **do not combine `--remote` with `--execution`**. Remote trajectory captures
need a separate [trajectory transfer](../reference/trajectory-workflows.md),
rather than a checkpoint pull.

### Authenticated remote marimo apps (experimental)

Install the `apps` extra in the **remote** checkout's virtual environment
(`uv sync --package pyrite-mc --no-default-groups --extra apps`, together with
any GPU extra needed by your existing environment). The login node must support
SLURM, and compute nodes must be SSH-reachable through the login host for
port forwarding. The node must share the remote checkout and job directories.

```bash
pyrite remote app analysis --material hopg
pyrite remote app viewer --gpu --time 02:00:00
pyrite remote app validation --edit
pyrite remote app attach JOBID
pyrite remote job list --kind app --json
pyrite remote job stop JOBID --yes
```

The client waits for marimo to listen on **127.0.0.1** on the allocated node,
then sets up a local SSH port forward and opens the tokenized URL. If it is a
separate compute node, SSH uses the login host as a jump host. **Ctrl-C only
closes the tunnel**; the job keeps running until cancelled or its SLURM
walltime expires. Remote apps never accept `--no-token`.

For a static export, use `pyrite remote app analysis --batch`. This executes
marimo in the allocation and pulls the completed HTML into the local
`pyrite-output/results/` directory. A pending export can be retrieved later via
`pyrite remote app attach JOBID`. Remote app jobs reuse the existing synced
checkout and require `marimo` to already be installed in its `.venv`; the app
launcher does not modify a shared virtual environment while other jobs run.

(amd-rocm-nodes)=
### AMD (ROCm) nodes

Create separate named targets for the two partitions:

```bash
pyrite remote target add lab-gpu --host MY_SSH_ALIAS --remote-dir '~/pyrite-gpu'
pyrite remote target add lab-amd --host MY_SSH_ALIAS --remote-dir '~/pyrite-amd' \
  --gpu-vendor amd --partition gpu-amd --nodelist amd-node --gres gpu:radeon8060s:1
pyrite remote target use lab-gpu
pyrite run quickstart --remote --target lab-amd --dry-run
```

Use `target list`, `show NAME`, `edit NAME --FIELD VALUE`, or `remove NAME --yes`
to manage targets. `--target NAME` selects a target on remote-contacting commands;
`PYRITE_REMOTE_TARGET` overrides the stored default. Named targets use their own
stored fields ahead of per-field environment variables; omitted fields fall back
to environment, flat settings, then defaults. Without a named
selection, existing flat `remote.*` settings and environment overrides still apply.
Use distinct remote directories when GPU vendors differ; shared directories also
share job bookkeeping. New jobs record their target remotely and locally; job
commands recover it by ID. If an ID exists on multiple targets, pass `--target`.

AMD jobs keep their own `.venv-amd`, build CuPy for HIP on the first sync (about
9 minutes on `qlmc-ace`), and pin `PYRITE_MC_BACKEND=rocm` so a broken HIP stack fails
instead of running on the CPU. The prelude finds the ROCm root and the GPU
architecture itself. On a distribution ROCm without a host C++ compiler it
builds with the HIP toolchain's clang. Where profiler libraries are missing it
adds header wrappers and no-op roctx/roctracer stubs under
`<remote dir>/.rocm-shim`, and the job's stderr names each one. A stubbed
roctx disables `cupy.cuda.nvtx` markers; a stubbed roctracer disables
`cudaProfilerStart/Stop`. On Ubuntu, `sudo apt install libroctx-dev` provides
the real roctx, which the prelude then uses. Ubuntu does not package roctracer,
and AMD's `roctracer-dev` is built only for Ubuntu 22.04 and 24.04.

(validation-and-limitations)=
Execution profiles currently refuse trajectory capture, transport-only dispatch,
wall-clock-bounded sweeps and performance wrappers. Neither successful execution
nor an execution plan demonstrates physical accuracy or cluster scaling. Use
[connectivity checks and limitations](../reference/cluster-execution.md#validation-and-limitations)
before submitting a large workload.


## Submit and follow a remote profile from Python

The [experimental remote client](../reference/python-api/remote.md) uses the same
named destinations and batch scripts as `pyrite run --remote --target NAME`:

```python
from pyrite.remote import RemoteTarget, submit

job = submit("quickstart", target="lab-amd", materials=["hopg"])
print(job.jobid, job.status().state)
finished = job.wait(timeout=3600, poll_interval=10)
if finished.succeeded:
    pulled = job.pull()
    print(pulled.paths)
else:
    print(job.logs())
```

To use a destination without storing it in config:

```python
destination = RemoteTarget(
    host="MY_SSH_ALIAS",
    remote_dir="/srv/pyrite",
    uv="uv",
    gpu_vendor="amd",
    partition="gpu-amd",
)
preview = submit("quickstart", target=destination, dry_run=True)
print(preview.script)
```

The client returns values without terminal output. `job.stop()` explicitly
cancels the job; a `wait()` timeout leaves it running. Catch `RemoteError` for
remote failures and inspect `error.result` for partial checkpoint pulls. Keep the
returned handle: it retains the target and checkpoint identities used at submit
time. Target selection does not enter physics or case identity.
