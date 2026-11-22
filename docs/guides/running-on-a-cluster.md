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
running until completion; on an HPC site run it as a compute-node job, never
on a login node.

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
include GPU allocation rules.

(checkpoints-retries-and-outputs)=
Only the coordinator updates the normal checkpoints; completed immutable case
artifacts can be reused after failure. Inspect the resulting dataset with
`pyrite checkpoint list`, then [plot saved results](working-with-results.md).
[Retry and output contracts](../reference/cluster-execution.md#checkpoints-retries-and-outputs)
explain operational metadata and queue limits.

(dask-with-slurm-provisioned-cpu-workers)=
(dask-with-an-existing-scheduler-cpu-or-gpu)=
## Run Dask workers on a SLURM site

Dask on SLURM fills whole worker jobs: each job runs one single-threaded
worker process per allocated core. Site details (account, QoS, modules,
Python, network interface) go in dask-jobqueue's own config file; the
execution profile keeps resources, partition and walltime. The
[Dask reference](../reference/cluster-execution.md#dask-on-an-hpc-cluster)
lists every field and a worked Hoffman2 example.

1. Answer the [site checklist](../reference/cluster-execution.md#site-checklist):
   account and QoS limits, whether compute nodes may run `sbatch`, the
   network interface, node-local disk and the partition's walltime.
2. Build the environment inside a short compute-node job, not on the login node:

   ```bash
   srun --time=00:30:00 --cpus-per-task=2 --mem=8G uv sync --extra dask
   ```

   Add the GPU vendor extra (for example `--extra nvidia`) for GPU workers.

3. Put the site settings in `~/.config/dask/jobqueue.yaml`, starting from
   the [starter file](../reference/cluster-execution.md#site-settings).
4. Write an execution profile with `cluster = "slurm"`, `resources.cpu_cores`
   per worker job and `workers` jobs, then preview it inside a short
   allocation; the preview submits nothing:

   ```bash
   pyrite run quickstart -m hopg --execution site-dask      --execution-config execution.toml --execution-plan
   ```

5. Submit the [coordinator job](../reference/cluster-execution.md#coordinator-job)
   with `sbatch`. It starts the scheduler, submits the worker jobs and writes
   the checkpoints. If it stops before finishing, submit it again; finished
   cases are reused.

If compute nodes may not call `sbatch`, start the scheduler and workers
yourself and use a profile with `cluster = "existing"` and `scheduler_file`.
For one-GPU workers, add the site's GPU directive (for example
`--gres=gpu:1`) to `job_extra_directives`; PyRITE refuses a GPU profile
without one.

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

(new-remote-host)=
### Set up a new remote host

The helper needs SSH access to a SLURM login node, `uv` on that host, and a
home or project directory that compute nodes also see. Do not install PyRITE
or fetch tables there yourself: each batch job runs `uv sync` (and downloads
a supported Python if needed) on its compute node, and `pyrite remote sync`
ships your local tables.

1. On the host, install `uv`, a single binary:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh   # ~/.local/bin/uv
   ```

   A different `uv` location goes in `--uv PATH`. `pyrite remote sync`
   refuses when it cannot find `uv`.
2. Choose the remote directory against the site's quotas. Checkpoints and the
   job environment live under it, so prefer a backed-up home or project space
   over purged scratch.
3. Read the site's SLURM policy, then preview a one-GPU request without
   submitting it:

   ```bash
   ssh MY_SSH_ALIAS 'sinfo -o "%P %l %G %f"; sacctmgr -nP show assoc user=$USER format=account,qos'
   ssh MY_SSH_ALIAS sbatch --test-only -p PARTITION -c 8 --gres=gpu:1 -t 30 \
     --constraint='FEATURE_A|FEATURE_B' --wrap=true
   ```

   Untyped `gpu:1` gres on a mixed partition can land on any GPU model;
   select models with node features (`--constraint`), not `--nodelist`, which
   requires every listed node. A partition `MaxTime` means jobs need a
   `--max-time` cap.
4. Add the target with the site policy. For example, a shared campus
   partition capped at 24 hours, on a site that reserves login nodes for
   editing and submission:

   ```bash
   pyrite remote target add campus --host MY_SSH_ALIAS --remote-dir '~/pyrite' \
     --partition campus --account MY_ACCOUNT \
     --constraint 'A100|H200' --max-time 1-00:00:00 \
     --host-exec 'srun --account=MY_ACCOUNT --time=00:30:00 --cpus-per-task=2 --mem=8G'
   pyrite remote sync --target campus
   ```

   Each job runs an exact `uv sync` of the vendor extra, which removes
   other extras from the shared environment. To keep the Dask backend there,
   add `--extras dask` (comma-separated for several).

   Simulations always run as `sbatch` jobs. Sync, job status, logs and stop
   are file and scheduler operations on the login node. `pull`, `remote gc`
   and `remote trajectories` also run PyRITE on the host (`pull` slims each
   checkpoint there and streams it back); `--host-exec` runs those inside a
   short allocation instead, at the cost of the allocation wait per command.
   Use a blocking `srun` without `--pty`: it returns the command's output
   over the same SSH session, which `sbatch` cannot.

   `--max-time` replaces an unlimited request with the cap and refuses longer
   finite requests before submission. A chunked slice requests three times
   `--chunk-minutes`, so a 24-hour cap allows `--chunk-minutes 480` at most.
   Each slice queues again, so on a busy shared partition use long chunks.
   Chunked slices are also submitted with `--nice=10000` by default, which
   yields a shared box to other jobs between slices. Where priority is mostly
   queue age (`ssh MY_SSH_ALIAS 'sprio -w'`), that penalty delays every slice;
   set `pyrite remote target edit campus --nice 0`.

   For email when jobs finish or fail, set SLURM mail events and, if the
   site does not deliver to your cluster username, a recipient:

   ```bash
   pyrite remote target edit campus --mail-type END,FAIL --mail-user you@example.org
   ```

   Every chunked slice is a separate SLURM job, so `END` mails once per
   slice; use `FAIL` alone for long chunked runs. An empty `--mail-type ''`
   turns notifications off.

   A site QOS goes in `--qos`. Any other sbatch option the site needs, such as
   a reservation or excluded nodes, goes in `--sbatch-extra` as long options:

   ```bash
   pyrite remote target edit campus --qos highp \
     --sbatch-extra '--reservation=lab_resv --exclude=node3'
   ```

   Options PyRITE writes itself (time, output, job name, signal, CPU and memory
   counts) or that have their own flag are rejected.
5. If the site routes file transfers through data-transfer nodes (DTNs), give
   the DTN its own SSH alias and add it as the target's transfer host:

   ```text
   Host MY_DTN_ALIAS
       HostName dtn.example.edu
       User MY_USER
       ServerAliveInterval 30
   ```

   ```bash
   ssh MY_DTN_ALIAS true   # accept and check its host key once, yourself
   pyrite remote target edit campus --transfer-host MY_DTN_ALIAS
   ```

   The transfer host must mount the remote directory at the same absolute
   path; a `~/` directory is checked by comparing the two homes. Then sync
   uploads, the table inventories and unpacking, and the downloads of `pull`,
   `remote trajectories` and `remote performance` go to the DTN. Scheduler
   commands, the sync lock and PyRITE itself stay on the login node or in
   `--host-exec`. A pull with a transfer host slims into
   `.pull-staging/` in the remote directory, then downloads with resumable
   `rsync`: rerun an interrupted pull to resume it, and `remote gc --yes`
   removes leftover staged files. For one-off multi-gigabyte moves, the site's
   Globus endpoint is faster than SSH.
6. Run something small and confirm the allocation:

   ```bash
   pyrite run quickstart -m hopg --remote --target campus --dry-run
   pyrite run quickstart -m hopg --remote --target campus --chunk-minutes 400 --detach
   pyrite remote job logs --follow
   ```

   The job log header names the node. Compute nodes must reach the package
   index for the first `uv sync`; on a site without outbound access, fill
   uv's cache once from an interactive allocation
   (`srun --time=01:00:00 --cpus-per-task=4 --mem=8G --pty $SHELL -l`) by
   running `uv sync --package pyrite-mc --no-default-groups --extra nvidia`
   in the remote directory.

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
