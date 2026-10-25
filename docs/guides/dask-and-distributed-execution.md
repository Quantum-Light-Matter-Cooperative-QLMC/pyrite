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
pyrite run standard -m hopg --execution cluster-nvidia \
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
[execution.cluster-dask]
backend = "dask"
max_attempts = 3
[execution.cluster-dask.resources]
cpu_cores = 4
memory_bytes = "8GiB"
accelerator = "cpu"
[execution.cluster-dask.dask]
cluster = "existing"
scheduler_address = "tcp://scheduler.internal:8786"
cases_per_task = 2
max_in_flight = 4
```

```shell
pyrite run standard -m hopg --execution cluster-dask \
  --execution-config execution.toml
```

For automatic SLURM worker provisioning via `dask-jobqueue`:

```toml
[execution.cluster-jobqueue]
backend = "dask"
max_attempts = 2
[execution.cluster-jobqueue.resources]
cpu_cores = 8
memory_bytes = "32GiB"
[execution.cluster-jobqueue.dask]
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

`max_in_flight` bounds submitted Futures, including queued and running tasks.
The default is twice the configured worker count, or twice `max_workers` for
an adaptive cluster. For an existing scheduler or borrowed Python `Client`,
set this explicitly to match the workers allocated to this campaign. Completed
Futures are released as their artifacts are consumed; cancellation and
coordinator callback failures cancel outstanding submissions. Work-unit
descriptors are staged incrementally, while the existing case sequence and
result bookkeeping still require memory proportional to campaign size.

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

## Checking a site's SLURM deployment

Dask needs compatible Python environments, shared paths and TCP connectivity
between scheduler and workers. `dask-jobqueue` submits ordinary SLURM jobs;
the process constructing `SLURMCluster` hosts the Dask scheduler. See the
upstream [SLURMCluster configuration](https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html)
for scheduler addresses, worker environments and site directives.

Check compute-to-scheduler networking before submitting a physics campaign.
If workers cannot reach a scheduler on the login node, try hosting the
coordinator on a compute node. A working local connection does not prove that
another node can reach the same listener.

The checkout includes `checks/dask_execution_smoke.py`. It runs three small
CPU HOPG cases with fixed seeds, compares local and Dask spectra, and checks
coordinator checkpoint recovery without recomputation. `--native-slurm` also
compares a native job array. The report records code identity, worker metadata
and numerical checks for the operator; review that metadata before sharing it.
The path timers exclude cluster provisioning and have different import/JIT
warm-up costs, so they are not scaling benchmarks. Install the optional Dask
dependencies and current physics tables, preserving any accelerator extras.

The node names and documentation-only IP addresses below are placeholders.
Replace them with your site's allocation and network settings after a clean
`pyrite remote sync`:

<!-- verify: skip (SLURM job template runs a Python smoke script and requires site resources) -->
```bash
#!/usr/bin/env bash
#SBATCH --job-name=pyrite-dask-smoke
#SBATCH --partition=cpu
#SBATCH --nodelist=coordinator-node
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:08:00
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"  # submit from the synced checkout
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/pyrite-output/dask-smoke-user-catalog"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYRITE_MC_BACKEND=cpu
uv run --no-sync python checks/dask_execution_smoke.py --mode slurm \
    --scheduler-host 192.0.2.10 --worker-node compute-node --native-slurm \
    --output "$PWD/pyrite-output/dask-smoke-$SLURM_JOB_ID"
```

The script also accepts `--mode local` for a process-worker smoke without
SLURM. Its output directory must be new, so previous results cannot make a
failed execution look successful.

### Isolated NVIDIA smoke

Run the coordinator in a **CPU** allocation, then let the smoke script request
a separate GPU worker. The coordinator must not reserve the sole GPU while
waiting for its worker. The site-configured worker requests a real SLURM GRES,
advertises `GPU=1`, and checks `CUDA_VISIBLE_DEVICES`, CuPy device count and
device identity. This uses a borrowed client through the existing-scheduler
interface; it does not enable generic GPU jobqueue provisioning in PyRITE
execution profiles.

On a single-GPU node, place the CPU coordinator and GPU worker on the same
host. Adapt the example above to select that node for the coordinator, then
replace the final command with:

<!-- verify: skip (Python CUDA smoke runner requires an allocated SLURM GPU) -->
```bash
export PYRITE_FP64=1
uv run --no-sync python checks/dask_execution_smoke.py --mode slurm \
    --scheduler-host 192.0.2.30 --worker-node gpu-node --partition gpu \
    --gpu --gres gpu:1 --native-slurm \
    --output "$PWD/pyrite-output/dask-gpu-smoke-$SLURM_JOB_ID"
```

The comparison retains `rtol=1e-10`, `atol=1e-18` for both energy grids and
CXR, bremsstrahlung and characteristic spectra. It exercises the runner's
automatic transport policy with GPU scoring; it does not independently
validate forced device-resident transport, ROCm, concurrent GPU tenants or
scaling efficiency. Device visibility and scheduler annotations do not prove
kernel-enforced GPU isolation.

### Network and containment troubleshooting

If a compute node reaches the login node's SSH/SLURM services but times out
on a Dask listener that works locally, inspect inbound filtering and routing.
Confirm the listener's bind address. On Windows/WSL hosts, an administrator may
also need to inspect Windows and Hyper-V firewall policies.

A candidate fixed-port plan is TCP 8786 for the scheduler, 8800–8815 for workers
and 8820–8835 for nannies. Configure the site-owned cluster/client with those
ports before opening them; PyRITE's default jobqueue setup chooses dynamic ports:

```python
scheduler_options = {"host": "192.0.2.30", "port": 8786, "dashboard": False}
worker_extra_args = ["--worker-port", "8800:8815", "--nanny-port", "8820:8835"]
```

The following example is a review-only Hyper-V rule, using placeholder
addresses and Microsoft's standard WSL creator identifier. Inspect existing
policy and any corresponding Windows host rule before applying a change.
Restrict access to the intended compute peers and retain the existing default
inbound policy. See [Microsoft's Hyper-V rule reference](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallhypervrule?view=windowsserver2022-ps).

```powershell
New-NetFirewallHyperVRule -Name 'PyRITE-Dask-Compute-Inbound' `
    -DisplayName 'PyRITE Dask from compute nodes' -Direction Inbound -Action Allow `
    -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP `
    -LocalAddresses '192.0.2.30' -LocalPorts '8786','8800-8815','8820-8835' `
    -RemoteAddresses '192.0.2.20','192.0.2.21' -WhatIf
```

SLURM reservations alone do not prove containment. If `TaskPlugin` is unset
or `CgroupPlugin=disabled`, inspect actual CPU affinity and memory limits.
Thread limits constrain cooperative applications but do not establish an
enforced allocation boundary. When accounting storage is disabled, use
`scontrol`, job logs and worker-side measurements instead of `sacct`.

An administrator evaluating cgroup v2 can review this candidate configuration:

```ini
# slurm.conf
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity

# cgroup.conf
CgroupPlugin=cgroup/v2
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
AllowedSwapSpace=0
ConstrainDevices=yes
```

Check the installed Slurm version, its [cgroup requirements](https://slurm.schedmd.com/cgroup_v2.html)
and [configuration reference](https://slurm.schedmd.com/cgroup.conf.html).
Systemd delegation and available controllers are prerequisites, not proof of
successful daemon startup. Back up controller/node configurations and arrange
an idle maintenance window. Validate startup, especially on WSL, before a
cluster-wide rollout; retain the old files for rollback if scope creation fails.
After restart, verify an allocated job's actual cpuset and finite `memory.max`,
bounded memory-limit failure, survivor cleanup and unchanged GPU admission.
WSL's `/dev/dxg` needs explicit device-controller validation.
