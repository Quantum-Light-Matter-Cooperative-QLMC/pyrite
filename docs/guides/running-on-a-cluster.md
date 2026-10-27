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

(validation-and-limitations)=
Execution profiles currently refuse trajectory capture, transport-only dispatch,
wall-clock-bounded sweeps and performance wrappers. Neither successful execution
nor an execution plan demonstrates physical accuracy or cluster scaling. Use
[connectivity checks and limitations](../reference/cluster-execution.md#validation-and-limitations)
before submitting a large workload.
