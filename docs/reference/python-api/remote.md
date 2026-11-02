# Remote compute targets

The **experimental** `pyrite.remote` client submits catalog profile runs over
SSH to SLURM. It owns remote destination selection and job lifecycle; local
simulation signatures and scientific identities are unchanged. This is separate
from the [execution-profile scheduler interface](execution.md).

Canonical imports: `RemoteTarget`, `resolve_target`, `submit`, `RemoteJob`,
`RemoteStatus`, `PullResult`, and `RemoteError` from `pyrite.remote`. Other
historical orchestration aliases in that package remain internal.

`submit(profile, target=NAME_OR_VALUE)` returns a resolved `RemoteJob` immediately
after submission. Its `status()`, `logs()`, `stop()`, `wait()`, and `pull()` methods
perform synchronous I/O without printing. The CLI and client use the same script
builder and admission checks. Material selection defaults to profile membership;
`materials=[...]` narrows it. Only ordinary profile runs are exposed; profiling,
validation, recomputation and streaming viewers remain CLI workflows.

## Selection and side effects

`resolve_target` performs no network I/O. Explicit target values are used as-is.
Names resolve stored fields ahead of per-field environment variables, flat
configuration and defaults; name selection defaults to `PYRITE_REMOTE_TARGET`,
then the configured default. Unknown/invalid selections raise `ValueError` or
`TypeError`. Remote paths use POSIX syntax; `~/` expands through a cached SSH
remote-home lookup when a path is used. For an offline dry run, provide absolute
`remote_dir` and `uv` paths (or an executable name such as `uv`).

Submission may sync the checkout and selected catalog, reserve checkpoints, and
submit an allocation. `no_sync=True` uses the existing remote code. Dry runs only
construct scripts and refuse subsequent job operations. Job handles retain their
resolved destination, generated script, and checkpoint stems so later config or
catalog edits cannot redirect their operations. Submission allocation IDs are
snapshots; `status()` reads the current ID after chunked resubmission.

`wait(timeout=..., poll_interval=...)` polls the persisted job state; time values
are seconds. A timeout raises `TimeoutError` and leaves the job running. It does
not impose a timeout on an individual SSH call. A terminal state can represent
failure; check `snapshot.succeeded`. `stop()` cancels the allocation and waits
for retirement before releasing reservations. `pull()` writes checkpoint
directories into the current local workspace, grid-filtered by default. It may
sync source for remote slimming unless `no_sync=True`.

Transport/submission failures raise `RemoteError`. Partial pulls attach a
`PullResult` to `error.result`, containing successful paths, failed stems and
error messages. An SSH loss during submission can be ambiguous: inspect remote
job metadata before retrying. No automatic retry or asynchronous client is
provided. Reconstructing a handle manually requires a resolved target; pulling
also requires the saved checkpoint stems from submission.

The interface is provisional. Promotion requires evidence from multiple SSH and
SLURM deployments, stable error and handle persistence contracts, and review of
transport concurrency. It has no independent scientific validation claim; it
runs the same physics and records as the CLI.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.remote.RemoteTarget
   pyrite.remote.resolve_target
   pyrite.remote.submit
   pyrite.remote.RemoteJob
   pyrite.remote.RemoteStatus
   pyrite.remote.PullResult
   pyrite.remote.RemoteError
```
