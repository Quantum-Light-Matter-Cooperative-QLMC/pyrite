"""Experimental synchronous SSH/SLURM client with typed, silent results."""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import config, lifecycle, queue, state, transport
from .config import RemoteTarget
from .reporting import quiet
from .selection import select_materials
from .targets import resolve_target


class RemoteError(RuntimeError):
    """Remote I/O or submission failed; ``result`` retains a partial pull report.

    An SSH failure during submission can leave an accepted scheduler job. Check
    its remote metadata before retrying; the client retains reservation safety.
    """

    def __init__(self, message: str, *, result: PullResult | None = None):
        super().__init__(message)
        self.result = result


@contextmanager
def _operation(target: RemoteTarget):
    with config.use_target(target), quiet():
        try:
            yield
        except (SystemExit, OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", None)
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise RemoteError(f"{exc}{': ' + detail.strip() if detail else ''}") from exc


@dataclass(frozen=True)
class RemoteStatus:
    """Persisted job state and current allocation ID (chunked jobs may change it)."""

    jobid: str
    state: str
    scheduler_id: str | None

    @property
    def terminal(self) -> bool:
        return self.state.lower().startswith(("done", "failed", "cancelled"))

    @property
    def succeeded(self) -> bool:
        return self.state.lower().startswith("done") and not self.state.lower().startswith(
            "done with "
        )


@dataclass(frozen=True)
class PullResult:
    """Completed checkpoint directories and failures from one explicit pull."""

    paths: tuple[Path, ...]
    failed: tuple[str, ...] = ()
    errors: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RemoteJob:
    """A resolved target and submission, independent of later config changes.

    ``script`` is the generated batch script; dry-run handles have no allocation
    and refuse lifecycle operations. ``checkpoint_stems`` preserves submit-time
    identities for pulling even if the local catalog changes. Methods perform
    blocking SSH I/O and return values without terminal output.
    """

    jobid: str
    target: RemoteTarget
    profile: str = "standard"
    materials: tuple[str, ...] = ()
    checkpoint_stems: tuple[str, ...] = ()
    script: str = ""
    scheduler_id: str | None = None
    dry_run: bool = False

    def __post_init__(self):
        if not isinstance(self.target, RemoteTarget):
            raise TypeError("target must be a resolved RemoteTarget")
        if not isinstance(self.jobid, str) or not self.jobid:
            raise ValueError("jobid must be a nonempty safe job identifier")
        try:
            transport._check_shell_tokens([self.jobid])
        except SystemExit as exc:
            raise ValueError(str(exc)) from exc

    def _require_submitted(self):
        if self.dry_run:
            raise ValueError("dry-run handles have no submitted remote job")

    def status(self) -> RemoteStatus:
        """Read persisted progress/terminal state and the current scheduler ID."""
        self._require_submitted()
        with _operation(self.target):
            value = state._job_state(self.jobid)
            if not value:
                raise RemoteError(f"job {self.jobid!r} has no remote state")
            return RemoteStatus(self.jobid, value, state._slurm_job_id(self.jobid))

    def logs(self, *, lines: int = 60) -> str:
        """Fetch the last ``lines`` diagnostic lines; no streaming or rendering."""
        self._require_submitted()
        if type(lines) is not int or lines < 1:
            raise ValueError("lines must be a positive integer")
        with _operation(self.target):
            directory = config.shell_remote_path(config.JOBS_SUBDIR, self.jobid)
            return transport._ssh_capture(f"tail -n {lines} -- {directory}/log")

    def stop(self) -> RemoteStatus:
        """Cancel this job, wait for its allocation to retire, and release reservations."""
        self._require_submitted()
        with _operation(self.target):
            scheduler_id = lifecycle._stop_jobid(self.jobid)
            return RemoteStatus(self.jobid, "cancelled", scheduler_id)

    def wait(self, *, timeout: float | None = None, poll_interval: float = 10.0) -> RemoteStatus:
        """Poll until terminal; timeout (seconds) raises ``TimeoutError`` without cancelling."""
        self._require_submitted()
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and greater than zero")
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative, or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            snapshot = self.status()
            if snapshot.terminal:
                return snapshot
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"remote job {self.jobid} is still active")
            time.sleep(poll_interval if remaining is None else min(poll_interval, remaining))

    def pull(self, *, grid: bool = True, no_sync: bool = False) -> PullResult:
        """Pull submitted checkpoints to the current local workspace.

        This may sync source code, run remote slimming, and write local checkpoint
        directories. ``no_sync=True`` uses the code already on the host. Partial
        failures raise ``RemoteError`` with a ``PullResult`` in ``error.result``.
        """
        self._require_submitted()
        if not self.checkpoint_stems:
            raise ValueError("pull requires the checkpoint_stems retained by submit()")
        from ..console.outputs import output_dir

        summary = {}
        try:
            with _operation(self.target):
                lifecycle.pull(
                    list(self.checkpoint_stems),
                    grid=grid,
                    no_sync=no_sync,
                    summary=summary,
                    _expand_surveys=False,
                )
        except RemoteError as exc:
            result = _pull_result(summary, output_dir("checkpoints"))
            raise RemoteError(str(exc), result=result) from exc
        return _pull_result(summary, output_dir("checkpoints"))


def _pull_result(summary, root) -> PullResult:
    return PullResult(
        tuple(root / stem for stem in summary.get("completed", [])),
        tuple(summary.get("failed", [])),
        tuple(summary.get("errors", {}).items()),
    )


def submit(
    profile: str = "standard",
    *,
    target: str | RemoteTarget | None = None,
    materials: Sequence[str] | None = None,
    quick: bool = False,
    workers: int | None = None,
    chunk_minutes: float = 10.0,
    parallel_materials: int | None = None,
    no_sync: bool = False,
    dry_run: bool = False,
) -> RemoteJob:
    """Submit a catalog profile to a named or explicit remote target.

    Defaults match ``pyrite run PROFILE --remote``: chunked allocation, sync
    enabled, profile material membership. Returns immediately after submission.
    Use ``job.wait()`` and ``job.pull()`` to follow and retrieve results. No GPU
    work runs locally. Named fields override environment, flat config and defaults;
    an explicit ``RemoteTarget`` is used as-is. Invalid selections raise
    ``ValueError``/``TypeError``; remote failures raise ``RemoteError``.

    ``chunk_minutes=0`` selects a monolithic allocation; only that mode accepts
    ``parallel_materials``. ``workers`` is a nonnegative CPU worker count.
    Dry runs generate a batch script without submission or sync. Absolute
    ``remote_dir`` and ``uv`` avoid the remote-home lookup needed for ``~/``.
    """
    if isinstance(materials, (str, bytes)):
        raise TypeError("materials must be a sequence of names, not a string")
    if workers is not None and (type(workers) is not int or workers < 0):
        raise ValueError("workers must be a nonnegative integer or None")
    if not math.isfinite(chunk_minutes) or chunk_minutes < 0:
        raise ValueError("chunk_minutes must be finite and nonnegative")
    if parallel_materials is not None:
        if (
            type(parallel_materials) is not int
            or not 1 <= parallel_materials <= config.MAX_PARALLEL_MATERIALS
        ):
            raise ValueError(
                f"parallel_materials must be an integer from 1 to {config.MAX_PARALLEL_MATERIALS}"
            )
        if chunk_minutes > 0:
            raise ValueError("parallel_materials requires chunk_minutes=0")
    selected = select_materials(profile, materials)
    with quiet():
        try:
            destination = resolve_target(target)
        except SystemExit as exc:
            raise ValueError(str(exc)) from exc
    with _operation(destination):
        submission = queue.start_queue(
            selected,
            quick=quick,
            workers=workers,
            chunk_minutes=chunk_minutes,
            parallel_materials=parallel_materials,
            no_sync=no_sync,
            dry_run=dry_run,
            catalog_profile=profile,
            _return_submission=True,
        )
    return RemoteJob(
        submission.jobid,
        destination,
        profile,
        tuple(selected),
        submission.stems,
        submission.script,
        submission.scheduler_id,
        dry_run,
    )
