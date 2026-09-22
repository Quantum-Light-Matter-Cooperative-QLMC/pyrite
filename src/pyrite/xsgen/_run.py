"""Run an external program in an isolated scratch directory.

All three codes resolve their data paths and their output paths against the
*process working directory*, with fixed names: ELSEPA opens ``./database`` and
writes ``dcs_xpyyyezz.dat`` and ``dcs.dat``; SBETHE opens ``./sdbase`` and
writes ``stp.dat``, ``asymptotic.dat``, ``OOS.dat`` and a ``<mname>.mat``
cache; BREMS opens ``./V`` and ``./CS_int``.

So two concurrent runs sharing a directory overwrite each other's output. For
SBETHE it is worse than lost output: the ``.mat`` cache is read back *in
preference to* the interactive prompts whenever it exists, so a sibling run's
stale ``.mat`` silently overrides the material that was asked for and the run
returns confident numbers for the wrong material. Per-run isolation is a
correctness requirement, not a tidiness one.

Each run therefore gets a fresh directory, the code's data directories linked
into it, and ``cwd`` set to it. Linking rather than copying keeps ELSEPA's
4.6 MB ``database/`` and SBETHE's 18.4 MB ``sdbase/`` single-instanced across
runs; where the filesystem refuses a symlink the link degrades to a copy,
which the codes cannot distinguish since they only ever open paths beneath it.
"""

import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import cache_dir
from ._errors import RunError

#: How a data directory was made available inside a scratch directory. Recorded
#: so a run that fell back to copying is visible in diagnostics rather than
#: being an invisible performance cliff.
LinkMode = str


@dataclass(frozen=True)
class RunResult:
    """Outcome of one external-program invocation.

    Parameters
    ----------
    argv
        Exact command executed.
    returncode
        Process exit status.
    stdout, stderr
        Captured streams, decoded as text.
    outputs
        Contents of the requested output files. Bytes, not paths: the scratch
        directory is gone by the time this is returned, so a path would be a
        trap. Callers needing the files on disk use :func:`scratch_dir`.
    workdir
        The scratch directory the run used. Already removed unless the run
        failed under ``keep_on_failure``; retained here for messages.
    link_modes
        How each data directory was provided: ``"symlink"`` or ``"copy"``.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    outputs: Mapping[str, bytes]
    workdir: Path
    link_modes: Mapping[str, LinkMode] = field(default_factory=dict)

    def text(self, name: str, *, encoding: str = "latin-1") -> str:
        """Decode one collected output.

        ``latin-1`` by default because these are fixed-width Fortran text
        files: every byte maps to one character, so decoding can neither fail
        nor shift a column, which is what a parser reading by position needs.
        """
        return self.outputs[name].decode(encoding)


def scratch_root() -> Path:
    """Return the parent directory for external-code scratch directories."""
    return cache_dir() / "xsgen" / "scratch"


def _link_dir(source: Path, destination: Path) -> LinkMode:
    """Make ``source`` reachable at ``destination``, preferring a symlink.

    Returns which mechanism was used. The copy fallback exists for filesystems
    that refuse symlinks -- Windows without developer mode, and some network
    mounts. The codes only ever ``OPEN`` files beneath the directory, never
    inspect or follow the link itself, so a copy is behaviourally identical
    and differs only in cost.
    """
    try:
        destination.symlink_to(source, target_is_directory=True)
    except OSError, NotImplementedError:
        shutil.copytree(source, destination)
        return "copy"
    return "symlink"


def _populate(workdir: Path, data_dirs: Mapping[str, Path] | None) -> dict[str, LinkMode]:
    """Link every declared data directory into ``workdir``."""
    modes: dict[str, LinkMode] = {}
    for name, source in (data_dirs or {}).items():
        if not source.is_dir():
            raise RunError(
                f"data directory {name!r} is missing from the code tree "
                f"(expected {source}); fetch it with `pyrite tables fetch`"
            )
        modes[name] = _link_dir(source, workdir / name)
    return modes


@contextmanager
def scratch_dir(
    data_dirs: Mapping[str, Path] | None = None,
    *,
    keep: bool = False,
    keep_on_failure: bool = False,
    prefix: str = "run-",
) -> Iterator[Path]:
    """Yield a fresh working directory with ``data_dirs`` linked into it.

    Parameters
    ----------
    data_dirs
        Directories to make reachable by their key name inside the scratch
        directory, e.g. ``{"database": <tree>/database}``.
    keep
        Leave the directory in place unconditionally.
    keep_on_failure
        Leave the directory in place only when the body raises. For debugging
        a failed run, whose inputs and outputs are otherwise gone by the time
        the error surfaces -- while a successful run still cleans up, which
        ``keep`` alone would not do.
    prefix
        Directory-name prefix, for recognising leftovers.

    Raises
    ------
    RunError
        If a declared data directory does not exist. Letting the program
        start without it produces an opaque Fortran I/O error hundreds of
        lines later.
    """
    root = scratch_root()
    root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    try:
        _populate(workdir, data_dirs)
        yield workdir
    except BaseException:
        if not (keep or keep_on_failure):
            shutil.rmtree(workdir, ignore_errors=True)
        raise
    else:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)


def run_program(
    binary: Path,
    *,
    data_dirs: Mapping[str, Path] | None = None,
    stdin_text: str | None = None,
    input_files: Mapping[str, str] | None = None,
    outputs: Sequence[str] = (),
    timeout: float | None = None,
    keep_on_failure: bool = False,
    env: Mapping[str, str] | None = None,
) -> RunResult:
    """Run ``binary`` in its own scratch directory and read back its outputs.

    Parameters
    ----------
    binary
        Executable to run, typically from :func:`pyrite.xsgen.toolchain.build`.
    data_dirs
        Data directories to link in, keyed by the name the program opens.
    stdin_text
        Text fed to the program's standard input. ELSEPA reads a deck this
        way (``elscata < deck.in``); SBETHE is prompt-driven and reads its
        answers the same way.
    input_files
        Extra files written into the scratch directory before the run, mapped
        from name to content.
    outputs
        Fixed output names to collect. Each is read into memory before the
        scratch directory is removed, and a name the program did not write is
        an error: a code that exits 0 having produced nothing has failed.
    timeout
        Seconds before the program is killed. ``None`` waits indefinitely;
        BREMS runs take minutes, so no default is imposed here.
    keep_on_failure
        Preserve the scratch directory when the program fails, and name it in
        the error.
    env
        Replacement environment. ``None`` inherits the current one.

    Returns
    -------
    RunResult
        Captured streams and the collected output paths. The scratch
        directory is already removed on success unless an output is still
        needed -- use :func:`scratch_dir` directly for that case.

    Raises
    ------
    RunError
        If the program cannot be started, times out, or exits non-zero.
    """
    resolved_binary = Path(binary)
    if not resolved_binary.is_file():
        raise RunError(f"external program {resolved_binary} does not exist")

    argv = (str(resolved_binary),)
    link_modes: dict[str, LinkMode] = {}
    with scratch_dir(data_dirs, keep_on_failure=keep_on_failure) as workdir:
        for name in data_dirs or {}:
            link_modes[name] = "symlink" if (workdir / name).is_symlink() else "copy"
        for name, content in (input_files or {}).items():
            (workdir / name).write_text(content, encoding="utf-8")
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, no shell
                list(argv),
                cwd=workdir,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=None if env is None else dict(env),
            )
        except subprocess.TimeoutExpired as exc:
            raise RunError(
                f"{resolved_binary.name} exceeded its {timeout}s time limit"
                f"{_kept(workdir, keep_on_failure)}"
            ) from exc
        except OSError as exc:
            raise RunError(f"could not run {resolved_binary}: {exc}") from exc

        if completed.returncode != 0:
            raise RunError(
                f"{resolved_binary.name} exited {completed.returncode}"
                f"{_kept(workdir, keep_on_failure)}\n"
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        missing = [name for name in outputs if not (workdir / name).is_file()]
        if missing:
            raise RunError(
                f"{resolved_binary.name} exited 0 but wrote no "
                f"{', '.join(missing)}{_kept(workdir, keep_on_failure)}\n"
                f"{completed.stdout.strip()[-2000:]}"
            )
        # Read the outputs while the directory still exists. Returning paths
        # into a directory that is about to be removed would be a trap;
        # callers needing the raw files use `scratch_dir` directly.
        materialized = {name: (workdir / name).read_bytes() for name in outputs}
        return RunResult(
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            outputs=materialized,
            workdir=workdir,
            link_modes=link_modes,
        )


def _kept(workdir: Path, keep: bool) -> str:
    return f"; scratch directory kept at {workdir}" if keep else ""


__all__ = [
    "RunResult",
    "run_program",
    "scratch_dir",
    "scratch_root",
]
