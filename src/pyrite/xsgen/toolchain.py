"""Find a Fortran compiler and build an external code, cached per source SHA.

gfortran is an optional *runtime* dependency (D9): absent on a fresh machine,
never installed by ``pip``, and never a reason to substitute a surrogate
model. Absence is therefore a normal, actionable condition, and the error
carries the exact compile command so a user can reproduce it by hand.

Builds are cached under :func:`pyrite.paths.cache_dir` keyed on the source
digest, so a tree that has not changed is compiled once per machine. The
digest covers only the files a program compiles, so patching a source
invalidates exactly the binaries built from it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..paths import cache_dir
from ._errors import BuildError, ToolchainUnavailableError
from .sources import ProgramSpec, ResolvedSource, source_digest

#: Compilers tried in order. ``gfortran`` is the documented one; the Intel and
#: LLVM drivers are accepted because a user who has one has a working Fortran
#: toolchain and no reason to install a second.
_CANDIDATES = ("gfortran", "ifx", "flang-new", "flang")

#: ``-std=legacy`` because ELSEPA and SBETHE are fixed-form F77-era sources
#: that modern gfortran rejects by default over obsolescent constructs, and
#: ``-fallow-argument-mismatch`` because both pass scalars where the dummy is
#: an array, which gfortran 10 turned from a warning into an error. Neither
#: changes generated code; both only unblock compilation of unmodified
#: upstream source.
_FLAGS = ("-O2", "-std=legacy", "-fallow-argument-mismatch")

#: Environment variable overriding compiler discovery, for a machine with a
#: compiler that is not on ``PATH`` under a known name.
_COMPILER_ENV = "PYRITE_XSGEN_FC"


@dataclass(frozen=True)
class Toolchain:
    """A usable Fortran compiler.

    Parameters
    ----------
    compiler
        Absolute path to the compiler driver.
    version
        First line of its ``--version`` output, recorded in every provenance
        manifest because compiler and flags are part of how a table was made.
    """

    compiler: Path
    version: str


def find_toolchain() -> Toolchain:
    """Locate a Fortran compiler.

    Returns
    -------
    Toolchain
        The first compiler found, honouring ``PYRITE_XSGEN_FC`` first.

    Raises
    ------
    ToolchainUnavailableError
        If no compiler is found, with install hints.
    """
    override = os.environ.get(_COMPILER_ENV)
    names = (override,) if override else _CANDIDATES
    for name in names:
        found = shutil.which(name)
        if found is None and override:
            candidate = Path(override).expanduser()
            found = str(candidate) if candidate.is_file() else None
        if found is None:
            continue
        return Toolchain(Path(found), _version_of(Path(found)))
    raise ToolchainUnavailableError(
        "no Fortran compiler found; PyRITE builds the external cross-section "
        f"codes from source and cannot substitute a model for them.\n"
        f"tried: {', '.join(name for name in names if name)}\n"
        "install one, for example:\n"
        "  Debian/Ubuntu:  sudo apt install gfortran\n"
        "  macOS:          brew install gcc\n"
        "  conda:          conda install -c conda-forge gfortran\n"
        f"or set {_COMPILER_ENV} to a compiler not on PATH."
    )


def _version_of(compiler: Path) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, no shell
            [str(compiler), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolchainUnavailableError(f"{compiler} is not runnable: {exc}") from exc
    if completed.returncode != 0:
        raise ToolchainUnavailableError(
            f"{compiler} --version exited {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    first, _, _ = completed.stdout.partition("\n")
    return first.strip()


def build_root() -> Path:
    """Return the directory holding every cached external-code binary."""
    return cache_dir() / "xsgen" / "build"


def build_command(
    source: ResolvedSource,
    program: ProgramSpec,
    toolchain: Toolchain,
    destination: Path,
) -> list[str]:
    """Return the exact argv used to build ``program``.

    Public because every failure message quotes it: a user who cannot build
    through PyRITE must be able to run the identical command themselves.
    """
    # ``-I`` at the tree root so Fortran ``INCLUDE`` lines resolve: ELSEPA's
    # ``elscata.f`` includes its two companion sources by bare name, and the
    # build runs from the cache directory, not the tree.
    return [
        str(toolchain.compiler),
        *_FLAGS,
        f"-I{source.root}",
        "-o",
        str(destination),
        *(str(source.root / name) for name in program.sources),
    ]


def build(
    source: ResolvedSource,
    program_name: str,
    *,
    toolchain: Toolchain | None = None,
    force: bool = False,
) -> Path:
    """Build one program from ``source``, reusing a cached binary when possible.

    Parameters
    ----------
    source
        Resolved code tree, from :func:`pyrite.xsgen.sources.resolve_source`.
    program_name
        Key into :attr:`CodeSpec.programs`.
    toolchain
        Compiler to use. Discovered when ``None``, which is the normal path;
        passing one avoids repeating discovery across a batch of builds.
    force
        Rebuild even when a cached binary for this source digest exists.

    Returns
    -------
    Path
        Absolute path to the built executable.

    Raises
    ------
    KeyError
        If ``program_name`` is not a program of this code.
    ToolchainUnavailableError
        If no Fortran compiler is available.
    BuildError
        If the compiler ran and failed.
    """
    try:
        program = source.spec.programs[program_name]
    except KeyError:
        known = ", ".join(source.spec.programs) or "(none: this code is read, not run)"
        raise KeyError(
            f"{source.spec.name} has no program {program_name!r}; known: {known}"
        ) from None

    digest = source_digest(source)
    destination = build_root() / f"{source.spec.name}-{digest[:16]}" / program.executable
    if destination.is_file() and not force:
        return destination

    resolved_toolchain = find_toolchain() if toolchain is None else toolchain
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = build_command(source, program, resolved_toolchain, destination)
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, no shell
            argv,
            cwd=destination.parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise BuildError(f"could not run the compiler:\n  {' '.join(argv)}\n{exc}") from exc
    if completed.returncode != 0 or not destination.is_file():
        # Leave no half-built binary behind: a zero-length or stale file at the
        # cache path would be served as a successful build by the next call.
        destination.unlink(missing_ok=True)
        raise BuildError(
            f"building {source.spec.name} {program.executable} failed "
            f"(exit {completed.returncode}).\ncommand:\n  {' '.join(argv)}\n"
            f"compiler: {resolved_toolchain.version}\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return destination


__all__ = [
    "Toolchain",
    "build",
    "build_command",
    "build_root",
    "find_toolchain",
]
