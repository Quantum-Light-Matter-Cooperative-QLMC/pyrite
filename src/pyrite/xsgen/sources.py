"""Resolve a usable source tree for each external code, and digest it.

Three codes, three ways their tree can arrive, and one rule for choosing
between them. A tree is either *vendored* inside the PyRITE distribution
(ELSEPA and ``sbethe.f``, per D4), *configured* by the user through the
context store, or a *conventional sibling checkout* beside the PyRITE
checkout. BremsLib is only ever the latter two: its sources are GPL-3 and its
library is 810 MB, so neither is redistributed.

Resolution order is explicit override, then configured path, then vendored
tree, then sibling checkout. Configured beats vendored so a user testing a
patched upstream is not silently served the packaged copy; vendored beats the
sibling so an installed wheel with no checkout anywhere still works offline,
which is what the ``pyrite remote`` cluster workflow needs.

When nothing resolves this raises rather than degrading. D9 forbids a silent
fallback to a surrogate model, so the failure has to name every location
tried, the config key that overrides them, and the upstream deposit.

The *code version* in a table key (D2) is a digest of the Fortran sources, not
a version string: upstream ships no version number in the source, users patch
trees, and a table generated from patched sources is a different table.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..console import config as _config
from ..paths import data_dir, user_data_dir
from ._errors import SourceUnavailableError

#: Chunk size for hashing Fortran sources. They are small (the largest is
#: ELSEPA's 1.1 MB ``elsepa2020.f``), so this only bounds peak memory.
_DIGEST_CHUNK = 1 << 20


@dataclass(frozen=True)
class ProgramSpec:
    """One buildable program within a code tree.

    Parameters
    ----------
    executable
        Basename of the built binary, without any platform suffix.
    sources
        Source file names relative to the tree root, in the order the compiler
        must see them.
    """

    executable: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class CodeSpec:
    """Everything about one external code that does not depend on its location.

    Parameters
    ----------
    name
        Registry key, and the value stored under ``"code"`` in a table key.
    sibling
        Directory name of the conventional checkout beside the PyRITE
        checkout.
    config_key
        Context-store key that overrides every other location.
    markers
        Paths, relative to a candidate directory, of which at least one must
        exist for that directory to be accepted as this code's tree. Guards
        against a configured path pointing at an empty or wrong directory,
        which would otherwise fail much later with a confusing compiler
        error. A pattern containing ``*`` is globbed, because BremsLib's
        library directory carries its own version in its name; several
        patterns are accepted because a deposit can be unpacked either as a
        whole or as the one directory PyRITE reads.
    digest_sources
        Source files whose bytes define the code version. Data directories are
        excluded: they are large, and upstream pins them by deposit DOI.
    data_dirs
        Directories the program resolves relative to its working directory.
        :mod:`pyrite.xsgen._run` links these into each scratch directory.
    programs
        Buildable programs, keyed by the name PyRITE uses to request them.
        Empty for BremsLib, which PyRITE reads rather than runs (D7).
    upstream
        Human-readable upstream location, quoted in the unavailable-source
        error.
    """

    name: str
    sibling: str
    config_key: str
    markers: tuple[str, ...]
    digest_sources: tuple[str, ...]
    data_dirs: tuple[str, ...]
    programs: Mapping[str, ProgramSpec]
    upstream: str
    vendored: bool


#: ``elscata`` only. ``elscatm`` is the molecular independent-atom path, which
#: the spec rejects for crystals: it sums atomic amplitudes coherently over a
#: randomly oriented molecule, which on a crystal cluster produces diffraction
#: that PyRITE already models separately and would double-count.
_ELSEPA = CodeSpec(
    name="elsepa",
    sibling="elsepa-2020",
    config_key="xsgen.elsepa_source",
    markers=("elscata.f",),
    digest_sources=("elscata.f", "elsepa2020.f", "radial.f"),
    data_dirs=("database",),
    # ``elscata.f`` opens with ``INCLUDE 'radial.f'`` and ``INCLUDE
    # 'elsepa2020.f'``, so it is compiled *alone*: naming all three on the
    # command line compiles the included bodies twice and the link fails on
    # dozens of duplicate symbols. The other two still appear in
    # ``digest_sources`` -- they are part of the binary, so patching one is a
    # different code even though the compiler is never pointed at them.
    programs={"elscata": ProgramSpec(executable="elscata", sources=("elscata.f",))},
    upstream="ELSEPA 2020, Mendeley Data doi:10.17632/w4hm5vymym.1",
    vendored=True,
)

_SBETHE = CodeSpec(
    name="sbethe",
    sibling="sbethe",
    config_key="xsgen.sbethe_source",
    markers=("sbethe.f",),
    digest_sources=("sbethe.f",),
    data_dirs=("sdbase",),
    programs={"sbethe": ProgramSpec(executable="sbethe", sources=("sbethe.f",))},
    upstream="SBETHE, Mendeley Data doi:10.17632/7zw25f428t.2",
    vendored=True,
)

#: No programs: PyRITE reads the precomputed library in place (D7). The
#: markers name a library *data* file rather than a source file, because the
#: GPL-3 sources are exactly what PyRITE does not need -- and because a tree
#: holding the codes but not the 810 MB library cannot answer a single
#: request. The deposit unpacks to a root holding ``BremsLib_v2.0.<patch>/``
#: beside the two code folders, and the directory name carries the patch
#: version, so the first pattern is globbed; the second accepts a checkout
#: pointed straight at the library directory.
_BREMSLIB = CodeSpec(
    name="bremslib",
    sibling="BremsLib_v2.0.8",
    config_key="xsgen.bremslib_source",
    markers=("BremsLib_v2.0*/SDCS/SDCS_*.txt", "SDCS/SDCS_*.txt"),
    digest_sources=(),
    data_dirs=(),
    programs={},
    upstream="BremsLib v2.0.8, Mendeley Data doi:10.17632/6zfsc9xsz8.9",
    vendored=False,
)

_CODES: Mapping[str, CodeSpec] = {spec.name: spec for spec in (_ELSEPA, _SBETHE, _BREMSLIB)}


def code_names() -> tuple[str, ...]:
    """Return every known code name, in stable display order."""
    return tuple(_CODES)


def code_spec(code: str) -> CodeSpec:
    """Return the :class:`CodeSpec` for ``code``.

    Raises
    ------
    KeyError
        If ``code`` is not one of :func:`code_names`.
    """
    try:
        return _CODES[code]
    except KeyError:
        known = ", ".join(code_names())
        raise KeyError(f"unknown external code {code!r}; known codes: {known}") from None


def vendored_root(code: str) -> Path:
    """Return where a vendored tree for ``code`` would live.

    The path is returned whether or not it exists, so callers can quote it in
    an error message.
    """
    return data_dir() / "xsgen" / code_spec(code).name


def fetched_data_dir(code: str, name: str) -> Path:
    """Return the user-data location for a fetched code data directory."""
    spec = code_spec(code)
    if name not in spec.data_dirs:
        known = ", ".join(spec.data_dirs) or "none"
        raise KeyError(f"{code!r} has no data directory {name!r}; known: {known}")
    return user_data_dir() / "xsgen" / "reference-data" / spec.name / name


def _accepts(spec: CodeSpec, candidate: Path) -> bool:
    return any(_marker_path(candidate, pattern) is not None for pattern in spec.markers)


def _marker_path(candidate: Path, pattern: str) -> Path | None:
    """Return the path ``pattern`` names under ``candidate``, or ``None``.

    Globbed only when the pattern needs it: ``Path.glob`` on a literal path
    still scans the parent directory, which for BremsLib's 86300-file
    ``DDCS/`` is the difference between one ``stat`` and a full listing.
    """
    if "*" not in pattern:
        direct = candidate / pattern
        return direct if direct.exists() else None
    return next((match for match in candidate.glob(pattern)), None)


@dataclass(frozen=True)
class ResolvedSource:
    """A usable source tree, and where it came from.

    Parameters
    ----------
    spec
        The code this tree provides.
    root
        Absolute path to the tree root.
    origin
        Short description of which resolution tier won, for display and for
        the provenance manifest.
    """

    spec: CodeSpec
    root: Path
    origin: str

    @property
    def data_dirs(self) -> dict[str, Path]:
        """Map required data names to tree-local or fetched directories."""
        resolved: dict[str, Path] = {}
        for name in self.spec.data_dirs:
            local = self.root / name
            fetched = fetched_data_dir(self.spec.name, name)
            resolved[name] = local if local.is_dir() else fetched
        return resolved


def resolve_source(code: str, override: str | Path | None = None) -> ResolvedSource:
    """Resolve the source tree for ``code``.

    Parameters
    ----------
    code
        One of :func:`code_names`.
    override
        Explicit tree root, beating every other tier. A path that does not
        hold the code raises rather than falling through, because an explicit
        path that is silently ignored is worse than an error.

    Returns
    -------
    ResolvedSource
        The winning tree and the tier it came from.

    Raises
    ------
    SourceUnavailableError
        If no tier yields a directory holding one of the code's markers.
    """
    spec = code_spec(code)
    tried: list[str] = []

    if override is not None:
        candidate = Path(override).expanduser()
        if _accepts(spec, candidate):
            return ResolvedSource(spec, candidate.resolve(), "explicit path")
        raise SourceUnavailableError(_unavailable_message(spec, [f"{candidate} (explicit)"]))

    resolved = _config.resolve(spec.config_key)
    if resolved.source != "built-in default":
        candidate = Path(resolved.value).expanduser()
        if _accepts(spec, candidate):
            return ResolvedSource(spec, candidate.resolve(), resolved.source)
        tried.append(f"{candidate} (from {resolved.source})")

    if spec.vendored:
        candidate = vendored_root(spec.name)
        if _accepts(spec, candidate):
            return ResolvedSource(spec, candidate.resolve(), "vendored")
        tried.append(f"{candidate} (vendored)")

    candidate = Path(resolved.value).expanduser()
    if _accepts(spec, candidate):
        return ResolvedSource(spec, candidate.resolve(), "sibling checkout")
    tried.append(f"{candidate} (sibling checkout)")

    raise SourceUnavailableError(_unavailable_message(spec, tried))


def _unavailable_message(spec: CodeSpec, tried: Sequence[str]) -> str:
    locations = "\n".join(f"  - {entry}" for entry in tried)
    return (
        f"no {spec.name} source tree found (looked for {_markers_text(spec)} in):\n"
        f"{locations}\n"
        f"obtain it from {spec.upstream}, then point PyRITE at it with:\n"
        f"  pyrite tables sources set {spec.name} <path>\n"
        f"(equivalently: pyrite config set {spec.config_key} <path>)"
    )


def _markers_text(spec: CodeSpec) -> str:
    """Render a code's markers for an error message."""
    return " or ".join(repr(pattern) for pattern in spec.markers)


def validate_source_path(code: str, value: str | Path) -> Path:
    """Return ``value`` resolved, having checked it holds ``code``.

    Used wherever a user names a code tree -- ``pyrite tables sources set`` and
    ``pyrite config set xsgen.*_source`` -- so both reject the same paths with
    the same message.

    The path is resolved rather than stored as typed: a stored value outlives
    the directory it was typed in, and a relative path means something
    different from every other working directory.

    Raises
    ------
    SourceUnavailableError
        If the path does not hold one of the code's markers.
    KeyError
        If ``code`` is unknown.
    """
    spec = code_spec(code)
    resolved = Path(value).expanduser().resolve()
    if not _accepts(spec, resolved):
        raise SourceUnavailableError(
            f"{resolved} does not look like a {spec.name} tree: no {_markers_text(spec)} in it"
        )
    return resolved


def source_digest(source: ResolvedSource) -> str:
    """Return the SHA-256 code version for a resolved tree.

    Hashes each file in :attr:`CodeSpec.digest_sources` under its own name, so
    reordering the tuple does not change the digest but editing any source
    does. A code with no digest sources -- BremsLib, which PyRITE does not
    build -- digests to the hash of its empty file list, which is a constant
    and fine: its tables are keyed on the library deposit version instead.

    Raises
    ------
    SourceUnavailableError
        If a declared source file is missing from the tree.
    """
    digest = hashlib.sha256()
    for name in sorted(source.spec.digest_sources):
        path = source.root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(_DIGEST_CHUNK):
                    digest.update(chunk)
        except OSError as exc:
            raise SourceUnavailableError(
                f"{source.spec.name} tree at {source.root} is missing {name}: {exc}"
            ) from exc
        digest.update(b"\0")
    return digest.hexdigest()


def missing_data_dirs(source: ResolvedSource) -> tuple[str, ...]:
    """Return the names of required data directories absent from the tree.

    Separate from :func:`resolve_source` because the two failures have
    different remedies: a missing tree is fetched or configured, while a
    missing ``sdbase/`` is downloaded by ``pyrite tables fetch sbethe``.
    """
    return tuple(name for name, path in source.data_dirs.items() if not path.is_dir())


def iter_sources(codes: Iterable[str] | None = None) -> list[tuple[str, ResolvedSource | str]]:
    """Resolve several codes, reporting failures instead of raising.

    Returns one entry per code: the :class:`ResolvedSource`, or the message
    explaining why it did not resolve. ``pyrite tables sources`` shows the
    whole picture, so one missing tree must not hide the others.
    """
    report: list[tuple[str, ResolvedSource | str]] = []
    for name in code_names() if codes is None else codes:
        try:
            report.append((name, resolve_source(name)))
        except SourceUnavailableError as exc:
            report.append((name, str(exc)))
    return report


__all__ = [
    "CodeSpec",
    "ProgramSpec",
    "ResolvedSource",
    "code_names",
    "code_spec",
    "fetched_data_dir",
    "iter_sources",
    "missing_data_dirs",
    "resolve_source",
    "source_digest",
    "validate_source_path",
    "vendored_root",
]
