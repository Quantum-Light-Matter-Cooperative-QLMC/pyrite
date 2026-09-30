"""Hash-pinned upstream datasets installed once into the workspace data root.

ADR-0014 class (b): large upstream data that is pinned by SHA-256 in the wheel
but not shipped in it. ``pyrite tables fetch eedl`` (or ``eadl``) downloads the
published file from its pinned URL, or copies a local file given with
``--archive PATH``, verifies the digest, and installs it at
``<data root>/datasets/<name>/<file>``. The data root is the same one generated
xsgen tables use: the platform user data directory by default, or an explicit
workspace (``PYRITE_HOME`` / ``workspace.root``) when one is selected, which is
how the remote box (``PYRITE_HOME=<remote dir>``) finds its synced copy.

A missing dataset is an error naming the fetch command. Nothing here falls back
to other data, and nothing downloads implicitly at run time: a compute node may
have no network, so installation stays an explicit step.
"""

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 1 << 20


@dataclass(frozen=True)
class Dataset:
    """One pinned upstream file.

    Parameters
    ----------
    name
        Short name used by ``pyrite tables fetch NAME`` and the install path.
    filename
        Installed file name, which the loaders open.
    url
        Pinned HTTPS download location.
    sha256
        SHA-256 of the published bytes. Loaders and ``pyrite tables verify``
        check it; it also enters the model identity markers.
    description
        One line for help and diagnostics.
    download_sha256
        SHA-256 of the bytes the URL serves, when PyRITE's installed bytes are
        a declared transformation of them; ``None`` when they are identical.
    strip_suffix
        The transformation: bytes removed from the end of the download.
    """

    name: str
    filename: str
    url: str
    sha256: str
    description: str
    download_sha256: str | None = None
    strip_suffix: bytes = b""


#: EPICS2025 EEDL (NDS-IAEA-226), electroionization and bremsstrahlung.
EEDL = Dataset(
    name="eedl",
    filename="EEDL.endf",
    url="https://nuclear.llnl.gov/EPICS/ENDF2025/EEDL2025.ALL",
    sha256="f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c",
    description="EPICS2025 EEDL electron data (25 MB): ionization and bremsstrahlung",
    # The published file ends with one more CRLF than the bytes PyRITE vetted
    # and packaged until #263. The pin (and the model identity markers built
    # from it) keep the vetted bytes, so the fetch drops that final CRLF; the
    # ENDF records are unchanged.
    download_sha256="ce37912435e0b8002f85878f98ccf7c5840cb168f1d46af3c9e915cd16c70ccc",
    strip_suffix=b"\r\n",
)
#: EPICS2025 EADL (NDS-IAEA-224), atomic relaxation.
EADL = Dataset(
    name="eadl",
    filename="EADL2025.ALL",
    url="https://nuclear.llnl.gov/EPICS/ENDF2025/EADL2025.ALL",
    sha256="78ccf8a4e07c1c120a2e3d94ff051aab2180d151f35e8bc3406d52df5af5e88c",
    description="EPICS2025 EADL atomic relaxation data (8 MB)",
)
#: Every fetched dataset, in report order.
DATASETS: dict[str, Dataset] = {dataset.name: dataset for dataset in (EEDL, EADL)}

#: Outcomes of :func:`verify_dataset`; ``ok`` is the only passing one.
OK = "ok"
MISSING = "missing"
MISMATCH = "mismatch"


class DatasetNotFoundError(FileNotFoundError):
    """A required pinned dataset is not installed; the message names the fix."""


class DatasetMismatchError(ValueError):
    """An installed dataset does not match its pinned SHA-256."""


@dataclass(frozen=True)
class DatasetFetchResult:
    """Outcome of :func:`fetch_dataset`."""

    name: str
    path: Path
    sha256: str
    installed: bool
    source: str | None


def get(name: str) -> Dataset:
    """Return the pinned dataset called ``name``."""
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(DATASETS)
        raise KeyError(f"unknown dataset {name!r}; choose from {known}") from None


def datasets_dir() -> Path:
    """Return the directory fetched datasets install into."""
    from .console.config import fetched_data_root

    return fetched_data_root() / "datasets"


def dataset_path(name: str) -> Path:
    """Return where dataset ``name`` is installed, whether or not it exists."""
    dataset = get(name)
    return datasets_dir() / dataset.name / dataset.filename


def fetch_hint(name: str) -> str:
    """Return the command that installs ``name``, for error messages."""
    return (
        f"`pyrite tables fetch {name}` (downloads the pinned file), or "
        f"`pyrite tables fetch {name} --archive PATH` with a local copy; "
        "on the remote box, `pyrite remote sync` from a machine that has it"
    )


def require_dataset(name: str) -> Path:
    """Return the installed path of ``name``, raising when it is absent.

    Existence only; the loaders verify the digest before parsing.

    Raises
    ------
    DatasetNotFoundError
        Naming the expected path and the fetch command.
    """
    path = dataset_path(name)
    if not path.is_file():
        dataset = get(name)
        raise DatasetNotFoundError(
            f"the pinned {dataset.description} is not installed: expected {path}. "
            f"Install it with {fetch_hint(name)}."
        )
    return path


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of one file, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(name: str) -> str:
    """Return :data:`OK`, :data:`MISSING`, or :data:`MISMATCH` for ``name``.

    Hashes the whole file (about 0.1 s for EEDL), so a job preflight checks
    content rather than presence.
    """
    path = dataset_path(name)
    if not path.is_file():
        return MISSING
    return OK if file_sha256(path) == get(name).sha256 else MISMATCH


def require_verified(name: str) -> Path:
    """Return the installed path of ``name`` after checking its SHA-256.

    Raises
    ------
    DatasetNotFoundError
        If it is not installed.
    DatasetMismatchError
        If the installed bytes differ from the pin.
    """
    path = require_dataset(name)
    actual = file_sha256(path)
    expected = get(name).sha256
    if actual != expected:
        raise DatasetMismatchError(
            f"{path} does not match the pinned {name} SHA-256: expected {expected}, "
            f"got {actual}. Remove it and run {fetch_hint(name)}."
        )
    return path


def _strip_suffix(path: Path, suffix: bytes) -> str:
    """Drop ``suffix`` from the end of ``path`` and return the new SHA-256."""
    size = path.stat().st_size
    with path.open("r+b") as stream:
        if suffix:
            stream.seek(size - len(suffix))
            if stream.read() != suffix:
                return ""
            stream.truncate(size - len(suffix))
    return file_sha256(path)


def fetch_dataset(name: str, source: str | Path | None = None) -> DatasetFetchResult:
    """Install dataset ``name`` from its pinned URL or a local copy.

    Either the published bytes (``download_sha256``, then the declared
    ``strip_suffix`` transformation) or bytes already in the pinned form are
    accepted. An installed file that already matches the pin returns without
    network access. An installed file that does not match is left alone and reported:
    replacing it would turn a repair command into a destructive one. The new
    file is verified before it is moved into place, so a failed fetch leaves
    nothing behind.

    Parameters
    ----------
    name
        Dataset name, e.g. ``"eedl"``.
    source
        A local copy of the pinned file, used instead of downloading.

    Raises
    ------
    pyrite.xsgen.DataFetchError
        If the file cannot be obtained, does not verify, or a different file
        occupies the install path.
    """
    from .xsgen._errors import DataFetchError
    from .xsgen.fetch import _download

    dataset = get(name)
    destination = dataset_path(name)
    if destination.is_file():
        if file_sha256(destination) == dataset.sha256:
            return DatasetFetchResult(name, destination, dataset.sha256, False, None)
        raise DataFetchError(
            f"{destination} exists but does not match the pinned {name} SHA-256; "
            f"move or remove it, then run `pyrite tables fetch {name}` again"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-fetch-", dir=destination.parent) as temp:
        staged = Path(temp) / dataset.filename
        if source is not None:
            origin = str(Path(source).resolve())
            try:
                shutil.copyfile(source, staged)
            except OSError as exc:
                raise DataFetchError(f"could not read {name} file {source}: {exc}") from exc
            actual = file_sha256(staged)
        else:
            origin = dataset.url
            actual = _download(dataset.url, staged, f"{name} ({dataset.url})")
        if actual != dataset.sha256 and actual == dataset.download_sha256:
            actual = _strip_suffix(staged, dataset.strip_suffix)
        if actual != dataset.sha256:
            expected = " or ".join(
                digest for digest in (dataset.sha256, dataset.download_sha256) if digest
            )
            raise DataFetchError(
                f"{name} SHA-256 mismatch: expected {expected}, received {actual} "
                f"from {origin}; nothing was installed"
            )
        try:
            os.replace(staged, destination)
        except OSError as exc:
            raise DataFetchError(f"could not install {name} at {destination}: {exc}") from exc
    return DatasetFetchResult(name, destination, dataset.sha256, True, origin)


__all__ = [
    "DATASETS",
    "EADL",
    "EEDL",
    "MISMATCH",
    "MISSING",
    "OK",
    "Dataset",
    "DatasetFetchResult",
    "DatasetMismatchError",
    "DatasetNotFoundError",
    "dataset_path",
    "datasets_dir",
    "fetch_dataset",
    "fetch_hint",
    "file_sha256",
    "get",
    "require_dataset",
    "require_verified",
    "verify_dataset",
]
