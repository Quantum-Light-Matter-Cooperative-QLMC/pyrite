"""Fetch large, pinned external-code data into the user data directory.

Two installs share this path. SBETHE's source ships with PyRITE, while its
``sdbase/`` directory is distributed inside a much larger upstream archive:
the installer downloads that immutable archive, verifies its published
SHA-256, and extracts only ``sdbase/``. BremsLib-derived tables for the
catalogue elements are too large for the wheel, so the wheel pins their
release archive instead and the installer places each verified table into the
user table directory, where :func:`pyrite.xsgen.store.resolve` finds it.

Either archive may come from a local copy rather than the network, which is
how a cluster without outbound access is provisioned; the pinned digest is
checked the same way.
"""

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from ._errors import DataFetchError
from .bremslib.release import (
    ReleaseIndex,
    archive_member,
    file_sha256,
    load_release_index,
)
from .elsepa.release import ElsepaReleaseIndex
from .elsepa.release import load_release_index as load_elsepa_release_index
from .sources import fetched_data_dir
from .store import arrays_digest, manifest_digest, resolve, user_table_dir

SBETHE_ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/7zw25f428t/files/"
    "a7d2eed6-9aab-4462-936d-b15adf0e7c16/file_downloaded"
)
SBETHE_ARCHIVE_SHA256 = "d5d4879c2073ada3bd799fe0727054549cd6ec65cc699acbd0ff25fd5c3c4144"
SBETHE_DEPOSIT = "10.17632/7zw25f428t.2"
_CHUNK = 1 << 20
_USER_AGENT = "PyRITE xsgen (+https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite)"
# The shared (non per-element) sdbase files. `atparams.tab` and `exp-param.tab`
# are the decisive ones: `sbethe.f` OPENs both, and neither exists in deposit
# version 1, so an sdbase left over from that version is rejected rather than
# failing later inside the Fortran run.
_REQUIRED_SBETHE_FILES = (
    "atparams.tab",
    "exp-param.tab",
    "pdatconf.p14",
    "pdcompos.pen",
    "shparams.tab",
)


@dataclass(frozen=True)
class FetchResult:
    """Result of installing one external reference-data directory."""

    code: str
    path: Path
    archive_sha256: str
    file_count: int
    installed: bool


def _installed_sbethe(path: Path) -> bool:
    """Return whether ``path`` has the minimum complete-install markers."""
    return path.is_dir() and all((path / name).is_file() for name in _REQUIRED_SBETHE_FILES)


def _data_file_count(path: Path) -> int:
    """Count installed upstream files, excluding PyRITE's provenance marker."""
    return sum(
        candidate.is_file() and candidate.name != ".pyrite-fetch.json"
        for candidate in path.rglob("*")
    )


def _download(url: str, destination: Path, label: str) -> str:
    """Stream ``url`` to ``destination`` and return its SHA-256."""
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned HTTPS URL
            with destination.open("wb") as stream:
                while chunk := response.read(_CHUNK):
                    stream.write(chunk)
                    digest.update(chunk)
    except (OSError, URLError) as exc:
        raise DataFetchError(f"could not download {label}: {exc}") from exc
    return digest.hexdigest()


def _obtain(archive: Path | None, url: str | None, work: Path, label: str, expected: str) -> Path:
    """Return a verified archive: a local copy if given, else a download.

    The digest is checked before anything is extracted, whichever the
    source, so a local copy is exactly as trusted as the network.
    """
    if archive is not None:
        path = Path(archive)
        try:
            actual = file_sha256(path)
        except OSError as exc:
            raise DataFetchError(f"could not read {label} archive {path}: {exc}") from exc
    else:
        if url is None:
            raise DataFetchError(f"no download location is pinned for {label}")
        path = work / "archive.zip"
        actual = _download(url, path, label)
    if actual != expected:
        raise DataFetchError(
            f"{label} archive SHA-256 mismatch: expected {expected}, received {actual}; "
            "nothing was installed"
        )
    return path


def _extract_sdbase(archive: Path, destination: Path) -> int:
    """Extract regular files below the archive's ``sdbase/`` member only."""
    count = 0
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                parts = PurePosixPath(member.filename).parts
                if "sdbase" not in parts:
                    continue
                index = parts.index("sdbase")
                relative = parts[index + 1 :]
                if not relative or member.is_dir():
                    continue
                if any(part in {"", ".", ".."} for part in relative):
                    raise DataFetchError(
                        f"SBETHE archive contains an unsafe path: {member.filename}"
                    )
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise DataFetchError(
                        f"SBETHE archive contains an unsupported symlink: {member.filename}"
                    )
                target = destination.joinpath(*relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                count += 1
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DataFetchError(f"could not extract the SBETHE archive: {exc}") from exc
    return count


def fetch_sbethe(archive: str | Path | None = None) -> FetchResult:
    """Install SBETHE's pinned ``sdbase/`` into the user data directory.

    Existing complete installs are returned without network access. An
    incomplete directory is never overwritten automatically; it may contain
    user data, and replacing it would turn a repair command into a destructive
    one.

    Parameters
    ----------
    archive
        A local copy of the pinned upstream zip, used instead of downloading.
    """
    destination = fetched_data_dir("sbethe", "sdbase")
    if _installed_sbethe(destination):
        return FetchResult(
            code="sbethe",
            path=destination,
            archive_sha256=SBETHE_ARCHIVE_SHA256,
            file_count=_data_file_count(destination),
            installed=False,
        )
    if destination.exists():
        raise DataFetchError(
            f"incomplete SBETHE data directory already exists at {destination}; "
            "move or remove it, then run `pyrite tables fetch sbethe` again"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sbethe-fetch-", dir=destination.parent) as temp:
        work = Path(temp)
        bundle = _obtain(
            None if archive is None else Path(archive),
            SBETHE_ARCHIVE_URL,
            work,
            f"SBETHE data ({SBETHE_DEPOSIT})",
            SBETHE_ARCHIVE_SHA256,
        )

        staged = work / "sdbase"
        staged.mkdir()
        file_count = _extract_sdbase(bundle, staged)
        missing = [name for name in _REQUIRED_SBETHE_FILES if not (staged / name).is_file()]
        if missing:
            raise DataFetchError(
                "SBETHE archive is missing required sdbase files: " + ", ".join(missing)
            )
        marker = {
            "archive_sha256": SBETHE_ARCHIVE_SHA256,
            "deposit": SBETHE_DEPOSIT,
            "file_count": file_count,
            # Exactly one of these is set: where the verified bytes actually
            # came from. The digest alone cannot say whether they were
            # downloaded or copied in from a local archive.
            "source_url": SBETHE_ARCHIVE_URL if archive is None else None,
            "source_archive": None if archive is None else str(Path(archive).resolve()),
        }
        (staged / ".pyrite-fetch.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            os.replace(staged, destination)
        except OSError as exc:
            raise DataFetchError(f"could not install SBETHE data at {destination}: {exc}") from exc

    return FetchResult(
        code="sbethe",
        path=destination,
        archive_sha256=SBETHE_ARCHIVE_SHA256,
        file_count=file_count,
        installed=True,
    )


def _release_installed(entry, name: str, code: str, *, exact: bool) -> bool:
    """Return whether ``entry`` already resolves as the pinned table.

    With ``exact``, a table resolving under the pinned key with a *different*
    manifest is not treated as missing: replacing it could discard a table the
    user put there, so it is reported instead. Without it, any table under the
    key counts, for codes whose key already fixes the numbers.
    """
    table = resolve(entry.key)
    if table is None:
        return False
    if exact and table.digest != entry.manifest_sha256:
        raise DataFetchError(
            f"a different table is already stored under the released key for {entry.label} "
            f"({table.path}); move or remove it, then run `pyrite tables fetch {code}` again"
        )
    return True


def _stage_release_table(bundle: zipfile.ZipFile, entry, name: str, staged: Path) -> None:
    """Extract and verify one released table into ``staged``.

    Three checks, each against something the table cannot vouch for itself:
    the manifest digest against the pinned index, the recomputed manifest
    digest against the stored one, and the loaded arrays against the digest
    the manifest records. The archive digest already covers all of this for
    an honest release; these catch a release index and archive that disagree.
    """
    try:
        manifest_bytes = bundle.read(archive_member(entry.key, ".json"))
        payload_bytes = bundle.read(archive_member(entry.key, ".npz"))
    except KeyError as exc:
        raise DataFetchError(f"{name} table archive lacks the {entry.label} table") from exc
    body = json.loads(manifest_bytes)
    if (
        body.get("manifest_sha256") != entry.manifest_sha256
        or manifest_digest(body) != entry.manifest_sha256
    ):
        raise DataFetchError(
            f"{name} table manifest for {entry.label} does not match the pinned release"
        )
    payload = staged / f"{entry.key}.npz"
    payload.write_bytes(payload_bytes)
    with np.load(payload) as loaded:
        stored = arrays_digest({name: loaded[name] for name in loaded.files})
    if stored != body.get("arrays_sha256"):
        raise DataFetchError(f"{name} table arrays for {entry.label} do not match their manifest")
    (staged / f"{entry.key}.json").write_bytes(manifest_bytes)


def _install_release(
    code: str, name: str, pinned, archive: str | Path | None, *, exact: bool
) -> FetchResult:
    """Install every missing table of a pinned release into the user table directory.

    Nothing is installed unless every missing table verifies, so a failed
    fetch leaves no partial release behind.
    """
    destination = user_table_dir()
    missing = [
        entry for entry in pinned.tables if not _release_installed(entry, name, code, exact=exact)
    ]
    if not missing:
        return FetchResult(
            code=code,
            path=destination,
            archive_sha256=pinned.archive_sha256,
            file_count=len(pinned.tables),
            installed=False,
        )
    if archive is None and pinned.url is None:
        raise DataFetchError(
            f"the pinned {name} table release is not published for download yet; "
            "install from a copy of the release archive with "
            f"`pyrite tables fetch {code} --archive PATH` (SHA-256 {pinned.archive_sha256})"
        )

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{code}-fetch-", dir=destination) as temp:
        work = Path(temp)
        label = f"{name} tables ({pinned.upstream})"
        bundle_path = _obtain(
            None if archive is None else Path(archive),
            pinned.url,
            work,
            label,
            pinned.archive_sha256,
        )
        staged = work / "staged"
        staged.mkdir()
        try:
            with zipfile.ZipFile(bundle_path) as bundle:
                for entry in missing:
                    _stage_release_table(bundle, entry, name, staged)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise DataFetchError(f"could not extract the {name} table archive: {exc}") from exc
        try:
            # Payload before manifest: :func:`resolve` requires both, so a
            # table interrupted between the two moves is simply not found.
            for entry in missing:
                for suffix in (".npz", ".json"):
                    file_name = f"{entry.key}{suffix}"
                    os.replace(staged / file_name, destination / file_name)
        except OSError as exc:
            raise DataFetchError(
                f"could not install {name} tables at {destination}: {exc}"
            ) from exc

    return FetchResult(
        code=code,
        path=destination,
        archive_sha256=pinned.archive_sha256,
        file_count=len(pinned.tables),
        installed=True,
    )


def fetch_bremslib(
    archive: str | Path | None = None, *, index: ReleaseIndex | None = None
) -> FetchResult:
    """Install the pinned BremsLib-derived tables into the user table directory.

    Tables already present are left alone and a complete install returns
    without network access. Nothing is installed unless every missing table
    verifies, so a failed fetch leaves no partial release behind.

    Parameters
    ----------
    archive
        A local copy of the release zip, used instead of downloading.
    index
        Release index to install. Defaults to the one shipped with PyRITE.

    Raises
    ------
    DataFetchError
        If this build pins no release, the archive cannot be obtained or does
        not verify, or a different table already holds a released key.
    """
    pinned = load_release_index() if index is None else index
    if pinned is None:
        raise DataFetchError(
            "this PyRITE build pins no BremsLib table release; generate tables from a "
            "BremsLib checkout with `pyrite tables generate --code bremslib`"
        )
    return _install_release("bremslib", "BremsLib", pinned, archive, exact=True)


def fetch_elsepa(
    archive: str | Path | None = None, *, index: ElsepaReleaseIndex | None = None
) -> FetchResult:
    """Install the pinned ELSEPA elastic tables into the user table directory.

    As :func:`fetch_bremslib`, except that a table already stored under a
    released key counts as installed even when its manifest differs: an ELSEPA
    key fixes the vendored source and the deck, so a locally generated table
    under it is an equally valid copy (it differs only in timestamp and
    compiler).

    Raises
    ------
    DataFetchError
        If this build pins no release, or the archive cannot be obtained or
        does not verify.
    """
    pinned = load_elsepa_release_index() if index is None else index
    if pinned is None:
        raise DataFetchError(
            "this PyRITE build pins no ELSEPA table release; generate tables with "
            "`pyrite tables generate --code elsepa --material NAME`"
        )
    return _install_release("elsepa", "ELSEPA", pinned, archive, exact=False)


__all__ = [
    "FetchResult",
    "SBETHE_ARCHIVE_SHA256",
    "SBETHE_ARCHIVE_URL",
    "fetch_bremslib",
    "fetch_elsepa",
    "fetch_sbethe",
]
