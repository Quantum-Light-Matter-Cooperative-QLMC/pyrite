"""Fetch large, pinned external-code reference databases.

Only SBETHE needs this path. Its source is small enough to ship with PyRITE,
while its ``sdbase/`` directory is distributed inside a much larger upstream
archive. The installer downloads that immutable archive, verifies its
published SHA-256, and extracts only ``sdbase/`` into the user data directory.
"""

from __future__ import annotations

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
from urllib.request import urlopen

from ._errors import DataFetchError
from .sources import fetched_data_dir

SBETHE_ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/7zw25f428t/files/"
    "4fe30d66-1e6b-4de3-ac9c-a9f9cc9dae96/file_downloaded"
)
SBETHE_ARCHIVE_SHA256 = "693d447d7189dff05045cf9288ba4a1704e06747c4502a3572b35ff30b70845d"
SBETHE_DEPOSIT = "10.17632/7zw25f428t.1"
_CHUNK = 1 << 20
_REQUIRED_SBETHE_FILES = ("atparams.tab", "pdcompos.pen", "shparams.tab")


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


def _download(destination: Path) -> str:
    """Stream the pinned SBETHE archive to ``destination`` and return its digest."""
    digest = hashlib.sha256()
    try:
        with urlopen(SBETHE_ARCHIVE_URL, timeout=60) as response:  # noqa: S310 - pinned HTTPS URL
            with destination.open("wb") as stream:
                while chunk := response.read(_CHUNK):
                    stream.write(chunk)
                    digest.update(chunk)
    except (OSError, URLError) as exc:
        raise DataFetchError(
            f"could not download SBETHE data from {SBETHE_DEPOSIT}: {exc}"
        ) from exc
    return digest.hexdigest()


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


def fetch_sbethe() -> FetchResult:
    """Install SBETHE's pinned ``sdbase/`` into the user data directory.

    Existing complete installs are returned without network access. An
    incomplete directory is never overwritten automatically; it may contain
    user data, and replacing it would turn a repair command into a destructive
    one.
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
        archive = work / "sbethe.zip"
        actual_digest = _download(archive)
        if actual_digest != SBETHE_ARCHIVE_SHA256:
            raise DataFetchError(
                "SBETHE archive SHA-256 mismatch: "
                f"expected {SBETHE_ARCHIVE_SHA256}, received {actual_digest}; "
                "nothing was installed"
            )

        staged = work / "sdbase"
        staged.mkdir()
        file_count = _extract_sdbase(archive, staged)
        missing = [name for name in _REQUIRED_SBETHE_FILES if not (staged / name).is_file()]
        if missing:
            raise DataFetchError(
                "SBETHE archive is missing required sdbase files: " + ", ".join(missing)
            )
        marker = {
            "archive_sha256": actual_digest,
            "deposit": SBETHE_DEPOSIT,
            "file_count": file_count,
            "source_url": SBETHE_ARCHIVE_URL,
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


__all__ = [
    "FetchResult",
    "SBETHE_ARCHIVE_SHA256",
    "SBETHE_ARCHIVE_URL",
    "fetch_sbethe",
]
