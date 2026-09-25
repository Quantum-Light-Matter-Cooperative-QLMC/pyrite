"""Pinned external reference files fetched on demand for validation checks."""

import hashlib
from pathlib import Path
from urllib.request import urlopen


def _verified(data: bytes, sha256: str, origin: object) -> bytes:
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ValueError(f"{origin} has SHA-256 {digest}; expected {sha256}")
    return data


def read_pinned(
    path: Path,
    url: str,
    sha256: str,
    *,
    download: bool,
    missing: type[FileNotFoundError],
    hint: str,
) -> bytes:
    """Return the verified bytes at ``path``, fetching ``url`` first when allowed."""
    if not path.exists():
        if not download:
            raise missing(f"{path} not found; {hint}")
        with urlopen(url, timeout=60) as response:
            data = _verified(response.read(), sha256, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return _verified(path.read_bytes(), sha256, path)
