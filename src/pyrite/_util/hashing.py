"""Streamed file digests and canonical-JSON identity digests."""

import hashlib
import json
import os
from typing import Any

_BLOCK_BYTES = 1 << 20


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the hex SHA-256 of a file's bytes, read in bounded blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any, *, allow_nan: bool = False) -> bytes:
    """Compact, sorted-key, ASCII-escaped JSON bytes of ``payload``.

    These bytes are hashed into persisted identities: changing the separators,
    key order, or escaping here changes every digest built on them.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=allow_nan).encode()


def canonical_json_digest(payload: Any, *, allow_nan: bool = False) -> str:
    """Hex SHA-256 of :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(payload, allow_nan=allow_nan)).hexdigest()
