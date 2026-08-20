"""Versioned HDF5 result encoding with permanent pickle compatibility.

New writes are HDF5 schema version 2 -- columnar record tables over a
content-addressed blob pool, encoded by :mod:`pyrite.checkpoints._result_v2` --
even though checkpoint path tokens retain their historical ``.pkl`` suffix.

:func:`load` sniffs bytes and permanently supports every earlier generation:
schema-version-1 HDF5 typed trees, and the zstd-framed, gzip-framed, and plain
pickles that preceded HDF5.

Nothing stored on disk is compressed. Deflate is measurably counterproductive
once array content is deduplicated, so the only compression is the whole-container
zstd frame :func:`dump_stream` puts on the wire for ``pyrite slim -o -``. See
``docs/repo-design/storage/result-schema.md``.
"""

import gzip
import io
import os
import pickle
import tempfile
from typing import IO, Any, cast

import h5py
import numpy as np

# stdlib since 3.14 (the version this project pins); ty's typeshed lags.
from compression import zstd

from . import _result_v2

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_GZIP_MAGIC = b"\x1f\x8b"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

SCHEMA = "pyrite.result"
SCHEMA_VERSION = _result_v2.SCHEMA_VERSION
#: Every container version this module can still read. Version 1 is read
#: forever; there is no flag day and no migration script.
READABLE_SCHEMA_VERSIONS = (1, SCHEMA_VERSION)
IDENTITY_VERSION = 1

# Retained as the public CLI compatibility range. It now selects the transfer
# frame's zstd level; stored artifacts are never compressed.
DEFAULT_LEVEL = 3
MAX_LEVEL = 19
LEVEL_RANGE = (1, 22)

_CHUNK = 1 << 20


def _read_node(node: h5py.Group | h5py.Dataset) -> Any:
    """Decode one schema-version-1 typed-tree node."""
    kind = node.attrs["kind"]
    if kind == "null":
        return None
    if kind == "mapping":
        return {
            _read_node(item["key"]): _read_node(item["value"]) for item in node["items"].values()
        }
    if kind in {"list", "tuple"}:
        values = [_read_node(item) for item in node["items"].values()]
        return tuple(values) if kind == "tuple" else values
    dataset = cast(h5py.Dataset, node)
    if kind == "string":
        return dataset.asstr()[()]
    if kind == "bytes":
        return dataset[...].tobytes()
    if kind == "unicode-array":
        return np.asarray(dataset.asstr()[...], dtype=str(dataset.attrs["numpy_dtype"]))
    if kind == "array":
        return dataset[...]
    if kind == "numpy-scalar":
        return dataset[()]
    if kind == "python-scalar":
        value = dataset[()]
        return value.item() if isinstance(value, np.generic) else value
    raise ValueError(f"unsupported result node kind: {kind!r}")


def _check_level(compresslevel: int | None) -> int:
    level = DEFAULT_LEVEL if compresslevel is None else compresslevel
    if not LEVEL_RANGE[0] <= level <= LEVEL_RANGE[1]:
        raise ValueError(f"compresslevel must be in {LEVEL_RANGE}, got {level}")
    return level


def _write_container(obj: Any, target: str, *, shuffle: bool) -> None:
    with h5py.File(target, "w") as h5:
        h5.attrs["schema"] = SCHEMA
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["identity_version"] = IDENTITY_VERSION
        _result_v2.write(h5, obj, shuffle=shuffle)


def _read_container(h5: h5py.File) -> Any:
    if h5.attrs.get("schema") != SCHEMA:
        raise ValueError(f"unsupported HDF5 result schema: {h5.attrs.get('schema')!r}")
    version = int(h5.attrs.get("schema_version", -1))
    if version == SCHEMA_VERSION:
        return _result_v2.read(h5)
    if version == 1:
        return _read_node(h5["value"])
    raise ValueError(f"unsupported result schema version: {version}")


def dump(obj: Any, path: str, *, compresslevel: int | None = None) -> None:
    """Encode ``obj`` as schema-version-2 HDF5 at ``path``.

    Callers continue to own atomic write-then-rename behavior. ``compresslevel``
    retains the historical 1--22 interface but no longer affects a stored
    artifact, which is written unfiltered; it selects the frame level in
    :func:`dump_stream`.
    """
    _check_level(compresslevel)
    _write_container(obj, path, shuffle=False)


def dump_stream(
    obj: Any, stream: IO[bytes] | io.RawIOBase, *, compresslevel: int | None = None
) -> None:
    """Encode ``obj`` to a binary stream, zstd-framed, leaving the stream open.

    HDF5 requires random-access output, so the container is staged on local disk
    and then compressed onto the caller's pipe. The frame is what makes a remote
    pull cheap: HDF5 object metadata is highly redundant, and the byte-transposed
    (``shuffle``) arrays it carries compress far better whole than per dataset.
    """
    level = _check_level(compresslevel)
    compressor = zstd.ZstdCompressor(level=min(level, LEVEL_RANGE[1]))
    with tempfile.TemporaryDirectory(prefix="pyrite-result-") as staging:
        staged = os.path.join(staging, "result.h5")
        _write_container(obj, staged, shuffle=True)
        with open(staged, "rb") as source:
            while chunk := source.read(_CHUNK):
                encoded = compressor.compress(chunk)
                if encoded:
                    stream.write(encoded)
    stream.write(compressor.flush())


def load(path: str) -> Any:
    """Load any generation: framed or bare HDF5, or any legacy pickle."""
    with open(path, "rb") as stream:
        head = stream.read(8)
    if head == _HDF5_MAGIC:
        with h5py.File(path, "r") as h5:
            return _read_container(h5)
    if head[:4] == _ZSTD_MAGIC:
        # Both a framed HDF5 artifact and the legacy zstd pickle start here.
        with zstd.ZstdFile(path, "rb") as stream:
            if stream.read(8) == _HDF5_MAGIC:
                stream.seek(0)
                payload = io.BytesIO(stream.read())
                with h5py.File(payload, "r") as h5:
                    return _read_container(h5)
            stream.seek(0)
            return pickle.load(stream)
    if head[:2] == _GZIP_MAGIC:
        with gzip.GzipFile(path, "rb") as stream:
            return pickle.load(stream)
    with open(path, "rb") as stream:
        return pickle.load(stream)
