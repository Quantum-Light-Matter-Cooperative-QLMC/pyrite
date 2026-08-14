"""Versioned HDF5 result encoding with permanent pickle compatibility.

New writes are HDF5 even though checkpoint path tokens retain their historical
``.pkl`` suffix.  :func:`load` sniffs bytes and permanently supports the three
older pickle generations: zstd-framed, gzip-framed, and plain.
"""

import gzip
import io
import pickle
import tempfile
from collections.abc import Mapping
from typing import IO, Any, cast

import h5py
import numpy as np

# stdlib since 3.14 (the version this project pins); ty's typeshed lags.
from compression import zstd  # ty: ignore[unresolved-import]

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_GZIP_MAGIC = b"\x1f\x8b"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

SCHEMA = "pyrite.result"
SCHEMA_VERSION = 1
IDENTITY_VERSION = 1

# Retained as the public CLI compatibility range. HDF5's portable gzip filter
# has levels 1--9, so values above 9 request level 9.
DEFAULT_LEVEL = 3
MAX_LEVEL = 19
LEVEL_RANGE = (1, 22)

_SPOOL_LIMIT = 64 << 20


def _compression_kwargs(value: np.ndarray, level: int) -> dict[str, Any]:
    if value.ndim == 0 or value.size < 2 or value.dtype.kind not in "biufc":
        return {"track_times": False}
    return {
        "compression": "gzip",
        "compression_opts": min(level, 9),
        "shuffle": True,
        "track_times": False,
    }


def _write_node(parent: h5py.Group, name: str, value: Any, level: int) -> None:
    if value is None:
        node = parent.create_group(name, track_order=True)
        node.attrs["kind"] = "null"
        return

    if isinstance(value, Mapping):
        node = parent.create_group(name, track_order=True)
        node.attrs["kind"] = "mapping"
        items = node.create_group("items", track_order=True)
        for index, (key, item_value) in enumerate(value.items()):
            item = items.create_group(f"{index:08d}", track_order=True)
            _write_node(item, "key", key, level)
            _write_node(item, "value", item_value, level)
        return

    if isinstance(value, (list, tuple)):
        node = parent.create_group(name, track_order=True)
        node.attrs["kind"] = "tuple" if isinstance(value, tuple) else "list"
        items = node.create_group("items", track_order=True)
        for index, item_value in enumerate(value):
            _write_node(items, f"{index:08d}", item_value, level)
        return

    if isinstance(value, str):
        node = parent.create_dataset(
            name,
            data=value,
            dtype=h5py.string_dtype(encoding="utf-8"),
            track_times=False,
        )
        node.attrs["kind"] = "string"
        return

    if isinstance(value, bytes):
        node = parent.create_dataset(
            name, data=np.frombuffer(value, dtype=np.uint8), track_times=False
        )
        node.attrs["kind"] = "bytes"
        return

    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError("object-dtype arrays are not supported by the result schema")
        if value.dtype.kind == "U":
            encoded = value.astype(h5py.string_dtype(encoding="utf-8"))
            node = parent.create_dataset(name, data=encoded, track_times=False)
            node.attrs["kind"] = "unicode-array"
            node.attrs["numpy_dtype"] = value.dtype.str
            return
        node = parent.create_dataset(name, data=value, **_compression_kwargs(value, level))
        node.attrs["kind"] = "array"
        return

    if isinstance(value, np.generic):
        node = parent.create_dataset(name, data=value, track_times=False)
        node.attrs["kind"] = "numpy-scalar"
        return

    if isinstance(value, (bool, int, float, complex)):
        node = parent.create_dataset(name, data=value, track_times=False)
        node.attrs["kind"] = "python-scalar"
        return

    raise TypeError(f"unsupported result value: {type(value).__qualname__}")


def _read_node(node: h5py.Group | h5py.Dataset) -> Any:
    kind = node.attrs["kind"]
    if kind == "null":
        return None
    if kind == "mapping":
        return {
            _read_node(item["key"]): _read_node(item["value"])
            for item in node["items"].values()
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


def _write_hdf5(obj: Any, target: str | IO[bytes], level: int) -> None:
    with h5py.File(target, "w", track_order=True) as h5:
        h5.attrs["schema"] = SCHEMA
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["identity_version"] = IDENTITY_VERSION
        _write_node(h5, "value", obj, level)


def dump(obj: Any, path: str, *, compresslevel: int | None = None) -> None:
    """Encode ``obj`` as schema-version-1 HDF5 at ``path``.

    Callers continue to own atomic write-then-rename behavior. ``compresslevel``
    retains the historical 1--22 interface and maps onto portable HDF5 gzip.
    """
    level = DEFAULT_LEVEL if compresslevel is None else compresslevel
    if not LEVEL_RANGE[0] <= level <= LEVEL_RANGE[1]:
        raise ValueError(f"compresslevel must be in {LEVEL_RANGE}, got {level}")
    with open(path, "wb") as stream:
        dump_stream(obj, stream, compresslevel=level)


def dump_stream(
    obj: Any, stream: IO[bytes] | io.RawIOBase, *, compresslevel: int | None = None
) -> None:
    """Encode ``obj`` to a binary stream without closing the caller's stream.

    HDF5 requires seekable output. A spooled temporary file supplies random
    access, then copies the finished container to pipes used by remote pull.
    """
    level = DEFAULT_LEVEL if compresslevel is None else compresslevel
    if not LEVEL_RANGE[0] <= level <= LEVEL_RANGE[1]:
        raise ValueError(f"compresslevel must be in {LEVEL_RANGE}, got {level}")
    with tempfile.SpooledTemporaryFile(max_size=_SPOOL_LIMIT, mode="w+b") as staged:
        _write_hdf5(obj, staged, level)
        staged.seek(0)
        while chunk := staged.read(1 << 20):
            stream.write(chunk)


def load(path: str) -> Any:
    """Load current HDF5 or any legacy checkpoint-pickle generation."""
    with open(path, "rb") as stream:
        head = stream.read(8)
    if head == _HDF5_MAGIC:
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("schema") != SCHEMA:
                raise ValueError(f"unsupported HDF5 result schema: {h5.attrs.get('schema')!r}")
            version = int(h5.attrs.get("schema_version", -1))
            if version != SCHEMA_VERSION:
                raise ValueError(f"unsupported result schema version: {version}")
            return _read_node(h5["value"])
    if head[:4] == _ZSTD_MAGIC:
        with zstd.ZstdFile(path, "rb") as stream:
            return pickle.load(stream)
    if head[:2] == _GZIP_MAGIC:
        with gzip.GzipFile(path, "rb") as stream:
            return pickle.load(stream)
    with open(path, "rb") as stream:
        return pickle.load(stream)
