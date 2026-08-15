"""Schema version 2: columnar record tables over a content-addressed blob pool.

Version 1 spent one HDF5 object per Python leaf, which cost roughly 840 B of
object-header metadata and roughly 125 us per leaf. A component store is not a
tree of unrelated leaves, though: it is ``{configuration: {E0_keV: record}}``
with one record key set, so it encodes columnar. Version 2 detects that shape
and writes a *record table* -- one dataset per flattened record column, plus a
key column per nesting level -- and falls back to an attribute-packed typed tree
for payloads that are not record sets (CAS runner blobs, analysis caches).

Arrays inside a record table are stored once in a per-artifact content-addressed
blob pool and referenced by index; the reference store holds 24.7 MB of arrays
across only 7.0 MB of distinct content. Readers materialize a fresh array per
reference, so deduplication is never observable as aliasing.

Nothing here is compressed. Stored artifacts are unfiltered; the ``slim``
transfer artifact takes the ``shuffle`` byte-transposition filter and is framed
whole by :mod:`compression.zstd` in :mod:`pyrite.checkpoints._checkpoint_io`.
See ``docs/repo-design/storage/result-schema.md`` for the measurements.
"""

import hashlib
import struct
from collections.abc import Mapping
from typing import Any, cast

import h5py
import numpy as np

SCHEMA_VERSION = 2

#: Record tables below this row count are written as typed trees instead. A
#: column costs a fixed group plus one to three datasets whatever the row
#: count, while the tree costs about three objects per cell, so the table wins
#: from two rows up; four keeps a margin and keeps tiny payloads legible.
MIN_TABLE_ROWS = 4

#: A record set may carry optional keys (``spec_coherent`` is present only for
#: coherent-emission runs). The set still encodes as one table while the average
#: record carries at least this fraction of the union of keys.
MIN_KEY_DENSITY = 0.5

_STR = h5py.string_dtype(encoding="utf-8")


# --------------------------------------------------------------------------
# scalar tags
# --------------------------------------------------------------------------


def _scalar_tag(value: Any) -> str | None:
    """Return the packed-scalar tag for ``value``, or ``None`` if it needs a node."""
    if value is None:
        return "null"
    if isinstance(value, np.generic):
        return f"np:{value.dtype.str}"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is complex:
        return "complex"
    if type(value) is str:
        return "str"
    return None


_TAG_DTYPE = {
    "bool": np.dtype(bool),
    "int": np.dtype(np.int64),
    "float": np.dtype(np.float64),
    "complex": np.dtype(np.complex128),
}


def _tag_dtype(tag: str) -> np.dtype:
    if tag.startswith("np:"):
        return np.dtype(tag[3:])
    return _TAG_DTYPE[tag]


def _from_tag(tag: str, raw: Any) -> Any:
    if tag == "null":
        return None
    if tag == "str":
        return raw.decode() if isinstance(raw, bytes) else str(raw)
    if tag.startswith("np:"):
        return np.dtype(tag[3:]).type(raw)
    if tag == "bool":
        return bool(raw)
    if tag == "int":
        return int(raw)
    if tag == "float":
        return float(raw)
    if tag == "complex":
        return complex(raw)
    raise ValueError(f"unsupported result scalar tag: {tag!r}")


def _key_id(key: Any) -> tuple[str, Any]:
    """Identity used to group mapping keys into columns.

    ``30`` and ``30.0`` are equal and hash-equal but must stay distinct columns,
    because version 1 preserved the exact key type.
    """
    return (type(key).__name__, key)


# --------------------------------------------------------------------------
# content addressing
# --------------------------------------------------------------------------


def _feed(h: Any, value: Any) -> None:
    if value is None:
        h.update(b"N")
    elif isinstance(value, np.ndarray):
        h.update(b"A" + value.dtype.str.encode() + repr(value.shape).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, np.generic):
        h.update(b"G" + value.dtype.str.encode() + value.tobytes())
    elif type(value) is bool:
        h.update(b"B" + bytes([value]))
    elif type(value) is int:
        h.update(b"I" + repr(value).encode())
    elif type(value) is float:
        h.update(b"F" + struct.pack("<d", value))
    elif type(value) is complex:
        h.update(b"C" + struct.pack("<dd", value.real, value.imag))
    elif isinstance(value, str):
        h.update(b"S" + repr(len(value)).encode() + value.encode())
    elif isinstance(value, bytes):
        h.update(b"Y" + repr(len(value)).encode() + value)
    elif isinstance(value, (list, tuple)):
        h.update((b"L" if isinstance(value, list) else b"T") + repr(len(value)).encode())
        for item in value:
            _feed(h, item)
    elif isinstance(value, Mapping):
        h.update(b"M" + repr(len(value)).encode())
        for key, item in value.items():
            _feed(h, key)
            _feed(h, item)
    else:
        raise TypeError(f"unsupported result value: {type(value).__qualname__}")


def _content_key(value: Any) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    _feed(digest, value)
    return digest.digest()


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------


def _storage_names(keys: list[Any]) -> list[str]:
    """HDF5 child/attribute names for ``keys``.

    String keys are used verbatim when they are safe and unique so ``h5dump``
    shows real field names; anything else gets an underscore-prefixed index.
    Keys that themselves start with ``_`` never take the literal form, so the
    two namespaces cannot collide.
    """
    used: set[str] = set()
    names: list[str] = []
    for index, key in enumerate(keys):
        literal = (
            type(key) is str
            and key != ""
            and "/" not in key
            and "." not in key
            and not key.startswith("_")
            and key not in used
        )
        name = key if literal else f"_{index:08d}"
        used.add(name)
        names.append(name)
    return names


def _str_array(values: list[str]) -> np.ndarray:
    # An empty object array has no inferable element type, and h5py rejects bare
    # dtype('O'); empty mappings are common (a trimmed `case`), so name the vlen
    # string type explicitly.
    return np.array(values, dtype=_STR if not values else object)


def _read_strs(raw: Any) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in raw]


# --------------------------------------------------------------------------
# record-set detection
# --------------------------------------------------------------------------


def _table_rows(obj: Mapping) -> tuple[int, list[tuple[tuple, Mapping]]] | None:
    """Return ``(levels, rows)`` if ``obj`` is a record set, else ``None``.

    ``levels`` counts the nesting levels of keys above the record layer; each
    row pairs its key tuple with the record mapping.
    """
    levels = 0
    layer: list[tuple[tuple, Any]] = [((), obj)]
    while True:
        if not all(isinstance(node, Mapping) and node for _, node in layer):
            break
        nxt = [(path + (key,), value) for path, node in layer for key, value in node.items()]
        if not all(isinstance(value, Mapping) for _, value in nxt):
            break
        levels += 1
        layer = nxt
    if levels < 1:
        return None
    rows = [(path, node) for path, node in layer]
    if len(rows) < MIN_TABLE_ROWS:
        return None
    union: dict[tuple[str, Any], None] = {}
    filled = 0
    for _, record in rows:
        for key in record:
            union.setdefault(_key_id(key), None)
            filled += 1
    if not union:
        return None
    if filled < MIN_KEY_DENSITY * len(rows) * len(union):
        return None
    return levels, rows


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------


class _Writer:
    def __init__(self, h5: h5py.File, *, shuffle: bool) -> None:
        self._h5 = h5
        self._shuffle = shuffle
        self._blobs: h5py.Group | None = None
        self._blob_index: dict[bytes, int] = {}

    # -- arrays ----------------------------------------------------------
    def _array_kwargs(self, value: np.ndarray) -> dict[str, Any]:
        if self._shuffle and value.ndim >= 1 and value.size >= 2 and value.dtype.kind in "biufc":
            return {"shuffle": True, "chunks": True, "track_times": False}
        return {"track_times": False}

    def blob_ref(self, value: np.ndarray) -> int:
        key = _content_key(value)
        index = self._blob_index.get(key)
        if index is not None:
            return index
        if self._blobs is None:
            self._blobs = self._h5.create_group("blobs")
        index = len(self._blob_index)
        self._blobs.create_dataset(f"{index:08d}", data=value, **self._array_kwargs(value))
        self._blob_index[key] = index
        return index

    # -- typed tree ------------------------------------------------------
    def write_node(self, parent: h5py.Group, name: str, value: Any) -> None:
        if isinstance(value, Mapping):
            plan = _table_rows(value)
            group = parent.create_group(name)
            if plan is not None:
                self._write_table(group, *plan)
            else:
                self._write_mapping(group, value)
            return
        if isinstance(value, (list, tuple)):
            self._write_sequence(parent, name, value)
            return
        if isinstance(value, np.ndarray):
            self._write_array_node(parent, name, value)
            return
        if isinstance(value, bytes):
            node = parent.create_dataset(
                name, data=np.frombuffer(value, dtype=np.uint8), track_times=False
            )
            node.attrs["kind"] = "bytes"
            return
        tag = _scalar_tag(value)
        if tag is None:
            raise TypeError(f"unsupported result value: {type(value).__qualname__}")
        # A scalar reached as a node (rather than packed into its parent's
        # attributes) is a root or sequence-element scalar; keep version 1's
        # dataset form so a bare scalar payload stays legible.
        group = parent.create_group(name)
        group.attrs["kind"] = "scalar"
        group.attrs["tag"] = tag
        if tag != "null":
            group.attrs["v"] = value
        return

    def _write_array_node(self, parent: h5py.Group, name: str, value: np.ndarray) -> None:
        if value.dtype.kind == "O":
            raise TypeError("object-dtype arrays are not supported by the result schema")
        if value.dtype.kind == "U":
            node = parent.create_dataset(
                name, data=value.astype(_STR), dtype=_STR, track_times=False
            )
            node.attrs["kind"] = "unicode-array"
            node.attrs["numpy_dtype"] = value.dtype.str
            return
        node = parent.create_dataset(name, data=value, **self._array_kwargs(value))
        node.attrs["kind"] = "array"

    def _write_sequence(self, parent: h5py.Group, name: str, value: list | tuple) -> None:
        container = "tuple" if isinstance(value, tuple) else "list"
        tags = {_scalar_tag(item) for item in value}
        if len(value) >= 2 and len(tags) == 1:
            tag = tags.pop()
            if tag is not None and tag != "null" and tag != "str":
                node = parent.create_dataset(
                    name, data=np.asarray(value, dtype=_tag_dtype(tag)), track_times=False
                )
                node.attrs["kind"] = "packed-sequence"
                node.attrs["container"] = container
                node.attrs["tag"] = tag
                return
        group = parent.create_group(name)
        group.attrs["kind"] = container
        self._write_entries(group, [f"{i:08d}" for i in range(len(value))], list(value))

    def _write_mapping(self, group: h5py.Group, value: Mapping) -> None:
        group.attrs["kind"] = "mapping"
        keys = list(value)
        names = _storage_names(keys)
        if all(type(key) is str for key in keys):
            group.attrs["keys_inline"] = True
            group.attrs["keys"] = _str_array(keys)
        else:
            group.attrs["keys_inline"] = False
            self._write_sequence(group, "_keys", keys)
        group.attrs["names"] = _str_array(names)
        self._write_entries(group, names, list(value.values()))

    def _write_entries(self, group: h5py.Group, names: list[str], values: list[Any]) -> None:
        """Pack scalar entries into ``group``'s attributes; write the rest as children.

        ``kinds`` is a dataset, not an attribute: for a long sequence or mapping
        (e.g. a cross-material comparison cache with thousands of records) it
        grows large enough to exceed HDF5's ~64 KiB object-header
        attribute-message ceiling, raising ``OSError: Unable to synchronously
        create attribute (object header message is too large)`` -- a limit that
        does not apply to dataset storage.
        """
        kinds: list[str] = []
        for name, value in zip(names, values, strict=True):
            tag = _scalar_tag(value)
            if tag is None:
                kinds.append("node")
                self.write_node(group, name, value)
            else:
                kinds.append(tag)
                if tag != "null":
                    group.attrs[f"v:{name}"] = value
        group.create_dataset("kinds", data=_str_array(kinds), track_times=False)

    # -- record table ----------------------------------------------------
    def _write_table(
        self, group: h5py.Group, levels: int, rows: list[tuple[tuple, Mapping]]
    ) -> None:
        group.attrs["kind"] = "record-table"
        group.attrs["levels"] = levels
        group.attrs["n_rows"] = len(rows)
        keys_group = group.create_group("keys")
        every = list(range(len(rows)))
        for level in range(levels):
            self._write_column(
                keys_group, f"{level:02d}", every, len(rows), [path[level] for path, _ in rows]
            )
        self._write_column_set(group.create_group("records"), [record for _, record in rows])

    def _write_column_set(self, group: h5py.Group, records: list[Mapping]) -> None:
        group.attrs["kind"] = "column-set"
        group.attrs["n_rows"] = len(records)
        order: dict[tuple[str, Any], int] = {}
        keys: list[Any] = []
        for record in records:
            for key in record:
                ident = _key_id(key)
                if ident not in order:
                    order[ident] = len(keys)
                    keys.append(key)
        names = _storage_names(keys)
        if all(type(key) is str for key in keys):
            group.attrs["keys_inline"] = True
            group.attrs["keys"] = _str_array(keys)
        else:
            group.attrs["keys_inline"] = False
            self._write_sequence(group, "_keys", keys)
        group.attrs["names"] = _str_array(names)

        # Per-row key order and presence, pooled: one entry covers a uniform
        # record set, and an optional key adds one more entry rather than a
        # mask per column.
        layouts: dict[tuple[int, ...], int] = {}
        layout_index = np.empty(len(records), dtype=np.int64)
        for row, record in enumerate(records):
            signature = tuple(order[_key_id(key)] for key in record)
            layout_index[row] = layouts.setdefault(signature, len(layouts))
        flat: list[int] = []
        offsets = [0]
        for signature in layouts:
            flat.extend(signature)
            offsets.append(len(flat))
        group.create_dataset("layout", data=layout_index, track_times=False)
        group.create_dataset(
            "layout_values", data=np.asarray(flat, dtype=np.int64), track_times=False
        )
        group.create_dataset(
            "layout_offsets", data=np.asarray(offsets, dtype=np.int64), track_times=False
        )

        columns = group.create_group("cols")
        present: list[list[int]] = [[] for _ in keys]
        values: list[list[Any]] = [[] for _ in keys]
        for row, record in enumerate(records):
            for key, value in record.items():
                column = order[_key_id(key)]
                present[column].append(row)
                values[column].append(value)
        for name, rows, column_values in zip(names, present, values, strict=True):
            self._write_column(columns, name, rows, len(records), column_values)

    def _write_column(
        self, parent: h5py.Group, name: str, rows: list[int], n_rows: int, values: list[Any]
    ) -> None:
        column = parent.create_group(name)
        column.attrs["kind"] = "column"
        dense = len(rows) == n_rows
        column.attrs["dense"] = dense
        if not dense:
            column.create_dataset("rows", data=np.asarray(rows, dtype=np.int64), track_times=False)
        self._encode_column(column, values)

    def _encode_column(self, column: h5py.Group, values: list[Any]) -> None:
        nulls = np.fromiter((value is None for value in values), dtype=bool, count=len(values))
        present = [value for value in values if value is not None]
        if not present:
            column.attrs["enc"] = "null"
            return
        if nulls.any():
            column.create_dataset("nulls", data=nulls, track_times=False)
        column.attrs["has_nulls"] = bool(nulls.any())

        if all(isinstance(value, Mapping) for value in present):
            column.attrs["enc"] = "column-set"
            filled = [value if value is not None else {} for value in values]
            self._write_column_set(column.create_group("set"), filled)
            return

        if all(isinstance(value, np.ndarray) and value.dtype.kind in "biufc" for value in present):
            column.attrs["enc"] = "array-ref"
            refs = np.fromiter(
                (-1 if value is None else self.blob_ref(value) for value in values),
                dtype=np.int64,
                count=len(values),
            )
            column.create_dataset("refs", data=refs, track_times=False)
            return

        tags = {_scalar_tag(value) for value in present}
        if len(tags) == 1:
            tag = tags.pop()
            if tag == "str":
                column.attrs["enc"] = "scalar"
                column.attrs["tag"] = tag
                column.create_dataset(
                    "values",
                    data=_str_array(["" if v is None else v for v in values]),
                    dtype=_STR,
                    track_times=False,
                )
                return
            if tag is not None:
                dtype = _tag_dtype(tag)
                filled_scalars = [dtype.type(0) if v is None else v for v in values]
                column.attrs["enc"] = "scalar"
                column.attrs["tag"] = tag
                column.create_dataset(
                    "values", data=np.asarray(filled_scalars, dtype=dtype), track_times=False
                )
                return

        if self._encode_sequence_column(column, values, present):
            return

        # Anything else -- mixed scalar types, nested sequences of strings,
        # unicode arrays -- becomes a content-addressed pool of typed trees plus
        # an index. Repeated values (the common case for resolved-case metadata)
        # collapse to one tree; wholly distinct values cost what version 1 cost.
        column.attrs["enc"] = "tree-pool"
        pool = column.create_group("pool")
        index: dict[bytes, int] = {}
        refs = np.empty(len(values), dtype=np.int64)
        for row, value in enumerate(values):
            key = _content_key(value)
            slot = index.get(key)
            if slot is None:
                slot = len(index)
                index[key] = slot
                self.write_node(pool, f"{slot:08d}", value)
            refs[row] = slot
        column.create_dataset("refs", data=refs, track_times=False)

    def _encode_sequence_column(
        self, column: h5py.Group, values: list[Any], present: list[Any]
    ) -> bool:
        """Encode a sequence-valued column as a 2D or ragged numeric dataset."""
        if not all(isinstance(value, (list, tuple)) for value in present):
            return False
        containers = {"tuple" if isinstance(value, tuple) else "list" for value in present}
        if len(containers) != 1:
            return False
        tags = {_scalar_tag(item) for value in present for item in value}
        if len(tags) != 1:
            return False
        tag = tags.pop()
        if tag is None or tag in {"null", "str"}:
            return False
        dtype = _tag_dtype(tag)
        column.attrs["container"] = containers.pop()
        column.attrs["tag"] = tag
        lengths = {len(value) for value in present}
        if len(lengths) == 1:
            width = lengths.pop()
            block = np.zeros((len(values), width), dtype=dtype)
            for row, value in enumerate(values):
                if value is not None:
                    block[row] = np.asarray(value, dtype=dtype)
            column.attrs["enc"] = "seq-fixed"
            column.create_dataset("values", data=block, track_times=False)
            return True
        flat: list[Any] = []
        offsets = [0]
        for value in values:
            if value is not None:
                flat.extend(value)
            offsets.append(len(flat))
        column.attrs["enc"] = "seq-ragged"
        column.create_dataset("values", data=np.asarray(flat, dtype=dtype), track_times=False)
        column.create_dataset(
            "offsets", data=np.asarray(offsets, dtype=np.int64), track_times=False
        )
        return True


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------


class _Reader:
    def __init__(self, h5: h5py.File) -> None:
        self._h5 = h5
        self._blobs = h5.get("blobs")
        self._blob_cache: dict[int, np.ndarray] = {}

    def blob(self, index: int) -> np.ndarray:
        cached = self._blob_cache.get(index)
        if cached is None:
            cached = self._blobs[f"{index:08d}"][...]
            self._blob_cache[index] = cached
        # Fresh array per reference: deduplication must never alias two record
        # fields to one mutable object.
        return cached.copy()

    def read_node(self, node: h5py.Group | h5py.Dataset) -> Any:
        # The kind tag is what decides group vs. dataset; the static type of a
        # child lookup cannot express that, so each branch narrows.
        kind = node.attrs["kind"]
        if kind == "record-table":
            return self._read_table(cast(h5py.Group, node))
        if kind == "mapping":
            group = cast(h5py.Group, node)
            if node.attrs["keys_inline"]:
                keys: list[Any] = _read_strs(node.attrs["keys"])
            else:
                keys = list(self.read_node(group["_keys"]))
            names = _read_strs(node.attrs["names"])
            return dict(zip(keys, self._read_entries(group, names), strict=True))
        if kind in {"list", "tuple"}:
            group = cast(h5py.Group, node)
            # "kinds" moved from attribute to dataset (a long sequence can
            # exceed HDF5's attribute-message size ceiling); read either
            # generation so archives written before the change still load.
            kinds_len = len(group["kinds"]) if "kinds" in group else len(group.attrs["kinds"])
            names = [f"{i:08d}" for i in range(kinds_len)]
            values = self._read_entries(group, names)
            return tuple(values) if kind == "tuple" else values
        if kind == "packed-sequence":
            tag = node.attrs["tag"]
            values = [_from_tag(tag, item) for item in node[...]]
            return tuple(values) if node.attrs["container"] == "tuple" else values
        if kind == "scalar":
            tag = node.attrs["tag"]
            return None if tag == "null" else _from_tag(tag, node.attrs["v"])
        if kind == "array":
            return node[...]
        if kind == "unicode-array":
            dataset = cast(h5py.Dataset, node)
            return np.asarray(dataset.asstr()[...], dtype=str(node.attrs["numpy_dtype"]))
        if kind == "bytes":
            return node[...].tobytes()
        raise ValueError(f"unsupported result node kind: {kind!r}")

    def _read_entries(self, group: h5py.Group, names: list[str]) -> list[Any]:
        raw_kinds = group["kinds"][...] if "kinds" in group else group.attrs["kinds"]
        kinds = _read_strs(raw_kinds)
        values: list[Any] = []
        for name, tag in zip(names, kinds, strict=True):
            if tag == "node":
                values.append(self.read_node(group[name]))
            elif tag == "null":
                values.append(None)
            else:
                values.append(_from_tag(tag, group.attrs[f"v:{name}"]))
        return values

    def _read_table(self, group: h5py.Group) -> dict:
        levels = int(group.attrs["levels"])
        n_rows = int(group.attrs["n_rows"])
        key_columns = [
            self._read_column(group["keys"][f"{level:02d}"], n_rows) for level in range(levels)
        ]
        records = self._read_column_set(group["records"])
        root: dict = {}
        for row in range(n_rows):
            node = root
            for level in range(levels - 1):
                node = node.setdefault(key_columns[level][row], {})
            node[key_columns[levels - 1][row]] = records[row]
        return root

    def _read_column_set(self, group: h5py.Group) -> list[dict]:
        n_rows = int(group.attrs["n_rows"])
        if group.attrs["keys_inline"]:
            keys: list[Any] = _read_strs(group.attrs["keys"])
        else:
            keys = list(self.read_node(group["_keys"]))
        names = _read_strs(group.attrs["names"])
        layout_index = group["layout"][...]
        layout_values = group["layout_values"][...]
        layout_offsets = group["layout_offsets"][...]
        layouts = [
            layout_values[layout_offsets[i] : layout_offsets[i + 1]]
            for i in range(len(layout_offsets) - 1)
        ]
        columns = [self._read_column(group["cols"][name], n_rows) for name in names]
        records: list[dict] = []
        for row in range(n_rows):
            record: dict = {}
            for column in layouts[layout_index[row]]:
                record[keys[column]] = columns[column][row]
            records.append(record)
        return records

    def _read_column(self, column: h5py.Group, n_rows: int) -> list[Any]:
        rows = range(n_rows) if column.attrs["dense"] else [int(r) for r in column["rows"][...]]
        values = self._decode_column(column, len(rows) if not column.attrs["dense"] else n_rows)
        if column.attrs["dense"]:
            return values
        out: list[Any] = [None] * n_rows
        for slot, row in enumerate(rows):
            out[row] = values[slot]
        return out

    def _decode_column(self, column: h5py.Group, count: int) -> list[Any]:
        enc = column.attrs["enc"]
        if enc == "null":
            return [None] * count
        nulls = column["nulls"][...] if column.attrs.get("has_nulls") else None
        if enc == "column-set":
            values: list[Any] = list(self._read_column_set(column["set"]))
        elif enc == "array-ref":
            refs = column["refs"][...]
            values = [None if ref < 0 else self.blob(int(ref)) for ref in refs]
        elif enc == "scalar":
            tag = column.attrs["tag"]
            raw = column["values"].asstr()[...] if tag == "str" else column["values"][...]
            values = [_from_tag(tag, item) for item in raw]
        elif enc == "seq-fixed":
            tag = column.attrs["tag"]
            container = column.attrs["container"]
            block = column["values"][...]
            values = [
                (tuple if container == "tuple" else list)(_from_tag(tag, x) for x in row)
                for row in block
            ]
        elif enc == "seq-ragged":
            tag = column.attrs["tag"]
            container = column.attrs["container"]
            flat = column["values"][...]
            offsets = column["offsets"][...]
            values = [
                (tuple if container == "tuple" else list)(
                    _from_tag(tag, x) for x in flat[offsets[i] : offsets[i + 1]]
                )
                for i in range(count)
            ]
        elif enc == "tree-pool":
            pool = column["pool"]
            cache: dict[int, Any] = {}
            values = []
            for ref in column["refs"][...]:
                slot = int(ref)
                if slot not in cache:
                    cache[slot] = self.read_node(pool[f"{slot:08d}"])
                values.append(_fresh(cache[slot]))
        else:
            raise ValueError(f"unsupported result column encoding: {enc!r}")
        if nulls is not None:
            values = [None if nulls[i] else values[i] for i in range(count)]
        return values


def _fresh(value: Any) -> Any:
    """Copy a pooled tree so two rows never share one mutable object."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _fresh(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fresh(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_fresh(item) for item in value)
    return value


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def write(h5: h5py.File, obj: Any, *, shuffle: bool = False) -> None:
    """Encode ``obj`` into the already-opened container ``h5``."""
    _Writer(h5, shuffle=shuffle).write_node(h5, "value", obj)


def read(h5: h5py.File) -> Any:
    """Decode a schema-version-2 container."""
    return _Reader(h5).read_node(h5["value"])
