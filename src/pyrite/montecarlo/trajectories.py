"""Opt-in HDF5 artifacts of a case's electron-transport result.

A run that requests trajectory capture hands the exact
:func:`~pyrite.montecarlo.transport.simulate_trajectories` mapping its case
transported -- before the spectrum phase consumes it -- to
:meth:`TrajectoryCapture.write`. Nothing is re-transported, no random draw is
consumed, and the mapping is never mutated, so spectra are identical with and
without capture. Capture is off unless a caller passes a
:class:`TrajectoryCapture`; ordinary runs never import :mod:`h5py` here.

Layout (``schema_version`` 3)::

    /                   attrs: format, schema_version, complete, created_utc,
                               pyrite_version, case_name, E0_keV, seed,
                               case_sha256, parameter_sha256, fields (JSON),
                               segment_count, row_order ("electron")
    /case               attrs: json -- the resolved case mapping (readable)
                               case_type -- "Case" or "mapping"
    /case/payload       the same case as a typed node tree (exact round trip)
    /provenance         attrs: json -- run-level identity (stem, profile, ...)
    /scene              optional attrs: json -- resolved physical scene (lab mm)
    /settings           attrs: json -- resolved transport controls
                        E_cut_by_electrons  (Ne_transport,) keV
    /spectrum_inputs    optional typed tree: every non-segment input the
                        spectrum phase reads (resolved line grid, continuum
                        grid, ``n_hat``, electron counts, groove, grid record)
    /transport_order    stored row i is transported row transport_order[i];
                        absent when rows were already electron-major
    /electron_index     electron_id, row_start, row_stop: the stored-row run
                        ``[row_start, row_stop)`` of each history, ascending
    /transport/<field>  one node per returned key, insertion order kept; row
                        fields (``pyrite_row``) stored electron-major

Schema 3 compacts row fields on disk without changing what readers see: they
are chunked (``_ROW_CHUNK`` rows), with LZF compression by default; integer IDs narrow to the
smallest dtype holding their range, recording the transported dtype in
``pyrite_dtype``; and a row field constant within every history (``t0_ang``)
is stored once per ``/electron_index`` entry (``pyrite_kind``
``"electron_rows"``, ``rows`` = segment count). Read row fields through
:func:`transport_view`, which restores dtype and per-row values for every
schema. Schema 2 stored every row field uncompressed at its transported dtype
and had no ``/electron_index``; it stays fully readable.

Schema 1 stored the case only as JSON (tuples became lists, NumPy scalars
Python numbers) and had no ``/spectrum_inputs``; it stays readable, but only a
schema-2 or later artifact with ``/spectrum_inputs`` can feed the spectrum
phase (see :func:`artifact_spectrum_inputs`).

Each ``/transport`` node carries a ``pyrite_kind`` attribute so readback
restores the original Python structure: ``array`` datasets keep dtype, shape,
and row order; ``dict``/``list``/``tuple`` groups nest; scalars keep their
Python or NumPy type; ``none`` marks ``None``. A second key naming the SAME
array object (``E_keV`` is ``E_start_keV``) is stored as an HDF5 hard link, so
compatibility aliases cost no extra space. Known per-field units ride on each
dataset's ``units`` attribute; see ``docs/physics/beam-transport/
transport-outputs.md`` for field semantics.

Completion is atomic: the file is written to ``<name>.partial`` and renamed
onto its final path only after ``complete`` is set and the file is closed, so
an interrupted write can never be read as a finished artifact.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._backend import _to_cpu, is_device_array

FORMAT = "pyrite.transport-trajectories"
SCHEMA_VERSION = 3
# First schema whose artifacts are sufficient spectrum-phase inputs (#186).
SPECTRUM_SCHEMA_VERSION = 2
PARTIAL_SUFFIX = ".partial"
# Rows scanned per read when one electron outgrows a streamed block (schema 2).
_BOUNDARY_SCAN_ROWS = 1 << 16
# Schema-3 row datasets: rows per HDF5 chunk and the chunk filters. Benchmarks
# behind the choice: docs/repo-design/storage/trajectory-artifacts.md.
_ROW_CHUNK = 1 << 12
_ROW_FILTERS: dict[str, Any] = {"compression": "lzf", "shuffle": True}
_STORAGE_POLICIES = frozenset({"compact", "throughput"})


def _validate_storage_policy(value: str) -> None:
    if value not in _STORAGE_POLICIES:
        raise ValueError(f"storage_policy must be 'compact' or 'throughput', got {value!r}")


# On-disk dtypes tried, narrowest first, for integer ID row fields.
_COMPACT_IDS: dict[str, tuple[type[np.signedinteger], ...]] = {
    "electron_id": (np.int32,),
    "track_id": (np.int32,),
    "parent_id": (np.int32,),
    "flight_id": (np.int32,),
    "substep_id": (np.int16, np.int32),
}
# Row fields stored once per history when every history holds one value.
_ELECTRON_CONSTANT_ROWS = frozenset({"t0_ang"})

# Device arrays are downloaded in bounded row blocks rather than as one host
# copy of the whole segment table.
_DEVICE_BLOCK_BYTES = 64 * 1024 * 1024

# Units of known transport fields. Anything else falls back to its suffix
# (_ang, _keV, _eV, _mm, _rad); identifiers, codes, counts, and unit vectors
# are dimensionless and carry "1".
_EXPLICIT_UNITS = {
    "r_mid": "angstrom",
    "v_hat": "1",
    "hard_secondary_v_hat": "1",
    "hard_radiative_direction": "1",
    "hard_radiative_target_momentum_eV_c": "eV/c",
    "hard_radiative_Z": "1",
    "initial_v_hat": "1",
    "elec_id": "1",
    "electron_id": "1",
    "vacuum_elec_id": "1",
    "layer": "1",
    "flight_id": "1",
    "substep_id": "1",
    "event_kind": "1",
    "track_id": "1",
    "parent_id": "1",
    "generation": "1",
    "hard_channel": "1",
}
_SUFFIX_UNITS = (
    ("_ang", "angstrom"),
    ("_keV", "keV"),
    ("_eV", "eV"),
    ("_mm", "mm"),
    ("_rad", "rad"),
)


class TrajectoryArtifactError(RuntimeError):
    """A trajectory artifact is missing, incomplete, foreign, or mismatched."""


class TrajectoryArtifactExistsError(TrajectoryArtifactError, FileExistsError):
    """A complete artifact already occupies a path this run would write."""


def field_units(name: str) -> str | None:
    """Unit string for a top-level transport field, or ``None`` when unknown."""
    if name in _EXPLICIT_UNITS:
        return _EXPLICIT_UNITS[name]
    if name.startswith("n_") or name in {"Ne", "n_layers"}:
        return "1"
    for suffix, unit in _SUFFIX_UNITS:
        if name.endswith(suffix):
            return unit
    return None


def case_key(case: Mapping[str, Any]) -> str:
    """Resume key of one case: its configuration name and incident energy."""
    return json.dumps([str(case["name"]), float(case["E0_keV"])])


def artifact_relpath(case: Mapping[str, Any]) -> Path:
    """Deterministic artifact path of ``case`` relative to a capture root.

    Configuration names contain spaces, ``/``, and ``=``, so the directory is a
    readable slug plus a short digest of the exact name (two names that slug
    alike never share a directory); the file names the incident energy.
    """
    name = str(case["name"])
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "case"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    energy = format(float(case["E0_keV"]), ".12g")
    return Path(f"{slug[:80]}-{digest}") / f"E0_{energy}keV.h5"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_device_array(value):
        return _jsonable(_to_cpu(value))
    return value


def _dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, default=repr)


def case_digest(case: Mapping[str, Any]) -> str:
    """SHA-256 of the resolved case's canonical JSON; identifies its physics."""
    return hashlib.sha256(_dumps(case).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrajectoryCapture:
    """Where and how a run writes per-case transport artifacts.

    Picklable, so it crosses into transport worker processes with the case.

    Parameters
    ----------
    root
        Directory holding this run's artifacts; each case writes
        ``root / artifact_relpath(case)``.
    overwrite
        Replace a complete artifact already at a case's path. False raises
        :class:`TrajectoryArtifactExistsError` instead.
    provenance
        JSON-safe run-level identity recorded in every artifact. A
        ``parameter_sha256`` entry also takes part in stale-artifact detection.
    scene
        Optional JSON-safe resolved downstream geometry, independent of the
        case digest. Preserved in every captured artifact.
    storage_policy
        ``"compact"`` (default) uses LZF + shuffle for chunked row fields.
        ``"throughput"`` retains schema-3 indexing and exact values but
        writes uncompressed chunks, trading disk space for capture speed.
    """

    root: str
    overwrite: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)
    scene: Mapping[str, Any] | None = None
    storage_policy: str = "compact"

    def __post_init__(self) -> None:
        _validate_storage_policy(self.storage_policy)

    def path_for(self, case: Mapping[str, Any]) -> Path:
        """Final artifact path of ``case``."""
        return Path(self.root) / artifact_relpath(case)

    def write(
        self,
        case: Mapping[str, Any],
        transport: Mapping[str, Any],
        settings: Mapping[str, Any] | None = None,
        spectrum_inputs: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write ``case``'s transport result atomically; return its path."""
        return write_trajectory_artifact(
            self.path_for(case),
            transport,
            case=case,
            settings=settings,
            provenance=self.provenance,
            scene=self.scene,
            overwrite=self.overwrite,
            spectrum_inputs=spectrum_inputs,
            storage_policy=self.storage_policy,
        )


def _pyrite_version() -> str:
    try:
        from importlib.metadata import version

        return version("pyrite-mc")
    except Exception:
        return "unknown"


def _write_array(group, name: str, value: Any, units: str | None):
    """Write one non-row array; device arrays download in bounded row blocks."""
    if is_device_array(value):
        dtype = np.dtype(value.dtype)
        dataset = group.create_dataset(name, shape=value.shape, dtype=dtype)
        if value.size:
            block = _row_block(value)
            for start in range(0, value.shape[0], block):
                stop = min(start + block, value.shape[0])
                dataset[start:stop] = _to_cpu(value[start:stop])
    else:
        dataset = group.create_dataset(name, data=value)
    dataset.attrs["pyrite_kind"] = "array"
    if units is not None:
        dataset.attrs["units"] = units
    return dataset


def _row_block(value: Any) -> int:
    """Rows per bounded write block; a whole number of chunks when it can be."""
    row_bytes = max(
        1, np.dtype(value.dtype).itemsize * int(np.prod(value.shape[1:], dtype=np.int64))
    )
    block = max(1, _DEVICE_BLOCK_BYTES // row_bytes)
    return block - block % _ROW_CHUNK if block >= _ROW_CHUNK else block


def _stored_rows(value: Any, order: np.ndarray | None, start: int, stop: int) -> np.ndarray:
    """Host copy of stored rows ``[start, stop)``; stored row i is ``order[i]``."""
    device = is_device_array(value)
    if order is None:
        rows = value[start:stop]
    elif device:
        from .._backend import BACKEND

        rows = value[BACKEND.xp.asarray(order[start:stop])]
    else:
        rows = value[order[start:stop]]
    return _to_cpu(rows) if device else np.asarray(rows)


def _canonical_row(name: str) -> str:
    from .transport.secondaries import _ALIASES

    return _ALIASES.get(name, name)


def _compact_dtype(name: str, value: Any) -> np.dtype:
    """Narrowest listed on-disk dtype holding every ID of ``value``."""
    dtype = np.dtype(value.dtype)
    candidates = _COMPACT_IDS.get(_canonical_row(name), ())
    if not candidates or dtype.kind not in "iu" or not value.size:
        return dtype
    low, high = int(value.min()), int(value.max())
    for kind in candidates:
        info = np.iinfo(kind)
        if info.bits < 8 * dtype.itemsize and info.min <= low and high <= info.max:
            return np.dtype(kind)
    return dtype


def _expand_electron_rows(per, starts: np.ndarray, stops: np.ndarray, a: int, b: int):
    """Rows ``[a, b)`` of a field stored once per ``[starts, stops)`` history."""
    first = int(np.searchsorted(stops, a, side="right"))
    last = int(np.searchsorted(starts, b, side="left"))
    counts = np.minimum(stops[first:last], b) - np.maximum(starts[first:last], a)
    return np.repeat(np.asarray(per[first:last]), counts, axis=0)


def _per_electron_values(
    value: Any, order: np.ndarray | None, index: Sequence[np.ndarray], block: int
) -> np.ndarray | None:
    """One value per history when ``value`` is bitwise constant within each."""
    _ids, starts, stops = index
    if len(value.shape) != 1:
        return None
    first = order[starts] if order is not None else starts
    if is_device_array(value):
        from .._backend import BACKEND

        per = np.asarray(_to_cpu(value[BACKEND.xp.asarray(first)]))
    else:
        per = np.asarray(value[first])
    n_rows = int(value.shape[0])
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        rows = np.ascontiguousarray(_stored_rows(value, order, start, stop))
        expected = np.ascontiguousarray(_expand_electron_rows(per, starts, stops, start, stop))
        if not np.array_equal(rows.view(np.uint8), expected.view(np.uint8)):
            return None
    return per


def _write_rows(
    group,
    name: str,
    value: Any,
    units: str | None,
    order: np.ndarray | None,
    index: Sequence[np.ndarray] | None,
    storage_policy: str = "compact",
):
    """Write a row field as lossless schema-3 chunks, with optional compression."""
    n_rows = int(value.shape[0])
    block = _row_block(value)
    per = (
        _per_electron_values(value, order, index, block)
        if index is not None and _canonical_row(name) in _ELECTRON_CONSTANT_ROWS
        else None
    )
    if per is not None:
        dataset = group.create_dataset(name, data=per)
        dataset.attrs["pyrite_kind"] = "electron_rows"
        dataset.attrs["rows"] = n_rows
    else:
        dtype = np.dtype(value.dtype)
        disk = _compact_dtype(name, value)
        filters = (
            dict(
                chunks=(min(_ROW_CHUNK, n_rows), *value.shape[1:]),
                **(_ROW_FILTERS if storage_policy == "compact" else {}),
            )
            if value.size
            else {}
        )
        dataset = group.create_dataset(name, shape=value.shape, dtype=disk, **filters)
        for start in range(0, n_rows, block):
            stop = min(start + block, n_rows)
            dataset[start:stop] = _stored_rows(value, order, start, stop).astype(disk, copy=False)
        dataset.attrs["pyrite_kind"] = "array"
        if disk != dtype:
            dataset.attrs["pyrite_dtype"] = dtype.str
    dataset.attrs["pyrite_row"] = True
    if units is not None:
        dataset.attrs["units"] = units
    return dataset


def _write_node(
    group,
    name: str,
    value: Any,
    links: dict[int, str],
    units: str | None,
    *,
    row: bool = False,
    order: np.ndarray | None = None,
    index: Sequence[np.ndarray] | None = None,
    storage_policy: str = "compact",
) -> None:
    if isinstance(value, np.ndarray) or is_device_array(value):
        # Aliases share one array object; link instead of duplicating rows.
        target = links.get(id(value))
        if target is not None:
            group[name] = group.file[target]
            return
        if row:
            dataset = _write_rows(group, name, value, units, order, index, storage_policy)
        else:
            dataset = _write_array(group, name, value, units)
        links[id(value)] = dataset.name
        return
    if isinstance(value, Mapping):
        sub = group.create_group(name, track_order=True)
        sub.attrs["pyrite_kind"] = "dict"
        for key, item in value.items():
            if not isinstance(key, str) or not key or "/" in key or key in {".", ".."}:
                raise TypeError(f"trajectory field {sub.name!r} has unsupported key {key!r}")
            _write_node(sub, key, item, links, None)
        return
    if isinstance(value, (list, tuple)):
        sub = group.create_group(name, track_order=True)
        sub.attrs["pyrite_kind"] = "tuple" if isinstance(value, tuple) else "list"
        sub.attrs["length"] = len(value)
        for position, item in enumerate(value):
            _write_node(sub, str(position), item, links, None)
        return
    if value is None:
        dataset = group.create_dataset(name, data=np.zeros(0, dtype=np.uint8))
        dataset.attrs["pyrite_kind"] = "none"
        return
    if _is_groove_spec(value):
        sub = group.create_group(name, track_order=True)
        sub.attrs["pyrite_kind"] = "groove_spec"
        for key in ("spacing_ang", "depth_ang", "tilt_polar_rad"):
            _write_node(sub, key, getattr(value, key), links, None)
        return
    if isinstance(value, np.generic):
        dataset = group.create_dataset(name, data=value)
        dataset.attrs["pyrite_kind"] = "numpy_scalar"
    elif isinstance(value, bool):
        dataset = group.create_dataset(name, data=np.bool_(value))
        dataset.attrs["pyrite_kind"] = "bool"
    elif isinstance(value, int):
        dataset = group.create_dataset(name, data=np.int64(value))
        dataset.attrs["pyrite_kind"] = "int"
    elif isinstance(value, float):
        dataset = group.create_dataset(name, data=np.float64(value))
        dataset.attrs["pyrite_kind"] = "float"
    elif isinstance(value, str):
        dataset = group.create_dataset(name, data=value)
        dataset.attrs["pyrite_kind"] = "str"
    elif isinstance(value, os.PathLike):
        dataset = group.create_dataset(name, data=os.fspath(value))
        dataset.attrs["pyrite_kind"] = "path"
    else:
        raise TypeError(
            f"trajectory field {group.name}/{name} has unsupported type {type(value).__name__}"
        )
    if units is not None:
        dataset.attrs["units"] = units


def _is_groove_spec(value: Any) -> bool:
    # Imported lazily: the groove module pulls in Numba, which plain artifact
    # readers (VTP export) never need.
    if type(value).__name__ != "GrooveSpec":
        return False
    from .groove import GrooveSpec

    return isinstance(value, GrooveSpec)


def _case_type(case: Mapping[str, Any]) -> str:
    from .case import Case

    return "Case" if isinstance(case, Case) else "mapping"


def _segment_count(transport: Mapping[str, Any]) -> int:
    rows = transport.get("electron_id", transport.get("elec_id"))
    return 0 if rows is None else int(rows.shape[0])


def row_fields() -> frozenset[str]:
    """Top-level transport fields that hold one entry per segment row."""
    from .transport.secondaries import _ALIASES, _ROW_KEYS

    return frozenset((*_ROW_KEYS, *_ALIASES, "track_id", "parent_id", "generation"))


def _host_electron_ids(transport: Mapping[str, Any]) -> np.ndarray | None:
    ids = transport.get("electron_id", transport.get("elec_id"))
    if ids is None:
        return None
    return np.asarray(_to_cpu(ids) if is_device_array(ids) else ids)


def _electron_order(ids: np.ndarray | None) -> np.ndarray | None:
    """Stable electron-major row order, or ``None`` when rows already are.

    Lockstep cores emit rows step by step, interleaving electrons; storing them
    electron-major lets a reader take whole electrons as contiguous slices.
    """
    if ids is None or ids.size < 2 or bool(np.all(ids[1:] >= ids[:-1])):
        return None
    return np.argsort(ids, kind="stable")


def _electron_index(
    ids: np.ndarray | None, order: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(electron_id, row_start, row_stop)`` of each stored history run."""
    if ids is None or ids.ndim != 1:
        return None
    stored = ids[order] if order is not None else ids
    starts = np.flatnonzero(np.r_[True, stored[1:] != stored[:-1]]) if stored.size else stored[:0]
    stops = np.r_[starts[1:], stored.size]
    return (
        stored[starts].astype(np.int64),
        starts.astype(np.int64),
        stops.astype(np.int64)[: starts.size],
    )


def _compact_index_array(group, name: str, data: np.ndarray, storage_policy: str = "compact"):
    """Row-index array narrowed to int32 when it fits, chunked and filtered."""
    info = np.iinfo(np.int32)
    if data.size and info.min <= int(data.min()) and int(data.max()) <= info.max:
        data = data.astype(np.int32)
    filters = (
        dict(
            chunks=(min(_ROW_CHUNK, data.size),),
            **(_ROW_FILTERS if storage_policy == "compact" else {}),
        )
        if data.size
        else {}
    )
    return group.create_dataset(name, data=data, **filters)


def write_trajectory_artifact(
    path: str | os.PathLike[str],
    transport: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    scene: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    spectrum_inputs: Mapping[str, Any] | None = None,
    storage_policy: str = "compact",
) -> Path:
    """Write one complete transport result to ``path`` atomically.

    Parameters
    ----------
    path
        Final artifact path; parent directories are created.
    transport
        Mapping returned by ``simulate_trajectories``. Host arrays are written
        in place; device-resident arrays are downloaded in bounded row blocks.
        Not mutated.
    case
        Resolved case mapping that produced ``transport``.
    settings
        Resolved transport controls. An ``E_cut_by_electrons`` array is stored
        as a dataset; everything else as JSON.
    provenance
        JSON-safe run-level identity.
    scene
        Optional resolved physical geometry in lab mm, separate from transport
        identity. Old artifacts without this group remain readable.
    overwrite
        Replace an existing complete artifact at ``path``.
    spectrum_inputs
        Every non-segment input the spectrum phase reads, as the runner built
        it after resolving the line grid. Without it the artifact is still a
        complete transport record but cannot feed the spectrum phase.
    storage_policy
        ``"compact"`` (default) uses LZF-compressed shuffled chunks;
        ``"throughput"`` writes uncompressed chunks but keeps schema-3
        indexing, compact IDs, and lossless readback.

    Returns
    -------
    Path
        The final artifact path.

    Raises
    ------
    TrajectoryArtifactExistsError
        If ``path`` exists and ``overwrite`` is false.
    TypeError
        If ``transport`` holds a value the schema cannot represent.
    """
    import h5py

    _validate_storage_policy(storage_policy)
    final = Path(path)
    if final.exists() and not overwrite:
        raise TrajectoryArtifactExistsError(f"trajectory artifact already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.name + PARTIAL_SUFFIX)
    settings = dict(settings or {})
    cutoffs = settings.pop("E_cut_by_electrons", None)
    try:
        with h5py.File(partial, "w", track_order=True) as handle:
            handle.attrs["format"] = FORMAT
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["complete"] = False
            handle.attrs["storage_policy"] = storage_policy
            handle.attrs["created_utc"] = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
            handle.attrs["pyrite_version"] = _pyrite_version()
            handle.attrs["case_name"] = str(case["name"])
            handle.attrs["E0_keV"] = float(case["E0_keV"])
            handle.attrs["seed"] = int(case.get("seed", 0))
            handle.attrs["case_sha256"] = case_digest(case)
            handle.attrs["fields"] = json.dumps(list(transport))
            handle.attrs["segment_count"] = _segment_count(transport)
            case_group = handle.create_group("case")
            case_group.attrs["json"] = _dumps(case)
            case_group.attrs["case_type"] = _case_type(case)
            _write_node(case_group, "payload", dict(case), {}, None)
            handle.create_group("provenance").attrs["json"] = _dumps(provenance or {})
            handle.attrs["parameter_sha256"] = str((provenance or {}).get("parameter_sha256", ""))
            if scene is not None:
                handle.create_group("scene").attrs["json"] = _dumps(scene)
                handle.attrs["scene_sha256"] = case_digest(scene)
            settings_group = handle.create_group("settings")
            settings_group.attrs["json"] = _dumps(settings)
            if cutoffs is not None:
                _write_array(settings_group, "E_cut_by_electrons", np.asarray(cutoffs), "keV")
            if spectrum_inputs is not None:
                _write_node(handle, "spectrum_inputs", dict(spectrum_inputs), {}, None)
            n_rows = _segment_count(transport)
            ids = _host_electron_ids(transport)
            order = _electron_order(ids)
            index = _electron_index(ids, order)
            del ids
            handle.attrs["row_order"] = "electron"
            if order is not None:
                # Stored row i is transported row order[i]; readers invert it.
                _compact_index_array(handle, "transport_order", order, storage_policy)
            if index is not None:
                index_group = handle.create_group("electron_index")
                for key, data in zip(("electron_id", "row_start", "row_stop"), index, strict=True):
                    _compact_index_array(index_group, key, data, storage_policy)
            rows = row_fields()
            group = handle.create_group("transport", track_order=True)
            group.attrs["pyrite_kind"] = "dict"
            links: dict[int, str] = {}
            for name, value in transport.items():
                row = (
                    name in rows
                    and (isinstance(value, np.ndarray) or is_device_array(value))
                    and len(value.shape) > 0
                    and value.shape[0] == n_rows
                )
                _write_node(
                    group,
                    name,
                    value,
                    links,
                    field_units(name),
                    row=row,
                    order=order if row else None,
                    index=index if row else None,
                    storage_policy=storage_policy,
                )
            handle.attrs["complete"] = True
        os.replace(partial, final)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return final


def _read_node(node) -> Any:
    kind = node.attrs.get("pyrite_kind", "array")
    if kind == "dict":
        return {key: _read_node(node[key]) for key in node}
    if kind in {"list", "tuple"}:
        items = [_read_node(node[str(index)]) for index in range(int(node.attrs["length"]))]
        return tuple(items) if kind == "tuple" else items
    if kind == "none":
        return None
    if kind == "groove_spec":
        from .groove import GrooveSpec

        return GrooveSpec(**{key: _read_node(node[key]) for key in node})
    if kind == "array":
        return node[()]
    value = node[()]
    if kind == "numpy_scalar":
        return value
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind in {"str", "path"}:
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return Path(text) if kind == "path" else text
    raise TrajectoryArtifactError(f"unknown trajectory node kind {kind!r} at {node.name}")


class _RowField:
    """A compact schema-3 row field read back at its transported dtype.

    Slices like the h5py dataset it wraps along rows (``[a:b]``, ``[i]``,
    ``[a:b, j]``, ``[()]``); per-history fields expand to one value per row.
    """

    def __init__(self, dataset, view: TransportView):
        self._dataset = dataset
        self._view = view
        self.attrs = dataset.attrs
        self.name = dataset.name
        self.dtype = np.dtype(dataset.attrs.get("pyrite_dtype", dataset.dtype))
        self._per_electron = dataset.attrs.get("pyrite_kind") == "electron_rows"
        rows = int(dataset.attrs["rows"]) if self._per_electron else int(dataset.shape[0])
        self.shape = (rows, *dataset.shape[1:])
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        first, rest = (key[0], key[1:]) if key else (slice(None), ())
        if first is Ellipsis:
            first = slice(None)
        n_rows = self.shape[0]
        scalar = isinstance(first, (int, np.integer))
        if scalar:
            start = int(first) + (n_rows if int(first) < 0 else 0)
            if not 0 <= start < n_rows:
                raise IndexError(f"row {int(first)} out of range for {self.name}")
            stop = start + 1
        elif isinstance(first, slice):
            start, stop, step = first.indices(n_rows)
            if step != 1:
                raise TypeError("row fields slice with unit step only")
            stop = max(start, stop)
        else:
            raise TypeError(f"unsupported row index {first!r}")
        if self._per_electron:
            index = self._view.electron_index()
            if index is None:
                raise TrajectoryArtifactError(f"{self.name} is stored per history without an index")
            _ids, starts, stops = index
            rows = _expand_electron_rows(self._dataset, starts, stops, start, stop)
        else:
            rows = self._dataset[start:stop]
        rows = rows.astype(self.dtype, copy=False)
        if rest:
            rows = rows[(slice(None), *rest)]
        return rows[0] if scalar else rows


class TransportView(Mapping[str, Any]):
    """``/transport`` of an open artifact, row fields read as transported.

    Maps each field name to its h5py node, except that schema-3 compact row
    fields come back as views restoring their transported dtype and per-row
    values. Schema 1/2 nodes pass through unchanged.
    """

    def __init__(self, handle):
        self._handle = handle
        self._group = handle["transport"]
        self._index: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._nodes: dict[str, Any] = {}
        self.attrs = self._group.attrs

    def __getitem__(self, key: str):
        # Memoized: h5py node and attribute lookups cost more than small reads.
        node = self._nodes.get(key)
        if node is None:
            node = self._group[key]
            attrs = node.attrs
            if attrs.get("pyrite_kind") == "electron_rows" or "pyrite_dtype" in attrs:
                node = _RowField(node, self)
            self._nodes[key] = node
        return node

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._group

    def __iter__(self) -> Iterator[str]:
        return iter(self._group)

    def __len__(self) -> int:
        return len(self._group)

    def row_fields(self) -> list[str]:
        """Names of every stored row field (aliases included)."""
        return [key for key in self._group if self._group[key].attrs.get("pyrite_row", False)]

    def electron_index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """``(electron_id, row_start, row_stop)`` per history; ``None`` before schema 3."""
        if self._index is None and "electron_index" in self._handle:
            group = self._handle["electron_index"]
            self._index = tuple(
                group[key][()].astype(np.int64) for key in ("electron_id", "row_start", "row_stop")
            )
        return self._index


def transport_view(handle) -> TransportView:
    """Row-field view of an open artifact's ``/transport`` group (any schema)."""
    return TransportView(handle)


def _read_transport(view: TransportView, order: np.ndarray | None) -> dict[str, Any]:
    rows = set(view.row_fields())
    transport = {}
    for key in view:
        if key not in rows:
            transport[key] = _read_node(view[key])
            continue
        stored = view[key][()]
        if order is not None:
            restored = np.empty_like(stored)
            restored[order] = stored
            stored = restored
        transport[key] = stored
    return transport


def _check_header(handle, path: Path) -> None:
    if handle.attrs.get("format") != FORMAT:
        raise TrajectoryArtifactError(f"not a PyRITE trajectory artifact: {path}")
    version = int(handle.attrs.get("schema_version", 0))
    if version > SCHEMA_VERSION or version < 1:
        raise TrajectoryArtifactError(
            f"unsupported trajectory schema_version {version} in {path}; "
            f"this PyRITE reads 1..{SCHEMA_VERSION}"
        )
    if not bool(handle.attrs.get("complete", False)):
        raise TrajectoryArtifactError(f"incomplete trajectory artifact: {path}")


@dataclass(frozen=True)
class TrajectoryArtifact:
    """One reopened trajectory artifact.

    ``transport`` reproduces the ``simulate_trajectories`` mapping (NumPy
    arrays on the host); ``units`` maps each top-level field to its unit where
    known. ``case`` is the exact typed case (a :class:`~pyrite.montecarlo.case.
    Case` when one was written) from schema 2, and the JSON form from schema 1.
    ``spectrum_inputs`` is ``None`` unless the writer recorded them.
    """

    path: Path
    schema_version: int
    case: Mapping[str, Any]
    provenance: dict[str, Any]
    settings: dict[str, Any]
    transport: dict[str, Any]
    units: dict[str, str]
    attrs: dict[str, Any]
    spectrum_inputs: dict[str, Any] | None = None
    scene: dict[str, Any] | None = None


def _header_attrs(handle) -> dict[str, Any]:
    attrs = {}
    for key, value in handle.attrs.items():
        attrs[key] = value.item() if isinstance(value, np.generic) else value
    attrs["fields"] = json.loads(attrs.get("fields", "[]"))
    return attrs


def read_trajectory_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a complete artifact's root attributes without loading arrays."""
    import h5py

    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as error:
        raise TrajectoryArtifactError(f"unreadable trajectory artifact {path}: {error}") from error
    with handle:
        _check_header(handle, path)
        return _header_attrs(handle)


def read_trajectory_artifact(
    path: str | os.PathLike[str], *, load_transport: bool = True
) -> TrajectoryArtifact:
    """Reopen a complete trajectory artifact.

    Rows come back in their transported order whatever order they are stored
    in. ``load_transport=False`` skips every ``/transport`` array (``transport``
    is then empty), for callers that stream segments with
    :func:`iter_electron_row_blocks`.

    Raises
    ------
    TrajectoryArtifactError
        If the file is not a complete artifact of a supported schema version.
    """
    import h5py

    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as error:
        raise TrajectoryArtifactError(f"unreadable trajectory artifact {path}: {error}") from error
    with handle:
        _check_header(handle, path)
        settings = json.loads(handle["settings"].attrs["json"])
        if "E_cut_by_electrons" in handle["settings"]:
            settings["E_cut_by_electrons"] = handle["settings/E_cut_by_electrons"][()]
        group = handle["transport"]
        units = {
            key: str(group[key].attrs["units"]) for key in group if "units" in group[key].attrs
        }
        return TrajectoryArtifact(
            path=path,
            schema_version=int(handle.attrs["schema_version"]),
            case=_read_case(handle["case"]),
            provenance=json.loads(handle["provenance"].attrs["json"]),
            scene=json.loads(handle["scene"].attrs["json"]) if "scene" in handle else None,
            settings=settings,
            transport=(
                _read_transport(
                    transport_view(handle),
                    handle["transport_order"][()] if "transport_order" in handle else None,
                )
                if load_transport
                else {}
            ),
            units=units,
            attrs=_header_attrs(handle),
            spectrum_inputs=(
                _read_node(handle["spectrum_inputs"]) if "spectrum_inputs" in handle else None
            ),
        )


def iter_electron_row_blocks(
    path: str | os.PathLike[str],
    max_rows: int,
    *,
    fields: tuple[str, ...] | None = None,
    include_shared: bool = True,
) -> Iterator[dict[str, Any]]:
    """Stream an artifact's transport result in whole-electron row blocks.

    Each block is the transport mapping with every row field sliced to one
    contiguous run of stored (electron-major) rows; all other fields are read
    once and shared. A block holds at most ``max_rows`` rows unless one
    electron alone has more, so host memory is bounded by the block, not the
    segment count. Blocks partition the rows and never split an electron.

    ``fields`` restricts row columns; ``include_shared=False`` skips non-row
    arrays and the eager history index. This mode bounds memory independently
    of the number of histories (apart from a single oversized history).

    Raises
    ------
    TrajectoryArtifactError
        Unreadable, incomplete, or pre-schema-2 artifact, invalid fields,
        or ``max_rows < 1``.
    """
    import h5py

    if max_rows < 1:
        raise TrajectoryArtifactError("max_rows must be positive")
    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as error:
        raise TrajectoryArtifactError(f"unreadable trajectory artifact {path}: {error}") from error
    with handle:
        _check_header(handle, path)
        if handle.attrs.get("row_order") != "electron":
            raise TrajectoryArtifactError(
                f"trajectory artifact {path} does not store electron-major rows; "
                f"only schema-{SPECTRUM_SCHEMA_VERSION} or later artifacts stream"
            )
        view = transport_view(handle)
        row_names = view.row_fields()
        shared = (
            {key: _read_node(view[key]) for key in view if key not in row_names}
            if include_shared
            else {}
        )
        if fields is not None:
            unknown = set(fields) - set(row_names)
            if unknown:
                raise TrajectoryArtifactError(f"not stored row fields: {sorted(unknown)}")
            row_names = list(fields)
        ids = view["electron_id" if "electron_id" in view else "elec_id"]
        n_rows = int(ids.shape[0])
        index = view.electron_index() if include_shared else None
        start = 0
        while True:
            stop = min(start + max_rows, n_rows)
            if stop < n_rows:
                stop = (
                    _electron_boundary(ids, start, stop)
                    if index is None
                    else _index_boundary(index[2], start, max_rows)
                )
            block = dict(shared)
            for key in row_names:
                block[key] = view[key][start:stop]
            yield block
            if stop >= n_rows:
                return
            start = stop


def _index_boundary(stops: np.ndarray, start: int, max_rows: int) -> int:
    """Last history end within ``max_rows`` of ``start``, else the next one."""
    fit = int(np.searchsorted(stops, start + max_rows, side="right"))
    if fit and stops[fit - 1] > start:
        return int(stops[fit - 1])
    return int(stops[np.searchsorted(stops, start, side="right")])


def _electron_boundary(ids, start: int, stop: int) -> int:
    """First row at or after an electron boundary near ``stop`` (> ``start``).

    Schema-2 fallback: schema 3 reads the boundary from ``/electron_index``.
    """
    owner = ids[stop]
    window = ids[start:stop]
    back = int(np.searchsorted(window, owner, side="left"))
    if back > 0:
        return start + back
    # One electron spans the whole window: extend to its last row.
    n_rows = int(ids.shape[0])
    while stop < n_rows:
        ahead = ids[stop : min(stop + _BOUNDARY_SCAN_ROWS, n_rows)]
        changed = np.flatnonzero(ahead != owner)
        if changed.size:
            return stop + int(changed[0])
        stop += ahead.size
    return n_rows


def _read_case(group) -> Mapping[str, Any]:
    if "payload" not in group:  # schema 1: JSON only
        return json.loads(group.attrs["json"])
    payload = _read_node(group["payload"])
    if group.attrs.get("case_type") == "Case":
        from .case import Case

        try:
            return Case(**payload)
        except (TypeError, ValueError) as error:
            raise TrajectoryArtifactError(
                f"trajectory artifact {group.file.filename} records a case this PyRITE "
                f"cannot rebuild: {error}"
            ) from error
    return payload


def artifact_spectrum_inputs(artifact: TrajectoryArtifact) -> dict[str, Any]:
    """Return ``artifact``'s spectrum-phase inputs, or refuse an insufficient one.

    Raises
    ------
    TrajectoryArtifactError
        The artifact predates schema 2 or was written without
        ``spectrum_inputs``, so the spectrum phase cannot be replayed from it.
    """
    if artifact.schema_version < SPECTRUM_SCHEMA_VERSION:
        raise TrajectoryArtifactError(
            f"trajectory artifact {artifact.path} uses schema_version "
            f"{artifact.schema_version}, which predates spectrum-phase inputs; "
            "re-run the case with --trajectories to capture a "
            f"schema-{SPECTRUM_SCHEMA_VERSION}-or-later artifact"
        )
    if artifact.spectrum_inputs is None:
        raise TrajectoryArtifactError(
            f"trajectory artifact {artifact.path} records no spectrum-phase inputs; "
            "only artifacts captured by a run can feed the spectrum phase"
        )
    return artifact.spectrum_inputs


@dataclass(frozen=True)
class CapturePlan:
    """Outcome of :func:`preflight_capture` for one run."""

    to_write: int
    replaced: int
    kept: int
    missing_cached: int


def preflight_capture(
    capture: TrajectoryCapture,
    cases: Iterable[Mapping[str, Any]],
    todo: Iterable[Mapping[str, Any]],
) -> CapturePlan:
    """Check existing artifacts before any case transports.

    ``cases`` is the whole requested set; ``todo`` the subset this run will
    transport (the rest were resumed or replayed from a cache and are never
    re-transported, so they cannot gain an artifact). Stale ``.partial`` files
    of to-run cases are removed.

    Raises
    ------
    TrajectoryArtifactExistsError
        A to-run case already has a complete artifact and ``overwrite`` is off.
    TrajectoryArtifactError
        An existing file is not a complete artifact, or a cached case's file
        records a different resolved case or run parameter digest.
    """
    todo_keys = {case_key(case) for case in todo}
    expected_parameters = capture.provenance.get("parameter_sha256")
    to_write = replaced = kept = missing_cached = 0
    existing_to_run = []
    for case in cases:
        key = case_key(case)
        path = capture.path_for(case)
        will_run = key in todo_keys
        if will_run:
            to_write += 1
            path.with_name(path.name + PARTIAL_SUFFIX).unlink(missing_ok=True)
        if not path.exists():
            if not will_run:
                missing_cached += 1
            continue
        header = read_trajectory_header(path)
        stored_parameters = header.get("parameter_sha256") or None
        mismatched = header.get("case_sha256") != case_digest(case) or (
            expected_parameters is not None
            and stored_parameters is not None
            and stored_parameters != expected_parameters
        )
        mismatched |= capture.scene is not None and header.get("scene_sha256") != case_digest(
            capture.scene
        )
        if will_run:
            if not capture.overwrite:
                existing_to_run.append(path)
            replaced += 1
        elif mismatched:
            raise TrajectoryArtifactError(
                f"trajectory artifact {path} records different physics or scene than the requested "
                "case; remove it or choose another trajectory directory"
            )
        else:
            kept += 1
    if existing_to_run:
        shown = ", ".join(str(path) for path in existing_to_run[:3])
        more = "" if len(existing_to_run) <= 3 else f" (+{len(existing_to_run) - 3} more)"
        raise TrajectoryArtifactExistsError(
            f"{len(existing_to_run)} trajectory artifact(s) already exist for cases this run "
            f"will transport: {shown}{more}; pass --overwrite-trajectories to replace them"
        )
    return CapturePlan(
        to_write=to_write, replaced=replaced, kept=kept, missing_cached=missing_cached
    )
