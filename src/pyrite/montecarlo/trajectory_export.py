"""Export trajectory-artifact segments to VTK XML PolyData (``.vtp``).

The HDF5 artifact (:mod:`pyrite.montecarlo.trajectories`) stays
authoritative; this is a lossy view for ParaView, VisIt, PyVista/VTK and other
VTK readers. Each straight flight becomes one two-point line cell from
``r_mid - L_ang v_hat / 2`` to ``r_mid + L_ang v_hat / 2`` in the slab frame
[angstrom]. Every per-row transport array rides along as cell data under its
own name, so ``electron_id`` (and ``track_id`` for secondaries) keeps each
cell's electron association and ``Threshold`` on it isolates one history.

Grooved runs append their non-radiating vacuum legs as further cells with
``is_vacuum == 1``; there ``electron_id``, ``E_start_keV``, ``t_start_ang``,
``t0_ang``, ``L_ang``, and ``v_hat`` come from the ``vacuum_*`` arrays and all
other row fields are NaN (floating) or -1 (integer).

Not represented: compatibility aliases (``E_keV``, ``t_ang``, ``elec_id``),
per-electron arrays (``initial_*``, ``straggle_dE_keV``), scalar tallies,
nested metadata (``inelastic``, ``radiative``, ``secondaries``,
``secondary_tracks``, ``stopping_tables``, ``transport_diagnostics``), case,
settings, provenance, and units. VTK carries no units; lengths are angstrom
and every other field keeps the unit recorded in the HDF5 artifact.

Data are written in the VTK ``appended``/``raw`` encoding in bounded row
blocks, so export memory does not scale with the segment count.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .trajectories import TrajectoryArtifactError, read_trajectory_header
from .trajectory_selection import TrajectorySelection


def _field_data(metadata):
    """VTK XML string arrays store null-terminated UTF-8 character codes."""
    xml = ["<FieldData>\n"]
    for name, value in metadata.items():
        value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        codes = " ".join(str(byte) for byte in value.encode("utf-8") + b"\0")
        xml.append(
            f'<Array type="String" Name={quoteattr(name)} NumberOfTuples="1" '
            f'format="ascii">{codes}</Array>\n'
        )
    return "".join([*xml, "</FieldData>\n"])


_BLOCK_ROWS = 1 << 20

# Compatibility aliases name the same arrays as canonical fields.
_ALIASES = frozenset({"E_keV", "t_ang", "elec_id"})

_VTK_TYPES = {
    np.dtype("int8"): "Int8",
    np.dtype("uint8"): "UInt8",
    np.dtype("int16"): "Int16",
    np.dtype("uint16"): "UInt16",
    np.dtype("int32"): "Int32",
    np.dtype("uint32"): "UInt32",
    np.dtype("int64"): "Int64",
    np.dtype("uint64"): "UInt64",
    np.dtype("float32"): "Float32",
    np.dtype("float64"): "Float64",
}

# Vacuum-leg sources of the row fields they can fill.
_VACUUM_SOURCES = {
    "electron_id": "vacuum_elec_id",
    "E_start_keV": "vacuum_E_keV",
    "t_start_ang": "vacuum_t_ang",
    "t0_ang": "vacuum_t0_ang",
}


def segment_fields() -> tuple[str, ...]:
    """Per-row transport fields, in export order."""
    from .transport.secondaries import _ROW_KEYS

    return (*_ROW_KEYS, "track_id", "parent_id", "generation")


def _storage_dtype(dtype: np.dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return np.dtype("uint8")
    if dtype not in _VTK_TYPES:
        raise TrajectoryArtifactError(f"dtype {dtype} has no VTK equivalent")
    return dtype


def _fill_value(dtype: np.dtype):
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.signedinteger):
        return -1
    return 0


def _pieces(ranges, block_rows):
    """Split ``[start, stop)`` runs into pieces of at most ``block_rows``."""
    for a, b in ranges:
        for start in range(a, b, block_rows):
            yield start, min(b, start + block_rows)


def export_segments_vtp(
    artifact: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    include_vacuum: bool = True,
    selection: TrajectorySelection | None = None,
    _rotation=None,
    _origin_ang=None,
    _metadata=None,
) -> dict[str, int | list[str]]:
    """Write one artifact's segments as a VTK XML PolyData file.

    Parameters
    ----------
    artifact
        Complete trajectory artifact written by a capture-enabled run.
    output
        Destination ``.vtp`` path; replaced atomically.
    include_vacuum
        Append grooved runs' vacuum legs as ``is_vacuum`` cells.
    selection
        Whole histories/tracks of ``artifact`` from
        :func:`~pyrite.montecarlo.trajectory_selection.select_trajectories`;
        only their rows are read. Adds a ``segment_id`` cell array (transported
        row index; -1 on vacuum legs) and a ``selection`` FieldData record.
        ``None`` exports every segment.

    Returns
    -------
    dict
        ``segments``, ``vacuum_legs``, ``cells``, and exported ``fields``.

    Raises
    ------
    TrajectoryArtifactError
        If ``artifact`` is not a complete trajectory artifact, or
        ``selection`` was made from another file.
    """
    import h5py

    artifact = Path(artifact)
    output = Path(output)
    read_trajectory_header(artifact)
    if selection is not None and Path(selection.path).resolve() != artifact.resolve():
        raise TrajectoryArtifactError(f"selection of {selection.path} applied to {artifact}")
    with h5py.File(artifact, "r") as handle:
        transport = handle["transport"]
        for required in ("r_mid", "v_hat", "L_ang"):
            if required not in transport:
                raise TrajectoryArtifactError(f"{artifact} has no {required!r} segment field")
        n_rows = int(transport["L_ang"].shape[0])
        fields = [name for name in segment_fields() if name in transport]
        has_vacuum = include_vacuum and "vacuum_start_ang" in transport
        if selection is None:
            material = ((0, n_rows),) if n_rows else ()
            n_legs = int(transport["vacuum_start_ang"].shape[0]) if has_vacuum else 0
            vacuum = ((0, n_legs),) if n_legs else ()
        else:
            material = selection.ranges
            vacuum = selection.vacuum_ranges if has_vacuum else ()
        n_seg = sum(b - a for a, b in material)
        n_vac = sum(b - a for a, b in vacuum)
        n_cells = n_seg + n_vac
        # Every array is emitted piece by piece in the same order: material
        # runs, then vacuum runs; ``vacuum`` marks which source a piece reads.
        pieces = [(False, a, b) for a, b in _pieces(material, _BLOCK_ROWS)]
        pieces += [(True, a, b) for a, b in _pieces(vacuum, _BLOCK_ROWS)]
        order = handle["transport_order"] if "transport_order" in handle else None

        def _rows(name, start, stop):
            return transport[name][start:stop]

        def _points(is_vacuum, a, b):
            if not is_vacuum:
                mid = _rows("r_mid", a, b).astype("<f8")
                half = 0.5 * _rows("L_ang", a, b)[:, None] * _rows("v_hat", a, b)
                points = np.stack([mid - half, mid + half], axis=1).reshape(-1, 3)
            else:
                points = np.stack(
                    [_rows("vacuum_start_ang", a, b), _rows("vacuum_end_ang", a, b)], axis=1
                ).reshape(-1, 3)
            if _rotation is not None:
                points = points @ _rotation.T
            if _origin_ang is not None:
                points = points + _origin_ang
            return points

        def _vacuum_values(name, a, b, dtype, components):
            shape = (b - a,) if components == 1 else (b - a, components)
            if name in _VACUUM_SOURCES:
                return _rows(_VACUUM_SOURCES[name], a, b)
            if name in {"L_ang", "v_hat"}:
                delta = _rows("vacuum_end_ang", a, b) - _rows("vacuum_start_ang", a, b)
                length = np.linalg.norm(delta, axis=1)
                if name == "L_ang":
                    return length
                with np.errstate(invalid="ignore", divide="ignore"):
                    return delta / length[:, None]
            return np.full(shape, _fill_value(dtype), dtype=dtype)

        arrays = []  # (name, vtk dtype, components)
        for name in fields:
            dataset = transport[name]
            if dataset.shape[0] != n_rows:
                continue
            dtype = _storage_dtype(dataset.dtype)
            components = 1 if dataset.ndim == 1 else int(np.prod(dataset.shape[1:]))
            if n_vac and name == "electron_id":
                dtype = np.promote_types(dtype, transport["vacuum_elec_id"].dtype)
            arrays.append((name, dtype, components))
        if has_vacuum:
            arrays.append(("is_vacuum", np.dtype("uint8"), 1))
        if selection is not None:
            arrays.append(("segment_id", np.dtype("int64"), 1))

        def _cell_block(name, dtype, components, is_vacuum, a, b):
            if name == "is_vacuum":
                return np.full(b - a, is_vacuum, dtype=np.uint8)
            if name == "segment_id":
                if is_vacuum:
                    return np.full(b - a, -1, dtype=np.int64)
                return order[a:b] if order is not None else np.arange(a, b)
            if is_vacuum:
                block = _vacuum_values(name, a, b, dtype, components)
            else:
                block = transport[name][a:b]
            if (
                name
                in {
                    "v_hat",
                    "hard_secondary_v_hat",
                    "hard_radiative_direction",
                    "hard_radiative_target_momentum_eV_c",
                }
                and _rotation is not None
            ):
                block = block @ _rotation.T
            if name == "r_mid" and _rotation is not None:
                block = block @ _rotation.T
                if _origin_ang is not None:
                    block = block + _origin_ang
            return np.asarray(block).astype(dtype, copy=False).reshape(b - a, components)

        # Appended-raw layout: each array is a UInt64 byte count then its bytes.
        offset = 0
        sizes = [n_cells * 2 * 3 * 8, n_cells * 2 * 8, n_cells * 8]
        sizes += [n_cells * components * dtype.itemsize for _, dtype, components in arrays]
        offsets = []
        for size in sizes:
            offsets.append(offset)
            offset += 8 + size

        def _attr(name, dtype, components, data_offset):
            return (
                f'        <DataArray type="{_VTK_TYPES[np.dtype(dtype)]}" Name={quoteattr(name)} '
                f'NumberOfComponents="{components}" format="appended" offset="{data_offset}"/>\n'
            )

        metadata = dict(_metadata or {})
        if selection is not None:
            metadata["selection"] = selection.request
        xml = [
            '<?xml version="1.0"?>\n',
            '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" '
            'header_type="UInt64">\n',
            "  <PolyData>\n",
            _field_data(metadata) if metadata else "",
            f'    <Piece NumberOfPoints="{2 * n_cells}" NumberOfVerts="0" '
            f'NumberOfLines="{n_cells}" NumberOfStrips="0" NumberOfPolys="0">\n',
            "      <Points>\n",
            _attr("Points", np.float64, 3, offsets[0]),
            "      </Points>\n",
            "      <Lines>\n",
            _attr("connectivity", np.int64, 1, offsets[1]),
            _attr("offsets", np.int64, 1, offsets[2]),
            "      </Lines>\n",
            "      <CellData>\n",
            *(
                _attr(name, dtype, components, offsets[3 + index])
                for index, (name, dtype, components) in enumerate(arrays)
            ),
            "      </CellData>\n",
            "    </Piece>\n",
            "  </PolyData>\n",
            '  <AppendedData encoding="raw">\n   _',
        ]

        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        try:
            with open(partial, "wb") as stream:
                stream.write("".join(xml).encode("ascii"))

                def _emit(size, blocks):
                    stream.write(np.uint64(size).astype("<u8").tobytes())
                    for block in blocks:
                        stream.write(np.ascontiguousarray(block).tobytes())

                def _cell_ranges():
                    cell = 0
                    for _, a, b in pieces:
                        yield cell, cell + b - a
                        cell += b - a

                _emit(sizes[0], (_points(*piece).astype("<f8") for piece in pieces))
                _emit(
                    sizes[1],
                    (np.arange(2 * a, 2 * b, dtype="<i8") for a, b in _cell_ranges()),
                )
                _emit(
                    sizes[2],
                    (np.arange(2 * a + 2, 2 * b + 2, 2, dtype="<i8") for a, b in _cell_ranges()),
                )
                for name, dtype, components in arrays:
                    little = np.dtype(dtype).newbyteorder("<")
                    _emit(
                        n_cells * components * dtype.itemsize,
                        (
                            _cell_block(name, dtype, components, *piece).astype(little, copy=False)
                            for piece in pieces
                        ),
                    )
                stream.write(b"\n  </AppendedData>\n</VTKFile>\n")
            os.replace(partial, output)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    return {
        "segments": n_seg,
        "vacuum_legs": n_vac,
        "cells": n_cells,
        "fields": [name for name, _, _ in arrays],
    }
