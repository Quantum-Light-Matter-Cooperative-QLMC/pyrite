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

import os
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .trajectories import TrajectoryArtifactError, read_trajectory_header

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


def export_segments_vtp(
    artifact: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    include_vacuum: bool = True,
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

    Returns
    -------
    dict
        ``segments``, ``vacuum_legs``, ``cells``, and exported ``fields``.

    Raises
    ------
    TrajectoryArtifactError
        If ``artifact`` is not a complete trajectory artifact.
    """
    import h5py

    artifact = Path(artifact)
    output = Path(output)
    read_trajectory_header(artifact)
    with h5py.File(artifact, "r") as handle:
        transport = handle["transport"]
        for required in ("r_mid", "v_hat", "L_ang"):
            if required not in transport:
                raise TrajectoryArtifactError(f"{artifact} has no {required!r} segment field")
        n_seg = int(transport["L_ang"].shape[0])
        fields = [name for name in segment_fields() if name in transport]
        has_vacuum = include_vacuum and "vacuum_start_ang" in transport
        n_vac = int(transport["vacuum_start_ang"].shape[0]) if has_vacuum else 0
        n_cells = n_seg + n_vac

        arrays = []  # (name, vtk type, components, writer)

        def _rows(name, start, stop):
            return transport[name][start:stop]

        def _points(start, stop):
            if stop <= n_seg:
                mid = _rows("r_mid", start, stop).astype("<f8")
                half = 0.5 * _rows("L_ang", start, stop)[:, None] * _rows("v_hat", start, stop)
                return np.stack([mid - half, mid + half], axis=1).reshape(-1, 3)
            a, b = start - n_seg, stop - n_seg
            return np.stack(
                [_rows("vacuum_start_ang", a, b), _rows("vacuum_end_ang", a, b)], axis=1
            ).reshape(-1, 3)

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

        for name in fields:
            dataset = transport[name]
            if dataset.shape[0] != n_seg:
                continue
            dtype = _storage_dtype(dataset.dtype)
            components = 1 if dataset.ndim == 1 else int(np.prod(dataset.shape[1:]))
            if n_vac and name == "electron_id":
                dtype = np.promote_types(dtype, transport["vacuum_elec_id"].dtype)
            arrays.append((name, dtype, components))
        if has_vacuum:
            arrays.append(("is_vacuum", np.dtype("uint8"), 1))

        def _cell_block(name, dtype, components, start, stop):
            if name == "is_vacuum":
                return (np.arange(start, stop) >= n_seg).astype(np.uint8)
            parts = []
            if start < n_seg:
                parts.append(transport[name][start : min(stop, n_seg)])
            if stop > n_seg:
                a, b = max(start, n_seg) - n_seg, stop - n_seg
                parts.append(_vacuum_values(name, a, b, dtype, components))
            block = np.concatenate(parts) if len(parts) > 1 else parts[0]
            return np.asarray(block).astype(dtype, copy=False).reshape(stop - start, components)

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

        xml = [
            '<?xml version="1.0"?>\n',
            '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" '
            'header_type="UInt64">\n',
            "  <PolyData>\n",
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

                def _ranges():
                    for start in range(0, n_cells, _BLOCK_ROWS):
                        yield start, min(start + _BLOCK_ROWS, n_cells)

                def _point_blocks():
                    for start, stop in _ranges():
                        # A block straddling the material/vacuum boundary splits.
                        cuts = sorted({start, stop, *([n_seg] if start < n_seg < stop else [])})
                        for a, b in zip(cuts, cuts[1:], strict=False):
                            yield _points(a, b).astype("<f8")

                _emit(sizes[0], _point_blocks())
                _emit(
                    sizes[1],
                    (np.arange(2 * a, 2 * b, dtype="<i8") for a, b in _ranges()),
                )
                _emit(
                    sizes[2],
                    (np.arange(2 * a + 2, 2 * b + 2, 2, dtype="<i8") for a, b in _ranges()),
                )
                for name, dtype, components in arrays:
                    little = np.dtype(dtype).newbyteorder("<")
                    _emit(
                        n_cells * components * dtype.itemsize,
                        (
                            _cell_block(name, dtype, components, a, b).astype(little, copy=False)
                            for a, b in _ranges()
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
