"""Bounded-memory VTK scene export with independently viewable scale groups."""

import json
import os
import shutil
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .._scene_geometry import (
    beam_footprint_outline,
    case_rotation,
    crystal_mesh,
    groove_mesh,
)
from .trajectories import TrajectoryArtifactError, read_trajectory_header
from .trajectory_export import _BLOCK_ROWS, _field_data, export_segments_vtp


def scene_output_paths(output: Path) -> tuple[Path, Path]:
    """Return close-up and instrument manifest destinations for a track VTP."""
    return output.with_suffix(".vtm"), output.with_name(output.stem + ".instrument.vtm")


def _polydata(path, points, *, triangles=None, lines=None, metadata):
    """Write small numeric scene blocks; tracks use the streaming writer."""
    triangles = np.asarray([] if triangles is None else triangles, dtype=np.int64).reshape(-1, 3)
    lines = [] if lines is None else lines
    xml = ET.Element("VTKFile", type="PolyData", version="1.0", byte_order="LittleEndian")
    poly = ET.SubElement(xml, "PolyData")
    poly.append(ET.fromstring(_field_data(metadata)))
    piece = ET.SubElement(
        poly,
        "Piece",
        NumberOfPoints=str(len(points)),
        NumberOfVerts="0",
        NumberOfLines=str(len(lines)),
        NumberOfStrips="0",
        NumberOfPolys=str(len(triangles)),
    )
    array = ET.SubElement(
        ET.SubElement(piece, "Points"),
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    )
    array.text = " ".join(format(float(value), ".17g") for value in np.asarray(points).ravel())
    for name, cells in (("Lines", lines), ("Polys", triangles)):
        node = ET.SubElement(piece, name)
        connectivity = ET.SubElement(
            node, "DataArray", type="Int64", Name="connectivity", format="ascii"
        )
        connectivity.text = " ".join(str(int(index)) for cell in cells for index in cell)
        offsets = ET.SubElement(node, "DataArray", type="Int64", Name="offsets", format="ascii")
        offsets.text = " ".join(
            str(int(value)) for value in np.cumsum([len(cell) for cell in cells])
        )
    ET.ElementTree(xml).write(path, encoding="utf-8", xml_declaration=True)


def _manifest(path, blocks, metadata):
    xml = ET.Element(
        "VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian"
    )
    composite = ET.SubElement(xml, "vtkMultiBlockDataSet")
    for index, (name, file) in enumerate(blocks):
        ET.SubElement(composite, "DataSet", index=str(index), name=name, file=file)
    xml.append(ET.fromstring(_field_data(metadata)))
    ET.ElementTree(xml).write(path, encoding="utf-8", xml_declaration=True)


def _extent(transport, thickness):
    """Fit sample x/y to captured endpoints, without loading the full shower."""
    low = np.zeros(2)
    high = np.zeros(2)
    for start in range(0, len(transport["L_ang"]), _BLOCK_ROWS):
        stop = start + _BLOCK_ROWS
        mid = transport["r_mid"][start:stop, :2]
        half = 0.5 * transport["L_ang"][start:stop][:, None] * transport["v_hat"][start:stop, :2]
        if len(mid):
            low = np.minimum(low, np.minimum(mid - half, mid + half).min(axis=0))
            high = np.maximum(high, np.maximum(mid - half, mid + half).max(axis=0))
    for name in ("vacuum_start_ang", "vacuum_end_ang"):
        if name in transport:
            for start in range(0, len(transport[name]), _BLOCK_ROWS):
                points = transport[name][start : start + _BLOCK_ROWS, :2]
                if len(points):
                    low = np.minimum(low, points.min(axis=0))
                    high = np.maximum(high, points.max(axis=0))
    pad = max(1.0, thickness * 0.02, float(np.max(high - low)) * 0.05)
    return low - pad, high + pad


def export_trajectory_scene(
    artifact: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    include_vacuum: bool = True,
    overwrite: bool = False,
    include_tracks: bool = True,
    extent_ang: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Path, Path]:
    """Export close-up lab-angstrom and instrument lab-mm scene manifests.

    ``output`` names the existing slab-frame VTP; it is neither read nor
    overwritten here. Each manifest references a private, immutable sidecar
    directory. Both manifests are staged after every sidecar succeeds, then
    replaced individually. Old sidecar directories remain usable on overwrite.
    An interrupted publication never leaves a manifest pointing at partial
    data. Old HDF5 artifacts without a scene omit filters/detector explicitly.
    ``include_tracks=False`` builds geometry only for bounded saved viewers;
    ``extent_ang`` supplies that selection's sample-frame close-up bounds.
    """
    import h5py

    artifact, output = Path(artifact), Path(output)
    header = read_trajectory_header(artifact)
    closeup, instrument = scene_output_paths(output)
    for target in (closeup, instrument):
        if target.exists() and not overwrite:
            raise FileExistsError(f"scene output already exists: {target}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = output.parent / f"{output.stem}.scene-{uuid.uuid4().hex}"
    directory.mkdir()
    published = False
    staged = [directory / "closeup.vtm", directory / "instrument.vtm"]
    try:
        with h5py.File(artifact, "r") as handle:
            case = json.loads(handle["case"].attrs["json"])
            scene = json.loads(handle["scene"].attrs["json"]) if "scene" in handle else None
            if scene is not None and (
                not isinstance(scene, dict)
                or scene.get("schema") != "pyrite.trajectory-scene.v1"
                or scene.get("units") != "mm"
                or scene.get("frame") != "lab"
            ):
                raise TrajectoryArtifactError("unsupported trajectory scene schema, units or frame")
            rotation = case_rotation(case)
            origin = np.asarray(
                (scene or {}).get("sample_origin_lab_mm", [0.0, 0.0, 0.0]), dtype=float
            )
            if origin.shape != (3,) or not np.all(np.isfinite(origin)):
                raise TrajectoryArtifactError(
                    "scene sample_origin_lab_mm must be a finite 3-vector"
                )
            layers = case.get("abs_layers")
            thickness = float(layers[-1][1] if layers else case["thickness_ang"])
            low, high = (
                extent_ang if extent_ang is not None else _extent(handle["transport"], thickness)
            )
            common: dict[str, Any] = {
                "case_sha256": header["case_sha256"],
                "parameter_sha256": header.get("parameter_sha256", ""),
                "scene_sha256": header.get("scene_sha256", ""),
                "provenance": json.loads(handle["provenance"].attrs["json"]),
                "field_units": {
                    name: str(node.attrs["units"])
                    for name, node in handle["transport"].items()
                    if "units" in node.attrs
                },
                "case_name": case["name"],
                "material": case.get("crystal", "unknown"),
                "frame": "lab",
                "sample_to_lab_R": rotation.tolist(),
                "sample_origin_lab_mm": origin.tolist(),
                "downstream_scene": "recorded" if scene is not None else "unavailable",
                "thickness_ang": thickness,
                "crystal_width_mm": case.get("crystal_width_mm"),
                "crystal_height_mm": case.get("crystal_height_mm"),
            }
            near = dict(common, units="angstrom", scale_group="closeup", extent="capture-fitted")
            far = dict(common, units="mm", scale_group="instrument")
            close_blocks = [("tracks", f"{directory.name}/tracks.vtp")] if include_tracks else []
            far_blocks = []

            def add(name, points, *, triangles=None, lines=None, distant=False, **metadata):
                blocks, base = (far_blocks, far) if distant else (close_blocks, near)
                filename = f"{len(close_blocks) + len(far_blocks):03d}.vtp"
                _polydata(
                    directory / filename,
                    points,
                    triangles=triangles,
                    lines=lines,
                    metadata=dict(base, block=name, **metadata),
                )
                blocks.append((name, f"{directory.name}/{filename}"))

            points, triangles = crystal_mesh(
                low[0], high[0], low[1], high[1], thickness, R=rotation
            )
            add("crystal", points + origin * 1e7, triangles=triangles)
            mesh = groove_mesh(case, low[0], high[0], low[1], high[1], 1.0, R=rotation)
            if mesh is not None:
                add("grooves", mesh[0] + origin * 1e7, triangles=mesh[1])
            elif case.get("groove_spacing_ang") is not None:
                near["grooves"] = "omitted: more than 200 periods"
            for index, (_, boundary, _) in enumerate((layers or [])[:-1]):
                sample = np.array(
                    [
                        [low[0], low[1], boundary],
                        [high[0], low[1], boundary],
                        [high[0], high[1], boundary],
                        [low[0], high[1], boundary],
                    ]
                )
                add(
                    f"layer-{index + 1}",
                    sample @ rotation.T + origin * 1e7,
                    triangles=[[0, 1, 2], [0, 2, 3]],
                    boundary_ang=float(boundary),
                )
            span = max(thickness, float(np.max(high - low)))
            add(
                "entry beam",
                np.array([[0, 0, -span], [0, 0, 0]]) + origin * 1e7,
                lines=[[0, 1]],
                representation="lab-axis reference, not individual electrons",
            )
            round_spot = not (case.get("transverse_distribution") or case.get("gdf_source")) and (
                case.get("beam_fwhm_y_mm") is None
                or case.get("beam_fwhm_y_mm") == case.get("beam_fwhm_mm")
            )
            outline = (
                beam_footprint_outline(case, 1e7, case.get("beam_fwhm_mm"), R=rotation)
                if round_spot
                else None
            )
            far["beam_footprint"] = (
                "round FWHM contour"
                if outline is not None
                else "unavailable: no round-spot dimensions"
            )

            if outline is not None:
                points = np.column_stack(outline) + origin
                add("beam footprint", points, lines=[list(range(len(points)))], distant=True)
            width, height = case.get("crystal_width_mm"), case.get("crystal_height_mm")
            if width is not None and height is not None:
                points, triangles = crystal_mesh(
                    -width / 2, width / 2, -height / 2, height / 2, thickness / 1e7, R=rotation
                )
                add("target footprint", points + origin, triangles=triangles, distant=True)
            scene = scene or {}
            max_distance = 1.0
            for index, plate in enumerate(scene.get("filters", [])):
                # corners_mm uses x/y/z sign ordering from FilterPlate, not slab ordering.
                points = np.asarray(plate["corners_mm"], dtype=float)[[0, 4, 6, 2, 1, 5, 7, 3]]
                _, triangles = crystal_mesh(0, 1, 0, 1, 1)
                add(
                    f"filters/{index}:{plate.get('name') or 'unnamed'}",
                    points,
                    triangles=triangles,
                    distant=True,
                    material=plate["material"],
                    thickness_mm=plate["thickness_mm"],
                )
                max_distance = max(max_distance, float(np.max(np.linalg.norm(points, axis=1))))
            detector = scene.get("detector")
            if detector is not None:
                points = np.asarray(detector["corners_mm"], dtype=float)
                add(
                    "detector",
                    points,
                    triangles=[[0, 1, 2], [0, 2, 3]],
                    distant=True,
                    pixels=detector.get("pixels"),
                )
                max_distance = max(max_distance, float(np.max(np.linalg.norm(points, axis=1))))
            add(
                "beam axis",
                np.array([[0, 0, -max_distance], [0, 0, 0]]) + origin,
                lines=[[0, 1]],
                distant=True,
                representation="incident lab-axis reference",
            )
        if include_tracks:
            export_segments_vtp(
                artifact,
                directory / "tracks.vtp",
                include_vacuum=include_vacuum,
                _rotation=rotation,
                _origin_ang=origin * 1e7,
                _metadata=near,
            )
        if read_trajectory_header(artifact) != header:
            raise TrajectoryArtifactError("trajectory artifact changed during scene export; retry")
        _manifest(staged[0], close_blocks, near)
        _manifest(staged[1], far_blocks, far)
        for source, destination in zip(staged, (closeup, instrument), strict=True):
            os.replace(source, destination)
            published = True
    except BaseException:
        if not published:
            shutil.rmtree(directory)
        raise
    return closeup, instrument
