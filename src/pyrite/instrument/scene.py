"""Resolved, JSON-safe physical geometry for captured trajectory scenes."""

import json
from dataclasses import asdict
from typing import Any

import numpy as np

from .model import FilterPlate, PlanarDetector


def scene_payload(
    filters: tuple[FilterPlate, ...], detector: PlanarDetector | None
) -> dict[str, Any]:
    """Snapshot ordered plates and detector geometry, without response objects.

    Poses use mm in lab axes relative to the target reference point (entrance
    origin). The transform is recorded explicitly; no slab-centre shift is
    applied. The transport case supplies the per-case sample-to-lab rotation.
    """
    size = None if detector is None else detector.size_mm
    if detector is not None:
        assert size is not None
    payload = {
        "schema": "pyrite.trajectory-scene.v1",
        "units": "mm",
        "frame": "lab",
        "sample_origin_lab_mm": [0.0, 0.0, 0.0],
        "reference": "target-reference-entrance-origin",
        "filters": [
            {
                "name": plate.name,
                "material": plate.material
                if isinstance(plate.material, str)
                else asdict(plate.material),
                "pose": asdict(plate.pose),
                "size_mm": list(plate.size_mm),
                "thickness_mm": plate.thickness_mm,
                "corners_mm": plate.corners_mm().tolist(),
            }
            for plate in filters
        ],
        "detector": None
        if detector is None
        else {
            "pose": asdict(detector.pose),
            "size_mm": list(size) if size is not None else [],
            "pixels": None if detector.pixels is None else asdict(detector.pixels),
            "corners_mm": detector_corners_mm(detector).tolist(),
        },
    }
    return json.loads(json.dumps(payload))


def detector_corners_mm(detector: PlanarDetector) -> np.ndarray:
    """Return detector face corners in lab mm, counterclockwise in local axes."""
    center = np.asarray(detector.pose.center_mm)
    x = np.asarray(detector.pose.x_axis)
    y = np.asarray(detector.pose.y_axis)
    assert detector.size_mm is not None
    width, height = detector.size_mm
    return np.asarray(
        [
            center + sx * width / 2 * x + sy * height / 2 * y
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )
