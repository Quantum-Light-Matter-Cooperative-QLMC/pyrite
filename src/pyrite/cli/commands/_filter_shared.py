"""Filter/physical-detector CLI parsing and TOML resolution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NotRequired, TypedDict, cast

import click
import tomlkit

from pyrite.instrument import FilterPlate, PixelGrid, PlanarDetector, PlanarPose


class _FilterRow(TypedDict):
    material: str
    thickness_mm: float
    size_mm: tuple[float, float]
    distance_mm: float
    polar_deg: float
    azimuth_deg: NotRequired[float]
    roll_deg: NotRequired[float]
    offset_mm: NotRequired[tuple[float, float]]
    name: NotRequired[str]


class _PhysicalDetectorRow(TypedDict):
    distance_mm: float
    polar_deg: NotRequired[float]
    azimuth_deg: NotRequired[float]
    roll_deg: NotRequired[float]
    offset_mm: NotRequired[tuple[float, float]]
    shape: NotRequired[tuple[int, int]]
    pitch_mm: NotRequired[tuple[float, float]]


def filter_cli_options(function):
    """Options that describe one finite filter plate."""
    for param_decls, kwargs in reversed(
        (
            (
                ("--offset-mm",),
                {
                    "type": click.Tuple((float, float)),
                    "metavar": "X Y",
                    "default": (0.0, 0.0),
                    "show_default": True,
                    "help": "Local x/y offset in mm.",
                },
            ),
            (
                ("--roll-deg",),
                {
                    "type": float,
                    "default": 0.0,
                    "show_default": True,
                    "help": "Local-roll angle in degrees.",
                },
            ),
            (
                ("--azimuth-deg",),
                {
                    "type": float,
                    "default": 0.0,
                    "show_default": True,
                    "help": "Observation azimuth in degrees.",
                },
            ),
            (
                ("--polar-deg",),
                {
                    "type": float,
                    "default": 90.0,
                    "show_default": True,
                    "help": "Observation polar angle in degrees [0, 180].",
                },
            ),
            (
                ("--distance-mm",),
                {
                    "type": click.FloatRange(min=0.0, min_open=True),
                    "required": True,
                    "metavar": "MM",
                    "help": "Source-to-plate distance in mm.",
                },
            ),
            (
                ("--size-mm",),
                {
                    "type": click.Tuple((float, float)),
                    "required": True,
                    "metavar": "WIDTH HEIGHT",
                    "help": "Plate width and height in mm.",
                },
            ),
            (
                ("--thickness-mm",),
                {
                    "type": click.FloatRange(min=0.0, min_open=True),
                    "required": True,
                    "metavar": "MM",
                    "help": "Plate thickness in mm.",
                },
            ),
            (
                ("--material", "material"),
                {"required": True, "metavar": "KEY", "help": "Catalog crystal or medium key."},
            ),
            (("--name",), {"help": "Optional display name; must be unique within the profile."}),
        )
    ):
        function = click.option(*param_decls, **cast(dict[str, Any], kwargs))(function)
    return function


def physical_detector_cli_options(function):
    """Optional physical-detector fields accepted while adding a filter."""
    for param_decls, kwargs in reversed(
        (
            (
                ("--pitch-mm", "detector_pitch_mm"),
                {
                    "type": click.Tuple((float, float)),
                    "metavar": "Y X",
                    "help": "Physical-detector pixel pitch (y, x) in mm.",
                },
            ),
            (
                ("--shape", "detector_shape"),
                {
                    "type": click.Tuple((int, int)),
                    "metavar": "ROWS COLS",
                    "help": "Physical-detector pixel shape; defaults to a Timepix3 chip.",
                },
            ),
            (
                ("--detector-offset-mm",),
                {
                    "type": click.Tuple((float, float)),
                    "metavar": "X Y",
                    "default": (0.0, 0.0),
                    "show_default": True,
                    "help": "Physical-detector local x/y offset in mm.",
                },
            ),
            (
                ("--detector-roll-deg",),
                {
                    "type": float,
                    "default": 0.0,
                    "show_default": True,
                    "help": "Physical-detector local roll in degrees.",
                },
            ),
            (
                ("--detector-azimuth-deg",),
                {
                    "type": float,
                    "default": 0.0,
                    "show_default": True,
                    "help": "Physical-detector azimuth in degrees.",
                },
            ),
            (
                ("--detector-polar-deg",),
                {
                    "type": float,
                    "default": 90.0,
                    "show_default": True,
                    "help": "Physical-detector polar angle in degrees [0, 180].",
                },
            ),
            (
                ("--detector-distance-mm",),
                {
                    "type": click.FloatRange(min=0.0, min_open=True),
                    "metavar": "MM",
                    "help": "Source-to-physical-detector distance in mm.",
                },
            ),
        )
    ):
        function = click.option(*param_decls, **cast(dict[str, Any], kwargs))(function)
    return function


def filter_from_row(row: Mapping[str, object]) -> FilterPlate:
    """Construct the validated public object from one TOML array row."""
    typed = cast(_FilterRow, row)
    pose = PlanarPose.from_observation(
        typed["distance_mm"],
        typed["polar_deg"],
        typed.get("azimuth_deg", 0.0),
        typed.get("roll_deg", 0.0),
        typed.get("offset_mm", (0.0, 0.0)),
    )
    return FilterPlate(
        material=typed["material"],
        thickness_mm=typed["thickness_mm"],
        size_mm=typed["size_mm"],
        pose=pose,
        name=typed.get("name"),
    )


def physical_detector_from_row(row: Mapping[str, object]) -> PlanarDetector:
    """Construct a planar pixel detector from the profile TOML table."""
    typed = cast(_PhysicalDetectorRow, row)
    pose = PlanarPose.from_observation(
        typed["distance_mm"],
        typed.get("polar_deg", 90.0),
        typed.get("azimuth_deg", 0.0),
        typed.get("roll_deg", 0.0),
        typed.get("offset_mm", (0.0, 0.0)),
    )
    shape = typed.get("shape", (256, 256))
    pitch = typed.get("pitch_mm", (0.055, 0.055))
    return PlanarDetector(pose=pose, pixels=PixelGrid(shape=shape, pitch_mm=pitch))


def filter_row(**values):
    """Validate CLI values through ``FilterPlate`` and serialize one TOML row."""
    plate = filter_from_row(values)
    row = tomlkit.table()
    for key, value in values.items():
        if key == "name" and value is None:
            continue
        row[key] = value
    # Accessing the object makes this single validation source explicit and
    # prevents a future writer from skipping its dataclass checks.
    assert plate is not None
    return row


def physical_detector_row(**values):
    """Validate and serialize an optional physical-detector update."""
    if values["detector_distance_mm"] is None:
        supplied = values.get("detector_options_explicit", ())
        if supplied:
            raise ValueError(
                f"{', '.join(supplied)} require --detector-distance-mm"
            )
        return None
    row_values = {
        "distance_mm": values["detector_distance_mm"],
        "polar_deg": values["detector_polar_deg"],
        "azimuth_deg": values["detector_azimuth_deg"],
        "roll_deg": values["detector_roll_deg"],
        "offset_mm": values["detector_offset_mm"],
    }
    if values["detector_shape"] is not None:
        row_values["shape"] = values["detector_shape"]
    if values["detector_pitch_mm"] is not None:
        row_values["pitch_mm"] = values["detector_pitch_mm"]
    physical_detector_from_row(row_values)
    row = tomlkit.table()
    for key, value in row_values.items():
        row[key] = value
    return row
