"""Filter CLI parsing and filter/physical-detector TOML resolution."""

from collections.abc import Mapping
from typing import Any, cast

import click
import tomlkit

from pyrite.campaign.observation import filter_from_config, physical_detector_from_config
from pyrite.instrument import FilterPlate, PlanarDetector


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


def filter_from_row(row: Mapping[str, object]) -> FilterPlate:
    """Construct the validated public object from one TOML array row."""
    return filter_from_config(row)


def physical_detector_from_row(row: Mapping[str, object]) -> PlanarDetector:
    """Construct a planar pixel detector from the profile TOML table."""
    return physical_detector_from_config(row)


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
