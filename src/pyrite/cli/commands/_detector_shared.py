"""Detector-geometry CLI options, validation, and TOML writers."""

import math

import click

from pyrite.detectors.editing import POINT_DETECTOR_KIND, patch_detector


def detector_cli_options(function):
    function = click.option(
        "--solid-angle",
        "solid_angle_sr",
        type=click.FloatRange(min=0.0, max=4.0 * math.pi, min_open=True),
        metavar="SR",
        help="Detector solid angle in sr; scalar replacement.",
    )(function)
    function = click.option(
        "--polar-acceptance",
        "polar_acceptance_deg",
        type=click.FloatRange(min=0.0, max=180.0, min_open=True),
        metavar="DEG",
        help="Full detector polar acceptance span in degrees; scalar replacement.",
    )(function)
    function = click.option(
        "--observation-angle",
        "observation_angle_deg",
        type=click.FloatRange(min=0.0, max=180.0),
        metavar="DEG",
        help="Detector observation angle in degrees [0, 180]; scalar replacement.",
    )(function)
    return function


def point_option(function):
    """``--point``: an ideal point observer with zero angular acceptance (#624)."""
    return click.option(
        "--point",
        is_flag=True,
        help=(
            'Make this a point detector (kind = "point"): one observation direction, '
            "zero angular acceptance; removes and excludes --polar-acceptance/--solid-angle."
        ),
    )(function)


def collect_detector_updates(
    observation_angle_deg, polar_acceptance_deg, solid_angle_sr, *, point=False
):
    updates = {
        key: value
        for key, value in {
            "observation_angle_deg": observation_angle_deg,
            "polar_acceptance_deg": polar_acceptance_deg,
            "solid_angle_sr": solid_angle_sr,
        }.items()
        if value is not None
    }
    if point:
        if updates.keys() & {"polar_acceptance_deg", "solid_angle_sr"}:
            raise click.UsageError(
                "--point has no angular acceptance; drop --polar-acceptance/--solid-angle"
            )
        updates["kind"] = POINT_DETECTOR_KIND
    return updates


def write_detector_fields(table, updates):
    for key, value in updates.items():
        table[key] = value


def apply_detector_fields(target, scalar, pixel):
    try:
        patch_detector(target, scalar, pixel)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from None
