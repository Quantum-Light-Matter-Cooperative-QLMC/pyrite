"""Detector-geometry CLI options, validation, and TOML writers."""

import math

import click

from pyrite.campaign.profile_edit import merge_pixel_detector


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


def collect_detector_updates(observation_angle_deg, polar_acceptance_deg, solid_angle_sr):
    return {
        key: value
        for key, value in {
            "observation_angle_deg": observation_angle_deg,
            "polar_acceptance_deg": polar_acceptance_deg,
            "solid_angle_sr": solid_angle_sr,
        }.items()
        if value is not None
    }


def write_detector_fields(table, updates):
    for key, value in updates.items():
        table[key] = value


def apply_detector_fields(target, scalar, pixel):
    geometry, scorer, response, acquisition = pixel
    if any(pixel):
        if scalar or any(
            key in target
            for key in ("observation_angle_deg", "polar_acceptance_deg", "solid_angle_sr")
        ):
            raise click.UsageError("pixel geometry cannot be combined with scalar acceptance")
        if "distance_mm" not in target and "distance_mm" not in geometry:
            raise click.UsageError("creating pixel geometry requires --distance-mm")
        try:
            merge_pixel_detector(
                target, geometry=geometry, scorer=scorer, response=response, acquisition=acquisition
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from None
    elif scalar and "distance_mm" in target:
        raise click.UsageError("pixel detector acceptance is derived from its geometry")
    write_detector_fields(target, scalar)
