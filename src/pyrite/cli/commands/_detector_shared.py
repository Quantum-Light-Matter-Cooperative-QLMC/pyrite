"""Detector-geometry CLI options, validation, and TOML writers."""

from __future__ import annotations

import math

import click


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
