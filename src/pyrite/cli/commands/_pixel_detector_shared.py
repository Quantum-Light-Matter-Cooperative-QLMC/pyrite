"""Shared pixel geometry, scoring, response and acquisition CLI fields."""

import math

import click

from pyrite.console.output import (
    FINITE_FLOAT,
    MEASURED_EDGES_CSV,
    NONNEGATIVE_FLOAT,
    NONNEGATIVE_INT,
    POSITIVE_FLOAT,
    POSITIVE_INT,
)


def _pair_positive(ctx, param, value):
    if value is not None and not all(math.isfinite(item) and item > 0 for item in value):
        raise click.BadParameter("values must be finite and positive", ctx, param)
    return value


def pixel_cli_options(function):
    @click.option(
        "--distance-mm",
        type=POSITIVE_FLOAT,
        metavar="MM",
        help="Source-to-detector distance in mm; required to create.",
    )
    @click.option(
        "--polar-deg",
        type=click.FloatRange(0.0, 180.0),
        metavar="DEG",
        help="Detector-centre polar angle from the beam axis in degrees [0, 180].",
    )
    @click.option(
        "--azimuth-deg",
        type=FINITE_FLOAT,
        metavar="DEG",
        help="Detector-centre azimuth in degrees.",
    )
    @click.option(
        "--roll-deg",
        type=FINITE_FLOAT,
        metavar="DEG",
        help="Detector roll about its normal in degrees.",
    )
    @click.option(
        "--offset-mm",
        type=click.Tuple((FINITE_FLOAT, FINITE_FLOAT)),
        metavar="X Y",
        help="Detector-local x/y offset of the centre in mm.",
    )
    @click.option(
        "--shape",
        type=click.Tuple((POSITIVE_INT, POSITIVE_INT)),
        metavar="ROWS COLS",
        help="Pixel counts; a new table defaults to one 256 x 256 Timepix3 chip.",
    )
    @click.option(
        "--pitch-mm",
        type=click.Tuple((float, float)),
        callback=_pair_positive,
        metavar="Y X",
        help="Pixel pitch (y, x) in mm; a new table defaults to 0.055 0.055.",
    )
    @click.option(
        "--angular-shape",
        type=click.Tuple((POSITIVE_INT, POSITIVE_INT)),
        metavar="ROWS COLS",
        help="Representative directions evaluated per transport; at most --shape.",
    )
    @click.option(
        "--reconstruction",
        type=click.Choice(("nearest_tile", "bilinear_tile")),
        help="Pixel spectrum from its own tile, or blended from neighbouring tiles.",
    )
    @click.option(
        "--response",
        type=click.Choice(("ideal", "timepix3")),
        help="Detector response: unit-efficiency 'ideal' counter or uncalibrated 'timepix3'.",
    )
    @click.option(
        "--timepix-thickness-um",
        type=POSITIVE_FLOAT,
        metavar="UM",
        help="Timepix3 sensor thickness in micrometres.",
    )
    @click.option(
        "--timepix-bias-v", type=POSITIVE_FLOAT, metavar="V", help="Timepix3 sensor bias in volts."
    )
    @click.option(
        "--timepix-input-bin-ev",
        type=POSITIVE_FLOAT,
        metavar="EV",
        help="Timepix3 response-model incident-energy bin width in eV.",
    )
    @click.option(
        "--timepix-output-bin-ev",
        type=POSITIVE_FLOAT,
        metavar="EV",
        help="Timepix3 native measured-energy bin width in eV (not the reporting bins).",
    )
    @click.option(
        "--timepix-samples",
        type=POSITIVE_INT,
        metavar="N",
        help="Timepix3 response Monte Carlo photons per input bin.",
    )
    @click.option(
        "--timepix-seed", type=NONNEGATIVE_INT, metavar="N", help="Timepix3 response-matrix seed."
    )
    @click.option(
        "--exposure-s", type=POSITIVE_FLOAT, metavar="S", help="Exposure (integration) time in s."
    )
    @click.option(
        "--measured-edges-ev",
        type=MEASURED_EDGES_CSV,
        metavar="EV,...",
        help="Explicit half-open reporting-bin edges in eV, strictly increasing.",
    )
    @click.option(
        "--measured-range-ev",
        type=click.Tuple((NONNEGATIVE_FLOAT, NONNEGATIVE_FLOAT)),
        metavar="MIN MAX",
        help="Uniform reporting range in eV; use with --measured-bin-width-ev.",
    )
    @click.option(
        "--measured-bin-width-ev",
        type=POSITIVE_FLOAT,
        metavar="EV",
        help="Uniform reporting-bin width in eV; output binning, not detector resolution.",
    )
    @click.option(
        "--hit-threshold-ev",
        type=NONNEGATIVE_FLOAT,
        metavar="EV",
        help="Post-response measured-energy cut in eV; separate from the response's discriminator.",
    )
    @click.option(
        "--realization",
        type=click.Choice(("expected", "poisson")),
        help="Report expected counts, or a seeded Poisson realization.",
    )
    @click.option(
        "--realization-seed",
        type=NONNEGATIVE_INT,
        metavar="N",
        help="Seed of the Poisson realization.",
    )
    def decorated():
        pass

    function.__click_params__ = [
        *getattr(function, "__click_params__", []),
        *getattr(decorated, "__click_params__", []),
    ]
    return function


def _acquisition_changes(values):
    edges = values["measured_edges_ev"]
    uniform = values["measured_range_ev"], values["measured_bin_width_ev"]
    if edges is not None and any(item is not None for item in uniform):
        raise click.UsageError(
            "use --measured-edges-ev or --measured-range-ev with --measured-bin-width-ev, not both"
        )
    if (uniform[0] is None) != (uniform[1] is None):
        raise click.UsageError("--measured-range-ev and --measured-bin-width-ev go together")
    if edges is not None and any(
        right <= left for left, right in zip(edges, edges[1:], strict=False)
    ):
        raise click.UsageError("--measured-edges-ev must be strictly increasing")
    if edges is not None and len(edges) < 2:
        raise click.UsageError("--measured-edges-ev needs at least two edges")
    if values["realization"] == "expected" and values["realization_seed"] is not None:
        raise click.UsageError("--realization-seed applies only to --realization poisson")
    changes = {}
    if values["exposure_s"] is not None:
        changes["exposure_s"] = values["exposure_s"]
    if edges is not None:
        changes["measured_edges_eV"] = list(edges)
    if uniform[0] is not None:
        minimum, maximum = uniform[0]
        changes.update(
            measured_min_eV=minimum, measured_max_eV=maximum, measured_bin_width_eV=uniform[1]
        )
    if values["hit_threshold_ev"] is not None:
        changes["hit_threshold_eV"] = values["hit_threshold_ev"]
    if values["realization"] is not None:
        changes["mode"] = values["realization"]
    if values["realization_seed"] is not None:
        changes["seed"] = values["realization_seed"]
    return changes


def pixel_changes(values):
    geometry = {
        key: values[key]
        for key in (
            "distance_mm",
            "polar_deg",
            "azimuth_deg",
            "roll_deg",
            "offset_mm",
            "shape",
            "pitch_mm",
        )
        if values[key] is not None
    }
    scorer = {
        key: values[key] for key in ("reconstruction", "angular_shape") if values[key] is not None
    }
    response = {
        key: values[option]
        for key, option in (
            ("kind", "response"),
            ("thickness_um", "timepix_thickness_um"),
            ("bias_v", "timepix_bias_v"),
            ("dE_mc", "timepix_input_bin_ev"),
            ("dE_out", "timepix_output_bin_ev"),
            ("n_mc", "timepix_samples"),
            ("seed", "timepix_seed"),
        )
        if values[option] is not None
    }
    acquisition = _acquisition_changes(values)
    return geometry, scorer, response, acquisition
