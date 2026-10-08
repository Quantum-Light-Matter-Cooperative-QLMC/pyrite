"""Beam distribution-field CLI options, validation, and TOML writers.

Backs ``pyrite beam``'s named ``[beams.NAME]`` object verbs, which are the only
CLI surface that sets beam phase space: ``pyrite profile`` attaches a beam by
name (``--beam NAME``) and never writes distribution fields itself (issue #54).
``create`` and ``set`` share this module so the two verbs validate identically.
"""

from pathlib import Path
from typing import Any

import click
import tomlkit

from pyrite.console.output import FINITE_FLOAT, NONNEGATIVE_FLOAT, POSITIVE_FLOAT

_ANALYTIC_FIELDS = {
    "transverse_fwhm_mm",
    "beam_fwhm_mm",
    "transverse_fwhm_x_mm",
    "transverse_fwhm_y_mm",
    "transverse",
    "longitudinal",
    "bunch_length_fs",
    "long_shape",
    "long_offsets_fs",
    "energy_spread_frac",
    "divergence_mrad",
}


def beam_cli_options(function):
    source_options: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
        (
            ("--source",),
            dict(
                type=click.Choice(("analytic", "gpt_gdf")),
                help="Beam source [default: analytic]. GDF imports correlated particle records.",
            ),
        ),
        (
            ("--gdf-path",),
            dict(
                type=click.Path(path_type=Path),
                help="Native GPT file; relative to cwd, saved as an absolute path.",
            ),
        ),
        (
            ("--gdf-time-s",),
            dict(
                type=NONNEGATIVE_FLOAT,
                help="Time output in seconds; required for multiple outputs. Excludes --gdf-screen-position-m.",
            ),
        ),
        (
            ("--gdf-time-tolerance-s",),
            dict(
                type=NONNEGATIVE_FLOAT,
                help="Absolute time selection tolerance in seconds [default: 1e-15].",
            ),
        ),
        (
            ("--gdf-screen-position-m",),
            dict(type=FINITE_FLOAT, help="GPT screen coordinate in meters; excludes --gdf-time-s."),
        ),
        (
            ("--gdf-screen-tolerance-m",),
            dict(
                type=NONNEGATIVE_FLOAT,
                help="Absolute screen selection tolerance in meters [default: 1e-9].",
            ),
        ),
        (
            ("--gdf-z-origin-m",),
            dict(
                type=FINITE_FLOAT,
                help="Required explicit physical target origin along GPT lab z in meters.",
            ),
        ),
        (
            ("--gdf-normalization",),
            dict(
                type=click.Choice(("pyrite_current", "gdf_charge")),
                help="pyrite_current uses configured charge/current; gdf_charge derives charge from the file instead of bunch_charge_pc and uses shared rep_rate_hz.",
            ),
        ),
        (
            ("--gdf-shape-only/--no-gdf-shape-only",),
            dict(
                default=None,
                help="Use profile sweep energies and discard imported crossing times; default imports energies and times.",
            ),
        ),
    )
    for names, options in source_options:
        function = click.option(*names, **options)(function)
    function = click.option(
        "--envelope-rms-fs",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="FS",
        help="RMS duration for gaussian or microtrain longitudinal policy.",
    )(function)
    function = click.option(
        "--longitudinal",
        "longitudinal_kind",
        type=click.Choice(("gaussian", "microtrain", "compressed")),
        help="Replace the complete declarative longitudinal policy.",
    )(function)
    function = click.option(
        "--bunch-charge-pc",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="PC",
        help="Physical charge per bunch in pC.",
    )(function)
    function = click.option(
        "--rep-rate-hz",
        type=POSITIVE_FLOAT,
        metavar="HZ",
        help="Shared bunch repetition rate in Hz [default: 5000]; also used by gdf_charge.",
    )(function)
    function = click.option(
        "--transverse-fwhm-mm",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="MM",
        help="Circular Gaussian transverse FWHM in mm.",
    )(function)
    function = click.option(
        "--energy-spread",
        "energy_spread_frac",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="FRAC",
        help="RMS relative energy spread, (E - <E>) / <E>.",
    )(function)
    # alpha is signed: negative is a diverging beam past its waist, so this is
    # deliberately NOT a FloatRange(min=0.0) like its neighbours above.
    function = click.option(
        "--twiss-alpha",
        "alpha_twiss",
        type=float,
        metavar="A",
        help="Courant-Snyder alpha; negative diverges. Requires --emittance.",
    )(function)
    function = click.option(
        "--twiss-beta",
        "beta_twiss_m",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="M",
        help="Courant-Snyder beta in m. Requires --emittance.",
    )(function)
    function = click.option(
        "--emittance",
        "normalized_emittance_mm_mrad",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="MM_MRAD",
        help="Normalized transverse emittance in mm*mrad; replaces the spot FWHM.",
    )(function)
    return function


def collect_beam_updates(
    transverse_fwhm_mm,
    rep_rate_hz,
    bunch_charge_pc,
    longitudinal_kind,
    envelope_rms_fs,
    normalized_emittance_mm_mrad=None,
    beta_twiss_m=None,
    alpha_twiss=None,
    energy_spread_frac=None,
    **source_options,
):
    if longitudinal_kind is None and envelope_rms_fs is not None:
        raise click.UsageError("--envelope-rms-fs requires --longitudinal")
    if longitudinal_kind == "compressed" and envelope_rms_fs is not None:
        raise click.UsageError("compressed derives its duration; omit --envelope-rms-fs")
    if longitudinal_kind in {"gaussian", "microtrain"} and envelope_rms_fs is None:
        raise click.UsageError(f"{longitudinal_kind} requires --envelope-rms-fs")
    # Spot FWHM, divergence and emittance are three numbers for two independent
    # second moments plus a correlation. Resolving that by precedence would
    # write a plausible profile that is not the requested beam, so it is a hard
    # usage error naming both flags.
    if normalized_emittance_mm_mrad is not None and transverse_fwhm_mm is not None:
        raise click.UsageError("--emittance replaces --transverse-fwhm-mm; pass only one")
    if normalized_emittance_mm_mrad is None and (
        beta_twiss_m is not None or alpha_twiss is not None
    ):
        raise click.UsageError("--twiss-beta and --twiss-alpha require --emittance")
    if normalized_emittance_mm_mrad is not None and beta_twiss_m is None:
        raise click.UsageError("--emittance requires --twiss-beta")
    updates = {
        key: value
        for key, value in {
            "transverse_fwhm_mm": transverse_fwhm_mm,
            "rep_rate_hz": rep_rate_hz,
            "bunch_charge_pc": bunch_charge_pc,
            "energy_spread_frac": energy_spread_frac,
        }.items()
        if value is not None
    }
    if longitudinal_kind is not None:
        policy = {"kind": longitudinal_kind}
        if envelope_rms_fs is not None:
            policy["envelope_rms_fs"] = envelope_rms_fs
        updates["longitudinal"] = policy
    if normalized_emittance_mm_mrad is not None:
        # Circular beam, matching --transverse-fwhm-mm's scope: the y-plane
        # keys mirror x when omitted. Elliptical beams stay TOML-only.
        transverse = {
            "normalized_emittance_x_mm_mrad": normalized_emittance_mm_mrad,
            "beta_twiss_x_m": beta_twiss_m,
        }
        if alpha_twiss is not None:
            transverse["alpha_twiss_x"] = alpha_twiss
        updates["transverse"] = transverse
    updates.update({key: value for key, value in source_options.items() if value is not None})
    if "gdf_path" in updates:
        updates["gdf_path"] = str(Path(updates["gdf_path"]).expanduser().resolve())
    if "gdf_time_s" in updates and "gdf_screen_position_m" in updates:
        raise click.UsageError("--gdf-time-s and --gdf-screen-position-m are mutually exclusive")
    if updates.get("source") == "gpt_gdf" and updates.keys() & _ANALYTIC_FIELDS:
        raise click.UsageError("gpt_gdf is incompatible with analytic distribution options")
    return updates


def write_beam_fields(table, updates):
    """Write ``updates`` directly onto ``table``, a beam-shaped TOML table --
    a ``[beams.NAME]`` row, or a hand-authored ``[profiles.NAME.beam]`` subtable
    that still decodes."""
    if not updates:
        return
    source = updates.get("source", table.get("source", "analytic"))
    if source == "gpt_gdf":
        if updates.keys() & _ANALYTIC_FIELDS:
            raise click.UsageError("gpt_gdf is incompatible with analytic distribution options")
        if table.get("source", "analytic") != source:
            for key in _ANALYTIC_FIELDS:
                table.pop(key, None)
        if "gdf_time_s" in updates:
            table.pop("gdf_screen_position_m", None)
        elif "gdf_screen_position_m" in updates:
            table.pop("gdf_time_s", None)
    elif updates.get("source") == "analytic":
        if any(key.startswith("gdf_") for key in updates):
            raise click.UsageError("GDF settings require --source gpt_gdf")
        for key in list(table):
            if key.startswith("gdf_"):
                table.pop(key)
    for key, value in updates.items():
        if key in ("longitudinal", "transverse"):
            policy = tomlkit.table()
            for policy_key, policy_value in value.items():
                policy[policy_key] = policy_value
            table[key] = policy
        else:
            table[key] = value
    # The two transverse spellings are mutually exclusive at decode, so writing
    # one has to retire the other -- otherwise the edit leaves behind a beam
    # that no longer loads.
    if "transverse" in updates:
        for legacy in ("transverse_fwhm_mm", "transverse_fwhm_x_mm", "transverse_fwhm_y_mm"):
            table.pop(legacy, None)
    elif "transverse_fwhm_mm" in updates:
        table.pop("transverse", None)


def validate_gdf_fields(table):
    """Validate external particle data before saving an edited named beam."""
    if "gdf_repetition_rate_hz" in table:
        raise ValueError("removed; replace gdf_repetition_rate_hz with rep_rate_hz")
    if table.get("source", "analytic") != "gpt_gdf":
        return
    from pyrite.campaign.sweep import BeamSpec, beam_replace

    fields = {key: value for key, value in table.items() if key != "label"}
    fields.setdefault("transverse_fwhm_x_mm", None)
    fields.setdefault("transverse_fwhm_y_mm", None)
    beam_replace(BeamSpec(), **fields).gdf_beam()
