"""Beam distribution-field CLI options, validation, and TOML writers.

Backs ``pyrite beam``'s named ``[beams.NAME]`` object verbs, which are the only
CLI surface that sets beam phase space: ``pyrite profile`` attaches a beam by
name (``--beam NAME``) and never writes distribution fields itself (issue #54).
``create`` and ``set`` share this module so the two verbs validate identically.
"""

import click
import tomlkit


def beam_cli_options(function):
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
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="HZ",
        help="Bunch repetition rate in Hz.",
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
    return updates


def write_beam_fields(table, updates):
    """Write ``updates`` directly onto ``table``, a beam-shaped TOML table --
    a ``[beams.NAME]`` row, or a hand-authored ``[profiles.NAME.beam]`` subtable
    that still decodes."""
    if not updates:
        return
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
