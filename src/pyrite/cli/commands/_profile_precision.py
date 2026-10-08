"""``pyrite profile precision``: a profile's adaptive electron-count policy."""

import click
from tomlkit.exceptions import ParseError

from pyrite._precision import OBSERVABLES, REQUIRED_FIELDS, Precision
from pyrite.campaign import profile_edit as _profile_edit
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli.commands._profile_shared import confirm_standard, write
from pyrite.console import json as cli_json
from pyrite.console.output import CLIError, emit_json_result, emit_result, output_option

_PRECISION_FIELD_NAMES = {
    "target-rse": "target_rse",
    "min-electrons": "min_electrons",
    "max-electrons": "max_electrons",
    "block-electrons": "block_electrons",
    "observable": "observables",
    "max-electron-share": "max_electron_share",
    "min-effective-electrons": "min_effective_electrons",
    "stability-blocks": "stability_blocks",
    "stability-fraction": "stability_fraction",
    "pilot-electrons": "pilot_electrons",
    "band-ev": "band_eV",
    "batch-means-band-ev": "batch_means_band_eV",
}


def _energy_band(_ctx, param, value):
    if value is None:
        return None
    parts = value.split(",")
    try:
        band = tuple(float(part) for part in parts)
    except ValueError:
        band = ()
    if len(band) != 2 or not 0.0 < band[0] < band[1]:
        raise click.BadParameter(
            f"{value!r} is not START,STOP with 0 < START < STOP in eV", param=param
        )
    return band


@click.group("precision")
def command():
    """Inspect and edit PROFILE's adaptive electron-count policy.

    With a policy, each incoherent case stops when the relative standard error
    of its line (and optionally bremsstrahlung) yield meets the target and its
    heavy-tail guards pass, or at the maximum count flagged statistics-limited.
    One count serves line and bremsstrahlung transport. Coherent emission and
    fixed electron counts cannot be combined with a policy.
    """


@command.command("show")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@output_option
def precision_show_command(name, json_output):
    """Show PROFILE's adaptive policy, or report fixed electron counts."""
    try:
        _text, document = _catalog_io.catalog_text()
        explicit = _profile_edit.profile_precision_values(
            _profile_edit.existing_profile(document, name)
        )
        effective = None if explicit is None else Precision.from_dict(explicit).to_dict()
    except (OSError, TypeError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.precision.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        payload = {
            "profile": name,
            "mode": "fixed" if effective is None else "adaptive",
            "explicit": explicit,
            "effective": effective,
        }
        emit_json_result(cli_json.JsonResult("cxr.profile.precision.show", payload))
        return 0
    if effective is None:
        emit_result(f"[{name} precision] fixed electron counts (no adaptive policy)")
        return 0
    emit_result(f"[{name} precision] adaptive")
    assert explicit is not None
    for flag, key in _PRECISION_FIELD_NAMES.items():
        value = effective[key]
        if value is None:
            display = "none"
        elif isinstance(value, tuple):
            display = ",".join(
                f"{item:g}" if isinstance(item, float) else str(item) for item in value
            )
        else:
            display = f"{value:g}" if isinstance(value, float) else str(value)
        emit_result(f"  {flag}: {display}{'' if key in explicit else ' (default)'}")
    return 0


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option(
    "--target-rse",
    type=click.FloatRange(min=0.0, min_open=True),
    metavar="FRACTION",
    help="Target relative standard error of each watched yield (0.05 = 5%).",
)
@click.option(
    "--min-electrons",
    type=click.IntRange(min=2),
    metavar="N",
    help="Smallest count at which the run may stop; a multiple of --block-electrons.",
)
@click.option(
    "--max-electrons",
    type=click.IntRange(min=2),
    metavar="N",
    help="Count at which the run stops regardless, flagged statistics-limited.",
)
@click.option(
    "--block-electrons",
    type=click.IntRange(min=1),
    metavar="N",
    help="Electrons per transport block; the rule is checked at block ends.",
)
@click.option(
    "--observable",
    type=click.Choice(OBSERVABLES),
    multiple=True,
    help="Yield that must converge; repeat for both (default line).",
)
@click.option(
    "--max-electron-share",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    metavar="FRACTION",
    help="Guard: largest share of a yield one electron may hold (default 0.05).",
)
@click.option(
    "--min-effective-electrons",
    type=click.FloatRange(min=0.0),
    metavar="N",
    help="Guard: effective sample size floor (sum m)^2 / sum m^2 (default 100).",
)
@click.option(
    "--stability-blocks",
    type=click.IntRange(min=0),
    metavar="K",
    help="Guard: block ends whose running mean must agree; 0 disables (default 3).",
)
@click.option(
    "--stability-fraction",
    type=click.FloatRange(min=0.0),
    metavar="FRACTION",
    help="Guard: allowed relative mean drift as a fraction of the target (default 0.5).",
)
@click.option(
    "--pilot-electrons",
    type=click.IntRange(min=1),
    metavar="N",
    help="Pilot count that projects the stopping count and skips checks until it.",
)
@click.option(
    "--band-ev",
    callback=_energy_band,
    metavar="START,STOP",
    help="Monitor band in eV (default: the case's line band).",
)
@click.option(
    "--batch-means-band-ev",
    callback=_energy_band,
    metavar="START,STOP",
    help="Band in eV for reported per-bin batch-means errors; costs one extra reduction.",
)
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def precision_set_command(name, yes, dry_run, **values):
    """Create or update PROFILE's adaptive policy.

    A new policy needs --target-rse, --min-electrons, --max-electrons and
    --block-electrons. Remove fixed line/bremsstrahlung electron counts first
    with 'pyrite profile numerics reset'.
    """
    updates = {
        _PRECISION_FIELD_NAMES[key.replace("_", "-")]: value
        for key, value in values.items()
        if value is not None and value != ()
    }
    if not updates:
        raise click.UsageError("provide at least one precision option")
    if "observables" in updates:
        updates["observables"] = list(dict.fromkeys(updates["observables"]))
    try:
        original, document = _catalog_io.catalog_text()
        current = _profile_edit.profile_precision_values(
            _profile_edit.existing_profile(document, name)
        )
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    missing = [
        f"--{flag}"
        for flag, key in _PRECISION_FIELD_NAMES.items()
        if key in REQUIRED_FIELDS and key not in updates and key not in (current or {})
    ]
    if missing:
        raise click.UsageError(f"a new precision policy needs {', '.join(missing)}")
    try:
        _profile_edit.set_precision(document, name, updates)
    except (TypeError, ValueError) as exc:
        raise CLIError(str(exc)) from None
    confirm_standard(name, "set adaptive precision on", yes, dry_run)
    return write(document, original, dry_run, f"updated precision for profile {name}")


@command.command("reset")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("fields", nargs=-1, type=click.Choice(tuple(_PRECISION_FIELD_NAMES)))
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def precision_reset_command(name, fields, yes, dry_run):
    """Reset optional FIELDs to defaults, or remove the policy when none are named.

    Removing the policy returns PROFILE to fixed electron counts.
    """
    keys = tuple(_PRECISION_FIELD_NAMES[field] for field in fields)
    try:
        original, document = _catalog_io.catalog_text()
        removed = _profile_edit.reset_precision(document, name, keys)
    except (OSError, TypeError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if not removed:
        emit_result(f"profile {name}: nothing to reset")
        return 0
    confirm_standard(name, "reset adaptive precision on", yes, dry_run)
    done = "removed precision policy" if not keys else "reset precision fields"
    return write(document, original, dry_run, f"{done} for profile {name}")
