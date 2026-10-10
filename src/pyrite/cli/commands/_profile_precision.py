"""``pyrite profile precision``: a profile's adaptive electron-count policy."""

import click
from tomlkit.exceptions import ParseError

from pyrite._precision import OBSERVABLES
from pyrite.campaign import profile_edit as _profile_edit
from pyrite.campaign.profiles import get_fidelity_preset
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli.commands._profile_shared import confirm_standard, write
from pyrite.console import json as cli_json
from pyrite.console import output as _output
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


def _policy_options(function):
    """Attach the adaptive-policy field options shared by ``set`` and ``enable``."""
    options = (
        click.option(
            "--target-rse",
            type=click.FloatRange(min=0.0, min_open=True),
            metavar="FRACTION",
            help="Target relative standard error of each watched yield (0.05 = 5%).",
        ),
        click.option(
            "--min-electrons",
            type=click.IntRange(min=2),
            metavar="N",
            help="Smallest count at which the run may stop; a multiple of --block-electrons.",
        ),
        click.option(
            "--max-electrons",
            type=click.IntRange(min=2),
            metavar="N",
            help="Count at which the run stops regardless, flagged statistics-limited.",
        ),
        click.option(
            "--block-electrons",
            type=click.IntRange(min=1),
            metavar="N",
            help="Electrons per transport block; the rule is checked at block ends.",
        ),
        click.option(
            "--observable",
            type=click.Choice(OBSERVABLES),
            multiple=True,
            help=(
                "Yield that must converge; repeat for both. Replaces the watched set "
                "(a new policy watches line and brem)."
            ),
        ),
        click.option(
            "--max-electron-share",
            type=click.FloatRange(min=0.0, max=1.0, min_open=True),
            metavar="FRACTION",
            help="Guard: largest share of a yield one electron may hold (default 0.05).",
        ),
        click.option(
            "--min-effective-electrons",
            type=click.FloatRange(min=0.0),
            metavar="N",
            help="Guard: effective sample size floor (sum m)^2 / sum m^2 (default 100).",
        ),
        click.option(
            "--stability-blocks",
            type=click.IntRange(min=0),
            metavar="K",
            help="Guard: block ends whose running mean must agree; 0 disables (default 3).",
        ),
        click.option(
            "--stability-fraction",
            type=click.FloatRange(min=0.0),
            metavar="FRACTION",
            help="Guard: allowed relative mean drift as a fraction of the target (default 0.5).",
        ),
        click.option(
            "--pilot-electrons",
            type=click.IntRange(min=1),
            metavar="N",
            help="Pilot count that projects the stopping count and skips checks until it.",
        ),
        click.option(
            "--band-ev",
            callback=_energy_band,
            metavar="START,STOP",
            help="Monitor band in eV (default: the case's line band).",
        ),
        click.option(
            "--batch-means-band-ev",
            callback=_energy_band,
            metavar="START,STOP",
            help="Band in eV for reported per-bin batch-means errors; costs one extra reduction.",
        ),
    )
    for option in reversed(options):
        function = option(function)
    return function


def _updates(values):
    updates = {
        _PRECISION_FIELD_NAMES[key.replace("_", "-")]: value
        for key, value in values.items()
        if value is not None and value != ()
    }
    if "observables" in updates:
        updates["observables"] = list(dict.fromkeys(updates["observables"]))
    return updates


def _display(value):
    if value is None:
        return "none"
    if isinstance(value, (tuple, list)):
        return ",".join(f"{item:g}" if isinstance(item, float) else str(item) for item in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _interactive():
    return _output._stdin_is_tty()


def show_line(precision):
    """Return the `profile show` summary line for a resolved precision payload."""
    effective = precision["effective"]
    if effective is None:
        return f"  electron counts: fixed ({precision['fixed_reason']})"
    default = " (default)" if precision["source"] == "default" else ""
    return (
        f"  electron counts: adaptive{default}, target RSE {effective['target_rse']:g}, "
        f"{effective['min_electrons']}-{effective['max_electrons']} electrons"
    )


def _mode_line(document, name):
    """One-line resulting electron-count mode after an edit."""
    _explicit, effective, source, reason = _profile_edit.precision_resolution(
        _profile_edit.existing_profile(document, name)
    )
    if effective is None:
        return f"profile {name}: fixed electron counts ({reason})"
    return f"profile {name}: adaptive ({source} policy)"


@click.group("precision")
def command():
    """Inspect and edit PROFILE's electron-count mode: adaptive policy or fixed counts.

    Profiles without fixed electron counts or a policy run adaptive by default
    (target RSE 0.05, 200-20000 electrons, blocks of 100, line and
    bremsstrahlung). Each incoherent case stops when the relative standard error
    of its line (and optionally bremsstrahlung) yield meets the target and its
    heavy-tail guards pass, or at the maximum count flagged statistics-limited.
    One count serves line and bremsstrahlung transport. Coherent emission,
    cascades, physical detectors, GDF beams, grooves and fixed electron counts
    keep fixed counts.

    \b
      enable   switch to adaptive: drop fixed counts, optionally customize
      set      customize fields of an adaptive policy
      reset    drop customized fields, or the whole table (back to the default)
      disable  opt out: drop any policy and select fixed counts

    Electron sampling is independent of the line grid; see
    'pyrite profile line-grid'.
    """


@command.command("show")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@output_option
def precision_show_command(name, json_output):
    """Show PROFILE's electron-count mode, its source, blockers and next steps."""
    try:
        _text, document = _catalog_io.catalog_text()
        profile = _profile_edit.existing_profile(document, name)
        explicit, effective, source, fixed_reason = _profile_edit.precision_resolution(profile)
        counts = _profile_edit.fixed_counts(profile)
        blockers = _profile_edit.precision_blockers(profile)
        grid = _profile_edit.line_grid_sources(document, profile)
    except (OSError, TypeError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.precision.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    automatic_grid = grid["automatic_for_every_case"]
    if json_output:
        payload = {
            "profile": name,
            "mode": "fixed" if effective is None else "adaptive",
            "source": source,
            "fixed_reason": fixed_reason,
            "explicit": explicit,
            "effective": effective,
            "fixed_counts": {
                _profile_edit.SAMPLING_FLAGS[key].replace("-", "_"): value
                for key, value in counts.items()
            },
            "blockers": blockers,
            "line_grid_automatic_for_every_case": automatic_grid,
        }
        emit_json_result(cli_json.JsonResult("cxr.profile.precision.show", payload))
        return 0
    if effective is None:
        emit_result(f"[{name} precision] fixed electron counts: {fixed_reason}")
        if counts:
            emit_result(f"  counts: {_profile_edit.describe_counts(counts)}")
    else:
        emit_result(f"[{name} precision] adaptive ({source} policy)")
        for flag, key in _PRECISION_FIELD_NAMES.items():
            marker = "" if explicit is not None and key in explicit else " (default)"
            emit_result(f"  {flag}: {_display(effective[key])}{marker}")
        emit_result(
            "  per material, fixed counts, GDF beams and grooved faces keep fixed electron counts"
        )
    for blocker in blockers:
        emit_result(f"  adaptive unsupported: {blocker}")
    emit_result(
        "  line grid: "
        + (
            "automatic for every case (line-grid policy)"
            if automatic_grid
            else "explicit or stored grids first, automatic for the rest"
        )
        + f"; independent of electron sampling, see 'pyrite profile line-grid show {name}'"
    )
    if effective is not None:
        emit_result(f"  to select fixed counts: pyrite profile precision disable {name}")
    elif counts and not blockers:
        emit_result(f"  to switch to adaptive: pyrite profile precision enable {name}")
    return 0


@command.command("enable")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_policy_options
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def precision_enable_command(name, yes, dry_run, **values):
    """Switch PROFILE to adaptive electron counts in one atomic edit.

    Removes the profile's fixed line-trials/brem-trials. Without options the
    default policy applies (no table is written); options write a policy whose
    unspecified fields come from the existing policy, else the default.
    Refuses, writing nothing, when the profile cannot run adaptive (coherent
    emission, cascades, positrons, a physical detector, or per-material count
    overrides). Emission, cascades, beam, geometry and line-grid selectors are
    never changed.
    """
    try:
        original, document = _catalog_io.catalog_text()
        removed = _profile_edit.enable_precision(document, name, _updates(values))
    except (OSError, TypeError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if document.as_string() == original:
        emit_result(f"profile {name}: already adaptive; nothing to change")
        return 0
    confirm_standard(name, "enable adaptive precision on", yes, dry_run)
    status = write(document, original, dry_run, f"enabled adaptive precision for profile {name}")
    if not dry_run:
        if removed:
            emit_result(f"  removed fixed counts: {_profile_edit.describe_counts(removed)}")
        emit_result(_mode_line(document, name))
    return status


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_policy_options
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def precision_set_command(name, yes, dry_run, **values):
    """Create or update PROFILE's adaptive policy.

    Fields left unspecified keep the existing policy's values; a new policy takes
    them from the default (target RSE 0.05, 200-20000 electrons, blocks of 100,
    line and brem). A profile with fixed counts is refused; switch it with
    'pyrite profile precision enable'.
    """
    updates = _updates(values)
    if not updates:
        raise click.UsageError(
            "provide at least one precision option; "
            f"'pyrite profile precision enable {name}' selects the default policy"
        )
    try:
        original, document = _catalog_io.catalog_text()
        _profile_edit.set_precision(document, name, updates)
    except (OSError, TypeError, ValueError, ParseError) as exc:
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

    Removing the policy returns PROFILE to the default adaptive policy unless it
    cannot run adaptive. Fixed counts are untouched; select them with
    'pyrite profile precision disable'.
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
    status = write(document, original, dry_run, f"{done} for profile {name}")
    if not dry_run:
        emit_result(_mode_line(document, name))
    return status


@command.command("disable")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option(
    "--line-trials",
    type=click.IntRange(min=1),
    metavar="N",
    help="Fixed Monte Carlo electron histories for the line spectrum.",
)
@click.option(
    "--brem-trials",
    type=click.IntRange(min=1),
    metavar="N",
    help="Fixed Monte Carlo electron histories for bremsstrahlung.",
)
@click.option(
    "-y", "--yes", is_flag=True, help="Use defaults for missing counts; skip confirmations."
)
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def precision_disable_command(name, line_trials, brem_trials, yes, dry_run):
    """Opt PROFILE out of adaptive sampling: remove any policy, set fixed counts.

    --line-trials/--brem-trials replace that count. A count the profile already
    sets is otherwise kept, including a sweep grid. A count it lacks is prompted
    for on a terminal; without a terminal, or with --yes or --dry-run, it takes
    the full-fidelity default (300 line, 150 brem).
    """
    try:
        original, document = _catalog_io.catalog_text()
        current = _profile_edit.fixed_counts(_profile_edit.existing_profile(document, name))
    except (OSError, TypeError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    preset = get_fidelity_preset("full")
    defaults = {"n_electrons": preset.n_electrons, "n_electrons_brem": preset.n_electrons_brem}
    prompt = _interactive() and not (yes or dry_run)
    counts = {}
    for key, value in (("n_electrons", line_trials), ("n_electrons_brem", brem_trials)):
        if value is None and key in current:
            continue
        if value is None and prompt:
            value = click.prompt(
                _profile_edit.SAMPLING_FLAGS[key],
                default=defaults[key],
                type=click.IntRange(min=1),
                err=True,
            )
        counts[key] = defaults[key] if value is None else value
    try:
        removed = _profile_edit.disable_precision(document, name, counts)
        result = _profile_edit.fixed_counts(_profile_edit.existing_profile(document, name))
    except (OSError, TypeError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if document.as_string() == original:
        emit_result(f"profile {name}: already uses fixed counts; nothing to change")
        return 0
    confirm_standard(name, "select fixed electron counts on", yes, dry_run)
    status = write(
        document, original, dry_run, f"selected fixed electron counts for profile {name}"
    )
    if not dry_run:
        if removed is not None:
            emit_result("  removed precision policy")
        emit_result(f"  counts: {_profile_edit.describe_counts(result)}")
    return status
