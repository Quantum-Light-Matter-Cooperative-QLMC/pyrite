"""`cxr line-grid` command group.

Job verbs (``status``/``attach``/``logs``/``stop``) delegate to ``cxr_mc.remote``
for output byte-identical to ``cxr remote``; ``derive``/``submit``/``apply``/
``set``/``set-brem``/``defaults``/``show``/``regen-golden`` call the package
modules. Heavy modules (``derive``, ``golden``) import lazily inside handlers so
``cxr`` startup stays cheap.
"""

from __future__ import annotations

import math
import tomllib
from copy import copy
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import click

from cxr_mc import _cli_completion, cli_json, remote
from cxr_mc._cli_core import (
    POSITIVE_FLOAT,
    POSITIVE_INT,
    CLIError,
    emit_json_result,
    emit_result,
    invoke_legacy,
)
from cxr_mc.line_grid import apply, defaults, job


def _floats(s):
    return [float(x) for x in s.split(",")] if s else None


class _CSV(click.ParamType):
    """Comma-separated finite floats with an optional numeric domain."""

    name = "numbers"

    def __init__(
        self,
        label,
        *,
        lower=None,
        lower_open=False,
        upper=None,
        upper_inclusive=True,
        preserve_text=False,
    ):
        self.label = label
        self.lower = lower
        self.lower_open = lower_open
        self.upper = upper
        self.upper_inclusive = upper_inclusive
        self.preserve_text = preserve_text

    def convert(self, value, param, ctx):
        try:
            values = _floats(value)
        except (TypeError, ValueError):
            self.fail(f"{self.label} must be comma-separated numbers", param, ctx)
        if not values:
            self.fail(f"{self.label} requires at least one value", param, ctx)
        for item in values:
            if not math.isfinite(item):
                self.fail(f"{self.label} values must be finite", param, ctx)
            if self.lower is not None:
                lower_ok = item > self.lower if self.lower_open else item >= self.lower
                if not lower_ok:
                    self._fail_domain(param, ctx)
            if self.upper is not None:
                upper_ok = item <= self.upper if self.upper_inclusive else item < self.upper
                if not upper_ok:
                    self._fail_domain(param, ctx)
        return value if self.preserve_text else values

    def _fail_domain(self, param, ctx):
        if self.lower == 0 and self.lower_open and self.upper is None:
            self.fail(f"{self.label} values must be finite and positive", param, ctx)
        relation = "<=" if self.upper_inclusive else "<"
        self.fail(
            f"{self.label} values must satisfy {self.lower:g} <= value {relation} {self.upper:g}",
            param,
            ctx,
        )


_ENERGY_CSV_TEXT = _CSV("energy", lower=0, lower_open=True, preserve_text=True)
_THICKNESS_CSV_TEXT = _CSV("thickness", lower=0, lower_open=True, preserve_text=True)
_TILT_CSV_TEXT = _CSV("tilt", lower=0, upper=90, upper_inclusive=False, preserve_text=True)
_AZIMUTH_CSV_TEXT = _CSV("azimuth", lower=0, upper=360, upper_inclusive=True, preserve_text=True)
_TILT_CSV = _CSV("tilt", lower=0, upper=90, upper_inclusive=False)
_AZIMUTH_CSV = _CSV("azimuth", lower=0, upper=360, upper_inclusive=True)
_THICKNESS_CSV = _CSV("thickness", lower=0, lower_open=True)


def _pull_combined(json_name=None):
    """scp the combined derivation JSON back from the remote box; return local path."""
    name = json_name or job.DEFAULT_JSON_OUT
    job._validate_remote_output_name(name)
    local = name
    remote_path = remote.remote_path(name)
    remote._run(["scp", remote.scp_remote_path(remote_path), local])
    return local


# --- Click wiring -----------------------------------------------------------


def _raise_for_status(status):
    """Preserve nonzero legacy statuses under Click's standalone runner."""
    if isinstance(status, int) and not isinstance(status, bool) and status:
        raise click.exceptions.Exit(status)
    return status


def _invoke_callback(function, /, *args, **kwargs):
    """Route legacy exits through shared stderr/status compatibility."""
    status = invoke_legacy(lambda _namespace: function(*args, **kwargs))
    return _raise_for_status(status)


def _expected_failure(exc):
    message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
    raise CLIError(str(message)) from None


def _derive_options(function):
    function = click.option(
        "--set-default",
        is_flag=True,
        help="Persist supplied geometry, energies, and materials as future defaults.",
    )(function)
    function = click.option(
        "--thickness",
        type=_THICKNESS_CSV_TEXT,
        metavar="ANGSTROM,...",
        help="Crystal thicknesses in angstrom; comma-separated and positive.",
    )(function)
    function = click.option(
        "--azimuths",
        type=_AZIMUTH_CSV_TEXT,
        metavar="DEG,...",
        help="Azimuths in degrees [0, 360]; comma-separated.",
    )(function)
    function = click.option(
        "--tilts",
        type=_TILT_CSV_TEXT,
        metavar="DEG,...",
        help="Polar tilts in degrees [0, 90); comma-separated.",
    )(function)
    function = click.option(
        "--energies",
        type=_ENERGY_CSV_TEXT,
        metavar="KEV,...",
        help="Beam energies in keV; comma-separated and positive.",
    )(function)
    return click.option(
        "--materials",
        metavar="KEY,...",
        help="Material keys; comma-separated. Omit to use persistent defaults.",
        shell_complete=_cli_completion.complete_material_csv,
    )(function)


@click.group(name="line-grid", no_args_is_help=False)
def command():
    """Derive and manage per-material line-grid bounds.

    Geometry flags override persistent defaults for one run. Use ``--set-default``
    to persist supplied values.

    \b
    Examples:
      cxr line-grid derive --materials mose2,wse2 --energies 30,60
      cxr line-grid submit --materials mose2 --dry-run
      cxr line-grid show mose2
    """


@command.command("derive")
@_derive_options
@click.option(
    "--brem-step",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Bremsstrahlung grid spacing in eV; overrides persistent default.",
)
def derive_command(
    materials,
    energies,
    tilts,
    azimuths,
    thickness,
    set_default,
    brem_step,
):
    """Derive line-grid bounds locally."""
    from cxr_mc.line_grid import derive

    argv = []
    for flag, value in (
        ("materials", materials),
        ("energies", energies),
        ("tilts", tilts),
        ("azimuths", azimuths),
        ("thickness", thickness),
    ):
        if value:
            argv.extend((f"--{flag}", value))
    if brem_step is not None:
        argv.extend(("--brem-step", str(brem_step)))
    if set_default:
        argv.append("--set-default")
    return _invoke_callback(derive.main, argv)


@command.command("submit")
@_derive_options
@click.option(
    "--slice-minutes",
    type=POSITIVE_FLOAT,
    default=job.DEFAULT_SLICE_MINUTES,
    show_default=True,
    metavar="MINUTES",
    help="Maximum duration of each self-resubmitting remote slice.",
)
@click.option("--no-sync", is_flag=True, help="Skip code upload before submission.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print batch script and submission command; do not connect or submit.",
)
def submit_command(
    materials,
    energies,
    tilts,
    azimuths,
    thickness,
    set_default,
    slice_minutes,
    no_sync,
    dry_run,
):
    """Submit sliced line-grid derivation remotely."""
    return _invoke_callback(
        job.start,
        materials=materials or job.DEFAULT_MATERIALS,
        energies=energies or job.DEFAULT_ENERGIES,
        tilts=tilts,
        azimuths=azimuths,
        thickness=thickness,
        set_default=set_default,
        slice_minutes=slice_minutes,
        no_sync=no_sync,
        dry_run=dry_run,
    )


@click.command("status")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Add allocation detail; repeat for case progress and recent logs.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit one versioned JSON object on stdout.",
)
def status_command(jobid, verbose, json_output):
    """Show line-grid job status. JOBID defaults to latest recorded job."""
    if json_output:
        return _invoke_callback(
            remote._cli_status,
            SimpleNamespace(jobid=jobid, verbose=verbose, json_output=True),
        )
    remote.job_status(jobid, detail=verbose)
    return 0


@click.command("attach")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
def attach_command(jobid):
    """Attach to line-grid job progress. JOBID defaults to latest recorded job."""
    remote.attach(jobid)
    return 0


@click.command("logs")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    help="Stream live; Ctrl-C disconnects viewer without stopping job.",
)
def logs_command(jobid, follow):
    """Print or follow line-grid job logs. JOBID defaults to latest recorded job."""
    return _invoke_callback(remote.tail_logs, jobid, follow)


@click.command("stop")
@click.argument("jobid", required=False, metavar="[JOBID]")
@click.option("--latest", is_flag=True, help="Target latest recorded job instead of JOBID.")
@click.option("--yes", is_flag=True, help="Cancel exact previewed job; otherwise preview.")
def stop_command(jobid, latest, yes):
    """Preview or stop one line-grid job."""
    if jobid and latest:
        raise click.UsageError("stop takes JOBID or --latest, not both")
    if not jobid and not latest:
        raise click.UsageError("stop needs JOBID, or use --latest")
    resolved_jobid = jobid or remote._latest_jobid()
    if not resolved_jobid:
        raise CLIError("no jobs to stop")
    if not yes:
        emit_result(f"would cancel remote job: {resolved_jobid}")
        emit_result("re-run with --yes to cancel")
        return 0
    remote._stop_jobid(resolved_jobid)
    return 0


@click.group("job", no_args_is_help=True)
def job_command():
    """Inspect, follow, or stop remote line-grid jobs."""


for _job_child in (status_command, attach_command, logs_command, stop_command):
    job_command.add_command(_job_child)
command.add_command(job_command)

for _legacy_job_child in (status_command, attach_command, logs_command, stop_command):
    _alias = copy(_legacy_job_child)
    _alias.hidden = True
    command.add_command(_alias)


@command.command("apply")
@click.argument("json_path", required=False, metavar="JSON")
@click.option(
    "--materials",
    metavar="KEY,...",
    help="Apply only listed material keys.",
    shell_complete=_cli_completion.complete_material_csv,
)
@click.option(
    "--pull",
    is_flag=True,
    help="Fetch default combined JSON from remote host; takes precedence over JSON.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace manually overridden rows; otherwise preserve them.",
)
@click.option(
    "--regen-golden",
    is_flag=True,
    help="Regenerate checked catalog snapshot after successful write.",
)
@click.option("--dry-run", is_flag=True, help="Print proposed diff; write nothing.")
def apply_command(json_path, materials, pull, force, regen_golden, dry_run):
    """Apply derived bounds to material catalog.

    Writes packaged ``materials.toml`` and provenance atomically after validation.

    \b
    Example:
      cxr line-grid apply combined_line_grid_bounds.json --materials mose2,wse2
    """
    path = _pull_combined() if pull else json_path
    if not path:
        raise click.UsageError("no JSON: pass a path or --pull")
    try:
        apply.apply_file(
            path,
            materials=materials,
            force=force,
            dry_run=dry_run,
            date=str(date.today()),
            regen_golden=regen_golden,
        )
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    if regen_golden and not dry_run:
        from cxr_mc.line_grid import golden

        golden.regen()
    return 0


@command.command("set")
@click.argument("material", shell_complete=_cli_completion.complete_material)
@click.option(
    "--energy", type=POSITIVE_FLOAT, required=True, metavar="KEV", help="Beam energy in keV."
)
@click.option(
    "--stop", type=POSITIVE_FLOAT, required=True, metavar="EV", help="Line-grid upper bound in eV."
)
@click.option(
    "--num",
    type=POSITIVE_INT,
    metavar="N",
    help="Grid point count; preserve current value if omitted.",
)
@click.option(
    "--start",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Line-grid lower bound in eV; preserve current value if omitted.",
)
@click.option("--note", help="Provenance note stored with manual override.")
def set_command(material, energy, stop, num, start, note):
    """Set one material line-grid row and mark it as a manual override."""
    try:
        apply.set_line_grid(material, energy, stop, num=num, start_eV=start, note=note)
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    return 0


@command.command("set-brem")
@click.argument("material", shell_complete=_cli_completion.complete_material)
@click.option(
    "--stop",
    type=POSITIVE_FLOAT,
    required=True,
    metavar="EV",
    help="Bremsstrahlung grid upper bound in eV.",
)
@click.option(
    "--step",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Grid spacing in eV; preserve current value if omitted.",
)
@click.option("--note", help="Provenance note stored with manual override.")
def set_brem_command(material, stop, step, note):
    """Set one material bremsstrahlung grid and mark it as a manual override."""
    try:
        apply.set_brem_grid(material, stop, step_eV=step, note=note)
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    return 0


@command.command("defaults")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit one versioned JSON object on stdout (show mode only).",
)
@click.option(
    "--set",
    "set_values",
    is_flag=True,
    help="Persist supplied values; otherwise only show defaults.",
)
@click.option(
    "--tilts", type=_TILT_CSV, metavar="DEG,...", help="Persistent polar tilts in degrees [0, 90)."
)
@click.option(
    "--azimuths",
    type=_AZIMUTH_CSV,
    metavar="DEG,...",
    help="Persistent azimuths in degrees [0, 360].",
)
@click.option(
    "--thickness",
    type=_THICKNESS_CSV,
    metavar="ANGSTROM,...",
    help="Persistent positive crystal thicknesses in angstrom.",
)
@click.option(
    "--brem-step",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Persistent positive bremsstrahlung spacing in eV.",
)
def defaults_command(json_output, set_values, tilts, azimuths, thickness, brem_step):
    """Show or update persistent derivation defaults."""
    supplied = [
        flag
        for flag, value in (
            ("--tilts", tilts),
            ("--azimuths", azimuths),
            ("--thickness", thickness),
            ("--brem-step", brem_step),
        )
        if value is not None
    ]
    if supplied and not set_values:
        raise click.UsageError(f"{', '.join(supplied)} require --set")
    if json_output and set_values:
        raise click.UsageError("--json is read-only and cannot be combined with --set")
    if set_values:
        defaults.update_defaults(
            tilts=tilts,
            azimuths=azimuths,
            thickness_ang=thickness,
            brem_step_ev=brem_step,
        )
    if json_output:
        try:
            values = defaults.load_defaults()
            source = "persisted" if defaults.DEFAULTS_PATH.exists() else "fallback"
            result = cli_json.line_grid_defaults(values, source=source)
        except (OSError, TypeError, ValueError) as exc:
            result = cli_json.failure("cxr.line-grid.defaults", {}, str(exc))
        emit_json_result(result)
        return 0
    values = defaults.load_defaults()
    for key, value in values.items():
        emit_result(f"{key} = {value}")
    return 0


@command.command("show")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit one versioned JSON object on stdout.",
)
@click.argument(
    "material",
    required=False,
    shell_complete=_cli_completion.complete_material,
)
def show_command(json_output, material):
    """Show configured line grids."""
    if json_output:
        try:
            with Path(apply._MATERIALS_TOML).open("rb") as stream:
                materials = tomllib.load(stream)["materials"]
            result = cli_json.line_grid_show(materials, apply._provenance.load(), selected=material)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            result = cli_json.failure("cxr.line-grid.show", {"materials": []}, str(exc))
        emit_json_result(result)
        return 0
    try:
        result = apply.show(material)
    except ValueError as exc:
        raise CLIError(str(exc)) from None
    emit_result(result)
    return 0


@command.command("regen-golden")
@click.option(
    "--check",
    is_flag=True,
    help="Check snapshot for drift; do not write (exit 1 when stale).",
)
def regen_golden_command(check):
    """Regenerate or check material-catalog golden snapshot.

    Requires source checkout because installed wheels do not contain test data.
    """
    from cxr_mc.line_grid import golden

    return _invoke_callback(golden.regen, check=check)
