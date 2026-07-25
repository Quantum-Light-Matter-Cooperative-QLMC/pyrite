"""`cxr line-grid` command group.

Job verbs (``status``/``attach``/``logs``/``stop``) delegate to ``cxr_mc.remote``
for output byte-identical to ``cxr remote``; ``derive``/``submit``/``apply``/
``set``/``set-brem``/``defaults``/``show``/``regen-golden`` call the package
modules. Heavy modules (``derive``, ``golden``) import lazily inside handlers so
``cxr`` startup stays cheap.
"""

from __future__ import annotations

import math
from datetime import date

import click

from cxr_mc import remote
from cxr_mc._cli_core import (
    POSITIVE_FLOAT,
    POSITIVE_INT,
    CLIError,
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
    function = click.option("--set-default", is_flag=True)(function)
    function = click.option("--thickness", type=_THICKNESS_CSV_TEXT)(function)
    function = click.option("--azimuths", type=_AZIMUTH_CSV_TEXT)(function)
    function = click.option("--tilts", type=_TILT_CSV_TEXT)(function)
    function = click.option("--energies", type=_ENERGY_CSV_TEXT)(function)
    return click.option("--materials")(function)


@click.group(name="line-grid")
def command():
    """Derive and manage per-material line-grid bounds."""


@command.command("derive")
@_derive_options
@click.option("--brem-step", type=POSITIVE_FLOAT)
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
@click.option("--slice-minutes", type=POSITIVE_FLOAT, default=job.DEFAULT_SLICE_MINUTES)
@click.option("--no-sync", is_flag=True)
@click.option("--dry-run", is_flag=True)
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


@command.command("status")
@click.argument("jobid", required=False)
@click.option("-v", "--verbose", count=True)
def status_command(jobid, verbose):
    """Show line-grid job status."""
    remote.job_status(jobid, detail=verbose)
    return 0


@command.command("attach")
@click.argument("jobid", required=False)
def attach_command(jobid):
    """Attach to line-grid job progress."""
    remote.attach(jobid)
    return 0


@command.command("logs")
@click.argument("jobid", required=False)
@click.option("-f", "--follow", is_flag=True)
def logs_command(jobid, follow):
    """Print or follow line-grid job logs."""
    return _invoke_callback(remote.tail_logs, jobid, follow)


@command.command("stop")
@click.argument("jobid", required=False)
def stop_command(jobid):
    """Stop one line-grid job."""
    resolved_jobid = jobid or remote._latest_jobid()
    if not resolved_jobid:
        raise CLIError("no jobs to stop")
    remote._stop_jobid(resolved_jobid)
    return 0


@command.command("apply")
@click.argument("json_path", required=False, metavar="JSON")
@click.option("--materials")
@click.option("--pull", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--regen-golden", is_flag=True)
@click.option("--dry-run", is_flag=True)
def apply_command(json_path, materials, pull, force, regen_golden, dry_run):
    """Apply derived bounds to material catalog."""
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
@click.argument("material")
@click.option("--energy", type=POSITIVE_FLOAT, required=True)
@click.option("--stop", type=POSITIVE_FLOAT, required=True)
@click.option("--num", type=POSITIVE_INT)
@click.option("--start", type=POSITIVE_FLOAT)
@click.option("--note")
def set_command(material, energy, stop, num, start, note):
    """Set one material line-grid row."""
    try:
        apply.set_line_grid(material, energy, stop, num=num, start_eV=start, note=note)
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    return 0


@command.command("set-brem")
@click.argument("material")
@click.option("--stop", type=POSITIVE_FLOAT, required=True)
@click.option("--step", type=POSITIVE_FLOAT)
@click.option("--note")
def set_brem_command(material, stop, step, note):
    """Set one material bremsstrahlung grid."""
    try:
        apply.set_brem_grid(material, stop, step_eV=step, note=note)
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    return 0


@command.command("defaults")
@click.option("--set", "set_values", is_flag=True)
@click.option("--tilts", type=_TILT_CSV)
@click.option("--azimuths", type=_AZIMUTH_CSV)
@click.option("--thickness", type=_THICKNESS_CSV)
@click.option("--brem-step", type=POSITIVE_FLOAT)
def defaults_command(set_values, tilts, azimuths, thickness, brem_step):
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
    if set_values:
        defaults.update_defaults(
            tilts=tilts,
            azimuths=azimuths,
            thickness_ang=thickness,
            brem_step_ev=brem_step,
        )
    for key, value in defaults.load_defaults().items():
        emit_result(f"{key} = {value}")
    return 0


@command.command("show")
@click.argument("material", required=False)
def show_command(material):
    """Show configured line grids."""
    try:
        result = apply.show(material)
    except ValueError as exc:
        raise CLIError(str(exc)) from None
    emit_result(result)
    return 0


@command.command("regen-golden")
@click.option("--check", is_flag=True)
def regen_golden_command(check):
    """Regenerate or check material-catalog golden snapshot."""
    from cxr_mc.line_grid import golden

    return _invoke_callback(golden.regen, check=check)
