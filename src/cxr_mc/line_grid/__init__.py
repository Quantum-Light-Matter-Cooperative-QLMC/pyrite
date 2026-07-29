"""`cxr energy-grid` command group.

Job verbs (``status``/``attach``/``logs``/``stop``) delegate to ``cxr_mc.remote``
for output byte-identical to ``cxr remote``; ``derive``/``submit``/``apply``/
``line set``/``brem set``/``defaults``/``show``/``regen-golden`` call the package
modules. Heavy modules (``derive``, ``golden``) import lazily inside handlers so
``cxr`` startup stays cheap.
"""

from __future__ import annotations

import tomllib
from copy import copy
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import click

from cxr_mc import remote
from cxr_mc.cli import _completion as _cli_completion
from cxr_mc.cli import json as cli_json
from cxr_mc.cli._core import (
    AZIMUTH_CSV,
    AZIMUTH_CSV_TEXT,
    ENERGY_CSV_TEXT,
    POSITIVE_FLOAT,
    POSITIVE_INT,
    THICKNESS_CSV,
    THICKNESS_CSV_TEXT,
    TILT_CSV,
    TILT_CSV_TEXT,
    CLIError,
    emit_json_result,
    emit_result,
    invoke_legacy,
)
from cxr_mc.line_grid import apply, defaults, job

_DEFAULT_FIELD_KEYS = {
    "tilts": "tilts",
    "azimuths": "azimuths",
    "thickness": "thickness_ang",
    "brem-step": "brem_step_ev",
    "energies": "energies",
    "materials": "materials",
}


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
        type=THICKNESS_CSV_TEXT,
        metavar="ANGSTROM,...",
        help="Crystal thicknesses in angstrom; comma-separated and positive.",
    )(function)
    function = click.option(
        "--azimuths",
        type=AZIMUTH_CSV_TEXT,
        metavar="DEG,...",
        help="Azimuths in degrees [0, 360]; comma-separated.",
    )(function)
    function = click.option(
        "--tilts",
        type=TILT_CSV_TEXT,
        metavar="DEG,...",
        help="Polar tilts in degrees [0, 90); comma-separated.",
    )(function)
    function = click.option(
        "--energies",
        type=ENERGY_CSV_TEXT,
        metavar="KEV,...",
        help="Beam energies in keV; comma-separated and positive.",
    )(function)
    return click.option(
        "--materials",
        metavar="KEY,...",
        help="Material keys; comma-separated. Omit to use persistent defaults.",
        shell_complete=_cli_completion.complete_material_csv,
    )(function)


@click.group(name="energy-grid", no_args_is_help=False)
def command():
    """Derive and manage per-material photon-energy grids.

    ``derive`` and ``submit`` measure both coherent-line and bremsstrahlung
    upper bounds. ``defaults`` controls that diagnostic derivation only;
    ``apply`` writes validated bounds into the material catalog. Physical scan
    profile defaults belong to ``cxr profile``; per-material range overrides
    belong to ``cxr material``.

    Scan ``--fidelity full|survey`` is separate. It controls later simulation
    cost and grid reduction; it never changes derivation or applied full bounds.

    Command-line derivation values override persistent defaults for one run.

    \b
    Examples:
      cxr energy-grid derive --materials mose2,wse2 --energies 30,60
      cxr energy-grid submit --materials mose2 --dry-run
      cxr energy-grid show mose2
    """


@command.group("line", no_args_is_help=True)
def line_command():
    """Inspect or manually set coherent line-energy grids."""


@command.group("brem", no_args_is_help=True)
def brem_command():
    """Inspect or manually set bremsstrahlung energy grids."""


@command.command("derive")
@_derive_options
@click.option(
    "--brem-step",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Derivation bremsstrahlung spacing in eV; overrides persistent default.",
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
    """Derive line and bremsstrahlung energy-grid bounds locally."""
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
    """Submit sliced line and bremsstrahlung bound derivation remotely."""
    persisted = defaults.load_defaults()
    return _invoke_callback(
        job.start,
        materials=materials
        or ",".join(str(value) for value in persisted["materials"])
        or job.DEFAULT_MATERIALS,
        energies=energies
        or ",".join(f"{float(value):g}" for value in persisted["energies"])
        or job.DEFAULT_ENERGIES,
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
    """Show energy-grid job status. JOBID defaults to latest recorded job."""
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
    """Attach to energy-grid job progress. JOBID defaults to latest recorded job."""
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
    """Print or follow energy-grid job logs. JOBID defaults to latest recorded job."""
    return _invoke_callback(remote.tail_logs, jobid, follow)


@click.command("stop")
@click.argument("jobid", required=False, metavar="[JOBID]")
@click.option("--latest", is_flag=True, help="Target latest recorded job instead of JOBID.")
@click.option("--yes", is_flag=True, help="Cancel exact previewed job; otherwise preview.")
def stop_command(jobid, latest, yes):
    """Preview or stop one energy-grid job."""
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
    """Inspect, follow, or stop remote energy-grid jobs."""


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

    Consumes combined JSON from ``derive``/``submit``. Writes line bounds into
    the shared per-material grid store, bremsstrahlung bounds into standard
    profile overrides, and adds derived beam energies to the standard profile.
    Catalog and provenance writes are atomic and validated.

    Manual line and bremsstrahlung overrides remain unchanged unless
    ``--force`` is passed. This command does not run a scan and does not select
    ``full`` or ``survey`` fidelity.

    \b
    Example:
      cxr energy-grid apply combined_line_grid_bounds.json --materials mose2,wse2
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


@line_command.command("set")
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


@line_command.command("delete")
@click.argument("material", shell_complete=_cli_completion.complete_material)
@click.option(
    "--energy",
    "energies",
    type=POSITIVE_FLOAT,
    metavar="KEV",
    multiple=True,
    required=True,
    help="Beam energy in keV; repeat for multiple rows.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed diff; delete nothing.")
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def delete_command(material, energies, yes, dry_run, json_output):
    """Delete MATERIAL's derived or manual line-grid rows; irreversible.

    The only way to remove bounds from the shared per-material derived-grid
    store -- editing a profile's energies never deletes them. Refuses (as a
    catalog validation failure) when a beam energy is still required by a
    profile's ``energy_keV`` grid.

    \b
    Example:
      cxr energy-grid line delete wse2 --energy 30 --energy 40
    """
    if dry_run and json_output:
        raise click.UsageError("--dry-run and --json cannot be combined")
    if json_output and not yes:
        raise click.UsageError("--json requires --yes; prompts are disabled in machine-output mode")
    if dry_run:
        try:
            apply.delete_line_grid(material, energies, dry_run=True)
        except (KeyError, ValueError, OSError) as exc:
            _expected_failure(exc)
        return 0
    if not yes:
        energy_list = ", ".join(f"{e:g}" for e in energies)
        click.confirm(
            f"delete {len(energies)} line-grid row(s) for {material} at {energy_list} keV? "
            "this cannot be undone",
            err=True,
            abort=True,
        )
    try:
        deleted = apply.delete_line_grid(material, energies)
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    if json_output:
        emit_json_result(
            cli_json.JsonResult(
                "cxr.energy-grid.line-delete",
                {"material": material, "deleted_energies_keV": deleted},
            )
        )
        return 0
    emit_result(f"deleted {material}: {', '.join(f'{e:g}' for e in deleted)} keV")
    apply._warn_stale_golden()
    return 0


@brem_command.command("set")
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
    "--clear",
    "clear_fields",
    type=click.Choice(tuple(_DEFAULT_FIELD_KEYS), case_sensitive=True),
    multiple=True,
    metavar="FIELD",
    help=(
        "Reset one field to inherited/built-in behavior; repeatable. "
        "Fields: tilts, azimuths, thickness, brem-step, energies, materials."
    ),
)
@click.option(
    "--reset",
    is_flag=True,
    help="Reset every persistent derivation field to inherited/built-in behavior.",
)
@click.option(
    "--tilts",
    type=TILT_CSV,
    metavar="DEG,...",
    help="Persistent derivation polar tilts in degrees [0, 90).",
)
@click.option(
    "--azimuths",
    type=AZIMUTH_CSV,
    metavar="DEG,...",
    help="Persistent azimuths in degrees [0, 360].",
)
@click.option(
    "--thickness",
    type=THICKNESS_CSV,
    metavar="ANGSTROM,...",
    help="Persistent positive crystal thicknesses in angstrom.",
)
@click.option(
    "--brem-step",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Persistent derivation bremsstrahlung spacing in eV.",
)
def defaults_command(
    json_output,
    set_values,
    clear_fields,
    reset,
    tilts,
    azimuths,
    thickness,
    brem_step,
):
    """Show, update, or clear persistent derivation inputs.

    These values feed ``derive`` and ``submit`` when their matching options are
    omitted. Geometry searches determine both line and bremsstrahlung upper
    bounds; ``brem-step`` controls only applied bremsstrahlung spacing.

    Empty ``tilts`` or ``azimuths`` mean inherit each material's catalog-profile
    angles. These are not physical scan defaults and do not select scan
    ``--fidelity full|survey``.
    """
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
    if set_values and not supplied:
        raise click.UsageError("--set requires at least one value option")
    mutation_modes = int(set_values) + bool(clear_fields) + int(reset)
    if mutation_modes > 1:
        raise click.UsageError("--set, --clear, and --reset cannot be combined")
    if json_output and mutation_modes:
        raise click.UsageError("--json is read-only and cannot be combined with mutations")
    if set_values:
        defaults.update_defaults(
            tilts=tilts,
            azimuths=azimuths,
            thickness_ang=thickness,
            brem_step_ev=brem_step,
        )
    elif clear_fields:
        defaults.reset_defaults(*(_DEFAULT_FIELD_KEYS[field] for field in clear_fields))
    elif reset:
        defaults.reset_defaults()
    if json_output:
        try:
            values = defaults.load_defaults()
            source = "persisted" if defaults.DEFAULTS_PATH.exists() else "fallback"
            result = cli_json.line_grid_defaults(values, source=source)
        except (OSError, TypeError, ValueError) as exc:
            result = cli_json.failure("cxr.energy-grid.defaults", {}, str(exc))
        emit_json_result(result)
        return 0
    values = defaults.load_defaults()
    for key, value in values.items():
        suffix = ""
        if key == "tilts" and not value:
            suffix = " (inherit each material's catalog-profile polar tilts)"
        elif key == "azimuths" and not value:
            suffix = " (inherit each material's catalog-profile azimuths)"
        emit_result(f"{key} = {value}{suffix}")
    return 0


def _show(json_output, material, *, band=None):
    """Show configured energy grids, optionally scoped to one band."""
    if json_output:
        try:
            with Path(apply._MATERIALS_TOML).open("rb") as stream:
                raw = tomllib.load(stream)
            materials = raw["materials"]
            energy_grids = raw.get("energy_grids", {})
            brem_by_material = {key: apply.effective_brem(raw, key) for key in materials}
            result = cli_json.line_grid_show(
                materials,
                energy_grids,
                brem_by_material,
                apply._provenance.load(),
                selected=material,
                band=band,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            result = cli_json.failure("cxr.energy-grid.show", {"materials": []}, str(exc))
        emit_json_result(result)
        return 0
    try:
        result = apply.show(material, band=band)
    except ValueError as exc:
        raise CLIError(str(exc)) from None
    emit_result(result)
    return 0


@command.command("show")
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
@click.argument("material", required=False, shell_complete=_cli_completion.complete_material)
def show_command(json_output, material):
    """Show line and bremsstrahlung grids together."""
    return _show(json_output, material)


@line_command.command("show")
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
@click.argument("material", required=False, shell_complete=_cli_completion.complete_material)
def line_show_command(json_output, material):
    """Show coherent line-energy grids."""
    return _show(json_output, material, band="line")


@brem_command.command("show")
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
@click.argument("material", required=False, shell_complete=_cli_completion.complete_material)
def brem_show_command(json_output, material):
    """Show bremsstrahlung energy grids."""
    return _show(json_output, material, band="brem")


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
