"""`pyrite energy-grid` command group.

Job verbs (``status``/``attach``/``logs``/``stop``) delegate to ``pyrite.remote``
for output byte-identical to ``pyrite remote``; ``derive``/``apply``/
``line set``/``brem set``/``defaults``/``show``/``regen-golden`` call the package
modules. Heavy modules (``derive``, ``golden``) import lazily inside handlers so
``pyrite`` startup stays cheap.
"""

import contextlib
import tempfile
from pathlib import Path

import click

from pyrite import remote
from pyrite.cli import _completion as _cli_completion
from pyrite.cli._deprecations import canonical_option
from pyrite.cli._options import remote_option
from pyrite.console import config as _cli_config
from pyrite.console import json as cli_json
from pyrite.console.output import (
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
    confirm_destructive,
    emit_diagnostic,
    emit_json_result,
    emit_result,
    invoke_legacy,
    output_option,
)
from pyrite.energy_grid import apply, defaults, job
from pyrite.energy_grid import gc as artifact_gc

#: ``--clear FIELD`` choices, mapped to persisted-defaults keys. D5 renamed the
#: matching flags, so the canonical field names are the singular ones; the
#: plurals stay accepted because they are values rather than flag spellings and
#: `RetiredOption` does not reach them.
_DEFAULT_FIELD_KEYS = {
    "polar": "tilts",
    "azimuth": "azimuths",
    "energy": "energies",
    "material": "materials",
    "thickness": "thickness_ang",
    "brem-step": "brem_step_ev",
    "tilts": "tilts",
    "azimuths": "azimuths",
    "energies": "energies",
    "materials": "materials",
}


def _pull_combined(json_name=None, *, dest_dir):
    """scp the combined derivation JSON into ``dest_dir``; return local path.

    ``json_name`` is the remote-side basename; the payload is consumed by
    ``apply.add_file`` and discarded, so callers pass a temporary directory
    rather than leaving a dated JSON in the working directory.
    """
    name = json_name or job.DEFAULT_JSON_OUT
    job._validate_remote_output_name(name)
    local = str(Path(dest_dir) / name)
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
    """D5 canonical geometry/selection flags, with their retired plural spellings.

    Destination names stay plural: they are the argv relay keys for the staged
    argparse handlers in `derive`/`job`, which are internal and not part of the
    surface D5 governs.
    """
    function = canonical_option(
        "--save-default",
        "set_default",
        is_flag=True,
        help="Persist supplied geometry, energies, and materials as future defaults.",
    )(function)
    function = click.option(
        "--thickness",
        type=THICKNESS_CSV_TEXT,
        metavar="ANGSTROM,...",
        help="Crystal thicknesses in angstrom; comma-separated and positive.",
    )(function)
    function = canonical_option(
        "--azimuth",
        "azimuths",
        type=AZIMUTH_CSV_TEXT,
        metavar="DEG,...",
        help="Azimuths in degrees [0, 360]; comma-separated.",
    )(function)
    function = canonical_option(
        "--polar",
        "tilts",
        type=TILT_CSV_TEXT,
        metavar="DEG,...",
        help="Polar tilts in degrees [0, 90); comma-separated.",
    )(function)
    function = canonical_option(
        "--energy",
        "energies",
        type=ENERGY_CSV_TEXT,
        metavar="KEV,...",
        help="Beam energies in keV; comma-separated and positive.",
    )(function)
    return canonical_option(
        "--material",
        "materials",
        metavar="KEY,...",
        help="Material keys; comma-separated. Omit to use persistent defaults.",
        shell_complete=_cli_completion.complete_material_csv,
    )(function)


@click.command("derive")
@_derive_options
@click.option(
    "--brem-step",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Derivation bremsstrahlung spacing in eV; overrides persistent default.",
)
@click.option(
    "--slice-minutes",
    type=POSITIVE_FLOAT,
    default=job.DEFAULT_SLICE_MINUTES,
    show_default=True,
    metavar="MINUTES",
    help="Maximum duration of each self-resubmitting remote slice.",
)
@click.option("--no-sync", is_flag=True, help="Skip code upload before remote submission.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print remote batch script and submission command; do not connect or submit.",
)
@click.option("--wait", is_flag=True, help="Wait for remote completion and pull the result.")
@click.option("--detach", is_flag=True, help="Return after remote submission.")
@click.option(
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Install derived grids for profile NAME; precedence: flag > configuration > standard.",
)
@remote_option
@click.pass_context
def derive_command(
    ctx,
    materials,
    energies,
    tilts,
    azimuths,
    thickness,
    set_default,
    brem_step,
    slice_minutes,
    no_sync,
    dry_run,
    wait,
    detach,
    catalog_profile,
    remote_target,
):
    """Derive and install line and bremsstrahlung grids locally or remotely."""
    if wait and detach:
        raise click.UsageError("--wait and --detach are mutually exclusive")
    remote_only = {
        "wait": "--wait",
        "detach": "--detach",
        "slice_minutes": "--slice-minutes",
        "no_sync": "--no-sync",
        "dry_run": "--dry-run",
    }
    if remote_target is None:
        explicit = [
            flag
            for parameter, flag in remote_only.items()
            if ctx.get_parameter_source(parameter) is click.core.ParameterSource.COMMANDLINE
        ]
        if explicit:
            raise click.UsageError(
                f"remote-only option(s) require -R/--remote: {', '.join(explicit)}"
            )
        return _derive_local(
            materials,
            energies,
            tilts,
            azimuths,
            thickness,
            set_default,
            brem_step,
            _cli_config.resolve("profile.current", catalog_profile).value,
        )
    return _derive_remote(
        materials=materials,
        energies=energies,
        tilts=tilts,
        azimuths=azimuths,
        thickness=thickness,
        set_default=set_default,
        brem_step=brem_step,
        slice_minutes=slice_minutes,
        no_sync=no_sync,
        dry_run=dry_run,
        detach=detach,
        remote_target=remote_target,
        persist_local=True,
        catalog_profile=_cli_config.resolve("profile.current", catalog_profile).value,
    )


def _derive_local(
    materials, energies, tilts, azimuths, thickness, set_default, brem_step, catalog_profile
):
    from pyrite.energy_grid import derive

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
    with tempfile.TemporaryDirectory() as workdir:
        json_out = str(Path(workdir) / job.DEFAULT_JSON_OUT)
        _invoke_callback(derive.main, [*argv, "--json-out", json_out])
        _install_derived(json_out, catalog_profile)
    return 0


def _install_derived(path: str, catalog_profile: str) -> None:
    try:
        apply.add_file(path, profile=catalog_profile)
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    emit_result(f"installed derived grids for profile {catalog_profile}")


def _derive_remote(
    *,
    materials,
    energies,
    tilts,
    azimuths,
    thickness,
    set_default,
    brem_step,
    slice_minutes,
    no_sync,
    dry_run,
    detach,
    remote_target,
    persist_local,
    catalog_profile,
):
    from pyrite.remote.config import override_remote_host

    persisted = defaults.load_defaults()
    if set_default and persist_local:
        persisted = defaults.update_defaults(
            materials=materials.split(",") if materials else None,
            energies=[float(value) for value in energies.split(",")] if energies else None,
            tilts=[float(value) for value in tilts.split(",")] if tilts else None,
            azimuths=[float(value) for value in azimuths.split(",")] if azimuths else None,
            thickness_ang=([float(value) for value in thickness.split(",")] if thickness else None),
            brem_step_ev=brem_step,
        )
    resolved_materials = (
        materials
        or ",".join(str(value) for value in persisted["materials"])
        or job.DEFAULT_MATERIALS
    )
    resolved_energies = (
        energies
        or ",".join(f"{float(value):g}" for value in persisted["energies"])
        or job.DEFAULT_ENERGIES
    )
    resolved_tilts = tilts or ",".join(f"{float(value):g}" for value in persisted["tilts"])
    resolved_azimuths = azimuths or ",".join(f"{float(value):g}" for value in persisted["azimuths"])
    resolved_thickness = thickness or ",".join(
        f"{float(value):g}" for value in persisted["thickness_ang"]
    )
    resolved_brem_step = brem_step if brem_step is not None else float(persisted["brem_step_ev"])
    with override_remote_host(None if remote_target == "__configured__" else remote_target):
        jobid = _invoke_callback(
            job.start,
            materials=resolved_materials,
            energies=resolved_energies,
            tilts=resolved_tilts or None,
            azimuths=resolved_azimuths or None,
            thickness=resolved_thickness or None,
            set_default=set_default,
            brem_step=resolved_brem_step,
            slice_minutes=slice_minutes,
            no_sync=no_sync,
            dry_run=dry_run,
        )
        if dry_run or detach:
            return 0
        if not remote.attach(jobid):
            emit_diagnostic(
                "energy-grid derivation is still active or its viewer disconnected; "
                "skipping automatic pull"
            )
            return 0
        if not remote._job_succeeded(jobid):
            raise CLIError("energy-grid derivation failed; skipping automatic pull")
        with tempfile.TemporaryDirectory() as workdir:
            path = _pull_combined(dest_dir=workdir)
            emit_result(f"pulled {Path(path).name}")
            _install_derived(path, catalog_profile)
    return 0


@click.command("add")
@click.argument("json_path", required=False, metavar="JSON")
@canonical_option(
    "--material",
    "materials",
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
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help=("Repoint profile NAME; precedence: flag > PYRITE_PROFILE > config store > standard."),
)
@click.option(
    "--regen-golden",
    is_flag=True,
    help="Regenerate checked catalog snapshot after successful write.",
)
@click.option("--dry-run", is_flag=True, help="Print proposed diff; write nothing.")
def add_command(json_path, materials, pull, force, catalog_profile, regen_golden, dry_run):
    """Add immutable derived-grid artifacts and repoint one profile.

    Consumes combined JSON from ``derive``. Artifact bytes are content-addressed
    and immutable; only the resolved profile's ``energy_grid_refs`` move.
    Legacy grid tables, scan ranges, and material overrides remain unchanged.

    Manual line and bremsstrahlung overrides remain unchanged unless
    ``--force`` is passed. This command does not run a scan and does not select
    ``full`` or ``survey`` fidelity.

    \b
    Example:
      pyrite-dev energy-grid add combined_line_grid_bounds.json --material mose2,wse2
    """
    with contextlib.ExitStack() as stack:
        if pull:
            workdir = stack.enter_context(tempfile.TemporaryDirectory())
            path = _pull_combined(dest_dir=workdir)
        else:
            path = json_path
        if not path:
            raise click.UsageError("no JSON: pass a path or --pull")
        resolved_profile = _cli_config.resolve("profile.current", catalog_profile).value
        try:
            apply.add_file(
                path,
                profile=resolved_profile,
                materials=materials,
                force=force,
                dry_run=dry_run,
            )
        except (KeyError, ValueError, OSError) as exc:
            _expected_failure(exc)
    if regen_golden and not dry_run:
        from pyrite.energy_grid import golden

        golden.regen()
    return 0


@click.command("set")
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
@click.option(
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Repoint profile NAME; precedence: flag > PYRITE_PROFILE > config store > standard.",
)
def set_command(material, energy, stop, num, start, note, catalog_profile):
    """Set one line-grid row by repointing an immutable artifact."""
    resolved_profile = _cli_config.resolve("profile.current", catalog_profile).value
    try:
        digest = apply.set_line_artifact(
            material,
            energy,
            stop,
            profile=resolved_profile,
            num=num,
            start_eV=start,
            note=note,
        )
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    emit_result(f"repointed {resolved_profile}/{material} -> {digest}")
    return 0


@click.command("rm")
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
@click.option("-y", "--yes", "yes", is_flag=True, help="Delete the exact previewed rows.")
@click.option("--dry-run", is_flag=True, help="Print proposed diff; delete nothing.")
@click.option(
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help=("Repoint profile NAME; precedence: flag > PYRITE_PROFILE > config store > standard."),
)
@output_option
def rm_command(material, energies, yes, dry_run, catalog_profile, json_output):
    """Remove line rows by repointing a profile to a new immutable artifact.

    Old artifact bytes remain recoverable until ``energy-grid gc`` reclaims
    them after its grace window.

    \b
    Example:
      pyrite-dev energy-grid rm wse2 --energy 30 --energy 40
    """
    if dry_run and json_output:
        raise click.UsageError("--dry-run and --output json cannot be combined")
    if json_output and not yes:
        raise click.UsageError(
            "--output json requires --yes; prompts are disabled in machine-output mode"
        )
    resolved_profile = _cli_config.resolve("profile.current", catalog_profile).value
    if dry_run:
        try:
            apply.remove_line_rows(material, energies, profile=resolved_profile, dry_run=True)
        except (KeyError, ValueError, OSError) as exc:
            _expected_failure(exc)
        return 0
    if not yes:
        energy_list = ", ".join(f"{e:g}" for e in energies)
        try:
            preview_original = apply._read_catalog(apply.active_catalog_path())
            apply.remove_line_rows(material, energies, profile=resolved_profile, dry_run=True)
        except (KeyError, ValueError, OSError) as exc:
            _expected_failure(exc)
        if not confirm_destructive(
            False,
            f"remove {len(energies)} line-grid row(s) from {resolved_profile}/{material} "
            f"at {energy_list} keV? old artifact remains recoverable until gc",
        ):
            return 0
    try:
        kwargs = {} if yes else {"expected_original": preview_original}
        deleted, digest = apply.remove_line_rows(
            material, energies, profile=resolved_profile, **kwargs
        )
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    if json_output:
        emit_json_result(
            cli_json.JsonResult(
                "cxr.energy-grid.rm",
                {
                    "material": material,
                    "profile": resolved_profile,
                    "removed_energies_keV": deleted,
                    "artifact_sha256": digest,
                },
            )
        )
        return 0
    emit_result(
        f"repointed {resolved_profile}/{material} -> {digest}: removed "
        f"{', '.join(f'{energy:g}' for energy in deleted)} keV"
    )
    return 0


@click.command("verify")
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    metavar="DIR",
    help="Checkpoint root whose campaign locks are reachability roots.",
)
def verify_command(checkpoint_dir):
    """Verify stored and profile/lock-referenced immutable artifacts."""
    try:
        report = artifact_gc.verify_artifacts(apply.active_catalog_path(), checkpoint_dir)
    except (OSError, ValueError, artifact_gc.ArtifactGCError) as exc:
        _expected_failure(exc)
    if not report.ok:
        details = "\n".join(
            f"- {issue.digest} ({issue.source}): {issue.message}" for issue in report.issues
        )
        raise CLIError(f"artifact verification failed:\n{details}")
    emit_result(
        f"verified {len(report.inventory)} stored artifact(s); {len(report.roots)} reachable ref(s)"
    )
    return 0


@click.command("gc")
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    metavar="DIR",
    help="Checkpoint root whose active/archive campaign locks remain reachable.",
)
@click.option(
    "--prune-all",
    is_flag=True,
    help="Ignore the 14-day orphan grace window and select every unreachable artifact.",
)
@click.option("-y", "--yes", is_flag=True, help="Delete the exact revalidated preview.")
def gc_command(checkpoint_dir, prune_all, yes):
    """Reclaim unreachable immutable artifacts after a 14-day grace window."""
    try:
        plan = artifact_gc.plan_gc(
            apply.active_catalog_path(),
            checkpoint_dir,
            prune_all=prune_all,
        )
    except (OSError, ValueError, artifact_gc.ArtifactGCError) as exc:
        _expected_failure(exc)
    if not plan.candidates:
        emit_result(
            f"nothing reclaimable ({len(plan.retained)} retained; "
            "new orphans enter the 14-day grace window)"
        )
        return 0
    emit_result("would delete unreachable energy-grid artifacts:")
    for candidate in plan.candidates:
        emit_result(f"  {candidate.path} (sha256={candidate.digest})")
    if not confirm_destructive(yes, "Delete these exact unreachable artifacts?"):
        return 0
    try:
        deleted = artifact_gc.execute_gc(plan)
    except (OSError, ValueError, artifact_gc.ArtifactGCError) as exc:
        _expected_failure(exc)
    emit_result(f"deleted {len(deleted)} unreachable artifact(s)")
    return 0


@click.command("set")
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
@click.option(
    "--spacing",
    type=click.Choice(["uniform", "geometric"]),
    default="uniform",
    show_default=True,
    help="Node spacing; geometric declares a nonuniform continuum override.",
)
@click.option(
    "--num",
    type=click.IntRange(min=2),
    metavar="N",
    help="Node count; required for --spacing geometric, rejected otherwise.",
)
@click.option(
    "--start",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Lowest node in eV for --spacing geometric; default is the medium's derived floor.",
)
@click.option("--note", help="Provenance note stored with manual override.")
@click.option(
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Repoint profile NAME; precedence: flag > PYRITE_PROFILE > config store > standard.",
)
def set_brem_command(material, stop, step, spacing, num, start, note, catalog_profile):
    """Set a bremsstrahlung grid by repointing an immutable artifact.

    --spacing geometric instead writes explicit nodes into the profile's
    override table, because artifact identity stores a uniform band only.
    """
    resolved_profile = _cli_config.resolve("profile.current", catalog_profile).value
    if spacing == "geometric":
        if step is not None:
            _expected_failure(
                ValueError("--step names a uniform band; drop it for --spacing geometric")
            )
        if num is None:
            _expected_failure(ValueError("--spacing geometric requires --num"))
        try:
            band = apply.set_brem_geometric(
                material,
                stop,
                num,
                profile=resolved_profile,
                start_eV=start,
                note=note,
            )
        except (KeyError, ValueError, OSError) as exc:
            _expected_failure(exc)
        emit_result(f"set {resolved_profile}/{material} brem grid -> {band}")
        return 0
    if num is not None:
        _expected_failure(ValueError("--num applies to --spacing geometric only"))
    if start is not None:
        _expected_failure(ValueError("--start applies to --spacing geometric only"))
    try:
        digest = apply.set_brem_artifact(
            material,
            stop,
            profile=resolved_profile,
            step_eV=step,
            note=note,
        )
    except (KeyError, ValueError, OSError) as exc:
        _expected_failure(exc)
    emit_result(f"repointed {resolved_profile}/{material} -> {digest}")
    return 0


@click.command("defaults")
@output_option
@canonical_option(
    "--save-default",
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
        "Fields: polar, azimuth, thickness, brem-step, energy, material."
    ),
)
@click.option(
    "--reset",
    is_flag=True,
    help="Reset every persistent derivation field to inherited/built-in behavior.",
)
@canonical_option(
    "--polar",
    "tilts",
    type=TILT_CSV,
    metavar="DEG,...",
    help="Persistent derivation polar tilts in degrees [0, 90).",
)
@canonical_option(
    "--azimuth",
    "azimuths",
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

    These values feed local and remote ``derive`` when matching options are
    omitted. Geometry searches determine both line and bremsstrahlung upper
    bounds; ``brem-step`` controls only applied bremsstrahlung spacing.

    Empty ``polar`` or ``azimuth`` mean inherit each material's catalog-profile
    angles. These are not physical scan defaults and do not select scan
    ``--fidelity full|survey``.
    """
    supplied = [
        flag
        for flag, value in (
            ("--polar", tilts),
            ("--azimuth", azimuths),
            ("--thickness", thickness),
            ("--brem-step", brem_step),
        )
        if value is not None
    ]
    if supplied and not set_values:
        raise click.UsageError(f"{', '.join(supplied)} require --save-default")
    if set_values and not supplied:
        raise click.UsageError("--save-default requires at least one value option")
    mutation_modes = int(set_values) + bool(clear_fields) + int(reset)
    if mutation_modes > 1:
        raise click.UsageError("--save-default, --clear, and --reset cannot be combined")
    if json_output and mutation_modes:
        raise click.UsageError("--output json is read-only and cannot be combined with mutations")
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


def _show(json_output, material, catalog_profile, *, band=None):
    """Show configured energy grids, optionally scoped to one band."""
    resolved_profile = _cli_config.resolve("profile.current", catalog_profile).value
    if json_output:
        try:
            raw, energy_grids, brem_by_material, refs = apply.resolved_show_inputs(
                profile=resolved_profile
            )
            materials = raw["materials"]
            result = cli_json.line_grid_show(
                materials,
                energy_grids,
                brem_by_material,
                apply._provenance.profile_records(resolved_profile),
                selected=material,
                band=band,
                profile=resolved_profile,
                artifact_refs=refs,
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            result = cli_json.failure(
                "cxr.energy-grid.show",
                {"profile": resolved_profile, "materials": []},
                str(exc),
            )
        emit_json_result(result)
        return 0
    try:
        result = apply.show(material, band=band, profile=resolved_profile)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CLIError(str(exc)) from None
    emit_result(result)
    return 0


@click.command("show")
@output_option
@click.argument("material", required=False, shell_complete=_cli_completion.complete_material)
@click.option(
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Resolve profile NAME; precedence: flag > PYRITE_PROFILE > config store > standard.",
)
def show_command(json_output, material, catalog_profile):
    """Show line and bremsstrahlung grids together."""
    return _show(json_output, material, catalog_profile)


@click.command("show")
@output_option
@click.argument("material", required=False, shell_complete=_cli_completion.complete_material)
@click.option(
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Resolve profile NAME; precedence: flag > PYRITE_PROFILE > config store > standard.",
)
def line_show_command(json_output, material, catalog_profile):
    """Show coherent line-energy grids."""
    return _show(json_output, material, catalog_profile, band="line")


@click.command("show")
@output_option
@click.argument("material", required=False, shell_complete=_cli_completion.complete_material)
@click.option(
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Resolve profile NAME; precedence: flag > PYRITE_PROFILE > config store > standard.",
)
def brem_show_command(json_output, material, catalog_profile):
    """Show bremsstrahlung energy grids."""
    return _show(json_output, material, catalog_profile, band="brem")
