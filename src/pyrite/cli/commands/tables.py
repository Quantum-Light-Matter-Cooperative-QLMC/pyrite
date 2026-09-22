"""Inspect generated cross-section tables and the external code trees.

Covers the store and the source configuration: what tables exist, where they
came from, where they live, and which code trees PyRITE can currently reach.

``generate`` exposes ELSEPA's free-atom path and SBETHE's material path,
whose options are disjoint: each is rejected for the other code rather than
silently ignored. ``fetch`` installs
SBETHE's pinned reference database without extracting the archive's prebuilt
executable or documentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import click

from ...console.output import CLIError, emit_json, emit_result, output_option
from .._groups import LazyGroup

#: Projectile names accepted by ``--projectile``. Spelled here rather than
#: imported so building the command group stays free of ``xsgen`` imports;
#: a test pins the two lists together.
_PROJECTILES = (
    "alpha",
    "antimuon",
    "antiproton",
    "electron",
    "muon",
    "positron",
    "proton",
)

#: Option spellings, for the two messages below.
_FLAGS = {
    "element": "--element",
    "energy": "--energy",
    "name": "--name",
    "element_count": "--element-count",
    "density": "--density",
    "mean_excitation": "--mean-excitation",
    "band_gap": "--band-gap",
}


def _elsepa_args(
    code: str, element: int | None, energies_ev: tuple[float, ...]
) -> tuple[int, tuple[float, ...]]:
    """Return ELSEPA's required options, or fail naming every missing one."""
    if element is None or not energies_ev:
        missing = [
            flag
            for flag, given in (("--element", element), ("--energy", energies_ev or None))
            if given is None
        ]
        raise CLIError(f"--code {code} requires {', '.join(missing)}")
    return element, energies_ev


def _sbethe_args(
    code: str,
    name: str | None,
    element_counts: tuple[str, ...],
    density: float | None,
    mean_excitation: float | None,
) -> tuple[str, dict[int, float], float, float]:
    """Return SBETHE's required options, or fail naming every missing one."""
    if name is None or not element_counts or density is None or mean_excitation is None:
        missing = [
            flag
            for flag, given in (
                ("--name", name),
                ("--element-count", element_counts or None),
                ("--density", density),
                ("--mean-excitation", mean_excitation),
            )
            if given is None
        ]
        raise CLIError(f"--code {code} requires {', '.join(missing)}")
    return name, _composition(element_counts), density, mean_excitation


def _reject_for(code: str, **given: object) -> None:
    """Fail when an option belonging to the other code was supplied.

    Ignoring it silently would generate a table for a target the caller did
    not ask for, which is the failure this command exists to make impossible.
    """
    extra = [_FLAGS[key] for key, value in given.items() if value is not None]
    if extra:
        raise CLIError(f"--code {code} does not accept {', '.join(sorted(extra))}")


def _composition(pairs: tuple[str, ...]) -> dict[int, float]:
    """Parse repeated ``Z:N`` options into a composition mapping."""
    composition: dict[int, float] = {}
    for pair in pairs:
        atomic_number, separator, count = pair.partition(":")
        if not separator:
            raise CLIError(f"--element-count expects Z:N, got {pair!r}")
        try:
            z = int(atomic_number)
            n = float(count)
        except ValueError as exc:
            raise CLIError(f"--element-count expects Z:N with numeric parts, got {pair!r}") from exc
        if z in composition:
            raise CLIError(f"--element-count repeats Z={z}")
        composition[z] = n
    return composition


@dataclass(frozen=True)
class _SourceRow:
    """One code's resolution status, for both renderings.

    A dataclass rather than a dict so the two renderers agree on the field
    names and ``missing_data_dirs`` stays a list of strings rather than
    ``object``.
    """

    code: str
    status: str
    origin: str | None
    root: str | None
    missing_data_dirs: tuple[str, ...]
    detail: str | None

    def payload(self) -> dict[str, object]:
        """Return the JSON spelling of this row."""
        return {
            "code": self.code,
            "status": self.status,
            "origin": self.origin,
            "root": self.root,
            "missing_data_dirs": list(self.missing_data_dirs),
            "detail": self.detail,
        }


def _source_rows() -> list[_SourceRow]:
    """Resolve every code, as rows, with failures represented not raised."""
    from ...xsgen.sources import ResolvedSource, iter_sources, missing_data_dirs

    rows: list[_SourceRow] = []
    for name, outcome in iter_sources():
        if isinstance(outcome, ResolvedSource):
            missing = missing_data_dirs(outcome)
            rows.append(
                _SourceRow(
                    code=name,
                    status="ready" if not missing else "incomplete",
                    origin=outcome.origin,
                    root=str(outcome.root),
                    missing_data_dirs=missing,
                    detail=None,
                )
            )
        else:
            rows.append(
                _SourceRow(
                    code=name,
                    status="missing",
                    origin=None,
                    root=None,
                    missing_data_dirs=(),
                    detail=outcome,
                )
            )
    return rows


@click.group("tables", cls=LazyGroup)
def command() -> None:
    """Inspect generated cross-section tables and external code trees.

    Tables are produced by external Fortran codes (ELSEPA, SBETHE, BremsLib)
    and resolved in two tiers: your own tables first, then the tables shipped
    with PyRITE. Consumers cannot tell the two apart.

    \b
    Examples:
      pyrite tables path
      pyrite tables list
      pyrite tables show 4f3a9c
      pyrite tables generate --code elsepa --element 79 --energy 1e3
      pyrite tables fetch sbethe
      pyrite tables sources list
      pyrite tables sources set elsepa ../elsepa-2020
    """


@command.command("path")
@output_option
def path_command(json_output: bool) -> None:
    """Print the directory your generated tables are written to.

    Tables live in your user data directory rather than the workspace: they are
    expensive and target-scoped, not run-specific, so they are shared across
    every workspace.
    """
    from ...xsgen.store import packaged_table_dir, user_table_dir

    user = user_table_dir()
    if json_output:
        emit_json(
            "pyrite.tables.path.v1",
            {"user": str(user), "packaged": str(packaged_table_dir())},
        )
        return
    # One bare line, so `cd "$(pyrite tables path)"` works.
    emit_result(str(user))


@command.command("list")
@output_option
def list_command(json_output: bool) -> None:
    """List stored tables, most-preferred tier first.

    A table present in both tiers is listed once, as the tier that would be
    served.
    """
    from ...xsgen.store import iter_stored

    rows = [
        {
            "key": table.key,
            "code": str(table.manifest.get("code", "")),
            "quantity": str(table.manifest.get("quantity", "")),
            "target": str(table.manifest.get("target_label", "")),
            "tier": table.tier,
        }
        for table in iter_stored()
    ]
    if json_output:
        emit_json("pyrite.tables.list.v1", {"tables": rows})
        return
    emit_result("KEY\tCODE\tQUANTITY\tTARGET\tTIER")
    for row in rows:
        emit_result(
            f"{row['key']}\t{row['code']}\t{row['quantity']}\t{row['target']}\t{row['tier']}"
        )


@command.command("show")
@click.argument("key")
@output_option
def show_command(key: str, json_output: bool) -> None:
    """Print the provenance manifest for the table named by KEY.

    KEY may be an unambiguous prefix of a table key, the way a commit is named
    by its short hash.
    """
    from ...xsgen.store import iter_stored

    matches = [table for table in iter_stored() if table.key.startswith(key)]
    if not matches:
        raise CLIError(f"no stored table whose key starts with {key!r}; see `pyrite tables list`")
    if len(matches) > 1:
        found = ", ".join(sorted(table.key[:12] for table in matches))
        raise CLIError(f"{key!r} is ambiguous; it matches {len(matches)} tables: {found}")

    table = matches[0]
    manifest = dict(table.manifest)
    if json_output:
        emit_json("pyrite.tables.show.v1", {"tier": table.tier, "manifest": manifest})
        return
    emit_result(json.dumps(manifest, indent=2, sort_keys=True))


@command.command("generate")
@click.option(
    "--code",
    type=click.Choice(["elsepa", "sbethe"], case_sensitive=False),
    required=True,
    help="External code to run: ELSEPA free atoms, or SBETHE materials.",
)
@click.option(
    "--element",
    type=click.IntRange(1, 103),
    metavar="Z",
    help="Atomic number of the free-atom target. ELSEPA only.",
)
@click.option(
    "--energy",
    "energies_ev",
    type=click.FloatRange(min=4.999),
    multiple=True,
    metavar="EV",
    help="Kinetic energy in eV; repeat for a native-grid table. ELSEPA only.",
)
@click.option(
    "--name",
    metavar="NAME",
    help="Material name recorded in the SBETHE output headers. SBETHE only.",
)
@click.option(
    "--element-count",
    "element_counts",
    multiple=True,
    metavar="Z:N",
    help="Stoichiometric index of one element, as Z:N; repeat per element. SBETHE only.",
)
@click.option(
    "--density",
    type=click.FloatRange(min=0.0, min_open=True),
    metavar="G_CM3",
    help="Mass density in g/cm^3. SBETHE only.",
)
@click.option(
    "--mean-excitation",
    type=click.FloatRange(min=1.0, min_open=True),
    metavar="EV",
    help="Mean excitation energy in eV. SBETHE only.",
)
@click.option(
    "--band-gap",
    type=click.FloatRange(min=0.0, min_open=True),
    metavar="EV",
    help="Gap energy for an insulator or semiconductor; omit for a conductor. SBETHE only.",
)
@click.option(
    "--projectile",
    type=click.Choice(sorted(_PROJECTILES), case_sensitive=False),
    default="electron",
    show_default=True,
    help="Projectile particle. SBETHE only.",
)
@click.option("--overwrite", is_flag=True, help="Regenerate and replace an existing key.")
@click.option(
    "--keep-on-failure",
    is_flag=True,
    help="Keep the scratch directory after an external-code failure.",
)
@output_option
def generate_command(
    code: str,
    element: int | None,
    energies_ev: tuple[float, ...],
    name: str | None,
    element_counts: tuple[str, ...],
    density: float | None,
    mean_excitation: float | None,
    band_gap: float | None,
    projectile: str,
    overwrite: bool,
    keep_on_failure: bool,
    json_output: bool,
) -> None:
    """Generate or reuse one external-code table.

    ELSEPA takes ``--element`` and one or more ``--energy`` values in eV.
    SBETHE takes ``--name``, ``--density``, ``--mean-excitation`` and one
    ``--element-count Z:N`` per element in the molecule.

    Generated files live in the user table store; rerunning the same
    normalized request reuses its table without compiling or running the
    external code.
    """
    from ...xsgen import XsgenError

    selected = code.lower()
    try:
        if selected == "elsepa":
            from ...xsgen.elsepa import generate_element

            _reject_for(
                selected,
                name=name,
                element_count=element_counts or None,
                density=density,
                mean_excitation=mean_excitation,
                band_gap=band_gap,
            )
            atomic_number, energies = _elsepa_args(selected, element, energies_ev)
            result = generate_element(
                atomic_number,
                energies,
                overwrite=overwrite,
                keep_on_failure=keep_on_failure,
            )
        else:
            from ...xsgen.sbethe import generate_material

            _reject_for(selected, element=element, energy=energies_ev or None)
            material, composition, density_value, excitation = _sbethe_args(
                selected, name, element_counts, density, mean_excitation
            )
            result = generate_material(
                material,
                composition,
                density_g_cm3=density_value,
                mean_excitation_eV=excitation,
                band_gap_eV=band_gap,
                projectile=projectile.lower(),
                overwrite=overwrite,
                keep_on_failure=keep_on_failure,
            )
    except (XsgenError, ValueError, FileExistsError) as exc:
        raise CLIError(str(exc)) from exc

    table = result.table
    payload = {
        "code": selected,
        "key": table.key,
        "path": str(table.path),
        "tier": table.tier,
        "generated": result.generated,
        "manifest_sha256": table.manifest.get("manifest_sha256"),
    }
    if json_output:
        emit_json("pyrite.tables.generate.v1", payload)
        return
    action = "generated" if result.generated else "reused"
    emit_result(f"{action}: {table.key}\npath: {table.path}")


@command.command("fetch")
@click.argument("code", type=click.Choice(["sbethe"], case_sensitive=False))
@output_option
def fetch_command(code: str, json_output: bool) -> None:
    """Fetch the pinned large reference database for CODE.

    SBETHE's source ships with PyRITE, but its 18 MB ``sdbase/`` directory is
    installed on demand into your user data directory. The complete upstream
    archive is SHA-256 verified; only ``sdbase/`` is extracted. A complete
    existing install returns successfully without network access.
    """
    from ...xsgen import DataFetchError
    from ...xsgen.fetch import fetch_sbethe

    try:
        result = fetch_sbethe()
    except DataFetchError as exc:
        raise CLIError(str(exc)) from exc

    payload = {
        "code": code.lower(),
        "path": str(result.path),
        "archive_sha256": result.archive_sha256,
        "file_count": result.file_count,
        "installed": result.installed,
    }
    if json_output:
        emit_json("pyrite.tables.fetch.v1", payload)
        return
    action = "installed" if result.installed else "already installed"
    emit_result(f"{action}: {result.path} ({result.file_count} files)")


@command.group("sources", cls=LazyGroup)
def sources_command() -> None:
    """Show and configure where the external code trees live."""


@sources_command.command("list")
@output_option
def sources_list_command(json_output: bool) -> None:
    """Report which external code trees PyRITE can currently reach.

    Reports every code even when one cannot be found, so a single missing tree
    does not hide the state of the others. A missing tree is a reported status,
    not a command failure.
    """
    rows = _source_rows()
    if json_output:
        emit_json("pyrite.tables.sources.v1", {"sources": [row.payload() for row in rows]})
        return
    emit_result("CODE\tSTATUS\tORIGIN\tROOT")
    for row in rows:
        emit_result(f"{row.code}\t{row.status}\t{row.origin or '-'}\t{row.root or '-'}")
        if row.missing_data_dirs:
            missing = ", ".join(row.missing_data_dirs)
            emit_result(f"\t\tmissing data directories: {missing}")


@sources_command.command("set")
@click.argument("code")
@click.argument("path", type=click.Path(file_okay=False))
def sources_set_command(code: str, path: str) -> None:
    """Persist PATH as the source tree for CODE.

    Stored resolved, so the value keeps its meaning from any working
    directory, and rejected up front if PATH does not hold that code.
    """
    from ...console import config as _config
    from ...xsgen.sources import SourceUnavailableError, code_names, validate_source_path

    if code not in code_names():
        known = ", ".join(code_names())
        raise click.BadParameter(
            f"unknown code {code!r}; choose one of: {known}", param_hint="CODE"
        )
    try:
        resolved = validate_source_path(code, path)
    except SourceUnavailableError as exc:
        raise click.BadParameter(str(exc), param_hint="PATH") from exc

    key = f"xsgen.{code}_source"
    try:
        _config.set_stored(key, str(resolved))
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc
    emit_result(f"{key} = {resolved}")
