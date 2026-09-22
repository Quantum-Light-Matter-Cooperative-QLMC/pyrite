"""Generate and cache material-scoped SBETHE tables."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .._errors import DataFetchError
from .._run import run_program
from ..sources import missing_data_dirs, resolve_source, source_digest
from ..store import (
    MaterialTarget,
    StoredTable,
    TableRequest,
    material_identity,
    resolve,
    store,
)
from ..toolchain import build, find_toolchain
from .deck import SbetheDeck
from .parse import parse_integrated, parse_oscillator, parse_stopping, table_arrays

#: Outputs this generator actually reads back. The program writes more, but a
#: name listed here that it did not write is an error, so the list stays to
#: what is parsed.
_READ_BACK = ("stp.dat", "asymptotic.dat", "OOS.dat")


@dataclass(frozen=True)
class GenerationResult:
    """A resolved table and whether this call ran the external program."""

    table: StoredTable
    generated: bool


def material_request(
    name: str,
    composition: Mapping[int, float],
    *,
    density_g_cm3: float,
    mean_excitation_eV: float,
    band_gap_eV: float | None = None,
    projectile: str = "electron",
    source_path: str | Path | None = None,
) -> TableRequest:
    """Return the exact store request for one SBETHE material table."""
    deck = SbetheDeck(
        name=name,
        composition=composition,
        density_g_cm3=density_g_cm3,
        mean_excitation_eV=mean_excitation_eV,
        band_gap_eV=band_gap_eV,
        projectile=projectile,
    )
    source = resolve_source("sbethe", source_path)
    identity = material_identity(
        composition=deck.composition,
        density_g_cm3=deck.density_g_cm3,
        mean_excitation_eV=deck.mean_excitation_eV,
        extra={} if deck.band_gap_eV is None else {"band_gap_eV": deck.band_gap_eV},
    )
    return TableRequest(
        code="sbethe",
        code_version=source_digest(source),
        target=MaterialTarget(key=deck.name, identity=identity),
        quantity="collision_stopping",
        model=deck.model_record(),
    )


def generate_material(
    name: str,
    composition: Mapping[int, float],
    *,
    density_g_cm3: float,
    mean_excitation_eV: float,
    band_gap_eV: float | None = None,
    projectile: str = "electron",
    source_path: str | Path | None = None,
    overwrite: bool = False,
    keep_on_failure: bool = False,
) -> GenerationResult:
    """Generate or resolve an SBETHE stopping-power and cross-section table.

    Parameters
    ----------
    name
        Material name, echoed into the SBETHE output headers.
    composition
        Atomic number to stoichiometric index.
    density_g_cm3
        Mass density.
    mean_excitation_eV
        Mean excitation energy, always supplied explicitly.
    band_gap_eV
        Gap energy for an insulator or semiconductor; ``None`` for a
        conductor.
    projectile
        Projectile name, from :data:`pyrite.xsgen.sbethe.deck.PROJECTILES`.
    source_path
        Override for the SBETHE code tree.
    overwrite
        Regenerate and replace an existing stored table.
    keep_on_failure
        Preserve the scratch directory when the program fails.

    Returns
    -------
    GenerationResult
        The stored table, and whether this call ran SBETHE.

    Raises
    ------
    DataFetchError
        If the fetched ``sdbase/`` reference data is not installed.
    """
    deck = SbetheDeck(
        name=name,
        composition=composition,
        density_g_cm3=density_g_cm3,
        mean_excitation_eV=mean_excitation_eV,
        band_gap_eV=band_gap_eV,
        projectile=projectile,
    )
    source = resolve_source("sbethe", source_path)
    request = material_request(
        name,
        composition,
        density_g_cm3=density_g_cm3,
        mean_excitation_eV=mean_excitation_eV,
        band_gap_eV=band_gap_eV,
        projectile=projectile,
        source_path=source_path,
    )
    existing = resolve(request.key)
    if existing is not None and not overwrite:
        return GenerationResult(existing, generated=False)

    missing = missing_data_dirs(source)
    if missing:
        raise DataFetchError(
            f"SBETHE reference data is not installed: {', '.join(missing)}; "
            "run `pyrite tables fetch sbethe`"
        )

    toolchain = find_toolchain()
    binary = build(source, "sbethe", toolchain=toolchain)
    run = run_program(
        binary,
        data_dirs=source.data_dirs,
        stdin_text=deck.render(),
        outputs=_READ_BACK,
        keep_on_failure=keep_on_failure,
    )
    arrays = table_arrays(
        parse_stopping(run.outputs["stp.dat"]),
        parse_integrated(run.outputs["asymptotic.dat"]),
        parse_oscillator(run.outputs["OOS.dat"]),
    )
    table = store(
        request,
        arrays,
        compiler=toolchain.version,
        source_origin=source.origin,
        upstream=source.spec.upstream,
        overwrite=overwrite,
    )
    return GenerationResult(table, generated=True)
