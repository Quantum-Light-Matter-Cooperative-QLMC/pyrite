"""Generate and cache free-atom and muffin-tin ELSEPA tables.

Validation: elsepa-vendor-reference
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .._run import run_program
from ..sources import resolve_source, source_digest
from ..store import (
    ElementTarget,
    MaterialTarget,
    StoredTable,
    TableRequest,
    material_identity,
    resolve,
    store,
)
from ..toolchain import build, find_toolchain
from .deck import DEFAULT_ABSORPTION_STRENGTH, ElsepaDeck
from .parse import parse_dcs, table_arrays


@dataclass(frozen=True)
class GenerationResult:
    """A resolved table and whether this call ran the external program."""

    table: StoredTable
    generated: bool


def element_request(
    z: int,
    energies_ev: Iterable[float],
    *,
    projectile: str = "electron",
    source_path: str | Path | None = None,
) -> tuple[TableRequest, ElsepaDeck]:
    """Return the store request and deck for one free-atom table."""
    deck = ElsepaDeck.free_atom(z, energies_ev, projectile=projectile)
    source = resolve_source("elsepa", source_path)
    request = TableRequest(
        code="elsepa",
        code_version=source_digest(source),
        target=ElementTarget(deck.z),
        quantity="elastic_dcs",
        model=deck.model_record(),
    )
    return request, deck


def muffin_tin_request(
    name: str,
    z: int,
    energies_ev: Iterable[float],
    *,
    radius_cm: float,
    projectile: str = "electron",
    density_g_cm3: float,
    absorption_strength: float = DEFAULT_ABSORPTION_STRENGTH,
    absorption_gap_eV: float | None = None,
    source_path: str | Path | None = None,
) -> tuple[TableRequest, ElsepaDeck]:
    """Return the store request and deck for one elementary-solid table.

    The target is the material, not the element: the muffin-tin radius and
    absorption gap are properties of the solid, so they enter the material
    identity and a changed lattice re-keys the table.
    """
    deck = ElsepaDeck.muffin_tin(
        z,
        energies_ev,
        radius_cm=radius_cm,
        projectile=projectile,
        absorption_strength=absorption_strength,
        absorption_gap_eV=absorption_gap_eV,
    )
    source = resolve_source("elsepa", source_path)
    extra: dict[str, float] = {"muffin_tin_radius_cm": float(radius_cm)}
    if absorption_gap_eV is not None:
        extra["absorption_gap_eV"] = float(absorption_gap_eV)
    identity = material_identity(
        composition={deck.z: 1.0}, density_g_cm3=density_g_cm3, extra=extra
    )
    request = TableRequest(
        code="elsepa",
        code_version=source_digest(source),
        target=MaterialTarget(key=str(name), identity=identity),
        quantity="elastic_dcs",
        model=deck.model_record(),
    )
    return request, deck


def _generate(
    request: TableRequest,
    deck: ElsepaDeck,
    *,
    source_path: str | Path | None,
    overwrite: bool,
    keep_on_failure: bool,
) -> GenerationResult:
    existing = resolve(request.key)
    if existing is not None and not overwrite:
        return GenerationResult(existing, generated=False)

    source = resolve_source("elsepa", source_path)
    toolchain = find_toolchain()
    binary = build(source, "elscata", toolchain=toolchain)
    run = run_program(
        binary,
        data_dirs=source.data_dirs,
        stdin_text=deck.render(),
        outputs=deck.output_names,
        keep_on_failure=keep_on_failure,
    )
    panels = [parse_dcs(run.outputs[name]) for name in deck.output_names]
    table = store(
        request,
        table_arrays(panels, energies_ev=deck.energies_ev),
        compiler=toolchain.version,
        source_origin=source.origin,
        upstream=source.spec.upstream,
        overwrite=overwrite,
    )
    return GenerationResult(table, generated=True)


def generate_element(
    z: int,
    energies_ev: Iterable[float],
    *,
    projectile: str = "electron",
    source_path: str | Path | None = None,
    overwrite: bool = False,
    keep_on_failure: bool = False,
) -> GenerationResult:
    """Generate or resolve a free-atom ELSEPA differential-cross-section table."""
    request, deck = element_request(z, energies_ev, projectile=projectile, source_path=source_path)
    return _generate(
        request,
        deck,
        source_path=source_path,
        overwrite=overwrite,
        keep_on_failure=keep_on_failure,
    )


def generate_muffin_tin(
    name: str,
    z: int,
    energies_ev: Iterable[float],
    *,
    radius_cm: float,
    projectile: str = "electron",
    density_g_cm3: float,
    absorption_gap_eV: float | None = None,
    source_path: str | Path | None = None,
    overwrite: bool = False,
    keep_on_failure: bool = False,
) -> GenerationResult:
    """Generate or resolve an elementary-solid (muffin-tin) ELSEPA table."""
    request, deck = muffin_tin_request(
        name,
        z,
        energies_ev,
        radius_cm=radius_cm,
        projectile=projectile,
        density_g_cm3=density_g_cm3,
        absorption_gap_eV=absorption_gap_eV,
        source_path=source_path,
    )
    return _generate(
        request,
        deck,
        source_path=source_path,
        overwrite=overwrite,
        keep_on_failure=keep_on_failure,
    )
