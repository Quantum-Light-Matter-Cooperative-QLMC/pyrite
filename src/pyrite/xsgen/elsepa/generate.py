"""Generate and cache free-atom ELSEPA tables."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .._run import run_program
from ..sources import resolve_source, source_digest
from ..store import ElementTarget, StoredTable, TableRequest, resolve, store
from ..toolchain import build, find_toolchain
from .deck import ElsepaDeck
from .parse import parse_dcs, table_arrays


@dataclass(frozen=True)
class GenerationResult:
    """A resolved table and whether this call ran the external program."""

    table: StoredTable
    generated: bool


def generate_element(
    z: int,
    energies_ev: Iterable[float],
    *,
    source_path: str | Path | None = None,
    overwrite: bool = False,
    keep_on_failure: bool = False,
) -> GenerationResult:
    """Generate or resolve a free-atom ELSEPA differential-cross-section table."""
    deck = ElsepaDeck.free_atom(z, energies_ev)
    source = resolve_source("elsepa", source_path)
    request = TableRequest(
        code="elsepa",
        code_version=source_digest(source),
        target=ElementTarget(deck.z),
        quantity="elastic_dcs",
        model=deck.model_record(),
    )
    existing = resolve(request.key)
    if existing is not None and not overwrite:
        return GenerationResult(existing, generated=False)

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
