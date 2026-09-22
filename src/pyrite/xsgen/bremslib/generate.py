"""Generate and cache per-element BremsLib tables.

No toolchain and no subprocess: BremsLib is read, not run (D7), so this path
needs neither gfortran nor a scratch directory. What it shares with the other
two generators is everything after that -- one normalized request, one key,
one manifest, and a resolve that reuses an existing table instead of reading
the library again.
"""

from dataclasses import dataclass
from pathlib import Path

from ..sources import resolve_source
from ..store import ElementTarget, StoredTable, TableRequest, resolve, store
from .convert import QUANTITY, build_table
from .read import COMPLETE_T1_MAX_MEV, iter_node_files, library_root, library_version


@dataclass(frozen=True)
class GenerationResult:
    """A resolved table and whether this call read the library."""

    table: StoredTable
    generated: bool


def generate_element(
    z: int,
    *,
    t1_max_MeV: float = COMPLETE_T1_MAX_MEV,
    source_path: str | Path | None = None,
    overwrite: bool = False,
) -> GenerationResult:
    """Generate or resolve a BremsLib bremsstrahlung table for element ``z``.

    Parameters
    ----------
    z
        Atomic number, 1 to 100 -- the range the library covers.
    t1_max_MeV
        Highest incident energy to include, bounding the table's size.
        Defaults to the library's complete range.
    source_path
        Override for the BremsLib checkout. Neither the GPL-3 sources nor the
        810 MB library is shipped, so a checkout is required for any element
        whose table PyRITE does not already carry.
    overwrite
        Rebuild and replace an existing stored table.

    Returns
    -------
    GenerationResult
        The stored table, and whether this call read the library.

    Raises
    ------
    SourceUnavailableError
        If no BremsLib checkout resolves, or the one that does holds no
        library data for ``z``.
    ValueError
        If ``z`` is outside the library's range, or the library's structure
        does not hold.
    """
    if not 1 <= int(z) <= 100:
        raise ValueError(f"BremsLib covers Z = 1 to 100, got {z!r}")
    if not t1_max_MeV > 0.0:
        raise ValueError(f"t1_max_MeV must be positive, got {t1_max_MeV!r}")

    source = resolve_source("bremslib", source_path)
    library = library_root(source.root)
    nodes = [node for node in iter_node_files(library, z) if node.t1_MeV <= t1_max_MeV]
    request = TableRequest(
        code="bremslib",
        code_version=library_version(library, int(z), nodes),
        target=ElementTarget(int(z)),
        quantity=QUANTITY,
        model={"t1_max_MeV": float(t1_max_MeV)},
    )
    existing = resolve(request.key)
    if existing is not None and not overwrite:
        return GenerationResult(existing, generated=False)

    arrays = build_table(library, int(z), t1_max_MeV=t1_max_MeV)
    table = store(
        request,
        arrays,
        # No compiler: nothing was built. Recording one would claim a
        # provenance this table does not have.
        compiler=None,
        source_origin=source.origin,
        # The deposit DOI pins the version the shipped tables must be
        # refreshed against; the directory name records which one was read,
        # so a table built from an older deposit is visible in its manifest.
        upstream=f"{source.spec.upstream}; library data {library.name}",
        overwrite=overwrite,
    )
    return GenerationResult(table, generated=True)


__all__ = ["GenerationResult", "generate_element"]
