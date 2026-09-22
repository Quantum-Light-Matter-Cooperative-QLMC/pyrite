"""Resolve BremsLib tables for the bremsstrahlung spectrum.

The physics core may not import :mod:`pyrite.xsgen`, so the spectrum takes
its BremsLib tables as an argument. This module is the driver-side half:
it resolves one stored table per element and stages it with
:func:`pyrite.montecarlo.spectrum.brem_bremslib.prepare_bremslib_table`.

Resolution order, per element:

1. An element the pinned release covers resolves to its released table. A
   released table that has not been fetched is an error naming
   ``pyrite tables fetch bremslib``, not a fallback: the fix is one command.
2. Any other element resolves to a locally generated table, generating it
   from a BremsLib checkout when none is stored yet.
3. With no release entry and no checkout, the element is left out with a
   ``RuntimeWarning``; :func:`~pyrite.montecarlo.spectrum.mc_brem_spectrum`
   then treats its emission as isotropic EEDL, the documented fallback.
"""

import warnings
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path

import numpy as np

from ...materials.atomic import Z_TABLE
from ...montecarlo.spectrum.brem_bremslib import (
    BremsLibBremsstrahlungTable,
    prepare_bremslib_table,
)
from .._errors import SourceUnavailableError
from ..store import StoredTable
from .generate import generate_element
from .release import catalogue_table, load_release_index


@cache
def _prepared(key: str, digest: str, path: str, z: int) -> BremsLibBremsstrahlungTable:
    # Keyed on the manifest digest as well as the key, so a table replaced
    # under the same key (``--overwrite``) is staged afresh.
    with np.load(path) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    return prepare_bremslib_table(arrays, atomic_number=z, key=key, digest=digest)


def resolve_element_table(z: int, *, source_path: str | Path | None = None) -> StoredTable:
    """Return the stored BremsLib table serving element ``z``.

    Raises
    ------
    TableNotFoundError
        If ``z`` is released but its table is not installed or not the pinned one.
    SourceUnavailableError
        If ``z`` is not released and no BremsLib checkout resolves.
    """
    index = load_release_index()
    if index is not None and index.entry(z) is not None:
        return catalogue_table(z)
    return generate_element(int(z), source_path=source_path).table


def load_bremsstrahlung_tables(
    elements: Iterable[str],
    *,
    source_path: str | Path | None = None,
) -> dict[str, BremsLibBremsstrahlungTable]:
    """Return staged BremsLib tables for ``elements``, keyed by symbol.

    Pass the result as ``bremslib_tables`` to
    :func:`pyrite.montecarlo.spectrum.mc_brem_spectrum` with
    ``cross_section_model="bremslib"``.

    Parameters
    ----------
    elements
        Element symbols, for example a composition's first column.
    source_path
        BremsLib checkout for elements outside the release.

    Raises
    ------
    ValueError
        For an unknown element symbol.
    TableNotFoundError
        If a released element's table has not been fetched.
    """
    tables: dict[str, BremsLibBremsstrahlungTable] = {}
    for element in dict.fromkeys(elements):
        try:
            z = Z_TABLE[element]
        except KeyError as exc:
            raise ValueError(f"unknown element {element!r}") from exc
        try:
            stored = resolve_element_table(z, source_path=source_path)
        except SourceUnavailableError as exc:
            warnings.warn(
                f"no BremsLib table for {element} (Z={z}): {exc}; its bremsstrahlung "
                "falls back to isotropic EEDL emission",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        tables[element] = _prepared(stored.key, stored.digest, str(stored.path), z)
    return tables


def table_identity(tables: Mapping[str, BremsLibBremsstrahlungTable]) -> dict[str, str]:
    """Map each table's key to its manifest digest, for run identity.

    The same shape :func:`pyrite.xsgen.store.identity_markers` returns, so it
    feeds ``xsgen_tables`` in :mod:`pyrite.campaign.profiles` directly.
    """
    return {table.key: table.digest for table in tables.values()}


__all__ = ["load_bremsstrahlung_tables", "resolve_element_table", "table_identity"]
