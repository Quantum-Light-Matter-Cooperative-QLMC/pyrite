"""Resolve BremsLib tables for the bremsstrahlung spectrum.

The physics core may not import :mod:`pyrite.xsgen`, so the spectrum takes
its BremsLib tables as an argument. This module is the driver-side half:
it resolves one stored table per element and stages it with
:func:`pyrite._bremslib_table.prepare_bremslib_table`.

Resolution order, per element:

1. An element the pinned release covers resolves to its released table. A
   released table that has not been fetched is an error naming
   ``pyrite tables fetch bremslib``, not a fallback: the fix is one command.
2. Any other element resolves to a locally generated table, generating it
   from a BremsLib checkout when none is stored yet.
3. With no release entry and no checkout, raise by default. Only explicit
   ``allow_fallback=True`` permits the isotropic EEDL substitute.
"""

import warnings
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Literal, cast

import numpy as np

from ..._bremslib_table import BremsLibBremsstrahlungTable, prepare_bremslib_table
from ..._physics_fallback import PhysicsFallbackWarning
from ...materials.atomic import Z_TABLE
from .._errors import SourceUnavailableError, TableNotFoundError
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
    allow_fallback: bool = False,
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
    allow_fallback
        Allow unresolvable elements to be omitted, with a
        :class:`pyrite._physics_fallback.PhysicsFallbackWarning`. By default
        every requested element requires a table.

    Raises
    ------
    ValueError
        For an unknown element symbol.
    TableNotFoundError
        If a released element's table has not been fetched.

    Validation: bremslib-angular-model
    """
    tables: dict[str, BremsLibBremsstrahlungTable] = {}
    for element in dict.fromkeys(elements):
        try:
            z = Z_TABLE[element]
        except KeyError as exc:
            raise ValueError(f"unknown element {element!r}") from exc
        try:
            stored = resolve_element_table(z, source_path=source_path)
        except (SourceUnavailableError, TableNotFoundError) as exc:
            message = (
                f"no BremsLib table for {element} (Z={z}): {exc}. Install it with "
                "`pyrite tables fetch bremslib`, or explicitly select "
                "bremsstrahlung_model='eedl'."
            )
            if not allow_fallback:
                raise TableNotFoundError(message) from exc
            warnings.warn(
                message + " Falling back to isotropic EEDL emission for this element.",
                PhysicsFallbackWarning,
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


def resolve_bremsstrahlung_model(
    model: str, elements: Iterable[str], *, allow_fallback: bool = False
) -> Literal["eedl", "bremslib"]:
    """Resolve a run's continuum source *before* computing any trajectories.

    ``auto`` requires every BremsLib table by default. Opting into fallback
    selects EEDL for the *whole* run, not just the missing elements. An
    explicit ``eedl`` request never probes BremsLib and needs no opt-in.
    """
    if model != "auto":
        return cast(Literal["eedl", "bremslib"], model)
    unique = tuple(dict.fromkeys(elements))
    try:
        load_bremsstrahlung_tables(unique)
    except TableNotFoundError as exc:
        if not allow_fallback:
            raise TableNotFoundError(
                f"BremsLib is required for bremsstrahlung_model='auto' "
                f"({', '.join(unique)}): {exc}"
            ) from exc
        warnings.warn(
            f"BremsLib tables for {', '.join(unique)} are not all installed: {exc}. "
            "Falling back to EEDL with isotropic photon angles for the entire run.",
            PhysicsFallbackWarning,
            stacklevel=2,
        )
        return "eedl"
    return "bremslib"


__all__ = [
    "load_bremsstrahlung_tables",
    "resolve_bremsstrahlung_model",
    "resolve_element_table",
    "table_identity",
]
