"""Check that every pinned release table and fetched dataset is intact.

The release indexes shipped in the wheel (``bremslib-tables.json``,
``elsepa-tables.json``, and ``sbethe-tables.json``) record each table's key and manifest digest. This
module resolves each key through the store and compares the stored manifest
against the pin. It reads table manifests only, never payloads, so it is cheap
enough to run as a job preflight; table payload integrity is the remote sync's
content-addressed inventory.

The fetched datasets (:mod:`pyrite.datasets`: EEDL, EADL, EPDL) are single files
pinned by SHA-256; they are hashed in full, about 0.1 s together.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass

from .store import manifest_digest, resolve

#: Codes whose release indexes pin tables, in report order.
TABLE_CODES = ("bremslib", "elsepa", "sbethe-tables")
#: Fetched single-file datasets, in report order (see :mod:`pyrite.datasets`).
DATASET_CODES = ("eedl", "eadl", "epdl")
#: Everything ``pyrite tables verify`` can require.
CODES = TABLE_CODES + DATASET_CODES

#: Per-table outcomes. ``ok`` is the only passing one.
OK = "ok"
MISSING = "missing"
MISMATCH = "mismatch"
CORRUPT = "corrupt"


@dataclass(frozen=True)
class TableCheck:
    """Outcome of checking one pinned table."""

    code: str
    label: str
    key: str
    status: str
    tier: str | None = None


def pinned_tables(codes: Iterable[str] = TABLE_CODES) -> list[tuple[str, str, str, str]]:
    """Return ``(code, label, key, manifest_sha256)`` for every pinned table.

    Dataset codes in ``codes`` are accepted and contribute no rows.
    """
    wanted = tuple(codes)
    unknown = [code for code in wanted if code not in CODES]
    if unknown:
        raise ValueError(f"unknown code(s) {unknown}; choose from {list(CODES)}")
    rows: list[tuple[str, str, str, str]] = []
    if "bremslib" in wanted:
        from .bremslib.release import load_release_index as load_bremslib

        index = load_bremslib()
        for entry in () if index is None else index.tables:
            rows.append(("bremslib", entry.label, entry.key, entry.manifest_sha256))
    if "elsepa" in wanted:
        from .elsepa.release import load_release_index as load_elsepa

        elsepa = load_elsepa()
        for entry in () if elsepa is None else elsepa.tables:
            rows.append(("elsepa", entry.label, entry.key, entry.manifest_sha256))
    if "sbethe-tables" in wanted:
        from .sbethe.release import load_release_index as load_sbethe

        sbethe = load_sbethe()
        for entry in () if sbethe is None else sbethe.tables:
            rows.append(("sbethe-tables", entry.label, entry.key, entry.manifest_sha256))
    return rows


def verify_pinned(codes: Iterable[str] = CODES) -> list[TableCheck]:
    """Resolve each pinned table and compare its manifest digest to the pin.

    ``corrupt`` means the stored manifest does not hash to its own recorded
    digest (edited or truncated), or a dataset could not be read;
    ``mismatch`` means it is internally consistent but is not the pinned
    table, or a dataset's bytes differ from its pinned SHA-256. Dataset rows
    carry the file name as ``label`` and the pinned SHA-256 as ``key``.
    """
    wanted = tuple(codes)
    checks: list[TableCheck] = []
    for code, label, key, pinned in pinned_tables(wanted):
        try:
            found = resolve(key)
        except OSError, ValueError:
            checks.append(TableCheck(code, label, key, CORRUPT))
            continue
        if found is None:
            checks.append(TableCheck(code, label, key, MISSING))
            continue
        try:
            recorded = found.digest
            consistent = manifest_digest(found.manifest) == recorded
        except KeyError, TypeError, ValueError, json.JSONDecodeError:
            checks.append(TableCheck(code, label, key, CORRUPT, found.tier))
            continue
        if not consistent:
            status = CORRUPT
        elif recorded != pinned:
            status = MISMATCH
        else:
            status = OK
        checks.append(TableCheck(code, label, key, status, found.tier))
    from .. import datasets

    for code in (code for code in DATASET_CODES if code in wanted):
        dataset = datasets.get(code)
        try:
            outcome = datasets.verify_dataset(code)
        except OSError:
            outcome = CORRUPT
        status = {datasets.OK: OK, datasets.MISSING: MISSING}.get(outcome, outcome)
        checks.append(TableCheck(code, dataset.filename, dataset.sha256, status))
    return checks
