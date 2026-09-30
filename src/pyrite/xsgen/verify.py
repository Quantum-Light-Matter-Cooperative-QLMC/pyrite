"""Check that every pinned release table is present and matches its pin.

The release indexes shipped in the wheel (``bremslib-tables.json`` and
``elsepa-tables.json``) record each table's key and manifest digest. This
module resolves each key through the store and compares the stored manifest
against the pin. It reads manifests only, never payloads, so it is cheap
enough to run as a job preflight; payload integrity is the remote sync's
content-addressed inventory.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass

from .store import manifest_digest, resolve

#: Codes whose release indexes pin tables, in report order.
CODES = ("bremslib", "elsepa")

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


def pinned_tables(codes: Iterable[str] = CODES) -> list[tuple[str, str, str, str]]:
    """Return ``(code, label, key, manifest_sha256)`` for every pinned table."""
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
    return rows


def verify_pinned(codes: Iterable[str] = CODES) -> list[TableCheck]:
    """Resolve each pinned table and compare its manifest digest to the pin.

    ``corrupt`` means the stored manifest does not hash to its own recorded
    digest (edited or truncated); ``mismatch`` means it is internally
    consistent but is not the pinned table.
    """
    checks: list[TableCheck] = []
    for code, label, key, pinned in pinned_tables(codes):
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
    return checks
