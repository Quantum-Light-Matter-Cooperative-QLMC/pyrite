"""Check that every pinned release table and fetched dataset is intact.

The release indexes shipped in the wheel (``bremslib-tables.json``,
``elsepa-tables.json``, and ``sbethe-tables.json``) record each table's key and manifest digest. This
module resolves each key through the store and compares the stored manifest
against the pin. It reads table manifests only, never payloads, so it is cheap
enough to run as a job preflight; table payload integrity is the remote sync's
content-addressed inventory.

The fetched datasets (:mod:`pyrite.datasets`: EEDL, EADL, EPDL) are single files
pinned by SHA-256; they are hashed in full, about 0.1 s together. SBETHE's
``sdbase/`` (``sbethe``) is checked for its required files, and its
``pdatconf.p14``, which every default-model run reads, against its pin.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from .store import manifest_digest, resolve

#: Codes whose release indexes pin tables, in report order.
TABLE_CODES = ("bremslib", "elsepa", "sbethe-tables")
#: Fetched single-file datasets, in report order (see :mod:`pyrite.datasets`).
DATASET_CODES = ("eedl", "eadl", "epdl")
#: Fetched reference-data directories, in report order.
REFERENCE_CODES = ("sbethe",)
#: Codes fixed by ``pyrite tables fetch CODE`` alone, without ``--archive``.
BARE_FETCH_CODES = DATASET_CODES + REFERENCE_CODES
#: Everything ``pyrite tables verify`` can require.
CODES = TABLE_CODES + DATASET_CODES + REFERENCE_CODES

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

    Dataset and reference-data codes in ``codes`` are accepted and
    contribute no rows.
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
    carry the file name as ``label`` and the pinned SHA-256 as ``key``. The
    ``sbethe`` row is ``mismatch`` when ``pdatconf.p14`` differs from its pin
    and ``corrupt`` when another required ``sdbase/`` file is absent.
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
    if "sbethe" in wanted:
        checks.append(_verify_sdbase())
    return checks


def _verify_sdbase() -> TableCheck:
    """Check the installed ``sdbase/`` the default shell model would read."""
    from ..datasets import PDATCONF_SHA256
    from .fetch import REQUIRED_SBETHE_FILES
    from .sources import installed_data_dir

    root = installed_data_dir("sbethe", "sdbase")
    label = "sdbase/pdatconf.p14"
    pdatconf = root / "pdatconf.p14"
    if not pdatconf.is_file():
        return TableCheck("sbethe", label, PDATCONF_SHA256, MISSING)
    try:
        digest = hashlib.sha256(pdatconf.read_bytes()).hexdigest()
    except OSError:
        return TableCheck("sbethe", label, PDATCONF_SHA256, CORRUPT)
    if digest != PDATCONF_SHA256:
        status = MISMATCH
    elif not all((root / name).is_file() for name in REQUIRED_SBETHE_FILES):
        status = CORRUPT
    else:
        status = OK
    return TableCheck("sbethe", label, PDATCONF_SHA256, status)
