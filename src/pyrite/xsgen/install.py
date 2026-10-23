"""Install the pinned reference data and tables, shared by the CLI and the Python API."""

from ._errors import DataFetchError

FETCH_CODES = (
    "eedl",
    "eadl",
    "epdl",
    "epdl-differential",
    "sbethe",
    "sbethe-tables",
    "elsepa",
    "bremslib",
    "bote-salvat",
)
DATASET_FETCH_CODES = ("eedl", "eadl", "epdl", "epdl-differential", "bote-salvat")
#: Opt-in codes a bare fetch skips; ``pyrite tables fetch CODE`` installs them.
OPTIONAL_FETCH_CODES = ("bote-salvat",)
_POSITRON_FETCH_CODES = ("sbethe-tables", "elsepa")
#: ``(code, projectile)`` pins a bare fetch installs, in order: each code, with
#: its positron pin right after the electron one (#385).
BARE_FETCH_STEPS = tuple(
    (code, projectile)
    for code in FETCH_CODES
    if code not in OPTIONAL_FETCH_CODES
    for projectile in (("electron", "positron") if code in _POSITRON_FETCH_CODES else ("electron",))
)


def step_name(code: str, projectile: str) -> str:
    """Return the display name of one fetch step."""
    return code if projectile == "electron" else f"{code} ({projectile})"


def fetch_one(code: str, archive: str | None, *, projectile: str = "electron") -> dict:
    """Install one code and return its machine-readable outcome."""
    if code in DATASET_FETCH_CODES:
        from ..datasets import fetch_dataset

        fetched = fetch_dataset(code, archive)
        return {
            "code": code,
            "path": str(fetched.path),
            "archive_sha256": fetched.sha256,
            "file_count": 1,
            "installed": fetched.installed,
        }
    from .fetch import fetch_bremslib, fetch_elsepa, fetch_sbethe, fetch_sbethe_tables

    if projectile != "electron":
        fetch_tables = {"elsepa": fetch_elsepa, "sbethe-tables": fetch_sbethe_tables}[code]
        result = fetch_tables(archive, projectile=projectile)
    else:
        fetch = {
            "bremslib": fetch_bremslib,
            "elsepa": fetch_elsepa,
            "sbethe": fetch_sbethe,
            "sbethe-tables": fetch_sbethe_tables,
        }[code]
        result = fetch(archive)
    return {
        "code": code,
        "path": str(result.path),
        "archive_sha256": result.archive_sha256,
        "file_count": result.file_count,
        "installed": result.installed,
    }


def fetch_tables(
    code: str | None = None, *, archive: str | None = None, projectile: str = "electron"
) -> list[dict]:
    """Download and install the pinned reference data and tables.

    The Python equivalent of ``pyrite tables fetch``: use it where no terminal
    is at hand (a notebook). Archives are SHA-256 verified; a complete install
    returns without network access.

    Parameters
    ----------
    code
        One of :data:`FETCH_CODES`, or ``None`` for everything (electron and
        positron pins).
    archive
        Local copy of ``code``'s pinned archive; needs ``code``.
    projectile
        ``"positron"`` installs that species' pin; needs ``code`` ``elsepa`` or
        ``sbethe-tables``.

    Returns
    -------
    list[dict]
        One outcome per step: ``code``, ``path``, ``archive_sha256``,
        ``file_count``, ``installed`` (``False`` when already present), plus
        ``projectile`` for positron steps.

    Raises
    ------
    ValueError
        For an unknown ``code`` or an invalid argument combination.
    DataFetchError
        If any step fails; the rest still run and the message names each failure.
    """
    projectile = projectile.lower()
    if projectile not in ("electron", "positron"):
        raise ValueError("projectile must be electron or positron")
    if code is not None:
        code = code.lower()
        if code not in FETCH_CODES:
            raise ValueError(f"unknown code {code!r}; choose one of: {', '.join(FETCH_CODES)}")
    if projectile == "positron" and code not in _POSITRON_FETCH_CODES:
        raise ValueError("projectile='positron' requires code 'elsepa' or 'sbethe-tables'")
    if code is None and archive is not None:
        raise ValueError("archive installs one code; name it, e.g. code='elsepa'")
    steps = BARE_FETCH_STEPS if code is None else ((code, projectile),)
    results, failed = [], []
    for selected, species in steps:
        try:
            payload = fetch_one(selected, archive, projectile=species)
        except DataFetchError as exc:
            failed.append(f"{step_name(selected, species)}: {exc}")
            continue
        if species != "electron":
            payload["projectile"] = species
        results.append(payload)
    if failed:
        raise DataFetchError(f"{len(failed)} of {len(steps)} failed:\n" + "\n".join(failed))
    return results


__all__ = ["BARE_FETCH_STEPS", "FETCH_CODES", "fetch_one", "fetch_tables", "step_name"]
