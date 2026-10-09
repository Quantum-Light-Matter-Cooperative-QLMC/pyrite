"""Guard positron tests on the pinned positron SBETHE/ELSEPA releases (#385).

A guard probing one material skipped a whole box that held only some positron
tables, and let the rest fail with a bare ``TableNotFoundError``. These guards
check every table the pinned release lists instead:

* none installed: skip, naming the fetch command (or fail under ``CI``, which
  fetches every pin, so a silent skip there would hide a provisioning gap);
* some installed: fail, naming the fetch command;
* all installed: pass.
"""

import os

import pytest

from pyrite.xsgen.store import resolve


def _pinned_keys(code: str) -> tuple[str, ...]:
    if code == "sbethe-tables":
        from pyrite.xsgen.sbethe.release import load_release_index
    else:
        from pyrite.xsgen.elsepa.release import load_release_index
    index = load_release_index(projectile="positron")
    return () if index is None else tuple(entry.key for entry in index.tables)


def _require(code: str) -> None:
    keys = _pinned_keys(code)
    label = {"sbethe-tables": "SBETHE", "elsepa": "ELSEPA"}[code]
    command = f"pyrite tables fetch {code} --projectile positron"
    if not keys:
        pytest.fail(f"this build pins no positron {label} table release")
    missing = [key for key in keys if resolve(key) is None]
    if not missing:
        return
    if len(missing) < len(keys):
        pytest.fail(
            f"{len(missing)} of {len(keys)} pinned positron {label} tables are missing "
            f"(e.g. {missing[0][:12]}…); run `{command}`"
        )
    message = f"positron {label} tables are not installed; run `{command}`"
    if os.environ.get("CI"):
        pytest.fail(message)
    pytest.skip(message)


def require_positron_sbethe_tables() -> None:
    """Skip or fail unless every pinned positron SBETHE table is installed."""
    _require("sbethe-tables")


def require_positron_elsepa_tables() -> None:
    """Skip or fail unless every pinned positron ELSEPA table is installed."""
    _require("elsepa")


def require_positron_tables() -> None:
    """Skip or fail unless both pinned positron releases are installed."""
    require_positron_sbethe_tables()
    require_positron_elsepa_tables()
