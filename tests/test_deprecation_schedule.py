"""Hold every removal target to the shipping ``__version__``.

The registries used to be checked only against their own arithmetic:
`cli/_deprecations.py` computes ``remove_in = _window(since)`` and
`tests/cli/test_deprecations.py` asserted ``remove_in == _window(since)`` --
the same function that produced the value. `tests/test_module_deprecations.py`
went further and hard-coded ``"0.4.0"``. No test imported ``__version__``, so
no schedule could fail as releases went by, and 142 of 178 rows sat at or past
their target while the package shipped 0.3.0 (issue #68).

This module is the missing comparison. It is deliberately the only place that
knows what "overdue" means, and it covers every shim family that carries a
removal target, so a new family cannot be added without either appearing here
or being conspicuously absent.
"""

from __future__ import annotations

import pytest

from pyrite import __version__
from pyrite._module_deprecations import MODULE_DEPRECATIONS
from pyrite.campaign.model import FROM_LEGACY_REMOVE_IN
from pyrite.cli._deprecations import DEPRECATED_FLAGS, DEPRECATIONS


def _minor(version: str) -> tuple[int, int]:
    """The ``(major, minor)`` pair a removal target is compared on.

    Patch level is dropped on purpose: a target of ``0.5.0`` means "gone by the
    time 0.5 ships", so 0.5.0rc1 and 0.5.3 are both at or past it.
    """
    major, minor, *_ = version.split(".")
    return int(major), int(minor.split("rc")[0].split("a")[0].split("b")[0])


SHIPPING = _minor(__version__)


def _overdue(rows: dict[str, str]) -> dict[str, str]:
    """Rows whose removal minor the shipping version has reached or passed."""
    return {label: target for label, target in rows.items() if _minor(target) <= SHIPPING}


def test_no_deprecated_command_spelling_is_past_its_removal_target() -> None:
    overdue = _overdue({path: entry.remove_in for path, entry in DEPRECATIONS.items()})

    assert not overdue, (
        f"pyrite-xray {__version__} still ships {len(overdue)} command spelling(s) "
        f"at or past their removal target: {sorted(overdue)}. Remove the alias and "
        f"its row together, then regenerate docs/repo-design/cli/cli-deprecations.md."
    )


def test_no_deprecated_option_spelling_is_past_its_removal_target() -> None:
    overdue = _overdue(
        {
            f"{command} {flag}": entry.remove_in
            for (command, flag), entry in DEPRECATED_FLAGS.items()
        }
    )

    assert not overdue, (
        f"pyrite-xray {__version__} still ships {len(overdue)} option spelling(s) "
        f"at or past their removal target: {sorted(overdue)}."
    )


def test_no_compatibility_module_path_is_past_its_removal_target() -> None:
    overdue = _overdue({name: entry.remove_in for name, entry in MODULE_DEPRECATIONS.items()})

    assert not overdue, (
        f"pyrite-xray {__version__} still ships {len(overdue)} compatibility import "
        f"path(s) at or past their removal target: {sorted(overdue)}."
    )


def test_the_public_d7_bridge_is_not_past_its_removal_target() -> None:
    """`Sweep.from_legacy()` is scheduled by a constant, not by a registry."""
    assert not _overdue({"Sweep.from_legacy": FROM_LEGACY_REMOVE_IN})


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param([entry.remove_in for entry in DEPRECATIONS.values()], id="commands"),
        pytest.param([entry.remove_in for entry in DEPRECATED_FLAGS.values()], id="options"),
        pytest.param([entry.remove_in for entry in MODULE_DEPRECATIONS.values()], id="modules"),
        pytest.param([FROM_LEGACY_REMOVE_IN], id="d7-bridge"),
    ],
)
def test_every_removal_target_is_a_parseable_version(targets: list[str]) -> None:
    for target in targets:
        assert _minor(target) >= (0, 0)


def test_the_comparison_actually_fails_on_an_overdue_row() -> None:
    """Guard the guard: the 0.1.0 cohort's own target must read as overdue.

    Without this, an accidental inversion in `_overdue` would make every test
    above pass vacuously -- which is exactly the failure mode issue #68 found.
    """
    assert _overdue({"retired-in-0.3.0": "0.3.0"}) == {"retired-in-0.3.0": "0.3.0"}
    assert _overdue({"older": "0.1.0"}) == {"older": "0.1.0"}
    assert _overdue({"future": "0.5.0"}) == {}
