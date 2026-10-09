"""Opt-in ``slow`` test tier (#381).

Tests marked ``slow`` (>5 s wall or a large peak RSS, or rarely needed) are
skipped unless ``PYRITE_SLOW_TESTS=1`` (``pyrite-dev test --slow``), a ``-m``
expression selects ``slow``, or the test is named by node id. CI sets the
variable, so every PR still runs them. Any ``-m`` expression that names
``slow`` (including ``not slow``) leaves selection to pytest. ``tests/conftest.py`` re-exports these
hooks.
"""

import os
import re
from pathlib import Path

import pytest

SLOW_SKIP_REASON = (
    "slow test: run with `pyrite-dev test --slow` or PYRITE_SLOW_TESTS=1, "
    "or select it by node id or -m slow"
)


def _explicit_node_ids(config: pytest.Config) -> list[str]:
    """Root-relative node ids named on the command line (``path::name...``)."""
    ids = []
    for arg in config.args:
        path, sep, rest = str(arg).partition("::")
        if not sep:
            continue
        try:
            relative = Path(path).resolve().relative_to(config.rootpath).as_posix()
        except ValueError:
            continue
        ids.append(f"{relative}::{rest}")
    return ids


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("PYRITE_SLOW_TESTS") == "1":
        return
    # A -m expression naming ``slow`` already chose; pytest deselects the rest.
    if re.search(r"\bslow\b", config.getoption("markexpr", "") or ""):
        return
    explicit = _explicit_node_ids(config)
    skip = pytest.mark.skip(reason=SLOW_SKIP_REASON)
    for item in items:
        if item.get_closest_marker("slow") is None:
            continue
        nodeid = item.nodeid
        if any(nodeid == node or nodeid.startswith((node + "::", node + "[")) for node in explicit):
            continue
        item.add_marker(skip)


def pytest_terminal_summary(terminalreporter) -> None:
    skipped = sum(
        1
        for report in terminalreporter.stats.get("skipped", [])
        if isinstance(report.longrepr, tuple) and SLOW_SKIP_REASON in str(report.longrepr[-1])
    )
    if skipped:
        terminalreporter.write_line(
            f"{skipped} slow tests skipped; run them with `pyrite-dev test --slow` "
            "or PYRITE_SLOW_TESTS=1"
        )
