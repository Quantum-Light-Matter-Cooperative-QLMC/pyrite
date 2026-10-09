"""Govern compatibility-module removal metadata against the live package tree."""

import ast
from pathlib import Path

from pyrite._module_deprecations import MODULE_DEPRECATIONS, PUBLIC_EXPORT_DEPRECATIONS, _window


def _live_module_shims() -> set[str]:
    package = Path(__file__).parents[1] / "src" / "pyrite"
    live: set[str] = set()
    for directory, prefix in ((package, "pyrite"), (package / "plots", "pyrite.plots")):
        for path in directory.glob("*.py"):
            tree = ast.parse(path.read_text())
            if (ast.get_docstring(tree) or "").startswith("Compatibility re-export for"):
                live.add(f"{prefix}.{path.stem}")
    return live


def test_module_deprecation_registry_matches_live_tree_bidirectionally() -> None:
    """The flat re-export cohort was removed in 0.5.0; none may reappear unregistered."""
    live = _live_module_shims()

    assert live == set(MODULE_DEPRECATIONS)


def test_module_deprecation_support_window() -> None:
    for module, entry in MODULE_DEPRECATIONS.items():
        assert entry.module == module
        assert entry.remove_in == _window(entry.deprecated_in)


def test_public_export_deprecation_support_window() -> None:
    for (module, name), entry in PUBLIC_EXPORT_DEPRECATIONS.items():
        assert (entry.module, entry.name) == (module, name)
        assert entry.remove_in == _window(entry.deprecated_in)
