"""Govern compatibility-module removal metadata against the live package tree."""

from __future__ import annotations

import ast
import importlib
import sys
import warnings
from pathlib import Path

import pytest

from pyrite._module_deprecations import MODULE_DEPRECATIONS, _window


def _announces_deprecation(tree: ast.Module) -> bool:
    """True when the module body calls ``warn_module_deprecation(__name__)``."""
    return any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "warn_module_deprecation"
        and [arg.id for arg in node.value.args if isinstance(arg, ast.Name)] == ["__name__"]
        for node in tree.body
    )


def _live_module_shims() -> dict[str, str]:
    package = Path(__file__).parents[1] / "src" / "pyrite"
    live: dict[str, str] = {}

    # The issue audit reported 22 root shims, but both its audited commit and
    # the live tree contain 24. Discover the surface instead of preserving that
    # stale count. It also reported 11 flat plots shims where the audited and
    # live trees contain 12, bringing the governed total to 36.
    for directory, prefix in ((package, "pyrite"), (package / "plots", "pyrite.plots")):
        for path in directory.glob("*.py"):
            tree = ast.parse(path.read_text())
            if not (ast.get_docstring(tree) or "").startswith("Compatibility re-export for"):
                continue

            targets = [
                node.module
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                and any(alias.name == "*" for alias in node.names)
            ]
            assert len(targets) == 1, f"expected one star-import target in {path}"

            module = f"{prefix}.{path.stem}"
            relative_target = targets[0]
            assert relative_target is not None
            parent_parts = module.split(".")[:-1]
            # Resolve the ImportFrom level from the star-import node itself.
            star_import = next(
                node
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                and any(alias.name == "*" for alias in node.names)
            )
            target_parts = parent_parts[: len(parent_parts) - (star_import.level - 1)]
            live[module] = ".".join((*target_parts, relative_target))

            assert _announces_deprecation(tree), (
                f"{module} re-exports silently; its body must call "
                f"warn_module_deprecation(__name__) so the removal window is "
                f"something importers were actually told about"
            )

    return live


def test_module_deprecation_registry_matches_live_tree_bidirectionally() -> None:
    live = _live_module_shims()

    assert len(live) == 36
    assert sum(module.count(".") == 1 for module in live) == 24
    assert sum(module.startswith("pyrite.plots.") for module in live) == 12
    assert live.keys() == MODULE_DEPRECATIONS.keys()
    for module, replacement in live.items():
        assert MODULE_DEPRECATIONS[module].replacement == replacement


def test_module_deprecation_support_window() -> None:
    for module, entry in MODULE_DEPRECATIONS.items():
        assert entry.module == module
        assert entry.remove_in == _window(entry.deprecated_in)
        # The paths were carved out in 0.2.0 but stayed silent until 0.3.0 wired
        # `warn_module_deprecation`, so 0.3.0 is the release the window counts
        # from -- a path that never warned has not spent its window (#68).
        assert entry.deprecated_in == "0.3.0"


def test_importing_a_compatibility_module_warns_once_naming_its_replacement() -> None:
    module = "pyrite.slim"
    entry = MODULE_DEPRECATIONS[module]
    sys.modules.pop(module, None)

    with pytest.warns(DeprecationWarning) as record:
        importlib.import_module(module)

    assert [str(warning.message) for warning in record] == [
        f"pyrite.slim is deprecated and will be removed in {entry.remove_in}; "
        f"import pyrite.checkpoints.slim instead"
    ]


def test_compatibility_module_warning_is_not_repeated_on_a_cached_import() -> None:
    """Module bodies run once, so a second import of the same path is silent."""
    module = "pyrite.slim"
    sys.modules.pop(module, None)
    importlib.import_module(module)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(module)

    assert [str(warning.message) for warning in caught] == []
