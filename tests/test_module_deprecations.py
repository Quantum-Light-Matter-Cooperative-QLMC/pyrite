"""Govern compatibility-module removal metadata against the live package tree."""

from __future__ import annotations

import ast
from pathlib import Path

from pyrite._module_deprecations import MODULE_DEPRECATIONS, _window


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
        assert entry.remove_in == _window(entry.deprecated_in) == "0.4.0"
