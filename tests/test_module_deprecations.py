"""Govern compatibility-module removal metadata against the live package tree."""

import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from pyrite._module_deprecations import MODULE_DEPRECATIONS, PUBLIC_EXPORT_DEPRECATIONS, _window


def _live_module_shims() -> set[str]:
    package = Path(__file__).parents[1] / "src" / "pyrite"
    live: set[str] = set()
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text())
        if (ast.get_docstring(tree) or "").startswith("Compatibility re-export for"):
            relative = path.relative_to(package).with_suffix("")
            parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
            live.add(".".join(("pyrite", *parts)))
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


@pytest.mark.parametrize(
    "module",
    [
        "pyrite._line_grid_policy",
        "pyrite._line_windows",
        "pyrite.cli.commands._profile_line_grid",
        "pyrite.montecarlo.runner.line_grid",
        "pyrite.montecarlo.spectrum.line_jit_kernel",
        "pyrite.montecarlo.spectrum.line_seeds",
        "pyrite.montecarlo.spectrum.lines",
    ],
)
def test_removed_cxr_module_paths_are_unavailable(module):
    assert importlib.util.find_spec(module) is None


def test_profile_edit_module_is_removed():
    assert importlib.util.find_spec("pyrite.campaign.profile_edit") is None


def test_profile_commands_import_without_the_deprecated_facade():
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error:pyrite.campaign.profile_edit:DeprecationWarning",
            "-c",
            "import sys; from pyrite.cli.commands import profile, calculation; "
            "assert 'pyrite.campaign.profile_edit' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
