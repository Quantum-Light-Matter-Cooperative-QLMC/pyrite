"""The Python 3.15 ``apps`` requirements track ``marimo[recommended]`` (#1421).

``marimo-lens`` declares Python <3.15, so from 3.15 ``pyproject.toml`` copies
marimo's ``recommended`` set without it. A relock that picks a marimo with a
different set must update that copy.
"""

import tomllib
from importlib.metadata import requires
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]


def _by_name(requirements: list[Requirement]) -> dict[str, tuple[set[str], SpecifierSet]]:
    """Merge requirements per distribution; marimo's own floor is not compared."""
    merged: dict[str, tuple[set[str], SpecifierSet]] = {}
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        extras, specifier = merged.get(name, (set(), SpecifierSet()))
        if name != "marimo":
            specifier = specifier & requirement.specifier
        merged[name] = (extras | requirement.extras, specifier)
    return merged


def test_apps_extra_on_315_is_marimo_recommended_without_lens() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    ours = [
        requirement
        for requirement in map(Requirement, project["optional-dependencies"]["apps"])
        if requirement.marker
        and requirement.marker.evaluate({"python_version": "3.15"})
        and not requirement.marker.evaluate({"python_version": "3.14"})
    ]
    recommended = [
        requirement
        for requirement in map(Requirement, requires("marimo") or [])
        if requirement.marker
        and requirement.marker.evaluate({"extra": "recommended"})
        and not requirement.marker.evaluate({"extra": ""})
    ]

    expected = _by_name(recommended)
    del expected["marimo-lens"]

    assert _by_name(ours) == expected
