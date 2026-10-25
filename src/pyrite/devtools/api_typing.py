"""Source-only checks that lazy public exports remain visible to type checkers.

A ``__getattr__``-based facade can import correctly at runtime while becoming
invisible to static analysis. This module checks explicit Python declarations,
not import resolution: actual object identity is checked by ``api check --imports``.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrite.devtools.api_surface import Export


def _is_typing_guard(expr: ast.expr) -> bool:
    """Recognize standard ``if TYPE_CHECKING:`` and ``if typing.TYPE_CHECKING:``."""
    return (isinstance(expr, ast.Name) and expr.id == "TYPE_CHECKING") or (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "typing"
        and expr.attr == "TYPE_CHECKING"
    )


def _declared_names(statements: list[ast.stmt]) -> set[str]:
    """Names visible to editors, not names hidden behind runtime ``__getattr__``."""
    names: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, ast.ImportFrom):
            for item in stmt.names:
                if item.name != "*":
                    names.add(item.asname or item.name)
        elif isinstance(stmt, ast.Import):
            for item in stmt.names:
                names.add(item.asname or item.name.split(".")[0])
        elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            # A literal module-level binding is also visible to type checkers.
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(stmt, ast.If) and _is_typing_guard(stmt.test):
            names.update(_declared_names(stmt.body))
    return names


def read_typing_names(source: Path) -> tuple[set[str], list[str]]:
    """Inspect declarations without importing the package or optional backends."""
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return set(), [f"{source}: cannot inspect typing declarations: {exc}"]
    return _declared_names(tree.body), []


def check_typing_exports(
    root: Path, exports: Mapping[str, Export], domains: tuple[str, ...]
) -> list[str]:
    """Every governed lazy root/facade name needs a static declaration.

    Existing non-governed members of mixed facades are not silently promoted
    to public API by this check. The supported set comes from the inventory.
    """
    sources = [("pyrite", root / "src" / "pyrite" / "__init__.py", set(exports))]
    for domain in domains:
        owned = {export.name for export in exports.values() if export.domain == domain}
        sources.append(
            (f"pyrite.{domain}", root / "src" / "pyrite" / domain / "__init__.py", owned)
        )

    errors: list[str] = []
    for label, path, expected in sources:
        names, parse_errors = read_typing_names(path)
        errors.extend(parse_errors)
        if parse_errors:
            continue
        for name in sorted(expected - names):
            errors.append(
                f"{label}.{name}: governed public export lacks a static declaration "
                "(import or TYPE_CHECKING import)"
            )
    return errors
