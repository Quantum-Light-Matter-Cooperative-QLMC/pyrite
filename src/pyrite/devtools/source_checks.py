"""Static source-size gate shared by developer tooling and layout tests."""

import ast
from pathlib import Path

SOURCE_LINE_LIMIT = 1_250
NOTEBOOK_LINE_LIMIT = 4_000


def source_line_limit(relative_path: Path, source: str) -> int:
    """Allow marimo entry points room for cells without exempting app helpers."""
    if relative_path.parts[0] != "apps":
        return SOURCE_LINE_LIMIT
    for statement in ast.parse(source).body:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            continue
        factory = statement.value.func
        if (
            isinstance(factory, ast.Attribute)
            and isinstance(factory.value, ast.Name)
            and factory.value.id == "marimo"
            and factory.attr == "App"
        ):
            return NOTEBOOK_LINE_LIMIT
    return SOURCE_LINE_LIMIT


def oversized_source_modules(package_root: Path) -> dict[Path, int]:
    """Return relative paths and counts for Python modules over the size budget."""
    oversized = {}
    for path in sorted(package_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(package_root)
        count = len(source.splitlines())
        if count > source_line_limit(relative_path, source):
            oversized[relative_path] = count
    return oversized
