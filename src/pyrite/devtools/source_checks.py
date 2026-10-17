"""Static source-size gate shared by developer tooling and layout tests."""

from pathlib import Path

SOURCE_LINE_LIMIT = 1_250


def oversized_source_modules(package_root: Path) -> dict[Path, int]:
    """Return relative paths and counts for Python modules over the size budget."""
    oversized = {}
    for path in sorted(package_root.rglob("*.py")):
        count = len(path.read_text(encoding="utf-8").splitlines())
        if count > SOURCE_LINE_LIMIT:
            oversized[path.relative_to(package_root)] = count
    return oversized
