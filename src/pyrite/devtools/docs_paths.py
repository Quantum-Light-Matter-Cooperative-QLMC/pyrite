"""Check literal ``docs/...`` references outside Sphinx's link graph."""

import re
from dataclasses import dataclass
from pathlib import Path

_DOC_PATH = re.compile(r"(?<![A-Za-z0-9_])docs/[A-Za-z0-9_./-]+\.(?:md|rst)")
_TEXT_SUFFIXES = frozenset({".md", ".py", ".rst", ".sh", ".toml", ".yaml", ".yml"})
_SKIP_PARTS = frozenset(
    {
        ".git",
        ".remember",
        ".serena",
        ".venv",
        ".worktrees",
        "_autosummary",
        "_build",
        "__pycache__",
        "agentdocs",
    }
)


@dataclass(frozen=True)
class StaleDocPath:
    """One nonexistent repository-relative documentation path."""

    source: Path
    line: int
    target: str


def find_stale_doc_paths(root: Path) -> list[StaleDocPath]:
    """Return nonexistent literal ``docs/*.md|rst`` paths in repository text."""
    findings: list[StaleDocPath] = []
    for source in sorted(root.rglob("*")):
        if (
            not source.is_file()
            or source.suffix not in _TEXT_SUFFIXES
            or any(part in _SKIP_PARTS for part in source.relative_to(root).parts)
        ):
            continue
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, 1):
            for match in _DOC_PATH.finditer(line):
                target = match.group()
                if not (root / target).is_file():
                    findings.append(StaleDocPath(source.relative_to(root), line_number, target))
    return findings


def check_doc_paths(root: Path) -> None:
    """Raise with a location-first report when literal documentation paths are stale."""
    findings = find_stale_doc_paths(root)
    if not findings:
        return
    details = "\n".join(
        f"  {finding.source}:{finding.line}: {finding.target}" for finding in findings
    )
    raise RuntimeError(f"stale documentation paths:\n{details}")
