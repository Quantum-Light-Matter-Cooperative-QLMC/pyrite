"""Navigation ownership and legacy URL guards for issue #759."""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"


def _toc_targets(page: Path) -> set[Path]:
    targets: set[Path] = set()
    for block in re.findall(r"```\{toctree\}\n(.*?)```", page.read_text(), re.S):
        for line in block.splitlines():
            entry = line.strip()
            if not entry or entry.startswith(":"):
                continue
            targets.add((page.parent / f"{entry}.md").resolve())
    return targets


def test_computation_pages_have_one_navigation_owner():
    owners = {
        "computation/statistical-methods.md": "physics/numerical-methods/index.md",
        "computation/execution-and-acceleration.md": "reference/configuration/index.md",
        "computation/memory-and-scheduling.md": "reference/configuration/index.md",
        "computation/precision-and-tolerances.md": "reference/configuration/index.md",
        "computation/random-streams.md": "reference/configuration/index.md",
        "physics/radiation-physics/energy-grid-semantics.md": "physics/radiation-physics/index.md",
    }
    parents = {DOCS / name for name in set(owners.values())}
    for child, parent in owners.items():
        target = (DOCS / child).resolve()
        assert target.is_file(), child  # Preserve the published URLs.
        containing = {page for page in parents if target in _toc_targets(page)}
        assert containing == {DOCS / parent}, (child, containing)


def test_computation_index_is_a_link_only_compatibility_page():
    index = (DOCS / "index.md").read_text()
    legacy = (DOCS / "computation/index.md").read_text()
    physics = (DOCS / "physics/index.md").read_text()

    assert "Computation <computation/index>" not in index
    assert "orphan: true" in legacy
    assert "```{toctree}" not in legacy
    assert "numerical-methods/index" in physics
    assert (DOCS / "physics/numerical-methods/index.md").is_file()
