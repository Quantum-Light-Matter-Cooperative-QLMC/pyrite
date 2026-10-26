"""Generate the static package-dependency region in ``docs/repo_map.md``."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import grimp

BEGIN = "<!-- BEGIN GENERATED PACKAGE DEPENDENCIES -->"
END = "<!-- END GENERATED PACKAGE DEPENDENCIES -->"
PACKAGE = "pyrite"


def _top_level_packages(source_root: Path) -> set[str]:
    package_root = source_root / PACKAGE
    return {
        child.name
        for child in package_root.iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    }


def _owner(module: str, packages: set[str]) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == PACKAGE and parts[1] in packages:
        return parts[1]
    return None


def direct_module_dependencies(source_root: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Return exact cross-package static module imports (importer, imported).

    Package facade imports and TYPE_CHECKING imports are intentionally excluded,
    exactly as in the generated repository map. Lazy imports are not excluded:
    grimp sees ordinary import statements even when they are inside functions.
    """
    packages = _top_level_packages(source_root)
    graph = grimp.build_graph(
        PACKAGE,
        exclude_type_checking_imports=True,
        cache_dir=None,
    )
    edges: set[tuple[str, str]] = set()
    for importer in graph.modules:
        importer_owner = _owner(importer, packages)
        if importer_owner is None:
            continue
        if importer == f"{PACKAGE}.{importer_owner}":
            # Package facades intentionally re-export their public surface and
            # would otherwise turn ownership edges into artificial cycles.
            continue
        for imported in graph.find_modules_directly_imported_by(importer):
            imported_owner = _owner(imported, packages)
            if imported_owner is not None and imported_owner != importer_owner:
                edges.add((importer, imported))
    return packages, edges


def _package_edges(module_edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return {(importer.split(".")[1], imported.split(".")[1]) for importer, imported in module_edges}


def package_dependencies(source_root: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Return canonical top-level packages and their direct static dependencies."""
    packages, module_edges = direct_module_dependencies(source_root)
    return packages, _package_edges(module_edges)


def _report_from_edges(packages: set[str], module_edges: set[tuple[str, str]]) -> dict[str, Any]:
    """Produce a stable, machine-readable graph and SCC edge inventory."""
    package_edges = _package_edges(module_edges)
    components = _strongly_connected_components(packages, package_edges)
    component_for = {name: component for component in components for name in component}
    cycles = [component for component in components if len(component) > 1]
    internal_edges = sorted(
        (source, target)
        for source, target in package_edges
        if component_for[source] == component_for[target] and len(component_for[source]) > 1
    )
    return {
        "schema": "pyrite.package-dependencies.v1",
        "summary": {
            "package_count": len(packages),
            "package_edge_count": len(package_edges),
            "module_edge_count": len(module_edges),
            "component_count": len(components),
            "cyclic_component_count": len(cycles),
            "largest_component_size": max((len(component) for component in components), default=0),
        },
        "packages": sorted(packages),
        "package_edges": [list(edge) for edge in sorted(package_edges)],
        "module_edges": [
            {"importer": importer, "imported": imported}
            for importer, imported in sorted(module_edges)
        ],
        "components": [list(component) for component in components],
        "cyclic_components": [list(component) for component in cycles],
        "cycle_edges": [list(edge) for edge in internal_edges],
    }


def dependency_report(source_root: Path) -> dict[str, Any]:
    """Report exact import edges and SCCs for offline baseline comparisons."""
    packages, module_edges = direct_module_dependencies(source_root)
    return _report_from_edges(packages, module_edges)


def _strongly_connected_components(
    nodes: Iterable[str], edges: set[tuple[str, str]]
) -> list[tuple[str, ...]]:
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        adjacency[source].add(target)

    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(adjacency[node]):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        components.append(tuple(sorted(component)))

    for node in sorted(adjacency):
        if node not in indices:
            visit(node)
    return sorted(components)


def build_dependency_region(source_root: Path) -> str:
    packages, edges = package_dependencies(source_root)
    components = _strongly_connected_components(packages, edges)
    component_for = {member: component for component in components for member in component}
    component_id = {component: f"p{index}" for index, component in enumerate(components)}
    condensed_edges = {
        (component_for[source], component_for[target])
        for source, target in edges
        if component_for[source] != component_for[target]
    }

    lines = [
        BEGIN,
        "<!-- Generated by pyrite.devtools.repo_map; update with `pyrite-dev repo-map --write`. -->",
        "",
        "Arrows point from importer to imported package. Strongly connected packages are",
        "collapsed into one node so this static condensation graph is a DAG. Imports hidden",
        "behind lazy dispatch, ``__getattr__``, or ``import_module`` are outside this scan.",
        "",
        "```text",
        "Components (a + b denotes a static import cycle):",
    ]
    for component in components:
        label = " + ".join(component)
        lines.append(f"  {component_id[component]}: {label}")
    lines.append("Edges (importer -> imported):")
    for source, target in sorted(
        condensed_edges,
        key=lambda edge: (component_id[edge[0]], component_id[edge[1]]),
    ):
        lines.append(f"  {component_id[source]} -> {component_id[target]}")
    lines.extend(["```", END])
    return "\n".join(lines)


def update_repo_map(document: str, generated: str) -> str:
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ValueError(f"repository map must contain exactly one {BEGIN!r}/{END!r} region")
    prefix, remainder = document.split(BEGIN, 1)
    _, suffix = remainder.split(END, 1)
    return f"{prefix}{generated}{suffix}"


def write_or_check(*, root: Path, check: bool) -> bool:
    target = root / "docs" / "repo_map.md"
    expected = update_repo_map(
        target.read_text(encoding="utf-8"),
        build_dependency_region(root / "src"),
    )
    if check:
        return target.read_text(encoding="utf-8") == expected
    target.write_text(expected, encoding="utf-8")
    return True
