"""Offline checks for governed Python root and canonical domain namespaces.

This is deliberately a source-tree tool, not part of import-time ``pyrite``.
The optional runtime identity check is separated from the static CI gate so
checking the manifest never initializes optional numerical engines.
"""

from __future__ import annotations

import ast
import importlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pyrite.devtools.api_packages import (
    check_package_contracts,
    check_runtime_package_contracts,
    read_package_inventory,
)
from pyrite.devtools.api_root import check_root_policy
from pyrite.devtools.api_typing import check_typing_exports

ALLOWED_STATUSES = frozenset({"stable", "advanced-stable", "experimental", "internal"})
ALLOWED_ROOT_ROLES = frozenset({"core", "specialized", "experimental"})
ALLOWED_DOMAINS = frozenset(
    {
        "beams",
        "geometry",
        "materials",
        "detectors",
        "observations",
        "config",
        "quantum",
        "transport",
        "radiation",
        "campaign",
        "execution",
        "optimization",
        "results",
        "storage",
        "plots",
        "experimental",
        "simulation",
        "datasets",
    }
)
CORE_ROOT = frozenset(
    {"Beam", "Target", "Detector", "Scene", "Sweep", "Physics", "Numerics", "simulate", "Result"}
)
ROOT_METADATA = ("DATA_DIR", "__version__", "states")


@dataclass(frozen=True)
class Export:
    name: str
    module: str
    attribute: str
    status: str
    domain: str
    root_role: str


def _parse_export(name: str, record: object) -> tuple[Export | None, list[str]]:
    """Validate one TOML export row, independent of the inventory scan."""
    if not isinstance(record, dict):
        return None, [f"{name}: expected a table of export properties"]
    if set(record) != {"target", "status", "domain", "root_role"}:
        return None, [f"{name}: expected target, status, domain and root_role only"]

    target = record["target"]
    status, domain, role = (record[key] for key in ("status", "domain", "root_role"))
    if not isinstance(target, str) or target.count(":") != 1:
        return None, [f"{name}: target must be 'module:attribute'"]
    module, attribute = target.split(":", 1)
    if not module.startswith("pyrite.") or not attribute.isidentifier():
        return None, [f"{name}: invalid target {target!r}"]
    if not isinstance(status, str) or status not in ALLOWED_STATUSES:
        return None, [f"{name}: invalid API status {status!r}"]
    if not isinstance(domain, str) or domain not in ALLOWED_DOMAINS:
        return None, [f"{name}: invalid domain {domain!r}"]
    if not isinstance(role, str) or role not in ALLOWED_ROOT_ROLES:
        return None, [f"{name}: invalid root role {role!r}"]

    export = Export(name, module, attribute, status, domain, role)
    errors = _export_policy_errors(export)
    return export, errors


def _export_policy_errors(export: Export) -> list[str]:
    """Cross-field constraints are separate from TOML schema validation."""
    errors = []
    if export.status == "internal":
        errors.append(f"{export.name}: root exports cannot have internal status")
    if export.root_role == "core" and export.status != "stable":
        errors.append(f"{export.name}: a core root export must be stable")
    if export.root_role == "experimental" and export.status != "experimental":
        errors.append(f"{export.name}: experimental root role requires experimental status")
    return errors


def read_manifest(path: Path) -> tuple[dict[str, Export], list[str]]:
    """Load the reviewed root-export snapshot, reporting schema errors."""
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        return {}, [f"{path}: cannot read API inventory: {exc}"]

    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("api-surface.toml: unsupported schema_version (expected 1)")
    rows = payload.get("root_exports")
    if not isinstance(rows, dict) or not rows:
        return {}, [*errors, "api-surface.toml: [root_exports] must be a nonempty table"]

    exports: dict[str, Export] = {}
    seen_targets: dict[tuple[str, str], str] = {}
    for name, record in rows.items():
        export, row_errors = _parse_export(name, record)
        errors.extend(row_errors)
        if export is None:
            continue
        target = (export.module, export.attribute)
        if target in seen_targets:
            target_name = f"{export.module}:{export.attribute}"
            errors.append(
                f"{name}: duplicate target {target_name!r} already owned by {seen_targets[target]}"
            )
        seen_targets[target] = name
        exports[name] = export

    actual_core = {export.name for export in exports.values() if export.root_role == "core"}
    if actual_core != CORE_ROOT:
        errors.append(
            f"core root allowlist drift: missing={sorted(CORE_ROOT - actual_core)}, "
            f"extra={sorted(actual_core - CORE_ROOT)}"
        )
    return exports, errors


def _assignment(stmt: ast.stmt) -> tuple[str, ast.expr] | None:
    """Extract a direct module-level assignment without accepting computed bindings."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        return None
    return target.id, stmt.value


def _literal_exports(
    expression: ast.expr, *, label: str
) -> tuple[dict[str, tuple[str, str]] | None, list[str]]:
    """Read a manifest-governed literal mapping while preserving diagnostics."""
    try:
        value = ast.literal_eval(expression)
    except (ValueError, TypeError, SyntaxError, MemoryError):
        return None, [f"{label} must remain a literal mapping"]
    if not isinstance(value, dict) or any(
        not isinstance(name, str)
        or not isinstance(item, tuple)
        or len(item) != 2
        or not all(isinstance(part, str) for part in item)
        for name, item in value.items()
    ):
        return None, [f"{label} must map names to (module, attribute)"]
    return value, []


def _root_all_shape(expression: ast.expr) -> bool:
    """The root remains metadata literals followed by the lazy export keys."""
    if not isinstance(expression, ast.List):
        return False
    elements = expression.elts
    return (
        len(elements) == len(ROOT_METADATA) + 1
        and all(
            isinstance(element, ast.Constant) and element.value == name
            for element, name in zip(elements, ROOT_METADATA, strict=False)
        )
        and isinstance(elements[-1], ast.Starred)
        and isinstance(elements[-1].value, ast.Name)
        and elements[-1].value.id == "_PUBLIC_OBJECTS"
    )


def read_source_exports(source: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Extract lazy root mappings via AST without executing ``pyrite`` imports."""
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return {}, [f"{source}: cannot parse root source: {exc}"]
    mapping: dict[str, tuple[str, str]] | None = None
    all_shape_ok = False
    for stmt in tree.body:
        assignment = _assignment(stmt)
        if assignment is None:
            continue
        name, expression = assignment
        if name == "_PUBLIC_OBJECTS":
            mapping, errors = _literal_exports(expression, label="root _PUBLIC_OBJECTS")
            if errors:
                return {}, errors
        elif name == "__all__":
            all_shape_ok = _root_all_shape(expression)

    errors = []
    if mapping is None:
        errors.append("root _PUBLIC_OBJECTS mapping is missing")
    if not all_shape_ok:
        errors.append("root __all__ shape changed; review metadata and lazy exports")
    return mapping or {}, errors


def read_facade_domains(path: Path) -> tuple[tuple[str, ...], list[str]]:
    """Read explicitly approved canonical facades, if any (schema v1 optional key)."""
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        return (), [f"{path}: cannot read API inventory: {exc}"]
    domains = payload.get("canonical_facades", [])
    if (
        not isinstance(domains, list)
        or any(not isinstance(domain, str) or domain not in ALLOWED_DOMAINS for domain in domains)
        or len(domains) != len(set(domains))
    ):
        return (), ["api-surface.toml: canonical_facades must be distinct known domains"]
    return tuple(domains), []


def _facade_all_shape(expression: ast.expr) -> tuple[bool, set[str] | None]:
    """Accept a literal string list or a lazy list(_EXPORTS) declaration."""
    if isinstance(expression, ast.List):
        names: list[str] = []
        for element in expression.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return False, None
            names.append(element.value)
        if len(names) != len(set(names)):
            return False, None
        return True, set(names)
    valid = (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "list"
        and len(expression.args) == 1
        and isinstance(expression.args[0], ast.Name)
        and expression.args[0].id == "_EXPORTS"
        and not expression.keywords
    )
    return valid, None


def read_facade_exports(source: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Inspect a lazy domain facade without executing its imports."""
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return {}, [f"{source}: cannot parse facade: {exc}"]
    mapping: dict[str, tuple[str, str]] | None = None
    all_shape_ok = False
    listed_names: set[str] | None = None
    for stmt in tree.body:
        assignment = _assignment(stmt)
        if assignment is None:
            continue
        name, expression = assignment
        if name == "_EXPORTS":
            mapping, errors = _literal_exports(expression, label=f"{source}: _EXPORTS")
            if errors:
                return {}, errors
        elif name == "__all__":
            all_shape_ok, listed_names = _facade_all_shape(expression)

    errors = []
    if mapping is None:
        errors.append(f"{source}: missing _EXPORTS mapping")
    if not all_shape_ok:
        errors.append(f"{source}: __all__ must be list(_EXPORTS) or a unique string list")
    if mapping is not None and listed_names is not None:
        omitted = sorted(mapping.keys() - listed_names)
        if omitted:
            errors.append(f"{source}: __all__ omits canonical exports: {omitted}")
    return mapping or {}, errors


def check_facades(root: Path, exports: dict[str, Export], domains: tuple[str, ...]) -> list[str]:
    """Reject undocumented public facade members and diverging bindings."""
    packages, _, errors = read_package_inventory(root / "api-surface.toml")
    for domain in domains:
        path = root / "src" / "pyrite" / domain / "__init__.py"
        actual, source_errors = read_facade_exports(path)
        errors.extend(source_errors)
        expected = {
            name: (export.module, export.attribute)
            for name, export in exports.items()
            if export.domain == domain
        }
        root_owned = bool(expected)
        for name, row in packages.get(f"pyrite.{domain}", {}).items():
            # Existing mixed facades also expose eager/legacy package members.
            # Their __all__, typing and runtime bindings are checked separately.
            if root_owned and name not in actual:
                continue
            binding = (row.module, row.attribute)
            if name in expected and expected[name] != binding:
                errors.append(f"{domain}.{name}: root and facade inventory bindings disagree")
            expected[name] = binding
        if not expected:
            errors.append(f"{domain}: canonical facade has no owned exports in inventory")
        for name in sorted(expected.keys() - actual.keys()):
            errors.append(f"{domain}.{name}: missing canonical facade export")
        for name in sorted(actual.keys() - expected.keys()):
            errors.append(f"{domain}.{name}: unreviewed canonical facade export")
        for name in sorted(actual.keys() & expected.keys()):
            if actual[name] != expected[name]:
                errors.append(
                    f"{domain}.{name}: facade binding {actual[name]!r} "
                    f"differs from inventory {expected[name]!r}"
                )
    return errors


def check_runtime_facades(
    exports: dict[str, Export], domains: tuple[str, ...], *, package: str = "pyrite"
) -> list[str]:
    """Confirm canonical facade and legacy-root imports refer to identical objects."""
    errors: list[str] = []
    try:
        root = importlib.import_module(package)
    except (ImportError, RuntimeError) as exc:
        return [f"cannot import {package}: {exc}"]
    for domain in domains:
        try:
            facade = importlib.import_module(f"{package}.{domain}")
        except (ImportError, RuntimeError) as exc:
            errors.append(f"{domain}: cannot import facade: {exc}")
            continue
        for export in exports.values():
            if export.domain != domain:
                continue
            try:
                alias = getattr(facade, export.name)
                previous = getattr(root, export.name)
            except (ImportError, AttributeError, RuntimeError) as exc:
                errors.append(f"{domain}.{export.name}: cannot resolve facade: {exc}")
                continue
            if alias is not previous:
                errors.append(f"{domain}.{export.name}: facade is not identical to root alias")
    return errors


def check_static(root: Path) -> tuple[dict[str, Export], list[str]]:
    """Check manifest ownership against actual lazy exports and source modules."""
    manifest = root / "api-surface.toml"
    exports, errors = read_manifest(manifest)
    actual, source_errors = read_source_exports(root / "src" / "pyrite" / "__init__.py")
    errors.extend(source_errors)
    if errors:
        return exports, errors
    expected_names = set(exports)
    actual_names = set(actual)
    for name in sorted(expected_names - actual_names):
        errors.append(f"{name}: listed in API inventory but missing from root exports")
    for name in sorted(actual_names - expected_names):
        errors.append(f"{name}: unreviewed root export; add an approved inventory row")
    for name in sorted(actual_names & expected_names):
        export = exports[name]
        expected = (export.module, export.attribute)
        if actual[name] != expected:
            errors.append(f"{name}: root binding {actual[name]!r} differs from {expected!r}")
        relative = Path(*export.module.split("."))
        module_path = root / "src" / relative
        if (
            not module_path.with_suffix(".py").is_file()
            and not (module_path / "__init__.py").is_file()
        ):
            errors.append(f"{name}: canonical target module {export.module!r} does not exist")
    facade_domains, facade_errors = read_facade_domains(manifest)
    errors.extend(facade_errors)
    if not facade_errors:
        errors.extend(check_facades(root, exports, facade_domains))
        errors.extend(check_typing_exports(root, exports, facade_domains))
    errors.extend(check_package_contracts(root, exports))
    errors.extend(check_root_policy(root, set(exports), CORE_ROOT))
    return exports, errors


def check_runtime_aliases(exports: dict[str, Export], *, package: str = "pyrite") -> list[str]:
    """Check actual alias identity, explicitly opt-in because imports may be heavy."""
    errors: list[str] = []
    try:
        root = importlib.import_module(package)
    except (ImportError, RuntimeError) as exc:
        return [f"cannot import {package}: {exc}"]
    for export in exports.values():
        try:
            module = importlib.import_module(export.module)
            canonical = getattr(module, export.attribute)
            alias = getattr(root, export.name)
        except (ImportError, AttributeError, RuntimeError) as exc:
            errors.append(f"{export.name}: cannot resolve public alias: {exc}")
            continue
        if alias is not canonical:
            errors.append(
                f"{export.name}: root alias is not identical to {export.module}:{export.attribute}"
            )
    return errors


def check(root: Path, *, imports: bool = False) -> list[str]:
    """Return actionable violations; do not mutate source, imports or inventory."""
    exports, errors = check_static(root)
    if not errors and imports:
        errors.extend(check_runtime_aliases(exports))
        facade_domains, _ = read_facade_domains(root / "api-surface.toml")
        errors.extend(check_runtime_facades(exports, facade_domains))
        errors.extend(check_runtime_package_contracts(root / "api-surface.toml"))
    return errors
