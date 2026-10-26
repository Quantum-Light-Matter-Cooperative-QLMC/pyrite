"""Govern selected documented subpackage APIs without promoting legacy ``__all__``.

The manifest deliberately separates *supported* package names from historical
compatibility imports. All static checks use AST/TOML and import no application
modules; runtime identity checks are opt-in through ``pyrite-dev api check --imports``.
"""

from __future__ import annotations

import ast
import importlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pyrite.devtools.api_typing import read_typing_names

if TYPE_CHECKING:
    from pyrite.devtools.api_surface import Export

_STATUSES = frozenset({"stable", "advanced-stable", "experimental"})


@dataclass(frozen=True)
class PackageExport:
    package: str
    name: str
    module: str
    attribute: str
    status: str


def _module_name(name: str, *, public: bool = True) -> bool:
    """Only fully qualified dotted PyRITE paths can enter the manifest."""
    return (
        name.startswith("pyrite.")
        and all(part.isidentifier() for part in name.split("."))
        and (not public or all(not part.startswith("_") for part in name.split(".")[1:]))
    )


def _read_all(path: Path) -> tuple[set[str], list[str]]:
    """Read a package's literal ``__all__`` without executing it."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return set(), [f"{path}: cannot inspect __all__: {exc}"]
    definitions = [
        stmt.value
        for stmt in tree.body
        if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets)
    ]
    if len(definitions) != 1:
        return set(), [f"{path}: expected exactly one literal __all__ assignment"]
    try:
        names = ast.literal_eval(definitions[0])
    except (ValueError, TypeError, SyntaxError, MemoryError):
        return set(), [f"{path}: __all__ must be a literal string list"]
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) for name in names)
        or len(names) != len(set(names))
    ):
        return set(), [f"{path}: __all__ must contain unique strings"]
    return set(names), []


def _parse_package_export(
    package: str, name: str, row: object
) -> tuple[PackageExport | None, list[str]]:
    """Validate one supported package binding independently of inventory loading."""
    label = f"{package}.{name}"
    if (
        not name.isidentifier()
        or name.startswith("_")
        or not isinstance(row, dict)
        or set(row) != {"target", "status"}
    ):
        return None, [f"{label}: expected public name, target and status"]
    target, status = row["target"], row["status"]
    if not isinstance(target, str) or target.count(":") != 1:
        return None, [f"{label}: target must be module:attribute"]
    module, attribute = target.split(":", 1)
    if not _module_name(module, public=False) or not attribute.isidentifier():
        return None, [f"{label}: invalid target {target!r}"]
    if not isinstance(status, str) or status not in _STATUSES:
        return None, [f"{label}: invalid supported status {status!r}"]
    return PackageExport(package, name, module, attribute, status), []


def read_package_inventory(
    manifest: Path,
) -> tuple[dict[str, dict[str, PackageExport]], dict[str, set[str]], list[str]]:
    """Read opt-in supported package contracts and legacy-private exceptions."""
    try:
        payload = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        return {}, {}, [f"{manifest}: cannot read package API inventory: {exc}"]

    errors: list[str] = []
    supported: dict[str, dict[str, PackageExport]] = {}
    records = payload.get("package_exports", {})
    if not isinstance(records, dict):
        errors.append("api-surface.toml: package_exports must be a table")
    else:
        for package, symbols in records.items():
            if not _module_name(package) or not isinstance(symbols, dict) or not symbols:
                errors.append(f"{package}: invalid or empty package_exports table")
                continue
            supported[package] = {}
            for name, row in symbols.items():
                entry, row_errors = _parse_package_export(package, name, row)
                errors.extend(row_errors)
                if entry is not None:
                    if (
                        package.startswith("pyrite.experimental.")
                        and entry.status != "experimental"
                    ):
                        errors.append(
                            f"{package}.{name}: experimental facade requires experimental status"
                        )
                    supported[package][name] = entry

    legacy: dict[str, set[str]] = {}
    exceptions = payload.get("legacy_private_all", {})
    if not isinstance(exceptions, dict):
        errors.append("api-surface.toml: legacy_private_all must be a table")
    else:
        for package, names in exceptions.items():
            if (
                not _module_name(package)
                or not isinstance(names, list)
                or any(not isinstance(n, str) or not n.startswith("_") for n in names)
                or len(names) != len(set(names))
            ):
                errors.append(f"{package}: legacy_private_all must list unique private names")
                continue
            legacy[package] = set(names)
    return supported, legacy, errors


def _namespace_source(root: Path, module: str) -> Path:
    """Find a declared namespace, whether a leaf module or package."""
    path = root.joinpath("src", *module.split("."))
    return path.with_suffix(".py") if path.with_suffix(".py").is_file() else path / "__init__.py"


def check_package_contracts(root: Path, root_exports: dict[str, Export]) -> list[str]:
    """Check explicitly supported package names and frozen private exceptions."""
    supported, legacy, errors = read_package_inventory(root / "api-surface.toml")
    experimental_dir = root / "src" / "pyrite" / "experimental"
    if experimental_dir.is_dir():
        for facade in experimental_dir.glob("*.py"):
            if facade.name == "__init__.py" or facade.stem.startswith("_"):
                continue
            package = f"pyrite.experimental.{facade.stem}"
            if package not in supported:
                errors.append(f"{package}: experimental facade missing API inventory")
    for package in sorted(set(supported) | set(legacy)):
        source = _namespace_source(root, package)
        declared, typing_errors = read_typing_names(source)
        errors.extend(typing_errors)
        names, all_errors = (
            _read_all(source) if "__all__" in declared or package in legacy else (declared, [])
        )
        errors.extend(all_errors)
        expected = supported.get(package, {})
        if package.startswith("pyrite.experimental.") and not all_errors:
            unreviewed = sorted(names - expected.keys())
            if unreviewed:
                errors.append(f"{package}: unreviewed experimental exports: {unreviewed}")
        for name in sorted(expected):
            entry = expected[name]
            if not all_errors and name not in names:
                errors.append(
                    f"{package}.{name}: supported export absent from __all__/declarations"
                )
            prior = root_exports.get(name)
            if prior is not None and (
                (prior.module, prior.attribute, prior.status)
                != (entry.module, entry.attribute, entry.status)
            ):
                errors.append(f"{package}.{name}: root and package API inventory disagree")
            module_path = root.joinpath("src", *entry.module.split("."))
            if not (
                module_path.with_suffix(".py").is_file() or (module_path / "__init__.py").is_file()
            ):
                errors.append(f"{package}.{name}: target module {entry.module} does not exist")
        if expected:
            if not typing_errors:
                for name in sorted(expected.keys() - declared):
                    errors.append(f"{package}.{name}: supported export lacks a typing declaration")
        if package in legacy and not all_errors:
            actual_private = {name for name in names if name.startswith("_")}
            missing = sorted(legacy[package] - actual_private)
            extra = sorted(actual_private - legacy[package])
            if missing or extra:
                errors.append(
                    f"{package}: legacy private __all__ drift: "
                    f"removed={missing}, unreviewed={extra}"
                )
    return errors


def check_runtime_package_contracts(manifest: Path) -> list[str]:
    """Confirm package symbols resolve to the reviewed implementation objects."""
    supported, _, errors = read_package_inventory(manifest)
    if errors:
        return errors
    for package, symbols in supported.items():
        try:
            namespace = importlib.import_module(package)
        except (ImportError, AttributeError, RuntimeError) as exc:
            errors.append(f"{package}: cannot import supported package: {exc}")
            continue
        for name, entry in symbols.items():
            try:
                expected = getattr(importlib.import_module(entry.module), entry.attribute)
                actual = getattr(namespace, name)
            except (ImportError, AttributeError, RuntimeError) as exc:
                errors.append(f"{package}.{name}: cannot resolve package export: {exc}")
                continue
            if expected is not actual:
                errors.append(f"{package}.{name}: package export is not identical to target")
    return errors
