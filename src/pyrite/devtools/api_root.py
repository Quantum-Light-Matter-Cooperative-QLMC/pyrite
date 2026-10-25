"""Static governance for root migration classifications and warning replacements."""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Set
from pathlib import Path
from typing import Any


def _literal(path: Path, name: str) -> object:
    """Read a reviewed assignment without executing package code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"missing literal {name} in {path}")


def check_root_policy(root: Path, names: set[str], core: frozenset[str]) -> list[str]:
    """Require a classification for every root binding and matching runtime policy."""
    try:
        payload = tomllib.loads((root / "api-surface.toml").read_text(encoding="utf-8"))
        policies = payload.get("root_policy", {})
        retained = _literal(root / "src/pyrite/__init__.py", "_ROOT_API")
        replacements = _literal(root / "src/pyrite/_api_deprecations.py", "ROOT_API_REPLACEMENTS")
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        return [f"root policy: cannot read reviewed declarations: {exc}"]
    if not isinstance(policies, dict):
        return ["root policy: expected [root_policy] table"]
    errors = []
    if set(policies) != names:
        errors.append(
            f"root policy coverage: missing={sorted(names - policies.keys())}, "
            f"extra={sorted(policies.keys() - names)}"
        )
    approved = core | ({"photon_energies"} & names)
    if not isinstance(retained, tuple) or not all(isinstance(name, str) for name in retained):
        errors.append("root policy: _ROOT_API must be a tuple of names")
    elif len(retained) != len(set(retained)) or set(retained) != approved:
        errors.append("root policy: _ROOT_API differs from approved foundational/convenience names")
    expected_replacements = {}
    for name, row in policies.items():
        if name not in names:
            continue
        row_errors = _check_row(name, row, approved, payload)
        errors.extend(row_errors)
        if not row_errors and row["classification"] == "deprecated":
            expected_replacements[f"pyrite.{name}"] = row["canonical"].replace(":", ".")
    if replacements != expected_replacements:
        errors.append("root policy: deprecation replacements differ from classified aliases")
    return errors


def _check_row(name: str, row: object, approved: Set[str], payload: dict[str, Any]) -> list[str]:
    """Validate classification independently of a type's scientific stability."""
    label = f"root policy {name}"
    if not isinstance(row, dict) or set(row) != {"classification", "canonical", "reason"}:
        return [f"{label}: expected classification, canonical and reason"]
    if not all(isinstance(value, str) for value in row.values()):
        return [f"{label}: classification, canonical and reason must be strings"]
    classification = row["classification"]
    canonical = row["canonical"]
    reason = row["reason"]
    if classification not in {
        "retained",
        "deprecated",
        "deferred",
    }:
        return [f"{label}: invalid classification"]
    if not reason.strip():
        return [f"{label}: missing rationale"]
    if canonical.count(":") != 1:
        return [f"{label}: canonical must be module:attribute"]
    module, attribute = canonical.split(":")
    if attribute != name or not all(part.isidentifier() for part in module.split(".")):
        return [f"{label}: invalid canonical import"]
    if classification == "retained":
        if name not in approved or canonical != f"pyrite:{name}":
            return [f"{label}: unapproved retained root export"]
    elif name in approved:
        return [f"{label}: foundational/convenience export must be retained"]
    elif classification == "deprecated":
        domain = payload["root_exports"][name]["domain"]
        if domain not in payload.get("canonical_facades", ()) or module != f"pyrite.{domain}":
            return [f"{label}: deprecation requires an established canonical facade"]
    else:
        target = payload["root_exports"][name]
        domain_path = f"pyrite.{target['domain']}:{name}"
        if canonical not in {target["target"], domain_path} or "#" not in reason:
            return [f"{label}: deferred export requires its current import and follow-up issue"]
    return []
