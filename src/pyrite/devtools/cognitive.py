"""Informational Complexipy reporting; never part of the Radon ratchet."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyrite.devtools.complexity import ComplexityToolError

if TYPE_CHECKING:
    from complexipy import CodeComplexity, FunctionComplexity, RefactorPlan

    from pyrite.devtools.complexity import Routine, Scan

COGNITIVE_THRESHOLDS = (15, 20, 30, 50, 100)


def _complexipy_api() -> Callable[..., CodeComplexity]:
    try:
        from complexipy import code_complexity
    except ImportError as exc:
        raise ComplexityToolError(
            "complexipy is not installed; restore the lint dependency group with `pixi install`"
        ) from exc
    return code_complexity


def _matching_routine(function: FunctionComplexity, candidates: list[Routine]) -> Routine | None:
    """Match an analyzer range to its outer def, not a nested closure.

    Complexipy includes decorators in line_start and uses :: for class names.
    Return no match when the outermost candidate is ambiguous.
    """
    leaf = function.name.rsplit("::", 1)[-1]
    matches = [
        routine
        for routine in candidates
        if function.line_start <= routine.lineno <= function.line_end
        and routine.name.split("#")[0].rsplit(".", 1)[-1] == leaf
    ]
    if not matches:
        return None
    first = min(matches, key=lambda routine: routine.lineno)
    if sum(routine.lineno == first.lineno for routine in matches) != 1:
        return None
    return first


def _plan_record(plan: RefactorPlan) -> dict[str, Any]:
    return {
        "rule_id": plan.rule_id,
        "title": plan.title,
        "line_start": plan.line_start,
        "line_end": plan.line_end,
        "estimated_reduction": plan.estimated_reduction,
        "reduction_is_measured": plan.reduction_is_measured,
        "description": plan.description,
        "doc_url": plan.doc_url,
    }


def build_cognitive_report(
    root: Path, current: Scan, *, top: int, suggest: bool = False
) -> dict[str, Any]:
    """Scan exactly the successfully parsed Radon files, ignoring inline suppressions."""
    analyze = _complexipy_api()
    by_path: dict[str, list[Routine]] = defaultdict(list)
    for routine in current.routines:
        by_path[routine.path].append(routine)
    rows: list[dict[str, Any]] = []
    failures = []
    matched = set()
    for path in sorted(current.module_lines):
        try:
            result = analyze(
                (root / path).read_text(encoding="utf-8"), no_ignore=True, with_plans=False
            )
        except (OSError, ValueError) as exc:
            failures.append(f"{path}: {exc}")
            continue
        for function in result.functions:
            routine = _matching_routine(function, by_path[path])
            if routine is not None:
                matched.add((routine.path, routine.lineno, routine.col))
            rows.append(
                {
                    "path": path,
                    "name": function.name,
                    "line_start": function.line_start,
                    "line_end": function.line_end,
                    "cognitive": function.complexity,
                    "radon_match": None if routine is None else routine.name,
                    "cc": None if routine is None else routine.cc,
                }
            )
    rows.sort(key=lambda row: (row["path"], row["line_start"], row["name"]))
    ranked = sorted(rows, key=lambda row: (-row["cognitive"], row["path"], row["line_start"]))
    hotspots = [row for row in ranked if row["cognitive"] > 15][:top]
    if suggest:
        failures.extend(_add_suggestions(root, hotspots, analyze))
    scores = [row["cognitive"] for row in rows]
    return {
        "version": version("complexipy"),
        "routines": len(rows),
        "mean": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "above": {str(n): sum(score > n for score in scores) for n in COGNITIVE_THRESHOLDS},
        "matched_radon": len(matched),
        "unmatched_radon": len(current.routines) - len(matched),
        "unmatched_cognitive": sum(row["radon_match"] is None for row in rows),
        "hotspots": hotspots,
        "per_routine": rows,
        "failures": failures,
    }


def _add_suggestions(
    root: Path, hotspots: list[dict[str, Any]], analyze: Callable[..., CodeComplexity]
) -> list[str]:
    """Compute plans only for files containing displayed hotspots; never apply them."""
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hotspots:
        by_path[row["path"]].append(row)
    failures = []
    for path, rows in by_path.items():
        try:
            result = analyze(
                (root / path).read_text(encoding="utf-8"), no_ignore=True, with_plans=True
            )
        except (OSError, ValueError) as exc:
            failures.append(f"{path} (suggestions): {exc}")
            continue
        functions = {(f.name, f.line_start, f.line_end): f for f in result.functions}
        for row in rows:
            function = functions.get((row["name"], row["line_start"], row["line_end"]))
            if function is None:
                failures.append(f"{path}:{row['line_start']}: suggestion identity changed")
                continue
            row["suggestions"] = [_plan_record(plan) for plan in function.refactor_plans]
    return failures


def render_cognitive(report: dict[str, Any]) -> str:
    lines = [
        f"Cognitive complexity (informational; complexipy {report['version']}): "
        f"{report['routines']} routines, mean {report['mean']}",
        "Above thresholds: " + " ".join(f">{n}: {v}" for n, v in report["above"].items()),
        f"Matched Radon locations: {report['matched_radon']}; "
        f"unmatched Radon: {report['unmatched_radon']}; "
        f"unmatched Complexipy: {report['unmatched_cognitive']}",
        "Nested functions may contribute to the enclosing cognitive score; "
        "matched locations do not imply equivalent scopes.",
        "Top cognitive hotspots (>15):",
    ]
    for row in report["hotspots"]:
        cc = "unmatched" if row["cc"] is None else f"CC {row['cc']}"
        lines.append(
            f"  {row['path']}:{row['line_start']} {row['name']} cognitive {row['cognitive']} / {cc}"
        )
        for plan in row.get("suggestions", []):
            lines.append(f"    {plan['rule_id']} line {plan['line_start']}: {plan['title']}")
    if not report["hotspots"]:
        lines.append("  none")
    lines.extend(f"  cognitive scan failure: {failure}" for failure in report["failures"])
    return "\n".join(lines)
