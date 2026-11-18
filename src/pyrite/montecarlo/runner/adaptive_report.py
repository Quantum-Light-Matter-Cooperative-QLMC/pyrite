"""Explain an adaptive stop that missed its target or guards (#1267).

The stopping record already holds each observable's final statistics and the
pass/fail of every criterion. :func:`limit_summary` turns them into one
warning sentence per failing population that names only the failed criteria,
their measured values and effective thresholds, and says whether the RSE
target itself was met, so a variance-sampling (VOV) refusal is not read as a
missed RSE target. It reads the record only: no stopping rule, threshold or
sampling identity depends on it.
"""

from collections.abc import Mapping
from typing import Any

#: Display label of each stopped population.
_LABELS = {"cxr": "CXR", "brem": "brem"}


def _num(value: float) -> str:
    return f"{value:.3g}"


def _failed_criteria(status: Mapping[str, Any], guards: Mapping[str, Any]):
    """Short descriptions of the failed criteria other than RSE, in rule order."""
    passes = status["passes"]
    failed = []
    if not passes["max_electron_share"]:
        share = status["max_electron_share"]
        limit = _num(guards["max_electron_share"])
        failed.append(
            "max electron share undefined (zero total)"
            if share is None
            else f"max electron share {_num(share)} > {limit}"
        )
    if not passes["effective_sample_size"]:
        ess = status["effective_sample_size"]
        failed.append(f"ESS {_num(ess)} < {_num(guards['min_effective_electrons'])}")
    if not passes["variance_sampling"]:
        vov, n = status["variance_of_variance"], status["n_electrons"]
        limit = _num(guards["max_variance_of_variance"])
        failed.append(
            "VOV undefined (zero variance or too few electrons)"
            if vov is None
            else f"VOV gate (VOV + 1/n) {_num(vov + 1.0 / n)} > {limit}"
        )
    if not passes["stability"]:
        move, limit = status["stability"], status["stability_limit"]
        failed.append(
            "mean stability not evaluable (too few decision points or zero mean)"
            if move is None
            else f"mean stability move {_num(move)} > {_num(limit)}"
        )
    return failed


def _rse_text(status: Mapping[str, Any], target: float) -> str:
    rse = status["relative_se"]
    if rse is None:
        return f"RSE undefined (zero mean or too few electrons; target {_num(target)})"
    relation = "<=" if status["passes"]["relative_se"] else ">"
    verdict = "met" if status["passes"]["relative_se"] else "missed"
    return f"RSE {_num(rse)} {relation} target {_num(target)} ({verdict})"


def failure_kind(status: Mapping[str, Any]) -> str:
    """``"rse"``, ``"vov"``, ``"rse+vov"``, ``"guards"``/``"rse+guards"``, or ``"none"``."""
    passes = status["passes"]
    rse = not passes["relative_se"]
    others = [key for key, ok in passes.items() if key != "relative_se" and not ok]
    if not rse and not others:
        return "none"
    if others == ["variance_sampling"]:
        return "rse+vov" if rse else "vov"
    if not others:
        return "rse"
    return "rse+guards" if rse else "guards"


def _population(name: str, status: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    target = status.get("target_rse", record["target_rse"])
    count = status["n_electrons"]
    cap = status.get("max_electrons", record["max_electrons"])
    head = f"{_LABELS.get(name, name)} at {count}/{cap} electrons"
    if status["converged"]:
        return f"{head}: converged"
    parts = [_rse_text(status, target), *_failed_criteria(status, record["guards"])]
    if failure_kind(status) == "vov":
        parts.append("a looser RSE target would not lift this refusal")
    return f"{head}: " + "; ".join(parts)


def limit_summary(record: Mapping[str, Any]) -> str:
    """One clause per population, naming only the criteria that failed."""
    statistics = record["statistics"]
    return " | ".join(_population(name, statistics[name], record) for name in statistics)
