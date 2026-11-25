"""Bounded scalar telemetry for live adaptive stopping decisions."""

import math

_METRICS = (
    "n_electrons",
    "relative_se",
    "variance_of_variance",
    "effective_sample_size",
    "max_electron_share",
    "stability",
    "stability_limit",
    "vov_trend_slope",
    "coefficient_of_variation",
    "tail_slope",
    "shift_over_se",
    "figure_of_merit",
    "target_rse",
    "min_electrons",
    "max_electrons",
    "checks",
    "max_variance_of_variance",
    "max_history_share",
    "min_effective_histories",
    "projected_electrons",
    "next_point",
)
_GUARDS = (
    "relative_se",
    "variance_sampling",
    "effective_sample_size",
    "max_electron_share",
    "stability",
)
_PLAN = (
    "block_histories",
    "chunk_floor",
    "chunk_budget_mib",
    "rows_per_electron",
    "decision_growth",
    "stability_checks",
    "stability_fraction",
)


def _numbers(data, keys):
    if not isinstance(data, dict):
        return {}
    return {
        key: value
        for key in keys
        if isinstance((value := data.get(key)), (int, float))
        and not isinstance(value, bool)
        and abs(value) < 1e100
        and math.isfinite(value)
    }


def clean_adaptive_progress(data):
    """Whitelist fields and finite numbers (also used on untrusted remote JSON)."""
    if not isinstance(data, dict):
        return {}
    populations = data.get("populations")
    if not isinstance(populations, dict):
        return {}
    clean = {}
    for name in ("cxr", "brem"):
        raw = populations.get(name)
        if not isinstance(raw, dict):
            continue
        row = _numbers(raw, _METRICS)
        passes = raw.get("passes")
        if isinstance(passes, dict):
            row["passes"] = {key: passes[key] for key in _GUARDS if type(passes.get(key)) is bool}
        if raw.get("stop_reason") in ("converged", "max_electrons"):
            row["stop_reason"] = raw["stop_reason"]
        clean[name] = row
    if not clean:
        return {}
    return {"populations": clean, **_numbers(data, _PLAN)}


def adaptive_snapshot(monitor, plan, block_histories):
    """Read the actual monitor, not an estimate from transported history counts."""
    record = monitor.record()
    populations = {}
    for name, status in record["statistics"].items():
        # Independent brem uses its own target, projection and decision grid.
        scalar = getattr(monitor, name, monitor)
        settings = scalar.settings
        populations[name] = {
            **status,
            "target_rse": settings.target_rse,
            "min_electrons": settings.min_histories,
            "max_electrons": settings.max_histories,
            "max_variance_of_variance": settings.max_variance_of_variance,
            "max_history_share": settings.max_history_share,
            "min_effective_histories": settings.min_effective_histories,
            "checks": scalar.checks,
            "projected_electrons": scalar.projected,
            "next_point": scalar.next_point(),
            "stop_reason": scalar.stop_reason,
        }
    return clean_adaptive_progress(
        {
            "populations": populations,
            "block_histories": block_histories,
            "chunk_floor": plan.floor,
            "chunk_budget_mib": plan.budget_bytes / 2**20,
            "rows_per_electron": plan.rows_per_electron,
            "decision_growth": monitor.settings.decision_growth,
            "stability_checks": monitor.settings.stability_checks,
            "stability_fraction": monitor.settings.stability_fraction,
        }
    )
