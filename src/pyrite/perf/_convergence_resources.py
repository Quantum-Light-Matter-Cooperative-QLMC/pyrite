"""Conservative #1041 confirmation forecasts from pilot summaries, never run admission.

Variance is per incident primary: Var(D) for saved-history paired scoring, or
Var(Y_left) + Var(Y_right) for independent equal-size arms. Summaries must be
independently audited against raw evidence before any job can be authorized.
"""

import math
from statistics import NormalDist


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {name}")
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"invalid {name}")
    return value


def _integer(value, name, *, zero=False):
    if type(value) is not int:
        raise ValueError(f"invalid integer {name}")
    return _positive(value, name, zero=zero)


def _validate_design(design):
    if design["schema"] != "pyrite.geant4-performance.convergence-design.v1":
        raise ValueError("unsupported convergence design")
    rows = design["expected_rows"]
    ids = [r["id"] for r in rows]
    m = _integer(design["multiplicity"], "multiplicity")
    if len(set(ids)) != len(ids) or len(ids) != m:
        raise ValueError("duplicate rows or changed multiplicity")
    if design["family_alpha"] != 0.05 or design["row_alpha"] != 0.05 / m:
        raise ValueError("changed family alpha or multiplicity allocation")
    for row in rows:
        _positive(row["allowance"], "allowance")
        if row["support"] not in ("available", "unsupported", "unresolved"):
            raise ValueError("unknown design support")
    eligibility = design["eligibility"]
    if (
        eligibility["variance_safety_factor"],
        eligibility["repeats"],
        eligibility["min_histories_per_repeat"],
    ) != (1.25, 3, 20000):
        raise ValueError("changed confirmation population policy")
    resources = design["resources"]
    for key in ("confirmation_cpu_core_hours", "confirmation_gpu_hours", "max_shard_minutes"):
        _positive(resources[key], key)
    _integer(resources["max_cpus"], "max_cpus")
    if resources["max_cpus"] > 8 or resources["max_shard_minutes"] > 30:
        raise ValueError("resource ceilings exceed design policy")
    return rows


def _validate_pilots(pilots, ids):
    seeds = pilots["seeds"]
    if (
        pilots["phase"] != "pilot"
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s not in (301, 302, 303) for s in seeds)
    ):
        raise ValueError("only declared planning seeds 301-303 are accepted")
    result = {}
    for pilot in pilots["rows"]:
        key = pilot["id"]
        if key not in ids or key in result:
            raise ValueError("unknown or duplicate pilot row")
        _positive(pilot["variance"], "pilot variance", zero=True)
        sampling = pilot["sampling"]
        if sampling not in ("paired", "independent"):
            raise ValueError("unknown sampling model")
        if len(pilot["arms"]) != (1 if sampling == "paired" else 2):
            raise ValueError(
                "paired forecasts need one replay population; independent need two arms"
            )
        result[key] = pilot
    return result


def _arm_forecast(arm, count, resources):
    histories = _integer(arm["histories"], "pilot histories")
    seconds = _positive(arm["elapsed_seconds"], "pilot elapsed seconds")
    overhead = _positive(arm["overhead_seconds"], "shard overhead seconds", zero=True)
    cpus = _integer(arm["cpus"], "cpus")
    gpus = _integer(arm["gpus"], "gpus", zero=True)
    memory = _integer(arm["peak_memory_bytes"], "peak memory")
    vram = _integer(arm["peak_vram_bytes"], "peak VRAM", zero=True)
    margin = _positive(arm["memory_margin_factor"], "memory margin")
    if cpus > resources["max_cpus"] or gpus > 1 or margin < 1:
        raise ValueError("invalid allocation or memory margin")
    if (gpus == 0) != (vram == 0):
        raise ValueError("GPU pilot requires positive VRAM; CPU pilot requires zero VRAM")
    if not all(
        isinstance(arm[k], str) and arm[k].strip()
        for k in ("hardware_identity", "runtime_identity")
    ):
        raise ValueError("explicit pilot hardware and runtime identities required")
    # Conservatively charge elapsed/N as the rate AND overhead for every future shard.
    # No undocumented subtraction of cold/setup costs or sharing across row forecasts.
    rate = seconds / histories
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("invalid per-primary rate")
    limit = resources["max_shard_minutes"] * 60
    capacity = math.floor(min(count, max(0, limit - overhead) / rate))
    if capacity < 1:
        return dict(status="over-budget", reason="one primary plus overhead exceeds shard ceiling")
    per_repeat = count // 3
    shards = (per_repeat + capacity - 1) // capacity
    largest = (per_repeat + shards - 1) // shards
    elapsed = count * rate + 3 * shards * overhead
    return dict(
        status="forecast",
        hardware_identity=arm["hardware_identity"],
        runtime_identity=arm["runtime_identity"],
        pilot_histories=histories,
        pilot_elapsed_seconds=seconds,
        peak_memory_bytes=memory,
        peak_vram_bytes=vram,
        memory_margin_factor=margin,
        seconds_per_primary=rate,
        overhead_seconds=overhead,
        cpus=cpus,
        gpus=gpus,
        histories_per_repeat=per_repeat,
        shards_per_repeat=shards,
        total_shards=3 * shards,
        max_histories_per_shard=largest,
        max_shard_seconds=largest * rate + overhead,
        elapsed_seconds=elapsed,
        cpu_core_hours=elapsed * cpus / 3600,
        gpu_hours=elapsed * gpus / 3600,
        memory_request_bytes=math.ceil(memory * margin),
        vram_request_bytes=math.ceil(vram * margin),
    )


def _row_forecast(row, pilot, design):
    base = dict(row)
    if row["support"] != "available":
        return base | dict(status=row["support"], reason=row.get("reason"))
    if pilot is None:
        return base | dict(status="missing", reason="no pilot summary")
    variance = pilot["variance"]
    if variance == 0:
        return base | dict(status="unresolved", reason="zero variance is not a population forecast")
    z = NormalDist().inv_cdf(1 - design["row_alpha"] / 2)
    try:
        required = 1.25 * 4 * z**2 * variance / row["allowance"] ** 2
    except (OverflowError, ZeroDivisionError):
        required = math.inf
    if not math.isfinite(required) or required > 2**63 - 1:
        return base | dict(status="over-budget", reason="population forecast exceeds finite range")
    count = 3 * max(20000, (math.ceil(required) + 2) // 3)
    arms = [_arm_forecast(a, count, design["resources"]) for a in pilot["arms"]]
    status = "forecast" if all(a["status"] == "forecast" for a in arms) else "over-budget"
    return base | dict(
        status=status,
        sampling=pilot["sampling"],
        variance_per_primary=variance,
        histories_per_arm=count,
        repeats=3,
        arms=arms,
    )


def resource_plan(design, pilots):
    """Retain every design row and charge every contrast separately without reuse.

    Summary provenance, pairing, stationarity, variance stability and seed maps
    are not certified here. A forecast is never a convergence or submission gate.
    The caller must retain file hashes for the design and pilot input summaries.
    """
    expected = _validate_design(design)
    lookup = _validate_pilots(pilots, {r["id"] for r in expected})
    rows = [_row_forecast(r, lookup.get(r["id"]), design) for r in expected]
    arms = [a for r in rows for a in r.get("arms", []) if a["status"] == "forecast"]
    cpu = math.fsum(a["cpu_core_hours"] for a in arms)
    gpu = math.fsum(a["gpu_hours"] for a in arms)
    complete = all(r["status"] == "forecast" for r in rows)
    limits = design["resources"]
    over = (
        cpu > limits["confirmation_cpu_core_hours"]
        or gpu > limits["confirmation_gpu_hours"]
        or any(r["status"] == "over-budget" for r in rows)
    )
    return dict(
        schema="pyrite.geant4-performance.convergence-resource-plan.v1",
        purpose="planning-only",
        headline=False,
        numerical_gate="unmeasured",
        submission_ready=False,
        multiplicity=design["multiplicity"],
        row_alpha=design["row_alpha"],
        pilot_seeds=pilots["seeds"],
        rows=rows,
        forecast_cpu_core_hours=cpu,
        forecast_gpu_hours=gpu,
        cost_complete=complete,
        budget_status="over-budget" if over else ("within-ceilings" if complete else "incomplete"),
        ceilings=limits,
        seed_policy=design.get("seed_policy"),
        blockers=[
            "Pilot summaries require raw-evidence, diagnostics and identity audits",
            "No per-shard seed/substream map or collision audit; no jobs authorized",
            "Each contrast charged separately; shared runs require a separately audited schedule",
            "Forecasts cover confirmation only, not pilot spend or timing qualification",
            "Missing/unsupported/unresolved rows retain multiplicity and have unforecast cost",
            "Hardware-specific stationarity and memory forecasts remain unqualified",
        ],
    )
