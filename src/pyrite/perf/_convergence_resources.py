"""Conservative #1041 confirmation forecasts from pilot summaries, never run admission.

Variance is per incident primary: Var(D) for saved-history paired scoring (one
replay population), Var(Y_right - Y_left) for common-random-number pairs whose
summary cites a primary pairing audit (both arms charged), or
Var(Y_left) + Var(Y_right) for independent equal-size arms. Summaries must be
independently audited against raw evidence before any job can be authorized.

Rows whose audited common-random-number arms agree per primary (v1.1
exact-agreement rule) have no variance to forecast from. For such a row the
mean difference obeys ``|mean D| <= tau + R * p_U`` deterministically outside
an event of probability ``alpha/2``: primaries with ``|D| <= tau`` contribute
at most ``tau``, each discordant primary (``|D| > tau``) at most the frozen
score-support bound ``R``, and with zero discordances in ``N`` the one-sided
exact binomial bound on the discordance rate is ``p_U = 1 - (alpha/2)**(1/N)``.
The row qualifies only with zero discordances and ``R * p_U <= b - tau``.
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


def zero_discordance_rate_bound(n, alpha):
    """Exact one-sided upper bound on a rate with zero events in ``n`` trials."""
    _integer(n, "histories")
    _positive(alpha, "alpha")
    return -math.expm1(math.log(alpha / 2) / n)


def exact_agreement_bound(n, discordant, score_bound, allowance, alpha, tolerance_fraction):
    """Verdict of the frozen exact-agreement rule for one paired contrast.

    ``passed`` needs zero discordant primaries and ``tau + R * p_U <= b``;
    otherwise the row is not decided by this rule (``not-applicable`` with
    discordances, ``unresolved`` when ``N`` is too small).
    """
    _integer(discordant, "discordant primaries", zero=True)
    _positive(score_bound, "score bound")
    _positive(allowance, "allowance")
    if not 0 < tolerance_fraction < 1:
        raise ValueError("invalid tolerance fraction")
    tau = tolerance_fraction * allowance
    if discordant:
        return dict(status="not-applicable", tolerance=tau, discordant=discordant)
    bound = tau + score_bound * zero_discordance_rate_bound(n, alpha)
    status = "passed" if bound <= allowance else "unresolved"
    return dict(status=status, tolerance=tau, discordant=0, bound=bound, histories=n)


def exact_agreement_histories(score_bound, allowance, alpha, tolerance_fraction):
    """Smallest N for which zero discordances satisfy the exact-agreement rule."""
    room = (1 - tolerance_fraction) * allowance / score_bound
    if room >= 1:
        return 1
    n = math.ceil(math.log(alpha / 2) / math.log1p(-room))
    while (
        exact_agreement_bound(n, 0, score_bound, allowance, alpha, tolerance_fraction)["status"]
        != "passed"
    ):
        n += 1  # guard against floating rounding at the boundary
    return n


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
    rule = design.get("exact_agreement")
    if rule is not None:
        if not 0 < rule["tolerance_fraction"] < 1 or rule["requires"] != "crn-paired":
            raise ValueError("invalid exact-agreement rule")
        for bound in rule["score_bounds"].values():
            if bound != "beam_energy_keV":
                _positive(bound, "score bound")
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
        if sampling not in ("paired", "crn-paired", "independent"):
            raise ValueError("unknown sampling model")
        if sampling == "crn-paired" and not pilots.get("pairing_audit_sha256"):
            raise ValueError("common-random-number pairs need a cited pairing audit")
        if "discordant" in pilot:
            if sampling != "crn-paired":
                raise ValueError("discordance counts need audited common-random-number pairs")
            _integer(pilot["discordant"], "discordant primaries", zero=True)
            _positive(pilot["discordance_threshold"], "discordance threshold")
        if len(pilot["arms"]) != (1 if sampling == "paired" else 2):
            raise ValueError(
                "paired forecasts need one replay population; other sampling needs two arms"
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


def score_bound(design, row):
    """Frozen per-primary |D| bound R for an exact-agreement row, else None."""
    rule = design.get("exact_agreement")
    if rule is None or row["workload"] not in rule["workloads"]:
        return None
    bound = rule["score_bounds"].get(row["observable"])
    if bound == "beam_energy_keV":
        scene = next(c["scene"] for c in design["configurations"] if c["id"] == row["config"])
        spec = design["scene_specification"]["scenes"]
        return float(next(s["energy_keV"] for s in spec if s["id"] == scene))
    return bound


def _exact_forecast(row, pilot, design):
    bound = score_bound(design, row)
    if bound is None or pilot["sampling"] != "crn-paired" or pilot.get("discordant") != 0:
        return None
    fraction = design["exact_agreement"]["tolerance_fraction"]
    if not math.isclose(pilot["discordance_threshold"], fraction * row["allowance"], rel_tol=1e-12):
        raise ValueError("pilot discordance threshold differs from the frozen tolerance")
    alpha = design["row_alpha"]
    required = exact_agreement_histories(bound, row["allowance"], alpha, fraction)
    count = 3 * max(20000, (required + 2) // 3)
    arms = [_arm_forecast(a, count, design["resources"]) for a in pilot["arms"]]
    status = "forecast" if all(a["status"] == "forecast" for a in arms) else "over-budget"
    return dict(row) | dict(
        status=status,
        sampling="exact-agreement",
        score_bound=bound,
        discordance_threshold=pilot["discordance_threshold"],
        required_histories=required,
        histories_per_arm=count,
        repeats=3,
        arms=arms,
    )


def _row_forecast(row, pilot, design):
    base = dict(row)
    if row["support"] != "available":
        return base | dict(status=row["support"], reason=row.get("reason"))
    if pilot is None:
        return base | dict(status="missing", reason="no pilot summary")
    exact = _exact_forecast(row, pilot, design)
    if exact is not None:
        return exact
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
    # v1.1 (owner, 2026-11-25): GPU jobs count against the GPU-hour ceiling only.
    cpu_only = design["resources"].get("cpu_ceiling_scope") == "cpu-only-jobs"
    cpu = math.fsum(a["cpu_core_hours"] for a in arms if not (cpu_only and a["gpus"]))
    gpu_job_cpu = math.fsum(a["cpu_core_hours"] for a in arms if a["gpus"])
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
        pairing_audit_sha256=pilots.get("pairing_audit_sha256"),
        rows=rows,
        forecast_cpu_core_hours=cpu,
        forecast_gpu_job_cpu_core_hours=gpu_job_cpu,
        cpu_ceiling_scope="cpu-only-jobs" if cpu_only else "all-jobs",
        forecast_gpu_hours=gpu,
        cost_complete=complete,
        budget_status="over-budget" if over else ("within-ceilings" if complete else "incomplete"),
        ceilings=limits,
        seed_policy=design.get("seed_policy"),
        exact_agreement=design.get("exact_agreement"),
        blockers=[
            "Exact-agreement rows qualify only with zero confirmation discordances; any "
            "discordance makes the rule not-applicable, never an outcome-dependent top-up",
            "Pilot summaries require raw-evidence, diagnostics and identity audits",
            "No per-shard seed/substream map or collision audit; no jobs authorized",
            "Each contrast charged separately; shared runs require a separately audited schedule",
            "Forecasts cover confirmation only, not pilot spend or timing qualification",
            "Missing/unsupported/unresolved rows retain multiplicity and have unforecast cost",
            "Hardware-specific stationarity and memory forecasts remain unqualified",
        ],
    )
