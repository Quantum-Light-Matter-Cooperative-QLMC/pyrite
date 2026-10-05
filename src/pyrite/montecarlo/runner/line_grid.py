"""Automatic case-local line-grid resolution for one run (issue #101).

Split out of ``runner/__init__`` to keep that module inside the source-size
budget; it is runner-internal and has no other consumer.
"""

import sys
import warnings
from time import perf_counter

import numpy as np

from ..._backend import BACKEND, REAL, _to_cpu
from ..._grid_semantics import resolution_num, validate_backend_spacing
from ..._line_grid_policy import (
    BANDWIDTH_PROXY_SAFETY,
    LOCAL_RESOLUTION_POLICY,
    RESONANCE_BANDWIDTH_POLICY,
    LineGridToleranceError,
    LineGridTruncationWarning,
    LineShapePrecisionWarning,
    LineYieldStatisticsWarning,
    cached_coordinates,
    coordinate_cache_key,
    lineshape_precision_warning,
    resolved_coordinates,
    store_coordinates,
    windowed_coordinates,
)
from ..._line_windows import build_window_plan, window_plan_from_payload
from ..spectrum.diagnostics import coherent_fringe_spacing, sinc_feature_spacing
from ..spectrum.line_seeds import (
    SEEDING_REVISION,
    ResonancePopulation,
    SeedContext,
    case_line_stop_eV,
    collect_feature_seeds,
    local_spacing_seeds,
)


def _resident_mib():
    """Return current Linux RSS for opt-in line-grid profiling."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _device_mib():
    """Pool-used and driver-used device MiB for opt-in profiling, if a GPU."""
    stats = BACKEND.allocator_stats()
    out = {
        "device_used_mib": stats.get("used_mib"),
        "device_reserved_mib": stats.get("reserved_mib"),
    }
    runtime = getattr(getattr(getattr(BACKEND, "cp", None), "cuda", None), "runtime", None)
    if runtime is not None:
        free, total = runtime.memGetInfo()
        out["device_driver_used_mib"] = (total - free) / (1 << 20)
    return out


def _profile_stage(profile, **fields):
    """Record opt-in line-grid stage fields and flush them to stderr.

    Immediate output survives a scheduler time limit that discards the report.
    Device occupancy is sampled at every stage.
    """
    fields.update(_device_mib())
    profile.update(fields)
    items = " ".join(f"{name}={value}" for name, value in fields.items())
    print(f"line-grid profile: {items}", file=sys.stderr, flush=True)


# Case fields the automatic line-grid resolution depends on. Transport is a
# deterministic function of these plus the seed, so they content-address the
# resolved grid. Deliberately NOT the whole case: the photon grids and the
# emission-side switches do not move a single electron.
_RESOLUTION_INPUT_KEYS = (
    "crystal",
    "composition",
    "hkl_list",
    "E0_keV",
    "thickness_ang",
    "crystal_width_mm",
    "crystal_height_mm",
    "beam_fwhm_mm",
    "beam_fwhm_y_mm",
    "energy_spread_frac",
    "longitudinal_distribution",
    "transverse_distribution",
    "theta_obs_rad",
    "tilt_deg",
    "tilt_azim_deg",
    "groove_spacing_ang",
    "straggling",
    "energy_model",
    "max_dE_frac",
    "inelastic_model",
    "inelastic_cutoff_eV",
    "secondary_threshold_eV",
    "radiative_model",
    "radiative_cutoff_eV",
    "pair_production_model",
    "atomic_electron_deflection",
    "elastic_model",
    "abs_layers",
    "Ne",
    "Ne_brem",
    "seed",
    "E_cut_lines_keV",
    "E_cut_brem_keV",
)


# Additional case fields the kinematic window seeds read: the reflection rows
# depend on orientation and mosaic spread, which the sinc spacing does not.
_WINDOW_INPUT_KEYS = (
    "beam_uvw",
    "surface_hkl",
    "azimuth_rad",
    "recip_miscut_rad",
    "mosaic_mc_fwhm_rad",
    "mosaic_mc_nodes",
    "layer_radiators",
)
_WEIGHT_INPUT_KEYS = ("B_ang2",)


def _cached_grid(cached):
    """Coordinates of a cache record, or ``None`` when its plan is stale."""
    plan_payload = cached.get("window_plan")
    if plan_payload is None:
        return np.linspace(cached["start_eV"], cached["stop_eV"], int(cached["num"]))
    try:
        return window_plan_from_payload(plan_payload).coordinates()
    except KeyError, TypeError, ValueError:
        return None


def _windowed_line_grid(payload, case, segments, n_hats, Ne, feature_widths_eV):
    """Seed, plan and validate a piecewise line axis from this run's segments.

    The backbone is the policy's maximum spacing. Every provider the policy
    names contributes windows for every observation direction in ``n_hats``,
    and each direction's kinematic window spacing follows the sinc feature
    width measured on these same segments along that direction. The plan
    depends only on the union of seeds, so one direction reproduces the
    single-direction plan exactly.
    """
    windows = payload["windows"]
    start = float(payload["bandwidth"]["start_eV"])
    stop = float(payload["bandwidth"]["stop_eV"])
    seeds = []
    summaries = []
    for n_hat, feature_width_eV in zip(n_hats, feature_widths_eV, strict=True):
        context = SeedContext(
            case=case,
            segments=segments,
            n_hat=np.asarray(n_hat, dtype=float),
            electron_limit=Ne,
            start_eV=start,
            stop_eV=stop,
            feature_width_eV=float(feature_width_eV),
            samples_per_feature=int(windows["samples_per_feature"]),
            aliased_weight_limit=float(payload["resolution"]["aliased_weight_limit"]),
            tail_widths=float(windows["tail_widths"]),
        )
        direction_seeds, direction_summary = collect_feature_seeds(context, windows["providers"])
        seeds.extend(direction_seeds)
        summaries.append(direction_summary)
    plan = build_window_plan(start, stop, float(payload["resolution"]["max_spacing_eV"]), seeds)
    grid, record = windowed_coordinates(payload, plan, dtype=REAL)
    record["feature_width_eV"] = float(min(feature_widths_eV))
    record["window_seeds"] = summaries[0] if len(summaries) == 1 else summaries
    return grid, record


def _direction_populations(case, segments, n_hat, Ne, abs_layers, groove, start, ceiling, profile):
    """Every production line along ``n_hat`` over ``(start, ceiling)``."""
    from . import _lines_for_segments

    audit = {"start_eV": start, "stop_eV": ceiling, "collect": []}
    stage_started = perf_counter() if profile is not None else 0.0
    _lines_for_segments(
        segments,
        np.array([start, ceiling]),
        case,
        n_hat,
        abs_layers,
        groove,
        coherent=False,
        Ne=Ne,
        truncation_audit=audit,
    )
    chunks = audit["collect"]
    if profile is not None:
        _profile_stage(
            profile,
            population_collect_wall_s=perf_counter() - stage_started,
            population_chunks=len(chunks),
            population_lines=sum(chunk[0].size for chunk in chunks),
            rss_after_population_collect_mib=_resident_mib(),
        )
    stage_started = perf_counter() if profile is not None else 0.0
    populations = [
        ResonancePopulation("production lines", energy, weight, width)
        for energy, width, weight in chunks
    ]
    if profile is not None:
        _profile_stage(
            profile,
            population_pack_wall_s=perf_counter() - stage_started,
            population_data_bytes=sum(
                array.nbytes
                for population in populations
                for array in (population.energy_eV, population.weight, population.width_eV)
            ),
            rss_after_population_pack_mib=_resident_mib(),
        )
    return populations


def _measured_line_grid(payload, case, segments, n_hats, Ne, target_step, abs_layers, groove):
    """``resonance-population`` axis of this case (#192), under the ceiling cap.

    Uniform at the sinc-Nyquist step, or -- under ``resonance-local`` -- fine
    only where narrow lines resonate, on a backbone at the maximum spacing.
    Each direction in ``n_hats`` measures its own line population and stop
    edge; the grid ends at the largest edge and, under ``resonance-local``,
    refines the union of every direction's local-spacing seeds, so one
    direction reproduces the single-direction axis exactly.
    Returns ``(grid, record, bandwidth_record)``.

    Validation: line-grid-resonance-local-spacing
    """
    bandwidth = payload["bandwidth"]
    resolution = payload["resolution"]
    profile_enabled = bool(case.get("_profile_line_grid_stages", False))
    profile = {} if profile_enabled else None
    if profile_enabled:
        _profile_stage(profile, rss_before_resolution_mib=_resident_mib())
    start = float(bandwidth["start_eV"])
    ceiling = float(bandwidth["stop_eV"])
    direction_populations = []
    bandwidth_records = []
    for n_hat in n_hats:
        populations = _direction_populations(
            case, segments, n_hat, Ne, abs_layers, groove, start, ceiling, profile
        )
        stage_started = perf_counter() if profile_enabled else 0.0
        _direction_stop, direction_record = case_line_stop_eV(
            case,
            populations,
            start_eV=start,
            ceiling_eV=ceiling,
            truncation_limit=float(bandwidth["truncation_limit"]),
            proxy_safety=BANDWIDTH_PROXY_SAFETY,
        )
        if profile_enabled:
            _profile_stage(
                profile,
                stop_search_wall_s=perf_counter() - stage_started,
                rss_after_stop_search_mib=_resident_mib(),
            )
        direction_populations.append(populations)
        bandwidth_records.append(direction_record)
    stop = max(record["stop_eV"] for record in bandwidth_records)
    if len(bandwidth_records) == 1:
        bandwidth_record = bandwidth_records[0]
    else:
        bandwidth_record = {"stop_eV": stop, "directions": bandwidth_records}
    if resolution["policy"] != LOCAL_RESOLUTION_POLICY:
        stage_started = perf_counter() if profile_enabled else 0.0
        grid, record = resolved_coordinates(payload, target_step, dtype=REAL, stop_eV=stop)
        if profile_enabled:
            _profile_stage(
                profile,
                grid_build_wall_s=perf_counter() - stage_started,
                rss_after_grid_build_mib=_resident_mib(),
            )
            record["line_grid_profile"] = profile
        return grid, record, bandwidth_record
    backbone = float(resolution["max_spacing_eV"])
    stage_started = perf_counter() if profile_enabled else 0.0
    seeds = []
    summaries = []
    for populations in direction_populations:
        direction_seeds, direction_summary = local_spacing_seeds(
            populations,
            start_eV=start,
            stop_eV=stop,
            floor_spacing_eV=min(float(target_step), backbone),
            max_spacing_eV=backbone,
            halo_limit=float(resolution["halo_limit"]),
        )
        seeds.extend(direction_seeds)
        summaries.append(direction_summary)
    del direction_populations
    if profile_enabled:
        _profile_stage(
            profile,
            local_spacing_wall_s=perf_counter() - stage_started,
            local_spacing_seeds=len(seeds),
            rss_after_local_spacing_mib=_resident_mib(),
        )
    stage_started = perf_counter() if profile_enabled else 0.0
    plan = build_window_plan(start, stop, backbone, seeds)
    grid, record = windowed_coordinates(payload, plan, dtype=REAL, stop_eV=stop)
    if profile_enabled:
        _profile_stage(
            profile,
            grid_build_wall_s=perf_counter() - stage_started,
            rss_after_grid_build_mib=_resident_mib(),
        )
    record["feature_width_eV"] = float(target_step)
    record["local_spacing"] = summaries[0] if len(summaries) == 1 else summaries
    if profile_enabled:
        record["line_grid_profile"] = profile
    return grid, record, bandwidth_record


def _refuse_coherent_resolution(case, segments, n_hat, Ne):
    """Refuse automatic resolution on the coherent route (issue #117).

    The automatic policy derives its spacing from ``sinc_feature_spacing``,
    whose band limit is the per-segment retardation increment. The coherent
    route sums the complex field across segments first, so its fringes follow
    the *total* span of that quantity and are finer by the segment count --
    a difference of principle, not of calibration. Measured on identical
    trajectories, the coherent yield is still ~5e-2 from convergence at the
    spacing where the incoherent yield reaches 3e-8.

    Resolving those fringes is not affordable (order 1e7 points over a keV
    band), so this refuses rather than silently aliasing or silently
    refining. An explicit ``E_grid_line`` is unaffected.

    Validation: coherent-line-grid-fringe-spacing.
    """
    if not bool(case.get("coherent_emission", False)):
        return
    step, span, _ = coherent_fringe_spacing(segments, n_hat, electron_limit=Ne)
    grouped_step, grouped_span, _ = coherent_fringe_spacing(
        segments, n_hat, electron_limit=Ne, grouped=True
    )
    raise ValueError(
        "automatic line-grid resolution does not support coherent_emission: the "
        "sinc feature width bounds the incoherent route only, and the coherent "
        "route's fringe spacing is derived from the retardation span. This case "
        f"needs {step:.3e} eV (span {span:.4g} Ang) to resolve the inter-electron "
        f"term and {grouped_step:.3e} eV (span {grouped_span:.4g} Ang) for the "
        "decoherence-grouped floor. Supply an explicit E_grid_line, or run with "
        "coherent_emission disabled. See issue #117."
    )


def _resolve_policy_line_grid(payload, case, segments, n_hat, Ne, abs_layers, groove):
    """Resolve an automatic case-local line grid from this run's trajectories.

    Consults the content-addressed cache first, so a repeated case pays the
    sinc-feature scan once. The cache is a speed cache only: the case payload
    already carries the policy, and the policy plus the case determines the
    coordinates, so a cold and a warm cache resolve identically. A windowed
    policy also keys the cache on the orientation inputs its seeds read, and a
    hit rebuilds the coordinates from the stored plan; a plan this build would
    not make is treated as a miss.
    """
    return _resolve_policy_grid(
        payload, case, segments, (n_hat,), Ne, abs_layers, groove, observation=False
    )


def _resolve_policy_grid(payload, case, segments, n_hats, Ne, abs_layers, groove, *, observation):
    """Resolve one policy grid that serves every direction in ``n_hats``.

    The uniform spacing is the finest any direction needs, and windowed or
    measured seeds are the union over directions. ``observation``
    distinguishes a physical detector's direction set from the case's own
    scalar direction in the speed-cache key; the scalar grid and its cache
    entries are unchanged.
    """
    profile_enabled = bool(case.get("_profile_line_grid_stages", False))
    profile_started = perf_counter() if profile_enabled else 0.0
    profile_rss_before = _resident_mib() if profile_enabled else None
    for n_hat in n_hats:
        _refuse_coherent_resolution(case, segments, n_hat, Ne)
    windowed = payload.get("windows") is not None
    measured = payload["bandwidth"]["policy"] == RESONANCE_BANDWIDTH_POLICY
    keys = (
        _RESOLUTION_INPUT_KEYS
        + (_WINDOW_INPUT_KEYS if windowed or measured else ())
        + (_WEIGHT_INPUT_KEYS if measured else ())
    )
    inputs = {key: case[key] for key in keys if case.get(key, None) is not None}
    inputs["backend_dtype"] = np.dtype(REAL).name
    if windowed or measured:
        inputs["seeding_revision"] = SEEDING_REVISION
    if observation:
        inputs["observation_directions"] = np.asarray(n_hats, dtype=float).tolist()
    key = coordinate_cache_key(payload, inputs)
    cached = cached_coordinates(key)
    if cached is not None:
        grid = _cached_grid(cached)
        if grid is not None:
            _warn_lineshape_precision(cached)
            record = {**cached, "cache": "hit", "cache_key": key}
            if profile_enabled:
                record["line_grid_profile"] = {
                    "cache_hit": True,
                    "resolution_wall_s": perf_counter() - profile_started,
                    "rss_before_resolution_mib": profile_rss_before,
                    "rss_after_resolution_mib": _resident_mib(),
                }
            return grid, record
    limit = float(payload["resolution"]["aliased_weight_limit"])
    spacings = [
        sinc_feature_spacing(segments, n_hat, electron_limit=Ne, aliased_weight_limit=limit)
        for n_hat in n_hats
    ]
    steps = [spacing[0] for spacing in spacings]
    target_step = min(steps)
    _, aliased_fraction, spacing_segments = spacings[steps.index(target_step)]
    bandwidth_record = None
    try:
        if windowed:
            grid, record = _windowed_line_grid(payload, case, segments, n_hats, Ne, steps)
        elif measured:
            grid, record, bandwidth_record = _measured_line_grid(
                payload, case, segments, n_hats, Ne, target_step, abs_layers, groove
            )
        else:
            grid, record = resolved_coordinates(payload, target_step, dtype=REAL)
    except LineGridToleranceError as exc:
        raise LineGridToleranceError(
            f"{case['name']} at {case['E0_keV']:g} keV (backend {np.dtype(REAL).name}): {exc}"
        ) from exc
    if bandwidth_record is not None:
        record["measured_bandwidth"] = bandwidth_record
    if observation:
        record["observation_direction_count"] = len(n_hats)
    record.update(
        {
            "aliased_weight_fraction": aliased_fraction,
            "aliased_weight_limit": float(payload["resolution"]["aliased_weight_limit"]),
            "backend_dtype": np.dtype(REAL).name,
            "backend_safety_ulps": float(payload["resolution"]["backend_safety_ulps"]),
            "n_spacing_segments": spacing_segments,
            "observable_class": record["governing_observable"],
        }
    )
    line_grid_profile = record.pop("line_grid_profile", None)
    store_coordinates(key, record)
    _warn_lineshape_precision(record)
    result = {**record, "cache": "miss", "cache_key": key}
    if profile_enabled:
        result["line_grid_profile"] = {
            **(line_grid_profile or {}),
            "cache_hit": False,
            "resolution_wall_s": perf_counter() - profile_started,
            "rss_before_resolution_mib": profile_rss_before,
            "rss_after_resolution_mib": _resident_mib(),
        }
    return grid, result


def _warn_lineshape_precision(record):
    message = lineshape_precision_warning(record)
    if message is not None:
        warnings.warn(message, LineShapePrecisionWarning, stacklevel=3)


def resolve_observation_line_grid(case, segments, n_hats, Ne, E_grid, abs_layers=None, groove=None):
    """Choose one line grid covering every physical-detector direction.

    An automatic policy resolves from all directions jointly -- finest sinc
    spacing, the union of feature windows, and the widest measured bandwidth
    -- so lines that move across the detector stay inside refined windows.
    The scalar case grid is not reused: its windows are seeded only along the
    case's own direction. An explicit or stored grid is already final and is
    returned unchanged, as is a diagnostic-derivation case.

    Returns ``(E_grid, record)``; ``record`` is ``None`` for a final grid.
    """
    policy_payload = case.get("line_grid_policy")
    if policy_payload is None or case.get("_diagnostic_line_grid") is not None:
        return E_grid, None
    return _resolve_policy_grid(
        policy_payload, case, segments, n_hats, Ne, abs_layers, groove, observation=True
    )


def resolve_line_grid(case, segments, n_hat, Ne, E_grid, abs_layers=None, groove=None):
    """Choose this case's line grid once its own trajectories exist.

    Three paths, all before the spectrum phase and none of which starts a second
    Monte Carlo job:

    ``_diagnostic_line_grid``
        the private marker ``energy_grid.derive`` emits for its coverage scan.
    ``line_grid_policy``
        automatic case-local resolution (issue #101), attached by
        ``build_cases`` when neither an explicit grid nor a stored catalog row
        covers the case.
    neither
        the case's ``E_grid_line`` coordinates are already final.

    Returns ``(E_grid, record)``; ``record`` is ``None`` on the third path.
    """
    diagnostic_grid = case.get("_diagnostic_line_grid")
    policy_payload = case.get("line_grid_policy")
    if diagnostic_grid is not None and policy_payload is not None:
        raise ValueError(
            "a case carries both _diagnostic_line_grid and line_grid_policy; "
            "diagnostic derivation and automatic resolution are exclusive"
        )
    if policy_payload is not None:
        return _resolve_policy_line_grid(
            policy_payload, case, segments, n_hat, Ne, abs_layers, groove
        )
    if diagnostic_grid is None:
        return E_grid, None
    target_step, aliased_fraction, spacing_segments = sinc_feature_spacing(
        segments,
        n_hat,
        electron_limit=Ne,
        aliased_weight_limit=diagnostic_grid["aliased_weight_limit"],
    )
    maximum_spacing = diagnostic_grid.get("maximum_spacing_eV")
    if maximum_spacing is not None:
        if not np.isfinite(maximum_spacing) or maximum_spacing <= 0.0:
            raise ValueError("diagnostic maximum spacing must be finite and positive")
        target_step = min(target_step, float(maximum_spacing))
    start = float(diagnostic_grid["start_eV"])
    stop = float(diagnostic_grid["stop_eV"])
    num = resolution_num(start, stop, target_step)
    actual_step = validate_backend_spacing(
        start,
        stop,
        num,
        dtype=REAL,
        safety_ulps=diagnostic_grid["backend_safety_ulps"],
    )
    return np.linspace(start, stop, num), {
        "target_spacing_eV": target_step,
        "actual_spacing_eV": actual_step,
        "aliased_weight_fraction": aliased_fraction,
        "aliased_weight_limit": float(diagnostic_grid["aliased_weight_limit"]),
        "backend_dtype": np.dtype(REAL).name,
        "backend_safety_ulps": float(diagnostic_grid["backend_safety_ulps"]),
        "n_spacing_segments": spacing_segments,
        "observable_class": "intrinsic_source",
    }


#: Relative standard error of the per-electron line mass above which a
#: measured bandwidth case is flagged ``statistics_limited`` and warned about
#: (#201). A rare electron scattered into the detector's 1/gamma cone can carry
#: most of a case's line yield; the case still runs.
LINE_YIELD_RELATIVE_SE_LIMIT = 0.1


def line_truncation_audit(case, E_grid, n_electrons=None):
    """Empty edge-truncation audit for a ``resonance-population`` case, else ``None``.

    With ``n_electrons`` the audit also sums line mass per line electron, so
    :func:`check_line_truncation` can report the line yield's standard error.
    """
    payload = case.get("line_grid_policy")
    if payload is None or payload["bandwidth"]["policy"] != RESONANCE_BANDWIDTH_POLICY:
        return None
    grid = np.asarray(E_grid, dtype=float)
    audit = {"start_eV": float(grid[0]), "stop_eV": float(grid[-1])}
    if n_electrons is not None:
        audit["n_electrons"] = int(n_electrons)
    return audit


def line_yield_statistics(electron_mass):
    """Relative standard error of the mean line mass per electron (#201).

    Every line electron is one sample, zero-mass electrons included. The
    estimate describes the sampled electrons only: a heavy-tailed population
    whose rare heavy electrons were not sampled reports a small error around a
    biased mean, so ``max_electron_share`` is reported beside it.
    """
    mass = np.asarray(_to_cpu(electron_mass), dtype=float)
    n = int(mass.size)
    total = float(mass.sum())
    if n < 2 or total <= 0.0:
        return {"n_electrons": n, "relative_se": None, "max_electron_share": None}
    return {
        "n_electrons": n,
        "relative_se": float(mass.std(ddof=1) * np.sqrt(n) / total),
        "max_electron_share": float(mass.max() / total),
    }


def check_line_truncation(case, audit):
    """Gate a measured bandwidth on its production-weight truncation (#192).

    The upper-edge fraction is the bandwidth share this policy spends and is
    refused above ``truncation_limit``. The lower edge is reported only: the
    ``start`` convention predates this policy and is not its share. An axis
    already at the closed-form ceiling is warned about instead: the policy cut
    nothing the automatic bandwidth keeps. With
    per-electron sums, the line yield's standard error is recorded and a
    statistics-limited case is warned about, not refused (#201).

    Validation: line-grid-resonance-bandwidth
    """
    total = float(_to_cpu(audit.get("line_mass", 0.0)))
    above = float(_to_cpu(audit.get("mass_above", 0.0)))
    below = float(_to_cpu(audit.get("mass_below", 0.0)))
    limit = float(case["line_grid_policy"]["bandwidth"]["truncation_limit"])
    record = {
        "start_eV": audit["start_eV"],
        "stop_eV": audit["stop_eV"],
        "truncation_limit": limit,
        "upper_fraction_bound": above / total if total > 0.0 else 0.0,
        "lower_fraction_bound": below / total if total > 0.0 else 0.0,
    }
    if "electron_mass" in audit:
        stats = line_yield_statistics(audit["electron_mass"])
        record["line_yield_statistics"] = stats
        rse = stats["relative_se"]
        limited = rse is not None and rse > LINE_YIELD_RELATIVE_SE_LIMIT
        stats["statistics_limited"] = limited
        if limited:
            warnings.warn(
                f"{case['name']} at {case['E0_keV']:g} keV: the incoherent line yield "
                f"has relative standard error {rse:.3g} over {stats['n_electrons']} "
                f"electrons (one electron carries {stats['max_electron_share']:.3g} of "
                f"it), above {LINE_YIELD_RELATIVE_SE_LIMIT:g}. Rare electrons scattered "
                "toward the detector dominate this yield and its measured bandwidth; "
                "treat the line yield as statistics-limited or run more electrons.",
                LineYieldStatisticsWarning,
                stacklevel=2,
            )
    ceiling = float(case["line_grid_policy"]["bandwidth"]["stop_eV"])
    record["capped_at_ceiling"] = float(audit["stop_eV"]) >= ceiling
    if record["upper_fraction_bound"] > limit and record["capped_at_ceiling"]:
        # The axis already reaches the closed-form kinematic ceiling that the
        # automatic policy uses unaudited; no line resonates above it, so only
        # sinc**2 tails cross the edge and this policy cut nothing.
        warnings.warn(
            f"{case['name']} at {case['E0_keV']:g} keV: the line axis reaches the "
            f"kinematic ceiling {ceiling:g} eV, yet line tails above it may hold "
            f"{record['upper_fraction_bound']:.3g} of the line yield (bound), above "
            f"the {limit:g} share. The automatic bandwidth truncates the same tails.",
            LineGridTruncationWarning,
            stacklevel=2,
        )
    elif record["upper_fraction_bound"] > limit:
        raise LineGridToleranceError(
            f"{case['name']} at {case['E0_keV']:g} keV: the measured line bandwidth "
            f"[{audit['start_eV']:g}, {audit['stop_eV']:g}] eV may truncate "
            f"{record['upper_fraction_bound']:.3g} of the line yield above its edge "
            f"with production weights, above the {limit:g} bandwidth share. Use the "
            "kinematic-ceiling bandwidth or supply an explicit grid."
        )
    return record
