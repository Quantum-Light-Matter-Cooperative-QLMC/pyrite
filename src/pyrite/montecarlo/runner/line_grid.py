"""Automatic case-local line-grid resolution for one run (issue #101).

Split out of ``runner/__init__`` to keep that module inside the source-size
budget; it is runner-internal and has no other consumer.
"""

import warnings

import numpy as np

from ..._backend import REAL
from ..._grid_semantics import resolution_num, validate_backend_spacing
from ..._line_grid_policy import (
    LineShapePrecisionWarning,
    cached_coordinates,
    coordinate_cache_key,
    lineshape_precision_warning,
    resolved_coordinates,
    store_coordinates,
    windowed_coordinates,
)
from ..._line_windows import build_window_plan, window_plan_from_payload
from ..spectrum.diagnostics import coherent_fringe_spacing, sinc_feature_spacing
from ..spectrum.line_seeds import SEEDING_REVISION, SeedContext, collect_feature_seeds

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


def _cached_grid(cached):
    """Coordinates of a cache record, or ``None`` when its plan is stale."""
    plan_payload = cached.get("window_plan")
    if plan_payload is None:
        return np.linspace(cached["start_eV"], cached["stop_eV"], int(cached["num"]))
    try:
        return window_plan_from_payload(plan_payload).coordinates()
    except KeyError, TypeError, ValueError:
        return None


def _windowed_line_grid(payload, case, segments, n_hat, Ne, feature_width_eV):
    """Seed, plan and validate a piecewise line axis from this run's segments.

    The backbone is the policy's maximum spacing. Every provider the policy
    names contributes windows, and kinematic window spacing follows the sinc
    feature width measured on these same segments.
    """
    windows = payload["windows"]
    start = float(payload["bandwidth"]["start_eV"])
    stop = float(payload["bandwidth"]["stop_eV"])
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
    seeds, summaries = collect_feature_seeds(context, windows["providers"])
    plan = build_window_plan(start, stop, float(payload["resolution"]["max_spacing_eV"]), seeds)
    grid, record = windowed_coordinates(payload, plan, dtype=REAL)
    record["feature_width_eV"] = float(feature_width_eV)
    record["window_seeds"] = summaries
    return grid, record


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


def _resolve_policy_line_grid(payload, case, segments, n_hat, Ne):
    """Resolve an automatic case-local line grid from this run's trajectories.

    Consults the content-addressed cache first, so a repeated case pays the
    sinc-feature scan once. The cache is a speed cache only: the case payload
    already carries the policy, and the policy plus the case determines the
    coordinates, so a cold and a warm cache resolve identically. A windowed
    policy also keys the cache on the orientation inputs its seeds read, and a
    hit rebuilds the coordinates from the stored plan; a plan this build would
    not make is treated as a miss.
    """
    _refuse_coherent_resolution(case, segments, n_hat, Ne)
    windowed = payload.get("windows") is not None
    keys = _RESOLUTION_INPUT_KEYS + (_WINDOW_INPUT_KEYS if windowed else ())
    inputs = {key: case[key] for key in keys if case.get(key, None) is not None}
    inputs["backend_dtype"] = np.dtype(REAL).name
    if windowed:
        inputs["seeding_revision"] = SEEDING_REVISION
    key = coordinate_cache_key(payload, inputs)
    cached = cached_coordinates(key)
    if cached is not None:
        grid = _cached_grid(cached)
        if grid is not None:
            _warn_lineshape_precision(cached)
            return grid, {**cached, "cache": "hit", "cache_key": key}
    target_step, aliased_fraction, spacing_segments = sinc_feature_spacing(
        segments,
        n_hat,
        electron_limit=Ne,
        aliased_weight_limit=float(payload["resolution"]["aliased_weight_limit"]),
    )
    if windowed:
        grid, record = _windowed_line_grid(payload, case, segments, n_hat, Ne, target_step)
    else:
        grid, record = resolved_coordinates(payload, target_step, dtype=REAL)
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
    store_coordinates(key, record)
    _warn_lineshape_precision(record)
    return grid, {**record, "cache": "miss", "cache_key": key}


def _warn_lineshape_precision(record):
    message = lineshape_precision_warning(record)
    if message is not None:
        warnings.warn(message, LineShapePrecisionWarning, stacklevel=3)


def resolve_line_grid(case, segments, n_hat, Ne, E_grid):
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
        return _resolve_policy_line_grid(policy_payload, case, segments, n_hat, Ne)
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
