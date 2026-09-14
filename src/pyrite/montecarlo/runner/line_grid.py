"""Automatic case-local line-grid resolution for one run (issue #101).

Split out of ``runner/__init__`` to keep that module inside the source-size
budget; it is runner-internal and has no other consumer.
"""

from __future__ import annotations

import numpy as np

from ..._backend import REAL
from ..._grid_semantics import resolution_num, validate_backend_spacing
from ..._line_grid_policy import (
    cached_coordinates,
    coordinate_cache_key,
    resolved_coordinates,
    store_coordinates,
)
from ..spectrum.diagnostics import sinc_feature_spacing

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
    "abs_layers",
    "Ne",
    "Ne_brem",
    "seed",
    "E_cut_lines_keV",
    "E_cut_brem_keV",
)


def _resolve_policy_line_grid(payload, case, segments, n_hat, Ne):
    """Resolve an automatic case-local line grid from this run's trajectories.

    Consults the content-addressed cache first, so a repeated case pays the
    sinc-feature scan once. The cache is a speed cache only: the case payload
    already carries the policy, and the policy plus the case determines the
    coordinates, so a cold and a warm cache resolve identically.
    """
    inputs = {key: case[key] for key in _RESOLUTION_INPUT_KEYS if case.get(key, None) is not None}
    inputs["backend_dtype"] = np.dtype(REAL).name
    key = coordinate_cache_key(payload, inputs)
    cached = cached_coordinates(key)
    if cached is not None:
        grid = np.linspace(cached["start_eV"], cached["stop_eV"], int(cached["num"]))
        return grid, {**cached, "cache": "hit", "cache_key": key}
    target_step, aliased_fraction, spacing_segments = sinc_feature_spacing(
        segments,
        n_hat,
        electron_limit=Ne,
        aliased_weight_limit=float(payload["resolution"]["aliased_weight_limit"]),
    )
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
    return grid, {**record, "cache": "miss", "cache_key": key}


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
