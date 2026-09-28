"""Windowed automatic line-grid resolution (issue #101).

Pins the ``windows`` policy block (off by default, schema 2 only when present),
its precedence and identity, the budget and precision refusals of
``windowed_coordinates``, and runner resolution from a case's own segments,
including the content-addressed cache.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from pyrite import _line_grid_policy
from pyrite._energy_grid_encoding import decode_energy_grid
from pyrite._grid_semantics import is_uniform_grid
from pyrite._line_grid_policy import (
    DEFAULT_WINDOW_PROVIDERS,
    DEFAULT_WINDOW_SAMPLES_PER_FEATURE,
    DEFAULT_WINDOW_TAIL_WIDTHS,
    LINE_GRID_POLICY_SCHEMA,
    WINDOWED_LINE_GRID_POLICY_SCHEMA,
    LineGridToleranceError,
    LineShapePrecisionWarning,
    lineshape_precision_warning,
    resolve_line_grid_policy,
    windowed_coordinates,
)
from pyrite._line_windows import FeatureSeed, build_window_plan
from pyrite.campaign.config import material_sweep
from pyrite.campaign.profiles import case_content_key
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo.geometry import tilted_geometry
from pyrite.montecarlo.runner import line_grid as runner_line_grid
from pyrite.montecarlo.runner.line_grid import resolve_line_grid, resolve_observation_line_grid
from pyrite.montecarlo.spectrum import line_seeds

_ENV_NAMES = (
    "PYRITE_ENERGY_GRID_RTOL",
    "PYRITE_ENERGY_GRID_RTOL_INTRINSIC_SOURCE",
    "PYRITE_ENERGY_GRID_RTOL_DETECTED_COUNTS",
    "PYRITE_ENERGY_GRID_MAX_SPACING_EV",
    "PYRITE_ENERGY_GRID_ULPS",
    "PYRITE_ENERGY_GRID_MAX_POINTS",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_line_grid_policy, "cache_dir", lambda: tmp_path)


def _uncovered_energy_sweep(energy_keV=77.0, **policy):
    sweep = material_sweep("hopg")
    assert energy_keV not in {float(e) for e in sweep.detector.energy_bins.line_by_energy}
    return replace(
        sweep,
        beam=replace(sweep.beam, energy_keV=energy_keV),
        line_grid_policy=policy or None,
    )


# ---- policy block ------------------------------------------------------------
def test_window_defaults_match_the_seed_module():
    assert DEFAULT_WINDOW_PROVIDERS == line_seeds.DEFAULT_SEED_PROVIDERS
    assert DEFAULT_WINDOW_SAMPLES_PER_FEATURE == line_seeds.DEFAULT_SAMPLES_PER_FEATURE
    assert DEFAULT_WINDOW_TAIL_WIDTHS == line_seeds.DEFAULT_TAIL_WIDTHS


def test_policy_without_windows_keeps_the_historical_payload():
    default = resolve_line_grid_policy(start_eV=10.0, stop_eV=10_000.0).payload()
    assert default["schema"] == LINE_GRID_POLICY_SCHEMA
    assert "windows" not in default and "windows" not in default["sources"]
    disabled = resolve_line_grid_policy(
        start_eV=10.0,
        stop_eV=10_000.0,
        per_call={"windows": False},
        stored={"windows": True},
    ).payload()
    assert disabled == default


def test_windows_take_defaults_bump_the_schema_and_record_their_source():
    payload = resolve_line_grid_policy(
        start_eV=10.0, stop_eV=10_000.0, per_call={"windows": True}
    ).payload()
    assert payload["schema"] == WINDOWED_LINE_GRID_POLICY_SCHEMA
    assert payload["windows"] == {
        "policy": "feature-windows",
        "providers": list(DEFAULT_WINDOW_PROVIDERS),
        "samples_per_feature": DEFAULT_WINDOW_SAMPLES_PER_FEATURE,
        "tail_widths": DEFAULT_WINDOW_TAIL_WIDTHS,
    }
    assert payload["sources"]["windows"] == "per-call"
    stored = resolve_line_grid_policy(
        start_eV=10.0, stop_eV=10_000.0, stored={"windows": {"samples_per_feature": 10}}
    ).payload()
    assert stored["windows"]["samples_per_feature"] == 10
    assert stored["sources"]["windows"] == "stored configuration"
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    ("windows", "error"),
    [
        ({"samples_per_feature": 0}, ValueError),
        ({"samples_per_feature": 2.5}, ValueError),
        ({"tail_widths": -1.0}, ValueError),
        ({"tail_widths": 0.0}, ValueError),
        ({"providers": "pxr-kinematic"}, TypeError),
        ({"providers": []}, ValueError),
        ({"providers": ["pxr-kinematic", "pxr-kinematic"]}, ValueError),
        ({"bogus": 1}, ValueError),
        ("yes", TypeError),
    ],
)
def test_malformed_windows_are_refused(windows, error):
    with pytest.raises(error):
        resolve_line_grid_policy(start_eV=10.0, stop_eV=10_000.0, per_call={"windows": windows})


def test_window_policy_changes_case_identity():
    keys = {
        case_content_key(build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0])
        for sweep in (
            _uncovered_energy_sweep(),
            _uncovered_energy_sweep(windows=True),
            _uncovered_energy_sweep(windows={"samples_per_feature": 10}),
            _uncovered_energy_sweep(windows={"providers": ["pxr-kinematic"]}),
        )
    }
    assert len(keys) == 4


# ---- windowed coordinates ----------------------------------------------------
def _windowed_payload(*, start_eV=50.0, stop_eV=16_400.0, **resolution):
    payload = resolve_line_grid_policy(
        start_eV=start_eV, stop_eV=stop_eV, per_call={"windows": True}
    ).payload()
    payload["resolution"].update(resolution)
    return payload


def _seed(centre, half, spacing):
    return FeatureSeed("test", f"{centre:g}", centre, half, half, spacing)


def test_windowed_coordinates_refuse_a_plan_over_budget():
    plan = build_window_plan(50.0, 16_400.0, 3.0, [_seed(2000.0, 50.0, 0.01)])
    with pytest.raises(LineGridToleranceError) as caught:
        windowed_coordinates(_windowed_payload(max_points=5000), plan)
    message = str(caught.value)
    assert "feature windows" in message and "samples_per_feature" in message
    assert "not coarsened automatically" in message


def test_windowed_coordinates_report_a_backend_precision_shortfall():
    plan = build_window_plan(50.0, 16_400.0, 3.0, [_seed(15_000.0, 0.05, 5.0e-3)])
    with pytest.raises(LineGridToleranceError, match="PYRITE_FP64=1"):
        windowed_coordinates(_windowed_payload(), plan, dtype=np.float32)
    grid, record = windowed_coordinates(_windowed_payload(), plan, dtype=np.float64)
    assert record["num"] == grid.size
    assert record["window_plan"] == plan.payload()
    assert record["min_spacing_eV"] <= 5.0e-3


@pytest.mark.parametrize(
    ("seed", "dtype", "reach"),
    [
        (_seed(16_390.0, 5.0, 0.1), np.float32, 16_395.0),
        (_seed(16_390.0, 5.0, 0.1), np.float64, None),
        (_seed(16_000.0, 5.0, 0.1), np.float32, None),
        (None, np.float32, None),  # a 3 eV backbone above the binade is not a window
    ],
)
def test_a_float32_window_in_the_20_kev_binade_is_flagged(seed, dtype, reach):
    stop = 16_400.0 if seed is None or seed.centre_eV > 16_384.0 else 16_100.0
    plan = build_window_plan(50.0, stop, 3.0, [] if seed is None else [seed])
    _grid, record = windowed_coordinates(_windowed_payload(stop_eV=stop), plan, dtype=dtype)
    assert record.get("float32_lineshape_window_eV") == reach
    message = lineshape_precision_warning(record)
    assert (message is None) == (reach is None)
    if message is not None:
        assert "PYRITE_FP64=1" in message and "FWHM" in message


def test_runner_warns_on_a_binade_window_from_a_cold_and_a_warm_cache(monkeypatch):
    # A float32 backend with the binade lowered into this 77 keV case's band.
    monkeypatch.setattr(runner_line_grid, "REAL", np.float32)
    monkeypatch.setattr(_line_grid_policy, "FLOAT32_LINESHAPE_BINADE_EV", 100.0)
    case, segments, n_hat = _case_and_segments(windows=True)
    placeholder = decode_energy_grid(case["E_grid_line"])
    with pytest.warns(LineShapePrecisionWarning, match="PYRITE_FP64=1"):
        _grid, record = resolve_line_grid(case, segments, n_hat, 2, placeholder)
    assert record["cache"] == "miss"
    with pytest.warns(LineShapePrecisionWarning):
        _grid, record = resolve_line_grid(case, segments, n_hat, 2, placeholder)
    assert record["cache"] == "hit"


def test_runner_budget_refusal_identifies_case_and_precision():
    case, segments, n_hat = _case_and_segments()
    policy = case["line_grid_policy"]
    case = replace(
        case,
        line_grid_policy={
            **policy,
            "resolution": {**policy["resolution"], "max_points": 100},
        },
    )
    placeholder = decode_energy_grid(case["E_grid_line"])

    with pytest.raises(LineGridToleranceError) as caught:
        resolve_line_grid(case, segments, n_hat, 2, placeholder)

    message = str(caught.value)
    assert case["name"] in message
    assert "at 77 keV" in message
    assert f"backend {np.dtype(runner_line_grid.REAL).name}" in message
    assert "above the 100-point budget" in message


def test_windowed_coordinates_refuse_a_plan_for_another_bandwidth():
    plan = build_window_plan(60.0, 16_400.0, 3.0, [_seed(2000.0, 5.0, 0.1)])
    with pytest.raises(ValueError, match="bandwidth"):
        windowed_coordinates(_windowed_payload(), plan)


# ---- runner ------------------------------------------------------------------
def _case_and_segments(**policy):
    case = build_cases(_uncovered_energy_sweep(**policy), n_electrons=2, n_electrons_brem=2)[0]
    beam, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    rng = np.random.default_rng(0)
    count = 400
    directions = beam + 0.15 * rng.standard_normal((count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    segments = {
        "E_keV": rng.uniform(60.0, 77.0, count),
        "L_ang": rng.uniform(200.0, 2000.0, count),
        "v_hat": directions,
        "elec_id": rng.integers(0, 2, count),
    }
    return case, segments, n_hat


def test_runner_resolves_a_windowed_grid_and_reuses_it_from_the_cache():
    case, segments, n_hat = _case_and_segments(windows=True)
    placeholder = decode_energy_grid(case["E_grid_line"])
    grid, record = resolve_line_grid(case, segments, n_hat, 2, placeholder)
    policy = case["line_grid_policy"]

    assert record["cache"] == "miss"
    assert record["schema"] == WINDOWED_LINE_GRID_POLICY_SCHEMA
    assert grid[0] == policy["bandwidth"]["start_eV"]
    assert grid[-1] == policy["bandwidth"]["stop_eV"]
    assert np.all(np.diff(grid) > 0.0) and not is_uniform_grid(grid)
    assert record["num"] == grid.size == record["window_plan"]["num"]
    assert record["min_spacing_eV"] < record["backbone_spacing_eV"]
    assert record["window_seeds"]["pxr-kinematic"]["seeds"] >= 1
    assert any(seed["label"] == "C K" for seed in record["window_plan"]["seeds"])

    again, cached = resolve_line_grid(case, segments, n_hat, 2, placeholder)
    assert cached["cache"] == "hit"
    np.testing.assert_array_equal(again, grid)


def test_a_stale_cached_plan_is_recomputed_not_trusted(tmp_path):
    case, segments, n_hat = _case_and_segments(windows=True)
    placeholder = decode_energy_grid(case["E_grid_line"])
    grid, record = resolve_line_grid(case, segments, n_hat, 2, placeholder)
    key = record["cache_key"]
    path = tmp_path / "line-grids" / key[:2] / f"{key}.json"
    stored = json.loads(path.read_text())
    stored["window_plan"]["pieces"][0][2] *= 0.5
    path.write_text(json.dumps(stored))

    again, recomputed = resolve_line_grid(case, segments, n_hat, 2, placeholder)
    assert recomputed["cache"] == "miss"
    np.testing.assert_array_equal(again, grid)


def test_uniform_automatic_resolution_is_unchanged():
    case, segments, n_hat = _case_and_segments()
    grid, record = resolve_line_grid(
        case, segments, n_hat, 2, decode_energy_grid(case["E_grid_line"])
    )
    assert record["schema"] == LINE_GRID_POLICY_SCHEMA
    assert "window_plan" not in record
    np.testing.assert_array_equal(
        grid, np.linspace(record["start_eV"], record["stop_eV"], record["num"])
    )


# ---- physical-detector observation grid -------------------------------------
def _rotated(n_hat, degrees):
    axis = np.cross(n_hat, [0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    return (
        n_hat * np.cos(angle)
        + np.cross(axis, n_hat) * np.sin(angle)
        + axis * np.dot(axis, n_hat) * (1.0 - np.cos(angle))
    )


def _seed_set(record):
    return {(seed["label"], seed["centre_eV"]) for seed in record["window_plan"]["seeds"]}


@pytest.mark.parametrize("windows", [True, None])
def test_one_observation_direction_reproduces_the_scalar_grid(windows):
    case, segments, n_hat = _case_and_segments(**({"windows": True} if windows else {}))
    placeholder = decode_energy_grid(case["E_grid_line"])
    scalar, scalar_record = resolve_line_grid(case, segments, n_hat, 2, placeholder)
    observed, record = resolve_observation_line_grid(case, segments, n_hat[None], 2, placeholder)

    np.testing.assert_array_equal(observed, scalar)
    assert record["cache_key"] != scalar_record["cache_key"]
    assert record["observation_direction_count"] == 1


def test_observation_windows_cover_every_direction():
    case, segments, n_hat = _case_and_segments(windows=True)
    placeholder = decode_energy_grid(case["E_grid_line"])
    other = _rotated(n_hat, 6.0)
    _, first = resolve_line_grid(case, segments, n_hat, 2, placeholder)
    _, second = resolve_observation_line_grid(case, segments, other[None], 2, placeholder)
    grid, joint = resolve_observation_line_grid(
        case, segments, np.stack([n_hat, other]), 2, placeholder
    )

    assert _seed_set(first) | _seed_set(second) == _seed_set(joint)
    assert _seed_set(second) - _seed_set(first)
    assert joint["feature_width_eV"] == min(first["feature_width_eV"], second["feature_width_eV"])
    assert np.all(np.diff(grid) > 0.0)


def test_uniform_observation_grid_takes_the_finest_direction():
    case, segments, n_hat = _case_and_segments()
    placeholder = decode_energy_grid(case["E_grid_line"])
    others = [_rotated(n_hat, degrees) for degrees in (-8.0, 8.0)]
    singles = [
        resolve_observation_line_grid(case, segments, direction[None], 2, placeholder)[1]
        for direction in others
    ]
    _, joint = resolve_observation_line_grid(case, segments, np.stack(others), 2, placeholder)

    assert joint["target_spacing_eV"] == min(single["target_spacing_eV"] for single in singles)


def test_observation_grid_keeps_an_explicit_grid():
    case, segments, n_hat = _case_and_segments()
    case = {key: value for key, value in case.items() if key != "line_grid_policy"}
    explicit = np.linspace(100.0, 200.0, 11)
    grid, record = resolve_observation_line_grid(case, segments, n_hat[None], 2, explicit)
    assert grid is explicit and record is None
