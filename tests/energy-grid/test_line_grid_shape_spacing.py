"""Uniform ``sinc-shape`` line spacing for bounded-band bin-mean cases (issue #1230).

Pins the single-line shape bound of ``shape_spacing_factor`` (peak height and
FWHM read from bin means, as the #944 harness reads them) and the runner's
measured uniform axis with its ``resonance-local`` fallback above the budget.
"""

import math

import numpy as np
import pytest

from pyrite._cxr_grid_policy import (
    DEFAULT_LOCAL_HALO_LIMIT,
    DEFAULT_SHAPE_MAX_POINTS,
    DEFAULT_SHAPE_RTOL,
    KINEMATIC_BANDWIDTH_POLICY,
    LOCAL_RESOLUTION_POLICY,
    NYQUIST_RESOLUTION_POLICY,
    OVERSAMPLED_RESOLUTION_POLICY,
    RESONANCE_BANDWIDTH_POLICY,
    SHAPE_RESOLUTION_POLICY,
    default_resolution_policy,
    resolve_cxr_grid_policy,
    shape_spacing_factor,
)
from pyrite.montecarlo.runner.cxr_grid import _measured_cxr_grid
from pyrite.montecarlo.spectrum.cxr._bin_quadrature import sincsq_bin_masses

#: Half-maximum abscissa of ``sinc^2(x)``, in first-zero widths.
_SINCSQ_HALF_MAX_X = 0.442946470689452


def _peak_metrics(centres, density):
    """Highest bin and linearly interpolated FWHM (the #944 harness reading)."""
    k = int(np.argmax(density))
    half = 0.5 * density[k]
    i = int(np.flatnonzero(density[:k] < half)[-1])
    j = k + int(np.flatnonzero(density[k:] < half)[0])
    x_l = centres[i] + (half - density[i]) * (centres[i + 1] - centres[i]) / (
        density[i + 1] - density[i]
    )
    x_r = centres[j - 1] + (half - density[j - 1]) * (centres[j] - centres[j - 1]) / (
        density[j] - density[j - 1]
    )
    return float(density[k]), float(x_r - x_l)


@pytest.mark.parametrize("shape_rtol", [1e-2, 1e-3])
def test_shape_step_bounds_single_line_height_and_fwhm(shape_rtol):
    """Validation: line-grid-sinc-shape-spacing"""
    width = 0.2  # eV, first-zero width W
    h = shape_spacing_factor(shape_rtol) * width
    worst_height = worst_fwhm = 0.0
    for offset in np.linspace(0.0, 1.0, 41)[:-1]:
        centres = (np.arange(-60, 61) + offset) * h
        edges = np.concatenate((centres - 0.5 * h, centres[-1:] + 0.5 * h))
        density = sincsq_bin_masses(math.pi / width, 0.0, edges)[0] / h
        height, fwhm = _peak_metrics(centres, density)
        worst_height = max(worst_height, abs(height - 1.0))
        worst_fwhm = max(worst_fwhm, abs(fwhm / (2 * _SINCSQ_HALF_MAX_X * width) - 1.0))
    # The quadratic bound is tight at the half-bin offset and never exceeded.
    assert worst_height <= shape_rtol
    assert worst_height == pytest.approx(shape_rtol, rel=0.02)
    assert worst_fwhm <= shape_rtol


def test_shape_factor_closed_form_and_domain():
    assert shape_spacing_factor(1e-2) == pytest.approx(0.3 / math.pi)
    for bad in (0.0, 1.0, -1e-3):
        with pytest.raises(ValueError, match="shape_rtol"):
            shape_spacing_factor(bad)


def _shape_policy(**per_call):
    return resolve_cxr_grid_policy(
        start_eV=50.0,
        stop_eV=100_000.0,
        per_call={
            "bandwidth": RESONANCE_BANDWIDTH_POLICY,
            "resolution": SHAPE_RESOLUTION_POLICY,
            "quadrature": "bin-mean",
            **per_call,
        },
    ).payload()


def test_shape_policy_payload_carries_its_tolerance_budget_and_fallback_halo():
    resolution = _shape_policy()["resolution"]
    assert resolution["policy"] == SHAPE_RESOLUTION_POLICY
    assert resolution["shape_rtol"] == DEFAULT_SHAPE_RTOL
    assert resolution["shape_max_points"] == DEFAULT_SHAPE_MAX_POINTS
    assert resolution["halo_limit"] == DEFAULT_LOCAL_HALO_LIMIT


@pytest.mark.parametrize(
    ("bandwidth", "quadrature", "windows", "expected"),
    [
        (RESONANCE_BANDWIDTH_POLICY, "bin-mean", False, SHAPE_RESOLUTION_POLICY),
        (RESONANCE_BANDWIDTH_POLICY, "node", False, OVERSAMPLED_RESOLUTION_POLICY),
        (KINEMATIC_BANDWIDTH_POLICY, "bin-mean", False, OVERSAMPLED_RESOLUTION_POLICY),
        (KINEMATIC_BANDWIDTH_POLICY, "node", True, NYQUIST_RESOLUTION_POLICY),
    ],
)
def test_unset_resolution_follows_bandwidth_quadrature_and_windows(
    bandwidth, quadrature, windows, expected
):
    """Bin-mean on the measured band defaults to sinc-shape, else sinc-oversampled (#1214).

    Bin means at W/8 bound an isolated peak only to (pi^2/9)/64 = 1.7e-2.
    """
    assert (
        default_resolution_policy(bandwidth=bandwidth, quadrature=quadrature, windows=windows)
        == expected
    )
    if windows:
        return
    payload = resolve_cxr_grid_policy(
        start_eV=50.0,
        stop_eV=1000.0,
        per_call={"bandwidth": bandwidth, "quadrature": quadrature},
    ).payload()
    assert payload["resolution"]["policy"] == expected
    assert "resolution" not in payload["sources"]
    assert (math.pi**2 / 9) * (1 / 8) ** 2 > DEFAULT_SHAPE_RTOL


def test_shape_resolution_needs_measured_bandwidth_and_bin_mean():
    with pytest.raises(ValueError, match="needs the resonance-population bandwidth"):
        resolve_cxr_grid_policy(
            start_eV=50.0,
            stop_eV=1000.0,
            per_call={"resolution": SHAPE_RESOLUTION_POLICY, "bandwidth": "kinematic-ceiling"},
        )
    with pytest.raises(ValueError, match="needs bin-mean quadrature"):
        _shape_policy(quadrature="node")


def _measured_shape_grid(monkeypatch, payload):
    from pyrite.montecarlo import runner

    def collect(_segments, axis, _case, _direction, _layers, _groove, **kwargs):
        line = (np.array([5000.0]), np.array([1.0]), np.array([1.0]))
        kwargs["truncation_audit"]["collect"].append(line)
        return np.zeros(axis.size)

    monkeypatch.setattr(runner, "_cxr_for_segments", collect)
    case = {"name": "hbn test", "E0_keV": 5000.0, "composition": [("B", 0.5), ("N", 0.5)]}
    return _measured_cxr_grid(payload, case, {}, (np.array([0.0, 0.0, 1.0]),), 2, 0.25, None, None)


def test_measured_shape_grid_is_uniform_at_the_shape_step(monkeypatch):
    grid, record, bandwidth = _measured_shape_grid(monkeypatch, _shape_policy())
    spacing = shape_spacing_factor(DEFAULT_SHAPE_RTOL) * 0.25
    assert grid[0] == 50.0
    assert grid[-1] == bandwidth["stop_eV"]
    steps = np.diff(grid.astype(np.float64))
    assert steps.max() <= spacing * (1 + 1e-6)
    assert steps.min() == pytest.approx(steps.max(), rel=1e-3)
    shape = record["shape_axis"]
    assert shape["route"] == "uniform"
    assert shape["spacing_eV"] == pytest.approx(spacing)
    assert shape["required_points"] == grid.size
    assert record["resolution_policy"] == SHAPE_RESOLUTION_POLICY
    assert "payload" not in shape


def test_measured_shape_grid_falls_back_to_resonance_local_above_budget(monkeypatch):
    payload = _shape_policy()
    payload["resolution"]["shape_max_points"] = 1000
    grid, record, _ = _measured_shape_grid(monkeypatch, payload)
    shape = record["shape_axis"]
    assert shape["route"] == LOCAL_RESOLUTION_POLICY
    assert "above the 1000-point sinc-shape budget" in shape["reason"]
    assert shape["required_points"] > 1000
    assert record["local_spacing"]["narrow_lines"] == 1
    assert np.diff(grid).max() == pytest.approx(3.0)
