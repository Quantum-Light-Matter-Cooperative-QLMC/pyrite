"""Bin-mean shape step of ``sinc-oversampled`` (issues #1230, #1265).

Pins the single-line shape bound of ``shape_spacing_factor`` (peak height and
FWHM read from bin means, as the #944 harness reads them), the payload and
default budget of a bin-mean ``sinc-oversampled`` axis, and the runner's
measured uniform axis at that step, which refuses rather than reroutes above
its budget.
"""

import math

import numpy as np
import pytest

from pyrite._cxr_grid_policy import (
    DEFAULT_BIN_MEAN_MAX_POINTS,
    DEFAULT_MAX_POINTS,
    DEFAULT_SHAPE_RTOL,
    KINEMATIC_BANDWIDTH_POLICY,
    LINE_NYQUIST_OVERSAMPLING,
    OVERSAMPLED_RESOLUTION_POLICY,
    RESONANCE_BANDWIDTH_POLICY,
    CxrGridToleranceError,
    resolve_cxr_grid_policy,
    resolved_coordinates,
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


def _policy(bandwidth=RESONANCE_BANDWIDTH_POLICY, quadrature="bin-mean", **per_call):
    return resolve_cxr_grid_policy(
        start_eV=50.0,
        stop_eV=100_000.0,
        per_call={"bandwidth": bandwidth, "quadrature": quadrature, **per_call},
    ).payload()


@pytest.mark.parametrize("bandwidth", [RESONANCE_BANDWIDTH_POLICY, KINEMATIC_BANDWIDTH_POLICY])
def test_bin_mean_oversampled_payload_carries_the_shape_tolerance_on_any_bandwidth(bandwidth):
    """Validation: line-grid-sinc-shape-spacing"""
    payload = _policy(bandwidth)
    resolution = payload["resolution"]
    assert resolution["policy"] == OVERSAMPLED_RESOLUTION_POLICY
    assert resolution["shape_rtol"] == DEFAULT_SHAPE_RTOL
    assert "oversampling" not in resolution
    assert resolution["max_points"] == DEFAULT_BIN_MEAN_MAX_POINTS
    assert "resolution" not in payload["sources"]


def test_node_oversampled_payload_is_unchanged():
    resolution = _policy(quadrature="node")["resolution"]
    assert resolution["oversampling"] == LINE_NYQUIST_OVERSAMPLING
    assert "shape_rtol" not in resolution
    assert resolution["max_points"] == DEFAULT_MAX_POINTS


def test_bin_mean_step_is_tighter_than_w_over_8():
    """``W/8`` bounds a bin-mean peak only to 1.7e-2; the shape step to 1e-2."""
    assert (math.pi**2 / 9) * (1 / LINE_NYQUIST_OVERSAMPLING) ** 2 > DEFAULT_SHAPE_RTOL
    assert shape_spacing_factor(DEFAULT_SHAPE_RTOL) < 1 / LINE_NYQUIST_OVERSAMPLING


def test_bin_mean_oversampled_axis_takes_the_shape_step_and_refuses_over_budget():
    """Validation: line-grid-sinc-shape-spacing"""
    payload = _policy(KINEMATIC_BANDWIDTH_POLICY)
    payload["bandwidth"]["stop_eV"] = 1000.0
    grid, record = resolved_coordinates(payload, 0.25)
    spacing = shape_spacing_factor(DEFAULT_SHAPE_RTOL) * 0.25
    assert record["target_spacing_eV"] == pytest.approx(spacing)
    assert record["oversampling"] == pytest.approx(1 / shape_spacing_factor(DEFAULT_SHAPE_RTOL))
    assert np.diff(grid.astype(np.float64)).max() <= spacing * (1 + 1e-6)
    payload["resolution"]["max_points"] = 1000
    with pytest.raises(CxrGridToleranceError, match="sinc-nyquist"):
        resolved_coordinates(payload, 0.25)


def _measured_grid(monkeypatch, payload):
    from pyrite.montecarlo import runner

    def collect(_segments, axis, _case, _direction, _layers, _groove, **kwargs):
        line = (np.array([5000.0]), np.array([1.0]), np.array([1.0]))
        kwargs["truncation_audit"]["collect"].append(line)
        return np.zeros(axis.size)

    monkeypatch.setattr(runner, "_cxr_for_segments", collect)
    case = {"name": "hbn test", "E0_keV": 5000.0, "composition": [("B", 0.5), ("N", 0.5)]}
    return _measured_cxr_grid(payload, case, {}, (np.array([0.0, 0.0, 1.0]),), 2, 0.25, None, None)


def test_measured_bin_mean_grid_is_uniform_at_the_shape_step(monkeypatch):
    grid, record, bandwidth = _measured_grid(monkeypatch, _policy())
    spacing = shape_spacing_factor(DEFAULT_SHAPE_RTOL) * 0.25
    assert grid[0] == 50.0
    assert grid[-1] == bandwidth["stop_eV"]
    steps = np.diff(grid.astype(np.float64))
    assert steps.max() <= spacing * (1 + 1e-6)
    assert steps.min() == pytest.approx(steps.max(), rel=1e-3)
    assert record["resolution_policy"] == OVERSAMPLED_RESOLUTION_POLICY
    assert "shape_axis" not in record


def test_measured_bin_mean_grid_refuses_above_budget_instead_of_rerouting(monkeypatch):
    payload = _policy()
    payload["resolution"]["max_points"] = 1000
    with pytest.raises(CxrGridToleranceError, match="1000-point budget") as refused:
        _measured_grid(monkeypatch, payload)
    # Decision 1 (#1265): the refusal names both opt-outs on the measured band.
    assert "sinc-nyquist" in str(refused.value)
    assert "resonance-local" in str(refused.value)
