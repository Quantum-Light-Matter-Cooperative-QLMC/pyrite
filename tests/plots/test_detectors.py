"""Smoke tests for plots/detectors.py: every public figure builder called on a
small synthetic result, Agg backend, asserting a Figure with non-empty axes.
Not a pixel/physics check -- ``tests/plots/test_exports.py`` documents why
that's the wrong bar here; the point is exercising the drawing code paths
(log scales, wide-brem overlay, azimuth collapse) that only run inside a
real ``plt.plot``/``ax.set`` call."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cxr_mc.config import default_settings
from cxr_mc.plots.detectors import (
    plot_eaglexo_charge,
    plot_eaglexo_charge_map,
    plot_eaglexo_detected,
    plot_eaglexo_efficiency,
    plot_timepix_detected,
    plot_timepix_efficiency,
    plot_timepix_poisson,
)

# small n_mc keeps the Timepix MC response build fast; the response is cached
# per (grid, hardware, n_mc, seed) so repeated calls in this module are cheap.
_N_MC = 300


def _record(name, E0, tilt, azim, amp, wide=False):
    E = np.linspace(1000.0, 5000.0, 60)
    spec = amp * np.exp(-(((E - 2500.0) / 40.0) ** 2))
    brem = np.linspace(0.5, 0.1, 60)
    rec = {
        "E_grid": E,
        "spec": spec,
        "brem": brem,
        "scale": 1.0,
        "case": {
            "name": name,
            "crystal": "HOPG",
            "E0_keV": E0,
            "tilt_deg": tilt,
            "tilt_azim_deg": azim,
            "thickness_ang": 5.0e4,
        },
    }
    if wide:
        Eb = np.linspace(1000.0, E0 * 1e3, 40)
        rec["E_grid_brem"] = Eb
        rec["brem_wide"] = np.linspace(1.0, 0.05, 40)
    return rec


def _store():
    store = {}
    for tilt in (10.0, 20.0):
        for azim in (0.0, 30.0):
            name = f"HOPG t{tilt} a{azim}"
            store[name] = {
                E0: _record(name, E0, tilt, azim, amp=abs(tilt) + E0, wide=(E0 == 30.0))
                for E0 in (30.0, 60.0)
            }
    return store


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_plot_timepix_efficiency_returns_figure_with_two_axes():
    fig = plot_timepix_efficiency(n_mc=_N_MC)
    assert len(fig.axes) >= 2
    assert fig.axes[0].lines


def test_plot_timepix_detected_one_figure_per_tilt():
    figs = plot_timepix_detected(_store(), default_settings(), n_mc=_N_MC)
    assert len(figs) == 2  # two distinct tilts in _store()
    for fig in figs:
        assert fig.axes[0].lines


def test_plot_timepix_detected_empty_results_returns_empty_list(capsys):
    figs = plot_timepix_detected({}, default_settings(), n_mc=_N_MC)
    assert figs == []
    assert "no results" in capsys.readouterr().out


def test_plot_timepix_poisson_one_panel_per_energy():
    fig = plot_timepix_poisson(_store(), default_settings(), n_mc=_N_MC)
    assert len(fig.axes) == 2  # two beam energies in _store()
    for ax in fig.axes:
        assert ax.lines


def test_plot_timepix_poisson_empty_results_returns_none(capsys):
    assert plot_timepix_poisson({}, default_settings(), n_mc=_N_MC) is None
    assert "no results" in capsys.readouterr().out


def test_plot_eaglexo_efficiency_returns_figure_with_two_axes():
    fig = plot_eaglexo_efficiency()
    assert len(fig.axes) >= 2
    assert fig.axes[0].lines


def test_plot_eaglexo_detected_covers_wide_brem_and_resolve_energy():
    figs = plot_eaglexo_detected(_store(), default_settings(), resolve_energy=True)
    assert len(figs) == 2
    for fig in figs:
        assert fig.axes[0].lines


def test_plot_eaglexo_charge_covers_wide_brem():
    figs = plot_eaglexo_charge(_store(), default_settings())
    assert len(figs) == 2
    for fig in figs:
        assert fig.axes[0].lines


def test_plot_eaglexo_charge_map_default_grid():
    fig = plot_eaglexo_charge_map(_store(), default_settings())
    assert fig is not None
    assert fig.axes


def test_plot_eaglexo_charge_map_well_fill_fraction():
    fig = plot_eaglexo_charge_map(_store(), default_settings(), exposure_s=1.0)
    assert fig is not None
    assert fig.axes


def test_plot_eaglexo_charge_map_empty_results_returns_none():
    assert plot_eaglexo_charge_map({}, default_settings()) is None
