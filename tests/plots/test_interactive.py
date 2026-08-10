"""Smoke tests for plots/mpl/interactive.py: the static-export path of ``browse``
for every ``kind``, the ipywidgets slider path (``_tilt_browser``), the Plotly
click-through (``browse_plotly``), ``plot_chunk``, and ``stream_chunk``. Agg
backend; asserts figures/traces get produced, not pixel content -- see
``tests/plots/test_exports.py`` for why."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyrite.config import default_settings
from pyrite.plots.mpl.interactive import (
    browse,
    browse_plotly,
    plot_chunk,
    stream_chunk,
)

_N_MC = 300  # keeps the Timepix MC response build fast for the "timepix" kind


def _record(name, E0, tilt, azim, amp, wide=False):
    E = np.linspace(1000.0, 5000.0, 60)
    spec = amp * np.exp(-(((E - 2500.0) / 40.0) ** 2))
    brem = np.linspace(0.5, 0.1, 60)
    rec = {
        "E_grid": E,
        "spec": spec,
        "brem": brem,
        "scale": 1.0,
        "fwhm": 20.0,
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


def _store(wide=False):
    store = {}
    for tilt in (10.0, 20.0):
        for azim in (0.0, 30.0):
            name = f"HOPG t{tilt} a{azim}"
            store[name] = {
                E0: _record(name, E0, tilt, azim, amp=abs(tilt) + E0, wide=wide)
                for E0 in (30.0, 60.0)
            }
    return store


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.mark.parametrize("kind", ["by_energy", "chunk", "eaglexo", "eaglexo_charge"])
def test_browse_static_one_figure_per_tilt(kind):
    figs = browse(_store(), default_settings(), kind=kind, static=True)
    assert len(figs) == 2
    for fig in figs:
        assert fig.axes[0].lines


def test_browse_static_timepix_kind():
    figs = browse(_store(), default_settings(), kind="timepix", static=True, n_mc=_N_MC)
    assert len(figs) == 2


def test_browse_static_full_kind_needs_wide_brem():
    figs = browse(_store(wide=True), default_settings(), kind="full", static=True)
    assert len(figs) == 2
    for fig in figs:
        assert fig.axes[0].lines


def test_browse_full_kind_without_wide_brem_prints_hint(capsys):
    result = browse(_store(wide=False), default_settings(), kind="full", static=True)
    assert result is None
    assert "E_grid_brem" in capsys.readouterr().out


def test_browse_empty_results_prints_message_and_returns_none(capsys):
    assert browse({}, default_settings(), static=True) is None
    assert "no results to browse" in capsys.readouterr().out


def test_browse_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind must be one of"):
        browse(_store(), default_settings(), kind="bogus")


def test_browse_interactive_path_builds_widgets_and_renders():
    # static=None auto-detects ipywidgets (installed here) -> _tilt_browser.
    result = browse(_store(), default_settings(), kind="chunk", static=None)
    assert result is None  # the widget path has no figure-list return


def test_browse_plotly_by_energy_has_one_slider_step_per_tilt():
    fig = browse_plotly(_store(), default_settings(), kind="by_energy")
    assert len(fig.data) > 0
    assert len(fig.layout.sliders[0].steps) == 2


def test_browse_plotly_full_kind_needs_wide_brem():
    fig = browse_plotly(_store(wide=True), default_settings(), kind="full")
    assert len(fig.data) > 0
    assert fig.layout.xaxis.type == "log"


def test_browse_plotly_full_kind_without_wide_brem_returns_none(capsys):
    assert browse_plotly(_store(wide=False), default_settings(), kind="full") is None
    assert "no results to browse" in capsys.readouterr().out


def test_browse_plotly_empty_results_returns_none(capsys):
    assert browse_plotly({}, default_settings()) is None


def test_browse_plotly_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind must be"):
        browse_plotly(_store(), default_settings(), kind="bogus")


def test_plot_chunk_one_figure_per_tilt():
    figs = plot_chunk(_store(), default_settings())
    assert len(figs) == 2
    for fig in figs:
        assert len(fig.axes) == 2


def test_plot_chunk_empty_results(capsys):
    assert plot_chunk({}, default_settings()) == []


def test_stream_chunk_prints_summary_table(capsys):
    stream_chunk(_store(), None, default_settings())
    # summary_table prints the QE/units caption line before the dataframe.
    assert "per-nA columns are intrinsic" in capsys.readouterr().out


def test_stream_chunk_without_azimuth_collapse(capsys):
    stream_chunk(_store(), None, default_settings(), collapse_azimuth=False)
    assert "per-nA columns are intrinsic" in capsys.readouterr().out
