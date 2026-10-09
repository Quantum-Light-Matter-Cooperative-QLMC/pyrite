"""Smoke tests for the trajectories.py builders that _draw_trajectory_panel's
own test (test_trajectories.py) and the phase-space test (
test_trajectory_phase_space.py) don't reach: the three public entry points
that run real (small-Ne) electron transport via ``_trajectory_data`` --
``plot_electron_trajectories``, ``plot_trajectory_grid`` (both the 2-D
polar x azimuth grid and the wrapped/subsampled layouts), and
``plot_penetration_survival``. Agg backend; asserts a Figure/Axes comes back
with drawn content, not pixel output."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.plots.mpl.trajectories import (
    plot_electron_trajectories,
    plot_penetration_survival,
    plot_trajectory_grid,
)

_NE = 20  # electron count kept tiny: these exercise the plotting code, not stats


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _cases(tilt_deg, tilt_azim_deg, energy_keV):
    sweep = Sweep(
        material="hopg",
        thickness_ang=1.0e4,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        beam=BeamSpec(energy_keV=energy_keV),
    )
    return build_cases(sweep)


@pytest.mark.slow
def test_plot_electron_trajectories_draws_one_axis():
    case = _cases(10.0, 0.0, 20.0)[0]
    ax = plot_electron_trajectories(case, Ne=_NE, seed=0)
    assert ax.images and ax.patches  # datashader raster + the grey slab polygon
    assert ax.get_title()


def test_plot_electron_trajectories_reuses_an_existing_axis():
    case = _cases(10.0, 0.0, 20.0)[0]
    _, ax = plt.subplots()
    out = plot_electron_trajectories(case, Ne=_NE, seed=0, ax=ax, colorbar=False)
    assert out is ax


def test_plot_trajectory_grid_polar_by_azimuth_layout():
    cases = _cases([10.0, 20.0], [0.0, 30.0], 20.0)
    fig = plot_trajectory_grid(cases, Ne=_NE, seed=0)
    assert fig is not None
    assert len(fig.axes) >= 4  # 2x2 combos + shared colorbar axis


def test_plot_trajectory_grid_wraps_single_swept_axis():
    cases = _cases([10.0, 20.0, 30.0], 0.0, 20.0)
    fig = plot_trajectory_grid(cases, Ne=_NE, seed=0, ncols=2)
    assert fig is not None


def test_plot_trajectory_grid_subsamples_past_max_panels():
    cases = _cases([10.0, 20.0], [0.0, 30.0], 20.0)
    fig = plot_trajectory_grid(cases, Ne=_NE, seed=0, max_panels=2)
    assert fig is not None


def test_plot_trajectory_grid_empty_cases_prints_message(capsys):
    assert plot_trajectory_grid([], Ne=_NE) is None
    assert "no cases/results to plot" in capsys.readouterr().out


def test_plot_penetration_survival_one_line_per_energy():
    cases = _cases(10.0, 0.0, [20.0, 40.0])
    fig = plot_penetration_survival(cases, Ne=_NE, seed=0, n_bins=10)
    assert fig is not None
    assert len(fig.axes[0].lines) == 2


def test_plot_penetration_survival_renders_expected_survival_values(monkeypatch):
    case = {
        "name": "hopg survival",
        "E0_keV": 20.0,
        "tilt_deg": 10.0,
        "thickness_ang": 1.0e4,
    }
    data = {
        "Ne": 3,
        "elec_id": np.array([0, 0, 1, 2]),
        "z_u": np.array([0.2, 0.5, 0.1, 0.8]),
        "thick": 1.0,
    }
    monkeypatch.setattr(
        "pyrite.plots.mpl.trajectories._trajectory_data",
        lambda *_args, **_kwargs: data,
    )

    fig = plot_penetration_survival([case], Ne=_NE, seed=0, n_bins=3)

    assert fig is not None
    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(line.get_ydata(), [100.0, 200.0 / 3.0, 0.0])


def test_plot_penetration_survival_absolute_depth_axis():
    cases = _cases(10.0, 0.0, 20.0)
    fig = plot_penetration_survival(cases, Ne=_NE, seed=0, n_bins=10, depth_frac=False)
    assert fig is not None


def test_plot_penetration_survival_empty_cases_returns_none(capsys):
    assert plot_penetration_survival([], Ne=_NE) is None
    assert "no cases/results to plot" in capsys.readouterr().out
