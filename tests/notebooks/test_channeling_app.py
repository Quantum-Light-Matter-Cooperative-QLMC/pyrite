"""Scientific and reactive contracts of the planar channeling demo."""

import ast
from pathlib import Path

import numpy as np
import pytest

from pyrite.apps.channeling import critical_angle_rad
from pyrite.materials.channeling import longitudinal_kinematics, planar_profile, transverse_states
from pyrite.materials.crystal import M_E_EV

APP = Path(__file__).parents[2] / "src/pyrite/apps/channeling_app.py"


def test_si110_critical_angle_uses_the_well_depth():
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    gamma, beta = longitudinal_kinematics(5e6)
    minimum, maximum = profile.extrema()
    expected = np.sqrt(2 * (maximum - minimum) / (gamma * M_E_EV * beta**2))
    assert critical_angle_rad(profile, 5e6) == pytest.approx(expected)
    assert expected > 0


def test_display_selectors_do_not_feed_expensive_cells():
    tree = ast.parse(APP.read_text())
    expensive = {"propagate", "channeling_spectrum", "entry_populations", "transverse_states"}
    found = set()
    for cell in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(cell)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        if called & expensive:
            found |= called & expensive
            assert not {"state_ui", "sample_ui", "conditional_ui"} & {
                arg.arg for arg in cell.args.args
            }
    assert found == expensive


def test_every_plotted_eigenfunction_has_unit_probability_per_period():
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    gamma, _ = longitudinal_kinematics(5e6)
    states = transverse_states(profile, gamma, n_plane_waves=16)
    x = np.linspace(0, profile.period_ang, 257)
    for state in range(states.energies_eV.size):
        assert np.trapezoid(np.abs(states.wavefunction(x, state)) ** 2, x) == pytest.approx(
            1,
            abs=1e-13,
        )


def test_axial_display_controls_do_not_feed_solve():
    tree = ast.parse(APP.read_text())
    solve_cells = [
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "solve_axial_demo"
            for node in ast.walk(cell)
        )
    ]
    assert len(solve_cells) == 1
    assert {arg.arg for arg in solve_cells[0].args.args} == {"axial_form", "mo"}


@pytest.mark.parametrize("mode", ["coherent", "mcwf"])
def test_axial_app_executes_same_api_and_reuses_solve(mode):
    from types import SimpleNamespace

    import matplotlib.pyplot as plt

    from pyrite.apps.axial_channeling import solve_axial_demo
    from pyrite.apps.channeling_app import app
    from pyrite.montecarlo.channeling.axial_diagnostic import axial_diagnostic

    values = dict(
        crystal="silicon",
        axis="001",
        energy=5,
        mode=mode,
        cells=1,
        grid=16,
        depth=50,
        step=5,
        electrons=2,
        seed=547,
        preset="Gaussian",
        waist=0.5,
        impact_x=0,
        impact_y=0,
        tilt_x=0,
        tilt_y=0,
        oam=False,
    )
    _, definitions = app.run(
        defs={
            "mode_ui": SimpleNamespace(value="Axial (experimental)"),
            "axial_form": SimpleNamespace(value=values),
        }
    )
    result = definitions["axial_result"]
    direct = axial_diagnostic(result.config, entry=result.entry)
    np.testing.assert_allclose(result.density_per_ang2, direct.density_per_ang2)
    assert solve_axial_demo(result.config, "Gaussian", 0.5, (0, 0)) is result
    assert (
        result.wavefunctions_per_ang is None
        if mode == "mcwf"
        else result.wavefunctions_per_ang is not None
    )
    plt.close("all")


def test_axial_app_bounds_and_cache():
    from dataclasses import replace

    import matplotlib.pyplot as plt

    from pyrite.apps.axial_channeling import axial_demo_figure, solve_axial_demo
    from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig

    config = AxialDiagnosticConfig(shape=(16, 16))
    result = solve_axial_demo(config, "Equal ±1 mixture")
    assert solve_axial_demo(config, "Equal ±1 mixture") is result
    assert result.wavefunctions_per_ang is None
    figure = axial_demo_figure(result, 1, True)
    assert "No unique phase" in figure.axes[2].texts[0].get_text()
    plt.close(figure)
    for updates in (
        {"shape": (64, 64)},
        {"n_electrons": 33},
        {"depths_ang": tuple(range(0, 500, 10))},
        {"depths_ang": (0.0, 1e8)},
        {"mode": "mcwf", "n_electrons": 32, "depths_ang": (0.0, 1e7)},
    ):
        with pytest.raises(ValueError, match="app limit"):
            solve_axial_demo(replace(config, **updates))


def test_axial_app_depths_reach_dechanneling_scale():
    from pyrite.apps.axial_channeling import APP_MAX_DEPTHS, axial_demo_depths

    depths = axial_demo_depths(1e7, 10)
    assert depths[0] == 0 and depths[-1] == 1e7 and len(depths) == APP_MAX_DEPTHS
    np.testing.assert_allclose(np.mod(depths, 10), 0)
    assert axial_demo_depths(100, 5) == tuple(float(5 * i) for i in range(21))
    with pytest.raises(ValueError, match="at least one step"):
        axial_demo_depths(1, 5)


def test_axial_maps_center_the_origin_string():
    import matplotlib.pyplot as plt

    from pyrite.apps.axial_channeling import axial_demo_figure, solve_axial_demo
    from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig

    result = solve_axial_demo(AxialDiagnosticConfig(shape=(16, 16), depths_ang=(0.0,)))
    figure = axial_demo_figure(result, 0)
    for ax, pick in ((figure.axes[0], np.argmin), (figure.axes[1], np.argmax)):
        image = ax.images[0]
        left, right, bottom, top = image.get_extent()
        shown = image.get_array()
        row, col = np.unravel_index(pick(shown), shown.shape)
        x = left + (col + 0.5) * (right - left) / shown.shape[1]
        y = bottom + (row + 0.5) * (top - bottom) / shown.shape[0]
        assert abs(x) < 1e-12 and abs(y) < 1e-12
        assert left < 0 < right and bottom < 0 < top
    plt.close(figure)


def test_axial_form_waits_for_run():
    from types import SimpleNamespace

    from pyrite.apps.channeling_app import app

    _, definitions = app.run(defs={"mode_ui": SimpleNamespace(value="Axial (experimental)")})
    assert definitions["axial_form"].value is None
    assert "axial_result" not in definitions


def test_axial_radiation_reports_vortex_helicity_and_far_field():
    import matplotlib.pyplot as plt

    from pyrite.apps.axial_channeling import (
        axial_demo_depths,
        axial_line_options,
        axial_radiation,
        axial_radiation_figures,
        solve_axial_demo,
    )
    from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig

    config = AxialDiagnosticConfig(shape=(32, 32), depths_ang=axial_demo_depths(500, 5))
    vortex = axial_radiation(solve_axial_demo(config, "Vortex +1"))
    options, strongest = axial_line_options(vortex)
    assert options[strongest] == 0 and len(vortex.catalog.lines) == 1
    photons = vortex.photons(len(vortex.depths_ang) - 1)
    assert photons[0] > 0
    spectrum, far, moments = axial_radiation_figures(vortex, len(vortex.depths_ang) - 1, 0)
    assert moments["J_z"] == pytest.approx(1.0, abs=1e-3)
    assert far is not None and "purity 1.000" in far.texts[0].get_text()
    plt.close("all")
    # A centred Gaussian enters 1s only: no line emission and no map.
    gaussian = axial_radiation(solve_axial_demo(config, "Gaussian"))
    assert axial_radiation_figures(gaussian, 1, 0)[1] is None
    plt.close("all")
    # [110] has four strings per projected cell: no on-column catalog.
    assert axial_radiation(solve_axial_demo(replace_axis(config), "Gaussian")) is None


def replace_axis(config):
    from dataclasses import replace

    return replace(config, axis=(1, 1, 0), shape=(16, 16), depths_ang=(0.0, 10.0))


def test_planar_far_field_components_mirror_each_other():
    from pyrite.apps.channeling import far_field_maps

    axis, intensity, phase, purity = far_field_maps(10.0, np.diag([1.0, 0.0]), n=41)
    assert purity == pytest.approx(1.0)
    np.testing.assert_allclose(intensity[0], intensity[1], atol=1e-12)
    lit = intensity[0] > 1e-6 * intensity.max()
    np.testing.assert_allclose(np.exp(1j * phase[0][lit]), np.exp(-1j * phase[1][lit]), atol=1e-9)
    assert axis[0] == -3.0 and axis[-1] == 3.0


def test_ring_winding_reads_the_orbital_index():
    from pyrite.apps.channeling import ring_winding

    assert ring_winding(10.0, np.array([[1.0, -1j], [1j, 1.0]])) == (0, 2)  # J_z = +1
    assert ring_winding(10.0, np.array([[1.0, 1j], [-1j, 1.0]])) == (-2, 0)  # J_z = -1
    assert ring_winding(10.0, np.diag([1.0, 0.0])) == (-2, 2)  # planar, outside the nulls
    assert ring_winding(10.0, np.diag([1.0, 0.0]), radius=0.5) == (0, 0)


@pytest.mark.slow
def test_stored_population_views_preserve_normalization_and_empty_ensembles():
    import matplotlib.pyplot as plt

    from pyrite.apps.channeling_metrics import population_figure

    depths = np.array([0.0, 10.0])
    populations = np.array([[0.2, 0.1], [0.0, 0.0]])
    survival = np.array([0.5, 0.0])
    figure = population_figure(depths, populations, survival, 0, conditional=True)
    np.testing.assert_allclose([bar.get_height() for bar in figure.axes[0].patches], [0.4, 0.2])
    np.testing.assert_allclose(figure.axes[2].lines[2].get_ydata(), [0.2, 0.0])
    empty = population_figure(depths, populations, survival, 1, conditional=True)
    np.testing.assert_array_equal([bar.get_height() for bar in empty.axes[0].patches], [0.0, 0.0])
    plt.close(figure)
    plt.close(empty)


def test_metric_views_are_cached_array_consumers():
    tree = ast.parse(APP.read_text())
    views = []
    for cell in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        calls = {
            node.func.id
            for node in ast.walk(cell)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if calls & {"metric_figure", "axial_metric_figure"}:
            views.append(cell)
            assert not calls & {"solve_axial_demo", "axial_diagnostic", "DensityMatrixSolver"}
    assert len(views) == 2
