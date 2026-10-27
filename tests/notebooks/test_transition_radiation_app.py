"""Reference-profile parity, normalization, and reactive TR/CTR demo contracts."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.apps.transition_radiation import (
    DEFAULT_LINE_HISTORIES,
    DemoConfig,
    comparison_figure,
    photon_grid,
    reference_physics,
    reference_scene,
    straight_track_demo,
    transport_figure,
    transported_demo,
)


@pytest.mark.parametrize("profile,mode", [("tr_demo", "history"), ("ctr_demo", "bunch")])
def test_reference_profiles_match_notebook_scene(profile, mode):
    import pyrite as pr
    from pyrite.api import build_case, build_configured_cases
    from pyrite.campaign.config import default_settings, material_sweep
    from pyrite.campaign.profiles import profile_run_settings
    from pyrite.detectors import EnergyBins
    from pyrite.materials import CATALOG

    settings = profile_run_settings(default_settings(), profile, "full", CATALOG)
    sweep = material_sweep("silicon", catalog_profile=profile)
    cases = build_configured_cases(sweep, settings)
    assert len(cases) == 1
    configured = cases[0].to_dict()
    direct = build_case(
        reference_scene(DemoConfig(), mode),
        pr.Numerics(
            cxr_trials=DEFAULT_LINE_HISTORIES,
            brem_trials=2 * DEFAULT_LINE_HISTORIES,
            energy_bins=EnergyBins(cxr=photon_grid(), brem=np.arange(200.0, 30000.0, 100.0)),
        ),
        physics=reference_physics(mode),
    ).to_dict()
    for key in (
        "crystal",
        "E0_keV",
        "thickness_ang",
        "tilt_deg",
        "tilt_azim_deg",
        "theta_obs_rad",
        "dtheta_obs_rad",
        "domega_sr",
        "Ne",
        "Ne_brem",
        "beam_fwhm_mm",
        "bunch_charge_pc",
        "bunch_length_fs",
        "long_shape",
        "longitudinal_distribution",
        "seed",
        "energy_model",
        "inelastic_model",
        "radiative_model",
        "bremsstrahlung_model",
        "transition_radiation",
        "crystal_width_mm",
        "crystal_height_mm",
    ):
        assert configured.get(key) == direct.get(key), key
    np.testing.assert_allclose(configured["E_grid_line"], direct["E_grid_line"])
    assert CATALOG.profile_materials(profile) == ("silicon",)


def test_straight_track_comparison_cache_and_bunch_scaling():
    import matplotlib.pyplot as plt

    config = DemoConfig()
    data = straight_track_demo(config)
    assert straight_track_demo(config) is data
    assert np.all(np.isfinite(data["film"])) and np.all(data["film"] > 0)
    assert np.max(data["ctr"] / data["spectrum"]) > 100
    long = straight_track_demo(replace(config, duration_as=100))
    np.testing.assert_allclose(long["ctr"], long["spectrum"], rtol=1e-12)
    fig = comparison_figure(data, config)
    bunch = comparison_figure(data, config, per_bunch=True, energy_eV=800)
    np.testing.assert_allclose(
        bunch.axes[0].lines[0].get_ydata(),
        fig.axes[0].lines[0].get_ydata() * config.population,
    )
    assert "800 eV" in bunch.axes[1].get_title()
    plt.close("all")


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"thickness_nm": 0}, "demo limit"),
        ({"energy_keV": 1e9}, "demo limit"),
        ({"duration_as": float("nan")}, "demo limit"),
        ({"spot_nm": -1}, "demo limit"),
        ({"angle_deg": 95}, "parallel"),  # 5° tilt: film plane
        ({"stop_eV": 40_000}, "kinetic"),
        ({"angle_min_deg": 10, "angle_max_deg": 0}, "min < max"),
        ({"energy_points": 4000, "angle_points": 4000}, "bound"),
    ],
)
def test_demo_rejects_unphysical_requests(updates, match):
    with pytest.raises(ValueError, match=match):
        replace(DemoConfig(), **updates)


def test_forward_relativistic_map_and_grazing_mask():
    from pyrite.apps.transition_radiation import map_panels

    config = DemoConfig(
        energy_keV=5000.0,
        angle_deg=3.0,
        tilt_deg=0.0,
        start_eV=500.0,
        stop_eV=5000.0,
        energy_points=8,
        angle_min_deg=-180.0,
        angle_max_deg=180.0,
        angle_points=9,
    )
    data = straight_track_demo(config)
    grazing = np.isclose(abs(data["angles"]), 90.0)
    assert np.isnan(data["film"][:, grazing]).all()
    # TR vanishes exactly along a normal-incidence track (0°, ±180°).
    off_axis = ~grazing & ~np.isclose(abs(data["angles"]) % 180, 0)
    assert np.all(data["film"][:, off_axis] > 0) and np.all(data["spectrum"] > 0)
    assert np.all(data["film"][:, np.isclose(data["angles"], 0)] < 1e-30)
    # Forward TR of a relativistic beam dominates the backward lobe.
    forward, backward = (
        np.nanmean(data["film"][:, np.isclose(data["angles"], a)]) for a in (45, 135)
    )
    assert forward > backward
    panels = map_panels(data, config, per_bunch=True)
    np.testing.assert_allclose(panels[0].values, data["film"].T * config.population)


def test_app_initial_export_has_analytic_plots_and_waits_for_transport():
    import matplotlib.pyplot as plt

    from pyrite.apps.transition_radiation_app import app

    _, definitions = app.run()
    assert definitions["scene_form"].value is None
    assert definitions["config"] == DemoConfig()
    assert definitions["analytic"]["energy"].size == 801
    assert definitions["analytic"]["angles"].size == 301
    assert definitions["trials_ui"].value == DEFAULT_LINE_HISTORIES
    assert "production" not in definitions
    plt.close("all")


def test_display_controls_do_not_feed_solvers():
    path = Path(__file__).parents[2] / "src/pyrite/apps/transition_radiation_app.py"
    tree = ast.parse(path.read_text())
    found = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if calls & {"straight_track_demo", "transported_demo"}:
            found |= calls & {"straight_track_demo", "transported_demo"}
            assert not {"per_bunch_ui", "cut_ui"} & {arg.arg for arg in node.args.args}
    assert found == {"straight_track_demo", "transported_demo"}


@pytest.mark.slow
def test_live_app_runs_production_and_preserves_signed_components():
    import matplotlib.pyplot as plt

    from pyrite.apps.transition_radiation_app import app

    _, definitions = app.run(
        defs={
            "transport_button": SimpleNamespace(value=True),
            "trials_ui": SimpleNamespace(value=32),
            "core_ui": SimpleNamespace(value="per-electron"),
        }
    )
    results = definitions["production"]
    assert transported_demo(DemoConfig(), trials=32, transport_core="per-electron")[0] is results
    for result in results.values():
        assert result.transition is not None
        assert np.all(np.isfinite(result.transition.density))
        np.testing.assert_allclose(
            result.combined_cxr().density - result.combined_cxr(include_transition=False).density,
            result.transition.density,
            atol=1e-20,
        )
    fig = transport_figure(results, DemoConfig())
    np.testing.assert_array_equal(
        fig.axes[1].lines[0].get_ydata(), results["bunch"].transition.density
    )
    plt.close("all")


def test_reference_scene_is_independent_of_transition_model():
    scene = reference_scene(DemoConfig(), "history")
    assert scene == reference_scene(DemoConfig(), "bunch")
    assert set(scene.__dict__) == {"beam", "target", "detector", "filters"}
    assert scene.physics.transition_radiation == "off"
    assert reference_physics("history").transition_radiation == "history"
    assert reference_physics("bunch").transition_radiation == "bunch"
