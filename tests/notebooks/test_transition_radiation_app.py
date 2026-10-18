"""Reference-profile parity, normalization, and reactive TR/CTR demo contracts."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.apps.transition_radiation import (
    DemoConfig,
    comparison_figure,
    photon_grid,
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
    from pyrite.materials import CATALOG

    settings = profile_run_settings(default_settings(), profile, "full", CATALOG)
    sweep = material_sweep("silicon", catalog_profile=profile)
    cases = build_configured_cases(sweep, settings)
    assert len(cases) == 1
    configured = cases[0].to_dict()
    direct = build_case(
        reference_scene(DemoConfig(), mode),
        pr.Numerics(
            cxr_trials=32,
            brem_trials=64,
            energy_bins=pr.EnergyBins(cxr=photon_grid(), brem=np.arange(200.0, 30000.0, 100.0)),
        ),
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
    "updates",
    [
        {"thickness_nm": 0},
        {"energy_keV": 1000},
        {"duration_as": float("nan")},
        {"charge_pc": 1},
        {"spot_nm": -1},
        {"angle_deg": 90},
    ],
)
def test_demo_rejects_unbounded_requests(updates):
    with pytest.raises(ValueError, match="demo limit"):
        replace(DemoConfig(), **updates)


def test_app_initial_export_has_analytic_plots_and_waits_for_transport():
    import matplotlib.pyplot as plt

    from pyrite.apps.transition_radiation_app import app

    _, definitions = app.run()
    assert definitions["scene_form"].value is None
    assert definitions["config"] == DemoConfig()
    assert definitions["analytic"]["energy"].size == 81
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

    _, definitions = app.run(defs={"transport_button": SimpleNamespace(value=True)})
    results = definitions["production"]
    assert transported_demo(DemoConfig())[0] is results
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
