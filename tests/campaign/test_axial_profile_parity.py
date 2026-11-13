"""#624: admitted axial fibre cases lowered from a profile equal the API cases.

A user-layer profile names the packaged ``point`` detector and a named target
with ``cut = {surface_hkl = [0, 0, 1]}``, ``footprint = "unbounded"`` (and, for
si-fibre/B, ``allow_normal_incidence``). Neither side declares a continuum
grid, so both resolve the same automatic brem grid. A profile case name carries
the campaign ``ne=<line>/<brem>`` count suffix used for checkpoint resume;
``name`` is excluded from ``case_content_key`` and is the only field allowed
to differ.
"""

from dataclasses import replace

import numpy as np
import pytest
from click.testing import CliRunner

import pyrite as pr
from pyrite.api import build_configured_cases
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.lowering import build_case
from pyrite.campaign.profiles import case_content_key, profile_run_settings
from pyrite.config import FixedSampling
from pyrite.detectors import EnergyBins, PointDetector
from pyrite.materials.catalog import load_material_catalog
from pyrite.quantum._admission import AxialAdmissionError
from tests.helpers.axial_profiles import BREM, LINES, TILT, axial_profiles

FIBRE = {"version": 2, "axis_uvw": [0, 0, 1]}
GRID = np.linspace(800.0, 10000.0, 47)


@pytest.fixture
def axial_catalog(_isolate_user_catalog):
    with axial_profiles(_isolate_user_catalog) as layer:
        yield layer


def api_case(key):
    """The public-API lowering of an admitted case, continuum grid undeclared."""
    tilt, azimuth = TILT[key]
    beam = pr.Beam(
        energy_keV=3500.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    target = pr.geometry.Slab(
        "silicon",
        thickness_ang=3000.0,
        tilt_deg=tilt,
        tilt_azim_deg=azimuth,
        width=None,
        height=None,
        cut=pr.geometry.CrystalCut(surface_hkl=(0, 0, 1)),
        allow_normal_incidence=tilt == 0.0,
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=LINES, brem_histories=BREM),
        energy_bins=EnergyBins(cxr=GRID),
    )
    return build_case(
        pr.Scene(beam, target, PointDetector()),
        numerics,
        physics=pr.Physics(emission="incoherent", radiative_model="uncoupled"),
        channeling=pr.Channeling("axial", axial=FIBRE),
    )


def profile_cases(profile):
    catalog = load_material_catalog(profile=profile)
    settings = profile_run_settings(default_settings(), profile, catalog)
    return build_configured_cases(material_sweep("silicon", catalog_profile=profile), settings)


def _without_name(case):
    return {key: value for key, value in case.to_dict().items() if key != "name"}


@pytest.mark.parametrize("key", ["A", "B"])
def test_profile_resolves_the_api_case(axial_catalog, key):
    (case,) = profile_cases(f"axial_fibre_{key.lower()}")
    expected = api_case(key)
    assert case["channeling"]["case_key"] == f"si-fibre/{key}"
    assert (case["theta_obs_rad"], case["dtheta_obs_rad"], case["domega_sr"]) == (0, 0, 0)
    assert case["name"] == f"{expected['name']} ne={LINES}/{BREM}"
    assert repr(_without_name(case)) == repr(_without_name(expected))
    assert case_content_key(case) == case_content_key(expected)


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ("axial_fibre_thin", "accept-inaccurate"),
        ("axial_fibre_finite", "laterally unbounded"),
        ("axial_fibre_aperture", "zero-aperture"),
        ("axial_fibre_off_axis", "zero-aperture"),
    ],
)
def test_profiles_outside_the_envelope_still_refuse(axial_catalog, profile, message):
    with pytest.raises(AxialAdmissionError, match=message):
        profile_cases(profile)


def test_point_profile_refuses_acceptance_overrides(axial_catalog):
    with pytest.raises(ValueError, match="no angular acceptance; remove polar_acceptance_deg"):
        material_sweep("silicon", catalog_profile="axial_fibre_b", dtheta_obs_deg=0.1)
    moved = material_sweep("silicon", catalog_profile="axial_fibre_b", theta_obs_deg=1.0)
    assert moved.detector == PointDetector(1.0)


@pytest.mark.parametrize("output", ["table", "json"])
def test_cli_refuses_unadmitted_profile_before_writes(axial_catalog, tmp_path, output):
    from pyrite.cli.commands import scan as scan_cli

    checkpoints = tmp_path / "checkpoints"
    result = CliRunner().invoke(
        scan_cli.command,
        ["axial_fibre_thin", "-m", "silicon", "--checkpoint-dir", str(checkpoints)]
        + ["--output", output],
        catch_exceptions=False,
    )
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    assert "accept-inaccurate" in (result.stdout if output == "json" else result.stderr)
    assert not checkpoints.exists()


@pytest.mark.slow
def test_cli_run_persists_the_api_output(axial_catalog, tmp_path):
    """The CLI checkpoint of si-fibre/B equals the API case run and stored."""
    import json

    from pyrite.checkpoints import _checkpoint_store
    from pyrite.cli.commands import scan as scan_cli
    from pyrite.montecarlo import runner
    from pyrite.results.store import store_result

    checkpoints = tmp_path / "checkpoints"
    human = CliRunner().invoke(
        scan_cli.command,
        ["axial_fibre_b", "-m", "silicon", "--checkpoint-dir", str(checkpoints)]
        + ["--workers", "0", "--no-progress"],
        catch_exceptions=False,
    )
    assert human.exit_code == 0, human.output
    assert "experimental axial fibre channeling (si-fibre/B)" in human.stderr
    machine = CliRunner().invoke(
        scan_cli.command,
        ["axial_fibre_b", "-m", "silicon", "--checkpoint-dir", str(checkpoints)]
        + ["--workers", "0", "--output", "json"],
        catch_exceptions=False,
    )
    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.stdout)
    assert payload["ok"] and payload["payload"]["completed_materials"] == ["silicon"]
    (stem,) = [path.name for path in checkpoints.iterdir() if path.name.startswith("silicon@")]
    loaded = _checkpoint_store.load(stem, checkpoints)
    persisted = next(iter(next(iter(loaded.values())).values()))

    case = api_case("B")
    records = {}
    store_result(records, case, dict(runner.run_case(case)))
    expected = next(iter(next(iter(records.values())).values()))
    assert np.isfinite(persisted["spec_channeling"]).all()
    arrays = sorted(key for key, value in persisted.items() if isinstance(value, np.ndarray))
    assert {"spec", "spec_channeling"} <= set(arrays)
    assert arrays == sorted(key for key, value in expected.items() if isinstance(value, np.ndarray))
    for key in arrays:
        np.testing.assert_array_equal(persisted[key], expected[key], err_msg=key)
    assert persisted["channeling_diagnostics"].keys() == expected["channeling_diagnostics"].keys()
    assert persisted["case"]["channeling"] == dict(case["channeling"])
    assert replace(case, name=persisted["case"]["name"]).to_dict().keys() == (
        persisted["case"].keys()
    )
