import json
import shutil

import pytest

from pyrite._catalog_layout import read_text
from pyrite.cli import _catalog_io
from pyrite.cli.commands import profile
from tests.helpers.cli import assert_clean_result, invoke

SET = ["physical-detector", "set"]
ACQUIRE = [
    "--exposure-s",
    "2",
    "--measured-range-ev",
    "0",
    "2000",
    "--measured-bin-width-ev",
    "400",
    "--hit-threshold-ev",
    "500",
]


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    from pyrite import DATA_DIR

    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    path = data / "materials.toml"
    path.write_text(read_text(data / "catalog"))
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", path)
    return path


def _show(name="standard"):
    result = invoke(profile.command, ["physical-detector", "show", name, "-o", "json"])
    assert_clean_result(result)
    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.profile.physical-detector.show"
    return document["payload"]


def _create(name="standard", *extra):
    return invoke(
        profile.command,
        [
            *SET,
            name,
            "-y",
            "--distance-mm",
            "100",
            "--polar-deg",
            "60",
            "--shape",
            "8",
            "10",
            *extra,
        ],
    )


def test_show_reports_a_profile_without_a_physical_detector(catalog) -> None:
    text = invoke(profile.command, ["physical-detector", "show", "standard"])
    assert_clean_result(text, stdout="standard: no physical detector\n")
    assert _show() == {
        "profile": "standard",
        "source": None,
        "physical_detector": None,
        "counting_observation": False,
        "measured_edges_eV": None,
        "scalar_projection": None,
    }


def test_creating_a_detector_needs_a_distance(catalog) -> None:
    before = catalog.read_text()
    result = invoke(profile.command, [*SET, "standard", "-y", "--polar-deg", "60"])
    assert result.exit_code == 2
    assert "requires --distance-mm" in result.stderr
    assert catalog.read_text() == before


def test_set_without_fields_is_a_usage_error(catalog) -> None:
    result = invoke(profile.command, [*SET, "standard", "-y"])
    assert result.exit_code == 2
    assert "nothing to set" in result.stderr


def test_geometry_then_acquisition_makes_a_counting_observation(catalog) -> None:
    created = _create()
    assert_clean_result(created, stdout="updated physical detector for profile standard\n")
    payload = _show()
    assert payload["source"] == "profile"
    assert payload["physical_detector"]["shape"] == [8, 10]
    assert not payload["counting_observation"]

    acquired = invoke(profile.command, [*SET, "standard", "-y", *ACQUIRE])

    assert acquired.exit_code == 0
    assert acquired.stdout == "updated physical detector for profile standard\n"
    assert "note: sweeps of standard use this detector's projection" in acquired.stderr
    payload = _show()
    assert payload["counting_observation"]
    assert payload["measured_edges_eV"] == [0.0, 400.0, 800.0, 1200.0, 1600.0, 2000.0]
    assert payload["scalar_projection"]["observation_angle_deg"] == pytest.approx(60.0)
    assert payload["physical_detector"]["distance_mm"] == 100.0

    human = invoke(profile.command, ["physical-detector", "show", "standard"])
    assert_clean_result(human)
    assert "counting observation: yes, 5 reporting bins 0-2000 eV" in human.stdout
    assert "response: ideal (default)" in human.stdout
    assert "scalar projection used by sweeps: observation angle 60 deg" in human.stdout
    shown = invoke(profile.command, ["show", "standard"])
    assert "    physical: pixel\n" in shown.stdout
    assert "      observation angle: 60 deg\n" in shown.stdout
    assert "      acquisition: {'exposure_s': 2.0" in shown.stdout


def test_reporting_axis_spellings_replace_each_other(catalog) -> None:
    _create("standard", *ACQUIRE)
    edges = invoke(profile.command, [*SET, "standard", "-y", "--measured-edges-ev", "0,500,3000"])
    assert edges.exit_code == 0
    acquisition = _show()["physical_detector"]["acquisition"]
    assert acquisition["measured_edges_eV"] == [0.0, 500.0, 3000.0]
    assert "measured_bin_width_eV" not in acquisition

    both = invoke(
        profile.command,
        [*SET, "standard", "-y", "--measured-edges-ev", "0,1", "--measured-bin-width-ev", "1"],
    )
    assert both.exit_code == 2
    assert "not both" in both.stderr
    unsorted = invoke(profile.command, [*SET, "standard", "-y", "--measured-edges-ev", "5,1"])
    assert unsorted.exit_code == 2
    assert "strictly increasing" in unsorted.stderr
    half = invoke(profile.command, [*SET, "standard", "-y", "--measured-bin-width-ev", "5"])
    assert half.exit_code == 2
    assert "go together" in half.stderr


def test_poisson_realization_needs_a_seed_and_expected_drops_it(catalog) -> None:
    _create("standard", *ACQUIRE)
    before = catalog.read_text()
    unseeded = invoke(profile.command, [*SET, "standard", "-y", "--realization", "poisson"])
    assert unseeded.exit_code == 2
    assert "required for poisson mode" in unseeded.stderr
    assert catalog.read_text() == before

    seeded = invoke(
        profile.command,
        [*SET, "standard", "-y", "--realization", "poisson", "--realization-seed", "7"],
    )
    assert seeded.exit_code == 0
    assert _show()["physical_detector"]["acquisition"]["seed"] == 7

    expected = invoke(profile.command, [*SET, "standard", "-y", "--realization", "expected"])
    assert expected.exit_code == 0
    acquisition = _show()["physical_detector"]["acquisition"]
    assert acquisition["mode"] == "expected"
    assert "seed" not in acquisition
    contradictory = invoke(
        profile.command,
        [*SET, "standard", "-y", "--realization", "expected", "--realization-seed", "1"],
    )
    assert contradictory.exit_code == 2


def test_response_kinds_replace_and_timepix_options_need_timepix(catalog) -> None:
    _create()
    stray = invoke(profile.command, [*SET, "standard", "-y", "--timepix-samples", "100"])
    assert stray.exit_code == 2
    assert "require --response timepix3" in stray.stderr

    timepix = invoke(
        profile.command,
        [*SET, "standard", "-y", "--response", "timepix3", "--timepix-samples", "100"],
    )
    assert timepix.exit_code == 0
    merged = invoke(profile.command, [*SET, "standard", "-y", "--timepix-seed", "3"])
    assert merged.exit_code == 0
    assert _show()["physical_detector"]["response"] == {
        "kind": "timepix3",
        "n_mc": 100,
        "seed": 3,
    }

    ideal = invoke(profile.command, [*SET, "standard", "-y", "--response", "ideal"])
    assert ideal.exit_code == 0
    assert _show()["physical_detector"]["response"] == {"kind": "ideal"}


def test_angular_shape_cannot_exceed_the_pixel_grid(catalog) -> None:
    _create()
    before = catalog.read_text()
    result = invoke(profile.command, [*SET, "standard", "-y", "--angular-shape", "9", "2"])
    assert result.exit_code == 2
    assert "cannot exceed physical detector shape" in result.stderr
    assert catalog.read_text() == before


def test_reconstruction_is_set_independently_of_angular_shape(catalog) -> None:
    _create("standard", "--angular-shape", "2", "3")

    result = invoke(profile.command, [*SET, "standard", "-y", "--reconstruction", "bilinear_tile"])

    assert result.exit_code == 0
    assert _show()["physical_detector"]["scorer"] == {
        "reconstruction": "bilinear_tile",
        "angular_shape": [2, 3],
    }
    reshaped = invoke(profile.command, [*SET, "standard", "-y", "--angular-shape", "1", "2"])
    assert reshaped.exit_code == 0
    assert _show()["physical_detector"]["scorer"]["reconstruction"] == "bilinear_tile"
    bad = invoke(profile.command, [*SET, "standard", "-y", "--reconstruction", "cubic"])
    assert bad.exit_code == 2
    assert "'cubic' is not one of 'nearest_tile', 'bilinear_tile'" in bad.stderr


def test_set_on_profile_without_detector_requires_geometry(catalog) -> None:
    _create("standard", *ACQUIRE)
    assert _show("high_energy")["source"] is None

    result = invoke(profile.command, [*SET, "high_energy", "--polar-deg", "45"])

    assert result.exit_code == 2
    assert "creating one requires --distance-mm" in result.stderr
    created = invoke(
        profile.command,
        [*SET, "high_energy", "--distance-mm", "300", "--polar-deg", "45"],
    )
    assert_clean_result(created, stdout="updated physical detector for profile high_energy\n")
    local = _show("high_energy")
    assert local["source"] == "profile"
    assert local["physical_detector"]["polar_deg"] == 45.0
    assert "acquisition" not in local["physical_detector"]
    assert _show()["physical_detector"]["polar_deg"] == 60.0


def test_dry_run_writes_nothing(catalog) -> None:
    before = catalog.read_text()
    result = invoke(profile.command, [*SET, "standard", "--distance-mm", "100", "--dry-run"])
    assert result.exit_code == 0
    assert "+[profiles.standard.physical_detector]" in result.stdout
    assert catalog.read_text() == before


def test_reset_sections_then_the_whole_detector(catalog) -> None:
    _create("standard", *ACQUIRE, "--angular-shape", "2", "2")

    section = invoke(
        profile.command, ["physical-detector", "reset", "standard", "acquisition", "-y"]
    )
    assert_clean_result(section, stdout="removed acquisition from profile standard\n")
    payload = _show()
    assert not payload["counting_observation"]
    assert payload["physical_detector"]["scorer"]["angular_shape"] == [2, 2]

    missing = invoke(profile.command, ["physical-detector", "reset", "standard", "response", "-y"])
    assert missing.exit_code == 2
    assert "has no response section" in missing.stderr

    before = catalog.read_text()
    preview = invoke(profile.command, ["physical-detector", "reset", "standard"])
    assert preview.exit_code == 0
    assert "re-run with -y/--yes" in preview.stdout
    assert catalog.read_text() == before

    whole = invoke(profile.command, ["physical-detector", "reset", "standard", "-y"])
    assert_clean_result(whole, stdout="removed the physical detector from profile standard\n")
    assert _show()["physical_detector"] is None


def test_reset_refuses_a_profile_without_detector(catalog) -> None:
    _create()
    result = invoke(profile.command, ["physical-detector", "reset", "high_energy", "-y"])
    assert result.exit_code == 2
    assert "has no physical detector" in result.stderr
