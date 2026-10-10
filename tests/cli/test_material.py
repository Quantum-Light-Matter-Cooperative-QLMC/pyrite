import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite import cli
from pyrite._catalog_layout import read_text
from pyrite.cli import _catalog_io
from pyrite.cli._deprecations import message
from pyrite.cli.commands import material
from tests.helpers.cli import assert_clean_result, invoke

_CATALOG = """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.standard.overrides.hopg]
thickness_ang = { values = [2000.0] }
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }

[profiles.survey]
materials = ["hopg"]
thickness_ang = { values = [500.0] }
energy_keV = { values = [40.0] }
tilt_deg = { values = [10.0] }
tilt_azim_deg = { values = [45.0] }

[profiles.survey.overrides.hopg]
energy_keV = { values = [60.0] }

[materials.hopg]

[materials.mose2]
"""


def _catalog(tmp_path, monkeypatch):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(_CATALOG)
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", catalog)
    monkeypatch.setattr(_catalog_io, "validate", lambda *_args: None)
    return catalog


def test_show_default_profile_reports_effective_sources(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(material.command, ["show", "hopg"])

    assert_clean_result(result)
    assert "hopg: profile standard" in result.stdout
    assert "thickness: [2000] (overridden)" in result.stdout
    assert "energy: [30] (inherited)" in result.stdout
    assert "profiles this material belongs to" not in result.stdout


def test_show_nonstandard_profile_json(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(material.command, ["show", "hopg", "--profile", "survey", "-o", "json"])

    assert_clean_result(result)
    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.material.show"
    assert document["payload"]["profile"] == "survey"
    rows = {row["name"]: row for row in document["payload"]["ranges"]}
    assert rows["energy"]["values"] == [60.0]
    assert rows["energy"]["source"] == "overridden"
    assert rows["thickness"]["values"] == [500.0]
    assert rows["thickness"]["source"] == "inherited"


def test_retired_material_set_is_unknown_command():
    result = invoke(material.command, ["set", "hopg", "--reset", "all"])

    assert result.exit_code == 2
    assert "No such command 'set'" in result.stderr


def test_unknown_names_report_actionable_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    unknown_material = invoke(material.command, ["show", "hpg"])
    unknown_profile = invoke(material.command, ["show", "hopg", "--profile", "missing"])

    assert unknown_material.exit_code == 1
    assert "unknown material: hpg. Did you mean: hopg?" in unknown_material.stderr
    assert unknown_profile.exit_code == 1
    assert "unknown profile: missing" in unknown_profile.stderr


def test_root_and_group_help_expose_new_ownership_only():
    root = invoke(cli.command, ["--help"])
    profile_help = invoke(cli.command, ["profile", "--help"])
    material_help = invoke(cli.command, ["material", "--help"])

    assert_clean_result(root)
    assert "material" in root.stdout
    assert "\n  blaze " not in root.stdout
    assert "\n  catalog " not in root.stdout
    assert "\n  sweep " not in root.stdout
    assert_clean_result(profile_help)
    assert "\n  members " not in profile_help.stdout
    assert "add-material" not in profile_help.stdout
    assert "remove-material" not in profile_help.stdout
    assert_clean_result(material_help)
    assert "show" in material_help.stdout
    assert "\n  set " not in material_help.stdout
    assert "validate" in material_help.stdout
    assert "blaze" in material_help.stdout
    assert "list" not in material_help.stdout


def test_material_group_lazily_routes_validate_and_blaze():
    validate_help = invoke(cli.command, ["material", "validate", "--help"])
    blaze_help = invoke(cli.command, ["material", "blaze", "--help"])

    assert_clean_result(validate_help)
    assert "Validate bundled material catalog" in validate_help.stdout
    assert_clean_result(blaze_help)
    assert "blazed-crystal MC sweep" in blaze_help.stdout


def test_simulate_formats_result_and_uses_single_scene_api(monkeypatch):
    calls = []
    spatial = SimpleNamespace(
        ray_map=SimpleNamespace(
            tile_index=np.zeros((1, 1), dtype=int),
            solid_angle_sr=np.ones((1, 1)),
            path_length_mm=np.zeros((1, 1, 1)),
        ),
        line=SimpleNamespace(
            intrinsic_by_tile=np.array([[1.0, 2.0]]),
            mu_by_filter_inv_mm=np.array([[0.0, 0.0]]),
        ),
    )
    result = SimpleNamespace(
        energy_eV=np.array([100.0, 200.0]),
        spectrum=np.array([1.0, 2.0]),
        background_energy_eV=np.array([50.0, 100.0]),
        background=np.array([0.1, 0.2]),
        spatial=spatial,
        provenance={
            "observation_identity_digest": "abc",
            "scene": SimpleNamespace(acquisition=None),
        },
    )
    monkeypatch.setattr(_catalog_io, "catalog_text", lambda: ("", object()))
    monkeypatch.setattr(
        material,
        "_simulation_scene",
        lambda *_args: (
            "beam",
            "target",
            "detector",
            ("filter",),
            "scorer",
            "acquisition",
            "numerics",
            "incoherent",
            "pixel_a",
        ),
    )

    import pyrite.api

    def fake_simulate(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(pyrite.api, "simulate", fake_simulate)

    machine = invoke(material.command, ["simulate", "hopg", "-o", "json"])
    assert_clean_result(
        machine,
        stderr=message(
            "material simulate", replacement="pyrite run standard -m hopg --ephemeral -o json"
        )
        + "\n",
    )
    payload = json.loads(machine.stdout)["payload"]
    assert payload["line"]["energy_eV"] == [100.0, 200.0]
    assert payload["pixel_grid"]["filter_count"] == 1
    assert calls[0][0][:3] == ("beam", "target", "detector")
    assert calls[0][1]["filters"] == ("filter",)
    assert calls[0][1]["pixel_scorer"] == "scorer"
    assert calls[0][1]["acquisition"] == "acquisition"
    assert payload["acquisition"] is None
    assert payload["detector_id"] == "pixel_a"

    wide = invoke(
        material.command,
        ["simulate", "hopg", "--profile", "my profile", "--detector", "pixel_a", "-o", "wide"],
    )
    assert wide.exit_code == 0
    assert wide.stderr.splitlines() == [
        message(
            "material simulate",
            replacement="pyrite run 'my profile' -m hopg --ephemeral --detector pixel_a -o wide",
        )
    ]
    assert "material=hopg" in wide.stdout
    assert "detector=pixel_a" in wide.stdout


def test_simulate_json_reports_resolution_errors(monkeypatch):
    monkeypatch.setattr(_catalog_io, "catalog_text", lambda: ("", object()))
    monkeypatch.setattr(
        material,
        "_simulation_scene",
        lambda *_args: (_ for _ in ()).throw(ValueError("physical_detector is required")),
    )

    result = invoke(material.command, ["simulate", "hopg", "-o", "json"])
    assert result.exit_code == 1
    assert result.stderr.count("is deprecated") == 1
    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.material.simulate"
    assert "physical_detector is required" in document["errors"][0]["message"]


def _single_scene_catalog(tmp_path, monkeypatch, *, energies="[30.0]", physical_extra=""):
    from pyrite import DATA_DIR

    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    catalog = data / "materials.toml"
    catalog.write_text(read_text(data / "catalog"))
    with catalog.open("a") as stream:
        stream.write(
            f"""
[profiles.single]
materials = ["hopg"]
thickness_ang = {{ values = [1000.0] }}
energy_keV = {{ values = {energies} }}
tilt_deg = {{ values = [5.0] }}
tilt_azim_deg = {{ values = [95.0] }}
E_grid_line = {{ values = [100.0, 200.0] }}
E_grid_brem = {{ values = [10.0, 20.0] }}
n_electrons = {{ values = [7] }}
n_electrons_brem = {{ values = [3] }}
straggling = true
energy_model = "midpoint"
max_dE_frac = 0.1

[profiles.single.physical_detector]
distance_mm = 400.0
polar_deg = 90.0
shape = [2, 3]
pitch_mm = [0.1, 0.2]
{physical_extra}
[[profiles.single.filters]]
name = "half"
material = "silicon"
thickness_mm = 0.1
size_mm = [2.0, 3.0]
distance_mm = 200.0
polar_deg = 90.0
"""
        )
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", catalog)
    return _catalog_io.catalog_text()[1]


def test_simulation_scene_resolves_real_profile_objects(tmp_path, monkeypatch):
    from pyrite.campaign.model import Beam, Numerics
    from pyrite.instrument import FilterPlate, PlanarDetector

    document = _single_scene_catalog(tmp_path, monkeypatch)
    beam, target, detector, filters, scorer, acquisition, numerics, emission, detector_id = (
        material._simulation_scene(document, "hopg", "single")
    )

    assert isinstance(beam, Beam)
    assert beam.energy_keV == 30.0
    assert target.material == "hopg"
    assert isinstance(detector, PlanarDetector)
    assert detector.pixels.shape == (2, 3)
    assert isinstance(filters[0], FilterPlate)
    assert filters[0].name == "half"
    assert isinstance(numerics, Numerics)
    assert numerics.n_electrons == 7
    assert numerics.n_electrons_brem == 3
    assert numerics.straggling is True
    assert numerics.energy_model == "midpoint"
    assert emission == "incoherent"
    assert detector_id == "physical"
    assert scorer.angular_shape == (1, 1)
    assert acquisition is None


def test_simulation_scene_uses_the_profile_counting_observation(tmp_path, monkeypatch):
    from pyrite.detectors import IdealPhotonCounter

    document = _single_scene_catalog(
        tmp_path,
        monkeypatch,
        physical_extra="""
[profiles.single.physical_detector.scorer]
angular_shape = [2, 3]

[profiles.single.physical_detector.acquisition]
exposure_s = 2.0
measured_edges_eV = [0.0, 100.0, 200.0]
""",
    )
    _beam, _target, detector, _filters, scorer, acquisition, _numerics, _emission, _id = (
        material._simulation_scene(document, "hopg", "single")
    )

    assert scorer.angular_shape == (2, 3)
    assert acquisition.exposure_s == 2.0
    assert acquisition.measured_edges_eV == (0.0, 100.0, 200.0)
    assert isinstance(detector.response, IdealPhotonCounter)
    assert detector.energy_bins.line is not None


def test_simulation_scene_rejects_non_singleton_profile_grid(tmp_path, monkeypatch):
    document = _single_scene_catalog(tmp_path, monkeypatch, energies="[30.0, 40.0]")

    with np.testing.assert_raises_regex(
        ValueError, "requires profile 'single' to resolve one value"
    ):
        material._simulation_scene(document, "hopg", "single")


def test_simulation_artifact_uses_exact_suffixless_path_and_never_overwrites(tmp_path):
    spatial = SimpleNamespace(
        ray_map=SimpleNamespace(
            tile_index=np.zeros((1, 1), dtype=int),
            solid_angle_sr=np.ones((1, 1)),
            path_length_mm=np.zeros((1, 1, 1)),
        ),
        line=SimpleNamespace(
            intrinsic_by_tile=np.array([[1.0, 2.0]]),
            mu_by_filter_inv_mm=np.array([[0.0, 0.0]]),
        ),
    )
    result = SimpleNamespace(
        energy_eV=np.array([100.0, 200.0]),
        spectrum=np.array([1.0, 2.0]),
        background_energy_eV=np.array([50.0, 100.0]),
        background=np.array([0.1, 0.2]),
        spatial=spatial,
    )
    output = Path(tmp_path) / "spatial-output"

    material._write_simulation_artifact(output, result)

    assert output.is_file()
    assert not output.with_suffix(".npz").exists()
    with np.load(output) as archive:
        np.testing.assert_array_equal(archive["line_energy_eV"], [100.0, 200.0])
    with np.testing.assert_raises_regex(ValueError, "output file already exists"):
        material._write_simulation_artifact(output, result)


def _spatial_result(acquisition=None):
    return SimpleNamespace(
        energy_eV=np.array([100.0, 200.0]),
        spectrum=np.array([1.0, 2.0]),
        background_energy_eV=np.array([50.0, 100.0]),
        background=np.array([0.1, 0.2]),
        spatial=SimpleNamespace(
            ray_map=SimpleNamespace(
                tile_index=np.zeros((2, 3), dtype=int),
                solid_angle_sr=np.ones((2, 3)),
                path_length_mm=np.zeros((2, 3, 1)),
            ),
            line=SimpleNamespace(
                intrinsic_by_tile=np.array([[1.0, 2.0]]),
                mu_by_filter_inv_mm=np.array([[0.0, 0.0]]),
            ),
        ),
        provenance={
            "observation_identity_digest": "abc",
            "scene": SimpleNamespace(acquisition=acquisition),
        },
        acquisition_image=lambda: np.ones((2, 3)),
    )


@pytest.mark.parametrize("output_format", ["table", "json", "wide"])
@pytest.mark.parametrize("counting", [False, True])
def test_ephemeral_matches_simulate_and_never_enters_sweep(
    tmp_path, monkeypatch, output_format, counting
):
    import pyrite.api
    from pyrite.runs import scan

    extra = (
        """
[profiles.single.physical_detector.acquisition]
exposure_s = 2.0
measured_edges_eV = [0.0, 100.0, 200.0]
"""
        if counting
        else ""
    )
    _single_scene_catalog(tmp_path, monkeypatch, physical_extra=extra)
    monkeypatch.chdir(tmp_path)
    calls = []

    def simulate(*args, **kwargs):
        calls.append((args, kwargs))
        print("transport diagnostic")
        return _spatial_result(kwargs["acquisition"])

    def forbidden(*args, **kwargs):
        pytest.fail("ephemeral simulation entered checkpoint sweep machinery")

    monkeypatch.setattr(pyrite.api, "simulate", simulate)
    monkeypatch.setattr(scan, "run", forbidden)
    monkeypatch.setattr(scan, "_run_json", forbidden)
    before = set(tmp_path.iterdir())
    legacy = invoke(
        cli.command,
        [
            "material",
            "simulate",
            "hopg",
            "--profile",
            "single",
            "-o",
            output_format,
        ],
    )
    ephemeral = invoke(
        cli.command,
        [
            "run",
            "single",
            "-m",
            "hopg",
            "--ephemeral",
            "-o",
            output_format,
        ],
    )
    assert legacy.exit_code == ephemeral.exit_code == 0
    assert legacy.stdout == ephemeral.stdout
    warning = message(
        "material simulate",
        replacement="pyrite run single -m hopg --ephemeral"
        + ("" if output_format == "table" else f" -o {output_format}"),
    )
    assert legacy.stderr == f"{warning}\n" + ephemeral.stderr
    assert ephemeral.stderr == "transport diagnostic\n"
    assert set(tmp_path.iterdir()) == before
    assert len(calls) == 2
    for old, new in zip(calls[0][0], calls[1][0], strict=True):
        assert repr(old) == repr(new)
    assert repr(calls[0][1]) == repr(calls[1][1])
    assert calls[1][1]["filters"][0].name == "half"
    if output_format == "json":
        envelope = json.loads(ephemeral.stdout)
        assert envelope["schema"] == "cxr.material.simulate"
        assert envelope["schema_version"] == 1
        assert envelope["payload"]["detector_id"] == "physical"
        if counting:
            assert envelope["payload"]["acquisition"]["total_counts"] == 6.0


def test_ephemeral_artifact_parity_and_existing_file_preflight(tmp_path, monkeypatch):
    import pyrite.api

    _single_scene_catalog(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        pyrite.api, "simulate", lambda *a, **k: calls.append(k) or _spatial_result()
    )
    old_path, new_path = tmp_path / "simulate", tmp_path / "ephemeral"
    legacy = invoke(
        cli.command,
        [
            "material",
            "simulate",
            "hopg",
            "--profile",
            "single",
            "--output-file",
            str(old_path),
        ],
    )
    ephemeral = invoke(
        cli.command,
        [
            "run",
            "single",
            "-m",
            "hopg",
            "--ephemeral",
            "--output-file",
            str(new_path),
        ],
    )
    assert legacy.exit_code == ephemeral.exit_code == 0
    with np.load(old_path) as old, np.load(new_path) as new:
        assert old.files == new.files
        for key in old.files:
            np.testing.assert_array_equal(old[key], new[key])
    original = new_path.read_bytes()
    repeated = invoke(
        cli.command,
        [
            "run",
            "single",
            "-m",
            "hopg",
            "--ephemeral",
            "--output-file",
            str(new_path),
            "-o",
            "json",
        ],
    )
    assert repeated.exit_code == 1
    assert "output file already exists" in json.loads(repeated.stdout)["errors"][0]["message"]
    assert len(calls) == 2
    assert new_path.read_bytes() == original


@pytest.mark.parametrize(
    "options",
    [
        ["--checkpoint-dir", "checkpoints"],
        ["--workers", "0"],
        ["--max-minutes", "1"],
        ["--quick"],
        ["--no-cache"],
        ["--recompute"],
        ["--trajectories", "tracks"],
        ["--overwrite-trajectories"],
        ["--preset", "zhai"],
        ["--ne", "20000"],
        ["--remote"],
        ["--wait"],
        ["--detach"],
        ["--dry-run"],
        ["--no-sync"],
        ["--n-families", "3"],
        ["--chunk-minutes", "0"],
    ],
)
def test_ephemeral_rejects_explicit_sweep_options_before_resolution(monkeypatch, options):
    monkeypatch.setattr(
        _catalog_io, "catalog_text", lambda: pytest.fail("resolved invalid invocation")
    )
    result = invoke(
        cli.command,
        [
            "run",
            "single",
            "-m",
            "hopg",
            "--ephemeral",
            "-o",
            "json",
            *options,
        ],
    )
    assert result.exit_code == 2
    assert "does not support" in json.loads(result.stdout)["errors"][0]["message"]
    assert "--ephemeral" in result.stderr


@pytest.mark.parametrize("output_format", ["table", "json"])
def test_ephemeral_requires_explicit_material(output_format):
    result = invoke(cli.command, ["run", "single", "--ephemeral", "-o", output_format])
    assert result.exit_code == 2
    assert "requires -m/--material" in result.stderr
    if output_format == "json":
        assert json.loads(result.stdout)["ok"] is False
    else:
        assert result.stdout == ""


@pytest.mark.parametrize("options", [["--detector", "physical"], ["--output-file", "spatial.npz"]])
def test_ephemeral_selectors_require_ephemeral(options):
    result = invoke(cli.command, ["run", "single", "-m", "hopg", *options])
    assert result.exit_code == 2
    assert "require --ephemeral" in result.stderr


def test_ephemeral_resolution_errors_use_usage_exit(tmp_path, monkeypatch):
    import pyrite.api

    _single_scene_catalog(tmp_path, monkeypatch, energies="[30.0, 40.0]")
    monkeypatch.setattr(pyrite.api, "simulate", lambda *a, **k: pytest.fail("transport started"))
    result = invoke(cli.command, ["run", "single", "-m", "hopg", "--ephemeral", "-o", "json"])
    assert result.exit_code == 2
    assert "run --ephemeral requires profile 'single'" in result.stderr
    assert "singleton" in json.loads(result.stdout)["errors"][0]["message"]


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, OSError])
def test_ephemeral_runtime_failure_has_single_json_envelope(tmp_path, monkeypatch, error_type):
    import pyrite.api

    _single_scene_catalog(tmp_path, monkeypatch)

    def failed(*args, **kwargs):
        print("transport diagnostic")
        raise error_type("transport failed")

    monkeypatch.setattr(pyrite.api, "simulate", failed)
    result = invoke(cli.command, ["run", "single", "-m", "hopg", "--ephemeral", "-o", "json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["errors"][0]["message"] == "transport failed"
    assert result.stderr == "transport diagnostic\n"


@pytest.mark.parametrize(
    "selection,expected",
    [
        ([], None),
        (["--detector", "a"], "a"),
        (["--detector", "b"], "b"),
        (["--detector", "scalar"], None),
        (["--detector", "missing"], None),
    ],
)
def test_ephemeral_selects_one_of_multiple_pixel_detectors(
    tmp_path, monkeypatch, selection, expected
):
    import pyrite.api

    _single_scene_catalog(tmp_path, monkeypatch)
    path = _catalog_io.active_catalog_path()
    text = path.read_text().replace(
        "[profiles.single.physical_detector]", "[profiles.single.detectors.a]"
    )
    path.write_text(
        text
        + """
[profiles.single.detectors.b]
distance_mm = 500.0
shape = [2, 3]
pitch_mm = [0.1, 0.2]
[profiles.single.detectors.scalar]
observation_angle_deg = 90.0
"""
    )
    calls = []
    monkeypatch.setattr(
        pyrite.api, "simulate", lambda *a, **k: calls.append(a) or _spatial_result()
    )
    result = invoke(
        cli.command,
        [
            "run",
            "single",
            "-m",
            "hopg",
            "--ephemeral",
            "-o",
            "json",
            *selection,
        ],
    )
    document = json.loads(result.stdout)
    if expected is None:
        assert result.exit_code == 2
        assert len(calls) == 0
        assert "a, b" in document["errors"][0]["message"]
    else:
        assert result.exit_code == 0
        assert document["payload"]["detector_id"] == expected
        assert len(calls) == 1


def test_ephemeral_rejects_profile_without_pixel_detector(tmp_path, monkeypatch):
    _single_scene_catalog(tmp_path, monkeypatch)
    path = _catalog_io.active_catalog_path()
    path.write_text(
        path.read_text().replace(
            "[profiles.single.physical_detector]\ndistance_mm = 400.0\npolar_deg = 90.0\nshape = [2, 3]\npitch_mm = [0.1, 0.2]",
            "[profiles.single.detector]\nobservation_angle_deg = 90.0",
        )
    )
    result = invoke(cli.command, ["run", "single", "-m", "hopg", "--ephemeral", "-o", "json"])
    assert result.exit_code == 2
    assert "available pixel detectors: none" in json.loads(result.stdout)["errors"][0]["message"]


def test_single_scene_and_checkpoint_lowering_share_seed_counts_and_explicit_grids(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from pyrite import api
    from pyrite.campaign import config
    from pyrite.campaign.model import Scene
    from pyrite.materials import load_material_catalog

    document = _single_scene_catalog(tmp_path, monkeypatch)
    path = _catalog_io.active_catalog_path()
    path.write_text(
        path.read_text().replace(
            "n_electrons = { values = [7] }", "n_electrons = { values = [200] }"
        )
    )
    catalog = load_material_catalog(path, profile="single")
    monkeypatch.setattr(config, "_catalog", lambda *_: catalog)
    beam, target, detector, filters, scorer, acquisition, numerics, emission, _id = (
        material._simulation_scene(document, "hopg", "single")
    )
    scene = Scene(beam, target, detector, filters=filters, pixel_scorer=scorer, emission=emission)
    single = api.build_case(scene, numerics)
    settings = replace(
        config.default_settings(),
        straggling=numerics.straggling,
        energy_model=numerics.energy_model,
        max_dE_frac=numerics.max_dE_frac,
    )
    checkpoint = api.build_configured_cases(
        config.material_sweep("hopg", catalog_profile="single", detector_id="physical"), settings
    )[0]
    assert single["Ne"] == checkpoint["Ne"] == 200
    assert single["Ne_brem"] == checkpoint["Ne_brem"] == 3
    assert single["seed"] == checkpoint["seed"]
    for key in ("E_grid", "E_grid_brem"):
        np.testing.assert_array_equal(single[key], checkpoint[key])
    for key in ("E0_keV", "thickness_ang", "tilt_deg", "tilt_azim_deg", "theta_obs_rad"):
        assert single[key] == checkpoint[key]


def test_ephemeral_and_simulate_real_api_have_identical_cases_and_spatial_output(
    tmp_path, monkeypatch
):
    from pyrite import api
    from pyrite.observations import plan

    _single_scene_catalog(tmp_path, monkeypatch)
    cases = []

    def directional(case, n_hats, *, transport_core):
        cases.append(case)
        return {
            "E_grid": np.array([100.0, 200.0]),
            "E_grid_brem": np.array([10.0, 20.0]),
            "spec_by_direction": np.ones((len(n_hats), 2)),
            "spec_characteristic_by_direction": np.zeros((len(n_hats), 2)),
            "brem_wide_by_direction": np.full((len(n_hats), 2), 0.5),
        }

    monkeypatch.setattr(api, "run_case_directions", directional)
    monkeypatch.setattr(api, "case_table_markers", lambda case: {})
    # Deterministic attenuation fixture keeps this CLI/API integration check
    # independent of fetched EPDL tables while retaining finite-filter geometry.
    monkeypatch.setattr(
        plan, "attenuation_matrix", lambda filters, energy: np.ones((len(filters), len(energy)))
    )
    legacy = invoke(
        cli.command, ["material", "simulate", "hopg", "--profile", "single", "-o", "json"]
    )
    ephemeral = invoke(cli.command, ["run", "single", "-m", "hopg", "--ephemeral", "-o", "json"])
    assert legacy.exit_code == ephemeral.exit_code == 0, (legacy.output, ephemeral.output)
    assert legacy.stdout == ephemeral.stdout
    assert cases[0] == cases[1]


def test_single_scene_named_gdf_clears_default_spot(tmp_path, monkeypatch):
    import tomlkit

    from tests.montecarlo.test_gdf import write_gdf

    document = _single_scene_catalog(tmp_path, monkeypatch)
    path = write_gdf(tmp_path / "beam.gdf")
    document["beams"]["gpt_import"] = {
        "source": "gpt_gdf",
        "gdf_path": str(path),
        "gdf_z_origin_m": 0.1,
        "gdf_normalization": "gdf_charge",
        "rep_rate_hz": 3000,
    }
    document["profiles"]["single"]["beam"] = "gpt_import"
    _catalog_io.atomic_write(_catalog_io.active_catalog_path(), tomlkit.dumps(document))
    beam, *_ = material._simulation_scene(document, "hopg", "single")
    assert beam.source == "gpt_gdf"
    assert beam.transverse_fwhm_x_mm is None
    assert beam.transverse_fwhm_y_mm is None
    assert beam.rep_rate_hz == 3000
    assert beam.gdf_beam().absolute_charge_c > 0
