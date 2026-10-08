"""GDF catalog parsing and named-beam CLI integration."""

import json
from argparse import Namespace

import numpy as np
import pytest
import tomlkit
from click.testing import CliRunner

from pyrite.campaign import config
from pyrite.campaign.sweep import build_cases
from pyrite.cli import _catalog_io
from pyrite.cli.commands import beam, profile
from pyrite.cli.commands import scan as scan_cli
from pyrite.materials import MaterialConfigError, load_material_catalog
from pyrite.runs import scan
from tests.materials.test_material_catalog import _minimal_catalog
from tests.montecarlo.test_gdf import fields, write_gdf


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    path = tmp_path / "materials.toml"
    path.write_text(_catalog_io.current_text())
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", path)
    monkeypatch.chdir(tmp_path)
    return path


def invoke(*args):
    return CliRunner().invoke(beam.command, list(args))


def create(path, *extra):
    return invoke(
        "create",
        "imported",
        "--source",
        "gpt_gdf",
        "--gdf-path",
        str(path),
        "--gdf-z-origin-m",
        "0.1",
        *extra,
    )


def test_catalog_path_and_source(tmp_path):
    path = tmp_path / "materials.toml"
    text = (
        _minimal_catalog(material_rows="[materials]", profile_extra='beam = "imported"')
        + """
[beams.imported]
source = "gpt_gdf"
gdf_path = "beam.gdf"
gdf_time_s = 1e-9
gdf_z_origin_m = 0.1
gdf_normalization = "gdf_charge"
rep_rate_hz = 1e6
"""
    )
    path.write_text(text)
    resolved = load_material_catalog(path).profile_beam("standard")
    assert resolved["gdf_path"] == str(tmp_path / "beam.gdf")
    assert resolved["rep_rate_hz"] == 1e6
    for bad, message in [
        ("\nenergy_spread_frac = 0.1", "analytic"),
        ("\ngdf_time_tolerance_s = -1", "finite"),
        ("\ngdf_repetition_rate_hz = 1e6", "replace gdf_repetition_rate_hz with rep_rate_hz"),
    ]:
        path.write_text(text + bad)
        with pytest.raises(MaterialConfigError, match=message):
            load_material_catalog(path)
    path.write_text(text.replace("rep_rate_hz = 1e6", ""))
    assert "rep_rate_hz" not in load_material_catalog(path).profile_beam("standard")


@pytest.mark.parametrize("normalization", ["gdf_charge", "pyrite_current"])
def test_named_beam_workflow(catalog, monkeypatch, normalization):
    path = write_gdf(catalog.parent / "beam.gdf")
    result = create("beam.gdf", "--gdf-normalization", normalization, "--rep-rate-hz", "1000")
    assert result.exit_code == 0, result.output
    row = tomlkit.parse(catalog.read_text())["beams"]["imported"]
    assert row["gdf_path"] == str(path)
    assert row["rep_rate_hz"] == 1000
    shown = invoke("show", "imported")
    assert shown.exit_code == 0, shown.output
    assert "source: gpt_gdf" in shown.output
    assert f"gdf_path: {path}" in shown.output
    machine = invoke("show", "imported", "-o", "json")
    assert json.loads(machine.output)["payload"]["gdf_normalization"] == normalization
    attached = CliRunner().invoke(profile.command, ["set", "standard", "--beam", "imported", "-y"])
    assert attached.exit_code == 0, attached.output
    resolved_catalog = load_material_catalog(catalog)
    monkeypatch.setattr(config, "CATALOG", resolved_catalog)
    import pyrite.materials

    monkeypatch.setattr(pyrite.materials, "CATALOG", resolved_catalog)
    captured = {}

    def run(args):
        settings, sweep, identity, stem = scan._resolved_run(args, "hopg")
        captured.update(sweep=sweep, identity=identity, stem=stem)
        assert not hasattr(args, "gdf_overrides")

    monkeypatch.setattr(scan, "run", run)
    result = CliRunner().invoke(scan_cli.command, ["standard", "-m", "hopg"])
    assert result.exit_code == 0, result.output
    sweep = captured["sweep"]
    assert sweep.beam.source == "gpt_gdf"
    assert sweep.beam.transverse_fwhm_x_mm is None
    assert captured["identity"]["resolved_parameters"]["sweep"]["gdf_source"]["sha256"]
    cases = build_cases(sweep, n_electrons=2, n_electrons_brem=2)
    assert len({case["E0_keV"] for case in cases}) == 1
    assert cases[0]["rep_rate_hz"] == 1000
    assert cases[0]["gdf_source"]["sha256"]
    # Execution revalidates the external file even though the catalog is unchanged.
    path.write_bytes(b"broken")
    with pytest.raises(MaterialConfigError, match="cannot read"):
        scan._resolved_run(
            Namespace(
                catalog_profile="standard", performance_profile=None, fidelity="full", quick=False
            ),
            "hopg",
        )


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--gdf-time-s", "1e-9", "--gdf-screen-position-m", "0"], "mutually exclusive"),
        (["--transverse-fwhm-mm", "1"], "analytic"),
        (["--energy-spread", "0.1"], "analytic"),
        (["--longitudinal", "gaussian", "--envelope-rms-fs", "10"], "analytic"),
        (["--emittance", "1", "--twiss-beta", "1"], "analytic"),
    ],
)
def test_conflicts_do_not_mutate(catalog, extra, message):
    write_gdf(catalog.parent / "beam.gdf")
    original = catalog.read_bytes()
    result = create("beam.gdf", *extra)
    assert result.exit_code != 0
    assert message in result.output
    assert catalog.read_bytes() == original


def test_invalid_files_and_selection_do_not_mutate(catalog):
    original = catalog.read_bytes()
    for path in ("missing.gdf", "broken.gdf"):
        (catalog.parent / "broken.gdf").write_bytes(b"bad")
        result = create(path)
        assert result.exit_code == 1, result.output
        assert catalog.read_bytes() == original
    write_gdf(catalog.parent / "beam.gdf", times=(0, 1e-9))
    result = create("beam.gdf")
    assert result.exit_code == 1
    assert "gdf_time_s is required" in result.output
    result = invoke("create", "imported", "--source", "gpt_gdf", "--gdf-path", "beam.gdf")
    assert result.exit_code != 0
    assert "gdf_z_origin_m" in result.output
    assert catalog.read_bytes() == original


def test_edit_source_and_selectors(catalog):
    arrays = fields()
    arrays["t"] = np.array([1e-9, 2e-9])
    path = write_gdf(catalog.parent / "screen.gdf", arrays, times=(0,), kind="position")
    assert invoke("create", "imported", "--transverse-fwhm-mm", "2").exit_code == 0
    result = invoke(
        "set",
        "imported",
        "--source",
        "gpt_gdf",
        "--gdf-path",
        str(path),
        "--gdf-screen-position-m",
        "0",
        "--gdf-z-origin-m",
        "0.1",
        "-y",
    )
    assert result.exit_code == 0, result.output
    row = tomlkit.parse(catalog.read_text())["beams"]["imported"]
    assert "transverse_fwhm_mm" not in row
    assert row["gdf_screen_position_m"] == 0
    original = catalog.read_bytes()
    invalid = invoke("set", "imported", "--energy-spread", "0.1", "-y")
    assert invalid.exit_code == 2
    assert catalog.read_bytes() == original
    path = write_gdf(catalog.parent / "beam.gdf")
    result = invoke("set", "imported", "--gdf-path", str(path), "--gdf-time-s", "1e-9", "-y")
    assert result.exit_code == 0, result.output
    row = tomlkit.parse(catalog.read_text())["beams"]["imported"]
    assert "gdf_screen_position_m" not in row
    result = invoke("set", "imported", "--source", "analytic", "--transverse-fwhm-mm", "1", "-y")
    assert result.exit_code == 0, result.output
    row = tomlkit.parse(catalog.read_text())["beams"]["imported"]
    assert not any(key.startswith("gdf_") for key in row)
    assert row["transverse_fwhm_mm"] == 1


def test_dry_run_and_invalid_edit(catalog):
    write_gdf(catalog.parent / "beam.gdf")
    original = catalog.read_bytes()
    result = create("beam.gdf", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "+[beams.imported]" in result.output
    assert catalog.read_bytes() == original
    assert create("beam.gdf").exit_code == 0
    original = catalog.read_bytes()
    result = invoke("set", "imported", "--rep-rate-hz", "2000", "--dry-run")
    assert result.exit_code == 0, result.output
    assert catalog.read_bytes() == original
    result = invoke("set", "imported", "--gdf-path", "missing.gdf", "-y")
    assert result.exit_code == 1
    assert catalog.read_bytes() == original


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--gdf-time-s", "-1"),
        ("--gdf-time-tolerance-s", "nan"),
        ("--rep-rate-hz", "0"),
        ("--rep-rate-hz", "inf"),
        ("--gdf-z-origin-m", "inf"),
        ("--gdf-screen-tolerance-m", "-1"),
    ],
)
def test_cli_invalid_values(flag, value):
    assert invoke("create", "imported", flag, value).exit_code == 2


@pytest.mark.parametrize(
    "flag",
    [
        "--source",
        "--gdf-path",
        "--gdf-time-s",
        "--gdf-time-tolerance-s",
        "--gdf-screen-position-m",
        "--gdf-screen-tolerance-m",
        "--gdf-z-origin-m",
        "--gdf-normalization",
        "--gdf-repetition-rate-hz",
        "--gdf-shape-only",
        "--no-gdf-shape-only",
    ],
)
def test_removed_run_options(flag):
    result = CliRunner().invoke(scan_cli.command, ["standard", flag])
    assert result.exit_code == 2
    assert "No such option" in result.output


@pytest.mark.parametrize(
    "flag,expected", [("--gdf-shape-only", True), ("--no-gdf-shape-only", False)]
)
def test_shape_only_named_beam(catalog, flag, expected):
    write_gdf(catalog.parent / "beam.gdf")
    result = create("beam.gdf", flag)
    assert result.exit_code == 0, result.output
    assert tomlkit.parse(catalog.read_text())["beams"]["imported"]["gdf_shape_only"] is expected
    result = invoke(
        "set", "imported", "--no-gdf-shape-only" if expected else "--gdf-shape-only", "-y"
    )
    assert result.exit_code == 0, result.output
    assert tomlkit.parse(catalog.read_text())["beams"]["imported"]["gdf_shape_only"] is not expected


def test_list_times(tmp_path):
    path = write_gdf(tmp_path / "beam.gdf", times=(0, 1e-9))
    result = invoke("gdf-times", str(path))
    assert result.exit_code == 0, result.output
    assert result.output.count("2 particles") == 2
    assert invoke("gdf-times", str(tmp_path / "missing")).exit_code == 1


def test_screen_inspection(tmp_path):
    arrays = fields()
    arrays["t"] = np.array([1e-9, 2e-9])
    path = write_gdf(tmp_path / "screen.gdf", arrays, times=(0,), kind="position")
    args = ("gdf-inspect", str(path), "--screen-position-m", "0")
    result = invoke(*args)
    assert result.exit_code == 0, result.output
    assert "--gdf-z-origin-m" in result.output
    assert "physical target" in result.output
    result = invoke(*args, "-o", "json")
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert json.loads(result.output)["payload"]["selected"]["screen_position_m"] == 0
    assert invoke(*args, "--time-s", "0").exit_code == 2


@pytest.mark.parametrize("rate", [None, 3000])
def test_default_and_inherited_shared_rate(catalog, monkeypatch, rate):
    write_gdf(catalog.parent / "beam.gdf")
    extra = () if rate is None else ("--rep-rate-hz", str(rate))
    result = create("beam.gdf", "--gdf-normalization", "gdf_charge", *extra)
    assert result.exit_code == 0, result.output
    attached = CliRunner().invoke(profile.command, ["set", "standard", "--beam", "imported", "-y"])
    assert attached.exit_code == 0, attached.output
    resolved = load_material_catalog(catalog)
    monkeypatch.setattr(config, "CATALOG", resolved)
    sweep = config.material_sweep("hopg")
    assert sweep.beam.rep_rate_hz == (5000 if rate is None else rate)
    assert build_cases(sweep)[0]["rep_rate_hz"] == sweep.beam.rep_rate_hz


def test_named_beam_rename_and_label_preserve_identity(catalog, monkeypatch):
    from pyrite.campaign.profiles import dataset_identity

    write_gdf(catalog.parent / "beam.gdf")
    assert create("beam.gdf", "--gdf-normalization", "gdf_charge").exit_code == 0
    assert (
        CliRunner()
        .invoke(profile.command, ["set", "standard", "--beam", "imported", "-y"])
        .exit_code
        == 0
    )
    monkeypatch.setattr(config, "CATALOG", load_material_catalog(catalog))
    before = dataset_identity(
        "hopg", "full", config.default_settings(), config.material_sweep("hopg")
    )
    assert invoke("set", "imported", "--label", "My lab beam", "-y").exit_code == 0
    assert invoke("rename", "imported", "renamed").exit_code == 0
    monkeypatch.setattr(config, "CATALOG", load_material_catalog(catalog))
    after = dataset_identity(
        "hopg", "full", config.default_settings(), config.material_sweep("hopg")
    )
    assert before["parameter_sha256"] == after["parameter_sha256"]


def test_dependent_field_removal_requires_confirmation(catalog):
    write_gdf(catalog.parent / "beam.gdf")
    assert invoke("create", "imported", "--transverse-fwhm-mm", "2").exit_code == 0
    original = catalog.read_bytes()
    result = invoke(
        "set",
        "imported",
        "--source",
        "gpt_gdf",
        "--gdf-path",
        "beam.gdf",
        "--gdf-z-origin-m",
        "0.1",
    )
    assert result.exit_code != 0
    assert "transverse_fwhm_mm" in result.output
    assert catalog.read_bytes() == original


@pytest.mark.parametrize(
    "command,options",
    [(scan_cli.command, ["--remote"]), (scan_cli.performance_command, ["--nsys"])],
)
def test_gdf_execution_path_restrictions(catalog, monkeypatch, command, options):
    write_gdf(catalog.parent / "beam.gdf")
    assert create("beam.gdf").exit_code == 0
    assert (
        CliRunner()
        .invoke(profile.command, ["set", "standard", "--beam", "imported", "-y"])
        .exit_code
        == 0
    )
    monkeypatch.setattr(config, "CATALOG", load_material_catalog(catalog))
    result = CliRunner().invoke(command, ["standard", *options])
    assert result.exit_code == 2, result.output
    assert "gpt_gdf beams require a local run" in result.output


def test_ambiguous_selection_and_invalid_particles_fail_before_write(catalog):
    path = write_gdf(catalog.parent / "beam.gdf", times=(0, 1e-9))
    original = catalog.read_bytes()
    result = create(path, "--gdf-time-s", "0", "--gdf-time-tolerance-s", "2e-9")
    assert result.exit_code == 1, result.output
    assert "ambiguous" in result.output
    assert catalog.read_bytes() == original
    arrays = fields()
    arrays["q"] *= -1
    write_gdf(path, arrays)
    result = create(path)
    assert result.exit_code == 1, result.output
    assert "q" in result.output
    assert catalog.read_bytes() == original


def test_shape_only_catalog_rejects_nonboolean(tmp_path):
    path = tmp_path / "materials.toml"
    path.write_text(
        _minimal_catalog(material_rows="[materials]", profile_extra='beam = "imported"')
        + """
[beams.imported]
source = "gpt_gdf"
gdf_path = "beam.gdf"
gdf_z_origin_m = 0
gdf_shape_only = "true"
"""
    )
    with pytest.raises(MaterialConfigError, match="boolean"):
        load_material_catalog(path)


def test_profile_run_persists_imported_normalization(catalog, monkeypatch):
    """Exercise real CLI/run/checkpoint plumbing with CPU rays and fixed yields."""
    from scipy.constants import elementary_charge

    import pyrite.materials
    from pyrite.checkpoints.persistence import load_checkpoint
    from pyrite.montecarlo.transport import simulate_trajectories
    from pyrite.runs import run as run_driver

    path = write_gdf(catalog.parent / "beam.gdf")
    assert create(path, "--gdf-normalization", "gdf_charge", "--rep-rate-hz", "3000").exit_code == 0
    document = tomlkit.parse(catalog.read_text())
    row = document["profiles"]["standard"]
    row["beam"] = "imported"
    row.pop("overrides", None)
    for key, values in {
        "energy_keV": [30],
        "thickness_ang": [10],
        "tilt_deg": [10],
        "tilt_azim_deg": [0],
        "n_electrons": [2],
        "n_electrons_brem": [2],
    }.items():
        row[key] = {"values": values}
    catalog.write_text(tomlkit.dumps(document))
    resolved = load_material_catalog(catalog)
    monkeypatch.setattr(config, "CATALOG", resolved)
    monkeypatch.setattr(pyrite.materials, "CATALOG", resolved)
    transported = []

    def run_cpu_rays(cases, *, callback, **kwargs):
        # Fixed source yields isolate normalization from optional radiation tables.
        for index, case in enumerate(cases):
            rays = simulate_trajectories(
                case["E0_keV"],
                case["Ne"],
                case["thickness_ang"],
                element="C",
                n_atoms_per_ang3=0.1,
                elastic_model="sr",
                seed=case["seed"],
                gdf_source=case["gdf_source"],
                transport_core="lockstep",
            )
            transported.append(rays)
            grid = np.arange(50.0, 151.0)
            callback(
                index,
                case,
                dict(
                    E_grid=grid,
                    spec=np.exp(-0.5 * ((grid - 100) / 3) ** 2),
                    brem=np.full_like(grid, 0.1),
                    eta=1.0,
                ),
            )

    monkeypatch.setattr(run_driver, "run_cases", run_cpu_rays)
    checkpoint_dir = catalog.parent / "checkpoints"
    result = CliRunner().invoke(
        scan_cli.command,
        [
            "standard",
            "-m",
            "hopg",
            "--workers",
            "0",
            "--no-cache",
            "--no-progress",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(transported) == 1
    assert (checkpoint_dir / "hopg" / "line.h5").is_file()
    stored = load_checkpoint("hopg", checkpoint_dir)
    record = next(iter(next(iter(stored.values())).values()))
    assert record["source_current_na"] == pytest.approx(10 * elementary_charge * 3000 * 1e9)
    assert record["case"]["rep_rate_hz"] == 3000
    assert record["case"]["gdf_source"]["sha256"]
