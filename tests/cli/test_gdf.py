"""GDF catalog parsing and public CLI integration."""

import pytest
from click.testing import CliRunner

from pyrite.cli.commands import beam
from pyrite.cli.commands import scan as scan_cli
from pyrite.materials import MaterialConfigError, load_material_catalog
from pyrite.runs import scan
from tests.materials.test_material_catalog import _minimal_catalog
from tests.montecarlo.test_gdf import write_gdf


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
gdf_repetition_rate_hz = 1e6
"""
    )
    path.write_text(text)
    catalog = load_material_catalog(path)
    fields = catalog.profile_beam("standard")
    assert fields["gdf_path"] == str(tmp_path / "beam.gdf")
    assert fields["gdf_repetition_rate_hz"] == 1e6
    for bad, message in [
        ("\nenergy_spread_frac = 0.1", "analytic"),
        ("\ngdf_time_tolerance_s = -1", "finite"),
    ]:
        path.write_text(text + bad)
        with pytest.raises(MaterialConfigError, match=message):
            load_material_catalog(path)
    path.write_text(text.replace("gdf_repetition_rate_hz = 1e6", ""))
    with pytest.raises(MaterialConfigError, match="repetition"):
        load_material_catalog(path)


def test_cli_overrides_reach_run(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(scan, "run", lambda args: captured.update(vars(args)))
    result = CliRunner().invoke(
        scan_cli.command,
        [
            "standard",
            "-m",
            "hopg",
            "--source",
            "gpt_gdf",
            "--gdf-path",
            str(tmp_path / "beam.gdf"),
            "--gdf-time-s",
            "1e-9",
            "--gdf-time-tolerance-s",
            "1e-15",
            "--gdf-z-origin-m",
            "0.1",
            "--gdf-normalization",
            "gdf_charge",
            "--gdf-repetition-rate-hz",
            "1e6",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["gdf_overrides"] == {
        "source": "gpt_gdf",
        "gdf_path": str(tmp_path / "beam.gdf"),
        "gdf_time_s": 1e-9,
        "gdf_time_tolerance_s": 1e-15,
        "gdf_z_origin_m": 0.1,
        "gdf_normalization": "gdf_charge",
        "gdf_repetition_rate_hz": 1e6,
    }


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--gdf-time-s", "-1"),
        ("--gdf-time-tolerance-s", "nan"),
        ("--gdf-repetition-rate-hz", "0"),
        ("--gdf-z-origin-m", "inf"),
    ],
)
def test_cli_invalid_values(flag, value):
    result = CliRunner().invoke(scan_cli.command, ["standard", flag, value])
    assert result.exit_code == 2


def test_list_times(tmp_path):
    path = write_gdf(tmp_path / "beam.gdf", times=(0.0, 1e-9))
    result = CliRunner().invoke(beam.command, ["gdf-times", str(path)])
    assert result.exit_code == 0, result.output
    assert result.output.count("2 particles") == 2
    result = CliRunner().invoke(beam.command, ["gdf-times", str(tmp_path / "missing")])
    assert result.exit_code == 1
    assert "cannot read" in result.output


def test_real_run_resolution_uses_overrides(tmp_path):
    from argparse import Namespace

    path = write_gdf(tmp_path / "beam.gdf")
    args = Namespace(
        catalog_profile="standard",
        performance_profile=None,
        fidelity="full",
        quick=False,
        gdf_overrides={"source": "gpt_gdf", "gdf_path": str(path), "gdf_z_origin_m": 0.1},
    )
    _, sweep, identity, stem = scan._resolved_run(args, "hopg")
    assert sweep.beam.source == "gpt_gdf"
    assert sweep.beam.transverse_fwhm_x_mm is None
    assert stem != "hopg"
    assert identity["parameter_sha256"]


def test_screen_inspection_and_run(tmp_path, monkeypatch):
    import json

    import numpy as np

    from tests.montecarlo.test_gdf import fields

    arrays = fields()
    arrays["t"] = np.array([1e-9, 2e-9])
    path = write_gdf(tmp_path / "screen.gdf", arrays, times=(0.0,), kind="position")
    args = ["gdf-inspect", str(path), "--screen-position-m", "0"]
    result = CliRunner().invoke(beam.command, args)
    assert result.exit_code == 0, result.output
    assert "--gdf-z-origin-m" in result.output
    assert "physical target" in result.output
    result = CliRunner().invoke(beam.command, args + ["-o", "json"])
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert json.loads(result.output)["payload"]["selected"]["screen_position_m"] == 0
    result = CliRunner().invoke(beam.command, args + ["--time-s", "0"])
    assert result.exit_code == 2
    captured = {}
    monkeypatch.setattr(scan, "run", lambda args: captured.update(vars(args)))
    result = CliRunner().invoke(scan_cli.command, ["standard", "--gdf-screen-position-m", "0"])
    assert result.exit_code == 0, result.output
    assert captured["gdf_overrides"]["gdf_screen_position_m"] == 0


@pytest.mark.parametrize(
    "flag,expected", [("--gdf-shape-only", True), ("--no-gdf-shape-only", False)]
)
def test_shape_only_cli_override(monkeypatch, flag, expected):
    captured = {}
    monkeypatch.setattr(scan, "run", lambda args: captured.update(vars(args)))
    result = CliRunner().invoke(scan_cli.command, ["standard", flag])
    assert result.exit_code == 0, result.output
    assert captured["gdf_overrides"]["gdf_shape_only"] is expected


def test_shape_only_catalog(tmp_path):
    path = tmp_path / "materials.toml"
    text = (
        _minimal_catalog(material_rows="[materials]", profile_extra='beam = "imported"')
        + """
[beams.imported]
source = "gpt_gdf"
gdf_path = "beam.gdf"
gdf_z_origin_m = 0
gdf_shape_only = true
"""
    )
    path.write_text(text)
    assert load_material_catalog(path).profile_beam("standard")["gdf_shape_only"] is True
    path.write_text(text.replace("gdf_shape_only = true", 'gdf_shape_only = "true"'))
    with pytest.raises(MaterialConfigError, match="boolean"):
        load_material_catalog(path)
