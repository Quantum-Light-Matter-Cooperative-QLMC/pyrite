"""Regression tests for profile-owned ``cxr run`` beam configuration."""

import pytest
from click.testing import CliRunner

from cxr_mc import scan


@pytest.mark.parametrize(
    "option",
    (
        "--beam-uvw",
        "--beam-transverse-fwhm-x-mm",
        "--beam-transverse-fwhm-y-mm",
        "--beam-bunch-length-fs",
        "--beam-long-shape",
        "--beam-rep-rate-hz",
        "--beam-bunch-charge-pc",
    ),
)
def test_run_rejects_removed_beam_override_options(option):
    result = CliRunner().invoke(scan.command, ["standard", "-m", "hopg", option, "1"])

    assert result.exit_code == 2
    assert f"No such option '{option}'" in result.output


def test_run_help_keeps_beam_overrides_profile_owned():
    result = CliRunner().invoke(scan.command, ["--help"])

    assert result.exit_code == 0
    assert "PROFILE defaults to standard" in result.output
    assert "--beam-" not in result.output


def test_run_internal_performance_profile_selects_same_catalog_profile(monkeypatch):
    captured = {}

    def capture(args):
        captured["identity"] = scan._resolved_run(args, "hopg")[2]

    monkeypatch.setattr(scan, "run", capture)
    result = CliRunner().invoke(
        scan.command,
        [
            "sub_100keV",
            "-m",
            "hopg",
            "--performance-profile",
            "sub_100keV",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["identity"]["catalog_profile"] == "sub_100keV"


def test_run_internal_performance_profile_rejects_profile_mismatch():
    result = CliRunner().invoke(
        scan.command,
        [
            "sub_100keV",
            "-m",
            "hopg",
            "--performance-profile",
            "standard",
        ],
    )

    assert result.exit_code == 2
    assert "must name the same catalog profile" in result.output


def test_run_perf_flag_defaults_to_full_profile_membership(monkeypatch):
    captured = {}

    def capture(args):
        captured["performance_profile"] = args.performance_profile
        captured["material"] = args.material

    monkeypatch.setattr(scan, "run", capture)
    result = CliRunner().invoke(
        scan.command,
        ["sub_100keV", "-p"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["performance_profile"] == "sub_100keV"
    assert captured["material"] is None


def test_run_perf_flag_with_material_profiles_single_member(monkeypatch):
    captured = {}

    def capture(args):
        captured["performance_profile"] = args.performance_profile
        captured["material"] = args.material

    monkeypatch.setattr(scan, "run", capture)
    result = CliRunner().invoke(
        scan.command,
        ["sub_100keV", "-m", "hopg", "-p"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["performance_profile"] == "sub_100keV"
    assert captured["material"] == "hopg"


def test_run_perf_interval_requires_perf_flag():
    result = CliRunner().invoke(scan.command, ["sub_100keV", "-i", "10"])

    assert result.exit_code == 2
    assert "--perf-interval requires -p/--perf" in result.output
