"""Regression tests for profile-owned ``cxr run`` beam configuration."""

import os

import pytest
from click.testing import CliRunner

from cxr_mc.runs import scan


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
    assert "PROFILE defaults to the current configured profile" in result.output
    assert "standard built-in" in result.output
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


@pytest.mark.parametrize("option", ("--spec-chunk", "--brem-chunk"))
def test_run_chunk_pins_require_perf(option):
    result = CliRunner().invoke(scan.command, ["sub_100keV", option, "128"])

    assert result.exit_code == 2
    assert "--spec-chunk/--brem-chunk require -p/--perf" in result.output


def test_run_nsys_requires_perf():
    result = CliRunner().invoke(scan.command, ["sub_100keV", "-m", "hopg", "--nsys"])

    assert result.exit_code == 2
    assert "--nsys requires -p/--perf" in result.output


def test_run_nsys_defaults_to_full_profile_membership(monkeypatch):
    captured = {}

    def fake_reexec(**kwargs):
        captured.update(kwargs)

    def fail_run(args):  # pragma: no cover - must not be reached
        raise AssertionError("run() must not execute in-process under --nsys")

    monkeypatch.setattr(scan, "_reexec_under_nsys", fake_reexec)
    monkeypatch.setattr(scan, "run", fail_run)
    result = CliRunner().invoke(
        scan.command,
        ["sub_100keV", "-p", "--nsys"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["material"] is None  # full membership, no -m required
    assert captured["performance_profile"] == "sub_100keV"


def test_run_chunk_pins_exported_before_runtime_import(monkeypatch):
    seen = {}

    def capture(args):
        seen["spec"] = os.environ.get("CXR_MC_SPEC_CHUNK")
        seen["brem"] = os.environ.get("CXR_MC_BREM_CHUNK")

    monkeypatch.setattr(scan, "run", capture)
    monkeypatch.delenv("CXR_MC_SPEC_CHUNK", raising=False)
    monkeypatch.delenv("CXR_MC_BREM_CHUNK", raising=False)
    result = CliRunner().invoke(
        scan.command,
        ["sub_100keV", "-p", "--spec-chunk", "128", "--brem-chunk", "64"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert seen["spec"] == "128"
    assert seen["brem"] == "64"


def test_run_nsys_reexecs_instead_of_running_in_process(monkeypatch):
    captured = {}

    def fake_reexec(**kwargs):
        captured.update(kwargs)

    def fail_run(args):  # pragma: no cover - must not be reached
        raise AssertionError("run() must not execute in-process under --nsys")

    monkeypatch.setattr(scan, "_reexec_under_nsys", fake_reexec)
    monkeypatch.setattr(scan, "run", fail_run)
    result = CliRunner().invoke(
        scan.command,
        ["sub_100keV", "-m", "hopg", "-p", "--nsys"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["material"] == "hopg"
    assert captured["performance_profile"] == "sub_100keV"


def test_nsys_reexec_command_builds_launcher_and_uncached_checkpoint():
    argv, trace_base = scan._nsys_reexec_command(
        catalog_profile="sub_100keV",
        material="hopg",
        performance_profile="sub_100keV",
        performance_dir=None,
        performance_interval=2.0,
        workers=0,
        fidelity="survey",
        quick=False,
        n_families=None,
    )

    assert argv[0] == "nsys" and argv[1] == "profile"
    assert "--output=performance-profiles/sub_100keV/hopg" in argv
    assert "-m" in argv and "cxr_mc._entry.scan" in argv
    assert "--nsys" not in argv  # child must not recurse
    assert "--performance-profile" in argv and "sub_100keV" in argv
    assert "--perf-interval" in argv and "2" in argv
    # isolated, always-uncached checkpoint dir so the trace covers real work
    assert "--checkpoint-dir" in argv
    assert "performance-profiles/sub_100keV/nsys-checkpoints/hopg" in argv
    assert str(trace_base) == "performance-profiles/sub_100keV/hopg"


def test_nsys_reexec_command_full_membership_uses_profile_stem():
    argv, trace_base = scan._nsys_reexec_command(
        catalog_profile="sub_100keV",
        material=None,
        performance_profile="sub_100keV",
        performance_dir=None,
        performance_interval=5.0,
        workers=None,
        fidelity="full",
        quick=False,
        n_families=None,
    )

    # full membership: only the `python -m cxr_mc._entry.scan` flag, no `-m <mat>`
    assert argv.count("-m") == 1
    assert "--output=performance-profiles/sub_100keV/sub_100keV" in argv
    assert "performance-profiles/sub_100keV/nsys-checkpoints/sub_100keV" in argv
    assert str(trace_base) == "performance-profiles/sub_100keV/sub_100keV"


def test_nsys_reexec_forwards_explicit_recompute():
    argv, _ = scan._nsys_reexec_command(
        catalog_profile="sub_100keV",
        material="hopg",
        performance_profile="sub_100keV",
        performance_dir=None,
        performance_interval=5.0,
        workers=None,
        fidelity="full",
        quick=False,
        n_families=None,
        recompute=True,
    )

    assert "--recompute" in argv


def test_reexec_under_nsys_errors_when_nsys_missing(monkeypatch):
    monkeypatch.setattr(scan.shutil, "which", lambda _name: None)
    with pytest.raises(scan.click.UsageError, match="nsys executable is not on PATH"):
        scan._reexec_under_nsys(
            catalog_profile="sub_100keV",
            material="hopg",
            performance_profile="sub_100keV",
            performance_dir=None,
            performance_interval=5.0,
            workers=None,
            fidelity="full",
            quick=False,
            n_families=None,
        )
