"""Regression tests for profile-owned ``pyrite run`` beam configuration."""

import os

import pytest
from click.testing import CliRunner

from pyrite.cli.commands import scan as scan_cli
from pyrite.cli.commands.scan import performance_command
from pyrite.runs import scan


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
    result = CliRunner().invoke(scan_cli.command, ["standard", "-m", "hopg", option, "1"])

    assert result.exit_code == 2
    assert f"No such option '{option}'" in result.output


def test_run_help_keeps_beam_overrides_profile_owned():
    result = CliRunner().invoke(scan_cli.command, ["--help"])

    assert result.exit_code == 0
    assert "PROFILE defaults to PYRITE_PROFILE or the configured profile.current" in " ".join(
        result.output.split()
    )
    assert "--beam-" not in result.output


@pytest.mark.parametrize(
    "option",
    (
        "--perf",
        "--performance-profile",
        "--performance-dir",
        "--perf-interval",
        "--spec-chunk",
        "--brem-chunk",
        "--nsys",
        "--cpu",
        "--cpu-only",
    ),
)
def test_run_rejects_performance_options(option):
    result = CliRunner().invoke(scan_cli.command, ["sub_100keV", option])

    assert result.exit_code == 2
    assert f"No such option '{option}'" in result.output


def test_perf_defaults_to_full_profile_membership(monkeypatch):
    captured = {}

    def capture(args):
        captured["performance_profile"] = args.performance_profile
        captured["material"] = args.material

    monkeypatch.setattr(scan, "run", capture)
    result = CliRunner().invoke(
        performance_command,
        ["sub_100keV"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["performance_profile"] == "sub_100keV"
    assert captured["material"] is None


def test_perf_with_material_profiles_single_member(monkeypatch):
    captured = {}

    def capture(args):
        captured["performance_profile"] = args.performance_profile
        captured["material"] = args.material

    monkeypatch.setattr(scan, "run", capture)
    result = CliRunner().invoke(
        performance_command,
        ["sub_100keV", "-m", "hopg"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["performance_profile"] == "sub_100keV"
    assert captured["material"] == "hopg"


def test_perf_nsys_defaults_to_full_profile_membership(monkeypatch):
    captured = {}

    def fake_reexec(**kwargs):
        captured.update(kwargs)

    def fail_run(args):  # pragma: no cover - must not be reached
        raise AssertionError("run() must not execute in-process under --nsys")

    monkeypatch.setattr(scan, "_reexec_under_nsys", fake_reexec)
    monkeypatch.setattr(scan, "run", fail_run)
    result = CliRunner().invoke(
        performance_command,
        ["sub_100keV", "--nsys"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["material"] is None  # full membership, no -m required
    assert captured["performance_profile"] == "sub_100keV"


def test_perf_chunk_pins_exported_before_runtime_import(monkeypatch):
    seen = {}

    def capture(args):
        seen["spec"] = os.environ.get("PYRITE_MC_SPEC_CHUNK")
        seen["brem"] = os.environ.get("PYRITE_MC_BREM_CHUNK")

    monkeypatch.setattr(scan, "run", capture)
    monkeypatch.delenv("PYRITE_MC_SPEC_CHUNK", raising=False)
    monkeypatch.delenv("PYRITE_MC_BREM_CHUNK", raising=False)
    result = CliRunner().invoke(
        performance_command,
        ["sub_100keV", "--spec-chunk", "128", "--brem-chunk", "64"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert seen["spec"] == "128"
    assert seen["brem"] == "64"


def test_perf_nsys_reexecs_instead_of_running_in_process(monkeypatch):
    captured = {}

    def fake_reexec(**kwargs):
        captured.update(kwargs)

    def fail_run(args):  # pragma: no cover - must not be reached
        raise AssertionError("run() must not execute in-process under --nsys")

    monkeypatch.setattr(scan, "_reexec_under_nsys", fake_reexec)
    monkeypatch.setattr(scan, "run", fail_run)
    result = CliRunner().invoke(
        performance_command,
        ["sub_100keV", "-m", "hopg", "--nsys"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["material"] == "hopg"
    assert captured["performance_profile"] == "sub_100keV"


def test_nsys_reexec_command_builds_launcher_and_uncached_checkpoint(tmp_path):
    argv, trace_base = scan._nsys_reexec_command(
        catalog_profile="sub_100keV",
        material="hopg",
        performance_profile="sub_100keV",
        performance_dir=None,
        performance_interval=2.0,
        workers=0,
        quick=False,
        n_families=None,
        max_minutes=1.5,
    )

    assert argv[0] == "nsys" and argv[1] == "profile"
    assert f"--output={tmp_path}/pyrite-output/performance/sub_100keV/hopg" in argv
    assert "-m" in argv and "pyrite._dev" in argv and "perf" in argv
    assert "--nsys" not in argv  # child must not recurse
    assert "--fidelity" not in argv
    assert "--performance-profile" not in argv and "sub_100keV" in argv
    assert "--perf-interval" in argv and "2" in argv
    assert "--max-minutes" in argv and "1.5" in argv
    # isolated, always-uncached checkpoint dir so the trace covers real work
    assert "--checkpoint-dir" in argv
    assert (
        f"{tmp_path}/pyrite-output/checkpoints/performance/sub_100keV/nsys-checkpoints/hopg" in argv
    )
    assert str(trace_base) == f"{tmp_path}/pyrite-output/performance/sub_100keV/hopg"


def test_nsys_reexec_command_full_membership_uses_profile_stem(tmp_path):
    argv, trace_base = scan._nsys_reexec_command(
        catalog_profile="sub_100keV",
        material=None,
        performance_profile="sub_100keV",
        performance_dir=None,
        performance_interval=5.0,
        workers=None,
        quick=False,
        n_families=None,
    )

    # full membership: only the `python -m pyrite._entry.scan` flag, no `-m <mat>`
    assert argv.count("-m") == 1
    assert f"--output={tmp_path}/pyrite-output/performance/sub_100keV/sub_100keV" in argv
    assert (
        f"{tmp_path}/pyrite-output/checkpoints/performance/sub_100keV/nsys-checkpoints/sub_100keV"
        in argv
    )
    assert str(trace_base) == f"{tmp_path}/pyrite-output/performance/sub_100keV/sub_100keV"


def test_nsys_reexec_forwards_explicit_recompute():
    argv, _ = scan._nsys_reexec_command(
        catalog_profile="sub_100keV",
        material="hopg",
        performance_profile="sub_100keV",
        performance_dir=None,
        performance_interval=5.0,
        workers=None,
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
            quick=False,
            n_families=None,
        )


def test_py_spy_reexec_command_samples_an_uncached_child(tmp_path):
    argv, output, status = scan._py_spy_reexec_command(
        ["py-spy"],
        catalog_profile="sub_100keV",
        material="hopg",
        performance_profile="sub_100keV",
        performance_dir=None,
        performance_interval=5.0,
        workers=None,
        quick=False,
        n_families=None,
    )

    separator = argv.index("--")
    assert argv[:2] == ["py-spy", "record"]
    assert {"--subprocesses", "--nonblocking", "--idle"} <= set(argv[:separator])
    assert argv[argv.index("--output") + 1] == str(output)
    assert str(output) == f"{tmp_path}/pyrite-output/performance/sub_100keV/hopg.py-spy.json"
    wrapper = argv[separator + 1 :]
    assert wrapper[1:3] == ["-m", "pyrite.perf.py_spy"]
    assert (
        wrapper[3]
        == str(status)
        == f"{tmp_path}/pyrite-output/performance/sub_100keV/hopg.py-spy.status"
    )
    child = wrapper[wrapper.index("--") + 1 :]
    assert child[1:4] == ["-m", "pyrite._dev", "perf"]
    assert "--py-spy" not in child  # child must not recurse
    assert (
        f"{tmp_path}/pyrite-output/checkpoints/performance/sub_100keV/py-spy-checkpoints/hopg"
        in child
    )


def test_perf_rejects_py_spy_with_nsys(monkeypatch):
    monkeypatch.setattr(
        scan, "_reexec_under_py_spy", lambda **_: pytest.fail("must not launch py-spy")
    )
    result = CliRunner().invoke(
        performance_command, ["sub_100keV", "-m", "hopg", "--py-spy", "--nsys"]
    )
    assert result.exit_code == 2
    assert "--py-spy and --nsys are mutually exclusive" in result.output
