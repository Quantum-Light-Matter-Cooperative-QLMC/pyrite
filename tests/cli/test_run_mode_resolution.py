"""Mode resolution of ``pyrite run``/``pyrite-dev perf`` before any dispatch."""

import pytest
from click.testing import CliRunner

from pyrite.cli.commands import _simulation, scan
from pyrite.cli.commands import remote as remote_cli
from pyrite.runs import scan as runs_scan
from tests.helpers.cli import assert_clean_result, invoke


@pytest.fixture
def dispatched(monkeypatch):
    """Record every execution sink the command can reach."""
    calls = []

    def sink(label):
        return lambda *args, **kwargs: calls.append((label, args, kwargs))

    monkeypatch.setattr(runs_scan, "run", sink("run"))
    monkeypatch.setattr(runs_scan, "_run_json", sink("run_json"))
    monkeypatch.setattr(runs_scan, "_reexec_under_nsys", sink("nsys"))
    monkeypatch.setattr(runs_scan, "_reexec_under_py_spy", sink("py_spy"))
    monkeypatch.setattr(remote_cli, "_cli_start", sink("remote_start"))
    monkeypatch.setattr(remote_cli, "remote_check", sink("remote_check"))
    monkeypatch.setattr(scan, "_reproduce_zhai", sink("zhai"))
    monkeypatch.setattr(scan, "set_canonical_env", sink("env"))
    monkeypatch.setattr(_simulation, "execute_simulation", sink("simulate"))
    return calls


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["standard", "-m", "hopg", "--no-sync"], "--no-sync/--dry-run require -R/--remote"),
        (["standard", "-m", "hopg", "--dry-run"], "--no-sync/--dry-run require -R/--remote"),
        (
            ["standard", "-m", "hopg", "--remote", "--wait", "--detach"],
            "--wait and --detach are mutually exclusive",
        ),
        (["standard", "--ne", "10"], "Zhai option(s) require --preset zhai: --ne"),
        (
            ["standard", "--preset", "zhai"],
            "--preset zhai does not support normal-run option(s): PROFILE",
        ),
        (
            ["--preset", "zhai", "--workers", "2", "--quick"],
            "--preset zhai does not support normal-run option(s): --workers, --quick",
        ),
    ],
)
def test_invalid_combinations_refuse_before_dispatch(dispatched, argv, message):
    result = invoke(scan.command, argv)

    assert result.exit_code == 2
    assert message in result.stderr
    assert dispatched == []


def test_ephemeral_counts_default_map_options_as_explicit(dispatched):
    result = CliRunner().invoke(
        scan.command,
        ["standard", "-m", "hopg", "--ephemeral"],
        default_map={"workers": 2},
    )

    assert result.exit_code == 2
    assert "--ephemeral does not support option(s): --workers" in result.stderr
    assert dispatched == []


def test_preset_ignores_default_map_normal_run_options(dispatched):
    result = CliRunner().invoke(
        scan.command, ["--preset", "zhai", "--ne", "10"], default_map={"workers": 2}
    )

    assert_clean_result(result)
    assert dispatched == [
        (
            "zhai",
            (),
            {"ne": 10, "ne_brem": 200, "ne_supp": 200, "tmd_azimuth": 0.0, "refresh": False},
        )
    ]


def test_environment_profile_reaches_local_run(dispatched, monkeypatch):
    monkeypatch.setenv("PYRITE_PROFILE", "standard")

    result = invoke(scan.command, ["-m", "hopg"])

    assert_clean_result(result)
    [(label, (args,), _kwargs)] = dispatched
    assert label == "run"
    assert (args.catalog_profile, args.material) == ("standard", "hopg")
    assert (args.cache_read, args.cache_write, args.performance_profile) == (True, True, None)


@pytest.mark.parametrize(
    ("extra", "cache"),
    [([], (False, False)), (["--recompute"], (False, True)), (["--no-cache"], (False, False))],
)
def test_perf_local_run_resolves_profile_and_cache_gates(dispatched, extra, cache):
    result = invoke(
        scan.performance_command, ["standard", "-m", "hopg", "--cxr-chunk", "4", *extra]
    )

    assert_clean_result(result)
    assert dispatched[0] == ("env", ("PYRITE_MC_CXR_CHUNK", "4"), {})
    [(label, (args,), _kwargs)] = dispatched[1:]
    assert label == "run"
    assert args.performance_profile == "standard"
    assert (args.cache_read, args.cache_write) == cache


@pytest.mark.parametrize(("flag", "label"), [("--nsys", "nsys"), ("--py-spy", "py_spy")])
def test_perf_profiler_reexec_replaces_the_local_run(dispatched, flag, label):
    result = invoke(scan.performance_command, ["standard", "-m", "hopg", flag])

    assert_clean_result(result)
    [(called, args, kwargs)] = dispatched
    assert (called, args) == (label, ())
    assert kwargs["performance_profile"] == "standard"
    assert (kwargs["catalog_profile"], kwargs["material"]) == ("standard", "hopg")


@pytest.mark.parametrize(("flags", "headless"), [(["--detach"], True), ([], False)])
def test_remote_dry_run_forwards_without_local_execution(dispatched, flags, headless):
    result = invoke(scan.command, ["standard", "-m", "hopg", "--remote", "--dry-run", *flags])

    assert_clean_result(result)
    [(label, _args, kwargs)] = dispatched
    assert label == "remote_start"
    assert (kwargs["dry_run"], kwargs["headless"]) == (True, headless)
