from __future__ import annotations

from pyrite.cli.commands import recompute as recompute_cli
from pyrite.cli.commands import scan
from pyrite.cli.commands.scan import performance_command
from pyrite.remote import cli as remote_cli
from pyrite.remote import config as remote_config
from tests.helpers.cli import assert_clean_result, invoke


def test_run_remote_modifier_delegates_and_restores_explicit_target(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote_config, "HOST", "configured-box")

    def start(**kwargs):
        seen.update(kwargs, host=remote_config.remote_host())

    monkeypatch.setattr(remote_cli, "_cli_start", start)

    result = invoke(
        scan.command,
        ["standard", "-m", "hopg", "--remote=box-a", "--detach"],
    )

    assert_clean_result(result)
    assert seen["host"] == "box-a"
    assert seen["catalog_profile"] == "standard"
    assert seen["materials"] == ["hopg"]
    assert seen["headless"] is True
    assert remote_config.remote_host() == "configured-box"


def test_bare_remote_uses_configured_target_and_waits_by_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote_config, "HOST", "configured-box")
    monkeypatch.setattr(
        remote_cli,
        "_cli_start",
        lambda **kwargs: seen.update(kwargs, host=remote_config.remote_host()),
    )

    result = invoke(scan.command, ["standard", "-m", "hopg", "--remote"])

    assert_clean_result(result)
    assert seen["host"] == "configured-box"
    assert seen["headless"] is False
    assert seen["no_pull"] is False


def test_remote_cache_modes_delegate_with_local_semantics(monkeypatch):
    seen = []
    monkeypatch.setattr(remote_cli, "_cli_start", lambda **kwargs: seen.append(kwargs))

    no_cache = invoke(
        scan.command,
        ["standard", "-m", "hopg", "--remote", "--no-cache", "--detach"],
    )
    recompute = invoke(
        scan.command,
        ["standard", "-m", "hopg", "--remote", "--recompute", "--detach"],
    )

    assert_clean_result(no_cache)
    assert_clean_result(recompute)
    assert (seen[0]["no_cache"], seen[0]["recompute"]) == (True, False)
    assert (seen[1]["no_cache"], seen[1]["recompute"]) == (False, True)


def test_remote_cache_modes_are_mutually_exclusive_before_submission(monkeypatch):
    def fail_submission(**_kwargs):
        raise AssertionError("cache conflict must not submit")

    monkeypatch.setattr(remote_cli, "_cli_start", fail_submission)

    result = invoke(
        scan.command,
        ["standard", "-m", "hopg", "--remote", "--no-cache", "--recompute"],
    )

    assert result.exit_code == 2
    assert "--no-cache and --recompute are mutually exclusive" in result.stderr


def test_remote_wait_detach_and_local_only_options_are_rejected():
    conflict = invoke(scan.command, ["--remote", "--wait", "--detach"])
    local_only = invoke(scan.command, ["--remote", "--checkpoint-dir", "elsewhere"])
    local_wait = invoke(scan.command, ["--wait"])

    assert conflict.exit_code == 2
    assert "--wait and --detach are mutually exclusive" in conflict.stderr
    assert local_only.exit_code == 2
    assert "local-only option(s): --checkpoint-dir" in local_only.stderr
    assert local_wait.exit_code == 2
    assert "--wait/--detach require -R/--remote" in local_wait.stderr


def test_cpu_profile_flags_require_remote_and_reach_the_job(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote_cli, "_cli_start", lambda **kwargs: seen.update(kwargs))

    local_cpu = invoke(performance_command, ["standard", "--cpu"])
    local_cpu_only = invoke(performance_command, ["standard", "--cpu-only"])
    remote = invoke(
        performance_command,
        ["standard", "-m", "hopg", "--remote=box-a", "--cpu", "--detach"],
    )

    assert local_cpu.exit_code == 2
    assert "--cpu/--cpu-only require -R/--remote" in local_cpu.stderr
    assert local_cpu_only.exit_code == 2
    assert "--cpu/--cpu-only require -R/--remote" in local_cpu_only.stderr
    assert_clean_result(remote)
    assert seen["cpu"] is True
    assert seen["cpu_only"] is False
    # --cpu implies --perf and an unchunked session; both are start_command's job.
    assert seen["performance_profile"] == "standard"
    assert seen["chunk_minutes"] == 0.0


def test_cpu_profile_flag_conflicts_are_rejected_before_submission():
    both = invoke(
        performance_command,
        ["standard", "--remote=box-a", "--cpu", "--cpu-only"],
    )
    with_nsys = invoke(
        performance_command,
        ["standard", "--remote=box-a", "--cpu-only", "--nsys"],
    )
    with_chunk = invoke(
        performance_command,
        ["standard", "--remote=box-a", "--cpu-only", "--spec-chunk", "8"],
    )

    assert both.exit_code == 2
    assert "--cpu and --cpu-only are mutually exclusive" in both.stderr
    assert with_nsys.exit_code == 2
    assert "--cpu-only cannot be combined with --nsys" in with_nsys.stderr
    assert with_chunk.exit_code == 2
    assert "--cpu-only cannot be combined with GPU chunk pins" in with_chunk.stderr


def test_explicit_remote_wait_delegates_without_detaching(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote_cli, "_cli_start", lambda **kwargs: seen.update(kwargs))

    result = invoke(scan.command, ["standard", "--remote", "--wait"])

    assert_clean_result(result)
    assert seen["headless"] is False
    assert seen["no_pull"] is False


def test_invalid_remote_target_is_usage_error_and_restores_host(monkeypatch):
    monkeypatch.setattr(remote_config, "HOST", "configured-box")

    result = invoke(scan.command, ["standard", "--remote=bad host"])

    assert result.exit_code == 2
    assert "Invalid value for '-R' / '--remote'" in result.stderr
    assert "expected host alias" in result.stderr
    assert remote_config.remote_host() == "configured-box"


def test_zhai_preset_waits_pulls_and_restores_explicit_target(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote_config, "HOST", "configured-box")

    def check(**kwargs):
        seen.update(kwargs, host=remote_config.remote_host())

    monkeypatch.setattr(remote_cli, "remote_check", check)

    result = invoke(
        scan.command,
        [
            "--preset",
            "zhai",
            "--remote=box-a",
            "--wait",
            "--ne",
            "11",
            "--ne-brem",
            "3",
            "--ne-supp",
            "5",
            "--tmd-azimuth",
            "35",
            "--refresh",
            "--no-sync",
        ],
    )

    assert_clean_result(result)
    assert seen == {
        "ne": 11,
        "ne_brem": 3,
        "ne_supp": 5,
        "tmd_azimuth": 35.0,
        "refresh": True,
        "no_sync": True,
        "detach": False,
        "dry_run": False,
        "host": "box-a",
    }
    assert remote_config.remote_host() == "configured-box"


def test_zhai_preset_detach_and_dry_run_flow_to_remote_workflow(monkeypatch):
    calls = []
    monkeypatch.setattr(remote_cli, "remote_check", lambda **kwargs: calls.append(kwargs))

    detached = invoke(scan.command, ["--preset", "zhai", "--remote", "--detach"])
    preview = invoke(scan.command, ["--preset", "zhai", "--remote", "--dry-run"])

    assert_clean_result(detached)
    assert_clean_result(preview)
    assert calls[0]["detach"] is True
    assert calls[0]["dry_run"] is False
    assert calls[1]["detach"] is False
    assert calls[1]["dry_run"] is True


def test_zhai_preset_rejects_locality_and_normal_run_inputs():
    local = invoke(scan.command, ["--preset", "zhai"])
    positional = invoke(scan.command, ["standard", "--preset", "zhai", "--remote"])
    normal_option = invoke(
        scan.command,
        ["--preset", "zhai", "--remote", "--material", "hopg"],
    )
    missing_preset = invoke(scan.command, ["--remote", "--ne", "11"])

    assert local.exit_code == 2
    assert "require -R/--remote" in local.stderr
    assert positional.exit_code == 2
    assert "normal-run option(s): PROFILE" in positional.stderr
    assert normal_option.exit_code == 2
    assert "normal-run option(s): --material" in normal_option.stderr
    assert missing_preset.exit_code == 2
    assert "require --preset zhai: --ne" in missing_preset.stderr


def test_remote_optional_value_parses_around_profile(monkeypatch):
    seen = []
    monkeypatch.setattr(
        remote_cli,
        "_cli_start",
        lambda **kwargs: seen.append((kwargs["catalog_profile"], remote_config.remote_host())),
    )

    before = invoke(scan.command, ["--remote=box-a", "standard"])
    after = invoke(scan.command, ["standard", "--remote=box-b"])

    assert_clean_result(before)
    assert_clean_result(after)
    assert seen == [("standard", "box-a"), ("standard", "box-b")]


def test_recompute_remote_modifier_delegates_and_restores_target(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote_config, "HOST", "configured-box")
    monkeypatch.setattr(
        remote_cli,
        "_cli_rebrem",
        lambda **kwargs: seen.update(kwargs, host=remote_config.remote_host()),
    )

    result = invoke(
        recompute_cli.brem_command,
        ["hopg", "--remote=box-a", "--detach", "--ne-brem", "25"],
    )

    assert_clean_result(result)
    assert seen["material"] == ["hopg"]
    assert seen["ne_brem"] == 25
    assert seen["detach"] is True
    assert seen["host"] == "box-a"
    assert remote_config.remote_host() == "configured-box"


def test_recompute_bare_remote_waits_by_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote_config, "HOST", "configured-box")
    monkeypatch.setattr(remote_cli, "_cli_reline", lambda **kwargs: seen.update(kwargs))

    result = invoke(recompute_cli.line_command, ["hopg", "--remote", "--wait"])

    assert_clean_result(result)
    assert seen["material"] == ["hopg"]
    assert seen["detach"] is False
    assert seen["chunk_minutes"] == 10.0


def test_recompute_locality_controls_are_rejected_when_incompatible():
    conflict = invoke(
        recompute_cli.brem_command,
        ["hopg", "--remote", "--wait", "--detach"],
    )
    local_remote_option = invoke(recompute_cli.line_command, ["hopg", "--no-sync"])
    remote_local_option = invoke(
        recompute_cli.line_command,
        ["hopg", "--remote", "--checkpoint-dir", "elsewhere"],
    )
    remote_json = invoke(
        recompute_cli.line_command,
        ["hopg", "--remote", "-o", "json"],
    )

    assert conflict.exit_code == 2
    assert "--wait and --detach are mutually exclusive" in conflict.stderr
    assert local_remote_option.exit_code == 2
    assert "remote-only option(s) require -R/--remote: --no-sync" in local_remote_option.stderr
    assert remote_local_option.exit_code == 2
    assert "local-only option(s): --checkpoint-dir" in remote_local_option.stderr
    assert remote_json.exit_code == 2
    assert "does not yet support --output json" in remote_json.stderr
