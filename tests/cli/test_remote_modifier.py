from __future__ import annotations

from cxr_mc._remote import cli as remote_cli
from cxr_mc._remote import config as remote_config
from cxr_mc.cli.commands import scan
from tests.helpers.cli import assert_clean_result, invoke


def test_run_remote_modifier_delegates_and_restores_explicit_target(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote_config, "HOST", "configured-box")

    def start(args):
        seen.update(vars(args), host=remote_config.remote_host())

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
        lambda args: seen.update(vars(args), host=remote_config.remote_host()),
    )

    result = invoke(scan.command, ["standard", "-m", "hopg", "--remote"])

    assert_clean_result(result)
    assert seen["host"] == "configured-box"
    assert seen["headless"] is False
    assert seen["no_pull"] is False


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
