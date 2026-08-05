from __future__ import annotations

from cxr_mc._remote import config as remote_config
from cxr_mc.cli import _config
from cxr_mc.cli.commands import config as config_command
from cxr_mc.cli.commands import scan
from tests.helpers.cli import assert_clean_result, invoke


def _isolated_store(monkeypatch, tmp_path):
    path = tmp_path / "cxr" / "config.toml"
    monkeypatch.setattr(_config, "CONFIG_PATH", path)
    monkeypatch.delenv("CXR_PROFILE", raising=False)
    monkeypatch.delenv("CXR_REMOTE_HOST", raising=False)
    monkeypatch.setattr(remote_config, "HOST", None)
    return path


def test_config_set_get_and_list_effective_values(monkeypatch, tmp_path):
    path = _isolated_store(monkeypatch, tmp_path)

    set_profile = invoke(config_command.command, ["set", "profile.current", "sub_100keV"])
    set_remote = invoke(config_command.command, ["set", "remote.target", "box-a"])
    get_profile = invoke(config_command.command, ["get", "profile.current"])
    listed = invoke(config_command.command, ["list"])

    assert_clean_result(set_profile, stdout="profile.current = sub_100keV\n")
    assert_clean_result(set_remote, stdout="remote.target = box-a\n")
    assert_clean_result(get_profile, stdout="sub_100keV\n")
    assert_clean_result(
        listed,
        stdout=(
            "KEY\tVALUE\tSOURCE\n"
            "profile.current\tsub_100keV\tconfig store\n"
            "remote.target\tbox-a\tconfig store\n"
        ),
    )
    assert path.is_file()


def test_shared_precedence_is_call_environment_store_default(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    _config.set_stored("profile.current", "sub_100keV")

    assert _config.resolve("profile.current").value == "sub_100keV"
    monkeypatch.setenv("CXR_PROFILE", "standard")
    assert _config.resolve("profile.current").value == "standard"
    assert _config.resolve("profile.current", "sub_100keV").value == "sub_100keV"

    monkeypatch.delenv("CXR_PROFILE")
    _config.CONFIG_PATH.unlink()
    resolved = _config.resolve("profile.current")
    assert (resolved.value, resolved.source) == ("standard", "built-in default")


def test_run_and_remote_host_read_shared_context(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    _config.set_stored("profile.current", "sub_100keV")
    _config.set_stored("remote.target", "box-a")
    seen = {}
    monkeypatch.setattr(scan._scan, "run", lambda args: seen.update(vars(args)))

    result = invoke(scan.command)

    assert_clean_result(result)
    assert seen["catalog_profile"] == "sub_100keV"
    assert remote_config.remote_host() == "box-a"


def test_config_rejects_unknown_profile_and_unsafe_remote(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)

    profile = invoke(config_command.command, ["set", "profile.current", "missing"])
    remote = invoke(config_command.command, ["set", "remote.target", "bad host"])

    assert profile.exit_code == 2
    assert "unknown profile" in profile.stderr
    assert remote.exit_code == 2
    assert "expected host alias" in remote.stderr
