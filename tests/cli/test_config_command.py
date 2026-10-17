import json

from pyrite.cli.commands import config as config_command
from pyrite.cli.commands import scan
from pyrite.console import config as _config
from pyrite.remote import config as remote_config
from tests.helpers.cli import assert_clean_result, invoke


def _isolated_store(monkeypatch, tmp_path):
    path = tmp_path / "pyrite" / "config.toml"
    monkeypatch.setattr(_config, "CONFIG_PATH", path)
    monkeypatch.delenv("PYRITE_PROFILE", raising=False)
    monkeypatch.delenv("PYRITE_REMOTE_HOST", raising=False)
    monkeypatch.delenv("PYRITE_MOTT_TABLES_DIR", raising=False)
    monkeypatch.setattr(remote_config, "HOST", None)
    for name in (
        "PYRITE_REMOTE_GPU_VENDOR",
        "PYRITE_REMOTE_PARTITION",
        "PYRITE_REMOTE_NODELIST",
        "PYRITE_REMOTE_GRES",
    ):
        monkeypatch.delenv(name, raising=False)
    for attribute in ("REMOTE_GPU_VENDOR", "SLURM_PARTITION", "SLURM_NODELIST", "SLURM_GRES"):
        monkeypatch.setattr(remote_config, attribute, None)
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
    assert_clean_result(listed)
    assert "\t" not in listed.stdout
    lines = listed.stdout.splitlines()
    assert lines[0].split() == ["KEY", "VALUE", "SOURCE"]
    assert len(lines) == len(_config.keys()) + 1
    assert all(any(line.startswith(key + " ") for line in lines) for key in _config.keys())
    assert "sub_100keV" in lines[1] and "config store" in lines[1]
    assert "box-a" in lines[2] and "config store" in lines[2]
    assert path.is_file()


def test_config_unset_removes_stored_value_and_reports_fallback(monkeypatch, tmp_path):
    path = _isolated_store(monkeypatch, tmp_path)
    _config.set_stored("profile.current", "sub_100keV")
    _config.set_stored("remote.target", "box-a")

    unset_profile = invoke(config_command.command, ["unset", "profile.current"])

    assert_clean_result(unset_profile, stdout="profile.current = standard (built-in default)\n")
    assert _config.resolve("remote.target").value == "box-a"
    assert "[profile]" not in path.read_text(encoding="utf-8")

    # A key the environment still sets reports that source as the winner.
    monkeypatch.setenv("PYRITE_REMOTE_HOST", "box-env")
    unset_remote = invoke(config_command.command, ["unset", "remote.target"])

    assert_clean_result(unset_remote, stdout="remote.target = box-env (PYRITE_REMOTE_HOST)\n")
    assert path.read_text(encoding="utf-8").strip() == ""


def test_config_unset_of_unstored_key_is_a_noop(monkeypatch, tmp_path):
    path = _isolated_store(monkeypatch, tmp_path)

    result = invoke(config_command.command, ["unset", "remote.gres"])

    assert_clean_result(
        result,
        stdout="remote.gres = gpu:1 (built-in default)\n",
        stderr="remote.gres is not set in the config store\n",
    )
    assert not path.exists()


def test_config_unset_rejects_unknown_key(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)

    result = invoke(config_command.command, ["unset", "remote.nope"])

    assert result.exit_code == 2
    assert "remote.nope" in result.stderr


def test_config_set_slurm_profile_feeds_remote_target(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)

    for key, value in (
        ("remote.gpu_vendor", "AMD"),
        ("remote.partition", "gpu-amd"),
        ("remote.nodelist", "qlmc-ace"),
        ("remote.gres", "gpu:radeon8060s:1"),
    ):
        assert invoke(config_command.command, ["set", key, value]).exit_code == 0

    assert remote_config.remote_gpu_vendor() == "amd"
    assert remote_config.slurm_partition() == "gpu-amd"
    assert remote_config.slurm_nodelist() == "qlmc-ace"
    assert remote_config.slurm_gres() == "gpu:radeon8060s:1"

    assert invoke(config_command.command, ["set", "remote.nodelist", "any"]).exit_code == 0
    assert remote_config.slurm_nodelist() is None


def test_config_set_rejects_unsafe_slurm_partition(monkeypatch, tmp_path):
    path = _isolated_store(monkeypatch, tmp_path)

    result = invoke(config_command.command, ["set", "remote.partition", "gpu --x"])

    assert result.exit_code == 2
    assert "SLURM partition name" in result.output
    assert not path.exists()


def test_config_keys_describes_every_key_without_resolving(monkeypatch):
    def unexpected(*args):
        raise AssertionError("help/key discovery resolved live configuration")

    monkeypatch.setattr(_config, "resolve", unexpected)
    for args in (["--help"], ["keys", "--help"], ["explain", "--help"]):
        assert_clean_result(invoke(config_command.command, args))
    help_result = invoke(config_command.command, ["--help"])
    assert "Keys (environment" not in help_result.stdout
    result = invoke(config_command.command, ["keys"])
    assert_clean_result(result)
    for key in _config.keys():
        env_name, default, text = _config.describe(key)
        assert f"{key}: {text}" in result.stdout
        assert f"{env_name}; default: {default}" in result.stdout


def test_config_subset_explain_and_unknown_prefix(monkeypatch, tmp_path):
    path = _isolated_store(monkeypatch, tmp_path)
    _config.set_stored("remote.target", "stored")
    monkeypatch.setenv("PYRITE_REMOTE_HOST", "environment")
    subset = invoke(config_command.command, ["list", "remote."])
    assert_clean_result(subset)
    assert "environment" in subset.stdout
    assert "profile.current" not in subset.stdout
    explained = invoke(config_command.command, ["explain", "remote.target"])
    assert_clean_result(explained)
    assert "source: PYRITE_REMOTE_HOST" in explained.stdout
    assert "effective: environment" in explained.stdout
    before = path.read_bytes()
    bad = invoke(config_command.command, ["list", "missing."])
    assert bad.exit_code == 2
    assert "config keys" in bad.stderr
    assert path.read_bytes() == before


def test_config_user_catalog_is_settable(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    monkeypatch.delenv("PYRITE_USER_CATALOG", raising=False)
    destination = tmp_path / "my-catalog"
    result = invoke(config_command.command, ["set", "catalog.user", str(destination)])
    assert_clean_result(result)
    assert _config.resolve("catalog.user").value == str(destination.resolve())


def test_config_set_workspace_root_normalizes_path(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)

    result = invoke(config_command.command, ["set", "workspace.root", "workspace"])

    expected = tmp_path / "workspace"
    assert_clean_result(result, stdout=f"workspace.root = {expected}\n")
    assert _config.resolve("workspace.root").value == str(expected)


def test_shared_precedence_is_call_environment_store_default(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    _config.set_stored("profile.current", "sub_100keV")

    assert _config.resolve("profile.current").value == "sub_100keV"
    monkeypatch.setenv("PYRITE_PROFILE", "standard")
    assert _config.resolve("profile.current").value == "standard"
    assert _config.resolve("profile.current", "sub_100keV").value == "sub_100keV"

    monkeypatch.delenv("PYRITE_PROFILE")
    _config.CONFIG_PATH.unlink()
    resolved = _config.resolve("profile.current")
    assert (resolved.value, resolved.source) == ("standard", "built-in default")


def test_run_and_remote_host_read_shared_context(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    _config.set_stored("profile.current", "quickstart")
    _config.set_stored("remote.target", "box-a")
    seen = {}
    monkeypatch.setattr(scan._scan, "run", lambda args: seen.update(vars(args)))

    result = invoke(scan.command)

    assert_clean_result(result)
    assert seen["catalog_profile"] == "quickstart"
    assert remote_config.remote_host() == "box-a"


def test_config_rejects_unknown_profile_and_unsafe_remote(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)

    profile = invoke(config_command.command, ["set", "profile.current", "missing"])
    remote = invoke(config_command.command, ["set", "remote.target", "bad host"])

    assert profile.exit_code == 2
    assert "unknown profile" in profile.stderr
    assert remote.exit_code == 2
    assert "expected host alias" in remote.stderr


def test_run_rejects_profile_falling_back_to_builtin_standard(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr(scan._scan, "run", lambda args: seen.update(vars(args)))

    result = invoke(scan.command)

    assert result.exit_code == 2
    assert seen == {}
    assert "Error: this run names no profile; pass PROFILE, set PYRITE_PROFILE" in result.stderr
    assert "pyrite config set profile.current NAME" in result.stderr


def test_ephemeral_run_rejects_implicit_profile_with_json_envelope(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)

    result = invoke(scan.command, ["--ephemeral", "-m", "hopg", "-o", "json"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["schema"] == "cxr.material.simulate"
    assert "this run names no profile" in result.stdout


def test_run_is_silent_for_an_explicitly_named_standard_profile(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    monkeypatch.setattr(scan._scan, "run", lambda args: None)

    assert_clean_result(invoke(scan.command, ["standard"]))


def test_config_set_mott_tables_dir_requires_a_directory(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    tables = tmp_path / "srd64"

    missing = invoke(config_command.command, ["set", "mott.tables_dir", str(tables)])
    assert missing.exit_code == 2
    assert "is not a directory" in missing.stderr

    tables.mkdir()
    stored = invoke(config_command.command, ["set", "mott.tables_dir", str(tables)])
    assert_clean_result(stored, stdout=f"mott.tables_dir = {tables.resolve()}\n")
    assert _config.resolve("mott.tables_dir").value == str(tables.resolve())
