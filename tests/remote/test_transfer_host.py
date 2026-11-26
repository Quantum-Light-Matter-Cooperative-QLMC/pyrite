"""Transfer-node routing for remote targets (#1377).

A target's optional ``transfer_host`` (a site DTN sharing ``remote_dir``)
carries bulk data: sync uploads, inventories and extraction, staged pull
downloads, trajectory and performance downloads. SLURM, state and PyRITE
Python stay on the login host. The end-to-end tests run real ``sh``/``rsync``
against a local directory standing in for the shared filesystem and record
which host each call targeted.
"""

import importlib
import shutil
import subprocess

import pytest

from pyrite.remote import config, transport
from pyrite.remote.config import RemoteTarget

pull_module = importlib.import_module("pyrite.remote.pull")

pytestmark = pytest.mark.usefixtures("empty_user_catalog")
needs_rsync = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")

LOGIN, DTN = "h2", "h2-dtn"


def _target(remote_dir, **extra):
    return RemoteTarget(host=LOGIN, remote_dir=str(remote_dir), pixi="pixi", **extra)


def test_transfer_host_routes_only_data_paths(monkeypatch):
    monkeypatch.setattr(config, "_remote_home", lambda host: "/u/home/a/user")
    with config.use_target(_target("~/pyrite", transfer_host=DTN)):
        assert config.remote_host() == LOGIN
        assert config.transfer_host() == DTN
        assert config.uses_transfer_host()
        assert config.rsync_remote_path("/u/x") == f"{DTN}:/u/x"
        assert config.scp_remote_path("/u/x y") == f"{DTN}:'/u/x y'"
    with config.use_target(_target("/srv/p")):
        assert not config.uses_transfer_host()
        assert config.transfer_host() == LOGIN
        assert config.rsync_remote_path("/u/x") == f"{LOGIN}:/u/x"
        assert config.scp_remote_path("/u/x") == f"{LOGIN}:/u/x"


def test_transfer_host_equal_to_login_host_is_no_transfer_host():
    with config.use_target(_target("/srv/p", transfer_host=LOGIN)):
        assert not config.uses_transfer_host()


def test_unselected_target_needs_no_host_to_answer(monkeypatch):
    monkeypatch.setattr(config, "_selected_target", lambda: None)
    monkeypatch.setattr(config, "remote_host", lambda: pytest.fail("contacted a host"))
    assert not config.uses_transfer_host()


def test_home_relative_checkout_requires_a_shared_home(monkeypatch):
    homes = {LOGIN: "/u/home/a/user", DTN: "/home/user"}
    monkeypatch.setattr(config, "_remote_home", homes.__getitem__)
    with config.use_target(_target("~/pyrite", transfer_host=DTN)):
        with pytest.raises(SystemExit, match="different homes"):
            config.transfer_host()
    # An absolute checkout is taken as shared; no home lookup is needed.
    monkeypatch.setattr(config, "_remote_home", lambda host: pytest.fail("looked up home"))
    with config.use_target(_target("/u/scratch/p", transfer_host=DTN)):
        assert config.transfer_host() == DTN


@pytest.mark.parametrize("value", ["-dtn", "dtn host", "dtn;ls", "a/b"])
def test_transfer_host_must_be_an_ssh_alias(value):
    with pytest.raises(ValueError):
        RemoteTarget(host=LOGIN, transfer_host=value)


def test_target_cli_saves_transfer_host(monkeypatch, tmp_path):
    from pyrite.cli.commands import remote_targets
    from pyrite.console import config as store
    from pyrite.remote import targets
    from tests.helpers.cli import invoke

    monkeypatch.setattr(store, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_REMOTE_TARGET", raising=False)
    result = invoke(
        remote_targets.command,
        ["add", "h2", "--host", LOGIN, "--remote-dir", "/srv/h2", "--transfer-host", DTN],
    )
    assert result.exit_code == 0, result.output
    assert targets.resolve_target("h2").transfer_host == DTN
    result = invoke(remote_targets.command, ["edit", "h2", "--transfer-host", ""])
    assert result.exit_code == 0, result.output
    assert targets.resolve_target("h2").transfer_host == ""


# --- sync -------------------------------------------------------------------


def _strip(arg):
    for host in (LOGIN, DTN):
        arg = arg.removeprefix(f"{host}:")
    return arg


def _fake_hosts(monkeypatch):
    """Run every remote call locally; return ``[(kind, host, command)]``."""
    calls = []

    def capture(command, *, host=None):
        calls.append(("capture", host or config.remote_host(), command))
        if "squeue" in command:
            return ""
        # The box's pixi is configured, not necessarily on this machine's PATH.
        command = command.replace('command -v "pixi"', "true")
        return subprocess.check_output(["sh", "-c", command], text=True)

    def run(argv, **_kwargs):
        if argv[0] == "rsync":
            dash_e = argv.index("-e")
            hosts = {a.split(":", 1)[0] for a in argv if a.startswith((f"{LOGIN}:", f"{DTN}:"))}
            calls.append(("rsync", hosts.pop(), argv[-1]))
            subprocess.check_call([_strip(a) for a in argv[:dash_e] + argv[dash_e + 2 :]])
        elif argv[0] == "scp":
            calls.append(("scp", argv[-1].split(":", 1)[0], argv[-1]))
            subprocess.check_call(["cp", argv[-2], _strip(argv[-1]).strip("'")])
        elif argv[0] == "ssh":
            host = next(a for a in argv[1:] if a in (LOGIN, DTN))
            calls.append(("ssh", host, argv[-1]))
            subprocess.check_call(["sh", "-c", argv[-1]])

    monkeypatch.setattr(transport, "_ssh_capture", capture)
    monkeypatch.setattr(transport, "_run", run)
    return calls


def _checkout(monkeypatch, local, mode):
    pkg = local / "src" / "pyrite"
    pkg.mkdir(parents=True)
    (pkg / "keep.py").write_text("KEEP = 1\n")
    (local / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    monkeypatch.setattr(config, "LOCAL_ROOT", local)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src", "pyproject.toml"])
    monkeypatch.setenv("PYRITE_SSH_MUX", "0")
    monkeypatch.setenv("PYRITE_SYNC_TRANSPORT", mode)
    monkeypatch.setattr(config, "_remote_home", lambda host: "/unused")


def _host_of(calls, needle):
    return {host for _kind, host, command in calls if needle in command}


@needs_rsync
@pytest.mark.parametrize("mode", ["rsync", "tar"])
def test_sync_moves_data_through_the_transfer_host(tmp_path, monkeypatch, mode):
    box = tmp_path / "box"
    box.mkdir()
    _checkout(monkeypatch, tmp_path / "local", mode)
    calls = _fake_hosts(monkeypatch)
    with config.use_target(_target(box, transfer_host=DTN)):
        transport.sync_code()
    assert (box / "src/pyrite/keep.py").read_text() == "KEEP = 1\n"
    assert (box / config.SYNC_STAMP_NAME).is_file()
    # Scheduler state and the pixi probe stay on the login host ...
    assert _host_of(calls, "squeue") == {LOGIN}
    if mode == "rsync":  # the pixi probe rides the inventory round trip only when one runs
        assert _host_of(calls, transport._PIXI_MISSING_MARK) == {LOGIN}
        assert _host_of(calls, "command -v rsync") == {DTN}
    # ... while every upload, inventory and extraction goes to the DTN.
    data = [(kind, host) for kind, host, _ in calls if kind in ("rsync", "scp")]
    assert data and {host for _, host in data} == {DTN}
    assert _host_of(calls, config.SYNC_STAMP_NAME) == {DTN}
    if mode == "tar":
        uploaded = next(command for kind, _host, command in calls if kind == "scp")
        assert uploaded.startswith(f"{DTN}:{box}/.pyrite-stage-pyrite_code.")
    assert not list(box.glob(".pyrite-stage*"))


@needs_rsync
def test_sync_without_transfer_host_keeps_the_login_host_and_tmp(tmp_path, monkeypatch):
    box = tmp_path / "box"
    box.mkdir()
    _checkout(monkeypatch, tmp_path / "local", "tar")
    calls = _fake_hosts(monkeypatch)
    monkeypatch.setattr(
        subprocess, "check_call", _tmp_scp(subprocess.check_call, tmp_path / "tmp"), raising=True
    )
    with config.use_target(_target(box)):
        transport.sync_code()
    assert {host for _kind, host, _command in calls} == {LOGIN}
    uploaded = next(command for kind, _host, command in calls if kind == "scp")
    assert uploaded.startswith(f"{LOGIN}:/tmp/pyrite_code.")


def _tmp_scp(check_call, tmp):
    """Redirect the box's /tmp into the test tree for the no-transfer tar path."""
    tmp.mkdir()

    def call(argv, *args, **kwargs):
        argv = [a.replace("/tmp/pyrite_code.", f"{tmp}/pyrite_code.") for a in argv]
        return check_call(argv, *args, **kwargs)

    return call


# --- staged pull --------------------------------------------------------------


def _fake_pixi(tmp_path, *, fail=False):
    """A ``host_run_command`` stand-in whose ``checkpoint slim ... -o OUT`` copies the checkpoint."""
    script = tmp_path / "fake_pixi.sh"
    count = tmp_path / "slims"
    body = 'while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift;; slim) ckpt=$2;; esac; shift; done\n'
    body += f'echo x >> "{count}"\n'
    body += 'echo partial > "$out"; exit 9\n' if fail else 'cat "$ckpt"/* > "$out"\n'
    script.write_text(body)
    return f"sh {script}", count


def _staged_box(tmp_path, monkeypatch, *, fail=False):
    box = tmp_path / "box"
    ckpt = box / "pyrite-output" / "checkpoints" / "hopg"
    ckpt.mkdir(parents=True)
    (ckpt / "part.bin").write_bytes(b"checkpoint-bytes" * 64)
    pixi, count = _fake_pixi(tmp_path, fail=fail)
    monkeypatch.setattr(config, "host_run_command", lambda extra_env="": pixi)
    monkeypatch.setenv("PYRITE_SSH_MUX", "0")
    return box, str(ckpt), count


def _staging(box):
    return sorted(p.name for p in (box / pull_module.PULL_STAGING).glob("*"))


@needs_rsync
def test_staged_pull_slims_on_login_and_downloads_from_transfer_host(tmp_path, monkeypatch):
    box, ckpt, count = _staged_box(tmp_path, monkeypatch)
    calls = _fake_hosts(monkeypatch)
    incoming = tmp_path / ".hopg.incoming.pkl"
    with config.use_target(_target(box, transfer_host=DTN)):
        pull_module._staged_transfer(ckpt, " --grid", incoming, "hopg")
    assert incoming.read_bytes() == b"checkpoint-bytes" * 64
    assert count.read_text().count("x") == 1
    assert _host_of(calls, "checkpoint slim") == {LOGIN}
    assert {host for kind, host, _ in calls if kind == "rsync"} == {DTN}
    assert _host_of(calls, "sha256sum") == {DTN}
    assert _staging(box) == []  # removed once the local copy verified


@needs_rsync
def test_interrupted_staged_pull_resumes_without_reslimming(tmp_path, monkeypatch):
    box, ckpt, count = _staged_box(tmp_path, monkeypatch)
    calls = _fake_hosts(monkeypatch)
    real_run = transport._run

    def broken(argv, **kwargs):
        if argv[0] == "rsync":
            raise subprocess.CalledProcessError(30, argv)
        return real_run(argv, **kwargs)

    incoming = tmp_path / ".hopg.incoming.pkl"
    with config.use_target(_target(box, transfer_host=DTN)):
        monkeypatch.setattr(transport, "_run", broken)
        with pytest.raises(subprocess.CalledProcessError):
            pull_module._staged_transfer(ckpt, "", incoming, "hopg")
        assert len(_staging(box)) == 2  # staged pickle + fingerprint survive
        monkeypatch.setattr(transport, "_run", real_run)
        pull_module._staged_transfer(ckpt, "", incoming, "hopg")
    assert count.read_text().count("x") == 1  # no second slim
    assert incoming.read_bytes() == b"checkpoint-bytes" * 64
    assert _staging(box) == []
    assert calls  # every call above was routed through the fakes


@needs_rsync
def test_changed_checkpoint_is_reslimmed(tmp_path, monkeypatch):
    box, ckpt, count = _staged_box(tmp_path, monkeypatch)
    _fake_hosts(monkeypatch)
    staged = pull_module._staged_transfer
    with config.use_target(_target(box, transfer_host=DTN)):
        monkeypatch.setattr(transport, "_run", _fail_rsync_once(transport._run), raising=True)
        with pytest.raises(subprocess.CalledProcessError):
            staged(ckpt, "", tmp_path / "in.pkl", "hopg")
        (box / "pyrite-output/checkpoints/hopg/more.bin").write_bytes(b"new")
        staged(ckpt, "", tmp_path / "in.pkl", "hopg")
    assert count.read_text().count("x") == 2
    assert b"new" in (tmp_path / "in.pkl").read_bytes()


def _fail_rsync_once(run):
    state = {"failed": False}

    def call(argv, **kwargs):
        if argv[0] == "rsync" and not state["failed"]:
            state["failed"] = True
            raise subprocess.CalledProcessError(30, argv)
        return run(argv, **kwargs)

    return call


def test_failed_slim_leaves_no_staging(tmp_path, monkeypatch):
    box, ckpt, _count = _staged_box(tmp_path, monkeypatch, fail=True)
    _fake_hosts(monkeypatch)
    with config.use_target(_target(box, transfer_host=DTN)):
        with pytest.raises(subprocess.CalledProcessError):
            pull_module._staged_transfer(ckpt, "", tmp_path / "in.pkl", "hopg")
    assert _staging(box) == []


def test_gc_sweeps_leftover_staging_but_keeps_a_running_slim(tmp_path, monkeypatch):
    box = tmp_path / "box"
    staging = box / pull_module.PULL_STAGING
    staging.mkdir(parents=True)
    for name in ("hopg.abc.pkl", "hopg.abc.pkl.fp", "hopg.abc.pkl.partial.123"):
        (staging / name).write_text("x")
    calls = _fake_hosts(monkeypatch)
    with config.use_target(_target(box, transfer_host=DTN)):
        previewed = pull_module.reclaim_pull_staging(yes=False)
        assert len(previewed) == 2 and len(list(staging.iterdir())) == 3
        removed = pull_module.reclaim_pull_staging(yes=True)
    assert sorted(p.rsplit("/", 1)[1] for p in removed) == ["hopg.abc.pkl", "hopg.abc.pkl.fp"]
    assert [p.name for p in staging.iterdir()] == ["hopg.abc.pkl.partial.123"]
    assert {host for _kind, host, _ in calls} == {DTN}


def test_gc_without_transfer_host_contacts_nothing(tmp_path, monkeypatch):
    calls = _fake_hosts(monkeypatch)
    with config.use_target(_target(tmp_path)):
        assert pull_module.reclaim_pull_staging(yes=True) == []
    assert calls == []


def test_pull_uses_staged_transfer_only_with_a_transfer_host(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        pull_module, "_staged_transfer", lambda *args: seen.append("staged") or _stop()
    )
    monkeypatch.setattr(
        transport, "_ssh_download", lambda *args, **kwargs: seen.append("stream") or _stop()
    )
    monkeypatch.setattr(pull_module, "_resolve_selectors", lambda stems, *_: stems)
    monkeypatch.setattr(pull_module, "_pull_selection", lambda stems, *_: stems)
    monkeypatch.setattr(pull_module, "output_dir", lambda kind: tmp_path / kind)
    monkeypatch.setattr(config, "_remote_home", lambda host: "/unused")
    for target, expected in (
        (_target("/srv/p", transfer_host=DTN), "staged"),
        (_target("/srv/p"), "stream"),
    ):
        seen.clear()
        with config.use_target(target), pytest.raises(SystemExit):
            pull_module.pull(["hopg"], no_sync=True)
        assert seen == [expected]


def _stop():
    raise RuntimeError("stop after transfer routing")
