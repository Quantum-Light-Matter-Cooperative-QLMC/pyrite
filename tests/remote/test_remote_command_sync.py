"""Remote sync tests."""

import contextlib
import tarfile

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from tests.remote._remote_test_helpers import (
    _live_job_line,
    _remote_main,
    _stub_sync_tree,
)


def test_sync_rejects_hostile_scp_host_before_transport(monkeypatch):
    monkeypatch.setattr(config, "HOST", "-oProxyCommand=touch-/tmp/pwn")
    monkeypatch.setattr(config, "SYNC_PATHS", [])
    monkeypatch.setattr(
        transport,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("invalid host must not reach scp"),
    )

    with pytest.raises(SystemExit, match="PYRITE_REMOTE_HOST"):
        transport.sync_code()


@pytest.mark.usefixtures("empty_user_catalog")
def test_sync_excludes_generated_caches(monkeypatch, tmp_path):
    source = tmp_path / "src"
    (source / "pkg" / "__pycache__").mkdir(parents=True)
    (source / "pkg" / ".pytest_cache").mkdir()
    (source / "pkg" / "module.py").write_text("VALUE = 1\n")
    (source / "pkg" / "__pycache__" / "module.pyc").write_bytes(b"bytecode")
    (source / "pkg" / ".pytest_cache" / "state").write_text("generated")
    archived = []

    def inspect_transfer(command, **_kwargs):
        if command[0] == "scp":
            with tarfile.open(command[1], "r:gz") as archive:
                archived.extend(archive.getnames())

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src"])
    monkeypatch.setattr(transport, "_run", inspect_transfer)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")

    transport.sync_code()

    assert archived == ["src/pkg/module.py"]


def test_sync_removes_stale_remote_python_sources_before_extract(monkeypatch, tmp_path):
    source = tmp_path / "src" / "pyrite"
    source.mkdir(parents=True)
    (source / "current.py").write_text("VALUE = 1\n")
    commands = []

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src"])
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")

    transport.sync_code()

    remote_command = commands[-1]
    assert remote_command[:3] == ["ssh", "-n", remote.HOST]
    assert "find \"$p\" -type f -name '*.py' -delete" in remote_command[3]
    assert remote_command[3].index("tar xzf") < remote_command[3].index("find")
    assert remote_command[3].index("find") < remote_command[3].index("cp -a")


def test_sync_uploads_to_unique_path_and_removes_only_that_file(monkeypatch, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("A = 1\n")
    commands = []
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src"])
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")

    transport.sync_code()
    transport.sync_code()

    uploads = [c[-1].split(":", 1)[-1] for c in commands if c[0] == "scp"]
    assert len(set(uploads)) == 2
    assert all(u.startswith("/tmp/pyrite_code.") and u != "/tmp/pyrite_code.tgz" for u in uploads)
    remotes = [c[-1] for c in commands if c[0] == "ssh"]
    for upload, remote_command in zip(uploads, remotes, strict=True):
        assert f"tar xzf '{upload}'" in remote_command
        assert "/tmp/pyrite_code.tgz" not in remote_command
        assert "rm -rf .pyrite-stage." in remote_command and f"'{upload}'; exit" in remote_command


def test_sync_stamp_is_written_after_swap_and_staging_is_always_cleaned(monkeypatch, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("A = 1\n")
    commands = []
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src"])
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")

    transport.sync_code()

    script = commands[-1][-1]
    assert script.index("cp -a") < script.index("printf %s") < script.index("rm -rf .pyrite-stage")
    assert script.endswith("; exit $rc") or "exit $rc" in script


def test_sync_runs_inside_lock_and_refusal_still_releases_it(monkeypatch, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("A = 1\n")
    events = []

    @contextlib.contextmanager
    def fake_lock():
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src"])
    monkeypatch.setattr(transport, "_remote_sync_lock", fake_lock)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: events.append(command[0]))
    transport.sync_code()
    assert events == ["lock", "scp", "ssh", "unlock"]

    def refuse(*_args, **_kwargs):
        events.append("check")
        raise SystemExit("refusing to sync")

    events.clear()
    monkeypatch.setattr(transport, "_refuse_conflicting_live_jobs", refuse)
    with pytest.raises(SystemExit, match="refusing"):
        transport.sync_code()
    assert events == ["lock", "check", "unlock"]


def test_sync_paths_do_not_include_retired_material_manifest():
    assert "mats_to_sim.toml" not in remote.SYNC_PATHS


def test_sync_paths_ship_checks_entry_shims_and_build_metadata():
    # The entry shims live under src/pyrite/_entry/ now, so they travel via "src";
    # checks/ still ships the real reproduce_all logic.
    assert "checks" in remote.SYNC_PATHS
    assert "src" in remote.SYNC_PATHS
    assert "pyproject.toml" in remote.SYNC_PATHS
    assert "README.md" in remote.SYNC_PATHS
    assert "PYPI.md" in remote.SYNC_PATHS
    assert "LICENSE.txt" in remote.SYNC_PATHS
    assert "THIRD-PARTY-NOTICES.md" in remote.SYNC_PATHS
    assert "vendor" in remote.SYNC_PATHS
    assert "scan.py" not in remote.SYNC_PATHS
    assert "reproduce_zhai.py" not in remote.SYNC_PATHS


def test_sync_refuses_when_a_live_job_runs_different_code(tmp_path, monkeypatch, capsys):
    """The #142 failure: a second tree's sync rewrote src/ under SLURM job 1772."""
    _stub_sync_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: _live_job_line("20260919-134800-ab12cd34", "f" * 64),
    )
    monkeypatch.setattr(
        transport, "_run", lambda *_args, **_kwargs: pytest.fail("refusal must not transfer")
    )

    with pytest.raises(SystemExit) as refusal:
        transport.sync_code()

    message = str(refusal.value)
    assert "20260919-134800-ab12cd34" in message
    assert "ffffffffffff" in message
    assert "deadbeef" in message
    assert "--force" in message


def test_sync_proceeds_when_the_live_job_runs_this_payload(tmp_path, monkeypatch):
    """Identical payloads sync silently: starting a second material mid-run is
    a supported workflow (cleanup._refuse_if_busy allows it)."""
    _stub_sync_tree(tmp_path, monkeypatch)
    digest = transport._payload_digest(transport._sync_entries())
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: _live_job_line("job-1", digest))
    transferred = []
    monkeypatch.setattr(transport, "_run", lambda command, **_kw: transferred.append(command[0]))

    transport.sync_code()

    assert transferred == ["scp", "ssh"]


def test_sync_reports_but_does_not_refuse_an_unstamped_live_job(tmp_path, monkeypatch, capsys):
    """A job queued before code stamping is unverifiable, not a conflict --
    refusing on absence would train every caller to pass --force."""
    _stub_sync_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "legacy\tFalse\thopg\t\t\n")
    transferred = []
    monkeypatch.setattr(transport, "_run", lambda command, **_kw: transferred.append(command[0]))

    transport.sync_code()

    assert transferred == ["scp", "ssh"]
    assert "legacy" in capsys.readouterr().err


def test_sync_force_overrides_conflict_and_names_the_jobs_at_risk(tmp_path, monkeypatch, capsys):
    _stub_sync_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: _live_job_line("job-1", "f" * 64) + _live_job_line("job-2", "e" * 64),
    )
    transferred = []
    monkeypatch.setattr(transport, "_run", lambda command, **_kw: transferred.append(command[0]))

    transport.sync_code(force=True)

    warning = capsys.readouterr().err
    assert transferred == ["scp", "ssh"]
    assert "WARNING" in warning
    assert "job-1" in warning
    assert "job-2" in warning


def test_sync_stamps_the_remote_checkout_once_after_extraction(tmp_path, monkeypatch):
    """The stamp must describe code that actually landed, so it is chained after
    a successful tar with && rather than sent as its own command."""
    _stub_sync_tree(tmp_path, monkeypatch)
    digest = transport._payload_digest(transport._sync_entries())
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")
    commands = []
    monkeypatch.setattr(transport, "_run", lambda command, **_kw: commands.append(command))

    transport.sync_code()

    assert [command[0] for command in commands] == ["scp", "ssh"]
    remote_command = commands[-1][3]
    assert remote_command.count(config.SYNC_STAMP_NAME) == 1
    assert remote_command.index("tar xzf") < remote_command.index(config.SYNC_STAMP_NAME)
    assert "&& printf %s " in remote_command
    assert f"code_digest: {digest}" in remote_command
    assert "code_revision: " in remote_command


def test_sync_stamp_records_revision_evidence_without_keying_identity_to_it(tmp_path, monkeypatch):
    """Revision is evidence; an unavailable one is recorded as dirty, never as a
    clean revision nobody could reproduce."""
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(transport, "_local_revision", lambda: ("unknown", True))

    stamp = transport._sync_stamp("a" * 64)

    rendered = stamp.render()
    assert rendered.startswith(f"code_digest: {'a' * 64}\n")
    assert "code_revision: unknown\n" in rendered
    assert "code_dirty: True\n" in rendered
    assert rendered.endswith("\n")
    assert "\n\n" not in rendered


def test_sync_guard_runs_before_any_transfer_or_inventory(tmp_path, monkeypatch):
    """A refusal costs one squeue join, never an scp or an artifact inventory."""
    _stub_sync_tree(tmp_path, monkeypatch)
    captured = []

    def capture(command):
        captured.append(command)
        return _live_job_line("job-1", "f" * 64)

    monkeypatch.setattr(transport, "_ssh_capture", capture)
    monkeypatch.setattr(
        transport, "_run", lambda *_a, **_kw: pytest.fail("refusal must not transfer")
    )

    with pytest.raises(SystemExit):
        transport.sync_code()

    assert len(captured) == 1
    assert "squeue" in captured[0]


def test_sync_cli_exposes_force_and_defaults_to_refusing(monkeypatch):
    seen = []
    monkeypatch.setattr(transport, "sync_code", lambda *, force=False: seen.append(force))

    assert _remote_main(["sync"]) in (0, None)
    assert _remote_main(["sync", "--force"]) in (0, None)

    assert seen == [False, True]
