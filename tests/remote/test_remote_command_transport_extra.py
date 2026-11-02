"""Additional remote transport tests."""

import os
import subprocess

import pytest

from pyrite import remote
from pyrite.remote import config, state, transport
from tests.remote._remote_test_helpers import (
    _stub_sync_tree,
)


def test_zhai_start_dry_run_prints_without_ssh_or_sync(monkeypatch, capsys):
    monkeypatch.setattr(state, "_live_jobs", lambda: pytest.fail("dry-run must not check busy"))
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        transport.subprocess, "run", lambda *a, **kw: pytest.fail("dry-run must not ssh")
    )

    jobid = remote.start_zhai_queue(dry_run=True)

    out = capsys.readouterr().out
    assert jobid in out and "python -m pyrite._entry.reproduce_zhai" in out


def test_payload_digest_ignores_line_endings_and_mtimes(tmp_path, monkeypatch):
    """Identity is CONTENT: a CRLF checkout and an LF one ship the same code.

    The digest must also survive a touched mtime, or every sync would look like
    a new revision and refuse against its own live jobs."""
    source = _stub_sync_tree(tmp_path, monkeypatch, text="VALUE = 1\nOTHER = 2\n")
    lf = transport._payload_digest(transport._sync_entries())

    (source / "module.py").write_bytes(b"VALUE = 1\r\nOTHER = 2\r\n")
    assert transport._payload_digest(transport._sync_entries()) == lf

    os.utime(source / "module.py", (0, 0))
    os.utime(source / "table.bin", (0, 0))
    assert transport._payload_digest(transport._sync_entries()) == lf


def test_payload_digest_tracks_every_synced_file(tmp_path, monkeypatch):
    """Changed text, changed binary, added or renamed file: all new identities."""
    source = _stub_sync_tree(tmp_path, monkeypatch)
    baseline = transport._payload_digest(transport._sync_entries())

    (source / "module.py").write_text("VALUE = 2\n", newline="")
    changed_text = transport._payload_digest(transport._sync_entries())
    assert changed_text != baseline

    (source / "table.bin").write_bytes(b"\x00\x01\x03")
    assert transport._payload_digest(transport._sync_entries()) != changed_text

    (source / "table.bin").write_bytes(b"\x00\x01\x02")
    (source / "module.py").write_text("VALUE = 1\n", newline="")
    assert transport._payload_digest(transport._sync_entries()) == baseline

    (source / "module.py").rename(source / "renamed.py")
    assert transport._payload_digest(transport._sync_entries()) != baseline


def test_ssh_download_streams_bytes_and_removes_partial_failure(monkeypatch, tmp_path):
    destination = tmp_path / "incoming.pkl"

    def succeed(command, **kwargs):
        assert command[:3] == ["ssh", "-n", remote.HOST]
        kwargs["stdout"].write(b"payload")

    monkeypatch.setattr(transport, "_run", succeed)
    transport._ssh_download("cat artifact", destination)
    assert destination.read_bytes() == b"payload"

    def fail(_command, **kwargs):
        kwargs["stdout"].write(b"partial")
        raise subprocess.CalledProcessError(7, "ssh")

    monkeypatch.setattr(transport, "_run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        transport._ssh_download("cat artifact", destination)
    assert not destination.exists()


def test_ssh_argv_shares_one_connection_per_host(monkeypatch, tmp_path):
    monkeypatch.setenv("PYRITE_SSH_MUX", "1")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    argv = remote.config.ssh_argv("-n", "box", "true")
    assert argv[0] == "ssh" and argv[-3:] == ["-n", "box", "true"]
    assert "ControlMaster=auto" in argv
    assert f"ControlPath={tmp_path}/pyrite-ssh/%C" in argv
    assert remote.config.scp_argv("a", "b")[0] == "scp"


def test_ssh_argv_mux_can_be_disabled(monkeypatch):
    monkeypatch.setenv("PYRITE_SSH_MUX", "0")
    assert remote.config.ssh_argv("box") == ["ssh", "box"]


def test_verbose_ssh_prints_raw_command_instead_of_label(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: None)

    with transport.verbose_ssh(True):
        transport._run(["ssh", "-n", "box", "echo hi"], label="Doing work...")
    err = capsys.readouterr().err
    assert err == "+ ssh -n box echo hi\n"

    # Restored after the context manager exits.
    transport._run(["ssh", "-n", "box", "echo hi"], label="Doing work...")
    assert capsys.readouterr().err == "Doing work...\n"


def test_scp_remote_path_quotes_hostile_but_valid_posix_path(monkeypatch):
    monkeypatch.setattr(config, "HOST", "remote-host")
    path = "/srv/pyrite data/$(touch SENTINEL)"

    rendered = config.scp_remote_path(path)

    assert rendered == "remote-host:'/srv/pyrite data/$(touch SENTINEL)'"
