"""Additional remote profile tests."""

from types import SimpleNamespace

from pyrite import remote
from pyrite.cli.commands import remote as cli
from pyrite.console.dashboard import render as dashboard_render
from pyrite.remote import lifecycle, scripts, state, transport, viewer
from pyrite.remote import queue as lifecycle_queue


def test_is_reapable_only_when_dead_and_past_guard():
    guard = 300.0
    # dead (no SLURM allocation) AND old enough -> reap
    assert remote._is_reapable(None, 301, guard) is True
    assert remote._is_reapable(None, 300, guard) is True
    # dead but still inside the staging-race window -> keep
    assert remote._is_reapable(None, 299, guard) is False
    # live allocation is never reaped, however old the lock looks
    assert remote._is_reapable("RUNNING", 10_000, guard) is False
    assert remote._is_reapable("PENDING", 10_000, guard) is False


def test_reline_queue_script_and_metadata():
    from pyrite.remote import scripts

    # Default remote reline is chunked like `pyrite remote run`.
    s = scripts._recxr_chunked_queue_script(
        "J1", ["mos2", "w"], cxr_ne=40000, line_step_eV=None, redo_all=True, chunk_minutes=10.0
    )
    assert "pyrite checkpoint recompute cxr" in s
    assert "--cxr-trials 40000" in s
    assert "--redo-all" in s
    assert "--max-minutes" in s
    assert "--nice=10000" in s
    assert "--progress-file" in s
    # The monolithic (chunk_minutes==0) path still recomputes lines per material.
    mono = scripts._recxr_queue_script("J1", ["mos2", "w"], 40000, None, True)
    assert (
        "pyrite checkpoint recompute cxr" in mono
        and "--cxr-trials 40000" in mono
        and "--redo-all" in mono
    )
    assert "--progress-file" in mono
    meta = scripts._recxr_queue_metadata("J1", ["mos2", "w"], 40000, None, True)
    assert "kind: reline" in meta
    assert "line_ne: 40000" in meta


def test_cli_rebrem_pulls_brem_dataset(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_rebrem_queue", lambda *a, **k: "J1")
    monkeypatch.setattr(viewer, "attach", lambda jobid: True)
    monkeypatch.setattr(state, "_completed_materials", lambda jobid, mats: ["mos2"])
    captured = {}
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **kw: captured.update(stems=stems, kw=kw))
    cli._cli_rebrem(
        material=["mos2"],
        all_=False,
        ne_brem=None,
        start=None,
        stop=None,
        step=None,
        redo_all=False,
        no_sync=False,
        dry_run=False,
        chunk_minutes=10.0,
        detach=False,
    )
    assert captured["stems"] == ["mos2"]
    assert captured["kw"].get("dataset") == "brem"


def test_failed_legacy_status_marks_incomplete_bar_as_last_batch():
    output = dashboard_render._legacy_progress(
        "cases: 40%|████      | 2/5 [00:03<00:04, 1.4s/it]\r",
        "FAILED (exit 1) now",
    )

    assert output is not None
    assert "last case batch" in output
    assert "×" in output
    assert "2/5" in output


def test_profile_jobdirs_parses_existing_dirs(monkeypatch):
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "sub_100keV\nsub_100keV-2\n")

    assert state._profile_jobdirs("sub_100keV") == {"sub_100keV", "sub_100keV-2"}


def test_same_second_starters_use_unique_exclusive_job_directories(monkeypatch):
    """Two submitters must not overwrite a shared second-resolution job dir."""

    real_datetime = scripts.datetime.datetime

    class FixedDatetime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 7, 15, 12, 0, 0)

    suffixes = iter(["a" * 32, "b" * 32])
    uploads = []
    monkeypatch.setattr(scripts.datetime, "datetime", FixedDatetime)
    monkeypatch.setattr(
        scripts,
        "uuid",
        SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=next(suffixes))),
        raising=False,
    )
    monkeypatch.setattr(lifecycle_queue, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda command, **_kwargs: uploads.append(command) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")

    first = remote.start_queue(["hopg"], no_sync=True)
    second = remote.start_queue(["hbn"], no_sync=True)

    assert first == "20260715-120000-aaaaaaaa"
    assert second == "20260715-120000-bbbbbbbb"
    assert first != second
    assert all("mkdir -p" not in upload[-1] for upload in uploads)
    assert all("mkdir '" in upload[-1] for upload in uploads)
