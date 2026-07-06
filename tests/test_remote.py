"""remote.py: material-name validation and the detached-queue runner generation.

These are pure-string/logic checks (no ssh), so they run anywhere."""

import pytest
import remote


def test_check_materials_accepts_crystal_keys():
    remote._check_materials(["mose2", "hopg", "mote2", "silicon"])  # no raise


@pytest.mark.parametrize("bad", ["rm -rf /", "a;b", "../etc", "a b", "", "m&n"])
def test_check_materials_rejects_injection(bad):
    with pytest.raises(SystemExit):
        remote._check_materials([bad])


def test_queue_script_has_per_material_scan_calls():
    s = remote._queue_script("20260101-000000", ["mose2", "wse2"], quick=True, workers=8)
    assert "scan.py" in s
    assert "--quick" in s and "--workers 8" in s
    assert "mose2" in s and "wse2" in s
    assert "20260101-000000" in s  # job id is embedded
    assert "mats=(mose2 wse2)" in s  # bash array drives the loop


def test_queue_script_no_flags_when_unset():
    s = remote._queue_script("j", ["mos2"], quick=False, workers=None)
    assert "--quick" not in s and "--workers" not in s


def test_stems_quick_suffix():
    assert remote._stems(["mose2", "wse2"], True) == ["mose2_quick", "wse2_quick"]
    assert remote._stems(["mose2"], False) == ["mose2"]


# ---- clear <material> (checkpoint lifecycle, component 3) ----------------------
def _no_live_jobs(monkeypatch):
    monkeypatch.setattr(remote, "_live_jobs", lambda: [])


def test_clear_refuses_when_a_live_job_produces_the_stem(monkeypatch):
    # a live job producing hopg -> clearing hopg must refuse before any ssh
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda *a: pytest.fail("must not ssh when refusing")
    )
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg", yes=True)


def test_clear_refuses_for_quick_stem_collision(monkeypatch):
    # a live --quick job producing hopg_quick still blocks a clear of hopg
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", True, ["hopg"])])
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg")


def test_clear_dry_preview_lists_but_does_not_delete(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "hopg.pkl\nhopg_quick.pkl\n")
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=False)
    out = capsys.readouterr().out
    assert "would delete" in out and "hopg.pkl" in out and "hopg_quick.pkl" in out
    assert runs == []  # nothing deleted in a dry preview


def test_clear_yes_deletes_existing_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "hopg.pkl hopg_quick.pkl")
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert len(runs) == 1
    deletion = " ".join(runs[0])
    assert "rm -f" in deletion and "hopg.pkl" in deletion and "hopg_quick.pkl" in deletion
    assert "cleared on the box" in capsys.readouterr().out


def test_clear_reports_nothing_when_no_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "\n")
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert "nothing to clear" in capsys.readouterr().out
    assert runs == []  # nothing to delete


def test_clear_rejects_bad_material(monkeypatch):
    monkeypatch.setattr(
        remote, "_live_jobs", lambda: pytest.fail("must validate before touching jobs")
    )
    with pytest.raises(SystemExit):
        remote.clear_remote("rm -rf /")
