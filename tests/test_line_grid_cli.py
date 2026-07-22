import argparse

from cxr_mc import line_grid


def _parse(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    line_grid.add_subparser(sub)
    return ap.parse_args(argv)


def test_status_delegates_to_remote_job_status_with_detail(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        line_grid.remote,
        "job_status",
        lambda jobid=None, detail=0: seen.update(jobid=jobid, detail=detail),
    )
    args = _parse(["line-grid", "status", "job7", "-vv"])
    args.func(args)
    assert seen == {"jobid": "job7", "detail": 2}


def test_apply_dispatches_with_pull_and_force(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid, "_pull_combined", lambda: "combined.json")
    monkeypatch.setattr(
        line_grid.apply, "apply_file", lambda path, **kw: seen.update(path=path, **kw)
    )
    args = _parse(["line-grid", "apply", "--pull", "--force"])
    args.func(args)
    assert seen["path"] == "combined.json" and seen["force"] is True


def test_stop_falls_back_to_latest_jobid(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid.remote, "_latest_jobid", lambda: "job9")
    monkeypatch.setattr(line_grid.remote, "_stop_jobid", lambda jobid: seen.update(jobid=jobid))
    args = _parse(["line-grid", "stop"])
    args.func(args)
    assert seen == {"jobid": "job9"}


def test_regen_golden_delegates_check(monkeypatch):
    from cxr_mc.line_grid import golden

    seen = {}

    def _fake_regen(check=False):
        seen["check"] = check
        return 0

    monkeypatch.setattr(golden, "regen", _fake_regen)
    args = _parse(["line-grid", "regen-golden", "--check"])
    assert args.func(args) == 0
    assert seen["check"] is True
