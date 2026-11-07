"""Additional remote validation tests."""

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as remote_cli
from pyrite.remote import lifecycle, state, viewer


def test_zhai_queue_script_has_ne_flags_and_meta():
    s = remote._zhai_queue_script(
        "20260101-000000",
        ne=11,
        ne_brem=3,
        ne_supp=5,
        tmd_azimuth=35.0,
        refresh=True,
    )
    assert "python -m pyrite._entry.reproduce_zhai" in s
    assert "--ne 11" in s and "--brem-histories 3" in s and "--ne-supp 5" in s and "--refresh" in s
    assert "--tmd-azimuth 35.0" in s
    assert "20260101-000000" in s
    assert "materials: zhai" not in s and "quick: False" not in s
    assert "started: $(date -Is)" in s
    assert "/pid" not in s


def test_zhai_queue_metadata_records_cache_and_detector_provenance():
    metadata = remote._zhai_queue_metadata("j", ne=11, ne_brem=3, ne_supp=5)

    assert "zhai_cache_schema: 4" in metadata
    assert "detector_observation_angle_deg: 119" in metadata
    assert "detector_polar_acceptance_deg: 16.6" in metadata
    assert "detector_solid_angle_sr: 0.066" in metadata


def test_zhai_queue_script_no_refresh_flag_when_unset():
    s = remote._zhai_queue_script("j", ne=1, ne_brem=1, ne_supp=1, tmd_azimuth=0.0, refresh=False)
    assert "--refresh" not in s


@pytest.mark.parametrize("job_state", ["FAILED (exit 1)", "cancelled [48291]"])
def test_failed_foreground_check_does_not_pull_stale_cache(monkeypatch, job_state):
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **_kw: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(state, "_job_succeeded", lambda _jobid: False, raising=False)
    monkeypatch.setattr(state, "_job_state", lambda _jobid: job_state, raising=False)
    monkeypatch.setattr(
        lifecycle, "pull_zhai_cache", lambda: pytest.fail("failed job must not pull stale cache")
    )

    with pytest.raises(SystemExit, match="did not complete successfully"):
        remote_cli.remote_check(no_sync=True)


def test_interrupted_foreground_check_does_not_pull(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **_kw: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: False)
    monkeypatch.setattr(
        lifecycle, "pull_zhai_cache", lambda: pytest.fail("must not pull after interruption")
    )

    remote_cli.remote_check(no_sync=True)
