"""Small-budget regressions for retained transport and qualification (#807)."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.montecarlo.transport import host_memory
from pyrite.montecarlo.transport.host_pass import _row_buffers

ROOT = Path(__file__).resolve().parents[2]


def test_max_steps_no_longer_preallocates_per_electron_host_rows(monkeypatch):
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 1 << 25)
    p = SimpleNamespace(
        keep_segments_on_device=False,
        Ne=20000,
        max_steps=200000,
        energy_model="midpoint",
        groove=None,
    )
    c = SimpleNamespace(transport_core="cuda", shell_mode=True, radiative_mode=True)
    buffers = _row_buffers(p, c, True)
    assert buffers["seg_mid"].shape == (0, 3)
    buffers["seg_mid"][:3] = np.arange(9, dtype=float).reshape(3, 3)
    buffers["seg_id"][:3] = [0, 0, 1]
    assert np.array_equal(buffers["seg_mid"][:3], np.arange(9).reshape(3, 3))
    assert np.array_equal(buffers["seg_id"][:3], [0, 0, 1])
    # Shell and radiative columns use the same measured-capacity store.
    assert buffers["seg_rad_k"].shape[0] == buffers["seg_mid"].shape[0]
    assert buffers["seg_hard_dir"].shape[0] == buffers["seg_mid"].shape[0]


def test_growth_accounts_for_old_and_new_arrays_before_allocating(monkeypatch):
    owner = host_memory.GrowingHostRows({"x": ((3,), np.float64), "id": ((), np.int64)})
    col = owner.column("x")
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 100)
    col[:2] = np.ones((2, 3))  # new arrays = 2 * (24+8) = 64 bytes
    assert owner.capacity == 2
    assert np.all(col[:2] == 1)
    with pytest.raises(host_memory.HostMemoryResourceError) as caught:
        col[:4] = np.ones((4, 3))  # 4 * 32 = 128 bytes > 100
    assert caught.value.required_bytes == 128
    assert caught.value.available_bytes == 100
    assert caught.value.phase == "retained transport output grow/assembly"
    assert owner.capacity == 2
    assert np.all(col[:2] == 1)  # no partial/truncated mutation


def test_continuation_noncontiguous_scatter_preserves_rows(monkeypatch):
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 10000)
    owner = host_memory.GrowingHostRows({"id": ((), np.int64)})
    col = owner.column("id")
    col.reserve(4)
    col[np.array([3, 0, 2, 1])] = np.array([10, 11, 12, 13])
    assert np.array_equal(col[:4], [11, 13, 12, 10])


def test_containment_requires_an_actual_memory_max():
    module = runpy.run_path(str(ROOT / "checks/equal_precision/contained_sample.py"))
    validate = module["validate_memory_containment"]
    with pytest.raises(RuntimeError, match="Slurm --mem is only a reservation"):
        validate(max_bytes=10000, budget=None)
    with pytest.raises(RuntimeError, match="above the required"):
        validate(max_bytes=10000, budget=(10001, 0))
    validate(max_bytes=10000, budget=(9000, 2000))


def test_qualification_missing_sample_fails_closed_and_keeps_partial(tmp_path, monkeypatch):
    module = runpy.run_path(str(ROOT / "checks/equal_precision/analyze_qualify_schedule.py"))
    main = module["main"]
    out = tmp_path / "summary.json"
    for histories in (5000, 20000, 100000):
        for seed in (79901,) if histories == 100000 else (79901, 79902):
            for scene in ("silicon", "tungsten", "pbwo4"):
                for schedule in ("full-history", "auto"):
                    stem = f"{scene}-{histories}-{seed}-{schedule}"
                    (tmp_path / f"{stem}.json").write_text(
                        json.dumps(
                            {
                                "measurement_scope": "runner-transport-only",
                                "scene": scene,
                                "histories": histories,
                                "seed": seed,
                                "schedule": schedule,
                                "warmup_wall_s": 1.0,
                                "measured": {
                                    "wall_s": 1.0,
                                    "transport_s": 1.0,
                                    "peak_pool_bytes": 0,
                                    "schedule_log": [],
                                    "digest": {"row": "fixed"},
                                },
                                "peak_rss_bytes": 0,
                            }
                        )
                    )
                    (tmp_path / f"{stem}.status.json").write_text('{"status":"ok","exit_code":0}')
    monkeypatch.setattr(sys, "argv", ["analyze", str(tmp_path), "--output", str(out)])
    main()
    summary = json.loads(out.read_text())
    assert summary["qualified"] is True
    assert summary["valid_samples"] == 30
    (tmp_path / "silicon-5000-79901-auto.status.json").unlink()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    summary = json.loads(out.read_text())
    assert summary["qualified"] is False
    assert summary["valid_samples"] == 29
    assert any("missing status" in reason for reason in summary["failed_samples"])


def _continuation_drive(**kwargs):
    from tests.montecarlo.test_transport_continuation import _drive  # noqa: PLC0415

    return _drive(**kwargs)


@pytest.mark.parametrize(
    ("staging", "phase"),
    [
        (False, "continuation retained round download"),
        (True, "continuation staged-round spill"),
    ],
)
def test_continuation_admission_fails_cleanly_when_host_budget_is_exhausted(
    monkeypatch, staging, phase
):
    from dataclasses import replace  # noqa: PLC0415

    from pyrite.montecarlo.transport.continuation import TransportContinuationConfig  # noqa: PLC0415

    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 0)
    cont = TransportContinuationConfig(rows_per_round=5, active_histories=3)
    # staging=True with a zero staging budget forces an immediate spill to host.
    cont = replace(cont, device_staging=staging, staging_budget_bytes=0 if staging else None)
    with pytest.raises(host_memory.HostMemoryResourceError) as caught:
        _continuation_drive(cont=cont)
    assert caught.value.phase == phase
    assert caught.value.required_bytes > caught.value.available_bytes == 0


def test_continuation_admitted_request_is_unchanged(monkeypatch):
    from dataclasses import replace  # noqa: PLC0415

    from pyrite.montecarlo.transport.continuation import TransportContinuationConfig  # noqa: PLC0415

    cont = TransportContinuationConfig(rows_per_round=5, active_histories=3)
    _c, base, _l, _p = _continuation_drive(cont=replace(cont, device_staging=False))
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 1 << 30)
    _c, rows, _l, _p = _continuation_drive(cont=replace(cont, device_staging=False))
    for column in (0, 2, 5, 7, 12):  # the stand-in core leaves other scratch unset
        np.testing.assert_array_equal(base[column], rows[column])
