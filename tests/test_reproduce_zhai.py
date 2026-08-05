"""Tests for the cxr_mc._entry.reproduce_zhai shim (the box-invokable entry
point for the remote Zhai preset, run as ``python -m cxr_mc._entry.reproduce_zhai``)
-- argument parsing and CLI wiring only; the actual MC work is reproduce_all,
tested in tests/test_anchor_figures.py."""

from pathlib import Path

from cxr_mc._entry import reproduce_zhai


def test_cli_defaults_match_app_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(reproduce_zhai, "reproduce_all", lambda **kw: calls.append(kw) or [])

    reproduce_zhai.main([])

    assert calls == [
        {
            "ne": 20_000,
            "ne_brem": 200,
            "ne_supp": 200,
            "tmd_exploratory_azimuth_deg": 0.0,
            "cache_dir": None,
            "refresh": False,
        }
    ]


def test_cli_forwards_overrides(monkeypatch, capsys):
    calls = []

    def fake_reproduce_all(**kw):
        calls.append(kw)
        return [("zhai-fig1c", Path("checkpoints/zhai_reproduction/zhai-abc.pkl"), True)]

    monkeypatch.setattr(reproduce_zhai, "reproduce_all", fake_reproduce_all)

    reproduce_zhai.main(
        [
            "--ne",
            "11",
            "--ne-brem",
            "3",
            "--ne-supp",
            "5",
            "--tmd-azimuth",
            "35",
            "--refresh",
            "--cache-dir",
            "/tmp/x",
        ]
    )

    assert calls == [
        {
            "ne": 11,
            "ne_brem": 3,
            "ne_supp": 5,
            "tmd_exploratory_azimuth_deg": 35.0,
            "cache_dir": "/tmp/x",
            "refresh": True,
        }
    ]
    out = capsys.readouterr().out
    assert "zhai-fig1c" in out and "cached" in out
