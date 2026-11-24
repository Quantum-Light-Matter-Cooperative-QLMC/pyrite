"""The fixed-population probe fixes both independently stopped populations."""

import runpy
from dataclasses import dataclass
from pathlib import Path

import pytest

from pyrite._precision import Precision
from pyrite.runs import scan


@pytest.mark.parametrize("independent_brem", [False, True])
def test_fixed_probe_resolves_equal_line_and_brem_counts(monkeypatch, independent_brem):
    @dataclass
    class Settings:
        precision: Precision

    precision = Precision(
        target_rse=0.001,
        min_histories=3000,
        max_histories=20_000_000,
        pilot_histories=3000,
        observables=("cxr", "brem") if independent_brem else ("cxr",),
        **(dict(brem_target_rse=0.001, brem_max_histories=20_000_000) if independent_brem else {}),
    )
    original = Settings(precision)
    monkeypatch.setattr(scan, "_resolved_run", lambda *_: (original, "sweep", "identity", "stem"))
    monkeypatch.setenv("PROFILE", "cleo_sub_1281")
    probe = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "checks/reflection_selection_subsample_1281.py")
    )
    probe["precision_override"](4096)
    resolved, *rest = scan._resolved_run(None, "mose2")
    assert rest == ["sweep", "identity", "stem"]
    assert resolved.precision.min_histories == resolved.precision.max_histories == 4096
    assert resolved.precision.pilot_histories is None
    if independent_brem:
        assert (
            resolved.precision.brem_min_histories == resolved.precision.brem_max_histories == 4096
        )
    else:
        assert resolved.precision.brem_target_rse is None
    assert original.precision.max_histories == 20_000_000
